import warnings

import meshio
import numpy

from .helpers import unique_rows

__all__ = ["_SimplexMesh"]


class _SimplexMesh:
    def __init__(self, points, cells, sort_cells=False):
        if sort_cells:
            # Sort cells, first every row, then the rows themselves. This helps in many
            # downstream applications, e.g., when constructing linear systems with the
            # cells/facets. (When converting to CSR format, the I/J entries must be
            # sorted.) Don't use cells.sort(axis=1) to avoid
            # ```
            # ValueError: sort array is read-only
            # ```
            cells = numpy.sort(cells, axis=1)
            cells = cells[cells[:, 0].argsort()]

        points = numpy.asarray(points)
        cells = numpy.asarray(cells)
        assert len(points.shape) == 2, f"Illegal point coordinates shape {points.shape}"
        assert len(cells.shape) == 2, f"Illegal cells shape {cells.shape}"
        self.n = cells.shape[1]
        assert self.n in [3, 4], f"Illegal cells shape {cells.shape}"

        # Assert that all vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = numpy.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # points = points[uvertices]
        # ```
        # helps.
        # is_used = numpy.zeros(len(points), dtype=bool)
        # is_used[cells] = True
        # assert numpy.all(is_used), "There are {} dangling points in the mesh".format(
        #     numpy.sum(~is_used)
        # )

        self._is_boundary_facet_local = None
        self._is_boundary_facet = None
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_cell = None
        self._boundary_facets = None
        self._interior_facets = None

        self._half_edge_coords = None
        self._ei_dot_ei = None
        self._ei_dot_ej = None

        self._points = numpy.asarray(points)
        # prevent accidental override of parts of the array
        self._points.setflags(write=False)

        self.facets = None
        self.cells = {"points": numpy.asarray(cells)}
        nds = self.cells["points"].T

        if cells.shape[1] == 3:
            # Create the idx_hierarchy (points->facets->cells), i.e., the value of
            # `self.idx_hierarchy[0, 2, 27]` is the index of the point of cell 27, facet
            # 2, point 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`, where `n` is
            # the number of cells. Make sure that the k-th facet is opposite of the k-th
            # point in the triangle.
            self.local_idx = numpy.array([[1, 2], [2, 0], [0, 1]]).T
        else:
            assert cells.shape[1] == 4
            # Arrange the point_face_cells such that point k is opposite of face k in
            # each cell.
            idx = numpy.array([[1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]).T
            self.point_face_cells = nds[idx]

            # Arrange the idx_hierarchy (point->facet->face->cells) such that
            #
            #   * point k is opposite of facet k in each face,
            #   * duplicate facets are in the same spot of the each of the faces,
            #   * all points are in domino order ([1, 2], [2, 3], [3, 1]),
            #   * the same facets are easy to find:
            #      - facet 0: face+1, facet 2
            #      - facet 1: face+2, facet 1
            #      - facet 2: face+3, facet 0
            #   * opposite facets are easy to find, too:
            #      - facet 0  <-->  (face+2, facet 0)  equals  (face+3, facet 2)
            #      - facet 1  <-->  (face+1, facet 1)  equals  (face+3, facet 1)
            #      - facet 2  <-->  (face+1, facet 0)  equals  (face+2, facet 2)
            #
            self.local_idx = numpy.array(
                [
                    [[2, 3], [3, 1], [1, 2]],
                    [[3, 0], [0, 2], [2, 3]],
                    [[0, 1], [1, 3], [3, 0]],
                    [[1, 2], [2, 0], [0, 1]],
                ]
            ).T

        # Map idx back to the points. This is useful if quantities which are in idx
        # shape need to be added up into points (e.g., equation system rhs).
        self.idx_hierarchy = nds[self.local_idx]

        # The inverted local index.
        # This array specifies for each of the three points which facet endpoints
        # correspond to it. For triangles, the above local_idx should give
        #
        #    [[(1, 1), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (0, 1)]]
        #
        self.local_idx_inv = [
            [tuple(i) for i in zip(*numpy.where(self.local_idx == k))]
            for k in range(self.n)
        ]

        self._edge_lengths = None

    # prevent overriding points without adapting the other mesh data
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        new_points = numpy.asarray(new_points)
        assert new_points.shape == self._points.shape
        self._points = new_points
        # reset all computed values
        self._reset_point_data()

    def set_points(self, new_points, idx=slice(None)):
        self.points.setflags(write=True)
        self.points[idx] = new_points
        self.points.setflags(write=False)
        self._reset_point_data()

    @property
    def half_edge_coords(self):
        if self._half_edge_coords is None:
            p = self.points[self.idx_hierarchy]
            self._half_edge_coords = p[1] - p[0]
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
            # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
            # TODO reorganize the data?
            self._ei_dot_ei = numpy.einsum(
                "...k, ...k->...", self.half_edge_coords, self.half_edge_coords
            )
        return self._ei_dot_ei

    @property
    def ei_dot_ej(self):
        if self._ei_dot_ej is None:
            # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
            # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
            # TODO reorganize the data?
            self._ei_dot_ej = numpy.einsum(
                "...k, ...k->...",
                self.half_edge_coords[[1, 2, 0]],
                self.half_edge_coords[[2, 0, 1]],
            )
        return self._ei_dot_ej

    def compute_centroids(self, idx=slice(None)):
        return numpy.sum(self.points[self.cells["points"][idx]], axis=1) / self.n

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all
        simplices."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_centroids()
        return self._cell_centroids

    @property
    def cell_barycenters(self):
        """See cell_centroids."""
        return self.cell_centroids

    @property
    def is_point_used(self):
        # Check which vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = numpy.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # points = points[uvertices]
        # ```
        # helps.
        if self._is_point_used is None:
            self._is_point_used = numpy.zeros(len(self.points), dtype=bool)
            self._is_point_used[self.cells["points"]] = True
        return self._is_point_used

    def write(self, filename, point_data=None, cell_data=None, field_data=None):
        if self.points.shape[1] == 2:
            n = len(self.points)
            a = numpy.ascontiguousarray(
                numpy.column_stack([self.points, numpy.zeros(n)])
            )
        else:
            a = self.points

        if self.cells["points"].shape[1] == 3:
            cell_type = "triangle"
        else:
            assert (
                self.cells["points"].shape[1] == 4
            ), "Only triangles/tetrahedra supported"
            cell_type = "tetra"

        meshio.write_points_cells(
            filename,
            a,
            {cell_type: self.cells["points"]},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    @property
    def edge_lengths(self):
        if self._edge_lengths is None:
            self._edge_lengths = numpy.sqrt(self.ei_dot_ei)
        return self._edge_lengths

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]["vertices"]

    def get_facet_mask(self, subdomain=None):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its facets are in.
        # An facet is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over the first index
        is_inside = numpy.all(is_in, axis=tuple(range(1)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet

        return is_inside

    def get_face_mask(self, subdomain):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its facets are in.
        # An facet is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last two (face_ids, cell_ids).
        n = len(is_in.shape)
        is_inside = numpy.all(is_in, axis=tuple(range(n - 2)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet_local

        return is_inside

    def get_cell_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain.is_boundary_only:
            # There are no boundary cells
            return numpy.array([])

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last one (cell_ids).
        n = len(is_in.shape)
        return numpy.all(is_in, axis=tuple(range(n - 1)))

    def _mark_vertices(self, subdomain):
        """Mark faces/facets which are fully in subdomain."""
        if subdomain is None:
            is_inside = numpy.ones(len(self.points), dtype=bool)
        else:
            is_inside = subdomain.is_inside(self.points.T).T

            if subdomain.is_boundary_only:
                # Filter boundary
                self.mark_boundary()
                is_inside = is_inside & self.is_boundary_point

        self.subdomains[subdomain] = {"vertices": is_inside}

    def create_facets(self):
        """Identify individual facets, identify the boundary etc."""
        # Reshape into individual faces
        s = self.idx_hierarchy.shape
        if self.n == 3:
            a = self.idx_hierarchy.reshape([s[0], -1])
        else:
            # Take the first point per facet. The face is fully characterized by it.
            a = self.idx_hierarchy[0].reshape([s[1], -1])

        # Sort the columns to make it possible for `unique()` to identify individual
        # faces.
        a = numpy.sort(a.T)
        # Find the unique faces
        a_unique, inv, cts = unique_rows(a)

        assert numpy.all(
            cts < 3
        ), "No facet has more than 2 cells. Are cells listed twice?"

        self._is_boundary_facet_local = (cts[inv] == 1).reshape(s[self.n - 2 :])
        self._is_boundary_facet = cts == 1

        self.facets = {"points": a_unique}

        # cell->faces relationship
        self.cells["facets"] = inv.reshape([self.n, -1]).T

        # Store the opposing points too
        # self.cells["opposing vertex"] = self.cells["points"]

        # # save for create_facet_cells
        # self._inv_faces = inv

        self._facets_cells = None
        self._facets_cells_idx = None

    @property
    def is_boundary_cell(self):
        if self._is_boundary_cell is None:
            assert self.is_boundary_facet_local is not None
            self._is_boundary_cell = numpy.any(self.is_boundary_facet_local, axis=0)
        return self._is_boundary_cell

    @property
    def is_boundary_facet_local(self):
        if self._is_boundary_facet_local is None:
            self.create_facets()
        return self._is_boundary_facet_local

    is_boundary_facet_local = is_boundary_facet_local

    @property
    def is_boundary_facet(self):
        if self._is_boundary_facet is None:
            self.create_facets()
        return self._is_boundary_facet

    @property
    def is_interior_facet(self):
        return ~self._is_boundary_facet

    @property
    def boundary_facets(self):
        if self._boundary_facets is None:
            self._boundary_facets = numpy.where(self.is_boundary_facet)[0]
        return self._boundary_facets

    @property
    def interior_facets(self):
        if self._interior_facets is None:
            self._interior_facets = numpy.where(~self.is_boundary_facet)[0]
        return self._interior_facets

    @property
    def is_boundary_point(self):
        if self._is_boundary_point is None:
            self._is_boundary_point = numpy.zeros(len(self.points), dtype=bool)
            self._is_boundary_point[
                self.idx_hierarchy[..., self.is_boundary_facet_local]
            ] = True
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self._is_interior_point = self.is_point_used & ~self.is_boundary_point
        return self._is_interior_point

    def mark_boundary(self):
        warnings.warn(
            "mark_boundary() does nothing. "
            "Boundary entities are computed on the fly."
        )

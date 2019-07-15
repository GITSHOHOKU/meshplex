from __future__ import print_function

import shutil

import hashlib
import os
from math import fsum
import numpy
import requests


# The tests files are located on sourceforge.
def download_mesh(name, md5):

    filename = os.path.join("/tmp", name)
    if not os.path.exists(filename):
        print("Downloading {}...".format(name))
        url = "https://nschloe.github.io/meshplex/"
        print(url + name)
        r = requests.get(url + name, stream=True)
        if not r.ok:
            raise RuntimeError(
                "Download error ({}, return code {}).".format(r.url, r.status_code)
            )

        # save the mesh in /tmp
        with open(filename, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

    # check MD5
    file_md5 = hashlib.md5(open(filename, "rb").read()).hexdigest()

    if file_md5 != md5:
        raise RuntimeError("Checksums not matching (%s != %s)." % (file_md5, md5))

    return filename


def near_equal(a, b, tol):
    return numpy.allclose(a, b, rtol=0.0, atol=tol)


def run(mesh, volume, convol_norms, ce_ratio_norms, cellvol_norms, tol=1.0e-12):
    # Check cell volumes.
    total_cellvolume = fsum(mesh.cell_volumes)
    assert abs(volume - total_cellvolume) < tol * volume
    norm2 = numpy.linalg.norm(mesh.cell_volumes, ord=2)
    norm_inf = numpy.linalg.norm(mesh.cell_volumes, ord=numpy.Inf)
    assert near_equal(cellvol_norms, [norm2, norm_inf], tol)

    # If everything is Delaunay and the boundary elements aren't flat, the
    # volume of the domain is given by
    #   1/n * edge_lengths * ce_ratios.
    # Unfortunately, this isn't always the case.
    # ```
    # total_ce_ratio = \
    #     fsum(mesh.edge_lengths**2 * mesh.get_ce_ratios_per_edge() / dim)
    # self.assertAlmostEqual(volume, total_ce_ratio, delta=tol * volume)
    # ```
    # Check ce_ratio norms.
    # TODO reinstate
    alpha2 = fsum((mesh.ce_ratios ** 2).flat)
    alpha_inf = max(abs(mesh.ce_ratios).flat)
    assert near_equal(ce_ratio_norms, [alpha2, alpha_inf], tol)

    # Check the volume by summing over the absolute value of the control
    # volumes.
    vol = fsum(mesh.control_volumes)
    assert abs(volume - vol) < tol * volume
    # Check control volume norms.
    norm2 = numpy.linalg.norm(mesh.control_volumes, ord=2)
    norm_inf = numpy.linalg.norm(mesh.control_volumes, ord=numpy.Inf)
    assert near_equal(convol_norms, [norm2, norm_inf], tol)

    return


def compute_polygon_area(pts):
    # shoelace formula
    return 0.5 * abs(
        numpy.dot(pts[0], numpy.roll(pts[1], -1))
        - numpy.dot(numpy.roll(pts[0], -1), pts[1])
    )

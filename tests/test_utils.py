"""Test utils package"""
import numpy as np

from polycrystal.utils import tensordata as td
from polycrystal.utils import unique_vectors as uv


# Testing tensordata
TOL = 1e-14
n = 6
mats = np.random.random((n, 3, 3))
tdata = td.TensorData(matrices=mats)


def test_symmdev():
    """Check tensor parts - symmetric & deviatoric"""
    part = tdata.symmdev
    tmp = td.TensorData.from_parts(symmdev=part)
    part_tmp = tmp.symmdev
    assert np.allclose(part, part_tmp)


def test_spherical():
    """spherical  / isotropic"""
    part = tdata.sph
    tmp = td.TensorData.from_parts(sph=part)
    part_tmp = tmp.sph
    assert np.allclose(part, part_tmp)


def test_skew():
    """skew part"""
    part = tdata.skew
    tmp = td.TensorData.from_parts(skew=part)
    part_tmp = tmp.skew
    assert np.allclose(part, part_tmp)


def test_all():
    """Check all parts together"""
    tmp = td.TensorData.from_parts(
        skew=tdata.skew, symmdev=tdata.symmdev, sph=tdata.sph
    )
    assert np.allclose(tmp.matrices, mats)


def test_unique_vectors():
    """test unique_vectors"""
    a = [[1, 2], [3.2, 1], [0, 4], [1, 2]]
    b = [[0, 4], [1, 2], [3.2, 1]]
    assert np.allclose(uv.unique_vectors(np.array(a)), np.array(b))

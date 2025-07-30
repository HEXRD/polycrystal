"""Test utils package"""
import numpy as np

from polycrystal.utils import tensordata as td
from polycrystal.utils tensor_data.elasticity import MandelForm
from polycrystal.utils import unique_vectors as uv


# Testing tensordata
TOL = 1e-14
n = 6
mats = np.random.random((n, 3, 3))
tdata = td.TensorData(matrices=mats)


class TestTensorDataOld:

    def test_symmdev(self):
        """Check tensor parts - symmetric & deviatoric"""
        part = tdata.symmdev
        tmp = td.TensorData.from_parts(symmdev=part)
        part_tmp = tmp.symmdev
        assert np.allclose(part, part_tmp)


    def test_spherical(self):
        """spherical  / isotropic"""
        part = tdata.sph
        tmp = td.TensorData.from_parts(sph=part)
        part_tmp = tmp.sph
        assert np.allclose(part, part_tmp)


    def test_skew(self):
        """skew part"""
        part = tdata.skew
        tmp = td.TensorData.from_parts(skew=part)
        part_tmp = tmp.skew
        assert np.allclose(part, part_tmp)


    def test_all(self):
        """Check all parts together"""
        tmp = td.TensorData.from_parts(
            skew=tdata.skew, symmdev=tdata.symmdev, sph=tdata.sph
        )
        assert np.allclose(tmp.matrices, mats)


def test_unique_vectors():
    """test unique_vectors"""
    a = np.array([[1, 2], [3.2, 1], [0, 4], [1, 2]])
    b = np.array([[0, 4], [1, 2], [3.2, 1]])
    assert np.allclose(uv.unique_vectors(a), b)

    au, index = uv.unique_vectors(a, return_index=True)
    assert np.all(au == b)
    assert(np.all(a[index] == b))

    au, inverse = uv.unique_vectors(a, return_inverse=True)
    assert(np.all(a == b[inverse]))

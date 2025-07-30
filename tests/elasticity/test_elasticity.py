"""Unit tests for single crystal elasticity"""
import numpy as np
import pytest

from polycrystal.elasticity import single_crystal
from polycrystal.elasticity.single_crystal import SingleCrystal

TOL = 1e-14


def maxdiff(a, b):
    return np.max(np.abs(a - b))

@pytest.fixture
def isotropic_eigen_basis():
    """Return basis for isotropic tensors"""
    return np.array([
        [1, 1, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 0],
        [1, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])


class TestSingleCrystal:

    ID_6X6 = np.identity(6)

    def test_identity(self):

        # From K, G
        sx = single_crystal.SingleCrystal.from_K_G(1/3, 1/2)
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # From E, nu
        sx = single_crystal.SingleCrystal.from_E_nu(1, 0)
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Isotropic: c11, c12
        sx = single_crystal.SingleCrystal(
            'isotropic', [1.0,  0.0]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Cubic
        sx = single_crystal.SingleCrystal(
            'cubic', [1.0,  0.0,  0.5]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Hexagonal
        sx = single_crystal.SingleCrystal(
            'hexagonal', [1.0,  0.0,  0.0,  1.0,  0.5]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

    @pytest.mark.parametrize("K,G", [(2.3, 1.5), (3.2, 2.7)])
    def test_isotropic(self, K, G, isotropic_eigen_basis):
        """Test isotropic materials"""
        mat = SingleCrystal.from_K_G(K, G)
        sig = mat.stiffness @ isotropic_eigen_basis.T
        print("hi!\n", mat.stiffness)

        assert np.allclose(sig.T[0], 3 * K * isotropic_eigen_basis[0])
        assert np.allclose(sig.T[1:], 2 * G * isotropic_eigen_basis[1:])

        # Now try the Mandel form


class TestThermalExpansion:

    def test_cte_none(self):

        # From float.
        sx = single_crystal.SingleCrystal.from_E_nu(1, 0)
        assert not hasattr(sx, "cte")

    def test_cte_float(self):

        # From float.
        cte = 1.2e-3
        sx = single_crystal.SingleCrystal.from_E_nu(1, 0, cte=cte)
        assert np.allclose(sx.cte, np.diag(3 * (cte,)))

    def test_cte_array33(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
            [3.1, 41, 5.9],
        ])
        sx = single_crystal.SingleCrystal.from_E_nu(1, 0, cte=arr)
        assert np.allclose(sx.cte, arr)

    def test_cte_array_shape(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
        ])
        with pytest.raises(RuntimeError):
            sx = single_crystal.SingleCrystal.from_E_nu(1, 0, cte=arr)


"""
class TestUtilities(unittest.TestCase):
    def test_matrix_rep(self):
        a6 = np.array([2.0, 3.1, 5.2, 7.3, 11.4, 13.5])
        m3x3 = sx_elas.to_3x3(a6)
        b6 = sx_elas.to_6vec(m3x3)
        for i in range(len(a6)):
            self.assertEqual(a6[i], b6[i])

    @unittest.skip("under development")
    def test_rotation(self):
        i3 = np.identity(3)
        R3 = np.vstack((i3[2], i3[0], i3[1]))
        LR = sx_elas.rotation_operator(R3)

        M6 = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # print np.dot(LR, M6)
"""

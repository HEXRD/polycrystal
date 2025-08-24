"""Unit tests for single crystal elasticity"""
import numpy as np
import pytest

from polycrystal.elasticity.single_crystal import SingleCrystal

TOL = 1e-14


def maxdiff(a, b):
    return np.max(np.abs(a - b))



class TestSingleCrystal:

    @pytest.fixture
    def ID_6X6(cls):
        return np.identity(6)

    def test_identity(self, ID_6X6):

        # From K, G
        sx = SingleCrystal.from_K_G(1/3, 1/2)
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # From E, nu
        sx = SingleCrystal.from_E_nu(1, 0)
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Isotropic: c11, c12
        sx = SingleCrystal(
            'isotropic', [1.0,  0.0]
        )
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Cubic
        sx = SingleCrystal(
            'cubic', [1.0,  0.0,  0.5]
        )
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Hexagonal
        sx = SingleCrystal(
            'hexagonal', [1.0,  0.0,  0.0,  1.0,  0.5]
        )
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

        # Triclinic
        cij = [
            1.0, 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  1.0,
            0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  1.0
        ]
        sx = SingleCrystal(
            'triclinic', cij, input_system="MANDEL"
        )
        assert maxdiff(sx.stiffness, ID_6X6) < TOL

    def test_cij_in_out(self):
        """Test cij for output system"""

        # Isotropic.
        sx = SingleCrystal(
            'isotropic', [2.3, 4.9],
            input_system = "VOIGT_GAMMA",
            output_system = "MANDEL",
        )
        assert np.all(sx.cij_in == sx.cij)
        assert np.allclose([2.3, 4.9], sx.cij_in)
        assert np.allclose([2.3, 4.9], sx.cij_out)

        # Cubic.
        sx = SingleCrystal(
            'cubic', [2.3, 4.5, 7.8],
            input_system = "VOIGT_GAMMA",
            output_system = "VOIGT_EPSILON",
        )
        assert np.allclose([2.3, 4.5, 7.8], sx.cij_in)
        assert np.allclose([2.3, 4.5, 2 * 7.8], sx.cij_out)

        # Hexagonal
        test_cij_eps = [1.0,  2.3, 3.4, 5.6, 6.6]
        test_cij_gam = [1.0,  2.3, 3.4, 5.6, 3.3]
        sx = SingleCrystal(
            'hexagonal', test_cij_eps,
            input_system = "VOIGT_EPSILON",
            output_system = "VOIGT_GAMMA",
        )
        assert np.all(sx.cij_in == sx.cij)
        assert np.allclose(test_cij_gam, sx.cij_out)

        # Triclinic
        test_cij = np.arange(21)
        sx = SingleCrystal(
            'triclinic', test_cij,
            input_system="MANDEL",
            output_system = "VOIGT_GAMMA"
        )
        top_left = np.array((0, 1, 2, 6, 7, 11))
        assert np.allclose(sx.cij_out[top_left], test_cij[top_left])
        top_right = np.array((3, 4, 5, 8, 9, 10, 12, 13, 14,))
        assert np.allclose(sx.cij_out[top_right], 1/np.sqrt(2) * test_cij[top_right])
        # Bottom Left.
        assert np.allclose(sx.cij_out[15:], 0.5 * test_cij[15:])


class TestThermalExpansion:

    def test_cte_none(self):

        # From float.
        sx = SingleCrystal.from_E_nu(1, 0)
        assert not hasattr(sx, "cte")

    def test_cte_float(self):

        # From float.
        cte = 1.2e-3
        sx = SingleCrystal.from_E_nu(1, 0, cte=cte)
        assert np.allclose(sx.cte, np.diag(3 * (cte,)))

    def test_cte_array33(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
            [3.1, 41, 5.9],
        ])
        sx = SingleCrystal.from_E_nu(1, 0, cte=arr)
        assert np.allclose(sx.cte, arr)

    def test_cte_array_shape(self):

        # From float.
        arr = np.array([
            [1.0, 2.9, 0.3],
            [-1.3, 4.6, 0.9],
        ])
        with pytest.raises(RuntimeError):
            sx = SingleCrystal.from_E_nu(1, 0, cte=arr)

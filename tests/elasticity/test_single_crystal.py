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

    def test_cij_out(self):
        """Test cij for output system"""
        sx = SingleCrystal(
            'cubic', [2.3, 4.5, 7.8],
            input_system = "VOIGT_GAMMA",
            output_system = "VOIGT_EPSILON",
        )
        assert np.allclose([2.3, 4.5, 7.8], sx.cij_in)
        assert np.allclose([2.3, 4.5, 2 * 7.8], sx.cij_out)




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

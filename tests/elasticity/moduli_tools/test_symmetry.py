"""Tests on building stiffness from moduli for certain symmetries"""

import numpy as np
import pytest

from polycrystal.elasticity.moduli_tools import Isotropic, Cubic, Hexagonal, Triclinic


SYSTEMS = Isotropic.SYSTEMS


@pytest.fixture
def IDENTITY_6():
    """6x6 identity matrix"""
    return np.identity(6)


@pytest.fixture
def IDENTITY_VG():
    """Identity for VOIGT_GAMMA system"""
    return np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


class TestIsotropic:

    def test_identity_KG(self, IDENTITY_6, IDENTITY_VG):
        """Isotropic moduli"""

        iso = Isotropic.from_K_G(1/3, 1/2, system=SYSTEMS.MANDEL)
        assert np.allclose(iso.stiffness.matrix, IDENTITY_6)

        iso = Isotropic.from_K_G(1/3, 1/2, system=SYSTEMS.VOIGT_EPSILON)
        assert np.allclose(iso.stiffness.matrix, IDENTITY_6)

        iso = Isotropic.from_K_G(1/3, 1/2, system=SYSTEMS.VOIGT_GAMMA)
        assert np.allclose(iso.stiffness.matrix, IDENTITY_VG)

    def test_identity_E_nu(self, IDENTITY_6):
        """Test instantiating from E & nu."""
        iso = Isotropic.from_E_nu(1.0, 0.0, system=SYSTEMS.MANDEL)
        assert np.allclose(iso.stiffness.matrix, IDENTITY_6)

    def test_eigenvalues(self):
        """Verify that eigenvalues are correct"""
        # This should give eigenvalues of 3.0 and 2.0.
        iso = Isotropic.from_K_G(1.0, 1.0)

        # First test bulk eigenvector.
        ev_3K_v = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        stiff_v = iso.stiffness.matrix @ ev_3K_v.T
        assert np.allclose(stiff_v.T, 3 * ev_3K_v)

        # Next test shear eigenvectors.
        ev_2G_v = np.diagflat(np.ones(5), -1)[:, :5]
        ev_2G_v[0, :2] = -1
        stiff_v = iso.stiffness.matrix @ ev_2G_v
        assert np.allclose(stiff_v, 2 * ev_2G_v)


class TestCubic:

    def test_identity(self, IDENTITY_6, IDENTITY_VG):
        """Test identiy in all systems"""
        for sys in SYSTEMS:
            cub = Cubic.from_K_Gd_Gs(1/3.0, 1/2.0, 1/2.0, system=sys)
            if sys is SYSTEMS.VOIGT_GAMMA:
                assert np.allclose(cub.stiffness.matrix, IDENTITY_VG)
            else:
                assert np.allclose(cub.stiffness.matrix, IDENTITY_6)

    def test_eigenvalues(self):
        """Cubic moduli"""

        # This should give eigenvalues of 3.0, 2.0 and 4.0
        K, Gd, Gs = 1.0, 1.0, 2.0
        cub = Cubic.from_K_Gd_Gs(K, Gd, Gs, system=SYSTEMS.MANDEL)

        # First test bulk eigenvector.
        ev_3K_v = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        stiff_v = cub.stiffness.matrix @ ev_3K_v.T
        assert np.allclose(stiff_v.T, 3 * ev_3K_v)

        # Next test Gd shear eigenvectors.
        ev_2Gd_v = np.zeros((2, 6))
        ev_2Gd_v[:, 0] = 1
        ev_2Gd_v[0, 1] = -1
        ev_2Gd_v[1, 2] = -1
        stiff_v = cub.stiffness.matrix @ ev_2Gd_v.T
        assert np.allclose(stiff_v, 2 * ev_2Gd_v.T)

        # Next test Gs shear eigenvectors.
        ev_2Gs_v = np.diagflat(np.ones(3), 3)[:3]
        stiff_v = cub.stiffness.matrix @ ev_2Gs_v.T
        assert np.allclose(stiff_v, 4 * ev_2Gs_v.T)

    def test_properties(self):
        """Test system-independent properties"""
        K, Gd, Gs = 3.1, 5.2, 7.3
        cub = Cubic.from_K_Gd_Gs(K, Gd, Gs, system=SYSTEMS.MANDEL)
        for sys in SYSTEMS:
            cub.system = sys
            assert cub.K == K
            assert cub.Gd == Gd
            assert cub.Gs == Gs
        assert cub.isotropic_G == 0.6 * Gs + 0.4 * Gd
        assert cub.zener_A == Gs / Gd


class TestHexagonal:

    def test_identity(self, IDENTITY_6, IDENTITY_VG):
        """Hexagonal moduli"""

        c11, c12, c13, c33, c44 = 1.0, 0.0, 0.0, 1.0, 1.0
        hex = Hexagonal(c11, c12, c13, c33, c44, SYSTEMS.MANDEL)
        assert np.allclose(hex.stiffness.matrix, IDENTITY_6)
        hex.system = SYSTEMS.VOIGT_EPSILON
        assert np.allclose(hex.stiffness.matrix, IDENTITY_6)
        hex.system = SYSTEMS.VOIGT_GAMMA
        assert np.allclose(hex.stiffness.matrix, IDENTITY_VG)
        hex.system = SYSTEMS.MANDEL
        assert np.allclose(hex.stiffness.matrix, IDENTITY_6)


class TestTriclinic:
    """Test triclinic (no) symmetry"""

    def test_systems(self):
        """Test that form changes correctly with systems"""
        cij = np.arange(21)
        tricl = Triclinic(cij, system=SYSTEMS.MANDEL)

        # Check symmetry of stiffness matrix.
        assert np.allclose(tricl.stiffness.matrix, tricl.stiffness.matrix.T)

        # Now check number of entries that change.
        tricl.system = SYSTEMS.VOIGT_EPSILON
        assert np.count_nonzero(cij == tricl.cij) == 12

        tricl.system = SYSTEMS.VOIGT_GAMMA
        assert np.count_nonzero(cij == tricl.cij) == 6

        tricl.system = SYSTEMS.MANDEL
        assert np.allclose(cij, tricl.cij)

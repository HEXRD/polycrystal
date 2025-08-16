"""Tests on building stiffness from moduli for certain symmetries"""

import numpy as np
import pytest

from polycrystal.elasticity.moduli_tools import Isotropic
from polycrystal.utils.tensor_data.mandel_system import MandelSystem


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

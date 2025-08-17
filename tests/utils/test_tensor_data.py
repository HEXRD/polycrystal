"""Test utils.tensor_data"""

import numpy as np
import pytest

from polycrystal.utils.tensor_data.mandel_system import MandelSystem
from polycrystal.utils.tensor_data.voigt_system import VoigtSystem
from polycrystal.utils.tensor_data.symmdev_system import SymmDevSystem


@pytest.fixture
def random_matrices():
    """Return an array of random matrices in a repeatable way"""
    RANDOM_SEED, NUM_MATRICES = 42, 10
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.random((NUM_MATRICES, 3, 3))


def are_symmetric(ma):
    """Check that matrices are symmetric

    Parameters
    ----------
    ma: array(n, 3, 3)
       array of matrices
    """
    return np.allclose(ma, ma.transpose((0, 2, 1)))


def are_skew(ma):
    """Check that matrices are skew"""
    return np.allclose(ma, -ma.transpose((0, 2, 1)))


def are_spherical(ma):
    """Check that matrices are skew"""
    da = [np.diag(3 * (np.trace(A)/3,)) for A in ma]
    return np.allclose(ma, da)


def are_symmdev(ma):
    """Check that matrices are symmetric and deviatoric"""
    tr = [np.trace(A) for A in ma]
    return are_symmetric(ma) and np.allclose(tr, 0)


@pytest.mark.parametrize("Sys", [MandelSystem, VoigtSystem, SymmDevSystem])
def test_system(Sys, random_matrices):
    """test component system

    parameters
    ----------
    Sys: class
       class to test
    m: array(n, 3, 3)
       array of matrices
    """
    ms = Sys(random_matrices)

    if hasattr(ms, "symm"):
        vsym = ms.symm
        msym = Sys.from_parts(symm=vsym)
        assert are_symmetric(msym.matrices)

    if hasattr(ms, "skew"):
        vskw = ms.skew
        mskew = Sys.from_parts(skew=vskw)
        assert are_skew(mskew.matrices)

    if hasattr(ms, "sph"):
        vsph = ms.sph
        msph = Sys.from_parts(sph=vsph)
        assert are_spherical(msph.matrices)

    if hasattr(ms, "symmdev"):
        vsdv = ms.symmdev
        msdv = Sys.from_parts(symmdev=vsdv)
        assert are_symmdev(msdv.matrices)

    # Now test that the original are fully recovered after taking parts and then
    # reconstructing from parts.

    # For Mandel & Voigt.
    if hasattr(ms, "symm"):
        ms2 = Sys.from_parts(symm=vsym, skew=vskw)
        assert np.allclose(ms.matrices, ms2.matrices)

    # For SymmDev.
    if hasattr(ms, "symmdev"):
        ms2 = Sys.from_parts(symmdev=vsdv, skew=vskw, sph=vsph)
        assert np.allclose(ms.matrices, ms2.matrices)

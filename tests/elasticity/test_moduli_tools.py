"""Unit tests for modui tools"""
import numpy as np
import pytest

from polycrystal.elasticity.moduli_tools import Cubic


def check_equal(c1, c2):
    """Check whether two Cubic instances are equal"""
    a1 = np.array([c1.K, c1.Gd, c1.Gs])
    a2 = np.array([c2.K, c2.Gd, c2.Gs])

    assert np.allclose(a1, a2)


@pytest.mark.parametrize(
    "c11, c12, c14", [(2.3, 1.5, 0.1), (3.2, 2.7, 1.3), (1.3, 2.9, 4.3,)]
)
class TestCubic:
    """Test Cubic class"""

    def test_K_Gd_Gs(self, c11, c12, c14):
        c = Cubic(c11, c12, c14)
        K, Gd, Gs = c.K, c.Gd, c.Gs
        cnew = Cubic.from_K_Gd_Gs(K, Gd, Gs)
        check_equal(c, cnew)

"""Tests on stiffness matrices in different systems"""

import numpy as np
import pytest

from polycrystal.elasticity.moduli_tools import stiffness_matrix


SYSTEMS = stiffness_matrix.MatrixComponentSystem


def check_block(m1, m2, block, factor=1.0):
    """Check that matrices agree on a 3x3 block up to a factor

    Parameters
    ----------
    m1, m2: array(6, 6)
       matrices
    block: int
       {0, 1, 2, 3} for upper left, upper right, lower left, lower right
    factor: float, default = 1.0
       scaling factor

    Returns
    -------
    bool:
       true if m2 == factor * m1 on the given block
    """
    s0, s3 = slice(0, 3), slice(3, 6)
    d = {0: (s0, s0), 1: (s0, s3), 2: (s3, s0), 3: (s3, s3)}
    sl = d[block]
    return np.allclose(factor * m1[sl], m2[sl])


class TestStiffness:

    @pytest.fixture
    def all_ones_vg(cls):
        all_1 = np.ones((6, 6))
        cij_upper = all_1[ np.triu_indices(6)]
        units = "MPa"
        return stiffness_matrix.StiffnessMatrix(
            cij_upper, SYSTEMS.VOIGT_GAMMA, units
        )

    def test_ones(self, all_ones_vg):
        """Use matrix of all ones to test scaling factors of each system"""
        all_1 = all_ones_vg.matrix.copy()

        all_ones_vg.system = SYSTEMS.VOIGT_EPSILON
        assert check_block(all_1, all_ones_vg.matrix, 0)
        assert check_block(all_1, all_ones_vg.matrix, 1, 2.0)
        assert check_block(all_1, all_ones_vg.matrix, 2)
        assert check_block(all_1, all_ones_vg.matrix, 3, 2.0)

        all_ones_vg.system = SYSTEMS.MANDEL
        s2 = np.sqrt(2)
        assert check_block(all_1, all_ones_vg.matrix, 0)
        assert check_block(all_1, all_ones_vg.matrix, 1, s2)
        assert check_block(all_1, all_ones_vg.matrix, 2, s2)
        assert check_block(all_1, all_ones_vg.matrix, 3, 2.0)

        # Test that you get the original when resetting to VOIGT_GAMMA.
        all_ones_vg.system = SYSTEMS.VOIGT_GAMMA
        assert check_block(all_1, all_ones_vg.matrix, 0)
        assert check_block(all_1, all_ones_vg.matrix, 1)
        assert check_block(all_1, all_ones_vg.matrix, 2)
        assert check_block(all_1, all_ones_vg.matrix, 3)

    def test_units(self, all_ones_vg):

        mat_gpa = all_ones_vg.matrix
        all_ones_vg.units = "MPa"
        np.allclose(all_ones_vg.matrix, 1000 * mat_gpa)

        # Now, change system too.
        all_ones_vg.system = SYSTEMS.VOIGT_EPSILON
        mat_ve_gpa = all_ones_vg.matrix
        all_ones_vg.units = "MPa"
        np.allclose(all_ones_vg.matrix, 1e-3 * mat_ve_gpa)

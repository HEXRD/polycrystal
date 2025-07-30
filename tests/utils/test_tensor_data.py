"""Test utils.tensor_data"""

import numpy as np

from polycrystal.utils.tensor_data.mandel_system import MandelSystem



class TestMandelSystem:

    def test_to_from(self):

        mat = np.arange(9).reshape(3, 3)
        m1 = MandelSystem(mat)
        m2 = MandelSystem.from_parts(m1.symm, m1.skew)

        assert np.allclose(m1.matrices, m2.matrices)

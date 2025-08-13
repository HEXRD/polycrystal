"""Handler for hexagonal moduli"""

from .base_moduli import BaseModuli
from .stiffness_matrix import StiffnessMatrix

import numpy as np


class Hexagonal(BaseModuli):
    """Class for handling hexagonal moduli

    Parameters
    ----------
    c11, c12: float
       elastic modulus coefficients
    system: Enum
       MatrixComponentSystem attribute
    """

    def __init__(self, c11, c12, c13, c33, c44, system=BaseModuli.DEFAULT_SYSTEM):
        self.c11 = c11
        self.c12 = c12
        self.c13 = c13
        self.c33 = c33
        self.c44 = c44
        self.init_system(system)

    def stiffness_from_moduli(self):
        """Independent moduli to matrix"""
        c11 = c22 = self.c11
        c12 = self.c12
        c13 = c23 = self.c13
        c33 = self.c33

        c44 = c55 = self.c44
        cdiff_12 = (c11 - c12)

        if self.system is self.SYSTEMS.VOIGT_GAMMA:
            c66 = 0.5 * cdiff_12
        else:
            c66 = cdiff_12

        cij = self._high_symmetry_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66)

        return StiffnessMatrix(cij, self.system)

    def moduli_from_stiffness(self):
        """Independent moduli to matrix"""
        m = self.stiffness.matrix
        self.c11 = m[0, 0]
        self.c12 = m[0, 1]
        self.c13 = m[0, 2]
        self.c33 = m[2, 2]
        self.c44 = m[3, 3]

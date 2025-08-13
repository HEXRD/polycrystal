"""Handler for triclinic moduli"""

from .base_moduli import BaseModuli
from .stiffness_matrix import StiffnessMatrix

import numpy as np


class Triclinic(BaseModuli):
    """Class for handling triclinic moduli

    Parameters
    ----------
    cij: array(21)
       elastic modulus coefficients
    system: Enum
       MatrixComponentSystem attribute
    """

    def __init__(self, cij, system=BaseModuli.SYSTEMS.MANDEL):
        self.cij = cij
        self.init_system(system)

    def stiffness_from_moduli(self):
        """Independent moduli to matrix"""
        return StiffnessMatrix(self.cij, self.system)

    def moduli_from_stiffness(self):
        """Independent moduli to matrix"""
        m = self.stiffness.matrix
        self.cij = m[np.triu_indices(6)]

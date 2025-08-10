"""Isotropic moduli handler"""

from .base_moduli import BaseModuli
from .stiffness_matrix import StiffnessMatrix

import numpy as np


class Isotropic(BaseModuli):
    """A class for computing c11 and c12 from various forms of the elastic
    moduli.

    Parameters
    ----------
    c11, c12: float
       elastic modulus coefficients
    system: Enum
       MatrixComponentSystem attribute
    """

    def __init__(self, c11, c12, system=BaseModuli.SYSTEMS.MANDEL):
        self.c11 = c11
        self.c12 = c12
        self._system = system
        self._stiffness = self.stiffness_from_moduli()

    @classmethod
    def from_K_G(cls, K, G):
        """Initialize from bulk and shear moduli

        Parameters
        ----------
        K: float
           bulk modulus
        G: float
           shear modulus
        """
        c11 = (3*K + 4*G)/3.
        c12 = (3*K - 2*G)/3.
        return cls(c11, c12)

    @classmethod
    def from_E_nu(cls, E, nu):
        """Initialize from Young's modulus and Poisson ratio


        Parameters
        ----------
        E: float
           bulk modulus
        nu: float
           shear modulus
       """
        K = E/(1 - 2*nu)/3.
        G = E/(1 + nu)/2.
        return cls.from_K_G(K, G)

    def stiffness_from_moduli(self):
        """Independent moduli to matrix"""
        c11 = c22 = c33 = self.c11
        c12 = c13 = c23 = self.c12
        cdiff_12 = c11 - c12
        if self.system is self.SYSTEMS.VOIGT_GAMMA:
            c44 = c55 = c66 = 0.5 * cdiff_12
        else:
            c44 = c55 = c66 = cdiff_12

        cij = self._high_symmetry_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66)

        return StiffnessMatrix(cij, self.system)

    def moduli_from_stiffness(self):
        """Independent moduli to matrix"""
        m = self.stiffness.matrix
        self.c11 = m[0, 0]
        self.c12 = m[0, 1]

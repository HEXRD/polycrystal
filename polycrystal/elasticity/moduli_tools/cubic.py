"""Isotropic moduli handler"""

from .base_moduli import BaseModuli
from .stiffness_matrix import StiffnessMatrix

import numpy as np


class Cubic(object):
    """Class for initializing cubic moduli

    Parameters
    ----------
    c11, c12, c44: float
       elastic modulus coefficients
    system: MatrixComponentSystem, default = MatrixComponentSystem.VOIGT
       enum values: MatrixComponentSystem.VOIGT or MatrixComponentSystem.MANDEL
    """
    def __init__(self, c11, c12, c44, system=MatrixComponentSystem.MANDEL):
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self._system = system
        self._stiffness = self.stiffness_from_moduli()


    @classmethod    def stiffness_from_moduli(self):
        """Independent moduli to matrix"""
        c11 = c22 = c33 = self.c11
        c12 = c13 = c23 = self.c12
        c44 = c55 = c66 = self.c44

        cij = self._high_symmetry_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66)

        return StiffnessMatrix(cij, self.system)

    def moduli_from_stiffness(self):
        """Independent moduli to matrix"""
        m = self.stiffness.matrix
        self.c11 = m[0, 0]
        self.c12 = m[0, 1]
        self.c44 = m[3, 3]


    def from_K_Gd_Gs(cls, K, Gd, Gs, system=MatrixComponentSystem.MANDEL)):
        """Initialize from bulk and anisotropic shear moduli

        Parameters
        ----------
        K: float
           bulk modulus
        Gd, Gs: float
           anisotropic shear moduli
        """
        c11 = (3*K + 4*Gd)/3.
        c12 = (3*K - 2*Gd)/3.
        c44 = Gs

        return cls(c11, c12, c44)

    @property
    def K(self):
        """Bulk modulus"""
        return (self.c11 + 2 * self.c12) / 3.

    @property
    def Gd(self):
        """Shear modulus involving diagonal elastic strains"""
        return (self.c11 - self.c12) / 2.

    @property
    def Gs(self):
        """Shear modulus involving off-diagonal elastic strains"""
        return self.c44 # this depends on system

    @property
    def isotropic_G(self):
        """Average isotropic modulus for uniform orientation distribution"""
        return 0.6 * self.Gs + 0.4 * Gd

    @property
    def zener_A(self):
        """Zener's anisotropic ratio"""
        return 2 * self.c44/(self.c11 - self.c12)

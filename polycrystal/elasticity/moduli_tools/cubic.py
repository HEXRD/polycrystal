"""Cubic moduli handler

TO DO
-----

* Update properties below.
"""

from .base_moduli import BaseModuli, DEFAULT_UNITS
from .stiffness_matrix import StiffnessMatrix

import numpy as np


class Cubic(BaseModuli):
    """Class for initializing cubic moduli

    Parameters
    ----------
    c11, c12, c44: float
       elastic modulus coefficients
    system: Enum
       MatrixComponentSystem
    """
    def __init__(self, c11, c12, c44,
                 system=BaseModuli.SYSTEMS.MANDEL,
                 units=DEFAULT_UNITS
                 ):
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self.init_system(system)

    @property
    def cij(self):
        return (self.c11, self.c12, self.c44)

    def stiffness_from_moduli(self):
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


    @classmethod
    def from_K_Gd_Gs(cls, K, Gd, Gs, system=BaseModuli.SYSTEMS.MANDEL):
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
        if system is BaseModuli.SYSTEMS.VOIGT_GAMMA:
            c44 = Gs
        else:
            c44 = 2.0 * Gs

        return cls(c11, c12, c44, system)

    @property
    def K(self):
        """Bulk modulus"""
        # This is independent of system.
        return (self.c11 + 2 * self.c12) / 3.

    @property
    def Gd(self):
        """Shear modulus involving diagonal elastic strains"""
        return (self.c11 - self.c12) / 2.

    @property
    def Gs(self):
        """Shear modulus involving off-diagonal elastic strains"""
        return self.c44 if self.system is self.SYSTEMS.VOIGT_GAMMA else 0.5 * self.c44

    @property
    def isotropic_G(self):
        """Average isotropic modulus for uniform orientation distribution"""
        return 0.6 * self.Gs + 0.4 * self.Gd

    @property
    def zener_A(self):
        """Zener's anisotropic ratio"""
        return self.Gs / self.Gd

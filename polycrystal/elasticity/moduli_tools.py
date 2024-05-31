"""Parameter conversions for moduli"""
import numpy as np


class Isotropic(object):
    """A class for computing c11 and c12 from various forms of the elastic
    moduli.

    Parameters
    ----------
    c11, c12: float
       elastic modulus coefficients
    """

    def __init__(self, c11, c12):
        self.c11 = c11
        self.c12 = c12

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


class Cubic(object):
    """Class for initializing cubic moduli

    Parameters
    ----------
    c11, c12, c44: float
       elastic modulus coefficients
    """
    def __init__(self, c11, c12, c44):
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44

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
        return self.c44

    @property
    def isotropic_G(self):
        """Average isotropic modulus for uniform orientation distribution"""
        return 0.6 * self.Gs + 0.4 * Gd

    @property
    def zener_A(self):
        """Zener's anisotropic ratio"""
        return 2 * self.c44/(self.c11 - self.c12)

    @classmethod
    def from_K_Gd_Gs(cls, K, Gd, Gs):
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

"""Parameter conversions for moduli"""

from enum import Enum, auto

import numpy as np


class ComponentSystem(Enum):
    """Possible systems for components of symmetric tensors"""
    VOIGT = auto()
    MANDEL = auto()


component_system_dict = {s.name: s for s in ComponentSystem}
print(component_system_dict)


class SymmetryNames(Enum):
    """Names for crystal symmetry groups"""
    TRICLINIC = "triclinic"
    ISOTROPIC = "isotropic"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


symmetry_names_dict = {sn.value: sn for sn in SymmetryNames}
print(symmetry_names_dict)


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
    system: ComponentSystem, default = ComponentSystem.VOIGT
       enum values: ComponentSystem.VOIGT or ComponentSystem.MANDEL
    """
    def __init__(self, c11, c12, c44, system=ComponentSystem.VOIGT):
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self.system = system

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


class MatrixBuilder:
    """Build 6x6 matrix acting on symmetric matrices

    Parameters
    ----------
    symm: str
       one of {"triclinic", "isotropic", "hexagonal", "cubic"};
       see :py:class:`SymmetryNames`
    """
    c44scale_d = {ComponentSystem.VOIGT: 2.0, ComponentSystem.MANDEL: 2.0}

    def __init__(self, symm):
        self.symm = symm
        print("symmetry: ", symm)

    def cij_to_stiffness(self, cij, system):
        """build stiffness from minimal set of cij for each symmetry

        Parameters
        ----------
        cij: array(n)
           array of independent moduli; length depends on symmetry
        system: ComponentSystem
           enum value indicating representation of symmetric matrices; values can
           be in {"MANDEL", "VOIGT"}
        """
        c44_scale = self.c44scale_d[system]

        if self.symm == SymmetryNames.TRICLINIC.value:
            return self._fill_matrix(cij)

        if self.symm == SymmetryNames.ISOTROPIC.value:
            c11 = c22 = c33 = cij[0]
            c12 = c13 = c23 = cij[1]
            c44 = c55 = c66 = 0.5 * (c11 - c12) * c44_scale

        elif self.symm == SymmetryNames.CUBIC.value:
            c11 = c22 = c33 = cij[0]
            c12 = c13 = c23 = cij[1]
            c44 = c55 = c66 = cij[2] * c44_scale

        elif self.symm == SymmetryNames.HEXAGONAL.value:
            #
            c11 = c22 = cij[0]
            c12 = cij[1]
            c13 = c23 = cij[2]
            c33 = cij[3]
            c44 = c55 = cij[4] * c44_scale
            c66 = 0.5 * (c11 - c12) * c44_scale

        return self.to_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66)


    @staticmethod
    def to_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66):
        z = 0.0
        return np.array(
            [[c11, c12, c13, z, z, z],
             [c12, c22, c23, z, z, z],
             [c13, c23, c33, z, z, z],
             [z, z, z,     c44, z, z],
             [z, z, z,     z, c55, z],
             [z, z, z,     z, z, c66]])

    @staticmethod
    def _fill_matrix(cij):
        n = 6
        mat = np.zeros((n, n))
        indices = np.triu_indices(n)
        mat[indices] = cij
        # Now fill in the bottom.
        for i in range(6):
            i1 = i + 1
            mat[i1:, i] = mat[i, i1:]

        return mat

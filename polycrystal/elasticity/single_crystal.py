"""Elasticity tools for single crystals"""

from collections.abc import Callable

import numpy as np

from polycrystal.utils.tensor_data.mandel_system import MandelSystem
from polycrystal.utils.tensor_data.voigt_system import VoigtSystem

from .moduli_tools import moduli_handler, component_system, Isotropic, Cubic
from .moduli_tools.stiffness_matrix import DEFAULTS, MatrixComponentSystem

SYSTEMS = MatrixComponentSystem


class SingleCrystal:
    """Elastic single crystal

    Parameters
    ----------
    sym: str
       name of symmetry; one of {"triclinic", "isotropic", "cubic", "hexagonal"}
    cij: list | tuple | array
       sequence of independent matrix values; by symmetry, they are:
         "isotropic": (c11, c12)
         "cubic": (c11, c12, c44)
         "hexagonal": (c11, c12, c13, c33, c44)
         "triclinic": (c11, c12, ...), 21 values upper triangle of matrix
    name: str, optional
       name to use for the material
    system: str, default = "VOIGT_GAMMA"
       system to use for representation of symmetric matrices; choices are
       {"VOIGT_GAMMA", VOIGT_EPSILON", "MANDEL"}
    units: str, default = "GPa"
       units of stiffness matrix
    cte: float | array(3, 3) | function, default = None
       coefficient of thermal expansion; a single value for isotropic materials
       or a 3 x 3 array in the crystal frame; in the most general case, it can
       be a function of reference temperature; note that the functional form can
       be made to handle units

    Attributes
    ----------
    system: Enum attribute
       matrix component system
    units: str
       units of stiffness matrix
    symm: BaseModuli
       moduli handler for symmetry
    cij: array(n)
       array of indpendent moduli for the material cyrstal symmetry
    name: str
       name of material
    stiffness: matrix(6, 6)
       stiffness matrix for `system`
    compliance: matrix(6, 6)
       compliance matrix for `system`

    Methods
    -------
    from_K_G:
       Instantiate from bulk and shear moduli.
    from_E_nu:
       Instantiate from Young's modulus and Poisson ratio.
    apply_stiffness:
       apply the stiffness to array of strain tensors, possibly in a rotated frame
    apply_compliance:
       apply the compliance to array of stress tensors, possibly in a rotated frame
    cte:
       coefficient of thermal expansion as a function of a reference temperature
    """

    _MSG_NOT_IMPLEMENTED = "This function is not currently implemented."
    DEFAULT_SYSTEM = SYSTEMS.VOIGT_GAMMA

    system_d = {
        SYSTEMS.VOIGT_GAMMA: VoigtSystem,
        SYSTEMS.VOIGT_EPSILON: VoigtSystem,
        SYSTEMS.MANDEL: MandelSystem,
    }

    def __init__(
            self, symm, cij,
            name='<no name>',
            system=DEFAULT_SYSTEM,
            units="GPa",
            cte=None
    ):
        self.symm = symm
        self.name = name

        self._system = component_system(system)
        ModuliHandler = moduli_handler(symm)
        if symm == "triclinic":
            self.moduli = ModuliHandler(cij, self.system, units)
        else:
            self.moduli = ModuliHandler(*cij, self.system, units)

        # Check CTE (coefficient of thermal expansion) value.
        if not isinstance(cte, (type(None), float, np.ndarray, Callable)):
            raise TypeError("Unexpected type for `cte` argument")

        if isinstance(cte, np.ndarray):
            if cte.shape != (3, 3):
                raise RuntimeError("CTE shape is not 3x3")

        self._cte = cte

    @classmethod
    def from_K_G(cls, K, G, **kwargs):
        """Instantiate from K and G

        Parameters
        ----------
        K: float
           bulk modulus
        G: float
           shear modulus
        """
        return cls("isotropic", Isotropic.cij_from_K_G(K, G), **kwargs)

    @classmethod
    def from_E_nu(cls, E, nu, **kwargs):
        """Instantiate from Young's modulus and Poisson ratio

        Parameters
        ----------
        E: float
           Young's modulus
        nu: float
           Poisson ratio
        """
        return cls("isotropic", Isotropic.cij_from_E_nu(E, nu), **kwargs)

    @classmethod
    def from_K_Gd_Gs(cls, K, Gd, Gs, **kwargs):
        """Instantiate from K and G

        Parameters
        ----------
        K: float
           bulk modulus
        G_d: float
           shear modulus for diagonal
        G_s: float
           shear modulus for off-diagonal
        """
        system = kwargs.get("system", cls.DEFAULT_SYSTEM)
        cij =  Cubic.cij_from_K_Gd_Gs(K, Gd, Gs, system)
        return cls("cubic", cij, **kwargs)

    @property
    def system(self):
        """Input system for matrix components"""
        return self._system

    @system.setter
    def system(self, v):
        """Set method for system"""
        self._system = component_system(v)
        self.moduli.system = self._system

    @property
    def units(self):
        """Output units for moduli"""
        return self.moduli.units

    @units.setter
    def units(self, v):
        """Set method for output_units"""
        self.moduli.units = v

    @property
    def cij(self):
        """Return moduli for the input system"""
        return self.moduli.cij

    @property
    def stiffness(self):
        """Stiffness matrix in crystal coordinates"""
        return self.moduli.stiffness.matrix

    @property
    def compliance(self):
        """Compliance matrix in crystal coordinates"""
        return np.linalg.inv(self.stiffness)

    def cte(self, reftemp):
        """Coefficient of Thermal Expansion

        This is a user-defined function that returns the CTE value for a given reference
        temperature. The input `reftemp` might include units information as well.

        Parameters
        ----------
        reftemp: user-defined
            reference temperature, possibly including units

        Returns
        -------
        array(3, 3):
           the 3 x 3  CTE matrix
        """
        if self._cte is None:
            raise RuntimeError("No CTE (Coefficient of Thermal Expansion) has been set.")
        elif isinstance(self._cte, float):
            return np.diag(3 * (self._cte,))
        elif isinstance(self._cte, np.ndarray):
            return self._cte
        elif isinstance(self._cte, Callable):
            return self._cte(reftemp)
        else:
            raise RuntimeError("Cannot create CTE function.")

    def apply_stiffness(self, eps, rmat=None):
        """Stress tensors from strain tensors in same reference frame

        Parameters
        ----------
        eps: array(n, 3, 3)
           array of strains, in a common (sample or crystal) reference frame
        rmat: None or array(n, 3, 3)
           array of rotation matrices taking crystal components to sample, or None,
           if the strains are already in crystal components

        Returns
        -------
        sig: array(n, 3, 3)
           array of stresses in same frame as strains
        """


        # Here is the main calculation.

        System = self.system_d[self.system]
        eps_s_mat = self._to_3d(eps)

        eps_c_mat = self._change_basis(eps_s_mat, rmat)

        eps_c_vec = System(eps_c_mat).symm
        if self.system is SYSTEMS.VOIGT_GAMMA:
            eps_c_vec[:, 3:] *= 2.0

        sig_c_vec = self.stiffness @ eps_c_vec.T

        sig_c_mat = System.from_parts(symm=sig_c_vec.T).matrices

        sig_s_mat = sig_c_mat if rmat is None else (
            self._change_basis(sig_c_mat, rmat.transpose((0, 2, 1)))
        )

        return sig_s_mat

    def apply_compliance(self, sig, rmat=None):
        """Stress tensors from strain tensors in same reference frame

        Parameters
        ----------
        sig: array(n, 3, 3)
           array of stresses, in a common (sample or crystal) reference frame
        rmat: None or array(n, 3, 3)
           array of rotation matrices taking crystal components to sample, or None,
           if the stresses are already in crystal components
        Returns
        -------
        eps: array(n, 3, 3)
           array of strains in same frame as stresses
        """

        System = self.system_d[self.system]
        sig_s_mat = self._to_3d(sig)

        sig_c_mat = self._change_basis(sig_s_mat, rmat)

        sig_c_vec = System(sig_c_mat).symm

        eps_c_vec = self.compliance @ sig_c_vec.T
        if self.system is SYSTEMS.VOIGT_GAMMA:
            eps_c_vec[3:, :] *= 0.5

        eps_c_mat = System.from_parts(symm=eps_c_vec.T).matrices

        eps_s_mat = eps_c_mat if rmat is None else (
            self._change_basis(eps_c_mat, rmat.transpose((0, 2, 1)))
        )

        return eps_s_mat

    @staticmethod
    def _to_3d(arr):
        """Make sure that array is 3-dimensional and of shape (n, 3, 3)"""
        if arr is None:
            return arr

        if arr.ndim == 2:
            if arr.shape != (3, 3):
                raise RuntimeError("array shape not 3x3")
            return arr.reshape((1, 3, 3))
        elif arr.ndim != 3:
            raise RuntimeError("array shape incrorrect")
        else:
            assert arr.ndim == 3
            return arr

    @staticmethod
    def _change_basis(mat, rot):
        """Change of basis taking M -> R @ M @ R.T

        The typical use here is converting a matrix in cyrstal components to
        one in sample components, where R is the matrix that takes cyrstal components
        of vectors to sample components. In that case:

        M_s = R @ M_c @ R.T

        because evec_c = R.T evec_s, and svec_c = M_c @ evec_c, and svec_s = R @ svec_c
        So svec_s = (R @ M_c @ R.T) evec_s.

        mat: array(n, 3, 3)
           array of matrices
        rot: array(n, 3, 3) or None
           array of rotation matrices, if reference frame is not crystal
        """
        if rot is None:
            return mat
        else:
            if len(mat) != len(rot):
                msg = (
                    "rotation array needs to be the same "
                    "length as the matrix array"
                )
                raise ValueError(msg)
            return rot @ mat @ rot.transpose((0, 2, 1))

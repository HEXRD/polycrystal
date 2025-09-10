"""Elasticity tools for single crystals"""

import numpy as np

from polycrystal.utils.tensor_data.mandel_system import MandelSystem
from polycrystal.utils.tensor_data.voigt_system import VoigtSystem

from .moduli_tools import moduli_handler, component_system, Isotropic
from .moduli_tools.stiffness_matrix import DEFAULT_UNITS

SYSTEMS = Isotropic.SYSTEMS


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
    input_system: str, default = "VOIGT_GAMMA"
       system to use for representation of symmetric matrices; choices are
       {"MANDEL", "VOIGT_GAMMA", VOIGT_EPSILON"}
    output_system: str, default = "MANDEL"
       system to use for representation of stiffness matrix; same choices
    input_units: str, default = DEFAULT_UNITS
       units of input moduli
    output_units: str, default=DEFAULT_UNITS
       units of output stiffness
    cte: float | array(3, 3)
       coefficient of thermal expansion; a single value for isotropic materials
       or a 3 x 3 array in the crystal frame

    Attributes
    ----------
    cte: array(3, 3) or float
       coefficient of thermal expansion, if specified
    input_system, output_system: Enum attribute
       matrix component system
    symm: BaseModuli
       moduli handler for symmetry
    cij: array(n)
       array of indpendent moduli for the material cyrstal symmetry
    nmae: str
       name of material
    stiffness: matrix(6, 6)
       stiffness matrix for output_system`
    compliance: matrix(6, 6)
       compliance matrix for output_system`

    Methods
    -------
    from_K_G:
       Instantiate from bulk and shear moduli.
    from_E_nu:
       Instantiate from Young's modulus and Poisson ratio.
    apply_stiffness:
       apply the stiffness to array of strain tensors, possibly in a rotated frame
    """

    _MSG_NOT_IMPLEMENTED = "This function is not currently implemented."

    system_d = {
        SYSTEMS.VOIGT_GAMMA: VoigtSystem,
        SYSTEMS.VOIGT_EPSILON: VoigtSystem,
        SYSTEMS.MANDEL: MandelSystem,
    }

    def __init__(
            self, symm, cij,
            name='<no name>',
            input_system="VOIGT_GAMMA",
            output_system="MANDEL",
            input_units=DEFAULT_UNITS,
            output_units=DEFAULT_UNITS,
            cte=None
    ):
        self.symm = symm
        self._cij = np.array(cij).copy()
        self.name = name

        # This section sets up the moduli handler. The `input_system` is used only
        # on instantiation and is read-only. The `output_system` is set after the
        # handler is instantiated because the handler uses the `output_system` to
        # get the right matrix output. The `output_system` set() method uses the
        # handler. The `units` are initialized with the `input_units` and then
        # reset to the `output_units`.

        self._input_system = component_system(input_system)
        ModuliHandler = moduli_handler(symm)
        if symm == "triclinic":
            self.moduli = ModuliHandler(
                self.cij, system=self.input_system, units=input_units
            )
        else:
            self.moduli = ModuliHandler(
                *self.cij, system=self.input_system, units=input_units
            )
        self.moduli.units = output_units

        self.output_system = component_system(output_system)

        # Now for the units.
        self._input_units = input_units

        # Set CTE (coefficient of thermal expansion)
        if cte is not None:
            if isinstance(cte, np.ndarray):
                if cte.shape != (3, 3):
                    raise RuntimeError("CTE shape is not 3x3")
                self.cte = cte
            elif isinstance(cte, (float, int)):
                self.cte = np.diag(3 * (cte,))

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
        iso = Isotropic.from_K_G(K, G)
        cij = [iso.c11, iso.c12]
        return cls("isotropic", cij, **kwargs)

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
        iso = Isotropic.from_E_nu(E, nu)
        cij = [iso.c11, iso.c12]
        return cls("isotropic", cij, **kwargs)

    @property
    def input_system(self):
        """Input system for matrix components"""
        return self._input_system

    @property
    def output_system(self):
        """Output system for matrix components"""
        return self._output_system

    @output_system.setter
    def output_system(self, v):
        """Set method for output_system"""
        self._output_system = component_system(v)
        self.moduli.system = self._output_system

    @property
    def input_units(self):
        return self._input_units

    @property
    def output_units(self):
        """Output units for moduli"""
        return self.moduli.units

    @output_units.setter
    def output_units(self, v):
        """Set method for output_units"""
        self.moduli.units = v

    @property
    def cij(self):
        """Return moduli for the input system"""
        return self._cij

    @property
    def cij_in(self):
        """Return moduli for the input system"""
        return self.cij

    @property
    def cij_out(self):
        """Return moduli for the output system"""
        return self.moduli.cij

    @property
    def stiffness(self):
        """Stiffness matrix in crystal coordinates"""
        return self.moduli.stiffness.matrix

    @property
    def compliance(self):
        """Compliance matrix in crystal coordinates"""
        return np.linalg.inv(self.stiffness)

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

        System = self.system_d[self.output_system]
        eps_s_mat = self._to_3d(eps)

        eps_c_mat = self._change_basis(eps_s_mat, rmat)

        eps_c_vec = System(eps_c_mat).symm
        if self.output_system is SYSTEMS.VOIGT_GAMMA:
            eps_c_vec[3:] *= 2.0

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

        System = self.system_d[self.output_system]
        sig_s_mat = self._to_3d(sig)

        sig_c_mat = self._change_basis(sig_s_mat, rmat)

        sig_c_vec = System(sig_c_mat).symm

        eps_c_vec = self.compliance @ sig_c_vec.T
        if self.output_system is SYSTEMS.VOIGT_GAMMA:
            eps_c_vec[3:] *= 0.5

        eps_c_mat = System.from_parts(symm=eps_c_vec.T).matrices

        eps_s_mat = eps_c_mat if rmat is None else (
            self._change_basis(eps_c_mat, rmat.transpose((0, 2, 1)))
        )

        return eps_s_mat

    @staticmethod
    def _to_3d(arr):
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

        Here the typical use here is converting a matrix in cyrstal components to
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

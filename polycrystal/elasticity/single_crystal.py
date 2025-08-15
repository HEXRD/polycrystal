"""Elasticity tools for single crystals"""

import numpy as np

from .moduli_tools import moduli_handler, component_system, Isotropic


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
    cte: float | array(3, 3)
       coefficient of thermal expansion; a single value for isotropic materials
       or a 3 x 3 array in the crystal frame

    Attributes
    ----------
    cte:
       coefficient of thermal expansion, if specified

    Methods
    -------
    from_K_G:
       Instantiate from bulk and shear moduli.
    from_E_nu:
       Instantiate from Young's modulus and Poisson ratio.
    sample_stiffness:
        Stiffness matrix in sample coordinates.
    sample_compliance:
        Compliance matrix in sample coordinates.
    write:
        Write to a text file.
    read:
        Read from a text file and return new instance.
    """

    _MSG_NOT_IMPLEMENTED = "This function is not currently implemented."

    def __init__(
            self, symm, cij,
            name='<no name>',
            input_system= "VOIGT_GAMMA",
            output_system= "MANDEL",
            cte=None
    ):
        self.symm = symm
        self.cij = np.array(cij).copy()
        self.name = name

        # This section sets up the moduli handler. The `input_system` is used only on
        # instantiation and is read-only. The `output_system` is set after the
        # handler is instantiated because the handler uses the `output_system` to
        # get the right matrix output. The `output_system` set() method uses the
        # handler.

        self._input_system = component_system(input_system)
        ModuliHandler = moduli_handler(symm)
        if symm == "triclinic":
            self.moduli = ModuliHandler(self.cij, system=self.input_system)
        else:
            self.moduli = ModuliHandler(*self.cij, system=self.input_system)

        self.output_system = component_system(output_system)

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
        iso = isotropic.Isotropic.from_E_nu(E, nu)
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
    def stiffness(self):
        """Stiffness matrix in crystal coordinates"""
        return self.moduli.stiffness.matrix

    @property
    def compliance(self):
        """Compliance matrix in crystal coordinates"""
        return np.linalg.inv(self.stiffness)

    def sample_stiffness(self, R):
        """Stiffness matrix in sample coordinates

        Parameters
        ----------
        R: orientation matrix taking crystal components to sample

        Returns
        -------
        matrix (6, 6)
           stiffness matrix in sample frame
        """
        raise NotImplementedError(self._MSG_NOT_IMPLEMENTED)

    def sample_compliance(self, R):
        """Compliance matrix in sample coordinates


        Parameters
        ----------
        R: orientation matrix taking crystal components to sample

        Returns
        -------
        matrix (6, 6)
           stiffness matrix in sample frame
        """
        raise NotImplementedError(self._MSG_NOT_IMPLEMENTED)


    def write(self, fname):
        """write to a text file

        Parameters
        ----------
        fname: str | Path
           name of file to write to
        """
        raise NotImplementedError(self._MSG_NOT_IMPLEMENTED)

    @classmethod
    def read(cls, fname):
        """Read from a text file and return new instance

        Parameters
        ----------
        fname: str | Path
           name of file to read
        """
        raise NotImplementedError(self._MSG_NOT_IMPLEMENTED)

"""Elasticity tools for single crystals"""
import numpy as np

from .moduli_tools import Isotropic


SCALE_C44 = 2.0


class SingleCrystal:
    """Elastic single crystal

    Parameters
    ----------
    sym: str
       name of symmetry
    cij: list | tuple | array
       sequence of independent matrix values; the order is (c11, c12) for
       isotropic; (c11, c12, c44) for cubic; and (c11, c12, c13, c33, c44) for
       hexagonal.
    name: str, optional
       name to use for the material
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

    def __init__(self, symm, cij, name='<no name>', cte=None):
        self.symm = symm
        self.cij = np.array(cij).copy()
        self.name = name

        self._stiffness = to_stiffness(symm, cij)

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
    def stiffness(self):
        """Stiffness matrix in crystal coordinates"""
        return self._stiffness

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
        return rotate_matrix(self.stiffness, R)

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
        return rotate_matrix(self.compliance, R)

    def write(self, fname):
        """write to a text file

        Parameters
        ----------
        fname: str | Path
           name of file to write to
        """
        with open(fname, "w") as f:
            f.write(self.name)
            f.write(self.symm)
            f.write(self.cij)

    @classmethod
    def read(cls, fname):
        """Read from a text file and return new instance

        Parameters
        ----------
        fname: str | Path
           name of file to read
        """
        with file(fname, "r") as f:
            lines = f.readlines()

        name = lines[0].strip()
        symm = lines[1].strip()
        cij = [float(fi) for fi in lines[2].split()]

        esx = cls(symm, cij, name=name)

        return esx

# ======================================== Utilities


def to_stiffness(sym, cij):
    """build stiffness from minimal set of cij for each symmetry"""
    if sym.startswith("iso"):
        c11 = c22 = c33 = cij[0]
        c12 = c13 = c23 = cij[1]
        c44 = c55 = c66 = cij[0] - cij[1]

    elif sym.startswith("cub"):
        c11 = c22 = c33 = cij[0]
        c12 = c13 = c23 = cij[1]
        c44 = c55 = c66 = cij[2]*SCALE_C44

    elif sym.startswith("hex"):
        #
        c11 = c22 = cij[0]
        c12 = cij[1]
        c13 = c23 = cij[2]
        c33 = cij[3]
        c44 = c55 = cij[4]*SCALE_C44
        #
        c66 = (c11 - c12)

    return to_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66)


def to_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66):
    z = 0.0
    return np.array(
        [[c11, c12, c13, z, z, z],
         [c12, c22, c23, z, z, z],
         [c13, c23, c33, z, z, z],
         [z, z, z,     c44, z, z],
         [z, z, z,     z, c55, z],
         [z, z, z,     z, z, c66]])


def to_6vec(A):
    """Return a six-vector representing matrix A"""
    return np.array([A[0,0], A[1,1], A[2,2], A[1,2], A[0,2], A[0,1]])


def to_3x3(a):
    """Return 3x3 matrix for six-vector a"""
    return np.array([[a[0], a[5], a[4]],
                     [a[5], a[1], a[3]],
                     [a[4], a[3], a[2]]])


def rotation_operator(R):
    """Return 6x6 matrix for applying a rotation to a 3x3 symmetric tensor

    If Rc = s (taking crystal components to sample), then
    L(M) = R*M*R^T, (taking crystal components, M, to sample components)
"""
    L = np.zeros((6,6))
    id = np.identity(6)
    for i in range(6):
        a = id[i]
        A = to_3x3(a)
        LA = np.dot(np.dot(R, A), R.T)
        L[:, i] = to_6vec(LA)

    return L


def rotate_matrix(matrix, R):
    """return rotated matrix using rotation R, where Rc=s"""
    LR = rotation_operator(R)
    LRT = rotation_operator(R.T)

    return np.dot(LR, np.dot(matrix, LRT))

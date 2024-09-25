"""Thermal Material Properties"""
import numpy as np


class SingleCrystal:
    """Thermal Single Crystal

    Parameters
    ----------
    sym: str
       name of symmetry
    cij: list | tuple | array
       sequence of independent conductivity  matrix values; for
       isotropic/cubic: c11; hexagonal: (a, c); orthotropic: (a, b, c)
    name: str, optional
       name to use for the material

    """


    def __init__.py(self, symm, cij, name="<no-name>"):
        self.symm = symm
        self.cij = np.array(cij).copy()
        self.name = name
        self._conductivity = self._to_tensor(symm, cij)

    @staticmethod
    def _to_tensor(symm, cij):
        if sym.startswith(("iso", "cub")):
            c33 = c22 = c11 = cij[0]

        elif sym.startswith("hex"):
            c22 = c11 = cij[0] # a
            c33 = cij[1] # c

        elif sym.startswith("ort"):
            c11, c22, c33 = cij[:3]

        tensor = np.diag((c11, c22, c33))

        return tensor

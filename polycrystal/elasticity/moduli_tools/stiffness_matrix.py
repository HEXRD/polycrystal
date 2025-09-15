"""Elastic Stiffness Matrix"""

from collections import namedtuple
from enum import Enum, auto

import numpy as np

import pint


DEFAULT_UNITS = "GPa"

_UT_INDICES = np.triu_indices(6)
_LT_INDICES = np.tril_indices(6, k=-1)
_LT_CIJ = [1, 2, 7, 3, 8, 12, 4, 9, 13, 16, 5, 10, 14, 17, 19]

_Scales = namedtuple("_Scales", ["up_right", "low_left", "low_right"])


class MatrixComponentSystem(Enum):
    """Possible systems for components 6x6 tensors acting on symmetric matrices

    Note that these are based on the component systems in the :py:module:`tensor_data`
    module for 6-component vectors.  They include 'Voigt', 'Mandel' and 'SymmDev'. For
    the 6x6 matrices, there are two options for 'Voigt', one applies the matrix to the
    `gammas` (twice epsilon shears) and the other to the `epsilons`.
    """
    VOIGT_GAMMA = auto()
    VOIGT_EPSILON = auto()
    MANDEL = auto()


component_system_dict = {s.name: s for s in MatrixComponentSystem}


# This gives scaling factors for converting from Voigt_Gamma to the other two
# systems.
_s2 = np.sqrt(2)
vg_to_ve = _Scales(2.0, 1.0, 2.0)
vg_to_md = _Scales(_s2, _s2, 2.0)
other = _Scales(*[1 / vg_to_md[i] for i in range(3)]),

# This is a useful abbreviation for the dictionary.
vg, ve, md = (
    MatrixComponentSystem.VOIGT_GAMMA,
    MatrixComponentSystem.VOIGT_EPSILON,
    MatrixComponentSystem.MANDEL
)
scales_dict = {
    (vg, vg): None,
    (vg, ve): vg_to_ve,
    (vg, md): vg_to_md,
    (ve, ve): None,
    (ve, vg): _Scales(*[1 / vg_to_md[i] for i in range(3)]),
    (ve, vg): _Scales(*[1 / vg_to_ve[i] for i in range(len(vg_to_ve))]),
    (ve, md): _Scales(*[vg_to_md[i] / vg_to_ve[i] for i in range(len(vg_to_ve))]),
    (md, md): None,
    (md, vg): _Scales(*[1 / vg_to_md[i] for i in range(len(vg_to_md))]),
    (md, ve): _Scales(*[vg_to_ve[i] / vg_to_md[i] for i in range(len(vg_to_ve))]),
}


class StiffnessMatrix:

    """Elastic Stiffness Matrix

    Parameters
    ----------
    cij: array(21)
       upper triangle of the 6x6 stiffness matrix
    system: Enum
       item in MatrixComponentSystem class
    units: str
       units of stress
    """

    ureg = pint.UnitRegistry()

    def __init__(self, cij, system, units=DEFAULT_UNITS):
        if len(cij) != 21:
            raise ValueError("cij must have length 21")

        if system not in MatrixComponentSystem:
            raise ValueError("`system` must be an attribtue of MatrixComponentSystem")
        self._system = system

        # Adding `str(units)` makes it work even when the input is a pint.Unit.
        self._units = self.ureg.parse_expression(str(units))
        self._matrix = self._fill_cij(cij) * self.units

    def _fill_cij(self, cij):
        mat = np.zeros((6, 6))
        mat[_UT_INDICES] = cij
        mat[_LT_INDICES] = cij[_LT_CIJ]

        # Note that the VOIGT_EPSILON system may not be symmetric as the lower
        # left 3x3 submatrix is one half the upper right. This corrects for
        # that.
        if self.system is MatrixComponentSystem.VOIGT_EPSILON:
            for i in range(3):
                for j in range(3,6):
                    mat[j, i] = 0.5 * mat[i, j]
        return mat

    @property
    def matrix(self):
        return self._matrix.to_tuple()[0]

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self,  v):
        if v not in MatrixComponentSystem:
            raise ValueError("`system` not in MatrixComponentSystem")

        convert = (self.system, v)
        scale = scales_dict[convert]
        if scale is not None:
            self._rescale_matrix(scale)
            self._system = v

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._matrix.ito(v)
        self._units = v

    def _rescale_matrix(self, scale):
        self._matrix[:3, 3:] *= scale.up_right
        self._matrix[3:, :3] *= scale.low_left
        self._matrix[3:, 3:] *= scale.low_right

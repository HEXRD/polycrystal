"""Voigt System for Representing 3D Tensors"""

import numpy as np

from .base_system import BaseSystem


class VoigtSystem(BaseSystem):
    """Voigt System for Representing 3D Tensors"""

    _system_name = "Voigt"

    @classmethod
    def to_components(cls, matrices):
        """This sets the `_components` attribute"""
        m = matrices
        return np.array([
            m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
            0.5 * (m[:, 1, 2] + m[:, 2, 1]),
            0.5 * (m[:, 0, 2] + m[:, 2, 0]),
            0.5 * (m[:, 0, 1] + m[:, 1, 0]),
            0.5 * (m[:, 2, 1] - m[:, 1, 2]),
            0.5 * (m[:, 0, 2] - m[:, 2, 0]),
            0.5 * (m[:, 1, 0] - m[:, 0, 1,]),
        ]).T

    @classmethod
    def to_matrices(cls, components):
        cm = components
        m11, m22, m33 = cm[:, 0], cm[:, 1], cm[:, 2]
        m12, m21 = (cm[:, 5] - cm[:, 8]), (cm[:, 8] + cm[:, 5])
        m13, m31 = (cm[:, 4] + cm[:, 7]), (cm[:, 4] - cm[:, 7])
        m23, m32 = (cm[:, 3] - cm[:, 6]), (cm[:, 6] + cm[:, 3])
        m = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        return m.transpose((2, 0, 1))

    @property
    def symm(self):
        """components of symmetric part"""
        return self.components[:, :6]

    @property
    def skew(self):
        """components of skew part"""
        return self.components[:, 6:]

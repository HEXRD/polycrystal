"""Mandel System for Representing 3D Tensors"""

import numpy as np

from .base_system import BaseSystem


_s2 = np.sqrt(2.)
_s2i = 1/_s2


class MandelSystem(BaseSystem):
    """Mandel System for Representing 3D Tensors"""

    @classmethod
    def from_parts(cls, symm=None, skew=None):
        """Build matrices from parts additively

        This builds the matrix by specfiying the symmetric and skew parts
        separately.

        Parameters
        ----------
        symm: array (n, 6)
          symmetric part as 6-vector in Mandel orthonormal basis
        skew: array (n, 3)
          skew part
        """
        len = cls._check_parts(symm, skew)
        if symm is None:
            symm = np.zeros((len, 6))
        if skew is None:
            skew = np.zeros((len, 3))

        comps = np.hstack((symm, skew))
        mats = cls.to_matrices(comps)
        ten = BaseSystem(mats)
        ten.components = comps

        return ten

    @staticmethod
    def _check_parts(symm, skew):
        len = 0
        if symm is not None:
            if symm.ndim > 2:
                raise ValueError("`symm` must be 1- or 2-dimensional")
            symm = np.atleast_2d(symm)
            if symm.shape[1] != 6:
                raise ValueError("`symm` must have second dimension equal to 6")
            len = symm.shape[0]

        if skew is not None:
            if skew.ndim > 2:
                raise ValueError("`skew` must be 1- or 2-dimensional")
            skew = np.atleast_2d(skew)
            if skew.shape[1] != 3:
                raise ValueError("`skew` must have second dimension equal to 3")
            lskew = skew.shape[0]
            if len == 0:
                len = lskew
            else:
                if lskew != len:
                    raise ValueError("`symm` and `skew` arrays have different lengths")

        if len == 0:
            raise ValueError("`symm` and `skew` cannot both be None")

        return len

    @property
    def components(self):
        """full components"""
        if not hasattr(self, "_components"):
            self.components = self.to_components(self.matrices)
        return self._components

    @components.setter
    def components(self, c):
        self._components = c

    @staticmethod
    def to_components(matrices):
        """This sets the `_components` attribute"""
        m = matrices
        return np.array([
            m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
            _s2i * (m[:, 1, 2] + m[:, 2, 1]),
            _s2i * (m[:, 0, 2] + m[:, 2, 0]),
            _s2i * (m[:, 0, 1] + m[:, 1, 0]),
            _s2i * (m[:, 2, 1] - m[:, 1, 2]),
            _s2i * (m[:, 0, 2] - m[:, 2, 0]),
            _s2i * (m[:, 1, 0] - m[:, 0, 1,]),
        ]).T

    @staticmethod
    def to_matrices(components):
        cm = components
        m11, m22, m33 = cm[:, 0], cm[:, 1], cm[:, 2]
        m12, m21 = _s2i * (cm[:, 5] - cm[:, 8]), _s2i * (cm[:, 8] + cm[:, 5])
        m13, m31 = _s2i * (cm[:, 4] + cm[:, 7]), _s2i * (cm[:, 4] - cm[:, 7])
        m23, m32 = _s2i * (cm[:, 3] - cm[:, 6]), _s2i * (cm[:, 6] + cm[:, 3])
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

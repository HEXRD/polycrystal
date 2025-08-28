"""Modified Mandel System for Representing 3D Tensors"""

import numpy as np

from .base_system import BaseSystem


class SymmDevSystem(BaseSystem):
    """Modified Mandel System for 3D Tensors with Symmetric Deviatoric Part"""

    _system_name = "SymmDev"

    @classmethod
    def from_parts(cls, symmdev=None, skew=None, sph=None):
        """Build matrices from parts additively

        This builds the matrix by specfiying the symmetric and skew parts
        separately.

        Parameters
        ----------
        symmdev: array (n, 5)
          symmetric deviatoric part as 5-vector in orthonormal basis
        skew: array (n, 3)
          skew part
        sph: array (n)
          spherical part
        """
        # First, check that everything has compatible shapes.
        dim_sym, dim_skw = 5, 3

        len_sym = cls._check_part(symmdev, dim_sym)
        len_skw = cls._check_part(skew, dim_skw)
        len_sph = cls._check_part(sph) # need to change check_part

        len = max(len_sym, len_skw, len_sph)
        if len == 0:
            raise ValueError("all parts are None")

        emsg = "symmdev, skew and spherical parts all must have same length"
        if len_sym == 0:
            symmdev = np.zeros((len, dim_sym))
        else:
            if len_sym != len:
                raise ValueError(emsg)
            symmdev = symmdev.reshape((len, dim_sym))

        if len_skw == 0:
            skew = np.zeros((len, dim_skw))
        else:
            if len_skw != len:
                raise ValueError(emsg)
            # This handles the 1D array case.
            skew = skew.reshape((len, dim_skw))

        if len_sph == 0:
            sph = np.zeros((len, 1))
        else:
            if len_sph != len:
                raise ValueError(emsg)
            sph = sph.reshape((len, 1))

        comps = np.hstack((sph, symmdev, skew))
        mats = cls.to_matrices(comps)
        ten = cls(mats)
        ten.components = comps

        return ten

    @classmethod
    def to_components(cls, matrices):
        """This sets the `_components` attribute"""
        m = matrices
        return np.array([
            cls._s3i * (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]),
            cls._s2i * (m[:, 0, 0] - m[:, 1, 1]),
            cls._s2i * (m[:, 0, 0] - m[:, 2, 2]),
            cls._s2i * (m[:, 1, 2] + m[:, 2, 1]),
            cls._s2i * (m[:, 0, 2] + m[:, 2, 0]),
            cls._s2i * (m[:, 0, 1] + m[:, 1, 0]),
            cls._s2i * (m[:, 2, 1] - m[:, 1, 2]),
            cls._s2i * (m[:, 0, 2] - m[:, 2, 0]),
            cls._s2i * (m[:, 1, 0] - m[:, 0, 1]),
        ]).T

    @classmethod
    def to_matrices(cls, components):
        cm = components
        m11 = (cls._s3 * cm[:, 0] + cls._s2 * cm[:, 1] + cls._s2 * cm[:, 2]) / 3.0
        m22 = m11 - cls._s2 * cm[:, 1]
        m33 = m11 - cls._s2 * cm[:, 2]
        m12, m21 = cls._s2i * (cm[:, 5] - cm[:, 8]), cls._s2i * (cm[:, 8] + cm[:, 5])
        m13, m31 = cls._s2i * (cm[:, 4] + cm[:, 7]), cls._s2i * (cm[:, 4] - cm[:, 7])
        m23, m32 = cls._s2i * (cm[:, 3] - cm[:, 6]), cls._s2i * (cm[:, 6] + cm[:, 3])
        m = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        return m.transpose((2, 0, 1))

    @property
    def symmdev(self):
        """components of symmetric part"""
        return self.components[:, 1:6]

    @property
    def skew(self):
        """components of skew part"""
        return self.components[:, 6:]

    @property
    def sph(self):
        """components of symmetric part"""
        return self.components[:, 0]

"""Handles Tensor Data including symmetric, deviatoric, and skew parts"""
import numpy as np


_s6i = 1/(_s6 := np.sqrt(6.))
_s3i = 1/(_s3 := np.sqrt(3.))
_s2i = 1/(_s2 := np.sqrt(2.))


class TensorData:
    """Initialize tensordata

    Full matrices are given here, but you may also construct the matrices
    by giving symmetric, symmetric-deviatoric, skew and/or spherical parts
    by using the `from_parts` class method.

    Parameters
    ----------
    matrices: (optional) array (n, 3, 3)
       array of matrices

    """

    def __init__(self, matrices=None):
        self._matrices = matrices

    @classmethod
    def from_parts(cls, symm6=None, symmdev=None, skew=None, sph=None):
        """Build matrices from parts additively

        This builds the matrix by specfiying the symmetric and skew parts
        separately. If you use `symm6`, then you cannot also specify
        `symmdev` or `sph`.

        Parameters
        ----------
        symm: array (n, 6)
          symmetric part as 6-vector in orthonormal basis
        symmdev: array (n, 5)
          symmetric and deviatoric part as 5-vector in orthonormal basis
        skew: array (n, 3)
          skew part
        sph: array (n)
          spherical part
        """
        # Check for specification of both symm and symmdev
        if ((symm6 is not None) and
            ((symmdev is not None) or (sph is not None))):
            raise ValueError("Cannot specify both symm6 and symmdev or sph")

        m, n = None, 0
        if symm6 is not None:
            m = cls._matrices_from_symm6(symm6)
            n = len(m)

        if symmdev is not None:
            m = cls._matrices_from_symmdev(symmdev)
            n = len(m)

        if skew is not None:
            if m is not None:
                m += cls._matrices_from_skew(skew)
            else:
                m = cls._matrices_from_skew(skew)
                n = len(m)

        if sph is not None:
            if m is not None:
                m += cls._matrices_from_sph(sph)
            else:
                m = cls._matrices_from_sph(sph)
                n = len(m)

        return cls(matrices=m)

    def __len__(self):
        return len(self.matrices)

    @property
    def matrices(self):
        return self._matrices

    @property
    def symm6(self):
        """Return symmetric part as 6-vector array"""
        m = self.matrices
        v = np.zeros((len(m), 6))
        v[:, 0] = m[:, 0, 0]
        v[:, 1] = m[:, 1, 1]
        v[:, 2] = m[:, 2, 2]
        v[:, 3] = 0.5 * (m[:, 1, 2] + m[:, 2, 1])
        v[:, 4] = 0.5 * (m[:, 0, 2] + m[:, 2, 0])
        v[:, 5] = 0.5 * (m[:, 0, 1] + m[:, 1, 0])

        return v

    @property
    def symmdev(self):
        """Return symmetric and deviatoric part as 5-vector array"""
        m = self.matrices
        v = np.zeros((len(m), 5))
        v[:, 0] = _s6i * (2 * m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])
        v[:, 1] = _s2i * (m[:, 1, 1] - m[:, 2, 2])
        v[:, 2] = _s2i * (m[:, 1, 2] + m[:, 2, 1])
        v[:, 3] = _s2i * (m[:, 0, 2] + m[:, 2, 0])
        v[:, 4] = _s2i * (m[:, 0, 1] + m[:, 1, 0])

        return v

    @property
    def skew(self):
        m = self.matrices
        v = np.zeros((len(m), 3))
        v[:, 0] = _s2i * (m[:, 1, 2] - m[:, 2, 1])
        v[:, 1] = _s2i * (m[:, 2, 0] - m[:, 0, 2])
        v[:, 2] = _s2i * (m[:, 0, 1] - m[:, 1, 0])

        return v

    @property
    def sph(self):
        """Return spherical/isotropic part as scalar array"""
        m = self.matrices
        return _s3i * (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])

    @property
    def trace(self):
        """Return trace of matrices"""
        return _s3 * self.sph

    @staticmethod
    def _matrices_from_symm6(v):
        """matrices from 6-vectors (symmetric part)"""
        m = np.zeros((len(v), 3, 3))
        m[:, 0, 0] = v[:, 0]
        m[:, 1, 1] = v[:, 1]
        m[:, 2, 2] = v[:, 2]
        m[:, 1, 2] = m[:, 2, 1] = v[:, 3]
        m[:, 0, 2] = m[:, 2, 0] = v[:, 4]
        m[:, 0, 1] = m[:, 1, 0] = v[:, 5]

        return m

    @staticmethod
    def _matrices_from_symmdev(v):
        """matrices from 6-vectors (symmetric part)"""
        m = np.zeros((len(v), 3, 3))
        m[:, 0, 0] = v[:, 0] * 2. * _s6i
        m[:, 1, 1] = -v[:, 0] * _s6i + v[:, 1] * _s2i
        m[:, 2, 2] = -v[:, 0] * _s6i - v[:, 1] * _s2i
        m[:, 1, 2] = m[:, 2, 1] = v[:, 2] * _s2i
        m[:, 0, 2] = m[:, 2, 0] = v[:, 3] * _s2i
        m[:, 0, 1] = m[:, 1, 0] = v[:, 4] * _s2i

        return m

    @staticmethod
    def _matrices_from_sph(v):
        """matrices from scalar (spherical part)"""
        diag = (0, 1, 2)
        m = np.zeros((len(v), 3, 3))
        m[:, diag, diag] = v * _s3i

        return m

    @staticmethod
    def _matrices_from_skew(v):
        """matrices from 3-vector (skew part)"""
        m = np.zeros((len(v), 3, 3))
        m[:, 1, 2] = v[:, 0] * _s2i
        m[:, 2, 1] = -m[:, 1, 2]
        m[:, 2, 0] = v[:, 1] * _s2i
        m[:, 0, 2] = -m[:, 2, 0]
        m[:, 0, 1] = v[:, 2] * _s2i
        m[:, 1, 0] = -m[:, 0, 1]

        return m

    @staticmethod
    def _matrices_from_sph(v):
        """matrices from scalar array (sph part)"""
        m = np.zeros((len(v), 3, 3))
        m[:, 2, 2] = m[:, 1, 1] = m[:, 0, 0] = _s3i * v

        return m

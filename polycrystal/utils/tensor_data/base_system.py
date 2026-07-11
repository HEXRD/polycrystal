"""Base class for systems for representing 3D tensors"""

from abc import ABC, abstractmethod

import numpy as np


class BaseSystem:
    """Base system for representation of 3D rank 2 tensors"""

    registry = dict()

    _s2 = np.sqrt(2)
    _s2i = 1 / _s2
    _s3 = np.sqrt(3)
    _s3i = 1 / _s3

    def __init__(self, matrices):
        """Initialize with matrices in standard dyadic basis

        Parameters
        ----------
        matrices: array(n, 3)
           matrices in standard dyadic basis
        """
        if matrices.ndim < 1 or matrices.ndim > 3:
            raise RuntimeError("bad dimension on `matrices: must be 2 or 3`")
        if matrices.ndim == 2:
            matrices = matrices.reshape(1, 3, 3)
        if matrices.shape[1] != 3 or matrices.shape[2] != 3:
            raise ValueError("`matrices` has incorrect shape")
        self._matrices = matrices

    def __init_subclass__(cls, **kwargs):
        """Subclass initialization"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_system_name"):
            cls.registry[cls._system_name] = cls

    def __len__(self):
        return len(self.matrices)

    @property
    def matrices(self):
        return self._matrices

    @property
    def components(self):
        """full components"""
        if not hasattr(self, "_components"):
            self.components = self.to_components(self.matrices)
        return self._components

    @components.setter
    def components(self, c):
        self._components = c

    @abstractmethod
    def to_components(cls, matrices):
        """Set components from matrices"""
        pass

    @abstractmethod
    def to_matrices(cls, components):
        """Set matrices from components"""
        pass

    @classmethod
    def from_parts(cls, symm=None, skew=None):
        """Build matrices from parts additively

        Parameters
        ----------
        symm: array (n, 6)
          symmetric part as 6-vector in orthonormal basis
        skew: array (n, 3)
          skew part
        """
        dim_sym, dim_skw = 6, 3

        len_sym = cls._check_part(symm, dim_sym)
        len_skw = cls._check_part(skew, dim_skw)

        n = max(len_sym, len_skw)
        if n == 0:
            raise ValueError("all parts are None")

        if len_sym == 0:
            symm = np.zeros((n, dim_sym))
        else:
            if len_sym != n:
                raise ValueError("symm and skew parts must have same length")
            symm = symm.reshape((n, dim_sym))

        if len_skw == 0:
            skew = np.zeros((n, dim_skw))
        else:
            if len_skw != n:
                raise ValueError("symm and skew parts must have same length")
            skew = skew.reshape((n, dim_skw))

        comps = np.hstack((symm, skew))
        mats = cls.to_matrices(comps)
        ten = cls(mats)
        ten.components = comps

        return ten

    @staticmethod
    def _check_part(part, dim=None):
        """checks for expected input shape (2D) and length

        Parameters
        ----------
        part: array
           the component part to check
        dim: int or None
           the expected second dimension of the array, or `None` if the array is 1D

        Returns
        -------
        int:
           length of array

        Raises
        ------
        ValueError
           if not correct dimension or length in second dimension
        """
        if part is None:
            return 0

        is_1d = dim is None

        if is_1d:
            if part.ndim > 1:
                raise ValueError("expected 1D array for `part`")
            else:
                part = np.atleast_1d(part)
                return len(part)

        # Now, we have a 2D part.
        comstr = "component string"
        if part.ndim > 2:
            raise ValueError("{comstr} must be 1- or 2-dimensional")
        part = np.atleast_2d(part)
        if part.shape[1] != dim:
            raise ValueError(f"{comstr} must have second dimension equal to {dim}")
        return part.shape[0]

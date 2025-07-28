"""Base class for systems for representing 3D tensors"""

import numpy as np


_s2i = 1/(_s2 := np.sqrt(2.))


class BaseSystem:
    """Base system for representation of 3D rank 2 tensors"""

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

    def __len__(self):
        return len(self.matrices)

    @property
    def matrices(self):
        return self._matrices

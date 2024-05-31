"""Single crystal microstructure"""
import numpy as np

from . import CgoMicrostructure


class SingleCrystal(CgoMicrostructure):
    """Single crystal microstructure

    This is primarily used for simple test cases, but it can also be used
    for isotropic models.

    Parameters
    ----------
    orientation: array (3, 3)
       rotation matrix
    """

    def __init__(self, orientation):

        self.orientation = np.array(orientation).reshape(1, 3, 3)

    @property
    def num_grains(self):
        return 1

    @property
    def num_phases(self):
        return 1

    def grain(self, x):
        return np.zeros(len(x))

    def phase(self, g):
        return np.zeros(len(g))

    @property
    def orientation_list(self):
        return self.orientation

    def grain_orientation(self, g):
        return self.orientation_list[g]

"""Analytic microstructure"""
import numpy as np

from . import CgoMicrostructure


class Analytic(CgoMicrostructure):
    """Microstructure defined analytically by python function

    Parameters
    ----------
    grain_f: function
        a function of positions array x that returns the grain ID for each
        position in x
    orientation_list: array (n, 3, 3)
        the list of orientation matrices by grain ID
    phase_array: (optional) int array (n)
        array giving phase ID for each grain ID, if there are more than
        one phase
    """

    def __init__(
            self, grain_f, orientation_list, phase_array=None
    ):
        self._orientation_list = orientation_list
        self._grain_f = grain_f
        self._phase_a = phase_array

    @property
    def num_grains(self):
        return len(self.orientation_list)

    @property
    def num_phases(self):
        if self._phase_a is None:
            return 1
        else:
            return self._phase_a.max() + 1

    def grain(self, x):
        """determine grains containing given points"""
        return self._grain_f(x)

    def phase(self, g):
        if self._phase_a is None:
            return np.zeros(len(g), dtype=int)
        else:
            return self._phase_a[g]

    @property
    def orientation_list(self):
        return self._orientation_list

    def grain_orientation(self, g):
        return self.orientation_list[g]

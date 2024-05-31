"""Armstrong-Frederick Model with Single Hardness"""
from collections import namedtuple

import numpy as np

from .abc import SlipModel


_flds = ["gamma_dot_0", "m", "H", "H_d"]
AF_SingleHardnessParameters = namedtuple(
    "AF_SingleHardnessParameters", _flds
)
del _flds
AF_SingleHardnessParameters.__doc__ = """Model Parameters

Parameters
----------
gamma_dot_0: float
   reference deformation rate
m: float, (0 < m <= 1)
   rate dependence
H: float (> 0)
   direct hardening coefficient
H_d: float
   dynamic hardening coefficient
"""


class AF_SingleHardness(SlipModel):
    """Armstrong-Frederick model with single hardness

    Parameters
    ----------
    params: instance of AF_SingleHardnessParameters
        material parameters for this model
    """

    def __init__(self, params):
        self.params = params

    def num_statevar(self, num_slipsys):
        return 1

    def gamma_dots(self, state_var, rss):

        gamdot0 = self.params.gamma_dot_0
        gamdot_max = self.gammadot_max
        n = 1/self.params.m
        g = state_var

        taunrm = np.abs(rss/g.reshape((len(g), 1)))
        if gamdot_max is not None:
            t_max = np.power(gamdot_max/gamdot0, self.params.m)
            taunrm = np.minimum(taunrm, t_max)

        return gamdot0 * np.power(taunrm, n) * np.sign(rss)

    def state_derivative(self, state_var, gamdot):

        sv = state_var.flatten()
        sum_absgdot = np.abs(gamdot).sum(1)

        return (self.params.H - self.params.H_d * sv) * sum_absgdot

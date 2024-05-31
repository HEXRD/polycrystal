"""Armstrong-Frederick model with zero backstress"""
from collections import namedtuple

import numpy as np

from .abc import SlipModel

_flds = ["gamma_dot_0", "m", "H", "H_d", "q12"]
AF_ZeroBackStressParameters = namedtuple(
    "AF_ZeroBackStressParameters", _flds
)
del _flds
AF_ZeroBackStressParameters.__doc__ = """Model Parameters

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
q12: float
   latent hardening ratio
"""


class AF_ZeroBackStress(SlipModel):
    """Armstrong-Frederick model with zero backstresses

    PARAMETERS
    ----------
    params: instance of AF_ZeroBackStressParameters
        material parameters for this model
    """

    def __init__(self, params):
        self.params = params

    def num_statevar(self, num_slipsys):
        return num_slipsys

    def gamma_dots(self, state_var, rss):

        gamdot0 = self.params.gamma_dot_0
        gamdot_max = self.gammadot_max
        n = 1/self.params.m
        g = state_var

        taunrm = np.abs(rss/g)
        if gamdot_max is not None:
            t_max = np.power(gamdot_max/gamdot0, self.params.m)
            taunrm = np.minimum(taunrm, t_max)

        return gamdot0 * np.power(taunrm, n) * np.sign(rss)

    def state_derivative(self, state_var, gamdot):

        qii_fac = self.params.q12 - 1
        sgdot = np.abs(gamdot).sum(1).reshape((len(gamdot), 1))

        direct = self.params.H * (
            self.params.q12 * sgdot - qii_fac * np.abs(gamdot)
        )
        dynamic = self.params.H_d * state_var * sgdot

        return direct - dynamic

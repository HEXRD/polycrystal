"""Models for Armstrong-Frederick hardening with backstress"""
from collections import namedtuple

import numpy as np

from .abc import SlipModel

_flds = ["gamma_dot_0", "m", "H", "H_d", "A", "A_d", "q12"]
ArmstrongFrederickParameters = namedtuple(
    "ArmstrongFrederickParameters", _flds
)
del _flds
ArmstrongFrederickParameters.__doc__ = """Model Parameters

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
A: float (> 0)
   direct backstress hardening coefficient
A_d: float
   dynamic backstress hardening coefficient
q12: float
   latent hardening ratio
"""


class ArmstrongFrederick(SlipModel):
    """Armstrong-Frederick Hardening

    For this model, we have a state variable consisting of two fields: the
    hardness filed (g) and back stress (xi); each have values for every slip
    system.

    Parameters
    ----------
    params: instance of ArmstrongFrederickParameters
        material parameters for this model

    References
    ----------
    .. [1] Bandyopadhyay, Ritwik, Veerappan Prithivirajan, and Michael D.
       Sangid. “Uncertainty Quantification in the Mechanical Response of
       Crystal Plasticity Simulations.” JOM 71, no. 8 (August 1, 2019):
       2612–24. https://doi.org/10.1007/s11837-019-03551-3.
    """
    def __init__(self, params):
        """Constructor for ArmstrongFrederick

        """
        self.params = params

        return

    def num_statevar(self, num_slipsys):
        return num_slipsys * 2

    def gamma_dots(self, state_var, rss):
        """Compute slip system shear strain rates

        The schmid tensors and the stress need to be in the same reference
        frame, but it could be crystal or sample.

        PARAMETERS
        ----------
        state_var: array (npts, 2 * nslip)
            current values of material state, slip system hardness and
            backstresses (all hardnesses before backstresses)
        rss: array (npts, nslip)
            resolved shear stress

        RETURNS
        -------
        array (npts, nslip)
            the slip system shear rates, gamma dots
        """
        gamdot0 = self.params.gamma_dot_0
        gamdot_max = self.gammadot_max
        n = 1/self.params.m

        shp = state_var.shape
        state_var = state_var.reshape(shp[0], 2, shp[1]//2)
        G, CHI = 0, 1
        g = state_var[:, G, :]
        chi = state_var[:, CHI, :]

        dsig = rss - chi
        dsignrm = np.abs(dsig/g)

        if gamdot_max is not None:
            d_max = np.power(gamdot_max/gamdot0, self.params.m)
            dsignrm = np.minimum(dsignrm, d_max)

        return gamdot0 * np.power(dsignrm, n) * np.sign(dsig)

    def state_derivative(self, state_var, gamdot):
        """Derivative of state variable

        Parameters
        ----------
        state_var: array (npts, 2 * nslip)
            current values of material state; first array is slip system
            hardness, and second is back stress
        gamdot: array (npts, nslip)

        Returns
        -------
        array (npts, 2 * nslip)
            the slip system shear rates, gamma dots
        """
        A, A_d = self.params.A, self.params.A_d

        shp = state_var.shape
        state_var = state_var.reshape(shp[0], 2, shp[1]//2)
        G, CHI = 0, 1
        g = state_var[:, G, :]
        chi = state_var[:, CHI, :]

        qii_fac = self.params.q12 - 1
        sgdot = np.abs(gamdot).sum(1).reshape((len(gamdot), 1))

        # Compute the hardness derivative.
        direct = self.params.H * (
            self.params.q12 * sgdot - qii_fac * np.abs(gamdot)
        )
        dynamic = self.params.H_d * g * sgdot
        gdot = direct - dynamic

        # Compute the backstress derivative.
        direct = self.params.A * gamdot
        dynamic = self.params.A_d * chi * np.abs(gamdot)
        chidot = direct - dynamic

        return np.hstack((gdot, chidot)).reshape(shp)

    @staticmethod
    def _hardening_matrix(self, nslip):
        """Hardening matrix q_alpha_beta"""
        # Maybe this should be moved to a common place in the future when we
        # have more slip models, so it can be available to all models.
        q_ab = 1.2 * np.ones((nslip, nslip))
        return np.fill_diagonal(q_ab, 1.0)

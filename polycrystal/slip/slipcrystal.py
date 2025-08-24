"""Slip based crystal"""
from collections import namedtuple

import numpy as np

from ..utils.tensor_data.symmdev_system import SymmDevSystem


_flds = [
    "resolved_shear_stress", "gamma_dots", "velocity_gradient",
    "state_derivative"
]
_SlipData = namedtuple(
    "_SlipData", _flds, defaults=len(_flds) * [None]
)


class SlipCrystal:
    """Crystal with slip

    Parameters
    ----------
    groups: list of SlipGroup instances
      list of slip system groups related by crystal symmetry
    model: SlipModel instance
      model used for deformation and hardness evolution
    """

    def __init__(self, groups, model):
        self.groups = groups
        self.model = model

        self._schmid_td = SymmDevSystem(
            np.hstack([g.schmid for g in self.groups])
        )

    @property
    def schmid(self):
        """Return Schmid Tensor matrices"""
        return self._schmid_td.matrices

    @property
    def schmid_5(self):
        """symmetric deviatoric part of Schmid tensors"""
        return self._schmid_td.symmdev

    @property
    def num_slipsys(self):
        """Number of slip systems"""
        return len(self.schmid)

    @property
    def num_statevar(self):
        """Number of state variables"""
        return self.model.num_statevar(self.num_slipsys)

    def resolved_shear_stress(self, cstress):
        """resolved shear stress on slip systems

        Parameters
        ----------
        cstress: array (npts, 3, 3)
          crystal stress in crystal reference frame

        Returns
        -------
        array (npts, nslip)
           array of resolved shear stresses for each slip system
        """
        return SymmDevSystem(cstress).symmdev @ self.schmid_5.T

    def velocity_gradient(self, gdots):
        """Plastic velocity gradient from gamma dots

        Parameters
        ----------
        gdots: array (m, nss)
           shearing rates on all `nss` slip systems at `m` points

        Returns
        -------
        array (m, 3, 3)
           plastic velocity gradient (Lp) at each point
        """
        return np.einsum("ij,jkl->ikl", gdots, self.schmid)

    def get(
        self, cstress, state,
            resolved_shear_stress=False,
            gamma_dots=False,
            velocity_gradient=False,
            state_derivative=False

    ):
        """Compute slip data

        Parameters
        ----------
        cstress: array (npts, 3, 3)
            crystal stress in crystal reference frame
        state: array (npts, nsv)
            microstructural state variables, e.g hardess, etc.

        resolved_shear_stress: (optional, default=False) bool
            return resolved shear stress
        gamma_dots: (optional, default=False) bool
            return slip system shear rates
        velocity_gradient: (optional, default=False) bool
            return plastic velocity_gradient
        state_derivative: (optional, default=False) bool
            return state variable derivatives

        Returns
        -------
        slipdata:
            namedtuple with requested data
        """
        out_data = _SlipData()

        rss = self.resolved_shear_stress(cstress)
        if resolved_shear_stress:
            out_data = out_data._replace(resolved_shear_stress=rss)

        # Compute gamma dots if any other output is requested.
        if gamma_dots or velocity_gradient or state_derivative:
            gdots = self.model.gamma_dots(state, rss)

        if gamma_dots:
            out_data = out_data._replace(gamma_dots=gdots)

        if velocity_gradient:
            vg = self.velocity_gradient(gdots)
            out_data = out_data._replace(velocity_gradient=vg)

        if state_derivative:
            sd = self.model.state_derivative(state, gdots)
            out_data = out_data._replace(state_derivative=sd)

        return out_data

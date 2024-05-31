"""Abstract base class for slip_model"""
from abc import ABC, abstractmethod


class SlipModel(ABC):

    @abstractmethod
    def num_statevar(self, nums_slipsys):
        """Number of state variables

        PARAMETERS
        ----------
        num_slipsys: int
           number of slip systems

        RETURNS
        ----------
        int
           number of state variables
        """
        pass

    @property
    def gammadot_max(self):
        """maximum allowable shear strain rate"""
        if not hasattr(self, "_gammadot_max"):
            self._gammadot_max = None
        return self._gammadot_max

    @gammadot_max.setter
    def gammadot_max(self, v):
        """Set method for gammadot_max"""
        self._gammadot_max = v

    @abstractmethod
    def gamma_dots(self, state_var, rss):
        """Compute slip system shear strain rates

        The schmid tensors and the stress need to be in the same reference
        frame, but it could be crystal or sample.

        PARAMETERS
        ----------
        state_var: array (npts)
            current values of material state, slip system hardness
        rss: array (npts, nslip)
            resolved shear stress

        RETURNS
        -------
        array (npts, nslip)
            the slip system shear rates, gamma dots
        """
        pass

    @abstractmethod
    def state_derivative(self, state_var, gamdot):
        """Derivative of state variable

        Parameters
        ----------
        state_var: array (npts)
          current values of material state (slip system strength)
        gamdot: array (npts, nslip)
          slip system shear rates (gamma dots)

        Returns
        -------
        array (npts, nslip)
            derivative of state variables at each point
        """
        pass

"""ODE solvers and data structures for plastic slip evolution.

Provides solver strategies (Pointwise, Euler, Block) operating on
StateData (grain orientations, state variables, and plastic deformation
gradients) driven by StressIncrement or StressSequence (time-dependent crystal
stress loading paths).
"""
from collections import namedtuple
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from ..orientations.quaternions import random_rmats


_flds = ["csig", "times"]
_StressSequence = namedtuple("_StressSequence", _flds)


class StressSequence(_StressSequence):
    """Ordered sequence of crystal stress states and their associated times.

    Fields
    ------
    csig: array (n_steps, n_pts, 3, 3)
        Crystal-frame stress tensor for each grain at each time in ``times``.
    times: array (n_steps,)
        Monotonically increasing time values (seconds).
    """

    @classmethod
    def from_file(cls, filename):
        """Load a StressSequence from a ``.npz`` file.

        Parameters
        ----------
        filename: str or Path
            Path to the ``.npz`` file.
        """
        data = np.load(filename)
        return cls(csig=data["csig"], times=data["times"])

    def save(self, filename):
        """Save the stress sequence to a ``.npz`` file.

        Parameters
        ----------
        filename: str or Path
            Path to the output ``.npz`` file.
        """
        np.savez(filename, **self._asdict())

    @staticmethod
    def cstress_from_sample(s_stress, orient):
        """Rotate a sample-frame stress tensor into each grain's crystal frame.

        Parameters
        ----------
        s_stress: array-like (3, 3)
            Symmetric stress tensor in the sample (lab) frame.
        orient: array (n, 3, 3)
            Rotation matrices for each grain (sample -> crystal).

        Returns
        -------
        array (n, 3, 3)
            Per-grain crystal-frame stress tensors.
        """
        s = np.array(s_stress).reshape((1, 3, 3))
        oriT = orient.transpose((0, 2, 1))
        return orient @ s @ oriT


class StateData:
    """Grain ensemble state: crystal orientations, state variables, and
    plastic deformation gradients.

    Stores and persists the per-grain data needed by the ODE solvers.
    State variables follow the Armstrong-Frederick convention: the first
    ``num_sv`` entries per grain are hardness values; if the model includes
    back-stress, the next 12 entries are back-stress values.
    """

    def __init__(self, material, orient, state, def_grad=None):
        """Initialize state data for a grain ensemble.

        Parameters
        ----------
        material:
            Material model object (e.g. SlipCrystal); must expose
            ``num_statevar``.
        orient: array (n, 3, 3)
            Rotation matrices mapping crystal components to sample components
            for each of the *n* grains.
        state: array (n, num_sv) or (n,)
            Initial state variable values (hardness + back-stress).
        def_grad: array (n, 3, 3), optional
            Initial plastic deformation gradients. Defaults to identity for
            all grains.
        """
        self.material = material
        self.orient = np.array(orient, copy=True)
        self.state = np.array(state, copy=True)
        if def_grad is None:
            self.def_grad = self.constant_defgrad(len(orient))
        else:
            self.def_grad = np.array(def_grad, copy=True)

    @staticmethod
    def constant_state_af(num_pts, value, num_sv):
        """Return a constant initial state array for Armstrong-Frederick models.

        Parameters
        ----------
        num_pts: int
            number of grains (points).
        value: float | 2-tuple
            if a float, then it is applied to all state variables; if a tuple
            of two values, then the values are split between the state variables
            with strength in the first half and backstress in the second, and
            `num_sv` must be even.
        num_sv: int
            Number of state variables per grain. Even number for models with
            a backstress.

        Returns
        -------
        array
            Shape ``(num_pts,)`` when *num_sv* is 1, else
            ``(num_pts, num_sv)``.
        """
        if isinstance(value, tuple):
            if np.mod(num_sv, 2) != 0:
                raise RuntimeError("num_sv must be even")
            n_2 = num_sv // 2
            strength, backstress = value
            g_init = n_2 * (strength,) + n_2 * (backstress,)

        elif num_sv == 1:
            g_init = (value,)

        else:
            g_init = num_sv * (value,)

        # Now tile the initial values over all the grains.
        rv = np.tile(g_init, (num_pts, 1))
        if num_sv == 1:
            rv = rv.flatten()

        return rv

    @staticmethod
    def constant_defgrad(num_pts, value=None):
        """Return an array of identical plastic deformation gradients.

        Parameters
        ----------
        num_pts: int
            Number of grains.
        value: array-like (3, 3), optional
            Deformation gradient to replicate. Defaults to the 3x3 identity.

        Returns
        -------
        array (num_pts, 3, 3)
        """
        if value is None:
            F_init = np.identity(3)
        else:
            F_init = np.array(value)

        return np.tile(F_init, (num_pts, 1, 1))

    @staticmethod
    def random_orientations(num_pts):
        """Return uniformly distributed random rotation matrices.

        Parameters
        ----------
        num_pts: int
            Number of grains.

        Returns
        -------
        array (num_pts, 3, 3)
        """
        return random_rmats(num_pts)

    @classmethod
    def from_file(cls, material, filename):
        """Load state data from a ``.npz`` file saved by :meth:`save_data`.

        Parameters
        ----------
        material:
            Material model object (must expose ``num_statevar``).
        filename: str or Path
            Path to the ``.npz`` archive containing ``orient``, ``state``,
            and ``def_grad`` arrays.
        """
        npz = np.load(filename)
        return cls(material, npz["orient"], npz["state"], npz["def_grad"])

    def save_data(self, filename):
        """Save orientation, state, and deformation gradient arrays to a ``.npz`` file.

        Parameters
        ----------
        filename: str or Path
        """
        np.savez(
            filename,
            orient=self.orient,
            state=self.state,
            def_grad=self.def_grad,
        )

    @property
    def num_sv(self):
        """Number of state variables per grain, as reported by the material model."""
        return self.material.num_statevar

    @property
    def num_pts(self):
        """Number of grains in the ensemble."""
        return len(self.orient)


class StressIncrement:
    """Time-dependent crystal stress loading path between two stress states.

    Represents a linear ramp from an initial crystal stress ``csig0`` at
    ``t=0`` to a final crystal stress ``csigT`` at ``t=T``. Crystal stresses
    are expressed in each grain's crystal frame, so they have shape ``(n, 3, 3)``.
    """

    def __init__(self, csig0, csigT, T):
        """Initialize with crystal stress arrays and duration.

        Parameters
        ----------
        csig0: array (n, 3, 3) or (3, 3)
            Initial crystal stress state at t=0.
        csigT: array (n, 3, 3) or (3, 3)
            Final crystal stress state at t=T.
        T: float
            Duration of the stress step (seconds).
        """
        self.csig0 = np.array(csig0)
        self.csigT = np.array(csigT)
        self.T = float(T)

    @classmethod
    def from_file(cls, filename):
        """Load stress data from a ``.npz`` file saved by :meth:`save`.

        Parameters
        ----------
        filename: str or Path
        """
        npz = np.load(filename)
        return cls(npz["csig0"], npz["csigT"], npz["T"])

    def save(self, filename):
        """Save stress data to a ``.npz`` file.

        Parameters
        ----------
        filename: str or Path
        """
        np.savez(filename, csig0=self.csig0, csigT=self.csigT, T=self.T)

    def cstress_t(self, t, ipt=None):
        """Return the linearly interpolated crystal stress at time *t*.

        Parameters
        ----------
        t: float or array
            Time value(s) in ``[0, T]``.
        ipt: int, optional
            If given, return stress only for grain index *ipt*; otherwise
            return stresses for all grains.

        Returns
        -------
        array (3, 3) or (n, 3, 3)
            Crystal stress tensor(s) at time *t*.
        """
        trel = t / self.T
        trel_c = 1.0 - trel
        if ipt is None:
            return trel_c * self.csig0 + trel * self.csigT
        else:
            if (lt := len(np.atleast_1d(t))) > 1:
                trel = trel.reshape(lt, 1, 1)
                trel_c = trel_c.reshape(lt, 1, 1)
            ipt1 = ipt + 1
            return trel_c * self.csig0[ipt:ipt1] + trel * self.csigT[ipt:ipt1]


class Solver:
    """Abstract base class for slip ODE solvers.

    Subclasses implement :meth:`run` using different integration strategies
    (pointwise BDF, explicit Euler, or block BDF) over the time interval
    ``[0, sdata.T]``.
    """

    def __init__(self, mdata, sdata):
        """Parameters
        ----------
        mdata: StateData
            Grain ensemble state (orientations, state variables, def gradients).
        sdata: StressIncrement
            Time-dependent crystal stress loading path.
        """
        self.mdata = mdata
        self.sdata = sdata

    @property
    def time_interval(self):
        """Integration interval ``[0, T]`` as a two-element list."""
        return [0.0, self.sdata.T]

    @property
    def num_sv(self):
        """Number of state variables per grain."""
        return self.mdata.material.num_statevar

    @property
    def num_pts(self):
        """Number of grains in the ensemble."""
        return self.mdata.num_pts

    def run(self):
        """Integrate the slip ODEs. Implemented by each subclass."""
        raise NotImplementedError("run not implemented: use subclass")

    def time_run(self, *args, **kwargs):
        """Call :meth:`run`, print the wall-clock time, and return ``(elapsed, result)``.

        All positional and keyword arguments are forwarded to :meth:`run`.

        Returns
        -------
        tuple
            ``(elapsed_seconds, run_result)``
        """
        t0 = time.perf_counter()
        y = self.run(*args, **kwargs)
        et = time.perf_counter() - t0
        print(f"elapsed time: {et:0.4f} seconds")
        return et, y

    def integrate_fp(self, t_hist):
        """Integrate F^p for given time/state history.

        Parameters
        ----------
        t_hist: list of array
            List of arrays of shape ``(1 + num_sv, n_points)``, one for each grain.
            Row 0 is the time points; remaining rows are state variable values.

        Returns
        -------
        array (num_pts, 3, 3)
            Updated plastic deformation gradient tensors at time T.
        """
        matl = self.mdata.material
        npts = len(t_hist)
        Fp0 = self.mdata.def_grad
        Fp = np.zeros_like(Fp0)
        for ipt in range(npts):
            p_hist = t_hist[ipt]
            t, state = p_hist[0], p_hist[1:].T
            cstress = self.sdata.cstress_t(t, ipt)
            out = matl.get(
                cstress,
                state,
                gamma_dots=True,
                velocity_gradient=True,
            )
            # Integrate 0 -> T
            dtime = (t[1:] - t[:-1]).reshape((len(t) - 1, 1, 1))
            vg = out.velocity_gradient
            vgmid = 0.5 * (vg[:-1] + vg[1:])
            F = Fp0[ipt].copy() if Fp0 is not None else np.identity(3)
            for dt, L in zip(dtime, vgmid):
                F = F + dt * (L @ F)
            Fp[ipt] = F

        return Fp


class Pointwise(Solver):
    """Run ODE solver on each point independently.

    Returns
    -------
    array (npts, num_sv)
        If ``time_history=False``, array of state values at time T.
    list of array
        If ``time_history=True``, list of ``(1 + num_sv, n_steps)`` arrays for each grain.
    """

    def run(self, time_history=False):
        """Integrate each grain independently using ``scipy.solve_ivp`` (BDF).

        Parameters
        ----------
        time_history: bool
            If ``True``, return a list of ``(1 + num_sv, n_steps)`` arrays —
            one per grain — where row 0 is the time vector and rows 1… are
            the state-variable trajectories. If ``False``, return only the
            final state array.

        Returns
        -------
        array (num_pts, num_sv)
            Final state values (``time_history=False``).
        list of array
            Per-grain ``(1 + num_sv, n_steps)`` arrays (``time_history=True``).
        """
        T = self.sdata.T
        if time_history:
            t_hist = []
            t_eval = None
        else:
            t_eval = [T]
        sol_0 = self.mdata.state
        sol_y = np.zeros((self.num_pts, self.num_sv))
        for ipt in range(self.num_pts):
            if ipt % 1000 == 0 and self.num_pts >= 1000:
                print(f"finished {ipt} points")
            s0 = np.atleast_1d(sol_0[ipt])
            sol = solve_ivp(
                lambda t, s: self.dsdt_p(ipt, t, s),
                self.time_interval,
                s0,
                t_eval=t_eval,
                method="BDF",
            )
            if time_history:
                t_hist.append(np.vstack((sol.t, sol.y)))
            else:
                sol_y[ipt] = sol.y.flatten()
        if self.num_pts >= 1000:
            print(f"finished all ({self.num_pts}) points")

        if time_history:
            return t_hist
        else:
            return sol_y

    def dsdt_p(self, ipt, t, s):
        """Time derivative of state variables for grain *ipt* at time *t*.

        Parameters
        ----------
        ipt: int
            Grain index.
        t: float
            Current time.
        s: array (num_sv,)
            Current state variable values for this grain.

        Returns
        -------
        array (num_sv,)
        """
        sa = np.atleast_2d(s)
        return self.mdata.material.get(
            self.sdata.cstress_t(t, ipt), sa, state_derivative=True
        ).state_derivative.flatten()


class Euler(Solver):
    """Solves for all grains simultaneously using explicit Euler time-stepping."""

    def run(self, nsteps=100, time_history=False):
        """Advance state variables using fixed-step Euler integration.

        Parameters
        ----------
        nsteps: int
            Number of equal time steps over ``[0, T]``.
        time_history: bool
            If ``True``, return ``(t, y)`` arrays for all steps;
            otherwise return only the final state array.

        Returns
        -------
        array (num_pts, num_sv) or tuple (t_array, y_array)
        """
        T = self.sdata.T
        dt = T / nsteps
        sol_y = self.mdata.state.copy()
        if time_history:
            y = np.zeros((self.num_pts, nsteps + 1, self.num_sv))
            istep = 0
            y[:, istep, :] = sol_y
        for step in range(nsteps):
            t = step * dt
            sdot = self.dsdt(t, sol_y)
            sol_y += dt * sdot
            if time_history:
                istep += 1
                y[:, istep, :] = sol_y

        if time_history:
            t = np.linspace(0, T, nsteps + 1)
            return t, y
        else:
            return sol_y

    def dsdt(self, t, s):
        """Derivative of state variables for all grains."""
        cstress = self.sdata.cstress_t(t)
        return self.mdata.material.get(
            cstress, s, state_derivative=True
        ).state_derivative


class Block(Solver):
    """Integrates the full grain ensemble as one large ODE system using BDF."""

    def run(self):
        """Integrate all grains simultaneously with ``scipy.solve_ivp`` (BDF).

        Returns
        -------
        array (num_pts, num_sv)
            Final state values for all grains.
        """
        T = self.sdata.T
        sol_0 = np.array(self.mdata.state).flatten()
        sol = solve_ivp(
            self.dsdt,
            self.time_interval,
            sol_0,
            t_eval=[T],
            method="BDF",
        )
        return sol.y.reshape((self.num_pts, self.num_sv))

    def dsdt(self, t, s):
        """Time derivative of the full flattened state vector."""
        cstress = self.sdata.cstress_t(t)
        sa = s.reshape((self.num_pts, self.num_sv))
        return self.mdata.material.get(
            cstress, sa, state_derivative=True
        ).state_derivative.flatten()

"""Scripts and workflow tools for polycrystal simulations."""
from ..slip.solve import StateData, StressSequence


def evolve_slip(*args, **kwargs):
    """Integrate a multi-step stress sequence and save step-NN.npz output files."""
    from .evolve_slip import evolve_slip as _evolve_slip
    return _evolve_slip(*args, **kwargs)


__all__ = [
    "StateData",
    "StressSequence",
    "evolve_slip",
]

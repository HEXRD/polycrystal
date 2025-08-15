"""Moduli handlers for various symmetries

Notes
-----
See Brannon [1] for discussion of the Voigt and Mandel systems for representing
symmetric matrices. This includes representation of the elastic stiffness and
compliance matrices.

References
----------
(1) Brannon, R. M. Rotation, Reflection, and Frame Changes: Orthogonal Tensors in Computational Engineering Mechanics; IOP Publishing, 2018.
"""
from .isotropic import Isotropic
from .cubic import Cubic
from .hexagonal import Hexagonal
from .triclinic import Triclinic


def moduli_handler(symmetry):
    """Return the moduli handler for the given symmetry"""
    # All classes share the same registry, so we can use any handler.

    reg = isotropic.Isotropic.subclass_registry
    if symmetry in reg:
        return reg[symmetry]
    else:
        raise ValueError(f"symmetry '{symmetry}' is not recognized")


def component_system(system_spec):
    """Returns Enum attribute corresponding to system name

    Parameters
    ----------
    system_spec: str or Enum
       name of system or attribute of `MatrixComponentSystem`
    """
    cls = isotropic.Isotropic
    if system_spec in cls.SYSTEMS:
        return system_spec
    d = {s.name: s for s in cls.SYSTEMS}
    if system_spec not in d:
        emsg = (
            f"component system name `{system_spec}` not found;"
            f"it must be one of: {list(d.keys())}"
        )
        raise ValueError(emsg)

    return d[system_spec]

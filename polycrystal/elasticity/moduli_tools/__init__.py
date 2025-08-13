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
from . import isotropic, cubic, hexagonal, triclinic


def moduli_handler(symmetry):
    """Return the moduli handler for the given symmetry"""
    # All classes share the same registry, so we can use any handler.

    reg = isotropic.Isotropic.subclass_registry
    if symmetry in reg:
        return reg[symmetry]
    else:
        raise ValueError(f"symmetry '{symmetry}' is not recognized")

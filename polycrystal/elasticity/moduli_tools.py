"""Parameter conversions for moduli

Notes
-----
See Brannon [1] for discussion of the Voigt and Mandel systems for representing
symmetric matrices. This includes representation of the elastic stiffness and
compliance matrices.

References
----------
(1) Brannon, R. M. Rotation, Reflection, and Frame Changes: Orthogonal Tensors in Computational Engineering Mechanics; IOP Publishing, 2018.
"""

from enum import Enum, auto

import numpy as np



class SymmetryNames(Enum):
    """Names for crystal symmetry groups"""
    TRICLINIC = "triclinic"
    ISOTROPIC = "isotropic"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


symmetry_names_dict = {sn.value: sn for sn in SymmetryNames}

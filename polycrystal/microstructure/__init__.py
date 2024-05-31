"""Microstructure subpackage

Here, we think of `microstructure` as the spatial assignment of phases and
grains and their associated data, primarily orientation.
"""
from . abc import Microstructure
from . abc import ConstantGrainOrientationMicrostructure

# This is a useful abbreviation.
CgoMicrostructure = ConstantGrainOrientationMicrostructure

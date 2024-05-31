"""Base class for conventions"""
from .conventionabc import ConventionABC
from .registry import ConventionRegistry


# Base Class
class Convention(ConventionABC, metaclass=ConventionRegistry):

    # Angle units
    IN_DEGREES = 0
    IN_RADIANS = 1

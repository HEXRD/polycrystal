"""Base class for moduli handlers"""

from abc import ABC, abstractmethod

from .stiffness_matrix import MatrixComponentSystem, StiffnessMatrix

import numpy as np


class BaseModuli(ABC):

    """Moduli Handlers"""
    SYSTEMS = MatrixComponentSystem
    DEFAULT_SYSTEM = MatrixComponentSystem.MANDEL

    subclass_registry = {}


    def __init_subclass__(cls, **kwargs):
        # This adds each subclass to the registry with a key based on the
        # lower cased subclass name.
        super().__init_subclass__(**kwargs)
        cls.subclass_registry[cls.__name__.lower()] = cls

    @abstractmethod
    def stiffness_from_moduli(self):
        """Independent moduli to matrix"""

    @abstractmethod
    def moduli_from_stiffness(self):
        """Independent moduli from stiffness matrix"""

    @property
    @abstractmethod
    def cij(self):
        """Array of independent moduli"""

    def init_system(self, system):
        """Initialize system

        This also builds intializes the stiffness matrix from the moduli. After
        initialization, changing `system` resets the moduli from the stiffness
        matrix.
        """
        self._system = system
        self._stiffness = self.stiffness_from_moduli()

    @property
    def system(self):
        """Return the matrix component system"""
        return self._system

    @system.setter
    def system(self,  v):
        self._check_system(v)
        self._system = v
        self.stiffness.system = v
        self.moduli_from_stiffness()

    @property
    def stiffness(self):
        return self._stiffness

    def _check_system(self, v):
        if v not in self.SYSTEMS:
            raise ValueError("`system` not in MatrixComponentSystem")

    @staticmethod
    def _high_symmetry_matrix(c11, c12, c13, c22, c23, c33, c44, c55, c66):
        """Simplified form of stiffness matrix for certain symmetries

        This returns the upper triangle of a stiffness matrix of the form:

            [[c11, c12, c13,    0,   0,   0],
             [c12, c22, c23,    0,   0,   0],
             [c13, c23, c33,    0,   0,   0],
             [0,     0,   0,  c44,   0,   0],
             [0,     0,   0,    0, c55,   0],
             [0,     0,   0,    0,   0, c66]])
        """
        cij_upper = (
            c11, c12, c13,    0,   0,   0,
            c22, c23,    0,   0,   0,
            c33,    0,   0,   0,
            c44,   0,   0,
            c55,   0,
            c66
        )
        return np.array(cij_upper)

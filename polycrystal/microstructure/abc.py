"""Abstract base class for Microstructure class

In the general case, orientation is a function of position and the orientation
may vary across a grain. In many cases, we treat each grain as having a
constant orientation. In that case, use the
ConstantGrainOrientationMicrostructure subclass with the grain_orientation()
method.

Notes
-----
For this package, orientations are considered as change of basis matrices that
take crystal coordinates to sample coordinates (:math:`Rc = s`). Assuming
right handed orthonormal bases for both the crystal and sample frames, the
orientations will be rotation matrices.
"""
from abc import ABC, abstractmethod
from collections.abc import Container


class Microstructure(ABC):

    @property
    @abstractmethod
    def num_grains(self):
        """Number of grains"""
        pass

    @property
    @abstractmethod
    def num_phases(self):
        """Number of phases"""
        pass

    @abstractmethod
    def grain(self, x):
        """grain ID by position

        Parameters
        ----------
        x: array (n, d)
           array of `n` points of dimension `d`

        Returns
        -------
        int array (n)
           array of grain IDs for each point in x
        """
        pass

    @abstractmethod
    def phase(self, g):
        """phase ID of grains

        Parameters
        ----------
        g: int array (n)
           array of grain IDs

        Returns
        -------
        int array (n)
           array of phase IDs for corresponding grains
        """
        pass

    @abstractmethod
    def orientation(self, x):
        """orientation of positions

        Parameters
        ----------
        x: array (n, d)
           array of `n` points of dimension `d`

        Returns
        -------
        array (n, 3, 3)
           orientation matrices for each positon
        """
        pass


class ConstantGrainOrientationMicrostructure(Microstructure):
    """Microstructure with grains that have a constant orientation"""

    @property
    @abstractmethod
    def orientation_list(self):
        """array of orientations by grain ID"""
        pass

    @abstractmethod
    def grain_orientation(self, g):
        """orientation by grain

        Parameters
        ----------
        g: array (n)
           array of grain IDs

        Returns
        -------
        array (n, 3, 3)
           array of rotation matrices for each grain
        """
        # Implementation
        return self.orientation_list[g]

    def orientation(self, x):
        # Implementation
        return self.grain_orientation(self.grain(x))

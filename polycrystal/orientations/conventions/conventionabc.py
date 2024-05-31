"""Abstract Base Class for orientation conventions"""
import abc


class ConventionABC(abc.ABC):

    @abc.abstractmethod
    def to_rmats(self, a):
        """convert from convention to rotation matrices

        Parameters
        ----------
        a: array (n, m)
           array of `n` parameter arrays

        Returns
        -------
        array (n, 3, 3)
           array of `n` 3x3 matrices
        """
        pass

    @abc.abstractmethod
    def from_rmats(self, r):
        """convert to convention from rotation matrices

        Parameters
        ----------
        r: array (n, 3, 3)
           array of `n` parameters

        Returns
        -------
        array (n, m)
           array of `n` parameter arrays
        """
        pass

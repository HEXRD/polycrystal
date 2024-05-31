"""Voronoi microstructure"""
import numpy as np
from numpy.random import rand

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from . import CgoMicrostructure
from ..orientations.quaternions import random_quats, random_rmats


class Voronoi(CgoMicrostructure):
    """Voronoi microstructure

    A Voronoi microstructure determines the grain ID at a point x
    according to the nearest seed point, as in a Voronoi tessellation.
    If the `matrix` argument is given, it alters the metric to elongate
    the grains in certain directions.  For example, if you gave it a
    diagonal matrix of (1, 3, 5), the grains would roughly be 3 times
    the length in y and 5 times in z.

    Parameters
    ----------
    seeds: array (n, d)
       `n` seed points in dimension `d`
    orientations: array(n, 3, 3)
       rotation matrices giving crystal orientations
    box: (optional) array (d, 2)
       array giving ranges of each coordinate
    matrix: (optional) array (3, 3)
       grain shape matrix
    """

    def __init__(self, seeds, orientations, box=None, matrix=None):

        self.seeds = seeds
        n, d = seeds.shape
        if orientations is None:
            self.orientations = np.tile(np.identity(d), (n, 1, 1))
        else:
            if len(orientations) != len(seeds):
                raise ValueError(
                    "number of orientations does not match number of seeds"
                )
            self.orientations = orientations


        if box is None:
            self.box = box
        else:
            self.box = np.array(box)

        if matrix is None:
            self.matrix = None
        else:
            self.matrix = np.array(matrix)
            self.matrix_inv = np.linalg.pinv(self.matrix)
        if matrix is None:
            self.kdtree = KDTree(self.seeds)
        else:
            self.kdtree = KDTree(self.seeds @ self.matrix_inv.T)

    @classmethod
    def from_file(cls, filename):
        """Voronoi from npz file with input arrays

        Parameters
        ----------
        filename: str
           name of npz file with seeds, orientations and possibly box and matrix
        """
        npz = np.load(filename)
        seeds = npz['seeds']
        orientations = npz['orientations']
        box = npz["box"] if "box" in npz else None
        mat = npz["matrix"] if "matrix" in npz else None

        return cls(seeds, orientations, box=box, matrix=mat)

    def contains(self, x):
        """Determine which points lie in the box

        Parameters
        ----------
        x: array (n, d)
           array of points in dimension `d`

        Returns
        -------
        bool array (n)
           array with True values for points in the box
        """
        if self.box is None:
            return True
        else:
            return self._inbox(x)

    def _inbox(self, x):
        """check whether x is in the box"""
        for i in range(self.dim):
            indim_i = (x[:, i] >= self.box[i, 0]) & (x[:, i] < self.box[i, 1])
            if i == 0:
                indim = indim_i
            else:
                indim = indim & indim_i
        return indim

    @property
    def dim(self):
        """Dimension of points"""
        return self.seeds.shape[1]

    @property
    def num_grains(self):
        return len(self.seeds)

    @property
    def num_phases(self):
        return 1

    def grain(self, x):
        if self.matrix is None:
            xq = x
        else:
            xq = x @ self.matrix_inv.T
        inds = self.kdtree.query(xq)[1]
        return inds

    def phase(self, g):
        return 0

    @property
    def orientation_list(self):
        return self.orientations

    def grain_orientation(self, g):
        return self.orientations[g]

    def save(self, fname):
        """Save data to an npz file

        Parameters
        ----------
        fname: str
            name of file to save arrays to
        """
        d = {
            'seeds': self.seeds,
            'orientations': self.orientations
        }
        if self.box is not None:
            d.update(box=self.box)

        if self.matrix is not None:
            d.update(matrix=self.matrix)

        np.savez(fname, **d)

    @classmethod
    def random_seeds(cls, n, box):
        """return a list of random seeds

        Parameters
        ----------
        n: int
           the number of seeds
        box: array (3, 2)
           the box of the region containing the seeds

        Returns
        -------
        array (n, 3) of random seed points

        """
        useeds = np.random.rand(n,3)
        b0 = box[:,0].reshape((1,3))
        db = (box[:,1] - box[:,0]).reshape((1,3))

        return  b0 + db*useeds

    @classmethod
    def random_voronoi(cls, n, box, fname=None, seedbox=None, matrix=None):
        """Generate random seeds and orientations

        Parameters
        ----------
        n: int
           number of grains
        box: array (3, 2)
           array containing minimum and maximum values of each coordinate
        fname: str
           name of file in which to save data
        matrix: array (3, 3)
           grain shape matrix
        seedbox: (optional) array  (3, 2)
           box containing seeds

        Returns
        -------
        Voronoi instance
        """
        # Maybe extend this to allow 2D random voronois as well.  For example
        # have optional dim = 3. Then have to generate random 2D matrices, but
        # that shouldn't be hard.
        #
        seeds = np.random.rand(n, 3)
        if seedbox is None:
            seedbox = box
        for i, b in enumerate(seedbox):
            scale = b[1] - b[0]
            seeds[:, i] = scale*seeds[:, i] + b[0]

        rmats = random_rmats(n)

        v = cls(seeds=seeds, orientations=rmats, box=box, matrix=matrix)
        if fname is not None:
            v.save(fname)

        return v

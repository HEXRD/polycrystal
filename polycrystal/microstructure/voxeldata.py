"""Voxel data based on regular grid"""
from polycrystal.microstructure import\
    ConstantGrainOrientationMicrostructure as CgoMicrostructure

import numpy as np


class VoxelData(CgoMicrostructure):
    """Voxel data based on regular grid


    Parameters
    ----------
    grain_ids: int array (l, m, n)
       array of grain IDs
    orientation_list: array (n, 3, 3)
       list of orientations
    voxel_dims: 3-tuple
       the voxel dimensions in each direction
    origin: tuple | array, default = (0,0,0)
       lower left corner of box
    direction: 3-tuple of bools, default = (True, True, True)
       voxel directions, True meaning values go from low to high
    """

    def __init__(
            self, grain_ids, orientation_list, voxel_dims,
                 origin=(0,0,0), direction=3 * (True,)
    ):
        self.gids = grain_ids
        self.shape = grain_ids.shape
        self._num_grains = self.gids.max() + 1
        self.vdims = np.array(voxel_dims)
        self.origin = np.array(origin)
        self.lowleft = self.origin
        self.upright = self.origin + self.shape * self.vdims
        self._orientations = orientation_list
        self.direction = direction

    @property
    def num_grains(self):
        return self._num_grains

    @property
    def num_phases(self):
        return 1

    @property
    def direction(self):
        """Voxel order direction (increasing/decreasing)"""
        return self._direction

    @direction.setter
    def direction(self, v):
        self._direction = v
        self.v0 = np.where(v, self.lowleft, self.upright)
        self.dv = np.where(v, 1, -1) * self.vdims

    @property
    def num_cells(self):
        """number of cells"""
        return np.prod(self.shape)

    def _in_box(self, x):
        """determine if point lies in the box of cells"""
        okll = np.all((x - self.lowleft) >= 0.)
        okur = np.all((self.upright - x) >= 0.)
        return okll and okur

    def voxel_ijk(self, x):
        """determine cell containing point x"""
        dx = x - self.v0
        xdivs = (dx / self.dv).astype(int)
        ds = np.array(self.shape)
        xdivs = np.minimum(xdivs, ds - 1)

        return tuple(xdivs)

    def grain(self, x):
        n = len(x)
        vox = np.zeros((n, 3), dtype=int)
        for i in range(n):
            vox[i] = self.voxel_ijk(x[i])

        return self.gids[vox[:,0], vox[:,1], vox[:,2]]

    def phase(self, x):
        """Determine grain ID for point x"""
        return 0

    @property
    def orientation_list(self):
        return self._orientations

    def grain_orientation(self, g):
        return self.orientation_list[g]

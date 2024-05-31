"""Tests for voronoi module"""
import numpy as np

import pytest

from polycrystal.microstructure import voxeldata


def identity_array(n):
    return np.tile(np.identity(3), (n, 1, 1))


@pytest.fixture
def voxeldata_2x3x4():
    shp = (4, 3, 2)
    w6 = np.ones(6)
    gids = np.hstack((0 * w6, 1 * w6, 2 * w6, 3 * w6)).reshape(shp)
    oris = identity_array(4)
    vdim = np.ones(3)
    return voxeldata.VoxelData(gids, oris, vdim)


def test_atts(voxeldata_2x3x4):
    assert voxeldata_2x3x4.shape == (4, 3, 2)
    assert np.all(voxeldata_2x3x4.vdims == 1)
    assert np.all(voxeldata_2x3x4.origin == 0)
    assert np.all(voxeldata_2x3x4.lowleft == 0)
    assert np.all(voxeldata_2x3x4.upright == (4, 3, 2))
    assert voxeldata_2x3x4.num_cells == 24
    ida = identity_array(4)
    assert np.all(voxeldata_2x3x4.orientation_list == ida)
    assert voxeldata_2x3x4.direction == (True, True, True)
    assert voxeldata_2x3x4.num_grains == 4
    assert voxeldata_2x3x4.num_phases == 1


def test_grain(voxeldata_2x3x4):
    """Test grain ID assignment"""
    x = np.linspace((0.5, 1.0, 1.0), (3.5, 1.0, 1.0), 4)
    assert np.all(voxeldata_2x3x4.grain(x) == (0, 1, 2, 3))

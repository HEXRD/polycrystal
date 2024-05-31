"""Tests for voronoi module"""
import numpy as np

import pytest

from polycrystal.microstructure import voronoi


@pytest.fixture
def voronoi_2():
    seeds = np.array([[0.25, 0], [0.75, 0]])
    matrix = np.array
    return voronoi.Voronoi(seeds, None)


@pytest.fixture
def voronoi_2m():
    """voronoi 2D with matrix"""
    seeds = np.array([[0.25, 0], [0.75, 0]])
    matrix = np.diag([2, 3])
    return voronoi.Voronoi(seeds, None, matrix=matrix)


@pytest.fixture
def voronoi_3():
    seeds = np.linspace((0, 0, 0), (0, 0, 1), 11)
    return voronoi.Voronoi(seeds, None)


@pytest.fixture
def voronoi_3b():
    seeds = np.linspace((0, 0, 0), (0, 0, 1), 11)
    box = [(0, 1), (0, 2), (-1, 1)]
    return voronoi.Voronoi(seeds, None, box=box)


def test_shape2(voronoi_2):
    assert voronoi_2.num_grains == 2
    assert voronoi_2.dim == 2
    assert voronoi_2.orientations.shape == (2, 2, 2)


def test_shape3(voronoi_3):
    assert voronoi_3.num_grains == 11
    assert voronoi_3.dim == 3
    assert voronoi_3.orientations.shape == (11, 3, 3)


def test_contains(voronoi_3b):
    pts = np.array(
        [(0, 0, 0), (1.1, 0 , 0), (0, 1.1, 0), (0, 3, 0), (0, 0, -0.5)]
    )
    in_box = np.array((1, 0, 1, 0, 1), dtype=bool)
    print(voronoi_3b.contains(pts))
    assert np.all(voronoi_3b.contains(pts) == in_box)

def test_grain(voronoi_2):
    x = np.linspace((0, 0), (1, 0), 10)
    grains = voronoi_2.grain(x)
    assert np.all(grains[:5] == 0) and np.all(grains[5:] == 1)


def test_grain_matrix(voronoi_2m):
    x = np.linspace((0, 0), (1, 0), 10)
    grains = voronoi_2m.grain(x)
    assert np.all(grains[:5] == 0) and np.all(grains[5:] == 1)


def test_save_load(voronoi_2, tmp_path):
    p = tmp_path / "v.npz"
    voronoi_2.save(p)
    v2 = voronoi.Voronoi.from_file(p)
    assert np.allclose(v2.seeds, voronoi_2.seeds)
    assert np.allclose(v2.orientations[0], np.identity(2))


def test_save_load_box(voronoi_3b, tmp_path):
    p = tmp_path / "v3b.npz"
    voronoi_3b.save(p)
    v3b = voronoi.Voronoi.from_file(p)
    assert np.allclose(v3b.box, voronoi_3b.box)
    assert np.allclose(v3b.seeds, voronoi_3b.seeds)
    assert np.allclose(v3b.orientations[0], np.identity(3))


def test_random():
    n = 20
    box = np.stack((np.zeros(3), np.ones(3)), axis=1)
    mat = np.diag((2, 3, 5))
    v = voronoi.Voronoi.random_voronoi(n, box, fname=None, matrix=mat)
    assert len(v.seeds) == n
    assert np.all(v.matrix == mat)
    assert np.all(v.box == box)


def test_save_load_random(tmp_path):
    p = tmp_path / "random.npz"
    n = 20
    box = np.stack((np.zeros(3), np.ones(3)), axis=1)
    mat = np.diag((2, 3, 5))
    v = voronoi.Voronoi.random_voronoi(n, box, fname=p, matrix=mat)
    v.save(p)
    vff = voronoi.Voronoi.from_file(p)
    assert np.allclose(vff.box, v.box)
    assert np.allclose(vff.seeds, v.seeds)
    assert np.allclose(vff.orientations, v.orientations)
    assert np.allclose(vff.matrix, v.matrix)

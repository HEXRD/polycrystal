"""Test utils package"""

import numpy as np

from polycrystal.utils import tensordata as td
from polycrystal.utils import unique_vectors as uv


# Testing tensordata
TOL = 1e-14
n = 6
mats = np.random.random((n, 3, 3))
tdata = td.TensorData(matrices=mats)


def test_unique_vectors():
    """test unique_vectors"""
    a = np.array([[1, 2], [3.2, 1], [0, 4], [1, 2]])
    b = np.array([[0, 4], [1, 2], [3.2, 1]])
    assert np.allclose(uv.unique_vectors(a), b)

    au, index = uv.unique_vectors(a, return_index=True)
    assert np.all(au == b)
    assert(np.all(a[index] == b))

    au, inverse = uv.unique_vectors(a, return_inverse=True)
    assert(np.all(a == b[inverse]))

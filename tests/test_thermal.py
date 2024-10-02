"""Test thermal package"""
import numpy as np

from polycrystal.thermal.single_crystal import SingleCrystal


def test_tensor():
    """Check tensor for various symmetries"""
    sc = SingleCrystal('isotropic', (1))
    assert np.allclose(sc.conductivity, np.identity(3))

    sc = SingleCrystal('cubic', (1))
    assert np.allclose(sc.conductivity, np.identity(3))

    sc = SingleCrystal('hexagonal', (1, 2))
    assert np.allclose(sc.conductivity, np.diag((1, 1, 2)))

    d = (4.3, 2, 3)
    sc = SingleCrystal('orthotropic', d)
    assert np.allclose(sc.conductivity, np.diag(d))

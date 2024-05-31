"""Unit testing for quaternions module
"""
import unittest
import numpy as np

from polycrystal.orientations.conventions import baseclass
from polycrystal.orientations.conventions import euler
from polycrystal.orientations.conventions import expmap


class TestRegistry(unittest.TestCase):

    def test_baseclass(self):
        b = baseclass.ConventionRegistry('Convention', (), {})
        msg = "Convention base class needs no keyword"
        self.assertTrue(isinstance(b, type), msg=msg)

    def test_no_convention(self):
        msg = "no convention keyword should raise RuntimeError"
        with self.assertRaises(RuntimeError, msg=msg):
            b = baseclass.ConventionRegistry('NewConvention', (), {})


class TestEulerAngles(unittest.TestCase):

    def test_identity(self):
        ea = euler.EulerAngles()
        a = np.zeros(3)
        r = np.identity(3)
        rmat = ea.to_rmats(np.zeros(3))
        self.assertTrue(np.linalg.norm(rmat - r) < 1.0e-12)

    def test_90deg(self):
        """simple 90 degree rotations & combinations"""
        ea = euler.EulerAngles()
        a = np.array([
            [90., 0., 0],
            [0., 90., 0],
            [0., 0., 90],
            [90., 90., 0],
            [90., 0., 90],
            [0., 90., 90],
            [90., 90., 90]
        ])
        r = np.array(
            [
                [[0.,-1,0],[1,0,0],[0,0,1]],
                [[1,0,0],[0,0,-1],[0,1,0]],
                [[0,-1,0],[1,0,0],[0,0,1]],
                [[0,0,1],[1,0,0],[0,1,0]],
                [[-1,0,0],[0,-1,0],[0,0,1]],
                [[0,-1,0],[0,0,-1],[1,0,0]],
                [[0,0,1],[0,-1,0],[1,0,0]]

            ])
        rmat = ea.to_rmats(a)
        self.assertTrue(np.linalg.norm(rmat - r) < 1.0e-12)


class TestExpMap(unittest.TestCase):

    def test_identity(self):
        conv = expmap.ExpMap()
        a = np.zeros(3)
        r = np.identity(3)
        rmat = conv.to_rmats(np.zeros(3))
        self.assertTrue(np.linalg.norm(rmat - r) < 1.0e-12)

    def test_90deg(self):
        """simple 90 degree rotations & combinations"""
        conv = expmap.ExpMap()
        p2 = np.pi/2.
        a = np.array([
            [p2, 0., 0],
            [0., p2, 0],
            [0., 0., p2]
        ])
        r = np.array(
            [
                [[1.,0,0],[0,0,-1],[0,1,0]],
                [[0,0,1],[0,1,0],[-1,0,0]],
                [[0,-1,0],[1,0,0],[0,0,1]]
            ])
        rmat = conv.to_rmats(a)
        self.assertTrue(np.linalg.norm(rmat - r) < 1.0e-12)


if __name__ == '__main__':
    unittest.main()

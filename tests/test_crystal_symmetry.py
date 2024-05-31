"""Unit tests for cyrstalsymmetry module
"""
import unittest

import numpy as np

from polycrystal.orientations import crystalsymmetry as cs
from polycrystal.orientations import quaternions as quats


class TestCrystalSymmetry(unittest.TestCase):
    """TestCrystalSymmetry"""

    def setUp(self):
        self.hsym = cs.get_symmetries('hexagonal')

    def test_nsymm(self):
        """number of crystal symmetries"""
        nsymmd = {'identity': 1,
                  'monoclinic': 2,
                  'orthorhombic': 4,
                  'cubic': 24,
                  'hexagonal': 12}
        for n in nsymmd:
            sym = cs.get_symmetries(n)
            msg = 'incorrect number of symmetries for: %s' % n
            self.assertEqual(sym.nsymm, nsymmd[n], msg)

    def test_rmats(self):
        """Test rmats property"""
        sym = cs.get_symmetries('monoclinic')
        r = sym.rmats
        self.assertEqual(r.shape, (2, 3, 3))
        self.assertTrue(np.allclose(r[0], np.identity(3)))
        r180y = np.diag((-1, 1, -1))
        self.assertTrue(np.allclose(r[1], r180y))
        # self.assertAlmostEqual(err.max(), 0., msg=msg)

    def test_to_fundamental_region(self):
        """testing referral to fundamental region"""
        id = quats.identity()
        for n in cs.list_symmetries():
            sym = cs.get_symmetries(n)
            qfr = sym.to_fundamental_region(sym.quats)
            err = np.linalg.norm(qfr - id, axis=1)
            msg = 'reflexive fundamental region test failed for'\
              ' symmetries: %s' % n
            self.assertAlmostEqual(err.max(), 0., msg=msg)

    def test_average(self):
        """average orientation"""
        id = quats.identity()
        for n in cs.list_symmetries():
            sym = cs.get_symmetries(n)
            avg = sym.average_orientation(sym.quats)
            err = np.linalg.norm(avg - id, axis=1)
            msg = 'average orientation failed for symmetries: %s' % n
            self.assertAlmostEqual(err.max(), 0., msg=msg)

        msg = 'failed for near-identity case'
        ndeg = 5.
        theta = 0.5*ndeg*np.pi/180.
        q = np.zeros((6,4))
        q[:,0] = np.cos(theta)
        q[0:3,1:] = np.sin(theta)*np.identity(3)
        q[3:6,1:] = -np.sin(theta)*np.identity(3)
        avg = self.hsym.average_orientation(q)
        err = np.linalg.norm(avg - id, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

    def test_misorientation(self):
        """misorientation"""
        delta = np.array([1,2,4])
        fac = 0.01
        qid = np.zeros((3,4))
        qid[:, 0] = 1.
        q = np.zeros((3,4))
        q[:,0] = np.cos(delta*fac)
        q[:,1] = np.sin(delta*fac)

        msg = 'failed for misorientation'
        mis = self.hsym.misorientation(qid, q)
        err = np.linalg.norm(mis - q, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

    pass  # end class


if __name__ == '__main__':
    unittest.main()

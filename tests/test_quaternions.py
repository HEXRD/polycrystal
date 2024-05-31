"""Unit testing for quaternions module
"""
import unittest
import numpy as np

from polycrystal.orientations import quaternions as quats

class TestQuats(unittest.TestCase):
    def setUp(self):
        # 90 degrees about axes
        ang = (np.pi/2)*np.ones((3, 1))
        ang2 = ang * 0.5
        id = np.eye(3)
        qs = np.cos(ang2)
        qv = np.sin(ang2)*id
        self.q90 = np.hstack((qs, qv))

        # 180 degrees about axes
        id_neg = -np.identity(3)
        r180 = np.tile(id_neg, (3,1,1))
        ind = (0,1,2)
        r180[(ind, ind, ind)] = 1.
        q180 = np.zeros((3, 4))
        q180[((0,1,2),(1,2,3))] = 1.
        self.r180 = r180
        self.q180 = q180

        # sequence about z-axis
        na = 11
        a = np.linspace(0, np.pi, na)
        # . set up matrices
        ca = np.cos(a)
        sa = np.sin(a)
        ra = np.zeros((na,3,3))
        ra[:,0,0] = ra[:,1,1] = ca
        ra[:,1,0] = sa
        ra[:,0,1] = -sa
        ra[:,2,2] = 1.
        # . set up quaternions
        ca = np.cos(a/2)
        sa = np.sin(a/2)
        qa = np.zeros((na,4))
        qa[:,0] = ca
        qa[:,3] = sa
        #
        self.rz = ra
        self.qz = qa

    def test_identity_values(self):
        self.assertEqual(quats.identity()[0, 0], 1.)
        self.assertEqual(quats.identity()[0, 1], 0.)
        self.assertEqual(quats.identity()[0, 2], 0.)
        self.assertEqual(quats.identity()[0, 3], 0.)

    def test_identity_mult(self):
        id = quats.identity()

        msg = 'multiplication by identity on left failed '
        qp = quats.multiply(id, self.q90)
        err = np.linalg.norm(qp - self.q90, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

        msg = 'multiplication by identity on right failed '
        qp = quats.multiply(self.q90, id)
        err = np.linalg.norm(qp - self.q90, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

    def test_inverse(self):
        id = quats.identity()
        qinv = quats.inverse(self.q90)

        msg = 'multiplication by inverse on left failed '
        qp = quats.multiply(qinv, self.q90)
        err = np.linalg.norm(qp - id, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

        msg = 'multiplication by inverse on right failed '
        qp = quats.multiply(self.q90, qinv)
        err = np.linalg.norm(qp - id, axis=1)
        self.assertAlmostEqual(err.max(), 0., msg=msg)

    def test_from_rmats(self):
        # identity
        msg = 'rmat to quat conversion failed for identity'
        idr = np.identity(3)
        idq = quats.identity()
        q = quats.from_rmats(idr)
        err = np.linalg.norm(q - idq)
        self.assertAlmostEqual(err, 0., places=12, msg=msg)

        # 180 degree rotations
        msg = 'rmat to quat conversion failed on binary rotations'
        q = quats.from_rmats(self.r180)
        err = np.linalg.norm(q - self.q180)
        self.assertAlmostEqual(err, 0., places=12, msg=msg)

        # general case: rotation sequence about z-axis
        msg = 'rmat to quat conversion failed on z-axis sequence'
        q = quats.from_rmats(self.rz)
        err = np.linalg.norm(q - self.qz, axis=1)
        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

    def test_to_rmats(self):
        """quaternion array to rotation matrices"""
        msg = 'failed on identity'
        id = quats.identity()
        ra = quats.to_rmats(id)
        err = np.linalg.norm(ra - np.eye(3), axis=(1,2))
        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

        msg = 'failed on binary rotations'
        r180 = quats.to_rmats(self.q180)
        err = np.linalg.norm(r180 - self.r180, axis=(1,2))
        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

        msg = 'failed on z-axis sequence'
        r = quats.to_rmats(self.qz)
        err = np.linalg.norm(r - self.rz, axis=(1,2))
        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

    def test_from_exp(self):
        """quaternions from exponential map"""
        w = np.zeros((7,3))
        w[((1,2,3),(0,1,2))] = np.pi/2.
        w[((4,5,6),(0,1,2))] = np.pi

        qstar = np.vstack((quats.identity(), self.q90, self.q180))

        msg = 'failed on 0-90-180 rotations'
        q = quats.from_exp(w)
        err = np.linalg.norm(q - qstar, axis=(1,))
        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

    def test_multiply(self):
        """quaternion multiplication"""
        msg = 'failed to agree with rotation matrix multiplication'
        n = 5
        q1 = np.linspace(0, 10, 4*n).reshape((n,4))
        q1 = q1/np.linalg.norm(q1, axis=1).reshape(n,1)
        R1 = quats.to_rmats(q1)

        q2 = np.linspace(5, 12, 4*n).reshape((n,4))
        q2 = q2/np.linalg.norm(q2, axis=1).reshape(n,1)
        R2 = quats.to_rmats(q2)

        qp = quats.multiply(q1, q2)
        Rp = np.einsum('ijk,ikl->ijl', R1, R2)
        qpR = quats.from_rmats(Rp)
        # account for equivalent quaternions (+/-)
        err1 = np.linalg.norm(qp - qpR, axis=(1,))
        err2 = np.linalg.norm(qp + qpR, axis=(1,))
        err = np.minimum(err1, err2)

        self.assertAlmostEqual(err.max(), 0., places=12, msg=msg)

if __name__ == '__main__':
    unittest.main()

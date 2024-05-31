"""Exponential map convention"""
import numpy as np

from .baseclass import Convention


class ExpMap(Convention):
    """Exponential map"""
    convention = 'exp-map'
    ANG_ZERO = 1.0e-16

    def to_rmats(self, a):
        w = np.atleast_2d(a)
        n, d = w.shape
        if not d==3:
            raise RuntimeError("Exponential map array has wrong shape")

        ang = np.linalg.norm(w, axis=1) + self.ANG_ZERO

        rmat = np.empty((n,d,d))
        c1 = np.sin(ang) / ang
        c2 = (1. - np.cos(ang)) / (ang*ang)

        Z = np.zeros((n,1))
        I3 = np.tile(np.identity(d), (n, 1, 1))

        _w0 = w[:,0:1]
        _w1 = w[:,1:2]
        _w2 = w[:,2:]
        W1 = np.hstack((
            Z,    -_w2, +_w1,
            +_w2,   Z,  -_w0,
            -_w1, +_w0,   Z
        )).reshape(n,3,3)
        W2 = np.matmul(W1, W1)

        c1W1 = np.tile(c1.reshape(n,1), (1,9)).reshape((n,3,3)) * W1
        c2W2 = np.tile(c2.reshape(n,1), (1,9)).reshape((n,3,3)) * W2

        return I3 + c1W1 + c2W2

    def from_rmats(self, r):
        msg = (
            "Conversion from rotation matrices not implemented for this "
            "convention"
        )
        raise NotImplementedError(msg)

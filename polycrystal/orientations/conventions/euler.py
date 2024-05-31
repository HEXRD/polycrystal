"""Euler Angles convention

* Degrees is default
"""
import numpy as np

from .baseclass import Convention


class EulerAngles(Convention):
    """Euler angles class"""
    convention = 'euler-angles'

    def to_rmats(self, a):
        ar = np.atleast_2d(np.radians(a))
        n, d = ar.shape
        if not d==3:
            raise RuntimeError("Euler angles array has wrong shape")
        ca = np.cos(ar);
        sa = np.sin(ar);

        rmat = np.empty((n,d,d))
        rmat[:,0,0] = ca[:, 0]*ca[:, 2] - sa[:,0]*ca[:,1]*sa[:,2]
        rmat[:,0,1] = -ca[:, 0]*sa[:, 2] - sa[:,0]*ca[:,1]*ca[:,2]
        rmat[:,0,2] = sa[:, 0]*sa[:, 1]

        rmat[:,1,0] = sa[:, 0]*ca[:, 2] + ca[:,0]*ca[:,1]*sa[:,2]
        rmat[:,1,1] = -sa[:, 0]*sa[:, 2] + ca[:,0]*ca[:,1]*ca[:,2]
        rmat[:,1,2] = -ca[:, 0]*sa[:, 1]

        rmat[:,2,0] = sa[:, 1]*sa[:, 2]
        rmat[:,2,1] = sa[:, 1]*ca[:, 2]
        rmat[:,2,2] = ca[:, 1]

        return rmat

    def from_rmats(self, r):
        msg = (
            "Conversion from rotation matrices not implemented for this "
            "convention"
        )
        raise NotImplementedError(msg)

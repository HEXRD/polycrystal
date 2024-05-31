"""Unit quaternions"""
import numpy as np

from .baseclass import Convention
from ..quaternions import to_rmats, from_rmats


class Quaternion(Convention):
    """Quaternion"""
    convention = 'quaternions'
    _to_error_msg = "Quaternion array has wrong shape"
    _from_error_msg = "Rotation array has wrong shape"

    def to_rmats(self, a):
        ar = np.atleast_2d(a)
        n, d = ar.shape
        if not d==4:
            raise RuntimeError(self._to_error_msg)

        return to_rmats(a)

    def from_rmats(self, r):
        ok = True
        if r.ndim == 2:
            r3 = r.reshape((1,) + r.shape)
        else:
            r3 = r

        shp = r3.shape
        if r3.ndim == 3:
            if shp[1] !=3 or shp[2]!= 3:
                ok = False
        else:
            ok = False

        if not ok:
            raise RuntimeError(self._from_error_msg + str(shp))

        return from_rmats(r3)

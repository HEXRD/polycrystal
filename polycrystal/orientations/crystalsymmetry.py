"""Crystal Symmetries

The primary interface to this module is through the `get_symmetries()`
function, in which you can get a crystal symmetry group by name.  The
`list_symmetries()` lists the names of available symmetry groups.
"""
import numpy as np

from . import quaternions as quats


def list_symmetries():
    """List available symmetry groups"""
    return _Registry.registry.keys()


def get_symmetries(name):
    """Return symmetry group by name"""
    return _Registry.get(name)


class CrystalSymmetry(object):
    """Base class for crystal symmetry types

    Parameters
    ----------
    name: str
       name of the symmetry group
    qsymm: array (n, 4)
       quaternions for the crystal symmetry group
    """

    def __init__(self, name, qsymm):
        self._name = name
        self._q = qsymm
        self._rmats = quats.to_rmats(qsymm)
        _Registry.register(name, self)

    @property
    def name(self):
        """name of symmetry group"""
        return self._name

    @property
    def quats(self):
        """return quaternions of symmetry group"""
        return self._q.copy()

    @property
    def rmats(self):
        """return rotation matrices of symmetry group"""
        return self._rmats.copy()

    @property
    def nsymm(self):
        """number of symmetries in group"""
        return len(self._q)

    def to_fundamental_region(self, q):
        """find equivalent quaternions in the fundamental region

        Parameters
        ----------
        q: array (n, 4)
           an array of `n` quaternions

        Returns
        -------
        array (n, 4):
           array of symmetrically equivalent quaternions in fundamental region
        """

        # * apply crystal symmetries on right for convention: Rc=s
        # * enforce scalar part of quaternion nonnegative

        qfr = np.zeros_like(q)
        for i in range(len(q)):
            qeqv = quats.multiply(q[i], self._q)
            j = np.abs(qeqv[:, 0]).argmax()
            qfr[i] = qeqv[j]
            if qfr[i, 0] < 0.:
                qfr[i] = -qfr[i]

        return qfr

    def average_orientation(self, q):
        """average orientation of clustered group of orientations

        Parameters
        ----------
        q: array (n, 4)
           an array of `n` quaternions

        Returns
        -------
        array (n, 4):
           the average orientation (mean quaternion after applying symmetries
           to group them)
        """
        q0 = q[0].copy()
        qtmp = quats.multiply(quats.inverse(q0), q)
        qtmp = self.to_fundamental_region(qtmp)
        qtmpavg = qtmp.mean(axis=0)
        qtmpavg = qtmpavg/np.linalg.norm(qtmpavg)
        qavg = quats.multiply(q0, qtmpavg)

        return self.to_fundamental_region(qavg)

    def misorientation(self, q1, q2):
        """misorientation between crystals

        Parameters
        ----------
        q1, q2: array (n, 4)
           arrays of `n` quaternions; either can be a single quaternion, but
           if neither are, they must be of the same length

        Returns
        -------
        array (n, 4):
           the misorientation quaternion (taking q1 to q2) of smallest angle
        """
        qmis = quats.multiply(quats.inverse(q1), q2)
        return self.to_fundamental_region(qmis)

    def misorientation_angle(self, q1, q2):
        """misorientation between crystals

         Parameters
        ----------
        q1, q2: array (n, 4)
           arrays of `n` quaternions; either can be a single quaternion, but
           if neither are, they must be of the same length

        Returns
        -------
        array (n):
           the angle of the smallest misorientation quaternion
        """
        return 2.0*np.arccos(self.misorientation(q1, q2)[:, 0])

    pass  # end class

# ==================== Registry


class _Registry(object):
    """Registry for symmetry type instances"""
    registry = dict()

    @classmethod
    def register(cls, name, inst):
        """Register instance"""
        cls.registry[name] = inst

    @classmethod
    def get(cls, name):
        """return instance associated with name"""
        return cls.registry[name]
    #
    pass  # end class


# ==================== Instances

_s2 = 1/np.sqrt(2)
_s3 = 1/np.sqrt(3)
_p3 = np.pi/3.
_p4 = np.pi/4.
_p6 = np.pi/6.
_c = np.cos
_s = np.sin

# Identity: trivial group (triclinic)

_qident = np.array([[1., 0., 0., 0.]])
_identity = CrystalSymmetry('identity', _qident)


# Monoclinic
# . conventionally, rotation about axis 2 (y)

_qmono = 1.0*np.array([
    [1,   0,   0,   0],
    [0,   0,   1,   0]
])
_monoclinic = CrystalSymmetry('monoclinic', _qmono)


# Orthorhombic

_qortho = 1.0*np.array([
    [1,   0,   0,   0],  # identity
    [0,   1,   0,   0],  # twofold about 100
    [0,   0,   1,   0],  # twofold about 010
    [0,   0,   0,   1],  # twofold about 001
])
_orthorhombic = CrystalSymmetry('orthorhombic', _qortho)


# Hexagonal
#
_qhex = np.array([
    [1., 0, 0, 0],

    [  _c(_p6), 0, 0,   _s(_p6)],  # c-axis rotations
    [_c(2*_p6), 0, 0, _s(2*_p6)],
    [        0, 0, 0,         1],
    [_c(4*_p6), 0, 0, _s(4*_p6)],
    [_c(5*_p6), 0, 0, _s(5*_p6)],

    [0,        1.,         0, 0],  # binary rotations
    [0,   _c(_p6),   _s(_p6), 0],
    [0, _c(2*_p6), _s(2*_p6), 0],
    [0,         0,         1, 0],
    [0, _c(4*_p6), _s(4*_p6), 0],
    [0, _c(5*_p6), _s(5*_p6), 0],
])
_hexagonal = CrystalSymmetry('hexagonal', _qhex)


# Cubic
_qcubic = 1.0*np.array([
    [1, 0, 0, 0],

    [  _c(_p4),   _s(_p4), 0, 0],  # about [1,0,0]
    [_c(2*_p4), _s(2*_p4), 0, 0],
    [_c(3*_p4), _s(3*_p4), 0, 0],

    [  _c(_p4), 0,   _s(_p4), 0],  # about [0, 1, 0]
    [_c(2*_p4), 0, _s(2*_p4), 0],
    [_c(3*_p4), 0, _s(3*_p4), 0],

    [  _c(_p4), 0, 0,   _s(_p4)],  # about [0, 0, 1]
    [_c(2*_p4), 0, 0, _s(2*_p4)],
    [_c(3*_p4), 0, 0, _s(3*_p4)],

    [_c(2*_p3), _s(2*_p3)*_s3, _s(2*_p3)*_s3, _s(2*_p3)*_s3],  # [1, 1, 1]
    [_c(4*_p3), _s(4*_p3)*_s3, _s(4*_p3)*_s3, _s(4*_p3)*_s3],

    [_c(2*_p3), -_s(2*_p3)*_s3, _s(2*_p3)*_s3, _s(2*_p3)*_s3],  # [-1, 1, 1]
    [_c(4*_p3), -_s(4*_p3)*_s3, _s(4*_p3)*_s3, _s(4*_p3)*_s3],

    [_c(2*_p3), -_s(2*_p3)*_s3, -_s(2*_p3)*_s3, _s(2*_p3)*_s3],  # [-1, -1, 1]
    [_c(4*_p3), -_s(4*_p3)*_s3, -_s(4*_p3)*_s3, _s(4*_p3)*_s3],

    [_c(2*_p3), _s(2*_p3)*_s3, -_s(2*_p3)*_s3, _s(2*_p3)*_s3],  # [1, -1, 1]
    [_c(4*_p3), _s(4*_p3)*_s3, -_s(4*_p3)*_s3, _s(4*_p3)*_s3],

    [0,   _s2,     _s2,     0],  # binary rotations
    [0,  -_s2,     _s2,     0],
    [0,   _s2,      0,    _s2],
    [0,     0,    _s2,    _s2],
    [0,  -_s2,      0,    _s2],
    [0,     0,   -_s2,    _s2]
])
_cubic = CrystalSymmetry('cubic', _qcubic)

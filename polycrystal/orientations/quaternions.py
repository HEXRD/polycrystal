"""Quaternion operations"""
import numpy as np

# default cutoff for angles (in radians) near 0/180
_DFLT_CUT = 1e-6


def _rankone(a, b):
    """return array of rank one matrices: ai*ai.T"""
    return np.einsum('ij,ik->ijk', a, b)


def _vec2skew(v):
    """axial vector to skew matrix"""
    W = np.zeros((len(v), 3, 3))
    W[:,2,1] = v[:, 0]
    W[:,0,2] = v[:, 1]
    W[:,1,0] = v[:, 2]
    W[:,1,2] = -W[:,2,1]
    W[:,2,0] = -W[:,0,2]
    W[:,0,1] = -W[:,1,0]
    return W


def multiply(q_1, q_2):
    """multiply two quaternion arrays

    Parameters
    ----------
    q1, q2: arrays (n, 4)
       two arrays of unit quaternions; the arrays are expected to be of the
       same length, but either array can be a single quaternion (1, 4), in
       which case the single quaternion is multiplied by all in the other array

    Returns
    -------
    q1 * q2: array (n, 4)
        the array of quaternion products of corresponding elements

    Notes
    -----
    The result is not processed to enforce nonnegativity of the first component
    (as was done in the matlab  OdfPf package).
    """
    # Reshape q1/q2 if 1D
    if q_1.ndim == 1:
        q1 = q_1.copy().reshape((1,4))
    else:
        q1 = q_1

    if q_2.ndim == 1:
        q2 = q_2.copy().reshape((1,4))
    else:
        q2 = q_2

    n1 = len(q1)
    n2 = len(q2)
    n = np.maximum(n1, n2)
    s1 = q1[:, 0].reshape((n1,1))
    v1 = q1[:, 1:].reshape((n1,3))
    s2 = q2[:, 0].reshape((n2,1))
    v2 = q2[:, 1:].reshape((n2,3))
    qs = s1*s2 - np.sum(v1*v2, axis=1).reshape((n,1))
    qv = (s1*v2) + (s2*v1) + np.cross(v1, v2)
    qprod = np.hstack((qs, qv))
    return qprod/np.linalg.norm(qprod, axis=1).reshape(n, 1)


def inverse(q):
    """inverse of quaternion or array

    Parameters
    -----------
    q: array (n, 4)
       array of unit quaternions

    Returns
    -------
    array (n, 4)
       an array of inverse quaternions
    """
    qi = q.copy()
    if qi.ndim == 1:
        qi = qi.reshape((1,4))
    qi[:, 1:] = -qi[:, 1:]
    return qi


def identity():
    """return identity quaternion"""
    return np.array([[1.,0.,0.,0.]])


def to_rmats(q):
    """Convert quaternions to rotation matrices

    Parameters
    ----------
    q: array (n, 4)
       array of unit quaternions

    Returns
    -------
    r: array (n, 3, 3)
       arary of rotation matrices
    """
    # with w = theta*n, c = c(theta), s = s(theta)
    # R(w) = cI + (1 - c)(I - NN^T) + s W(n)
    #      = (qs^2 - qv^2) I + 2qv*qv.T + 2qs*W(qv)
    n = len(q)

    qs = q[:, 0].reshape((n,1))
    qv = q[:, 1:4]
    qvqv = np.sum(qv*qv, axis=1).reshape(n,1)

    r = _rankone(qv, 2.*qv) + _vec2skew(2.*qs*qv)
    for i in range(3):
        r[:,i,i] += (qs*qs - qvqv).reshape((n))

    return r


def from_rmats(r, cut=_DFLT_CUT):
    """Convert rotation matrices to quaternions

    Parameters
    ----------
    r: array (n, 3, 3)
       arary of rotation matrices

    Returns
    -------
    q: array (n, 4)
       array of unit quaternions
    """
    if r.ndim == 3:
        rm = r
    else:
        # line below failed due to matrix reshape issue
        # rm = r.reshape((1,3,3))
        rm = np.asarray(r).copy().reshape((1,3,3))
    nq = len(rm)

    # find angle
    ca = 0.5*(rm.trace(axis1=1, axis2=2) - 1);
    ca = np.minimum(ca, 1.);
    ca = np.maximum(ca, -1.);
    angle = np.arccos(ca)

    # find axis (a little harder)
    w0 = (rm[:, 2, 1] - rm[:, 1, 2]).reshape((nq, 1))
    w1 = (rm[:, 0, 2] - rm[:, 2, 0]).reshape((nq, 1))
    w2 = (rm[:, 1, 0] - rm[:, 0, 1]).reshape((nq, 1))
    w = np.hstack((w0, w1, w2))

    w[np.where(angle < cut)] = np.ones(3)

    nearpi = np.nonzero(angle > np.pi - cut)[0]
    n = len(nearpi)
    wpi = np.zeros((n, 3))
    if n > 0:
        rpi = (rm[nearpi] - ca[nearpi].reshape((n,1,1))*np.eye(3))
        rpi = 0.5* (rpi + rpi.transpose((0,2,1)))
        di = np.diag_indices(3)
        for i in range(len(rpi)):
            ind = np.argmax(rpi[i][di])
            wpi[i] = rpi[i, ind, :]
        w[nearpi] = wpi

    sca = np.cos(0.5*angle).reshape(nq, 1)
    vec = w*(np.sin(0.5*angle)/np.linalg.norm(w, axis=1)).reshape((nq,1))
    q = np.hstack((sca, vec))

    return q


def from_exp(w, cut=_DFLT_CUT):
    """Quaternions from exponential map

    Parameters
    ----------
    w: array (n, 3)
       arary of axial vectors

    Returns
    -------
    q: array (n, 4)
       array of unit quaternions

    Notes
    -----
    The exponential map refers to the matrix exponential of a skew matrix.
    If W is a skew matrix, the exp(W) is a rotation matrix. The axial vector
    w of W is the exponential map parameter, and the rotation matrix has
    axis parallel to w and rotation angle the length of w.

"""
    n = len(w)
    a = np.linalg.norm(w, axis=1).reshape((n,1))
    a2 = a/2.
    c = np.cos(a2)
    s = np.sin(a2)

    # handle small angles near the origin
    asmall = np.where(a < cut)
    a[asmall] = np.sqrt(3)*np.ones_like(a[asmall])
    axis = w.copy()
    axis[asmall] = np.ones_like(axis[asmall])
    axis = axis/a
    return np.hstack((c, s*axis))


def random_quats(n, return_matrices=False):
    """Generate n random orientations

    Parameters
    ----------
    n: int
       the number of orientations to generate
    return_matrices: bool, defaul = True
       if True return matrices, otherwise quaternions
    """
    q = np.random.standard_normal((n, 4))
    nrm = np.linalg.norm(q, axis=1)
    if nrm.min() == 0.0:
        raise ValueError("generated zero magnitude vector, try again")
    qrand = q/nrm.reshape((n, 1))

    if return_matrices:
        return to_rmats(qrand)
    else:
        return qrand


def random_rmats(n):
    """return random uniformly distributed rotation matrices
    Parameters
    ----------
    n: int
       the number of orientations to generate
    """
    return random_quats(n, return_matrices=True)

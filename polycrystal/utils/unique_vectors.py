"""Unique vector function"""
import numpy as np


DEFAULT_TOL = 1.0E-8


def unique_vectors(
        a, tol=DEFAULT_TOL, return_index=False, return_inverse=False
):
    """return vectors in a which are unique within the tolerance

    Parameters
    ----------
    a: array (n, k)
       input array with `n` rows
    tol: float, default = DEFAULT_TOL
       tolerance to compare array values
    return_index: bool, default = False
       if True, return indices of `a` that give the unique vector
    return_inverse: bool, default = False
       if True, return indices that reconstruct the input

    Returns
    -------
    u: array (m, k)
       unique values of `a` within the tolerance
    index: int array(m)
       indices of unique elements of `a` (if `return_index` is True)
    inverse: int array(n)
       indices that reconstruct `a` (if `return_inverse` is True)
    """
    nr, nc = a.shape
    a_ind = a.argsort(axis=0)
    arank = _to_ranks(a, tol=tol)
    out = np.unique(
        arank, axis=0, return_index=return_index, return_inverse=return_inverse
    )

    if not return_index and not return_inverse:
        return out

    # Now, output is a tuple.
    u = out[0]
    retvals = [u]
    ind = 1
    if return_index:
        retvals.append(out[ind])
        ind += 1
    if return_inverse:
        retvals.append(out[ind])

    return tuple(retvals)


def _to_ranks(a, tol=DEFAULT_TOL):
    """Change floats to ranks for each column"""
    nr, nc = a.shape
    a_ind = a.argsort(axis=0)
    arank = np.zeros_like(a, dtype=int)
    rnk = np.zeros(nr, dtype=int)
    for j in range(nc):
        # rank values in each column
        col = a[:, j]
        ind = np.argsort(col)
        srt = col[ind]
        rnk[0] = 0
        for i in range(1,nr):
            rnk[i] = rnk[i-1]
            if np.abs(srt[i] - srt[i-1]) > tol:
                rnk[i] += 1
        #
        inv = np.argsort(ind)
        arank[:,j] = rnk[inv]

    return arank

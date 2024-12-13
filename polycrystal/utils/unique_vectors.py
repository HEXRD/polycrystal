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
        arank, axis=0, return_index=True, return_inverse=return_inverse
    )

    # Output is a tuple since we always need the index, as we are sorting
    # by rank.

    if return_inverse:
        _, index, inverse = out
    else:
        _, index = out

    # Now, set up output to return.

    u = a[index]

    ret = (u,)
    if return_index:
        ret += (index,)
    if return_inverse:
        ret += (inverse,)

    return u if len(ret) == 1 else ret


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

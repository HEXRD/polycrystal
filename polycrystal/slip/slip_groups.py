"""Groups of slip systems realted by crystal symmetry

This provides some standard slip system groups. The interface is provided
here by the `get_group` and `list_groups` functions.

REFERENCES
----------
.. [1] Pagan, Darren C., Kenneth M. Peterson, Paul A. Shade, Adam L. Pilchak,
   and David Dye. “Using the Ti–Al System to Understand Plasticity and Its
   Connection to Fracture and Fatigue in α Ti Alloys.” Metallurgical and
   Materials Transactions A 54, no. 9 (September 1, 2023): 3373–88.
   https://doi.org/10.1007/s11661-023-07114-9.
"""
import numpy as np

from ..orientations import crystalsymmetry
from .slipgroup import SlipGroup


__all__ = ['get_group', 'list_groups']


def get_group(name, cbya=None):
    """Get a slip group by name

    PARAMETERS
    ----------
    name: str
        name of the group; see `list_groups` for available names
    cbya: float or None, default = None
        the c/a ratio for hexagonal slip groups
    """
    if name.startswith(("fcc", "bcc")):
        return registry[name]()
    elif name.startswith("hcp"):
        if cbya is None:
            raise RuntimeError("c/a ratio not specified for HCP")
        return registry[name](cbya)


def list_groups():
    """list available slip system groups"""
    return list(registry.keys())


# This is the registry for slip system selection.

registry = dict()

# ======================================== Cubic

_cubsymm = crystalsymmetry.get_symmetries('cubic')

# FCC: {111}{1 1_ 0}
def get_fcc():
    return SlipGroup(np.array([1., 1, 1]), np.array([1., -1, 0]), _cubsymm)


registry['fcc'] = get_fcc

# BCC:


def get_bcc():
    return SlipGroup(np.array([1., -1, 0]), np.array([1., 1, 1]), _cubsymm)


def get_bcc_112():
    return SlipGroup(np.array([1., 1, -2]), np.array([1., 1, 1]), _cubsymm)


def get_bcc_123():
    return SlipGroup(np.array([1., 2, -3]), np.array([1., 1, 1]), _cubsymm)


registry['bcc'] = get_bcc
registry['bcc:112'] = get_bcc_112
registry['bcc:123'] = get_bcc_123

# ======================================== HCP
#
# NOTE:
# * We choose basis so that crystallographic a_1 aligns with (1,0,0)
#   coordinate vector. This is relevant as to how the orientation
#   matrix is defined.
# * Orthonormal system is:
#   e_1 || a_1,
#   e_3 || c,
#   e_2 = e3 X e1

_hexsymm = crystalsymmetry.get_symmetries('hexagonal')


def _miller_bravais_direction(uvtw, c_over_a):
    u, v, t, w = uvtw
    s120 = np.sqrt(3)/2
    return (1.5 * u, s120 * (2*v + u), c_over_a * w)


def _miller_bravais_normal(hkil, c_over_a):
    h, k, i, l = hkil
    s120 = np.sqrt(3)/2
    return (h, (k + 0.5*h)/s120, l/c_over_a)


# Basal: (000 1) [1 1 -2 0]
def get_hcp_basal(cbya):
    return SlipGroup(
        _miller_bravais_normal((0, 0, 0, 1), cbya),
        _miller_bravais_direction((1, 1, -2, 0), cbya),
        _hexsymm
    )


# Prismatic: (1 -1 0  0) [1 1 -2  0]
def get_hcp_prismatic(cbya):
    return SlipGroup(
        _miller_bravais_normal((1, -1, 0, 0), cbya),
        _miller_bravais_direction((1, 1, -2, 0), cbya),
        _hexsymm
    )


# Pyramidal a: (1 1_ 0  1) [1 1_ 2_  0]
def get_hcp_pyramidal_a(cbya):
    return SlipGroup(
        _miller_bravais_normal((0, 1, -1, 1), cbya),
        _miller_bravais_direction((2, -1, -1, 0), cbya),
        _hexsymm
    )


# Pyramidal c+a: (0 1 1_  1) [2 1_ 1_   3]
def get_hcp_pyramidal_ca(cbya):
    return SlipGroup(
        _miller_bravais_normal((0, 1, -1, 1), cbya),
        _miller_bravais_direction((-1, -1, 2, 3), cbya),
        _hexsymm
    )


registry['hcp:basal'] = get_hcp_basal
registry['hcp:prismatic'] = get_hcp_prismatic
registry['hcp:pyramidal_a'] = get_hcp_pyramidal_a
registry['hcp:pyramidal_c+a'] = get_hcp_pyramidal_ca

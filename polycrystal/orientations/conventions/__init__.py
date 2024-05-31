"""Orientation Conventions

The primary interface here is through the `to_rmats()`, `from_rmats()` and
`convert()` functions. To add a new convention, make a subclass of the
`Convention` class in `baseclass.py`.  Add the convention name in the
`convention` attribute. Then implement the `to_rmats()` and `from_rmats()`
methods for that class. The convention will be automatically registered here.

The conventions here are not meant to be exhaustive. They are mainly for
internal use, particularly the quaternions. The `scipy` Rotation class
is a good choice for more complete conversions.

See Also
---------

* `scipy.transform.Rotation`_

.. _scipy.transform.Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
"""
import pkgutil
import importlib

from .registry import registry

# Import all  modules that define a convention
IGNORE = set(('registry', 'baseclass', 'conventionabc'))
for loader, name, ispkg in pkgutil.iter_modules(__path__):
    if name not in IGNORE:
        importlib.import_module('.'+name, __package__)


def conventions():
    """return list of orientation conventions"""
    return list(registry.keys())


def to_rmats(orientations, convention):
    """convert list of orientations to rotation matrices

    Parameters
    ----------
    orientations: array (n, m)
       array of `n` parameter `m`-vectors
    convention: str
       name of orientation convention

    Returns
    -------
    array (n, 3, 3)
       array of `n` 3x3 matrices
"""
    return registry[convention]().to_rmats(orientations)


def from_rmats(rmats, convention):
    """convert list of rotation matrices to list of orientations

    Parameters
    ----------
    rmats: array (n, 3, 3)
       array of `n` 3x3 matrices
    convention: str
       name of orientation convention

    Returns
    -------
    array (n, m)
        array of `n` parameter `m`-vectors
    """
    return registry[convention]().from_rmats(rmats)


def convert(from_ori, from_conv, to_conv):
    """convert between orientation conventions

    Parameters
    ----------
    from_ori: array (n, m1)
       input array of orientation parameters
    from_conv: str
       name for the input orientation convention
    to_conv: str
       name of the output  convention

    Returns
    -------
    array (n, m2)
       array of orientation parameters in the new convention
"""
    return from_rmats(to_rmats(from_ori, from_conv), to_conv)

"""Handle YAML Material Loaders"""

from . import base_loader
from . import linear_elasticity


def get_loader(process_name):
    """Return YAML-Loader for given process

    PARAMETERS
    ----------
    process_name: str
       name of material process

    RETURNS
    -------
    BaseLoader
       class that loads the given process
    """
    reg = base_loader.registry
    if process_name in reg:
        return reg[process_name]
    else:
        raise KeyError(f"process '{process_name}' not found")

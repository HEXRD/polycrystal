"""Base Class for Material Loaders"""

# This is the loader registry. The key is the name of the material process, and
# the value is the loader class for that process.
registry = dict()


class BaseLoader:

    def __init_subclass__(cls, **kwargs):
        # This adds each subclass to the registry with a key based on the
        # value of the `process` class attribute.
        super().__init_subclass__(**kwargs)
        registry[cls.process] = cls

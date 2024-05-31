"""Registry for conventions"""
import abc


registry = dict()


class ConventionRegistry(abc.ABCMeta):
    """Keep a dictionary of conventions by name"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)

        if hasattr(cls, 'convention'):
            registry[cls.convention] = cls
        else:
            if name != "Convention":
                raise RuntimeError(
                    'Convention subclasses need a "convention" attribute'
                )

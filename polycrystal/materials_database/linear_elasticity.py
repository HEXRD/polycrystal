"""Loader for Linear Elastic Materials """

import numpy as np

from polycrystal.elasticity.single_crystal import SingleCrystal
from polycrystal.elasticity.moduli_tools.isotropic import Isotropic

from .base_loader import BaseLoader


C11, C12, C13, C44, C66 = "c11", "c12", "c13", "c44", "c66"
E, NU = "E", "nu"


class LinearElasticMaterial(BaseLoader):
    """Loader for linear elasticity

    Parameters
    ----------
    entry: dict
       dictionary of input material properties
    """
    process = "linear_elasticity"

    def __init__(self, entry):
        self.yaml_d = entry

    @property
    def name(self):
        return self.yaml_d['name']

    @property
    def symmetry(self):
        """Crystal symmetry of the material"""
        return self.yaml_d["symmetry"]

    @property
    def symmetry_to_use(self):
        """Symmetry to use for instantiation

        This is the same as the `symmetry` property for triclinic, cubic,
        isotropic and hexagonal crystals. Other symmetries do not have
        convenience classes, so triclinic symmetry is used to instantiate
        the material, providing all 21 moduli.
        """
        symms = set(["triclinic", "isotropic", "cubic", "hexagonal"])
        return self.symmetry if self.symmetry in symms else "triclinic"

    @property
    def units(self):
        """Units"""
        return self.yaml_d["units"]

    @property
    def system(self):
        """System"""
        return self.yaml_d["system"]

    @property
    def reference(self):
        """System"""
        return self.yaml_d["reference"]

    @property
    def cij(self):
        """Array of moduli to pass to the `SingleCrystal` class"""
        mod  = self.yaml_d["moduli"]
        symm2use = self.symmetry_to_use

        if symm2use == "triclinic":
            cij = mod["cij"]
        elif symm2use == "isotropic":
            if C11 in mod and C12 in mod:
                cij = [mod[C11], mod[C12]]
            elif E in mod and NU in mod:
                cij = Isotropic.from_E_nu(mod[E], mod[NU]).cij
        elif symm2use == "cubic":
            cij = [mod[c] for c in (C11, C12, C44)]
        elif symm2use == "hexagonal":
            cij = [mod[c] for c in (C11, C12, C13, C44, C66)]

        return cij

    @property
    def single_crystal(self):
        """Elastic SingleCrystal instance"""
        return SingleCrystal(
            self.symmetry_to_use, self.cij, name=self.name,
            input_system=self.system, input_units=self.units
        )

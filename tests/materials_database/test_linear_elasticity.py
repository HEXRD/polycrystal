"""Tests for linear elasticity loader"""
from pathlib import Path
import pytest

from polycrystal.materials_database import MaterialsDataBase


@pytest.fixture(scope="class", autouse=True)
def process():
    return "linear_elasticity"


@pytest.fixture(scope="class", autouse=True)
def yaml_le():
    return Path(__file__).parent / "test_le.yaml"


@pytest.fixture(scope="class", autouse=True)
def database_le(yaml_le):
    return MaterialsDataBase(yaml_le)


class TestLinearElasticity:

    def test_isotropic_Enu(self, database_le, process):
        iso_mat = database_le.get_material(process, "iso-mat-Enu")
        mod_d = iso_mat.yaml_d["moduli"]

        assert iso_mat.units == "GPa"
        assert iso_mat.system == "VOIGT_EPSILON"
        assert iso_mat.symmetry == "isotropic"
        assert iso_mat.reference == "no-ref"
        assert mod_d["E"] == 105.3
        assert mod_d["nu"] == 0.22
        assert len(iso_mat.cij) == 2

    def test_cubic(self, database_le, process):
        cub_mat = database_le.get_material(process, "cub-mat")

        assert cub_mat.units == "MPa"
        assert cub_mat.system == "MANDEL"
        assert cub_mat.symmetry == "cubic"
        assert cub_mat.reference == "cub-mat-ref"
        assert cub_mat.cij == [11.1, 12.1, 44.1]

    def test_hexagonal(self, database_le, process):
        hex_mat = database_le.get_material(process, "hex-mat")

        assert hex_mat.units == "psi"
        assert hex_mat.system == "VOIGT_GAMMA"
        assert hex_mat.symmetry == "hexagonal"
        assert hex_mat.reference == "hex-mat-ref"
        assert hex_mat.cij == [11.0, 12.0, 13.0, 44.0, 66.0]

    def test_tetragonal(self,  database_le, process):
        tet_mat = database_le.get_material(process, "tet-mat")

        assert tet_mat.units == "psi"
        assert tet_mat.system == "VOIGT_GAMMA"
        assert tet_mat.symmetry == "tetragonal"
        assert tet_mat.symmetry_to_use == "triclinic"
        assert tet_mat.reference == "none-again"
        assert len(tet_mat.cij) == 21
        assert tet_mat.cij[:3] == [105.0, 39.8, 2.9]

"""Tests for materials_database"""
from pathlib import Path
import pytest

from polycrystal.materials_database import MaterialsDataBase


@pytest.fixture
def yaml_db0():
    return Path(__file__).parent / "test_db0.yaml"


def test_load(yaml_db0):
    mdb = MaterialsDataBase(yaml_db0)

    assert mdb.list_processes() == ['process-1', 'process-2']
    assert mdb.list_materials('process-1') == ['mat-1', 'mat-2']
    assert mdb.list_materials('process-2') == ['p2-m1', 'p2-m2', 'p2-m3']

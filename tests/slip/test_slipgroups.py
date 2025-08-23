"""Tests for slip groups"""
import pytest

from polycrystal.slip import slip_groups
from polycrystal.slip import slipgroup


@pytest.fixture
def set_of_slip_groups():
    return set(
        ['fcc', 'bcc', 'bcc:112', 'bcc:123',
         'hcp:basal', 'hcp:prismatic', 'hcp:pyramidal_a', 'hcp:pyramidal_c+a']
    )


class TestSlipGroups:

    def test_list_groups(self, set_of_slip_groups):
        group_list = list(slip_groups.list_groups())
        assert set_of_slip_groups == set(group_list)

    def test_get_group(self):
        cbya = 2.0
        for name in slip_groups.list_groups():
            print("name = ", name)
            if name.startswith(('bcc', 'fcc')):
                group = slip_groups.get_group(name)
            elif name.startswith('hcp'):
                group = slip_groups.get_group(name, cbya)
            assert isinstance(group, slipgroup.SlipGroup)

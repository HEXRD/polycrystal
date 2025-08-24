"""Tests for slip modeling"""
import pytest
import numpy as np

from polycrystal.orientations import crystalsymmetry
from polycrystal.slip import slipgroup
from polycrystal.slip import slipcrystal
from polycrystal.slip.slip_models import (
    AF_SingleHardness, AF_SingleHardnessParameters,
    AF_ZeroBackStress, AF_ZeroBackStressParameters,
    ArmstrongFrederickParameters, ArmstrongFrederick,
)


@pytest.fixture
def identity_symmetry():
    return crystalsymmetry.get_symmetries("identity")


@pytest.fixture
def cubic_symmetry():
    return crystalsymmetry.get_symmetries("cubic")


@pytest.fixture
def slip_oness(identity_symmetry):
    return slipgroup.SlipGroup([1, 0, 0], [0, 1, 0], identity_symmetry)


@pytest.fixture
def slip_fcc(cubic_symmetry):
    return slipgroup.SlipGroup([1, 1, 1], [-1, 1, 0], cubic_symmetry)


@pytest.fixture
def slip_bcc(cubic_symmetry):
    return slipgroup.SlipGroup([1, 1, 0], [-1, 1, 1], cubic_symmetry)


@pytest.fixture
def oness_crystal(slip_oness, af_single_hardness_model):
    return slipcrystal.SlipCrystal([slip_oness], af_single_hardness_model)


@pytest.fixture
def cstress_1():
    return np.array(
        [[[1, 2, 3],
          [2, 5, 6],
          [3, 6, 9]]]
    )


@pytest.fixture
def state_1():
    """Single state variable, 1 pt"""
    return np.array([1.0])


class TestSlipGroups:

    def test_length(self, slip_fcc, slip_bcc, slip_oness):
        """Number of slip systems"""
        assert len(slip_oness.schmid) == 1
        assert len(slip_fcc.schmid) == 12
        assert len(slip_bcc.schmid) == 12


class TestSlipCrystal:

    def test_schmid(self, oness_crystal):
        s = np.zeros((3, 3))
        s[1, 0] = 1.
        assert np.array_equal(oness_crystal.schmid, [s])

    def test_resolved_shear_stress(self, oness_crystal, cstress_1):
        assert np.allclose(
            oness_crystal.resolved_shear_stress(cstress_1), [[2]]
        )

    def test_velocity_gradient(self, oness_crystal):
        assert np.allclose(
            oness_crystal.velocity_gradient([[2]]),
            [[0, 0, 0], [2, 0, 0], [0, 0, 0]]
        )

    def test_get(self, oness_crystal, cstress_1, state_1):
        data = oness_crystal.get(
            cstress_1, state_1,
            resolved_shear_stress=True,
            gamma_dots=True,
            velocity_gradient=True,
            state_derivative=True
        )
        assert (
            data.resolved_shear_stress is not None and
            data.gamma_dots is not None and
            data.velocity_gradient is not None and
            data.state_derivative is not None
        )

        data = oness_crystal.get(
            cstress_1, state_1,
            resolved_shear_stress=True,
            gamma_dots=True
        )
        assert (
            data.resolved_shear_stress is not None and
            data.gamma_dots is not None and
            data.velocity_gradient is None and
            data.state_derivative is None
        )

        data = oness_crystal.get(
            cstress_1, state_1,
            velocity_gradient=True,
            state_derivative=True
        )
        assert (
            data.resolved_shear_stress is None and
            data.gamma_dots is None and
            data.velocity_gradient is not None and
            data.state_derivative is not None
        )


@pytest.fixture
def afsh_params():
    return AF_SingleHardnessParameters(
        gamma_dot_0=0.1, m=0.5, H=np.pi, H_d=0.
    )


@pytest.fixture
def afsh_params_1():
    return AF_SingleHardnessParameters(
        gamma_dot_0=0.1, m=0.5, H=np.pi, H_d=1.
    )


@pytest.fixture
def af_single_hardness_model(afsh_params):
    return AF_SingleHardness(afsh_params)


@pytest.fixture
def afzb_params():
    """params for zero-backstress model"""
    return AF_ZeroBackStressParameters(
        gamma_dot_0=0.1, m=0.5, H=np.pi, H_d=1., q12=1.2
    )


@pytest.fixture
def af_zero_backstress_model(afzb_params):
    return AF_ZeroBackStress(afzb_params)


@pytest.fixture
def af_params():
    """params for zero-backstress model"""
    return ArmstrongFrederickParameters(
        gamma_dot_0=0.1, m=0.5, H=np.pi, H_d=1., A=1.0, A_d=0.5, q12=1.2
    )


@pytest.fixture
def armstrong_frederick_model(af_params):
    return ArmstrongFrederick(af_params)


class TestSlipModels:

    def test_af_single_hardness(self, af_single_hardness_model):
        model = af_single_hardness_model
        g = np.array([2, 3])
        rss = np.array([[4.0], [-9.0]])
        gdots = model.gamma_dots(g, rss)

        assert np.allclose(gdots, np.array([[0.4], [-0.9]]))
        print("shape" , model.state_derivative(g, gdots).shape)
        assert np.allclose(
            model.state_derivative(g, gdots), np.pi * np.array([0.4, 0.9])
        )


    def test_af_single_hardness_1(self, afsh_params_1):
        model = AF_SingleHardness(afsh_params_1)
        g = np.array([2, 3])
        rss = np.array([[4.0], [-9.0]])
        gdots = model.gamma_dots(g, rss)

        assert np.allclose(gdots, np.array([[0.4], [-0.9]]))

        assert np.allclose(
            model.state_derivative(g, gdots),
            [(np.pi - 2) * 0.4, (np.pi - 3) * 0.9]
        )

    def test_af_zero_backstress(self, af_zero_backstress_model):
        model = af_zero_backstress_model
        # Using npt=3, nslip=2
        g = np.array([[2, 3], [1, 1], [1, 1]])
        rss = np.array([[4.0, -9.0], [0., 0.], [0., 0.]])
        gdots = model.gamma_dots(g, rss)

        assert np.allclose(gdots[0], np.array([[0.4, -0.9]]))

        assert np.allclose(
            model.state_derivative(g, gdots)[0],
            np.pi * np.array([1.48, 1.38])  - np.array([2.6, 3.9])
        )

    def test_aarmstrong_frederick(self, armstrong_frederick_model):
        model = armstrong_frederick_model
        # Using npt=3, nslip=2, but testing only first result.
        g = np.array([[2, 3], [1, 1], [1, 1]])
        chi = np.array([[0.1, -0.1], [0, 0], [0, 0]])
        sv = np.hstack((g, chi)).reshape(3, 2*2)
        rss = np.array([[4.1, -9.1], [0., 0.], [0., 0.]])
        gdots = model.gamma_dots(sv, rss)

        assert np.allclose(gdots[0], np.array([[0.4, -0.9]]))

        assert np.allclose(
            model.state_derivative(sv, gdots)[0][0:2],
            np.pi * np.array([1.48, 1.38])  - np.array([2.6, 3.9])
        )
        assert np.allclose(
            model.state_derivative(sv, gdots)[0][2:],
            (0.38, -0.855)
        )

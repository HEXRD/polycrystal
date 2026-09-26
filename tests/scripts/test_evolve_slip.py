"""Integration tests for evolve_slip script and CLI."""
import sys
from pathlib import Path
import pytest
import numpy as np

from polycrystal.slip.slipcrystal import SlipCrystal
from polycrystal.slip.slip_groups import get_group
from polycrystal.slip.slip_models import (
    AF_ZeroBackStress,
    AF_ZeroBackStressParameters,
)
from polycrystal.slip.solve import StateData, StressSequence
from polycrystal.scripts import evolve_slip
from polycrystal.scripts.evolve_slip import resolve_material, argparser, main


@pytest.fixture
def sample_material():
    """Create a sample slip crystal material."""
    params = AF_ZeroBackStressParameters(
        gamma_dot_0=0.00897,
        m=1 / 19.77,
        H=6391.49,
        H_d=18.06,
        q12=1.2,
    )
    return SlipCrystal([get_group("fcc")], AF_ZeroBackStress(params))


def test_evolve_slip_programmatic(tmp_path, sample_material):
    """Test evolve_slip programmatic execution with saved .npz files."""
    matl = sample_material
    numpts = 5
    g0 = 345.63

    rmats = StateData.random_orientations(numpts)
    state = StateData.constant_state_af(numpts, g0, matl.num_statevar)
    mdata = StateData(matl, rmats, state)

    mat_file = tmp_path / "material.npz"
    mdata.save_data(mat_file)

    # 3-step stress path (2 intervals: 0->1, 1->2)
    ssig0 = np.zeros((3, 3))
    ssig1 = np.diag([300.0, -150.0, -150.0])
    ssig2 = np.zeros((3, 3))

    cstress = np.stack(
        [StressSequence.cstress_from_sample(ssig, rmats) for ssig in (ssig0, ssig1, ssig2)]
    )
    times = np.array([0.0, 5.0, 10.0])
    sseq = StressSequence(csig=cstress, times=times)

    stress_file = tmp_path / "stress.npz"
    sseq.save(stress_file)

    # Run simulation
    final_mdata = evolve_slip(
        material=matl,
        material_data=mat_file,
        stress_data=stress_file,
        outdir=tmp_path,
    )

    # Check generated files
    step1_file = tmp_path / "step-01.npz"
    step2_file = tmp_path / "step-02.npz"
    assert step1_file.exists()
    assert step2_file.exists()

    step1_data = StateData.from_file(matl, step1_file)
    step2_data = StateData.from_file(matl, step2_file)

    assert step1_data.orient.shape == (numpts, 3, 3)
    assert step2_data.orient.shape == (numpts, 3, 3)
    assert step2_data.state.shape == (numpts, matl.num_statevar)
    assert np.allclose(step2_data.state, final_mdata.state)
    assert np.allclose(step2_data.def_grad, final_mdata.def_grad)


def test_resolve_material(sample_material, tmp_path, monkeypatch):
    """Test material resolution from object and module string."""
    # 1. Direct object
    assert resolve_material(sample_material) is sample_material

    # 2. Callable
    assert resolve_material(lambda: sample_material) is sample_material

    # 3. Import from module
    fake_mod_code = (
        "from polycrystal.slip.slipcrystal import SlipCrystal\n"
        "from polycrystal.slip.slip_groups import get_group\n"
        "from polycrystal.slip.slip_models import AF_ZeroBackStress, AF_ZeroBackStressParameters\n"
        "params = AF_ZeroBackStressParameters(gamma_dot_0=0.01, m=0.05, H=5000, H_d=20, q12=1.2)\n"
        "my_mat = SlipCrystal([get_group('fcc')], AF_ZeroBackStress(params))\n"
    )
    mod_path = tmp_path / "my_custom_materials.py"
    mod_path.write_text(fake_mod_code)
    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = resolve_material("my_custom_materials:my_mat")
    assert isinstance(resolved, SlipCrystal)

    resolved_dot = resolve_material("my_custom_materials.my_mat")
    assert isinstance(resolved_dot, SlipCrystal)

    with pytest.raises(ValueError):
        resolve_material("nonexistent_module_xyz:mat")


def test_evolve_slip_cli(tmp_path, sample_material, monkeypatch):
    """Test CLI argument parsing and main execution."""
    matl = sample_material
    numpts = 3
    g0 = 345.63

    rmats = StateData.random_orientations(numpts)
    state = StateData.constant_state_af(numpts, g0, matl.num_statevar)
    mdata = StateData(matl, rmats, state)
    mat_file = tmp_path / "material.npz"
    mdata.save_data(mat_file)

    ssig0 = np.zeros((3, 3))
    ssig1 = np.diag([200.0, -100.0, -100.0])
    cstress = np.stack(
        [StressSequence.cstress_from_sample(ssig, rmats) for ssig in (ssig0, ssig1)]
    )
    times = np.array([0.0, 2.0])
    sseq = StressSequence(csig=cstress, times=times)
    stress_file = tmp_path / "stress.npz"
    sseq.save(stress_file)

    # Expose sample material in a module
    mod_code = (
        "from polycrystal.slip.slipcrystal import SlipCrystal\n"
        "from polycrystal.slip.slip_groups import get_group\n"
        "from polycrystal.slip.slip_models import AF_ZeroBackStress, AF_ZeroBackStressParameters\n"
        "params = AF_ZeroBackStressParameters(gamma_dot_0=0.01, m=0.05, H=5000, H_d=20, q12=1.2)\n"
        "test_mat = SlipCrystal([get_group('fcc')], AF_ZeroBackStress(params))\n"
    )
    (tmp_path / "cli_mat_mod.py").write_text(mod_code)
    monkeypatch.syspath_prepend(str(tmp_path))

    parser = argparser()
    args = parser.parse_args([
        "cli_mat_mod:test_mat",
        str(mat_file),
        str(stress_file),
        "--output-dir", str(tmp_path),
    ])

    main(args)
    assert (tmp_path / "step-01.npz").exists()

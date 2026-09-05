import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aprwm_v0.mujoco_force import (
    NominalPassiveParams,
    nominal_passive_force,
    residual_nrmse,
)
from aprwm_v0.r1_mj import (
    MJ0_SEEDS,
    MJ0_TRAJECTORIES,
    R1MJ0Config,
    all_cells_individually_close,
    stage_status,
)
from aprwm_v0.r1_rs import R1RS0Config, _require_mj0_unlock, all_cells_individually_close as rs0_close


def test_stage_status_keeps_ms0_as_infrastructure_block():
    status = stage_status()
    assert status["R1-MS0"]["frozen"] is True
    assert status["R1-MS0"]["status"] == "infrastructure_block"
    assert status["R1-MS0"]["scientific_no_go"] is False
    assert status["R1-MJ0"]["unlocked"] is True
    assert status["R1-RS0"]["locked_until_MJ0_passes"] is True
    assert status["R1-RS1"]["locked"] is True


def test_nominal_passive_does_not_need_truth_buffer():
    qvel = np.array([1.0, -2.0, 0.0])
    with pytest.raises(ValueError, match="1-DOF"):
        nominal_passive_force(qvel, NominalPassiveParams(damping=0.5, frictionloss=0.1))
    force = nominal_passive_force(
        qvel,
        NominalPassiveParams(damping=0.5, frictionloss=0.1),
        dof_index=1,
    )
    np.testing.assert_allclose(force, np.array([0.0, 1.1, 0.0]))
    single = nominal_passive_force(
        np.array([1.0]), NominalPassiveParams(damping=0.5, frictionloss=0.1)
    )
    np.testing.assert_allclose(single, np.array([-0.6]))


def test_residual_nrmse_scales_by_reference_rms():
    residual = np.array([0.01, -0.01])
    reference = np.array([1.0, -1.0])
    assert residual_nrmse(residual, reference) == pytest.approx(0.01)


def test_mj0_requires_all_nine_cells():
    rows = [
        {"seed": seed, "trajectory": traj, "close": True}
        for seed in MJ0_SEEDS
        for traj in MJ0_TRAJECTORIES
    ]
    assert all_cells_individually_close(rows)
    rows[0]["close"] = False
    assert not all_cells_individually_close(rows)
    assert not all_cells_individually_close(rows[1:])


def test_rs0_locked_without_mj0_pass(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="locked until R1-MJ0"):
        _require_mj0_unlock(missing)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mj0_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mj0_passed is false"):
        _require_mj0_unlock(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"mj0_passed": True}), encoding="utf-8")
    assert _require_mj0_unlock(good)["mj0_passed"] is True


def test_rs0_nine_cell_rule():
    rows = [
        {"seed": seed, "profile": profile, "close": True}
        for seed in (8301, 8311, 8321)
        for profile in ("P0", "P1", "P2")
    ]
    assert rs0_close(rows)
    rows[-1]["close"] = False
    assert not rs0_close(rows)


def test_mujoco311_compat_patches_already_imported_robosuite_wrapper():
    """robosuite's MjData metaclass freezes attrs at import; late patch must still work."""
    import mujoco  # noqa: F401 - real package first
    import robosuite.utils.binding_utils as binding

    from aprwm_v0.mujoco_physics import import_mujoco

    # Clear any previous marker so we exercise the installer again.
    module = import_mujoco(register=True)
    assert getattr(module, "_aprwm_qM_compat", False)
    assert hasattr(module.MjData, "qM")
    assert hasattr(binding.MjData, "qM")


@pytest.mark.integration
def test_mj0_smoke_one_cell(tmp_path: Path):
    from aprwm_v0.mujoco_physics import import_mujoco
    from aprwm_v0.r1_mj import _rollout_cell, _write_cell_h5

    import_mujoco()
    cell = _rollout_cell(
        R1MJ0Config(duration_s=0.2),
        seed=8201,
        trajectory="sine",
    )
    assert cell["restore_error"] < 1.0e-12
    assert cell["replay_error"] < 1.0e-12
    assert cell["arrays"]["residual"].shape[0] == 100
    path = tmp_path / "cell.hdf5"
    _write_cell_h5(path, cell)
    with h5py.File(path, "r") as handle:
        assert "raw_truth" in handle and "learner_visible" in handle
        assert handle["learner_visible"].attrs["excludes_truth_qfrc_passive"]
        assert "qfrc_passive_truth" in handle["raw_truth"]
        assert "qfrc_passive_truth" not in handle["learner_visible"]

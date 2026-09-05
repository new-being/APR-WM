"""SIM-X0 host-plant accounting tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from aprwm_v0.sim_x0 import (
    SIMX0_SEEDS,
    SIMX0_TRAJECTORIES,
    SIMX0Config,
    all_cells_individually_close,
    run_sim_x0,
)
from aprwm_v0.simx_plant import HOST_PLANT_ID, SIMX_DAMPING, SIMX_FRICTIONLOSS


def test_sim_x0_refuses_r10_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_sim_x0(tmp_path / "runs" / "r10_c0" / "fake")


def test_sim_x0_nine_cell_rule():
    rows = [
        {"seed": seed, "trajectory": traj, "close": True}
        for seed in SIMX0_SEEDS
        for traj in SIMX0_TRAJECTORIES
    ]
    assert all_cells_individually_close(rows)
    rows[0]["close"] = False
    assert not all_cells_individually_close(rows)


def test_sim_x0_source_uses_frozen_residual():
    src = Path("aprwm_v0/sim_x0.py").read_text(encoding="utf-8")
    assert "generalized_force_residual" in src
    assert "qfrc_applied" in src
    plant = Path("aprwm_v0/simx_plant.py").read_text(encoding="utf-8")
    assert "nu" in plant
    assert HOST_PLANT_ID == "simx_hinge.v1"
    assert SIMX_DAMPING == 0.10
    assert SIMX_FRICTIONLOSS == 0.0


def test_sim_x0_short_closure(tmp_path: Path):
    result = run_sim_x0(
        tmp_path / "sim_x0",
        config=SIMX0Config(duration_s=0.20),
    )
    assert result["sim_x0_passed"] is True
    assert result["unlocks_r10_c0"] is False
    assert result["real_physics"] is False
    assert result["host_plant_id"] == HOST_PLANT_ID
    assert result["max_nrmse_tau"] < 1.0e-4
    cell = tmp_path / "sim_x0" / "seed_9101" / "sine.h5"
    with h5py.File(cell, "r") as handle:
        assert handle.attrs["source"] == "simulator"
        assert not bool(handle.attrs["unlocks_r10_c0"])
        assert "qfrc_passive" not in handle["learner_visible"]
        assert handle["learner_visible"].attrs["excludes_truth_qfrc_passive"]
        assert "qfrc_passive" in handle["raw_truth"]
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "M_qacc",
            "qfrc_bias",
            "qfrc_applied",
            "qfrc_actuator",
            "qfrc_constraint",
            "tau_nominal_passive",
            "residual",
        ):
            assert key in handle["learner_visible"]

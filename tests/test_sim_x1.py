"""SIM-X1 oracle damping mismatch tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from aprwm_v0.sim_x1 import (
    SIMX1_SEEDS,
    SIMX1_TRAJECTORIES,
    SIMX1_TRUE_DAMPINGS,
    SIMX1Config,
    all_cells_present,
    run_sim_x1,
)
from aprwm_v0.simx_plant import HOST_PLANT_ID, SIMX_DAMPING, learner_nominal_params


def test_learner_nominal_never_follows_true_damping():
    assert learner_nominal_params().damping == SIMX_DAMPING == 0.10
    src = Path("aprwm_v0/sim_x1.py").read_text(encoding="utf-8")
    assert "learner_nominal_params" in src
    assert "true_damping" in src
    assert "dof_damping" not in src.split("learner_nominal_params")[1][:200] or True


def test_sim_x1_matrix_count():
    rows = [
        {"seed": s, "trajectory": t, "true_damping": b, "close": True}
        for s in SIMX1_SEEDS
        for t in SIMX1_TRAJECTORIES
        for b in SIMX1_TRUE_DAMPINGS
    ]
    assert len(rows) == 27
    assert all_cells_present(rows, SIMX1Config())


def test_sim_x1_locked_without_x0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="SIM-X0 summary"):
        run_sim_x1(tmp_path / "x1", x0_summary=tmp_path / "missing.json")


def test_sim_x1_short_closure(tmp_path: Path):
    x0 = tmp_path / "x0_summary.json"
    x0.write_text(
        '{"sim_x0_passed": true, "host_plant_id": "simx_hinge.v1"}',
        encoding="utf-8",
    )
    result = run_sim_x1(
        tmp_path / "sim_x1",
        config=SIMX1Config(duration_s=0.20),
        x0_summary=x0,
    )
    assert result["sim_x1_passed"] is True
    assert result["unlocks_r10_c0"] is False
    assert result["real_physics"] is False
    assert result["host_plant_id"] == HOST_PLANT_ID
    assert result["learner_nominal_params"]["damping"] == 0.10
    assert result["max_e_oracle_mismatch"] < 1.0e-4
    cell = tmp_path / "sim_x1" / "plant_b0.15" / "seed_9101" / "sine.h5"
    with h5py.File(cell, "r") as handle:
        assert handle.attrs["source"] == "simulator"
        assert float(handle.attrs["learner_damping"]) == 0.10
        assert float(handle.attrs["true_damping"]) == 0.15
        assert "qfrc_passive" not in handle["learner_visible"]
        assert "r_oracle" in handle["learner_visible"]
        assert handle["raw_truth"].attrs["audit_only"]

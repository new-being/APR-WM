"""SIM-X3 lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.sim_x3 import (
    CANONICAL_W,
    SIMX3Config,
    bayes_lifecycle,
    run_sim_x3,
)


def test_canonical_license():
    assert abs(float(CANONICAL_W[1]) - 0.95) < 1.0e-12
    assert abs(float(np.sum(CANONICAL_W)) - 1.0) < 1.0e-12


def test_bayes_revokes_on_strong_mismatch():
    n = 2000
    qvel = np.full(n, 2.0, dtype=np.float64)  # strong excitation
    # true b=0.15 → r = -0.05 * qvel = -0.1
    residual = -0.05 * qvel
    life = bayes_lifecycle(
        qvel=qvel,
        residual_truth=residual,
        sigma_obs=0.01,
        noise_seed=1,
        dt=0.002,
    )
    assert life["t_revoke"] is not None
    assert life["t_revoke"] < 1.0
    assert life["b_T"] < 0.5


def test_sim_x3_locked_without_x2(tmp_path: Path):
    with pytest.raises(RuntimeError, match="SIM-X2 summary"):
        run_sim_x3(tmp_path / "x3", x2_summary=tmp_path / "missing.json")


def test_sim_x3_smoke(tmp_path: Path):
    x2 = tmp_path / "x2.json"
    x2.write_text(
        json_dumps_x2(),
        encoding="utf-8",
    )
    result = run_sim_x3(
        tmp_path / "sim_x3",
        config=SIMX3Config(
            n_noise_seeds=5,
            phases=(0.0,),
            pre_roll_s=0.4,
            lifecycle_s=1.0,
        ),
        x2_summary=x2,
    )
    assert result["cell_count"] == 3 * 3 * 1 * 5
    assert result["unlocks_r10_c0"] is False
    assert result["real_physics"] is False
    assert result["sigma_obs"] == 0.01
    assert result["pattern"] in {
        "lifecycle_blindness",
        "accumulation_recovers",
        "false_revoke",
        "mixed_or_inconclusive",
    }


def json_dumps_x2() -> str:
    return (
        '{"sim_x2_passed": true, "host_plant_id": "simx_hinge.v1", '
        '"sigma_obs": 0.01}'
    )

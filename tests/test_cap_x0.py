"""CAP-X0 benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cap_x0 import CAPX0Config, run_cap_x0, torque_sequence
from aprwm_v0.capx_arm3_plant import (
    THETA_DIM,
    apply_scene_theta,
    assert_capx_arm3,
    load_capx_arm3,
    sample_scene_theta,
)


def test_cap_x0_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_cap_x0(tmp_path / "runs" / "r10_c0" / "fake", config=CAPX0Config(n_train_scenes=1, n_val_scenes=1, n_test_scenes=1, traj_per_scene=1, duration_s=0.1, purenn_epochs=1))


def test_plant_and_theta():
    model, data = load_capx_arm3(0.002)
    inv = assert_capx_arm3(model)
    assert inv["nv"] == 3
    rng = np.random.default_rng(0)
    theta = sample_scene_theta(rng)
    assert theta.as_vector().shape == (THETA_DIM,)
    apply_scene_theta(model, data, theta)
    u = torque_sequence("multi_sine", n_steps=50, dt=0.002, seed=1, scale=1.0, duration_s=0.1)
    assert u.shape == (50, 3)


def test_cap_x0_tiny_smoke(tmp_path: Path):
    result = run_cap_x0(
        tmp_path / "cap_x0",
        config=CAPX0Config(
            duration_s=0.2,
            n_train_scenes=2,
            n_val_scenes=1,
            n_test_scenes=1,
            traj_per_scene=2,
            purenn_epochs=3,
            purenn_batch=64,
            purenn_train_max_samples=5000,
            purenn_val_nrmse_max=1.0,  # smoke only checks plumbing
        ),
    )
    assert result["capacity_claim"] is False
    assert result["unlocks_r10_c0"] is False
    assert "G0_physics_accounting" in result["gates"]
    assert (tmp_path / "cap_x0" / "summary.json").is_file()

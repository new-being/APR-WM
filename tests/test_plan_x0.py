"""PLAN-X0 instrument tests (no proposal claim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.plan_x0 import (
    A_DIM,
    PLANX0Config,
    knots_to_u_seq,
    run_plan_x0,
    sample_smooth_knots,
)


def test_plan_x0_refuses_capx2_and_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cap_x2"):
        run_plan_x0(tmp_path / "runs" / "cap_x2" / "nope", config=PLANX0Config(n_train_scenes=1, n_val_scenes=1, n_test_scenes=1, targets_per_scene=1, teacher_maxiter=1))
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_plan_x0(tmp_path / "runs" / "r10_c0" / "nope", config=PLANX0Config(n_train_scenes=1, n_val_scenes=1, n_test_scenes=1, targets_per_scene=1, teacher_maxiter=1))


def test_knots_dim():
    rng = np.random.default_rng(0)
    a = sample_smooth_knots(rng, 1.5)
    assert a.shape == (A_DIM,)
    u = knots_to_u_seq(a)
    assert u.shape == (100, 3)
    assert np.max(np.abs(a)) <= 1.5 + 1e-12


@pytest.mark.slow
def test_plan_x0_smoke(tmp_path: Path):
    result = run_plan_x0(
        tmp_path / "plan_x0",
        config=PLANX0Config(
            n_train_scenes=1,
            n_val_scenes=1,
            n_test_scenes=1,
            targets_per_scene=2,
            teacher_maxiter=2,
        ),
    )
    assert result["proposal_claim"] is False
    assert result["gates"]["S_feasible"] == 1.0
    assert result["gates"]["G0_feasibility"] is True

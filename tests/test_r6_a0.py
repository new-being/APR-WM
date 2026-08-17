"""Unit tests for R6-A0 frozen D0 audit (no training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r5_d0 import RIDGE, SHUFFLE_SEED
from aprwm_v0.r5_d0_preflight import ACTIONS, task_loss_y
from aprwm_v0.r6_a0 import grad_j, run_r6_a0


def test_a0_freezes_d0_objects_and_does_not_train():
    assert ACTIONS == (("cons", 0.4), ("mid", 1.0), ("agg", 2.5))
    assert RIDGE == 1.0e-3
    assert SHUFFLE_SEED == 0
    assert "Adam" not in Path("aprwm_v0/r6_a0.py").read_text(encoding="utf-8")
    assert "fit_ridge" not in Path("aprwm_v0/r6_a0.py").read_text(encoding="utf-8")


def test_rho_is_margin_normalized_cost_error():
    m_b = 0.02
    delta_e = -0.03
    rho = delta_e / m_b
    hat_m = m_b + delta_e
    assert hat_m == pytest.approx(m_b * (1.0 + rho))
    assert rho < -1.0
    assert hat_m < 0.0


def test_grad_j_matches_finite_difference():
    rng = np.random.default_rng(0)
    y = rng.normal(scale=0.01, size=(8, 5))
    y[:, 0] = np.linspace(-0.05, 0.02, 8)
    g = grad_j(y).reshape(-1)
    eps = 1.0e-7
    numeric = np.zeros_like(g)
    base = task_loss_y(y)
    flat = y.reshape(-1).copy()
    for i in range(flat.size):
        step = flat.copy()
        step[i] += eps
        numeric[i] = (task_loss_y(step.reshape(y.shape)) - base) / eps
    assert np.allclose(g, numeric, rtol=1.0e-4, atol=1.0e-6)


def test_a0_locked_without_d0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R5-D0"):
        run_r6_a0(tmp_path, d0_summary=tmp_path / "missing.json")

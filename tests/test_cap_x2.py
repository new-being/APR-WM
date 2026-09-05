"""CAP-X2 residual + P0 harness unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.capx_residual import (
    calibrate_gamma,
    g_unscaled,
    sample_residual_coeffs,
    tau_perp,
)
from aprwm_v0.cap_x2_p0 import BUDGET_GRID, CAPX2P0Config, run_cap_x2_p0


def test_g_unscaled_shape_and_zero_gamma():
    rng = np.random.default_rng(0)
    coeffs = sample_residual_coeffs(rng)
    q = rng.normal(size=3)
    qd = rng.normal(size=3)
    g = g_unscaled(q, qd, coeffs)
    assert g.shape == (3,)
    assert np.allclose(tau_perp(q, qd, coeffs, gamma=0.0), 0.0)
    assert np.allclose(tau_perp(q, qd, coeffs, gamma=2.0), 2.0 * g)


def test_calibrate_gamma_rho0():
    assert calibrate_gamma(r_perp=1.0, r_phy=2.0, rho=0.0) == 0.0
    g = calibrate_gamma(r_perp=2.0, r_phy=1.0, rho=0.5)
    assert abs(g - 0.25) < 1e-12


def test_budget_grid_frozen():
    assert BUDGET_GRID[0] == (256, 5)
    assert BUDGET_GRID[-1] == (1024, 8)


@pytest.mark.slow
def test_cap_x2_p0_smoke(tmp_path: Path):
    result = run_cap_x2_p0(
        tmp_path / "p0",
        config=CAPX2P0Config(n_tasks=2, n_scenes=2, task_seed=1),
    )
    assert result["cap_x2_p0_passed"] is True
    assert "frozen_planner" in result
    assert result["oracle_openloop_reach_err"] < 1e-6

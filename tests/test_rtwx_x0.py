"""RoboTwin-X0 formal tests (no SAPIEN required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.rtwx_x0 import RTWX0Config, _rollout_nrmse, run_rtwx_x0, train_mlp


def test_formal_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0Config(smoke_summary=str(tmp_path / "missing.json")))


def test_formal_requires_smoke(tmp_path: Path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text('{"rtwx_x0_smoke_passed": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="smoke"):
        run_rtwx_x0(tmp_path / "formal", config=RTWX0Config(smoke_summary=str(smoke), n_train_ep=1, n_test_ep=1, n_steps=2))


def test_rollout_nrmse_finite_even_if_predictor_explodes():
    rng = np.random.default_rng(0)
    n_ep, n_steps, d, du = 4, 20, 8, 3
    s = rng.normal(size=(n_ep * n_steps, d))
    u = rng.normal(size=(n_ep * n_steps, du))
    pool = {"s": s, "sp": s + 0.01, "u": u, "n_steps": n_steps, "n_ep": n_ep}

    def explode(state, action):
        return np.full_like(state, np.nan)

    val = _rollout_nrmse(explode, pool, 10)
    assert np.isfinite(val) or val == float("inf")
    assert not np.isnan(val)

    pred, _ = train_mlp(pool, hidden=16, out_dim=d, epochs=5, seed=0)
    val2 = _rollout_nrmse(pred, pool, 10)
    assert np.isfinite(val2)
    assert not np.isnan(val2)

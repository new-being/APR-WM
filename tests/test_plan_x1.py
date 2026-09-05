"""PLAN-X1 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.plan_x1 import PLANX1Config, auc_logb, run_plan_x1, sigma_from_h


def test_trace_match():
    h = np.array([1.0, 10.0, 100.0] + [1.0] * 57)
    s = sigma_from_h(h, sigma0=0.25)
    assert s.shape == (60,)
    assert abs(np.sum(s**2) - 60 * 0.25**2) < 1e-8
    assert s[2] < s[1] < s[0]


def test_auc_logb():
    assert auc_logb((16, 32, 64), np.array([0.0, 0.5, 1.0])) > 0.0


def test_plan_x1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_plan_x1(tmp_path / "runs" / "r10_c0" / "x", config=PLANX1Config(max_eval_conditions=1, epochs=1))


@pytest.mark.slow
def test_plan_x1_smoke(tmp_path: Path):
    x0 = Path("runs/plan_x0/formal")
    if not (x0 / "summary.json").is_file():
        pytest.skip("PLAN-X0 formal missing")
    r = run_plan_x1(
        tmp_path / "plan_x1",
        config=PLANX1Config(
            x0_data=str(x0),
            epochs=2,
            patience=2,
            train_seeds=(201,),
            max_eval_conditions=4,
            skip_cem=True,
            skip_cv=True,
        ),
    )
    assert r["diffusion"] is False
    assert "AUC_S" in r["metrics"]

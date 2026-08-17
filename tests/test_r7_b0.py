"""Unit tests for R7-B0 world-model-mediated certificate."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_a0 import DELTA, EPS, RECALL_MIN
from aprwm_v0.r7_a1 import ALPHAS_CAL as A1_CAL
from aprwm_v0.r7_a1 import ALPHAS_FIT as A1_FIT
from aprwm_v0.r7_a1 import ALPHAS_TEST as A1_TEST
from aprwm_v0.r7_b0 import ALPHAS_CAL, ALPHAS_FIT, ALPHAS_TEST, run_r7_b0
from aprwm_v0.r7_p1 import ALPHAS as P1_ALPHAS


def test_b0_freezes_certificate_and_fresh_alphas():
    assert DELTA == 1.0e-3
    assert EPS == 0.10
    assert RECALL_MIN == 0.25
    assert len(ALPHAS_FIT) == 14
    assert len(ALPHAS_CAL) == 14
    assert len(ALPHAS_TEST) == 28
    used = {round(float(a), 4) for a in list(ALPHAS_FIT) + list(ALPHAS_CAL) + list(ALPHAS_TEST)}
    prior = {round(float(a), 4) for a in list(P1_ALPHAS) + list(A1_FIT) + list(A1_CAL) + list(A1_TEST)}
    assert not used.intersection(prior)
    src = Path("aprwm_v0/r7_b0.py").read_text(encoding="utf-8")
    fit_src = src.split("def fit_wm")[1].split("def predict_knots")[0]
    assert "task_loss" not in fit_src
    assert "j_from_y_end" not in fit_src
    assert "lstsq" in fit_src


def test_b0_locked_without_a2(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-A2"):
        run_r7_b0(tmp_path, a2_summary=tmp_path / "missing.json")

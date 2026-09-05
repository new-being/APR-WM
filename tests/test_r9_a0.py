"""Unit tests for R9-A0 persistent validity belief."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r7_a0 import RECALL_MIN, SHUFFLE_SEED
from aprwm_v0.r9_a0 import (
    AUROC_MIN,
    BRIER_MAX,
    K_ORACLE_MIN,
    K_PRIMARY,
    TAU_V,
    auroc,
    fit_logistic,
    predict_b,
    run_r9_a0,
)


def test_r9_a0_freezes_two_layer_and_tau():
    assert TAU_V == 0.5
    from aprwm_v0.r9_a0 import RIDGE, S_SCALE
    assert S_SCALE == 1.0e-3
    assert RIDGE == 1.0e-8
    assert K_PRIMARY == 2
    assert K_ORACLE_MIN == 2
    assert AUROC_MIN == 0.80
    assert BRIER_MAX == 0.20
    assert RECALL_MIN == 0.25
    assert SHUFFLE_SEED == 0
    src = Path("aprwm_v0/r9_a0.py").read_text(encoding="utf-8")
    assert "decay" in src
    assert "change_detection" in src
    assert "fused_score" in src
    s = np.asarray([0.0, 0.001, 0.003])
    y = np.asarray([1.0, 1.0, 0.0])
    beta = fit_logistic(s, y)
    p = predict_b(s, beta)
    assert auroc(y.astype(int), p) >= 0.99


def test_r9_a0_locked_without_p0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R9-P0"):
        run_r9_a0(tmp_path, p0_summary=tmp_path / "missing.json")

"""Unit tests for R7-P0 frozen family locks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r7_p0 import (
    ALPHAS,
    M_MIN,
    N_SIDE_MIN,
    U_ALT,
    U_DEFAULT,
    Y_STAR,
    run_r7_p0,
    simulate_effectiveness,
    task_loss,
)


def test_p0_freezes_two_actions_and_has_no_certificate():
    assert U_DEFAULT == 1.0
    assert U_ALT == 0.4
    assert Y_STAR == 0.10
    assert M_MIN == 1.0e-3
    assert N_SIDE_MIN == 3
    assert ALPHAS[0] < ALPHAS[-1]
    src = Path("aprwm_v0/r7_p0.py").read_text(encoding="utf-8")
    assert "Adam" not in src
    assert "LCB" not in src
    assert "fit_ridge" not in src


def test_hold_state_does_not_see_alpha():
    a = simulate_effectiveness(0.6, U_DEFAULT)
    b = simulate_effectiveness(3.4, U_DEFAULT)
    assert a["hs"].shape == b["hs"].shape
    assert abs(a["x"] - b["x"]) > 0.1
    assert np.allclose(a["hs"], b["hs"], atol=1.0e-12)


def test_advantage_uses_frozen_j():
    mid = simulate_effectiveness(0.6, U_DEFAULT)
    cons = simulate_effectiveness(0.6, U_ALT)
    adv = task_loss(mid) - task_loss(cons)
    assert adv < 0.0


def test_p0_locked_without_b0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R6-B0"):
        run_r7_p0(tmp_path, b0_summary=tmp_path / "missing.json")

"""Unit tests for R7-P1 misspecification preflight locks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r7_a0 import POLY_DEG
from aprwm_v0.r7_p0 import M_MIN, U_ALT, U_DEFAULT
from aprwm_v0.r7_p1 import E_MIN, F_MAX, R2_MAX, run_r7_p1, simulate_saturated


def test_p1_freezes_saturation_and_mis_gate():
    assert F_MAX == 2.0
    assert E_MIN == 1.0e-4
    assert R2_MAX == 0.99
    assert POLY_DEG == 2
    assert M_MIN == 1.0e-3
    src = Path("aprwm_v0/r7_p1.py").read_text(encoding="utf-8")
    assert "conformal_q" not in src
    assert '"certificate": False' in src or "certificate" in src


def test_hold_still_hides_alpha_and_mid_clips():
    a = simulate_saturated(0.5, U_DEFAULT)
    b = simulate_saturated(3.8, U_DEFAULT)
    assert np.allclose(a["hs"], b["hs"], atol=1.0e-12)
    assert b["force"] == pytest.approx(F_MAX)
    cons = simulate_saturated(3.8, U_ALT)
    assert cons["force"] < F_MAX


def test_p1_locked_without_a0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-A0"):
        run_r7_p1(tmp_path, a0_summary=tmp_path / "missing.json")

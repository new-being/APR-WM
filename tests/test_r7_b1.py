"""Unit tests for R7-B1 frozen dynamics-shift audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_a0 import DELTA, EPS
from aprwm_v0.r7_b0 import ALPHAS_TEST as B0_TEST
from aprwm_v0.r7_b1 import F_MAX_TEST, ALPHAS_TEST, run_r7_b1
from aprwm_v0.r7_p1 import F_MAX


def test_b1_freezes_b0_objects_and_one_shift():
    assert DELTA == 1.0e-3
    assert EPS == 0.10
    assert F_MAX == 2.0
    assert F_MAX_TEST == 1.5
    assert F_MAX_TEST != F_MAX
    assert ALPHAS_TEST == B0_TEST
    src = Path("aprwm_v0/r7_b1.py").read_text(encoding="utf-8")
    assert "fit_wm" not in src
    assert "conformal_q" not in src
    assert "adapted" in src


def test_b1_locked_without_b0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-B0"):
        run_r7_b1(tmp_path, b0_summary=tmp_path / "missing.json")

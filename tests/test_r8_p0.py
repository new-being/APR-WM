"""Unit tests for R8-P0 active certificate-validity preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_p1 import F_MAX
from aprwm_v0.r8_p0 import (
    F_MAX_BENIGN,
    F_MAX_ID,
    F_MAX_INVALID,
    GAP_MIN,
    N_EPI,
    s_epi,
    run_r8_p0,
)


def test_r8_p0_freezes_probe_and_three_classes():
    assert F_MAX_ID == F_MAX == 2.0
    assert F_MAX_BENIGN == 2.2
    assert F_MAX_INVALID == 1.5
    assert N_EPI == 50
    assert GAP_MIN == 1.0e-3
    assert s_epi(0.2, 0.1) == 0.1
    src = Path("aprwm_v0/r8_p0.py").read_text(encoding="utf-8")
    assert "tau" in src
    assert "def s_epi" in src
    fit = src.split("def s_epi")[1].split("def _hold_hs")[0]
    assert "f_max" not in fit
    assert "task_loss" not in fit


def test_r8_p0_locked_without_b1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-B1"):
        run_r8_p0(tmp_path, b1_summary=tmp_path / "missing.json")

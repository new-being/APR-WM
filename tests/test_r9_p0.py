"""Unit tests for R9-P0 amortized validity-acquisition preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_p0 import PROBE_S
from aprwm_v0.r8_p0 import N_EPI, U_EPI
from aprwm_v0.r8_r0 import KD, KP
from aprwm_v0.r9_p0 import K_MAX, T_RECOVER_MAX, T_TASK, k_min_from_costs, run_r9_p0


def test_r9_p0_freezes_amortization_constants():
    assert U_EPI == 1.0
    assert N_EPI == 50
    assert T_TASK == PROBE_S == 0.50
    assert T_RECOVER_MAX == 5.0
    assert K_MAX == 20
    assert abs(KP - 50.0) < 1.0e-12
    assert abs(KD - 8.0) < 1.0e-12
    assert k_min_from_costs(3.0, 1.0) == 4
    src = Path("aprwm_v0/r9_p0.py").read_text(encoding="utf-8")
    assert "gain_retune" in src
    assert "block_stationary" in src


def test_r9_p0_locked_without_r0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R8-R0"):
        run_r9_p0(tmp_path, r0_summary=tmp_path / "missing.json")


def test_r9_p0_locked_unless_r8_stopped(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text('{"stage": "R8-R0", "r8_family_stop": false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not stopped"):
        run_r9_p0(tmp_path, r0_summary=path)

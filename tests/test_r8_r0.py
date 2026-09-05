"""Unit tests for R8-R0 billed probe→reset feasibility."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from aprwm_v0.r7_p0 import PROBE_S
from aprwm_v0.r8_p0 import GAP_MIN, N_EPI, U_EPI
from aprwm_v0.r8_p1 import RETURN_MAX
from aprwm_v0.r8_r0 import KD, KP, N_TOTAL, T_TOTAL, reset_u, run_r8_r0


def test_r0_freezes_p0_probe_and_nominal_pd():
    assert U_EPI == 1.0
    assert N_EPI == 50
    assert T_TOTAL == PROBE_S == 0.50
    assert N_TOTAL == 250
    assert abs(KP - 50.0) < 1.0e-12
    assert abs(KD - 8.0) < 1.0e-12
    assert GAP_MIN == 1.0e-3
    assert RETURN_MAX == 2.2e-3
    names = tuple(inspect.signature(reset_u).parameters)
    assert names == ("y", "v")
    src = Path("aprwm_v0/r8_r0.py").read_text(encoding="utf-8")
    body = src.split("def reset_u")[1].split("def reset_is_state_only")[0]
    assert "f_max" not in body
    assert "S_epi" not in body
    assert "task_loss" not in body


def test_r0_locked_without_p2(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R8-P2"):
        run_r8_r0(tmp_path, p2_summary=tmp_path / "missing.json")


def test_r0_locked_unless_family_frozen(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text('{"stage": "R8-P2", "open_loop_probe_family_frozen": false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not frozen"):
        run_r8_r0(tmp_path, p2_summary=path)

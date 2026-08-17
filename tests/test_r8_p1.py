"""Unit tests for R8-P1 state-neutral validity probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r8_p0 import GAP_MIN, N_EPI, U_EPI
from aprwm_v0.r8_p1 import PHASE_N, PHASE_SIGN, RETURN_MAX, epi_command, run_r8_p1, s_epi


def test_p1_freezes_waveform_not_sweep():
    assert U_EPI == 1.0
    assert N_EPI == 50
    assert PHASE_N == (13, 12, 12, 13)
    assert PHASE_SIGN == (1, -1, -1, 1)
    assert sum(PHASE_N) == N_EPI
    cmd = epi_command()
    assert len(cmd) == 50
    assert cmd[0] == 1.0 and cmd[-1] == 1.0
    assert GAP_MIN == 1.0e-3
    assert RETURN_MAX == 2.2e-3
    assert s_epi([0.0, 1.0], [0.0, 1.0]) == 0.0
    src = Path("aprwm_v0/r8_p1.py").read_text(encoding="utf-8")
    assert "waveform_search" in src
    assert "0.04" not in src


def test_p1_locked_without_p0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R8-P0"):
        run_r8_p1(tmp_path, p0_summary=tmp_path / "missing.json")

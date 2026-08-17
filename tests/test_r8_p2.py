"""Unit tests for R8-P2 damping-aware returning probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r8_p0 import GAP_MIN, N_EPI, U_EPI
from aprwm_v0.r8_p1 import RETURN_MAX
from aprwm_v0.r8_p2 import PHASE_N, PHASE_SIGN, epi_command, run_r8_p2


def test_p2_freezes_damped_three_phase():
    assert U_EPI == 1.0
    assert N_EPI == 50
    assert PHASE_SIGN == (1, -1, 1)
    assert PHASE_N == (13, 25, 12)
    assert sum(PHASE_N) == N_EPI
    assert PHASE_N[0] == 13
    cmd = epi_command()
    assert len(cmd) == 50
    assert cmd[0] == 1.0 and cmd[13] == -1.0
    assert GAP_MIN == 1.0e-3
    assert RETURN_MAX == 2.2e-3
    src = Path("aprwm_v0/r8_p2.py").read_text(encoding="utf-8")
    assert "max_d_initial" in src
    assert "S_epi tuning" in src


def test_p2_locked_without_p1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R8-P1"):
        run_r8_p2(tmp_path, p1_summary=tmp_path / "missing.json")

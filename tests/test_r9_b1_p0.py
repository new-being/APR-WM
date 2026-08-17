"""Unit tests for R9-B1-P0 periodic revalidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r9_a0 import K_ORACLE_MIN
from aprwm_v0.r9_b1_p0 import K_CHANGE, M, PROBE_K, run_r9_b1_p0


def test_r9_b1_p0_freezes_m_equals_kmin():
    assert M == K_ORACLE_MIN == 2
    assert PROBE_K == (1, 3, 5)
    assert K_CHANGE["at_probe"] == 3
    assert K_CHANGE["after_probe"] == 2
    src = Path("aprwm_v0/r9_b1_p0.py").read_text(encoding="utf-8")
    assert "m_sweep" in src
    assert "policy_induced_epistemic_blindness" in src
    assert "cusum" in src


def test_r9_b1_p0_locked_without_r9_b0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R9-B0"):
        run_r9_b1_p0(tmp_path, b0_r9_summary=tmp_path / "missing.json")

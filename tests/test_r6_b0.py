"""Unit tests for R6-B0 frozen abstention locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r6_a1 import Z_LOW
from aprwm_v0.r6_b0 import Z_MIN, apply_abstention, run_r6_b0


def test_b0_freezes_zmin_and_does_not_resimulate():
    assert Z_MIN == 1.0
    assert Z_MIN == Z_LOW
    src = Path("aprwm_v0/r6_b0.py").read_text(encoding="utf-8")
    assert "simulate_selfstress" not in src
    assert "Adam" not in src
    assert "threshold_swept" in src


def test_abstention_only_leaves_pi0_when_z_at_least_one():
    assert apply_abstention("mid", "mid", None) == ("mid", "no_proposal")
    assert apply_abstention("mid", "cons", 0.474) == ("mid", "abstain")
    assert apply_abstention("mid", "cons", 1.0) == ("cons", "accepted")
    assert apply_abstention("mid", "cons", 2.0) == ("cons", "accepted")


def test_b0_locked_without_a1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R6-A1"):
        run_r6_b0(
            tmp_path,
            a1_summary=tmp_path / "missing.json",
            d0_summary=tmp_path / "d0.json",
        )

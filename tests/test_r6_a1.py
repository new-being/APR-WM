"""Unit tests for R6-A1 pairwise margin-uncertainty locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r5_d0 import RIDGE
from aprwm_v0.r5_d0_preflight import ACTIONS
from aprwm_v0.r6_a1 import NAMES, PAIRS, Z_LOW, pair_key, run_r6_a1


def test_a1_freezes_pairs_scale_and_no_delta_method():
    assert ACTIONS == (("cons", 0.4), ("mid", 1.0), ("agg", 2.5))
    assert RIDGE == 1.0e-3
    assert NAMES == ("cons", "mid", "agg")
    assert len(PAIRS) == 6
    assert ("cons", "mid") in PAIRS and ("mid", "cons") in PAIRS
    assert Z_LOW == 1.0
    src = Path("aprwm_v0/r6_a1.py").read_text(encoding="utf-8")
    assert "Adam" not in src
    assert "grad_j" not in src
    assert "delta_method" in src


def test_z_is_abs_margin_over_sigma():
    hat_m = -9.2e-8
    sigma = 1.0e-5
    z = abs(hat_m) / sigma
    assert z < Z_LOW
    assert pair_key("mid", "cons") == "mid|cons"


def test_a1_locked_without_a0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R6-A0"):
        run_r6_a1(
            tmp_path,
            a0_summary=tmp_path / "missing.json",
            d0_summary=tmp_path / "d0.json",
        )

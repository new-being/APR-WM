"""Unit tests for R7-A2 frozen-certificate recall recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_a0 import DELTA, EPS
from aprwm_v0.r7_a1 import ALPHAS_CAL, ALPHAS_FIT, ALPHAS_TEST
from aprwm_v0.r7_a2 import KNOT_ALPHA, KNOT_X, run_r7_a2
from aprwm_v0.r7_p1 import F_MAX


def test_a2_freezes_certificate_and_one_spline():
    assert DELTA == 1.0e-3
    assert EPS == 0.10
    assert F_MAX == 2.0
    assert KNOT_ALPHA == (1.2, 2.0, 2.8)
    assert KNOT_X == (0.30, 0.50, 0.70)
    src = Path("aprwm_v0/r7_a2.py").read_text(encoding="utf-8")
    assert "model_search" in src
    assert "degree 3" not in src.lower()
    from aprwm_v0.r7_a2 import ALPHAS_FIT as A2_FIT
    from aprwm_v0.r7_a2 import ALPHAS_TEST as A2_TEST

    assert A2_FIT == ALPHAS_FIT
    assert A2_TEST == ALPHAS_TEST
    assert ALPHAS_CAL[0] == 0.55


def test_a2_locked_without_a1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-A1"):
        run_r7_a2(tmp_path, a1_summary=tmp_path / "missing.json")

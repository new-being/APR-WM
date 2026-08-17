"""Unit tests for R7-A1 frozen misspecified-plant certificate."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r7_a0 import DELTA, EPS, POLY_DEG
from aprwm_v0.r7_a1 import ALPHAS_CAL, ALPHAS_FIT, ALPHAS_TEST, run_r7_a1
from aprwm_v0.r7_p1 import ALPHAS as P1_ALPHAS
from aprwm_v0.r7_p1 import F_MAX


def test_a1_keeps_a0_certificate_and_fresh_alphas():
    assert POLY_DEG == 2
    assert DELTA == 1.0e-3
    assert EPS == 0.10
    assert F_MAX == 2.0
    assert len(ALPHAS_TEST) == 28
    p1 = {round(float(a), 4) for a in P1_ALPHAS}
    used = list(ALPHAS_FIT) + list(ALPHAS_CAL) + list(ALPHAS_TEST)
    assert not p1.intersection({round(float(a), 4) for a in used})
    src = Path("aprwm_v0/r7_a1.py").read_text(encoding="utf-8")
    assert "from .r7_a0 import" in src
    assert "model_upgraded" in src


def test_a1_locked_without_p1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-P1"):
        run_r7_a1(tmp_path, p1_summary=tmp_path / "missing.json")

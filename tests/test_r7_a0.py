"""Unit tests for R7-A0 frozen certificate locks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r7_a0 import (
    ALPHAS_CAL,
    ALPHAS_FIT,
    ALPHAS_TEST,
    DELTA,
    EPS,
    POLY_DEG,
    SHUFFLE_SEED,
    conformal_q,
    run_r7_a0,
)
from aprwm_v0.r7_p0 import ALPHAS as P0_ALPHAS


def test_a0_splits_are_fresh_and_delta_frozen():
    p0 = {round(float(a), 4) for a in P0_ALPHAS}
    used = list(ALPHAS_FIT) + list(ALPHAS_CAL) + list(ALPHAS_TEST)
    assert len(used) == len(set(round(float(a), 4) for a in used))
    assert not p0.intersection({round(float(a), 4) for a in used})
    assert DELTA == 1.0e-3
    assert EPS == 0.10
    assert POLY_DEG == 2
    assert SHUFFLE_SEED == 0
    src = Path("aprwm_v0/r7_a0.py").read_text(encoding="utf-8")
    assert "Adam" not in src
    assert "symmetric_z" in src


def test_conformal_q_is_upper_order_statistic():
    scores = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    q = conformal_q(scores, 0.10)
    assert q == pytest.approx(9.0)


def test_a0_locked_without_p0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R7-P0"):
        run_r7_a0(tmp_path, p0_summary=tmp_path / "missing.json")

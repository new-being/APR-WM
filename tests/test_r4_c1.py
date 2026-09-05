"""Unit tests for R4-C1 future-macro consequence helper."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r4_c1 import C_MIN, FT_PROBE, PROBE_S, relative_c


def test_relative_c_zero_when_identical():
    y = np.ones((4, 3))
    assert relative_c(y, y) == 0.0


def test_relative_c_positive_when_future_forks():
    a = np.zeros((8, 2))
    b = np.zeros((8, 2))
    b[-3:] = 1.0
    assert relative_c(a, b) > C_MIN
    assert FT_PROBE == 4.8
    assert PROBE_S == 0.40

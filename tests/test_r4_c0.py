"""Unit tests for R4-C0 warrant helper (no physics retune)."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r4_c0 import EPS_INFO, REL_EVID_MIN, warrant_from_macro


def test_warrant_nulls_matched_macros():
    a = np.ones((16, 3))
    b = np.ones((16, 3))
    w, relative, active = warrant_from_macro(a, b, rel_min=REL_EVID_MIN)
    assert relative == 0.0
    assert not active
    assert np.allclose(w, 0.0)


def test_warrant_ramps_when_macros_diverge():
    a = np.zeros((16, 2))
    b = np.zeros((16, 2))
    b[:, 0] = np.linspace(0.0, 1.0, 16)
    w, relative, active = warrant_from_macro(a, b, rel_min=REL_EVID_MIN)
    assert active
    assert relative > REL_EVID_MIN
    assert w[0] <= w[-1]
    assert np.isclose(w[-1], 1.0)
    assert EPS_INFO == 0.005

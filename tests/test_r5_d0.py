"""Unit tests for R5-D0 locks."""

from __future__ import annotations

from aprwm_v0.r5_d0 import REL_MIN, RIDGE, SHUFFLE_SEED
from aprwm_v0.r5_d0_preflight import ACTIONS


def test_d0_reuses_preflight_actions_and_loo_gates():
    assert ACTIONS == (("cons", 0.4), ("mid", 1.0), ("agg", 2.5))
    assert REL_MIN == 0.05
    assert RIDGE == 1.0e-3
    assert SHUFFLE_SEED == 0

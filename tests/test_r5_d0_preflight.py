"""Unit tests for R5-D0-preflight frozen action set and loss."""

from __future__ import annotations

from aprwm_v0.r5_d0_preflight import ACTIONS, BETA_LIM, BETA_U, Y_STAR


def test_d0_preflight_actions_and_loss_are_frozen():
    assert ACTIONS == (("cons", 0.4), ("mid", 1.0), ("agg", 2.5))
    assert Y_STAR == 0.010
    assert BETA_U == 1.0e-5
    assert BETA_LIM == 10.0

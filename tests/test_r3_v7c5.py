"""Unit tests for V7C.5 temporal gate."""

from __future__ import annotations

from aprwm_v0.r3_v7c5 import EPS_TEMPORAL, HELD_SEEDS, temporal_gate


def test_fresh_held_seeds_and_eps():
    assert HELD_SEEDS == (26131, 26141)
    assert EPS_TEMPORAL == 0.005


def test_temporal_gate():
    assert not temporal_gate(0.02, 0.019)
    assert temporal_gate(0.02, 0.01)
    assert not temporal_gate(0.004, 0.004)
    assert not temporal_gate(float("nan"), 0.0)

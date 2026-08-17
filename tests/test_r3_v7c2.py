"""Unit tests for V7C.2 locus rule and shuffle."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r3_v7c2 import EPS_INFO, HELD_SEEDS, classify_locus, shuffle_taxel


def test_locus_priority_field_then_encoder_then_fusion():
    assert classify_locus(0.001, 0.02, tactile_useful_vs_n=False, eps=EPS_INFO) == "FIELD"
    assert classify_locus(0.02, 0.0, tactile_useful_vs_n=False, eps=EPS_INFO) == "ENCODER"
    assert classify_locus(0.02, 0.02, tactile_useful_vs_n=False, eps=EPS_INFO) == "FUSION"
    assert classify_locus(0.02, 0.02, tactile_useful_vs_n=True, eps=EPS_INFO) == "UNRESOLVED"
    assert HELD_SEEDS == (23131, 23141)


def test_shuffle_preserves_frames_not_order():
    rng = np.random.default_rng(0)
    maps = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    shuffled = shuffle_taxel(maps, rng)
    assert shuffled.shape == maps.shape
    orig = {tuple(frame.ravel()) for frame in maps}
    got = {tuple(frame.ravel()) for frame in shuffled}
    assert orig == got

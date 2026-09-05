"""Unit tests for V7B.3 evidence weights and Spearman."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r3_v7b3 import DELTA, GAMMA, HELD_SEEDS, evidence_weights, spearman


def test_evidence_weights_are_causal_and_normalized():
    c0 = np.zeros(5)
    c1 = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0])
    weights = evidence_weights(c0, c1)
    assert weights[0] == 0.0
    assert abs(weights[-1] - 1.0) < 1.0e-9
    assert np.all(np.diff(weights) >= -1.0e-12)
    assert DELTA == 0.02 and GAMMA == 0.05
    assert HELD_SEEDS == (20131, 20141)


def test_spearman_tracks_monotone_evidence():
    w = np.linspace(0.0, 1.0, 20)
    p = w + 0.01 * np.sin(np.arange(20))
    assert spearman(p, w) > 0.8
    assert spearman(np.ones(10), np.linspace(0, 1, 10)) == 0.0

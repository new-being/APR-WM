"""Unit tests for R3-V7B.1 IBS and C0 occupancy metrics."""

from __future__ import annotations

import math

import numpy as np

from aprwm_v0.r3_v7b1 import (
    DELTA,
    HELD_SEEDS,
    TRAIN_SEEDS,
    VAL_SEED,
    brier_at_frac,
    c0_brier_mass,
    event_mean_p,
    first_phase_index,
    integrated_brier,
)


def test_ibs_episode_balanced():
    long_bad = np.ones(100) * 0.9
    short_good = np.zeros(4)
    # Equal episode weight: mean of 0.81 and 0, not length-weighted ~0.81.
    score = integrated_brier([long_bad, short_good], [0, 0])
    assert abs(score - 0.405) < 1.0e-9


def test_c0_brier_mass_and_frac_curve():
    a = np.asarray([0.2, 0.4, 0.6])
    b = np.asarray([0.0, 0.0, 0.0])
    brier, mass = c0_brier_mass([a, b])
    assert abs(brier - 0.5 * (np.mean(a * a) + 0.0)) < 1.0e-12
    assert abs(mass - 0.5 * (np.mean(a) + 0.0)) < 1.0e-12
    labels = [1, 0]
    assert math.isfinite(brier_at_frac([a, b], labels, 1.0))
    assert DELTA == 0.02


def test_event_alignment_evaluator_only():
    phase = np.asarray([0.0, 2.0, 2.0, 3.0, 2.5, 5.0])
    probs = np.asarray([0.1, 0.8, 0.7, 0.2, 0.6, 0.05])
    assert first_phase_index(phase, 2.0) == 1
    assert first_phase_index(phase, 5.0) == 5
    assert abs(event_mean_p(probs, phase, 5.0, half=0) - 0.05) < 1.0e-12
    assert TRAIN_SEEDS == (18101, 18111)
    assert VAL_SEED == 18121
    assert HELD_SEEDS == (18131, 18141)

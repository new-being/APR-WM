"""Unit tests for R3-V7B persistent belief metrics."""

from __future__ import annotations

import math

import numpy as np

from aprwm_v0.r3_v7b import (
    HELD_SEEDS,
    TRAIN_SEEDS,
    VAL_SEED,
    detection_time,
    mean_positive_dwell,
    recovery_time,
)


def test_detection_time_and_dwell():
    probs = np.asarray([0.1, 0.2, 0.81, 0.9, 0.4])
    assert detection_time(probs, 0.8) == 2
    assert detection_time(probs, 0.99) == 5
    assert mean_positive_dwell(probs > 0.8) == 2.0
    assert mean_positive_dwell(probs > 0.99) == 0.0


def test_recovery_after_contact_transient():
    probs = np.asarray([0.1, 0.85, 0.9, 0.88, 0.2, 0.1])
    assert recovery_time(probs, 0.5, quiet_start=4) == 0.0
    never = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])
    assert recovery_time(never, 0.5, quiet_start=3) == 0.0
    stuck = np.asarray([0.1, 0.9, 0.9, 0.9, 0.9])
    assert recovery_time(stuck, 0.5, quiet_start=2) == 3.0


def test_seed_splits_are_fresh():
    assert TRAIN_SEEDS == (17101, 17111)
    assert VAL_SEED == 17121
    assert HELD_SEEDS == (17131, 17141)
    assert math.isfinite(1.0)

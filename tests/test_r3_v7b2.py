"""Unit tests for R3-V7B.2 fast/slow structure and event-rise metric."""

from __future__ import annotations

import math

import numpy as np
import torch

from aprwm_v0.r3_v7b import STEP_DIM
from aprwm_v0.r3_v7b2 import (
    B2_HIDDEN,
    B4_HIDDEN,
    DELTA,
    HELD_SEEDS,
    WIDE_HIDDEN,
    FastSlowEpi,
    count_module,
    event_delta_p,
    max_event_rise,
    n_params_fastslow,
    n_params_gru,
)


def test_capacity_match_within_five_percent():
    b4 = n_params_fastslow(STEP_DIM, B4_HIDDEN)
    wide = n_params_gru(STEP_DIM, WIDE_HIDDEN)
    b2 = n_params_gru(STEP_DIM, B2_HIDDEN)
    assert b4 > b2
    assert abs(b4 - wide) / b4 < 0.05
    model = FastSlowEpi(STEP_DIM, B4_HIDDEN)
    assert count_module(model) == b4
    assert DELTA == 0.02
    assert HELD_SEEDS == (19131, 19141)


def test_event_rise_ignores_drops():
    phase = np.asarray([0.0, 0.0, 2.0, 2.0, 3.0, 2.5])
    dropping = np.asarray([0.5, 0.4, 0.2, 0.2, 0.1, 0.1])
    assert event_delta_p(dropping, phase, 2.0) == 0.0
    rising = np.asarray([0.1, 0.1, 0.8, 0.8, 0.1, 0.1])
    assert event_delta_p(rising, phase, 2.0) > 0.4
    assert math.isfinite(max_event_rise(rising, phase))


def test_slow_head_not_fast():
    model = FastSlowEpi(STEP_DIM, 8)
    padded = torch.zeros(1, 5, STEP_DIM)
    lengths = torch.tensor([5])
    probs, fast = model(padded, lengths)
    assert probs.shape == (1, 5)
    assert fast.shape[0] == 1 and fast.shape[1] == 5
    # p comes from slow recurrence; sequence is finite and in (0,1).
    assert torch.all((probs > 0.0) & (probs < 1.0))

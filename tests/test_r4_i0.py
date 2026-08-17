"""Unit tests for R4-I0 v2 stiffness-swap helpers."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r4_i0 import (
    D_TAXEL_MIN,
    d_taxel,
    mean_solref_timeconst,
    mechanics_matched,
)
from aprwm_v0.r4_i1 import AUROC_LEAK_FAIL, AUROC_PASS_MAX, GAIN_MIN, auroc, flip_x


def test_swap_preserves_mean_solref():
    assert mean_solref_timeconst("stiff_left") == mean_solref_timeconst("stiff_right")


def test_mechanics_matched_and_taxel_gap():
    base = {
        "q": 0.0,
        "v": 0.0,
        "fn": 1.0,
        "ft": 0.5,
        "u_finger": 0.2,
        "ft_cmd": 0.4,
        "q_finger": 0.002,
        "p": np.ones((2, 4, 4)),
        "tau_x": np.zeros((2, 4, 4)),
        "tau_y": np.zeros((2, 4, 4)),
    }
    other = dict(base)
    other["tau_x"] = np.ones((2, 4, 4))
    assert mechanics_matched(base, dict(base))
    leak = dict(base)
    leak["fn"] = 2.0
    assert not mechanics_matched(base, leak)
    assert d_taxel(base, other) > D_TAXEL_MIN


def test_auroc_and_leak_constants():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0
    assert AUROC_LEAK_FAIL == 0.95
    assert AUROC_PASS_MAX == 0.90
    assert GAIN_MIN == 0.10


def test_flip_x_reverses_pad_columns():
    grid = np.zeros((2, 4, 4))
    grid[0, 0, 0] = 1.0
    flipped = flip_x(grid)
    assert flipped[0, 0, -1] == 1.0
    assert flipped[0, 0, 0] == 0.0

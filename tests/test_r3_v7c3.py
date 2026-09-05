"""Unit tests for V7C.3 observation vectors and stop rule."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r1_rs2 import _patch_geometry
from aprwm_v0.r3_v7c3 import (
    EPS_INFO,
    HELD_SEEDS,
    observation_vector,
    stop_tactile_branch,
    sufficient_candidates,
)


def test_patch_geometry_zero_without_mass():
    geom = _patch_geometry(np.zeros(0), np.zeros(0), np.zeros(0))
    assert geom.shape == (7,)
    assert np.all(geom == 0.0)
    assert HELD_SEEDS == (24131, 24141)


def test_observation_dims_and_stop_rule():
    t = 4
    row = {
        "T": t,
        "taxel": np.ones((t, 8, 8)),
        "taxel_shear": np.ones((t, 8, 8)) * 2,
        "tactile_geom": np.ones((t, 7)) * 3,
        "tactile_surf": np.ones((t, 9)) * 4,
    }
    assert observation_vector(row, "N").shape == (t, 64)
    assert observation_vector(row, "NS").shape == (t, 128)
    assert observation_vector(row, "NSG").shape == (t, 135)
    assert observation_vector(row, "NSGM").shape == (t, 144)
    deltas = {"N": 0.001, "NS": 0.004, "NSG": 0.004, "NSGM": 0.002}
    assert stop_tactile_branch(deltas, EPS_INFO)
    assert sufficient_candidates(deltas, EPS_INFO) == []
    deltas["NS"] = 0.01
    assert not stop_tactile_branch(deltas, EPS_INFO)
    assert sufficient_candidates(deltas, EPS_INFO) == ["NS"]

"""Unit tests for V7C.4 representation trichotomy and NSG tensors."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r3_v7c4 import (
    EPS_INFO,
    EPS_RETAIN,
    HELD_SEEDS,
    OBS_NAME,
    classify_representation,
    nsg_geom,
    nsg_maps,
)


def test_held_seeds_are_fresh():
    assert HELD_SEEDS == (25131, 25141)
    assert OBS_NAME == "NSG"
    assert EPS_RETAIN == 0.02
    assert EPS_INFO == 0.005


def test_nsg_maps_stack_normal_and_shear():
    t = 3
    row = {
        "T": t,
        "taxel": np.ones((t, 8, 8)),
        "taxel_shear": np.full((t, 8, 8), 2.0),
        "tactile_geom": np.full((t, 7), 3.0),
    }
    maps = nsg_maps(row)
    assert maps.shape == (t, 2, 8, 8)
    assert np.all(maps[:, 0] == 1.0)
    assert np.all(maps[:, 1] == 2.0)
    assert nsg_geom(row).shape == (t, 7)


def test_trichotomy():
    assert classify_representation(0.001, 0.04) == "OBSERVATION_NOT_REPLICATED"
    assert classify_representation(0.057, 0.001) == "DESTROYS"
    assert classify_representation(0.057, 0.02) == "LOSES"
    assert classify_representation(0.057, 0.04) == "SUFFICIENT"
    assert classify_representation(0.057, 0.08) == "SUFFICIENT"

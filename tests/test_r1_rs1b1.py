"""Unit tests for R1-RS1B.1 support-risk calibration (no robosuite)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aprwm_v0.r1_rs1b1 import (
    FORMAL_SEEDS,
    _aggregate,
    _d_tol,
    _region,
    modeled_distance,
    spearman,
)


def test_modeled_distance_zero_inside_and_positive_outside():
    lo, hi, margin = 0.0, 1.0, 0.05
    inside = modeled_distance(np.asarray([0.2, 0.5, 0.8]), lo, hi, margin)
    assert np.allclose(inside, 0.0)
    below = float(modeled_distance(0.0, lo, hi, margin))
    above = float(modeled_distance(1.0, lo, hi, margin))
    assert below > 0.0
    assert above > 0.0


def test_spearman_monotone_and_undefined():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert math.isnan(spearman([1, 1, 1], [2, 3, 4]))


def test_d_tol_formula_and_regions():
    d_exit = np.asarray([0.0, 0.0, 0.2, 0.4])
    i_exit = np.asarray([0.0, 0.0, 1.0, 1.0])
    d_tol = _d_tol(d_exit, i_exit)
    assert d_tol == pytest.approx(0.15)
    assert _region(0.0, d_tol) == "supported"
    assert _region(0.10, d_tol) == "extrapolative"
    assert _region(0.20, d_tol) == "risky"


def test_aggregate_nonimplication_and_no_hard_filter():
    rows = [
        {"seed": 1, "revise_worthy": True, "accepted": True},
        {"seed": 2, "revise_worthy": True, "accepted": True},
    ]
    queries = []
    for seed in (1, 2):
        for index in range(4):
            d_exit = 0.0 if index < 2 else 0.2 * (index)
            rmse = 0.001 + d_exit
            queries.append(
                {
                    "seed": seed,
                    "accepted": True,
                    "D_exit": d_exit,
                    "I_exit": float(d_exit > 0.0),
                    "rmse_h32_revised": rmse,
                    "gain_h32": 0.002,
                    "harmful": False,
                }
            )
    summary = _aggregate(rows, queries)
    assert summary["installs_support_exit_hard_filter"] is False
    assert summary["gates"]["nonimplication"]["pass"] is True
    assert summary["gates"]["seed_mean_spearman_D_rmse"]["pass"] is True
    assert summary["rs1b1_go"] is True


def test_formal_seeds_are_held_out_from_rs1b():
    assert FORMAL_SEEDS == (9951, 9961, 9971, 9981, 9991)
    assert len(FORMAL_SEEDS) == 5

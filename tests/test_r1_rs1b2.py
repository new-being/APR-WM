"""Unit tests for R1-RS1B.2 band identification."""

from __future__ import annotations

import math

import torch

from aprwm_v0.r1_rs1b1 import modeled_distance
from aprwm_v0.r1_rs1b2 import (
    FORMAL_SEEDS,
    OCCUPANCY_MIN,
    _aggregate,
    realized_band,
    sample_band_queries,
)


def test_realized_bins_are_frozen():
    assert realized_band(0.0) == "B0_inside"
    assert realized_band(0.04) == "B1_mild"
    assert realized_band(0.0400001) == "B2_mid"
    assert realized_band(0.08) == "B2_mid"
    assert realized_band(0.0800001) == "B3_far"


def test_targeted_queries_cover_intended_initial_distance():
    generator = torch.Generator().manual_seed(0)
    queries, labels = sample_band_queries(
        lo=0.0, hi=0.4, margin=0.05, n_per_band=8, generator=generator
    )
    assert len(labels) == 32
    by_band = {name: [] for name in ("B0_inside", "B1_mild", "B2_mid", "B3_far")}
    for state, name in zip(queries, labels):
        d0 = float(modeled_distance(float(state[0]), 0.0, 0.4, 0.05))
        by_band[name].append(d0)
    assert max(by_band["B0_inside"]) == 0.0
    assert min(by_band["B1_mild"]) > 0.0
    assert max(by_band["B1_mild"]) <= 0.04 + 1.0e-6
    assert min(by_band["B2_mid"]) > 0.04 - 1.0e-6
    assert max(by_band["B2_mid"]) <= 0.08 + 1.0e-6
    assert min(by_band["B3_far"]) > 0.08 - 1.0e-6


def test_occupancy_go_requires_all_four_bands():
    def queries_for(counts):
        names = ("B0_inside", "B1_mild", "B2_mid", "B3_far")
        out = []
        d_map = {"B0_inside": 0.0, "B1_mild": 0.02, "B2_mid": 0.06, "B3_far": 0.12}
        for name, n in zip(names, counts):
            for _ in range(n):
                out.append(
                    {
                        "accepted": True,
                        "realized_band": name,
                        "intended_band": name,
                        "D_exit": d_map[name],
                        "I_exit": float(d_map[name] > 0),
                        "rmse_h32_revised": 0.001 + d_map[name],
                        "gain_h32": 0.002,
                    }
                )
        return out

    filled = _aggregate([{"accepted": True, "revise_worthy": True}], queries_for((12, 12, 12, 12)))
    assert filled["rs1b2_go"] is True
    assert filled["installs_support_exit_hard_filter"] is False
    empty_mild = _aggregate([{"accepted": True, "revise_worthy": True}], queries_for((20, 0, 12, 12)))
    assert empty_mild["rs1b2_go"] is False
    assert empty_mild["mechanism"] == "unfillable_intermediate"


def test_held_out_seeds_and_occupancy_bar():
    assert FORMAL_SEEDS == (10001, 10011, 10021, 10031, 10041)
    assert OCCUPANCY_MIN == 12
    assert math.isfinite(OCCUPANCY_MIN)

"""Unit tests for R1-RS1C layered policy (no robosuite)."""

from __future__ import annotations

from aprwm_v0.r1_rs1b import FORMAL_SEEDS as RS1B_SEEDS
from aprwm_v0.r1_rs1b1 import FORMAL_SEEDS as RS1B1_SEEDS
from aprwm_v0.r1_rs1b2 import FORMAL_SEEDS as RS1B2_SEEDS
from aprwm_v0.r1_rs1c import (
    ALPHAS,
    AMP_SCALES,
    FORMAL_SEEDS,
    SMOKE_JOBS,
    SMOKE_SEED,
    R1RS1CConfig,
    _aggregate,
    _bucket,
    _monitor_intensity,
    _policy_gain,
)


def test_formal_matrix_and_held_out_seeds():
    assert len(FORMAL_SEEDS) * len(ALPHAS) * len(AMP_SCALES) == 100
    assert FORMAL_SEEDS == (10101, 10111, 10121, 10131, 10141)
    overlap = set(FORMAL_SEEDS) & (set(RS1B_SEEDS) | set(RS1B1_SEEDS) | set(RS1B2_SEEDS))
    assert not overlap
    assert SMOKE_SEED == 9041
    assert SMOKE_JOBS == (
        (9041, 0.0, 0.5),
        (9041, 0.0, 1.5),
        (9041, -0.24, 0.5),
        (9041, -0.24, 1.5),
    )


def test_allocation_buckets_and_monitor_flag():
    assert _bucket(consequential=False, detected=True) == "tolerate"
    assert _bucket(consequential=True, detected=False) == "probe"
    assert _bucket(consequential=True, detected=True) == "revise_worthy"
    assert _monitor_intensity(installed=False, i_exit=0.4) == "none"
    assert _monitor_intensity(installed=True, i_exit=0.0) == "normal"
    assert _monitor_intensity(installed=True, i_exit=0.2) == "elevated"


def test_support_exit_does_not_veto_policy_gain():
    row = {
        "accepted": True,
        "accepted_always": True,
        "I_exit": 0.5,
        "gain_h32": 0.01,
    }
    assert _policy_gain(row, "rs1c") == 0.01
    assert _policy_gain(row, "veto") == 0.0
    assert _policy_gain(row, "none") == 0.0
    assert _policy_gain({**row, "accepted": False}, "rs1c") == 0.0


def test_aggregate_go_and_monitor_leakage():
    cfg = R1RS1CConfig(seeds=(10101, 10111))
    rows = []
    queries = []
    for seed in cfg.seeds:
        rows.append(
            {
                "seed": seed,
                "bucket_initial": "revise_worthy",
                "bucket_final": "revise_worthy",
                "revise_worthy": True,
                "promoted": False,
                "accepted": True,
                "accepted_always": True,
                "accepted_flipped_by_support": False,
                "passivity_violation": False,
                "monitor_intensity": "elevated",
                "I_exit": 0.2,
                "gain_h32": 0.01,
            }
        )
        rows.append(
            {
                "seed": seed,
                "bucket_initial": "tolerate",
                "bucket_final": "tolerate",
                "revise_worthy": False,
                "promoted": False,
                "accepted": False,
                "accepted_always": False,
                "accepted_flipped_by_support": False,
                "passivity_violation": False,
                "monitor_intensity": "none",
                "I_exit": 0.0,
                "gain_h32": 0.0,
            }
        )
        queries.append(
            {
                "accepted": True,
                "episode_I_exit": 0.2,
                "harmful": False,
            }
        )
    summary = _aggregate(rows, queries, cfg)
    assert summary["installs_support_exit_hard_filter"] is False
    assert summary["rewrites_rs1b_go"] is False
    assert summary["rs1c_go"] is True
    assert summary["gates"]["gain_vs_veto"]["pass"] is True

    rows[0]["accepted_flipped_by_support"] = True
    leaked = _aggregate(rows, queries, cfg)
    assert leaked["rs1c_go"] is False
    assert leaked["fail_layer"] == "monitoring_leakage_into_accept"

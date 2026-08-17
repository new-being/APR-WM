"""Unit tests for RS5A indexed calibration and abstention."""

from __future__ import annotations

import math

from aprwm_v0.r1_rs4a import REGIMES
from aprwm_v0.r1_rs5a import (
    DEV_SEEDS,
    HELD_SEEDS,
    SEEDS,
    R1RS5AConfig,
    _aggregate,
    coarse_index,
)


def test_coarse_index_does_not_read_family_label():
    pull = {
        "family": "contact_push",
        "contact_frac": 1.0,
        "sign_coverage": 0.49,
    }
    assert coarse_index(pull) == "contact"
    assert coarse_index({"contact_frac": 0.0, "family": "contact_push"}) == "mode_a"


def _row(family: str, seed: int, regime: str) -> dict:
    y = 0 if regime == "C0" else 1
    if family == "mode_a":
        contact, sign, log_n, span = 0.0, 0.48, 8.5, 0.40
        log_d0 = -6.3 + 1.2 * y
    elif family == "contact_pull":
        contact, sign, log_n, span = 1.0, 0.10, 5.5, 0.04
        log_d0 = -7.6 + 1.2 * y
    else:
        contact, sign, log_n, span = 1.0, 0.49, 6.2, 0.04
        log_d0 = -7.4 + 1.2 * y
    d0 = math.exp(log_d0)
    return {
        "family": family,
        "seed": seed,
        "regime": regime,
        "script": None if family == "mode_a" else ("fast_pull" if family == "contact_pull" else "pull_push"),
        "y_inadequate": y,
        "D0": d0,
        "log_d0": log_d0,
        "contact_frac": contact,
        "sign_coverage": sign,
        "log_n": log_n,
        "q_span": span,
    }


def test_aggregate_abstain_on_unseen_without_using_push_label():
    rows = []
    for seed in SEEDS:
        for family in ("mode_a", "contact_pull", "contact_push"):
            for regime in REGIMES:
                rows.append(_row(family, seed, regime))
    assert set(DEV_SEEDS) == {15101, 15111, 15121}
    assert set(HELD_SEEDS) == {15131, 15141}
    summary, scored = _aggregate(rows, R1RS5AConfig())
    assert summary["uses_pull_push_label_in_policy"] is False
    assert summary["one_hot_closed_world"] is False
    assert summary["unlocks_rs5b"] is False
    push = [row for row in scored if row["family"] == "contact_push"]
    assert push and all(row["coarse_I"] == "contact" for row in push)
    assert summary["h3_unseen_false_confidence"]["fcr"]["global"] == 1.0
    assert summary["h3_unseen_false_confidence"]["fcr"]["indexed"] == 1.0
    assert summary["h3_unseen_false_confidence"]["pass"] is True
    assert summary["h4_calibrated_contact_abstain"]["pass"] is True
    assert summary["rs5a_go"] is True

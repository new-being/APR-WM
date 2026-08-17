"""Unit tests for R3-V7A mixture-trained contextual belief."""

from __future__ import annotations

import math

from aprwm_v0.r1_rs4a import FEATURE_NAMES, REGIMES
from aprwm_v0.r3_v7a import DEV_SEEDS, HELD_SEEDS, SEEDS, R3V7AConfig, _aggregate


def _row(family: str, seed: int, regime: str, amp: float = 1.0) -> dict:
    y = 0 if regime == "C0" else 1
    if family == "mode_a":
        contact, sign, log_n, span = 0.0, 0.48, 8.5, 0.40
        log_d0 = -6.3 + 1.4 * y + 0.05 * (amp - 1.0)
        script = None
    elif family == "contact_pull":
        contact, sign, log_n, span = 1.0, 0.10, 5.5, 0.04
        log_d0 = -7.6 + 1.4 * y
        script = "fast_pull"
    else:
        contact, sign, log_n, span = 1.0, 0.49, 6.2, 0.04
        log_d0 = -7.4 + 1.4 * y
        script = "pull_push"
    d0 = math.exp(log_d0)
    row = {
        "family": family,
        "domain": "mode_a" if family == "mode_a" else "contact",
        "seed": seed,
        "regime": regime,
        "script": script,
        "scale": amp,
        "y_inadequate": y,
        "D0": d0,
        "log_d0": log_d0,
        "log_sv_min": -4.0 + 0.2 * y,
        "log_cond_J": 2.0 + 0.1 * contact,
        "sign_coverage": sign,
        "log_n": log_n,
        "contact_frac": contact,
        "log_rms_v": -2.0 + 0.3 * (1.0 - contact),
        "log_cond_G": 1.5,
        "abs_corr_absvv_v2": 0.2 * y,
        "q_span": span,
    }
    assert set(FEATURE_NAMES).issubset(row)
    return row


def test_mixture_go_without_family_label():
    rows = []
    for seed in SEEDS:
        for family in ("mode_a", "contact_pull", "contact_push"):
            for regime in REGIMES:
                amps = (1.0, 1.5) if family == "mode_a" else (1.0,)
                for amp in amps:
                    rows.append(_row(family, seed, regime, amp))
    assert set(DEV_SEEDS) == {16101, 16111, 16121}
    assert set(HELD_SEEDS) == {16131, 16141}
    summary = _aggregate(rows, R3V7AConfig())
    assert summary["uses_family_label_in_policy"] is False
    assert summary["unlocks_rs5b"] is False
    assert summary["is_old_v07_routing"] is False
    assert summary["h1_mixture_brier"]["pass"] is True
    assert summary["h2_held_c0_fpr"]["pass"] is True
    assert summary["h3_within_family_auroc"]["pass"] is True
    assert summary["v7a_go"] is True
    diag = summary["diagnostic_loio_push"]
    assert diag["n_test"] > 0
    assert math.isfinite(diag["brier_loio"])

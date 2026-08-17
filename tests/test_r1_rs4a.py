"""Unit tests for RS4A geometry-conditioned evidence (no domain ID)."""

from __future__ import annotations

import math

import numpy as np

from aprwm_v0.r1_rs2 import SCRIPTS
from aprwm_v0.r1_rs4a import (
    FEATURE_NAMES,
    LOIO,
    MODE_A_AMPS,
    PULL_JOBS,
    PUSH_JOBS,
    REGIMES,
    SEEDS,
    R1RS4AConfig,
    _aggregate,
    _fit_logit,
    _predict_logit,
    geometry_features,
)


def test_rs4a_matrix_is_three_families_and_not_rs2_scripts():
    cfg = R1RS4AConfig()
    assert cfg.seeds[0] == 14101
    assert "C2" not in REGIMES
    assert "pull_push" not in SCRIPTS
    n = (
        len(cfg.seeds) * len(REGIMES) * len(MODE_A_AMPS)
        + len(cfg.seeds) * len(REGIMES) * len(PULL_JOBS)
        + len(cfg.seeds) * len(REGIMES) * len(PUSH_JOBS)
    )
    assert n == 90
    held = {hold for hold, _train in LOIO}
    assert held == {"mode_a", "contact_pull", "contact_push"}
    assert "domain" not in FEATURE_NAMES
    assert "X_phi_perp" not in FEATURE_NAMES


def test_geometry_features_are_finite_and_ignore_labels():
    rng = np.random.default_rng(0)
    tangent = rng.normal(size=(80, 2))
    q = rng.uniform(-0.1, 0.4, size=80)
    v = rng.normal(scale=0.2, size=80)
    geom = geometry_features(tangent=tangent, q=q, velocity=v, d0=1.0e-3, contact_frac=0.2)
    assert set(geom) == set(FEATURE_NAMES)
    assert all(math.isfinite(value) for value in geom.values())
    assert geom["contact_frac"] == 0.2
    assert 0.0 <= geom["sign_coverage"] <= 0.5 + 1.0e-9


def test_logit_recovers_separable_labels():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 2))
    y = (x[:, 0] > 0).astype(np.float64)
    weights = _fit_logit(x, y, ridge=0.1)
    probs = _predict_logit(x, weights)
    acc = float(np.mean((probs > 0.5) == y))
    assert acc >= 0.85


def _synth_row(family: str, regime: str, seed: int) -> dict:
    y = 0 if regime == "C0" else 1
    contact = 0.0 if family == "mode_a" else (0.55 if family == "contact_pull" else 0.70)
    d0 = (2.0e-4 if y == 0 else 2.0e-3) * (1.0 + 0.4 * contact)
    log_d0 = math.log(d0 + 1.0e-12)
    return {
        "family": family,
        "domain": "mode_a" if family == "mode_a" else "contact",
        "seed": seed,
        "regime": regime,
        "script": None
        if family == "mode_a"
        else ("fast_pull" if family == "contact_pull" else "pull_push"),
        "y_inadequate": y,
        "D0": d0,
        "log_d0": log_d0,
        "log_sv_min": -2.0,
        "log_cond_J": 1.0,
        "sign_coverage": 0.4 if family == "contact_push" else 0.05,
        "log_n": 5.5 if family == "mode_a" else 5.0,
        "contact_frac": contact,
        "log_rms_v": -2.0 + 0.3 * y,
        "log_cond_G": 2.0,
        "abs_corr_absvv_v2": 0.2 if family == "mode_a" else 0.9,
        "q_span": 0.5 if family == "mode_a" else 0.08,
    }


def test_aggregate_loio_does_not_unlock_rs4b():
    rows = []
    for seed in SEEDS:
        for family in ("mode_a", "contact_pull", "contact_push"):
            for regime in REGIMES:
                for copy in range(2):
                    rows.append(_synth_row(family, regime, seed + copy))
    summary = _aggregate(rows, R1RS4AConfig())
    assert summary["uses_domain_id"] is False
    assert summary["one_hot_thresholds"] is False
    assert summary["unlocks_rs4b"] is False
    assert summary["unlocks_rs3b"] is False
    assert "rs4a_go" in summary

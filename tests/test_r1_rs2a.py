"""Unit tests for RS2A visibility geometry (no policy retune)."""

from __future__ import annotations

import math

import numpy as np

from aprwm_v0.r1_rs2a import (
    CONTACT_JOBS,
    MODE_A_AMPS,
    REGIMES,
    R1RS2AConfig,
    _aggregate,
    _visibility,
)
from aprwm_v0.r1_rs2_formal import FORMAL_SEEDS


def test_rs2a_matrix_excludes_c2_and_keeps_policy_frozen():
    cfg = R1RS2AConfig()
    assert cfg.seeds == FORMAL_SEEDS
    assert REGIMES == ("C0", "C1-L", "C1-H")
    assert "C2" not in REGIMES
    assert MODE_A_AMPS == (1.0, 1.5)
    assert len(CONTACT_JOBS) == 4
    n_mode = len(cfg.seeds) * len(cfg.regimes) * len(cfg.mode_a_amps)
    n_contact = len(cfg.seeds) * len(cfg.regimes) * len(CONTACT_JOBS)
    assert n_mode + n_contact == 90


def test_visibility_is_one_minus_r_squared_on_tangent():
    rng = np.random.default_rng(0)
    tangent = rng.normal(size=(200, 2))
    velocity = rng.normal(size=200)
    vis = _visibility(tangent, velocity, ridge=1.0e-5, epsilon=1.0e-18)
    assert 0.0 <= vis["kappa_perp"] <= 1.0 + 1.0e-6
    assert vis["X_phi_perp"] <= vis["X_phi"] + 1.0e-9

    aligned_v = rng.normal(size=200)
    aligned_tangent = np.stack([aligned_v, np.abs(aligned_v) * aligned_v], axis=1)
    # Φ is exactly the second tangent column, so P_T Φ ≈ Φ.
    aligned = _visibility(
        aligned_tangent, aligned_v, ridge=1.0e-12, epsilon=1.0e-18
    )
    assert aligned["kappa_perp"] < 0.05


def test_aggregate_h2_h3_do_not_rewrite_rs2():
    rows = []
    for seed in FORMAL_SEEDS:
        for regime, alpha in (("C1-L", -0.12), ("C1-H", -0.24)):
            x_phi = 4.0e-4
            k_m = 0.90
            k_c = 0.10
            rows.append(
                {
                    "domain": "mode_a",
                    "seed": seed,
                    "regime": regime,
                    "script": None,
                    "scale": math.nan,
                    "amp_scale": 1.5,
                    "alpha": alpha,
                    "D0": abs(alpha) * math.sqrt(k_m * x_phi),
                    "X_phi": x_phi,
                    "X_phi_perp": k_m * x_phi,
                    "kappa_perp": k_m,
                    "cond_G": 10.0,
                    "corr_abs_v_v_signed_v2": 0.2,
                }
            )
            rows.append(
                {
                    "domain": "contact",
                    "seed": seed,
                    "regime": regime,
                    "script": "pull_release",
                    "scale": 2.0,
                    "amp_scale": 1.5,
                    "alpha": alpha,
                    "D0": abs(alpha) * math.sqrt(k_c * x_phi),
                    "X_phi": x_phi,
                    "X_phi_perp": k_c * x_phi,
                    "kappa_perp": k_c,
                    "cond_G": 40.0,
                    "corr_abs_v_v_signed_v2": 0.8,
                }
            )
    summary = _aggregate(rows, R1RS2AConfig())
    assert summary["h1_raw_exposure"]["pass"] is True
    assert summary["h2_visibility"]["pass"] is True
    assert summary["h3a_spearman"]["pass"] is True
    assert summary["h3b_nrmse"]["pass"] is True
    assert summary["rs2a_go"] is True
    assert summary["rewrites_rs2_go"] is False
    assert summary["unlocks_threshold_retune"] is False
    assert summary["unlocks_rs2b"] is False

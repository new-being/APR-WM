"""Unit tests for RS3A.1 excess-risk evidence (no S_perp repair)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r1_rs2 import FROZEN_ADAPTER_SCALE
from aprwm_v0.r1_rs2a import CONTACT_JOBS, PRIMARY_CONTACT
from aprwm_v0.r1_rs3a1 import (
    MODE_A_AMPS,
    REGIMES,
    SEEDS,
    R1RS3A1Config,
    _aggregate,
    _require_rs3a,
    excess_risk_evidence,
)


def test_rs3a1_matrix_matches_rs3a_and_excludes_c2():
    cfg = R1RS3A1Config()
    assert cfg.seeds == SEEDS
    assert cfg.seeds[0] == 13101
    assert REGIMES == ("C0", "C1-L", "C1-H")
    assert "C2" not in REGIMES
    assert MODE_A_AMPS == (1.0, 1.5)
    assert PRIMARY_CONTACT == ("pull_release", FROZEN_ADAPTER_SCALE)
    n_mode = len(cfg.seeds) * len(cfg.regimes) * len(cfg.mode_a_amps)
    n_contact = len(cfg.seeds) * len(cfg.regimes) * len(CONTACT_JOBS)
    assert n_mode + n_contact == 90
    assert cfg.k_fit == cfg.k_hold == 100


def test_excess_risk_near_zero_when_linear_is_enough():
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.2, 0.6, size=240)
    v = rng.normal(scale=0.2, size=240)
    tangent = np.stack([np.ones(240), v], axis=1)
    residual = 0.05 * tangent[:, 0] + 0.02 * tangent[:, 1] + 1.0e-4 * rng.normal(size=240)
    features = np.stack([q, v], axis=1)
    out = excess_risk_evidence(
        tangent=tangent,
        residual=residual,
        features=features,
        k_fit=100,
        k_hold=100,
        param_ridge=1.0e-5,
        residual_ridge=2.0e-3,
        seed=3,
    )
    assert -1.0 <= out["E_XR"] <= 1.0
    assert out["E_XR"] < 0.15


def test_excess_risk_positive_when_rbf_structure_remains():
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.3, 0.7, size=240)
    v = rng.normal(scale=0.35, size=240)
    tangent = np.stack([np.ones(240), v], axis=1)
    bump = np.exp(-0.5 * ((q - 0.4) / 0.25) ** 2 - 0.5 * ((v - 0.0) / 0.45) ** 2)
    residual = 0.01 * v + 0.25 * bump + 1.0e-4 * rng.normal(size=240)
    features = np.stack([q, v], axis=1)
    out = excess_risk_evidence(
        tangent=tangent,
        residual=residual,
        features=features,
        k_fit=100,
        k_hold=100,
        param_ridge=1.0e-5,
        residual_ridge=2.0e-3,
        seed=4,
    )
    assert out["E_XR"] > 0.15
    assert out["L_flex"] < out["L_par"]


def test_aggregate_requires_c0_near_zero_and_does_not_unlock_rs3b():
    rows = []
    for seed in SEEDS:
        for amp in (1.0, 1.5):
            rows.append(
                {
                    "domain": "mode_a",
                    "seed": seed,
                    "regime": "C0",
                    "script": None,
                    "scale": math.nan,
                    "amp_scale": amp,
                    "D0": 1.0e-4,
                    "E_XR": 0.04 if amp > 1.2 else 0.02,
                }
            )
        for script, scale in CONTACT_JOBS:
            rows.append(
                {
                    "domain": "contact",
                    "seed": seed,
                    "regime": "C0",
                    "script": script,
                    "scale": scale,
                    "amp_scale": 1.0,
                    "D0": 8.0e-5,
                    "E_XR": 0.03,
                }
            )
        for regime in ("C1-L", "C1-H"):
            rows.append(
                {
                    "domain": "mode_a",
                    "seed": seed,
                    "regime": regime,
                    "script": None,
                    "scale": math.nan,
                    "amp_scale": 1.5,
                    "D0": 2.0e-3,
                    "E_XR": 0.55,
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
                    "D0": 2.0e-4,
                    "E_XR": 0.50,
                }
            )
    summary = _aggregate(rows, R1RS3A1Config())
    assert summary["h1a_c0_near_zero"]["pass"] is True
    assert summary["h1b_shared_threshold"]["pass"] is True
    assert summary["h2_evidence_gap"]["pass"] is True
    assert summary["rs3a1_go"] is True
    assert summary["rewrites_rs3a_go"] is False
    assert summary["unlocks_rs3b"] is False
    assert summary["last_scalar_attempt"] is True
    contact_c0 = [row for row in rows if row["domain"] == "contact" and row["regime"] == "C0"]
    for row in contact_c0:
        row["E_XR"] = 0.40
    failed = _aggregate(rows, R1RS3A1Config())
    assert failed["h1a_c0_near_zero"]["pass"] is False
    assert failed["rs3a1_go"] is False


def test_require_rs3a_rejects_smoke(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"smoke": True, "scientific_result": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific"):
        _require_rs3a(path)

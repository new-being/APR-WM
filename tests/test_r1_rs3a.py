"""Unit tests for RS3A transport-calibrated evidence (no policy)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r1_rs2 import FROZEN_ADAPTER_SCALE
from aprwm_v0.r1_rs2a import CONTACT_JOBS, PRIMARY_CONTACT
from aprwm_v0.r1_rs3a import (
    MODE_A_AMPS,
    REGIMES,
    SEEDS,
    R1RS3AConfig,
    _aggregate,
    _require_rs2b,
    structural_surprisal,
)


def test_rs3a_matrix_excludes_c2_and_keeps_rs2_frozen():
    cfg = R1RS3AConfig()
    assert cfg.seeds == SEEDS
    assert REGIMES == ("C0", "C1-L", "C1-H")
    assert "C2" not in REGIMES
    assert MODE_A_AMPS == (1.0, 1.5)
    assert PRIMARY_CONTACT == ("pull_release", FROZEN_ADAPTER_SCALE)
    n_mode = len(cfg.seeds) * len(cfg.regimes) * len(cfg.mode_a_amps)
    n_contact = len(cfg.seeds) * len(cfg.regimes) * len(CONTACT_JOBS)
    assert n_mode == 30
    assert n_contact == 60
    assert n_mode + n_contact == 90


def test_chi2_sf_matches_known_values():
    from aprwm_v0.r1_rs3a import _chi2_sf

    # χ²_2 survival: P(T≥t) = exp(-t/2)
    assert abs(_chi2_sf(2.0, 2.0) - math.exp(-1.0)) < 1.0e-8
    assert _chi2_sf(0.0, 10.0) == 1.0
    assert _chi2_sf(1.0e3, 2.0) < 1.0e-10


def test_structural_surprisal_is_scale_free_under_h0():
    rng = np.random.default_rng(0)
    r_h0 = rng.normal(scale=3.0e-4, size=2000)
    r_pr = rng.normal(scale=3.0e-4, size=200)
    scores = structural_surprisal(r_h0_perp=r_h0, r_probe_perp=r_pr)
    assert scores["S_perp"] < 10.0
    assert scores["p_perp"] > 1.0e-4
    loud = structural_surprisal(r_h0_perp=r_h0, r_probe_perp=r_pr * 40.0)
    assert loud["S_perp"] > scores["S_perp"]
    assert loud["D0"] > scores["D0"]


def test_aggregate_h1_h2_do_not_rewrite_rs2_or_unlock_rs3b():
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
                    "S_perp": 1.0 + 0.1 * amp,
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
                    "S_perp": 0.8,
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
                    "S_perp": 12.0,
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
                    "S_perp": 11.0,
                }
            )
    summary = _aggregate(rows, R1RS3AConfig())
    assert summary["h1_null_calibration"]["pass"] is True
    assert summary["h1_null_calibration"]["fpr_mode_a"] == 0.0
    assert summary["h2_evidence_gap"]["pass"] is True
    assert summary["rs3a_go"] is True
    assert summary["rewrites_rs2_go"] is False
    assert summary["unlocks_rs3b"] is False
    assert summary["unlocks_domain_threshold"] is False
    assert summary["uses_true_operator"] is False
    assert summary["uses_domain_id"] is False
    contact_c0 = [
        row for row in rows if row["domain"] == "contact" and row["regime"] == "C0"
    ]
    contact_c0[0]["S_perp"] = 20.0
    contact_c0[1]["S_perp"] = 20.0
    failed = _aggregate(rows, R1RS3AConfig())
    assert failed["h1_null_calibration"]["fpr_contact"] > 0.05
    assert failed["rs3a_go"] is False


def test_require_rs2b_rejects_smoke(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"smoke": True, "scientific_result": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific"):
        _require_rs2b(path)
    missing = tmp_path / "absent.json"
    with pytest.raises(RuntimeError, match="locked"):
        _require_rs2b(missing)

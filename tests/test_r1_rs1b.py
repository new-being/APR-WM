"""Unit tests for R1-RS1B intake and long-horizon gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.r1_rs1b import (
    ALPHAS,
    AMP_SCALES,
    FORMAL_SEEDS,
    R1RS1BConfig,
    _aggregate,
    _operator_value,
    _outside_fit_support,
    _require_rs1a5_unlock,
    _revision_force,
)


def test_formal_matrix_and_frozen_population():
    assert len(FORMAL_SEEDS) * len(ALPHAS) * len(AMP_SCALES) == 30
    assert ALPHAS == (-0.12, -0.18, -0.24)
    assert AMP_SCALES == (1.5, 2.0)


def test_rs1a5_unlock_requires_passing_frozen_policy(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked"):
        _require_rs1a5_unlock(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rs1a5_go": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rs1a5_go"):
        _require_rs1a5_unlock(bad)

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "rs1a5_go": True,
                "aggregate": {
                    "gates": {"C_tol": 0.001},
                    "cells": {"1.5": {"threshold": 0.002}},
                },
            }
        ),
        encoding="utf-8",
    )
    payload = _require_rs1a5_unlock(good)
    assert payload["aggregate"]["gates"]["C_tol"] == 0.001


def test_drag_revision_is_dissipative():
    force = _revision_force(-0.24, 4)
    velocities = np.asarray([-0.5, -0.2, 0.2, 0.5])
    forces = np.asarray([force(0.2, float(v)) for v in velocities])
    assert np.all(velocities * forces <= 0.0)
    assert np.allclose(
        _operator_value(4, np.full_like(velocities, 0.2), velocities),
        velocities * np.abs(velocities),
    )


def test_fit_support_exit_is_typed():
    box = {"q_min": 0.1, "q_max": 0.3, "v_min": -0.4, "v_max": 0.4}
    outside = _outside_fit_support(
        np.asarray([0.2, 0.31, 0.2]),
        np.asarray([0.0, 0.0, -0.41]),
        box,
    )
    assert outside.tolist() == [False, True, True]


def test_aggregate_go_and_passivity_regression():
    cfg = R1RS1BConfig(seeds=(1, 2))
    rows = []
    for seed in cfg.seeds:
        rows.append(
            {
                "seed": seed,
                "revise_worthy": True,
                "pipeline_accepted": True,
                "accepted": True,
                "passivity_violation": False,
                "stable_h32": True,
                "gain_h32": 0.02,
                "expansion_excess_max": -0.1,
            }
        )
    summary = _aggregate(rows, cfg)
    assert summary["rs1b_go"] is True
    rows[0]["passivity_violation"] = True
    failed = _aggregate(rows, cfg)
    assert failed["gates"]["passivity_violations"]["pass"] is False
    assert failed["rs1b_go"] is False

"""RTWX-O0HYB0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.relative_validity import (
    calibrate_threshold_youden,
    symmetric_support_score,
)
from aprwm_v0.rtwx_o0hyb0 import (
    CAL_SEED,
    FORMAL_SEED,
    RTWXO0HYB0Config,
    _oracle_pick,
    run_rtwx_o0hyb0,
)


def _cloud(rng, n=80, c=None):
    c = np.zeros(3) if c is None else np.asarray(c, float)
    return rng.normal(0, 0.01, (n, 3)) + c


def test_cli():
    assert build_parser().parse_args(["rtwx-o0hyb0"]).command == "rtwx-o0hyb0"


def test_t1_coincident_support_near_one():
    rng = np.random.default_rng(0)
    p = _cloud(rng)
    q = symmetric_support_score(p, p.copy(), np.eye(3), np.zeros(3), 0.44)
    assert q > 0.9


def test_t2_disjoint_support_near_zero():
    rng = np.random.default_rng(1)
    q = symmetric_support_score(_cloud(rng), _cloud(rng, c=[1, 1, 1]), np.eye(3), np.zeros(3), 0.44)
    assert q < 0.05


def test_t6_youden_max_tau_tiebreak():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.array([False, False, True, True])
    tau = calibrate_threshold_youden(scores, labels)
    assert tau == 0.6


def test_t7_oracle_pick():
    assert _oracle_pick(True, False, 5.0, 20.0) == "rel"
    assert _oracle_pick(False, True, 20.0, 5.0) == "abs"


def test_frozen_seeds():
    assert CAL_SEED == 37607
    assert FORMAL_SEED == 37608


def test_smoke(tmp_path: Path):
    out = run_rtwx_o0hyb0(
        tmp_path / "hyb0",
        config=RTWXO0HYB0Config(
            smoke=True,
            natural_cache="/root/APR-WM/runs/rtwx_o0e0",
            p0_cache="/root/APR-WM/runs/rtwx_o0e0p0",
            r5_run="/root/APR-WM/runs/rtwx_o0e0r5",
        ),
    )
    assert out["pattern"] in {
        "prior_anchor_failure",
        "hybrid_complementarity_insufficient",
        "hybrid_router_failure",
        "hybrid_state_update_insufficient",
        "hybrid_state_update_supported",
    }
    assert "methods_table" in out

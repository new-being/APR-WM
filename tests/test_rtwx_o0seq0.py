"""RTWX-O0SEQ0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.relative_rigid_registration import RelativeICPConfig
from aprwm_v0.geometry.relative_state_chain import ChainState, propagate_state, run_adjacent_chain, run_direct_anchor
from aprwm_v0.rtwx_o0seq0 import (
    CHECKPOINTS,
    SEQ_SEED,
    RTWXO0SEQ0Config,
    _largest_h_star,
    run_rtwx_o0seq0,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0seq0"]).command == "rtwx-o0seq0"


def test_identity_chain_16_steps():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.01, (64, 3))
    clouds = [p.copy() for _ in range(17)]
    n0 = np.array([0.0, 1.0, 0.0])
    p0 = np.zeros(3)
    ck, steps = run_adjacent_chain(clouds, p0, n0, 0.44, cfg=RelativeICPConfig(), checkpoints=CHECKPOINTS)
    assert ck[1].n[1] > 0.99
    assert np.linalg.norm(ck[16].p) < 0.02


def test_axis_normalized():
    st = ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    r = np.array([[0.866, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.866]])
    st2 = propagate_state(st, r, np.zeros(3))
    assert abs(np.linalg.norm(st2.n) - 1.0) < 1e-9


def test_h_star_contiguous():
    assert _largest_h_star({1: True, 2: True, 4: True, 8: False, 16: False}) == 4
    assert _largest_h_star({1: True, 2: False, 4: True}) == 1


def test_direct_uses_frame0():
    rng = np.random.default_rng(1)
    c0 = rng.normal(0, 0.01, (80, 3))
    c1 = (c0 + np.array([0.02, 0, 0])).copy()
    p, n, res = run_direct_anchor(c0, c1, np.zeros(3), np.array([0, 1, 0.0]), 0.44)
    assert res.valid
    assert np.linalg.norm(p) > 0.01


def test_frozen_seed():
    assert SEQ_SEED == 37609


def test_smoke(tmp_path: Path):
    out = run_rtwx_o0seq0(
        tmp_path / "seq0",
        config=RTWXO0SEQ0Config(
            smoke=True,
            natural_cache="/root/APR-WM/runs/rtwx_o0e0",
            p0_cache="/root/APR-WM/runs/rtwx_o0e0p0",
            r5_run="/root/APR-WM/runs/rtwx_o0e0r5",
        ),
    )
    assert "H_star" in out
    assert out["pattern"] in {
        "sequence_observation_failure",
        "sequence_anchor_failure",
        "relative_sequence_insufficient",
        "relative_sequence_supported",
    }

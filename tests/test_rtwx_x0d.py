"""RTWX-X0D tests (numpy closed-loop plant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0d import RTWX0DConfig, run_rtwx_x0d


def test_cli_has_rtwx_x0d():
    args = build_parser().parse_args(["rtwx-x0d", "--output", "runs/rtwx_x0d"])
    assert args.command == "rtwx-x0d"


def test_x0d_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0d(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0DConfig(backend="numpy", n_train_ep=2, n_val_ep=2, n_test_ep=2, n_steps=24),
        )


def test_numpy_closed_loop_structure(tmp_path: Path):
    out = tmp_path / "x0d"
    s = run_rtwx_x0d(
        out,
        config=RTWX0DConfig(
            backend="numpy",
            n_train_ep=6,
            n_val_ep=4,
            n_test_ep=4,
            n_steps=80,
            seed=5,
        ),
    )
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "E1_identity_min=0.02" in header
    assert "capacity_claim=false" in header
    assert s["capacity_claim"] is False
    assert s["neural"] is False
    assert s["G0"] is True
    assert s["G1"] is True
    assert s["G2"] is True
    assert s["G3"] is True
    assert s["pattern"] == "closed_loop_structure_supported"
    assert s["rtwx_x0d_passed"] is True
    assert s["test"]["M0"]["E_1"] >= 0.02
    assert s["test"]["M1"]["E_1"] <= 0.75 * s["test"]["M0"]["E_1"]
    assert s["not_official_robotwin_observation_benchmark"] is True
    assert s["unlocks_x0d1"] is True

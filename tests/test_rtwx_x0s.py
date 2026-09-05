"""RTWX-X0S tests (numpy plant; no official RoboTwin required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0s import DELTA_QDD, RTWX0SConfig, run_rtwx_x0s


def test_cli_has_rtwx_x0s():
    parser = build_parser()
    args = parser.parse_args(["rtwx-x0s", "--output", "runs/rtwx_x0s"])
    assert args.command == "rtwx-x0s"


def test_x0s_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0s(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0SConfig(backend="numpy", n_train_ep=2, n_test_ep=2, n_steps=20),
        )


def test_numpy_m2_plant_beats_identity(tmp_path: Path):
    out = tmp_path / "x0s"
    summary = run_rtwx_x0s(
        out,
        config=RTWX0SConfig(backend="numpy", n_train_ep=8, n_test_ep=4, n_steps=40, seed=3),
    )
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "delta_qdd=0.05" in header
    assert "capacity_claim=false" in header
    assert summary["capacity_claim"] is False
    assert summary["G0"] is True
    assert summary["G1"] is True
    assert summary["E_qdd_identity"] > summary["families"][-1]["E_qdd"] + DELTA_QDD * 0.5
    assert summary["rtwx_x0s_passed"] == (summary["pattern"] == "structure_closed")

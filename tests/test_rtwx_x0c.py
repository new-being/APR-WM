"""RTWX-X0C tests (numpy PD+rigid-body plant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0c import RTWX0CConfig, run_rtwx_x0c


def test_cli_has_rtwx_x0c():
    args = build_parser().parse_args(["rtwx-x0c", "--output", "runs/rtwx_x0c"])
    assert args.command == "rtwx-x0c"
    smoke = build_parser().parse_args(["rtwx-x0c", "--smoke", "--backend", "numpy"])
    assert smoke.smoke is True


def test_x0c_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0c(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0CConfig(backend="numpy", n_train_ep=2, n_val_ep=2, n_test_ep=2, n_steps=24),
        )


def test_numpy_pd_plant_passes_g1_g2(tmp_path: Path):
    out = tmp_path / "x0c"
    s = run_rtwx_x0c(
        out,
        config=RTWX0CConfig(
            backend="numpy",
            n_train_ep=6,
            n_val_ep=4,
            n_test_ep=4,
            n_steps=80,
            seed=3,
        ),
    )
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "E_qdd_max=0.75" in header
    assert "capacity_claim=false" in header
    assert (out / "run.json").is_file()
    assert (out / "metrics.json").is_file()
    assert s["capacity_claim"] is False
    assert s["neural"] is False
    assert s["G0"] is True
    assert s["G1"] is True
    assert s["G2"] is True
    assert s["pattern"] == "closed_loop_structure_closed"
    assert s["rtwx_x0c_passed"] is True
    assert s["test"]["M0"]["E_qdd"] == 1.0
    assert s["test"]["M2"]["E_qdd"] <= 0.75
    assert s["test"]["M3"]["E_qdd"] < s["test"]["M0"]["E_qdd"]
    assert s["test"]["M3"]["E_roll10"] < s["test"]["M0"]["E_roll10"]
    assert s["drive_mode"] == "force"
    assert s["not_official_robotwin_observation_benchmark"] is True

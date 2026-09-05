"""RTWX-X0F tests (numpy plant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0f import RTWX0FConfig, run_rtwx_x0f


def test_cli_has_rtwx_x0f():
    args = build_parser().parse_args(["rtwx-x0f", "--output", "runs/rtwx_x0f"])
    assert args.command == "rtwx-x0f"


def test_x0f_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0f(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0FConfig(backend="numpy", n_train_ep=2, n_test_ep=2, n_steps=24))


def test_numpy_channel_reproduced(tmp_path: Path):
    s = run_rtwx_x0f(
        tmp_path / "x0f",
        config=RTWX0FConfig(backend="numpy", n_train_ep=4, n_test_ep=2, n_steps=40, seed=1),
    )
    assert (tmp_path / "x0f" / "run.header.txt").read_text(encoding="utf-8").find("rho_min=0.3") >= 0
    assert s["G0"] is True
    assert s["pattern"] == "channel_reproduced"
    assert s["train_best"]["min_abs_corr"] >= 0.3
    assert s["heldout_same_triple"]["min_abs_corr"] >= 0.3
    assert s["capacity_claim"] is False
    assert s["phi_fit"] is False

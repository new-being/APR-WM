"""RTWX-X0E tests (numpy linear increment plant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0e import RTWX0EConfig, run_rtwx_x0e


def test_cli_has_rtwx_x0e():
    args = build_parser().parse_args(["rtwx-x0e", "--output", "runs/rtwx_x0e"])
    assert args.command == "rtwx-x0e"


def test_x0e_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0e(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0EConfig(backend="numpy", n_train_ep=2, n_val_ep=2, n_test_ep=2, n_steps=24),
        )


def test_numpy_direct_increment(tmp_path: Path):
    out = tmp_path / "x0e"
    s = run_rtwx_x0e(
        out,
        config=RTWX0EConfig(
            backend="numpy",
            n_train_ep=6,
            n_val_ep=4,
            n_test_ep=4,
            n_steps=80,
            seed=7,
        ),
    )
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "E1_identity_min=0.02" in header
    assert "capacity_claim=false" in header
    assert "no_euler" in header
    assert s["capacity_claim"] is False
    assert s["neural"] is False
    assert s["not_x0d_patch"] is True
    assert s["G0"] is True
    assert s["G1"] is True
    assert s["G2"] is True
    assert s["G3"] is True
    assert s["pattern"] == "native_window_increment_supported"
    assert s["rtwx_x0e_passed"] is True
    assert s["test"]["M0"]["E_1"] >= 0.02
    assert s["test"]["M1"]["E_1"] <= 0.75 * s["test"]["M0"]["E_1"]
    assert s["unlocks_x0e1"] is True

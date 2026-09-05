"""RTWX-X0E1 tests (numpy increment plant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0e1 import RTWX0E1Config, run_rtwx_x0e1


def test_cli_has_rtwx_x0e1():
    args = build_parser().parse_args(["rtwx-x0e1", "--output", "runs/rtwx_x0e1"])
    assert args.command == "rtwx-x0e1"


def test_x0e1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0e1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0E1Config(backend="numpy", n_train_ep=2, n_val_ep=2, n_test_ep=2, n_steps=16, smoke=True),
        )


def test_numpy_capacity_grid(tmp_path: Path):
    out = tmp_path / "x0e1"
    s = run_rtwx_x0e1(
        out,
        config=RTWX0E1Config(
            backend="numpy",
            smoke=True,
            n_train_ep=5,
            n_val_ep=3,
            n_test_ep=3,
            n_steps=80,
            seed=7,
            pure_widths=(16, 32),
            res_widths=(4, 8),
            train_seeds=(201,),
            epochs=40,
        ),
    )
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "no_x0e_retune=true" in header
    assert s["not_x0e_retune"] is True
    assert s["unlocks_r10"] is False
    assert s["B1"]["robust"] is True
    assert s["B1"]["metrics"]["test"]["E_1"] < 0.05 * s["M0"]["test"]["E_1"]
    assert s["pattern"] in {
        "capacity_substitution_supported",
        "structure_not_in_robust_set",
        "reference_failure",
        "neural_no_match",
        "instrument_failure",
    }
    assert (out / "cache_splits.npz").is_file()
    assert (out / "metrics.json").is_file()

"""RTWX-O0D2 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0d2 import FOV_MIN, LUT_MAX, SEED, RTWXO0D2Config, run_rtwx_o0d2


def test_cli_has_rtwx_o0d2():
    args = build_parser().parse_args(["rtwx-o0d2", "--output", "runs/rtwx_o0d2"])
    assert args.command == "rtwx-o0d2"


def test_o0d2_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0d2(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0D2Config(backend="numpy", smoke=True, epoch_lut=2, epoch_mlp=2, epoch_cnn=2, n_mem=8))


def test_frozen():
    assert SEED == 19601
    assert LUT_MAX == 1.0e-3
    assert FOV_MIN == 0.90


def test_numpy_lookup_and_mlp(tmp_path: Path):
    out = tmp_path / "d2"
    result = run_rtwx_o0d2(
        out,
        config=RTWXO0D2Config(
            backend="numpy",
            smoke=True,
            n_train_ep=2,
            n_steps=4,
            seed=19,
            n_mem=16,
            epoch_lut=400,
            epoch_mlp=800,
            epoch_cnn=40,
        ),
    )
    assert result["A_lookup"]["ok"] is True
    assert result["B_flatten_mlp"]["ok"] is True
    assert result["FOV"]["ok"] is True
    assert result["pattern"] == "instrument_closed"
    assert result["does_not_run_o0r"] is True

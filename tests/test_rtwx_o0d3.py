"""RTWX-O0D3 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0d3 import CAMERA, G1_MAX, SEED, RTWXO0D3Config, run_rtwx_o0d3


def test_cli_has_rtwx_o0d3():
    args = build_parser().parse_args(["rtwx-o0d3", "--output", "runs/rtwx_o0d3"])
    assert args.command == "rtwx-o0d3"


def test_o0d3_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0d3(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0D3Config(backend="numpy", smoke=True, n_train_ep=1, n_steps=8, n_scalar=8, epoch_s=2, epoch_p=2, epoch_r=2),
        )


def test_frozen():
    assert CAMERA == "head_camera"
    assert SEED == 20601
    assert G1_MAX == 0.05


def test_numpy_qualifies(tmp_path: Path):
    result = run_rtwx_o0d3(
        tmp_path / "d3",
        config=RTWXO0D3Config(
            backend="numpy",
            smoke=True,
            n_train_ep=2,
            n_steps=16,
            seed=21,
            rgb_size=64,
            n_scalar=16,
            epoch_s=80,
            epoch_p=50,
            epoch_r=40,
        ),
    )
    assert result["G0"]["ok"] is True
    assert result["G1"]["ok"] is True
    assert result["does_not_run_o0r"] is True
    assert result["pattern"] in {
        "coverage_failure",
        "scalar_spatial_failure",
        "capacity_control_failure",
        "position_memorization_failure",
        "rotation_memorization_failure",
        "rotation_symmetry_limited",
        "instrument_qualified",
    }
    assert result["G1"]["ok"] is True
    assert result["G2"]["ok"] is True
    assert result["G3"]["ok"] is True
    assert result["pattern"] in {"instrument_qualified", "rotation_symmetry_limited", "rotation_memorization_failure"}

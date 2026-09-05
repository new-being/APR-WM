"""RTWX-O0R tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0r import CAMERA, RGB_SIZE, SEED, RTWXO0RConfig, run_rtwx_o0r


def test_cli_has_rtwx_o0r():
    args = build_parser().parse_args(["rtwx-o0r", "--output", "runs/rtwx_o0r"])
    assert args.command == "rtwx-o0r"


def test_o0r_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0r(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0RConfig(backend="numpy", smoke=True, n_train_ep=1, n_val_ep=1, n_test_ep=1, n_steps=8, vis_epochs=2),
        )


def test_frozen_instrument():
    assert CAMERA == "head_camera"
    assert RGB_SIZE == 64
    assert SEED == 21601


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0r(
        tmp_path / "o0r",
        config=RTWXO0RConfig(
            backend="numpy",
            smoke=True,
            n_train_ep=3,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=16,
            seed=23,
            rgb_size=64,
            vis_epochs=8,
        ),
    )
    assert result["G0"]["ok"] is True
    assert result["header"]["camera"] == "head_camera"
    assert result["pattern"] in {
        "coverage_failure",
        "object_pose_failure",
        "orientation_symmetry_limited",
        "pose_supported_velocity_failed",
        "explicit_object_pose_supported",
    }
    assert result["symmetry_secondary"]["does_not_override_primary"] is True

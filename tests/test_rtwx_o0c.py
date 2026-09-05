"""RTWX-O0C tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0c import SEEDS, RTWXO0CConfig, run_rtwx_o0c
from aprwm_v0.rtwx_o0v import CAM_HEAD, CAM_OBS


def test_cli_has_rtwx_o0c():
    args = build_parser().parse_args(["rtwx-o0c", "--output", "runs/rtwx_o0c"])
    assert args.command == "rtwx-o0c"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0c(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0CConfig(
                backend="numpy",
                smoke=True,
                n_train_ep=1,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=8,
                seeds=(41,),
                epochs_unet=2,
                epochs_ori=2,
            ),
        )


def test_frozen():
    assert SEEDS == (27601, 27602, 27603)
    assert CAM_HEAD == "head_camera"
    assert CAM_OBS == "observer_camera"


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0c(
        tmp_path / "o0c",
        config=RTWXO0CConfig(
            backend="numpy",
            smoke=True,
            n_train_ep=2,
            n_val_ep=1,
            n_test_ep=1,
            n_steps=12,
            seeds=(41, 42),
            epochs_unet=6,
            epochs_ori=10,
        ),
    )
    assert result["R_BO_is_GT_nuisance"] is False
    assert result["pattern"] in {
        "coverage_failure",
        "orientation_failure",
        "orientation_position_interface_failure",
        "composite_pose_failure",
        "composite_object_pose_supported",
    }
    assert "orientation_to_position_propagation" in result
    assert result["G0"]["ok"] is True

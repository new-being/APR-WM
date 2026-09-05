"""RTWX-O0G4 G0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g4 import KEYPOINT_NAMES, SEEDS, RTWXO0G4Config, load_cad_keypoints, run_rtwx_o0g4


def test_cli_has_rtwx_o0g4():
    assert build_parser().parse_args(["rtwx-o0g4", "--output", "runs/rtwx_o0g4"]).command == "rtwx-o0g4"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g4(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G4Config(backend="numpy", smoke=True, seeds=(41,)),
        )


def test_frozen_keypoints():
    assert SEEDS == (29601, 29602, 29603)
    assert len(KEYPOINT_NAMES) == 4
    kp = load_cad_keypoints("/root/RoboTwin")
    assert kp["K_O"].shape == (4, 3)


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g4(
        tmp_path / "g4",
        config=RTWXO0G4Config(backend="numpy", smoke=True, seeds=(41, 42)),
    )
    assert result["unlocks_o1"] is False
    assert result["pattern"] in {
        "sparse_keypoints_unobservable",
        "sparse_geometry_insufficient",
        "sparse_oracle_orientation_supported",
    }
    assert result["G0_orientation"]["ok"] is True

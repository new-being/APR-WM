"""RTWX-O0G2 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g2 import CAMERA, RGB_SIZE, SEED, RTWXO0G2Config, run_rtwx_o0g2


def test_cli_has_rtwx_o0g2():
    args = build_parser().parse_args(["rtwx-o0g2", "--output", "runs/rtwx_o0g2"])
    assert args.command == "rtwx-o0g2"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g2(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G2Config(backend="numpy", smoke=True, n_train_ep=1, n_val_ep=1, n_test_ep=1, n_steps=8, epochs=2),
        )


def test_frozen():
    assert CAMERA == "head_camera"
    assert RGB_SIZE == 64
    assert SEED == 24601


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g2(
        tmp_path / "g2",
        config=RTWXO0G2Config(
            backend="numpy",
            smoke=True,
            n_train_ep=3,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=16,
            seed=37,
            rgb_size=64,
            epochs=10,
        ),
    )
    assert result["no_rgb_to_p_regression"] is True
    assert result["unlocks_o1"] is False
    assert result["G0"]["ok"] is True
    assert result["pattern"] in {
        "coverage_failure",
        "oracle_geometry_failure",
        "learned_localization_failure",
        "localization_geometry_failure",
        "geometry_mediated_position_supported",
    }
    # synthetic blobs should allow oracle + learned path to work
    assert result["G1_oracle"]["ok"] is True
    assert result["G2_localization"]["ok"] is True

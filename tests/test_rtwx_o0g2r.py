"""RTWX-O0G2R tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g2r import SEED, RTWXO0G2RConfig, run_rtwx_o0g2r
from aprwm_v0.rtwx_o0v import CAM_HEAD, CAM_OBS


def test_cli_has_rtwx_o0g2r():
    args = build_parser().parse_args(["rtwx-o0g2r", "--output", "runs/rtwx_o0g2r"])
    assert args.command == "rtwx-o0g2r"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g2r(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G2RConfig(backend="numpy", smoke=True, n_train_ep=1, n_val_ep=1, n_test_ep=1, n_steps=8, epochs=2),
        )


def test_frozen():
    assert SEED == 26601
    assert CAM_HEAD == "head_camera"
    assert CAM_OBS == "observer_camera"


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g2r(
        tmp_path / "g2r",
        config=RTWXO0G2RConfig(
            backend="numpy",
            smoke=True,
            n_train_ep=3,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=16,
            seed=41,
            epochs=8,
        ),
    )
    assert result["R_BO_is_GT_nuisance"] is True
    assert result["unlocks_o1"] is False
    assert result["G0"]["ok"] is True
    assert result["pattern"] in {
        "dual_view_coverage_failure",
        "oracle_fusion_geometry_failure",
        "dual_view_localization_failure",
        "dual_view_position_failure",
        "dual_view_geometry_position_supported",
    }
    assert result["G1_oracle_fusion"]["ok"] is True
    assert result["G2_localization"]["ok"] is True

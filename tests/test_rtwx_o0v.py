"""RTWX-O0V tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, SEEDS, RTWXO0VConfig, run_rtwx_o0v


def test_cli_has_rtwx_o0v():
    args = build_parser().parse_args(["rtwx-o0v", "--output", "runs/rtwx_o0v"])
    assert args.command == "rtwx-o0v"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0v(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0VConfig(backend="numpy", smoke=True, seeds=(1,), n_ep=1, n_steps=8),
        )


def test_frozen():
    assert CAM_HEAD == "head_camera"
    assert CAM_OBS == "observer_camera"
    assert SEEDS == (25601, 25602, 25603)
    assert P_ANY_AGG == 0.98


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0v(
        tmp_path / "o0v",
        config=RTWXO0VConfig(backend="numpy", smoke=True, seeds=(11, 12, 13), n_ep=2, n_steps=48),
    )
    assert result["no_training"] is True
    assert result["no_pose_claim"] is True
    assert result["unlocks_o1"] is False
    assert result["pattern"] in {"dual_view_coverage_supported", "dual_view_coverage_insufficient"}
    assert result["B1_head_or_observer"]["P_visible_any_agg"] >= result["B0_head_only"]["P_visible_agg"]

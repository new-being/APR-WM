"""RTWX-O0E0R0 instrument + smoke tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r0 import (
    RTWXO0E0R0Config,
    SEED,
    YAW_OCTANT_DIAG_BETA,
    _pattern,
    run_rtwx_o0e0r0,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r0"]).command == "rtwx-o0e0r0"
    args = build_parser().parse_args(["rtwx-o0e0r0", "--p0-cache", "runs/foo"])
    assert args.p0_cache == "runs/foo"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r0(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0E0R0Config(smoke=True, p0_cache=str(tmp_path / "p0")),
        )


def test_frozen_constants():
    assert SEED == 38601
    assert YAW_OCTANT_DIAG_BETA == 20.0


def test_patterns():
    assert _pattern(g0=False, g1=True, g2=True, b0_excluded=True) == "controlled_pose_detection_failure"
    assert _pattern(g0=True, g1=False, g2=True, b0_excluded=True) == "effective_axis_failure"
    assert _pattern(g0=True, g1=True, g2=False, b0_excluded=True) == "effective_position_failure"
    assert _pattern(g0=True, g1=True, g2=True, b0_excluded=False) == "constant_baseline_not_excluded"
    assert _pattern(g0=True, g1=True, g2=True, b0_excluded=True) == "effective_pose_supported"


def test_smoke_pipeline(tmp_path: Path):
    p0_out = tmp_path / "p0"
    r0_out = tmp_path / "r0"
    # numpy P0 with enough test frames for G0c yaw octant coverage
    p0 = run_rtwx_o0e0p0(
        p0_out,
        config=replace(RTWXO0E0P0Config(backend="numpy"), n_train=25, n_val=10, n_test=200),
    )
    assert p0["pattern"] == "controlled_pose_observation_qualified"
    r0 = run_rtwx_o0e0r0(
        r0_out,
        config=RTWXO0E0R0Config(smoke=True, p0_cache=str(p0_out)),
    )
    assert r0["pattern"] in {
        "controlled_pose_detection_failure",
        "effective_axis_failure",
        "effective_position_failure",
        "constant_baseline_not_excluded",
        "effective_pose_supported",
    }
    assert (r0_out / "summary.json").is_file()
    assert "mechanism_diagnostics" in r0 or r0["pattern"] == "controlled_pose_detection_failure"


def test_requires_p0_cache(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="O0E0P0 cache missing"):
        run_rtwx_o0e0r0(
            tmp_path / "r0",
            config=RTWXO0E0R0Config(smoke=True, p0_cache=str(tmp_path / "missing_p0")),
        )

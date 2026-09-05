"""RTWX-X0RGB tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0rgb import K_EXECUTE, L_HIST, RTWX0RGBConfig, run_rtwx_x0rgb


def test_cli_has_rtwx_x0rgb():
    args = build_parser().parse_args(["rtwx-x0rgb", "--output", "runs/rtwx_x0rgb"])
    assert args.command == "rtwx-x0rgb"


def test_x0rgb_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0rgb(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0RGBConfig(backend="numpy", smoke=True),
        )


def test_k_and_hist_frozen():
    assert K_EXECUTE == 4
    assert L_HIST == 2

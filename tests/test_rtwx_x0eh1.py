"""RTWX-X0EH1 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0eh1 import K_EXECUTE, RTWX0EH1Config, run_rtwx_x0eh1


def test_cli_has_rtwx_x0eh1():
    args = build_parser().parse_args(["rtwx-x0eh1", "--output", "runs/rtwx_x0eh1"])
    assert args.command == "rtwx-x0eh1"


def test_x0eh1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0eh1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0EH1Config(backend="numpy", smoke=True),
        )


def test_k_execute_frozen():
    assert K_EXECUTE == 4

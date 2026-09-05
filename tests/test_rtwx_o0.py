"""RTWX-O0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0 import E_P_MAX, L_HIST, RGB_SIZE, TASK, RTWXO0Config, run_rtwx_o0


def test_cli_has_rtwx_o0():
    args = build_parser().parse_args(["rtwx-o0", "--output", "runs/rtwx_o0"])
    assert args.command == "rtwx-o0"


def test_o0_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Config(backend="numpy", smoke=True))


def test_o0_frozen():
    assert TASK == "place_empty_cup"
    assert RGB_SIZE == 224
    assert L_HIST == 4
    assert E_P_MAX == 0.30

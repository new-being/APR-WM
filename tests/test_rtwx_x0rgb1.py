"""RTWX-X0RGB1 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0rgb1 import E_Q_MAX, E_QD_MAX, L_TEMP, RGB_SIZE, RTWX0RGB1Config, run_rtwx_x0rgb1


def test_cli_has_rtwx_x0rgb1():
    args = build_parser().parse_args(["rtwx-x0rgb1", "--output", "runs/rtwx_x0rgb1"])
    assert args.command == "rtwx-x0rgb1"


def test_x0rgb1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0rgb1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0RGB1Config(backend="numpy", smoke=True),
        )


def test_gates_and_l_frozen():
    assert E_Q_MAX == 0.35
    assert E_QD_MAX == 0.50
    assert L_TEMP == 4
    assert RGB_SIZE == 128

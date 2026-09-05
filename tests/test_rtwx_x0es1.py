"""RTWX-X0ES1 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0es1 import RTWX0ES1Config, run_rtwx_x0es1


def test_cli_has_rtwx_x0es1():
    args = build_parser().parse_args(["rtwx-x0es1", "--output", "runs/rtwx_x0es1"])
    assert args.command == "rtwx-x0es1"


def test_x0es1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0es1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0ES1Config(backend="numpy", smoke=True, n_ep=2, n_steps=60, old_cache=str(tmp_path / "no.npz")),
        )

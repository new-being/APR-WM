"""RTWX-X0EH tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0eh import RTWX0EHConfig, run_rtwx_x0eh


def test_cli_has_rtwx_x0eh():
    args = build_parser().parse_args(["rtwx-x0eh", "--output", "runs/rtwx_x0eh"])
    assert args.command == "rtwx-x0eh"


def test_x0eh_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0eh(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0EHConfig(backend="numpy", smoke=True, old_cache=str(tmp_path / "no.npz")),
        )

"""RTWX-X0E-S tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0es import RTWX0ESConfig, run_rtwx_x0es


def test_cli_has_rtwx_x0es():
    args = build_parser().parse_args(["rtwx-x0es", "--output", "runs/rtwx_x0es"])
    assert args.command == "rtwx-x0es"


def test_x0es_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0es(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0ESConfig(cache=str(tmp_path / "missing.npz")))


def test_x0es_requires_cache(tmp_path: Path):
    with pytest.raises(RuntimeError, match="requires cache"):
        run_rtwx_x0es(tmp_path / "x0es", config=RTWX0ESConfig(cache=str(tmp_path / "nope.npz")))

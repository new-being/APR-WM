"""RTWX-X0EH2 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0eh2 import HOLDOUTS, RTWX0EH2Config, run_rtwx_x0eh2


def test_cli_has_rtwx_x0eh2():
    args = build_parser().parse_args(["rtwx-x0eh2", "--output", "runs/rtwx_x0eh2"])
    assert args.command == "rtwx-x0eh2"


def test_x0eh2_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0eh2(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0EH2Config(backend="numpy", smoke=True),
        )


def test_holdouts_exclude_source():
    names = {t for t, _ in HOLDOUTS}
    assert "put_object_cabinet" not in names
    assert names == {"place_empty_cup", "stamp_seal"}

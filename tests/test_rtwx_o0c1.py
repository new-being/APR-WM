"""RTWX-O0C1 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0c1 import SEEDS, RTWXO0C1Config, run_rtwx_o0c1


def test_cli_has_rtwx_o0c1():
    args = build_parser().parse_args(["rtwx-o0c1", "--output", "runs/rtwx_o0c1"])
    assert args.command == "rtwx-o0c1"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0c1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0C1Config(smoke=True, seeds=(41,), epochs_ori=2, o0c_cache=str(tmp_path / "missing")),
        )


def test_frozen_seeds():
    assert SEEDS == (27601, 27602, 27603)


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0c1(
        tmp_path / "o0c1",
        config=RTWXO0C1Config(
            smoke=True,
            seeds=(41, 42),
            epochs_ori=6,
            o0c_cache=str(tmp_path / "no_cache"),
        ),
    )
    assert result["diagnostic_only"] is True
    assert result["unlocks_o1"] is False
    assert result["does_not_rewrite_o0c"] is True
    assert result["pattern"] in {
        "orientation_state_symmetry_mismatch",
        "orientation_fusion_failure",
        "dual_view_both_wrong_mode",
        "view_dependent_observability",
        "orientation_tail_inconclusive",
    }
    assert "D1_distribution" in result
    assert "oracle_view_selector" in result

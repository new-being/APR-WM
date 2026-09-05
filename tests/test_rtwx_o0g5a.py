"""RTWX-O0G5A smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g5a import SEEDS, RTWXO0G5AConfig, run_rtwx_o0g5a


def test_cli():
    assert build_parser().parse_args(["rtwx-o0g5a"]).command == "rtwx-o0g5a"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g5a(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G5AConfig(backend="numpy", smoke=True, seeds=(41,), n_test_ep=1, n_steps=8, rgb_size=16),
        )


def test_frozen_seeds():
    assert SEEDS == (31601, 31602, 31603)


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0g5a(
        tmp_path / "g5a",
        config=RTWXO0G5AConfig(backend="numpy", smoke=True, seeds=(41,), n_test_ep=2, n_steps=16, rgb_size=32),
    )
    assert r["pattern"] in {
        "cad_surface_support_failure",
        "local_descriptor_ambiguous",
        "cad_descriptor_observable",
    }
    assert r["unlocks_o1"] is False

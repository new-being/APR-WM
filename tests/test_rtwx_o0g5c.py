"""RTWX-O0G5C smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g5a import SEEDS as TRAIN_SEEDS
from aprwm_v0.rtwx_o0g5b import K_ANCHOR
from aprwm_v0.rtwx_o0g5c import (
    SEEDS_FRESH,
    RTWXO0G5CConfig,
    _cond_entropy,
    _pattern,
    run_rtwx_o0g5c,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0g5c"]).command == "rtwx-o0g5c"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g5c(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G5CConfig(smoke=True),
        )


def test_frozen():
    assert TRAIN_SEEDS == (31601, 31602, 31603)
    assert SEEDS_FRESH == (33601, 33602, 33603)
    assert K_ANCHOR == 512


def test_patterns():
    assert _pattern(g0=False, b1=True, b2=True) == "coverage_failure"
    assert _pattern(g0=True, b1=False, b2=False) == "canonical_identity_generalization_failure"
    assert _pattern(g0=True, b1=True, b2=False) == "local_identity_supported"
    assert _pattern(g0=True, b1=True, b2=True) == "both_supported"
    assert _pattern(g0=True, b1=False, b2=True) == "global_context_supported"


def test_cond_entropy_low_when_cell_pure():
    u = np.array([0.1, 0.1, 0.9, 0.9])
    v = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    h = _cond_entropy(u, v, y, bins=2, min_n=2)
    assert h["n_cells"] == 2
    assert h["H_Y_given_UV"] < 1e-6


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0g5c(tmp_path / "g5c", config=RTWXO0G5CConfig(smoke=True))
    assert r["unlocks_o1"] is False
    assert r["unlocks_o0c2"] is False
    assert r["pattern"] in {
        "coverage_failure",
        "canonical_identity_generalization_failure",
        "local_identity_supported",
        "both_supported",
        "global_context_supported",
    }
    if r["pattern"] in {"coverage_failure", "canonical_identity_generalization_failure"}:
        assert r["unlocks_o0g5r_prereg"] is False
    else:
        assert r["unlocks_o0g5r_prereg"] is True
    assert "B1_local_fresh" in r and "B2_global_fresh" in r
    assert "B1_uv_shuffle" in r
    assert "H_Y_given_UV" in r
    assert "mask_area_strata" in r
    assert r["header"]["no_ransac"] is True
    assert r["header"]["no_pose_rescue"] is True

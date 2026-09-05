"""RTWX-O0G6A smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g5c import SEEDS_FRESH
from aprwm_v0.rtwx_o0g6a import (
    EPS_BASIN_DEG,
    K_HYP,
    RTWXO0G6AConfig,
    _axis_angle,
    _build_generators,
    _geodesic_deg,
    _pattern,
    run_rtwx_o0g6a,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0g6a"]).command == "rtwx-o0g6a"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g6a(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0G6AConfig(smoke=True))


def test_frozen():
    assert SEEDS_FRESH == (33601, 33602, 33603)
    assert K_HYP == 72
    assert EPS_BASIN_DEG == 15.0


def test_patterns():
    assert _pattern(g0=False, g1=True, g2=True) == "surface_support_failure"
    assert _pattern(g0=True, g1=False, g2=True) == "global_geometry_ambiguous"
    assert _pattern(g0=True, g1=True, g2=False) == "global_geometry_ambiguous"
    assert _pattern(g0=True, g1=True, g2=True) == "global_shape_orientation_supported"


def test_geodesic_and_generators():
    I = np.eye(3)
    assert _geodesic_deg(I, I) < 1e-6
    assert abs(_geodesic_deg(I, _axis_angle(np.array([0.0, 1.0, 0.0]), 90.0)) - 90.0) < 1e-6
    gens, names = _build_generators()
    assert gens.shape == (72, 3, 3)
    assert names[0] == "I"
    assert "Ry90" in names
    basin = [_geodesic_deg(R, I) <= 15.0 for R in gens]
    assert basin[0]
    assert sum(basin) >= 1


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0g6a(tmp_path / "g6a", config=RTWXO0G6AConfig(smoke=True))
    assert r["unlocks_o1"] is False
    assert r["unlocks_o0g5r"] is False
    assert r["header"]["no_icp"] is True
    assert r["pattern"] in {
        "surface_support_failure",
        "global_geometry_ambiguous",
        "global_shape_orientation_supported",
    }
    if r["pattern"] != "global_shape_orientation_supported":
        assert r["unlocks_o0g6r_prereg"] is False
    else:
        assert r["unlocks_o0g6r_prereg"] is True
    assert "G1_top1" in r and "G2_margin" in r

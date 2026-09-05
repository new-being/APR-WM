"""RTWX-O0G1b tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g1b import (
    CAMERA,
    MED_MAX,
    SEED,
    STRONG_MED,
    RTWXO0G1BConfig,
    _align_pose,
    _quat_to_R,
    load_delta_O,
    run_rtwx_o0g1b,
)


def test_cli_has_rtwx_o0g1b():
    args = build_parser().parse_args(["rtwx-o0g1b", "--output", "runs/rtwx_o0g1b"])
    assert args.command == "rtwx-o0g1b"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g1b(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G1BConfig(backend="numpy", smoke=True, n_ep=1, n_steps=8),
        )


def test_frozen():
    assert CAMERA == "head_camera"
    assert SEED == 23601
    assert MED_MAX == 0.05
    assert STRONG_MED == 0.02


def test_delta_from_asset_not_test_bias():
    meta = load_delta_O("/root/RoboTwin")
    d = meta["delta_O"]
    assert meta["not_from_o0g_test_bias"] is True
    assert abs(float(d[1]) - 0.04475) < 1e-3  # mesh Y AABB center * scale
    # must NOT equal the O0G empirical +5.11cm world-z hack
    assert abs(float(d[1]) - 0.0511) > 1e-3


def test_object_frame_align_roundtrip():
    delta = np.array([0.0, 0.045, 0.0])
    quat = np.array([0.5, 0.5, 0.5, 0.5])
    p = np.array([[0.1, -0.05, 0.74]])
    p_surf = p + (_quat_to_R(quat) @ delta)
    p_hat = _align_pose(p_surf, quat.reshape(1, 4), delta)
    assert np.linalg.norm(p_hat - p) < 1e-12


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g1b(
        tmp_path / "g1b",
        config=RTWXO0G1BConfig(backend="numpy", smoke=True, n_ep=2, n_steps=16, seed=31),
    )
    b0, b1 = result["B0_surface_vs_pose"], result["B1_aligned_vs_pose"]
    assert b0["median_ep_m"] > 0.03  # ~||δ_O|| residual without alignment
    assert b1["ok_primary"] is True
    assert b1["median_ep_m"] < b0["median_ep_m"]
    assert result["pattern"] in {"geometry_reference_aligned", "geometry_usable_but_coarse"}
    assert result["unlocks_o1"] is False
    assert result["does_not_fit_o0g_test_bias"] is True
    assert result["unlocks_g2_prereg"] is True
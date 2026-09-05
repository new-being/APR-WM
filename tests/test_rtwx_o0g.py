"""RTWX-O0G tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0d1 import _project
from aprwm_v0.rtwx_o0g import (
    CAMERA,
    G0_EP_MAX,
    SEED,
    RTWXO0GConfig,
    _unproject_uvd,
    run_rtwx_o0g,
)


def test_cli_has_rtwx_o0g():
    args = build_parser().parse_args(["rtwx-o0g", "--output", "runs/rtwx_o0g", "--stage", "g0"])
    assert args.command == "rtwx-o0g"
    assert args.stage == "g0"


def test_o0g_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0GConfig(backend="numpy", smoke=True, stage="g0", n_ep=1, n_steps=8),
        )


def test_frozen():
    assert CAMERA == "head_camera"
    assert SEED == 22601
    assert G0_EP_MAX == 0.05


def test_unproject_roundtrip():
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    e = np.eye(4)
    e[:3, 3] = [0.1, -0.05, 0.9]
    p = np.array([0.12, -0.08, 0.74])
    pr = _project(k, e, p, 640, 480)
    phat = _unproject_uvd(k, e, pr["u"], pr["v"], pr["z"])
    assert np.linalg.norm(phat - p) < 1e-9


def test_numpy_g01_smoke(tmp_path: Path):
    result = run_rtwx_o0g(
        tmp_path / "o0g",
        config=RTWXO0GConfig(
            backend="numpy",
            smoke=True,
            stage="g01",
            n_ep=2,
            n_steps=16,
            seed=29,
        ),
    )
    assert result["G0"]["ok"] is True
    assert result["G1"]["ok"] is True
    assert result["unlocks_o1"] is False
    assert result["pauses_rgb_to_p"] is True
    assert result["pattern"] == "geometry_feasible_localization_pending"

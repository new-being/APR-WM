"""RTWX-O0G3 G0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g3 import SEEDS, RTWXO0G3Config, _kabsch, run_rtwx_o0g3
import numpy as np


def test_cli_has_rtwx_o0g3():
    args = build_parser().parse_args(["rtwx-o0g3", "--output", "runs/rtwx_o0g3"])
    assert args.command == "rtwx-o0g3"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g3(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G3Config(backend="numpy", smoke=True, seeds=(41,), n_ep=1, n_steps=8),
        )


def test_frozen():
    assert SEEDS == (28601, 28602, 28603)


def test_kabsch_recovers_identity():
    rng = np.random.default_rng(0)
    X_O = rng.normal(size=(100, 3))
    R = np.eye(3)
    t = np.array([0.1, -0.2, 0.3])
    X_B = (R @ X_O.T).T + t
    Rh, th, rms = _kabsch(X_O, X_B)
    assert rms < 1e-6
    assert np.allclose(Rh, R, atol=1e-6)
    assert np.allclose(th, t, atol=1e-6)


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g3(
        tmp_path / "g3",
        config=RTWXO0G3Config(backend="numpy", smoke=True, seeds=(41, 42), n_ep=2, n_steps=16),
    )
    assert result["unlocks_o1"] is False
    assert result["icp_not_primary"] is True
    assert result["pattern"] in {
        "oracle_orientation_geometry_supported",
        "orientation_geometry_insufficient",
    }
    assert result["G0_orientation"]["ok"] is True
    assert result["mode_mass"]["P_eR_in_75_105"] < 0.01

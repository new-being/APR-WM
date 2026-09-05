"""RTWX-O0G3R tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g3 import _kabsch
from aprwm_v0.rtwx_o0g3r import SEEDS, RTWXO0G3RConfig, run_rtwx_o0g3r


def test_cli_has_rtwx_o0g3r():
    args = build_parser().parse_args(["rtwx-o0g3r", "--output", "runs/rtwx_o0g3r"])
    assert args.command == "rtwx-o0g3r"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g3r(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G3RConfig(
                backend="numpy",
                smoke=True,
                seeds=(41,),
                n_train_ep=1,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=8,
                epochs_mask=2,
                epochs_corr=2,
                epochs_b0=2,
            ),
        )


def test_frozen_seeds():
    assert SEEDS == (29601, 29602, 29603)


def test_kabsch_roundtrip():
    rng = np.random.default_rng(0)
    X_O = rng.normal(size=(80, 3))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.1, -0.2, 0.3])
    X_B = (R @ X_O.T).T + t
    Rh, th, rms = _kabsch(X_O, X_B)
    assert rms < 1e-5


def test_numpy_smoke(tmp_path: Path):
    result = run_rtwx_o0g3r(
        tmp_path / "g3r",
        config=RTWXO0G3RConfig(
            backend="numpy",
            smoke=True,
            seeds=(41,),
            n_train_ep=3,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=16,
            epochs_mask=8,
            epochs_corr=10,
            epochs_b0=8,
        ),
    )
    assert result["unlocks_o1"] is False
    assert result["pattern"] in {
        "coverage_failure",
        "correspondence_instrument_failure",
        "geometry_mediated_orientation_failure",
        "geometry_mediated_orientation_supported",
    }
    assert "G2_corr_descriptive" in result
    assert "G4_mode_reduction" in result
    assert "B0_direct_RGB_to_R" in result

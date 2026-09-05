"""RTWX-O0D1 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0d1 import R1_MAX, SEED, RTWXO0D1Config, denorm_smoke, run_rtwx_o0d1


def test_cli_has_rtwx_o0d1():
    args = build_parser().parse_args(["rtwx-o0d1", "--output", "runs/rtwx_o0d1"])
    assert args.command == "rtwx-o0d1"


def test_o0d1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0d1(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0D1Config(backend="numpy", smoke=True))


def test_frozen_and_denorm():
    assert SEED == 18601
    assert R1_MAX == 0.05
    sm = denorm_smoke(np.array([2.0, -1.0]), np.array([0.5, 3.0]))
    assert sm["ok"] is True
    np.testing.assert_allclose(sm["hat_from_z1"], [2.5, 2.0])


def test_numpy_repair_smoke(tmp_path: Path):
    out = tmp_path / "o0d1"
    result = run_rtwx_o0d1(
        out,
        config=RTWXO0D1Config(
            backend="numpy",
            smoke=True,
            n_train_ep=2,
            n_steps=8,
            seed=17,
            rgb_size=32,
            n_scalar=16,
            n_pose=24,
            epoch_scalar=80,
            epoch_pose=50,
        ),
    )
    assert result["does_not_unlock_o1"] is True
    assert result["does_not_run_o0r"] is True
    assert result["norm_smoke"]["ok"] is True
    assert result["R1"]["ok"] is True
    assert result["R3"]["ok"] is True
    assert result["pattern"] in {
        "scalar_memorization_failure",
        "pose_memorization_failure",
        "visibility_contract_failure",
        "instrument_repair_supported",
    }
    assert result["R2"]["ok"] is True
    assert result["pattern"] == "instrument_repair_supported"

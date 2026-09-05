"""RTWX-O0E0R3 tests."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r3 import FRESH_TEST_SEED, RTWXO0E0R3Config, SEED, _pattern, run_rtwx_o0e0r3


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r3"]).command == "rtwx-o0e0r3"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r3(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0R3Config(smoke=True))


def test_patterns():
    assert _pattern(data_ok=False, instrument_ok=True, formal_ok=True) == "observation_support_failure"
    assert _pattern(data_ok=True, instrument_ok=False, formal_ok=False) == "segmentation_cloud_failure"
    assert _pattern(data_ok=True, instrument_ok=True, formal_ok=False) == "reference_coupled_failure"
    assert _pattern(data_ok=True, instrument_ok=True, formal_ok=True) == "effective_pose_supported"


def test_frozen():
    assert SEED == 41601
    assert FRESH_TEST_SEED == 37603
    assert FRESH_TEST_SEED != 37602


def test_smoke(tmp_path: Path):
    p0 = tmp_path / "p0"
    nat = tmp_path / "nat"
    run_rtwx_o0e0p0(p0, config=replace(RTWXO0E0P0Config(backend="numpy"), n_train=20, n_val=8, n_test=40))
    shutil.copytree(p0 / "cache", nat / "cache")
    for split in ("train", "val"):
        obs = np.load(p0 / "cache" / split / "obs.npz")
        np.savez_compressed(
            nat / "cache" / split / "seg_gt.npz",
            rgb_h=obs["rgb_h"], rgb_o=obs["rgb_o"], mask_h=obs["mask_h"], mask_o=obs["mask_o"],
        )
    r3 = run_rtwx_o0e0r3(
        tmp_path / "r3",
        config=RTWXO0E0R3Config(smoke=True, p0_cache=str(p0), natural_cache=str(nat), fresh_test_seed=37603),
    )
    assert r3["pattern"] in {
        "observation_support_failure", "segmentation_cloud_failure",
        "reference_coupled_failure", "effective_pose_supported",
    }

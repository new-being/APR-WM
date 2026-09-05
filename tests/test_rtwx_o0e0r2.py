"""RTWX-O0E0R2 tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r0 import RTWXO0E0R0Config, run_rtwx_o0e0r0
from aprwm_v0.rtwx_o0e0r2 import FRESH_TEST_SEED, RTWXO0E0R2Config, SEED, _pattern, run_rtwx_o0e0r2


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r2"]).command == "rtwx-o0e0r2"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r2(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0R2Config(smoke=True))


def test_patterns():
    assert _pattern(instrument_pass=False, formal_pass=False, training_insufficient=True)[0] == "segmentation_cloud_failure"
    assert _pattern(instrument_pass=True, formal_pass=True, training_insufficient=False)[0] == "effective_pose_supported"
    assert _pattern(instrument_pass=True, formal_pass=False, training_insufficient=False)[0] == "reference_coupled_failure"


def test_frozen():
    assert SEED == 40601
    assert FRESH_TEST_SEED == 37602


def test_smoke(tmp_path: Path):
    p0 = tmp_path / "p0"
    r0 = tmp_path / "r0"
    nat = tmp_path / "nat"
    run_rtwx_o0e0p0(p0, config=replace(RTWXO0E0P0Config(backend="numpy"), n_train=20, n_val=8, n_test=40))
    run_rtwx_o0e0r0(r0, config=RTWXO0E0R0Config(smoke=True, p0_cache=str(p0)))
    # minimal natural cache from p0 train renamed
    import shutil

    shutil.copytree(p0 / "cache", nat / "cache")
    for split in ("train", "val", "test"):
        d = nat / "cache" / split
        if (p0 / "cache" / split / "obs.npz").is_file():
            import numpy as np

            obs = np.load(p0 / "cache" / split / "obs.npz")
            np.savez_compressed(
                d / "seg_gt.npz",
                rgb_h=obs["rgb_h"], rgb_o=obs["rgb_o"],
                mask_h=obs["mask_h"], mask_o=obs["mask_o"],
            )
    out = tmp_path / "r2"
    r2 = run_rtwx_o0e0r2(
        out,
        config=RTWXO0E0R2Config(smoke=True, p0_cache=str(p0), natural_cache=str(nat)),
    )
    assert r2["pattern"] in {
        "segmentation_cloud_failure", "reference_coupled_failure", "effective_pose_supported"
    }
    assert (out / "cache" / "fresh_test" / "obs.npz").is_file()

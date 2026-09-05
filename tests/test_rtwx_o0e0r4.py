"""RTWX-O0E0R4 tests."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r4 import (
    K_VALUES,
    MV_TEST_SEED,
    R3_DIAG_SEED,
    RTWXO0E0R4Config,
    SEED,
    _pattern,
    fuse_multiview,
    run_rtwx_o0e0r4,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r4"]).command == "rtwx-o0e0r4"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r4(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0R4Config(smoke=True))


def test_frozen():
    assert SEED == 42601
    assert MV_TEST_SEED == 37604
    assert R3_DIAG_SEED == 37603
    assert MV_TEST_SEED != R3_DIAG_SEED
    assert K_VALUES == (1, 2, 4, 8)


def test_patterns():
    pat, tags = _pattern(any_k_pass=True, k8_g1=False, k1_g1=False)
    assert pat == "multiview_reference_supported"
    assert tags == []
    pat, tags = _pattern(any_k_pass=False, k8_g1=False, k1_g1=False)
    assert pat == "multiview_reference_insufficient"
    assert "multiview_reference_insufficient" in tags
    assert "single_observation_reference_coupled" in tags


def test_fuse_multiview():
    rng = np.random.default_rng(0)
    views = [np.random.randn(100, 3), np.random.randn(80, 3)]
    P = fuse_multiview(views, 2, rng=rng)
    assert P.shape[0] <= 256
    assert P.shape[1] == 3


def test_smoke(tmp_path: Path):
    p0 = tmp_path / "p0"
    nat = tmp_path / "nat"
    r3 = tmp_path / "r3"
    run_rtwx_o0e0p0(p0, config=replace(RTWXO0E0P0Config(backend="numpy"), n_train=20, n_val=8, n_test=40))
    shutil.copytree(p0 / "cache", nat / "cache")
    for split in ("train", "val"):
        obs = np.load(p0 / "cache" / split / "obs.npz")
        np.savez_compressed(
            nat / "cache" / split / "seg_gt.npz",
            rgb_h=obs["rgb_h"], rgb_o=obs["rgb_o"], mask_h=obs["mask_h"], mask_o=obs["mask_o"],
        )
    r4 = run_rtwx_o0e0r4(
        tmp_path / "r4",
        config=RTWXO0E0R4Config(
            smoke=True,
            p0_cache=str(p0),
            natural_cache=str(nat),
            r3_cache=str(r3),
        ),
    )
    assert r4["pattern"] in {"multiview_reference_supported", "multiview_reference_insufficient"}
    assert "center_lambda_diagnostic" in r4
    assert set(r4["multiview_formal"].keys()) == {"1", "2", "4", "8"}

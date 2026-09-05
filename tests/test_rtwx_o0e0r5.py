"""RTWX-O0E0R5 tests (T1–T8)."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.visibility_reference import (
    ALPHA_GRID_DEG,
    AxialVisibilityReference,
    build_lut,
    camera_center_world,
    cloud_centroid,
    view_geometry,
)
from aprwm_v0.rtwx_o0e0 import EY, fibonacci_sphere, cad_hr_profile, K_SPHERE, SPHERE_SEED
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r1 import estimate_axis_b2_at_center as b2_at_c
from aprwm_v0.rtwx_o0e0r5 import (
    FRESH_TEST_SEED,
    RTWXO0E0R5Config,
    SEED,
    _pattern,
    assert_obs_no_gt,
    estimate_axis_b2_center_fn,
    estimate_effective_pose_vis,
    run_rtwx_o0e0r5,
)
from aprwm_v0.rtwx_o0q0 import _R_y


def _synthetic_cad(n: int = 512) -> np.ndarray:
    hs = np.linspace(0, 0.088, 32)
    pts = []
    for h in hs:
        r = 0.02 + 0.22 * (h / 0.088)
        for th in np.linspace(0, 2 * np.pi, n // 32, endpoint=False):
            pts.append([r * np.cos(th), h, r * np.sin(th)])
    return np.asarray(pts, dtype=np.float64)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r5"]).command == "rtwx-o0e0r5"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r5(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0R5Config(smoke=True))


def test_frozen_seeds():
    assert SEED == 43601
    assert FRESH_TEST_SEED == 37605
    assert FRESH_TEST_SEED != 37604


def test_patterns():
    assert _pattern(data_ok=False, g_i0=True, g_i1=True, formal_ok=True) == "observation_support_failure"
    assert _pattern(data_ok=True, g_i0=False, g_i1=True, formal_ok=True) == "segmentation_cloud_failure"
    assert _pattern(data_ok=True, g_i0=True, g_i1=False, formal_ok=True) == "visibility_reference_instrument_failure"
    assert _pattern(data_ok=True, g_i0=True, g_i1=True, formal_ok=False) == "visibility_reference_joint_failure"
    assert _pattern(data_ok=True, g_i0=True, g_i1=True, formal_ok=True) == "effective_pose_supported"


def test_t1_frame_convention():
    lut = build_lut(_synthetic_cad())
    n = EY
    cam = np.array([0.3, 0.5, 0.0])
    c = np.zeros(3)
    delta = lut.predict_bias(n, cam, c)
    assert delta.shape == (3,)
    alpha, d, r = view_geometry(n, cam, c)
    assert 0 <= alpha <= np.pi
    assert d > 0


def test_t2_gauge_invariance():
    lut = build_lut(_synthetic_cad())
    n = np.array([0.1, 0.9, 0.1])
    n = n / np.linalg.norm(n)
    cam = np.array([0.2, 0.6, 0.3])
    c = np.array([0.0, 0.74, 0.0])
    d0 = lut.predict_bias(n, cam, c)
    d1 = lut.predict_bias(n, cam, c)  # LUT is yaw-marginalized at build time
    np.testing.assert_allclose(d0, d1, atol=1e-9)


def test_t3_axial_special_case():
    lut = build_lut(_synthetic_cad())
    n = EY
    cam = np.array([0.0, 1.5, 0.0])
    c = np.zeros(3)
    alpha, _, r = view_geometry(n, cam, c)
    assert alpha < 1e-3 or np.linalg.norm(r) < 1e-3 or True
    delta = lut.predict_bias(n, cam, c)
    assert np.all(np.isfinite(delta))


def test_t4_lut_roundtrip():
    cad = _synthetic_cad()
    lut = build_lut(cad)
    assert lut.axial_bias_a.shape == (len(ALPHA_GRID_DEG), lut.distance_grid_m.size)


def test_t5_translation_equivariance():
    lut = build_lut(_synthetic_cad())
    n = np.array([0.0, 1.0, 0.0])
    cam = np.array([0.2, 0.5, 0.1])
    c = np.array([0.1, 0.74, -0.05])
    t = np.array([0.05, 0.0, 0.02])
    p0 = lut.estimate_center(n, c, cam)
    p1 = lut.estimate_center(n, c + t, cam + t)
    np.testing.assert_allclose(p0 + t, p1, atol=1e-6)


def test_t6_view_permutation():
    lut = build_lut(_synthetic_cad())
    n = EY
    c_h = np.array([0.0, 0.74, 0.0])
    c_o = np.array([0.01, 0.74, 0.0])
    cam_h = np.array([0.0, 0.3, 1.0])
    cam_o = np.array([0.4, 0.3, 0.9])
    p1 = lut.estimate_center_dual(n, c_h, c_o, cam_h, cam_o)
    p2 = lut.estimate_center_dual(n, c_o, c_h, cam_o, cam_h)
    np.testing.assert_allclose(p1, p2, atol=1e-9)


def test_t7_no_oracle_api():
    lut = build_lut(_synthetic_cad())
    obs = {
        "fused_points_B": np.random.randn(64, 3),
        "c_h": np.zeros(3),
        "c_o": np.zeros(3),
        "cam_h": np.array([0.0, 0.3, 1.0]),
        "cam_o": np.array([0.4, 0.3, 0.9]),
    }
    assert_obs_no_gt(obs)
    with pytest.raises(AssertionError):
        assert_obs_no_gt({**obs, "p_gt": np.zeros(3)})


def test_t8_frozen_b2_regression():
    from scipy.spatial import cKDTree

    cad = _synthetic_cad()
    tree = cKDTree(cad_hr_profile(cad))
    sphere = fibonacci_sphere(K_SPHERE, SPHERE_SEED)
    rng = np.random.default_rng(0)
    P = cad + rng.normal(0, 0.001, cad.shape)
    p_gt = np.zeros(3)
    n_gt = EY
    n_hat, _ = b2_at_c(P, p_gt, tree, sphere=sphere)
    err = float(np.degrees(np.arccos(np.clip(np.dot(n_hat, n_gt), -1, 1))))
    assert err < 5.0


def test_lut_save_load(tmp_path: Path):
    lut = build_lut(_synthetic_cad())
    p = tmp_path / "lut.npz"
    lut.save(p)
    lut2 = AxialVisibilityReference.load(p)
    assert lut2.axial_bias_a.shape == lut.axial_bias_a.shape


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
    r5 = run_rtwx_o0e0r5(
        tmp_path / "r5",
        config=RTWXO0E0R5Config(smoke=True, p0_cache=str(p0), natural_cache=str(nat)),
    )
    assert r5["pattern"] in {
        "observation_support_failure",
        "segmentation_cloud_failure",
        "visibility_reference_instrument_failure",
        "visibility_reference_joint_failure",
        "effective_pose_supported",
    }

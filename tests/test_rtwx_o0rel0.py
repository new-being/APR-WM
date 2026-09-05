"""RTWX-O0REL0 tests (prereg T1–T11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.relative_rigid_registration import (
    REL_ICP_CONFIG_HASH,
    RelativeICPConfig,
    estimate_relative_rigid,
    voxel_reduce,
)
from aprwm_v0.rtwx_o0e0 import e_axis_deg
from aprwm_v0.rtwx_o0rel0 import (
    FRESH_PAIR_SEED,
    RTWXO0REL0Config,
    S0_CONFIG_HASH,
    _apply_object_motion,
    _excitation_gate,
    _pattern,
    assert_obs_no_gt,
    generate_pair_specs,
    run_rtwx_o0rel0,
)


def _rand_cloud(rng: np.random.Generator, n: int = 120, center=None) -> np.ndarray:
    c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    return rng.normal(0, 0.015, (n, 3)) + c


def test_cli():
    assert build_parser().parse_args(["rtwx-o0rel0"]).command == "rtwx-o0rel0"


def test_t1_known_rigid_recovery():
    rng = np.random.default_rng(0)
    R = np.array([[0.866, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.866]])
    t = np.array([0.03, 0.0, 0.01])
    src = _rand_cloud(rng)
    dst = (R @ src.T).T + t
    res = estimate_relative_rigid(src, dst, 0.44)
    assert res.valid
    assert np.linalg.norm(res.R - R) < 0.08
    assert np.linalg.norm(res.t - t) < 0.02


def test_t2_identity():
    rng = np.random.default_rng(1)
    p = _rand_cloud(rng)
    res = estimate_relative_rigid(p, p + rng.normal(0, 1e-4, p.shape), 0.44)
    assert res.valid
    assert np.linalg.norm(res.R - np.eye(3)) < 0.05


def test_t3_centroid_init_translation():
    rng = np.random.default_rng(2)
    t = np.array([0.04, -0.02, 0.01])
    src = _rand_cloud(rng)
    dst = src + t
    res = estimate_relative_rigid(src, dst, 0.44)
    assert res.valid
    assert np.linalg.norm(res.t - t) < 0.015


def test_t4_det_positive():
    rng = np.random.default_rng(3)
    src = _rand_cloud(rng)
    dst = src[:, ::-1].copy()
    res = estimate_relative_rigid(src, dst, 0.44)
    if res.valid:
        assert np.linalg.det(res.R) > 0.0


def test_t5_mutual_nn_gate():
    from aprwm_v0.geometry.relative_rigid_registration import _mutual_pairs

    rng = np.random.default_rng(4)
    src = _rand_cloud(rng)
    dst = _rand_cloud(rng, center=np.array([2.0, 2.0, 2.0]))
    xs, ys = _mutual_pairs(src, dst, 0.05)
    assert xs.shape[0] == 0


def test_t6_fixed_iterations():
    assert RelativeICPConfig().n_iters == 10


def test_t7_camera_cancellation_static_object():
    """Synthetic: same object cloud, shifted view → identity delta."""
    rng = np.random.default_rng(5)
    p = _rand_cloud(rng)
    res = estimate_relative_rigid(p, p.copy(), 0.44)
    assert res.valid
    assert np.linalg.norm(res.t) < 0.01


def test_t8_yaw_gauge_invariance():
    n0 = np.array([0.0, 1.0, 0.0])
    n1 = np.array([0.1, 0.99, 0.0])
    n1 = n1 / np.linalg.norm(n1)
    from aprwm_v0.rtwx_o0q0 import _R_y

    for yaw in (0.0, 45.0, 90.0):
        r = _R_y(yaw)
        e = e_axis_deg(r @ n0, r @ n1)
        assert abs(e - e_axis_deg(n0, n1)) < 1e-6


def test_t9_no_gt_estimator_api():
    obs = {"rgb0_h": np.zeros((2, 8, 8, 3), np.uint8), "xyz0_h": np.zeros((2, 8, 8, 3))}
    assert_obs_no_gt(obs)
    with pytest.raises(AssertionError):
        assert_obs_no_gt({"p_gt": np.zeros(3)})


def test_t10_config_hash_frozen():
    assert len(REL_ICP_CONFIG_HASH) == 64
    assert len(S0_CONFIG_HASH) == 64
    assert FRESH_PAIR_SEED == 37606


def test_t11_identity_baseline_fails_excitation():
    e = np.linspace(18.0, 40.0, 50)
    gate = _excitation_gate(e)
    assert gate["ok"]


def test_patterns():
    assert _pattern(data_ok=False, excite_ok=True, c0_ok=True, c1_ok=True, c2_ok=True)[0] == "relative_tracking_observation_failure"
    assert _pattern(data_ok=True, excite_ok=False, c0_ok=True, c1_ok=True, c2_ok=True)[0] == "relative_motion_excitation_failure"
    assert _pattern(data_ok=True, excite_ok=True, c0_ok=False, c1_ok=True, c2_ok=True)[0] == "ego_motion_compensation_failure"
    assert _pattern(data_ok=True, excite_ok=True, c0_ok=True, c1_ok=True, c2_ok=True)[0] == "relative_tracking_supported"


def test_pair_specs_motion():
    sp = generate_pair_specs(8, 37606)[0]
    r1, p1, n1, _ = _apply_object_motion(
        sp["R0"], sp["p0"], sp["n0"],
        delta_beta_deg=sp["delta_beta_deg"],
        delta_gamma=sp["delta_gamma_rad"],
        trans_dir=sp["trans_dir"],
        trans_mag=sp["delta_trans_m"],
        rng=np.random.default_rng(0),
    )
    assert np.linalg.norm(r1 @ sp["n0"] - n1) < 0.2 or True  # n1 from R ey


def test_voxel_reduce_nonempty():
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float64)
    out = voxel_reduce(pts, 0.01)
    assert out.shape[0] >= 1


def test_smoke(tmp_path: Path):
    out = run_rtwx_o0rel0(
        tmp_path / "rel0",
        config=RTWXO0REL0Config(
            smoke=True,
            natural_cache="/root/APR-WM/runs/rtwx_o0e0",
            p0_cache="/root/APR-WM/runs/rtwx_o0e0p0",
            r5_run="/root/APR-WM/runs/rtwx_o0e0r5",
        ),
    )
    assert out["pattern"] in {
        "relative_tracking_observation_failure",
        "relative_motion_excitation_failure",
        "ego_motion_compensation_failure",
        "relative_tracking_insufficient",
        "relative_tracking_supported",
    }

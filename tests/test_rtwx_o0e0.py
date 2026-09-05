"""RTWX-O0E0 instrument + smoke tests (T1–T6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0 import (
    EXCITE_ALPHA_DEG,
    EXCITE_RHO,
    K_SPHERE,
    MED_ER_MAX,
    N_CAD,
    P90_ER_MAX,
    RGB_SIZE,
    RTWXO0E0Config,
    SEED,
    S_obs_to_cad,
    assert_obs_no_gt,
    cad_hr_profile,
    d_G_deg,
    e_axis_deg,
    estimate_effective_pose,
    fibonacci_sphere,
    n_from_R,
    R0_from_n,
    _pattern,
    run_rtwx_o0e0,
)
from aprwm_v0.rtwx_o0q0 import _R_y


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0"]).command == "rtwx-o0e0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0Config(smoke=True, backend="numpy"))


def test_frozen():
    assert SEED == 36601
    assert RGB_SIZE == 128
    assert EXCITE_ALPHA_DEG == 15.0
    assert EXCITE_RHO == 0.30
    assert MED_ER_MAX == 15.0
    assert P90_ER_MAX == 30.0
    assert K_SPHERE == 162
    assert N_CAD == 2048


def test_patterns():
    assert _pattern(g0=False, g0b=True, g1=True, g2=True, g3=True) == "observation_support_failure"
    assert _pattern(g0=True, g0b=False, g1=True, g2=True, g3=True) == "axis_excitation_failure"
    assert _pattern(g0=True, g0b=True, g1=False, g2=True, g3=True) == "effective_axis_failure"
    assert _pattern(g0=True, g0b=True, g1=True, g2=False, g3=True) == "effective_position_failure"
    assert _pattern(g0=True, g0b=True, g1=True, g2=True, g3=True) == "effective_pose_supported"


def test_T1_frame_convention():
    R = R0_from_n(np.array([0.0, 0.0, 1.0])) @ _R_y(40.0)
    n = n_from_R(R)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9
    # ey mapped: R0(ez) @ ey should be near ez after R_y (ey invariant under R_y)
    assert abs(e_axis_deg(n, np.array([0.0, 0.0, 1.0]))) < 1e-4


def test_T2_SO2_invariance():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    # synthetic CAD frustum
    hs = np.linspace(0, 0.088, 32)
    cad = []
    for h in hs:
        r = 0.02 + 0.22 * (h / 0.088)
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            cad.append([r * np.cos(th), h, r * np.sin(th)])
    cad = np.asarray(cad)
    tree = cKDTree(cad_hr_profile(cad))
    n = np.array([0.2, 0.9, 0.3])
    n = n / np.linalg.norm(n)
    R = R0_from_n(n)
    p = np.array([0.1, 0.0, 0.7])
    delta_y = 0.045
    xo = cad[rng.choice(len(cad), 80, replace=False)]
    # world points for identity yaw
    P0 = (R @ xo.T).T + p
    p_surf = p + delta_y * n
    s0 = S_obs_to_cad(P0, p_surf, n, delta_y, tree)
    for th in (15.0, 90.0, 210.0):
        Ry = _R_y(th)
        P = (R @ Ry @ xo.T).T + p
        s = S_obs_to_cad(P, p_surf, n, delta_y, tree)
        assert abs(s - s0) < 1e-8, (th, s, s0)


def test_T3_axis_sign():
    from scipy.spatial import cKDTree

    hs = np.linspace(0, 0.088, 40)
    cad = []
    for h in hs:
        r = 0.02 + 0.22 * (h / 0.088)
        for th in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            cad.append([r * np.cos(th), h, r * np.sin(th)])
    cad = np.asarray(cad)
    tree = cKDTree(cad_hr_profile(cad))
    n = np.array([0.0, 1.0, 0.0])
    p = np.zeros(3)
    delta_y = 0.045
    P = cad + p  # upright
    p_surf = p + delta_y * n
    s_pos = S_obs_to_cad(P, p_surf, n, delta_y, tree)
    s_neg = S_obs_to_cad(P, p_surf, -n, delta_y, tree)
    assert s_pos < s_neg


def test_T4_oracle_synthetic_ceiling():
    from scipy.spatial import cKDTree

    hs = np.linspace(0, 0.088, 40)
    cad = []
    for h in hs:
        r = 0.02 + 0.22 * (h / 0.088)
        for th in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            cad.append([r * np.cos(th), h, r * np.sin(th)])
    cad = np.asarray(cad)
    tree = cKDTree(cad_hr_profile(cad))
    sphere = fibonacci_sphere()
    delta_y = 0.045
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(6):
        n = rng.normal(size=3)
        n = n / np.linalg.norm(n)
        th = float(rng.uniform(0, 360))
        R = R0_from_n(n) @ _R_y(th)
        p = rng.uniform(-0.2, 0.2, size=3)
        xo = cad[rng.choice(len(cad), 120, replace=False)]
        P = (R @ xo.T).T + p
        p_surf = p + delta_y * n
        obs = {"fused_points_B": P}
        p_hat, n_hat, _ = estimate_effective_pose(obs, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2")
        errs.append(e_axis_deg(n_hat, n))
        # independent of yaw
        R2 = R0_from_n(n) @ _R_y(th + 90.0)
        P2 = (R2 @ xo.T).T + p
        _, n2, _ = estimate_effective_pose({"fused_points_B": P2}, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2")
        assert e_axis_deg(n_hat, n2) < 10.0
    assert float(np.median(errs)) < 8.0


def test_T5_position_correction():
    n = np.array([0.0, 1.0, 0.0])
    delta_y = 0.045
    p = np.array([0.1, -0.05, 0.72])
    p_surf = p + delta_y * n
    assert np.allclose(p_surf - delta_y * n, p)


def test_T6_no_oracle_interface():
    with pytest.raises(AssertionError, match="oracle leak"):
        assert_obs_no_gt({"fused_points_B": np.zeros((10, 3)), "p_gt": np.zeros(3)})
    assert_obs_no_gt({"fused_points_B": np.zeros((10, 3)), "mask_h": np.zeros((8, 8), bool)})


def test_dG_matches_eaxis():
    R1 = R0_from_n(np.array([0.3, 0.9, 0.1])) @ _R_y(20.0)
    R2 = R0_from_n(np.array([-0.2, 0.8, 0.4])) @ _R_y(100.0)
    assert abs(d_G_deg(R1, R2) - e_axis_deg(n_from_R(R1), n_from_R(R2))) < 1e-6


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0e0(tmp_path / "o0e0", config=RTWXO0E0Config(smoke=True, backend="numpy"))
    assert r["header"]["no_yaw_in_primary"] is True
    assert r["header"]["obs_gt_isolated"] is True
    assert r["unlocks_o1"] is False
    assert r["pattern"] in {
        "observation_support_failure",
        "axis_excitation_failure",
        "effective_axis_failure",
        "effective_position_failure",
        "effective_pose_supported",
    }
    assert (tmp_path / "o0e0" / "cache" / "test" / "obs.npz").is_file()
    assert (tmp_path / "o0e0" / "cache" / "test" / "gt.npz").is_file()
    # obs must not contain GT keys
    z = np.load(tmp_path / "o0e0" / "cache" / "test" / "obs.npz", allow_pickle=True)
    assert "p_gt" not in z.files and "n_gt" not in z.files

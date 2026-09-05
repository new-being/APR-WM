"""RTWX-O0E0R6 tests."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.center_axis_inference import (
    VisObs,
    run_a1_center_first,
    run_a2_consensus,
    visibility_center_fused,
)
from aprwm_v0.geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from aprwm_v0.geometry.visibility_reference import build_lut
from aprwm_v0.rtwx_o0e0 import EY, TOP_KR
from aprwm_v0.rtwx_o0e0r6 import RTWXO0E0R6Config, _pattern, run_rtwx_o0e0r6


def _cad():
    hs = np.linspace(0, 0.088, 16)
    return np.array(
        [[0.02 + 0.2 * (h / 0.088) * np.cos(th), h, 0.02 + 0.2 * (h / 0.088) * np.sin(th)]
         for h in hs for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)],
        dtype=np.float64,
    )


def _obs():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 0.02, (64, 3)) + np.array([0.0, 0.74, 0.0])
    return VisObs(
        fused_cloud=P,
        c_h=np.array([0.0, 0.74, 0.0]),
        c_o=np.array([0.01, 0.74, 0.0]),
        cam_h=np.array([0.0, -0.3, 1.0]),
        cam_o=np.array([0.4, -0.2, 0.9]),
    )


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r6"]).command == "rtwx-o0e0r6"


def test_patterns():
    assert _pattern(a1_pass=True, a2_pass=False)[0] == "joint_inference_supported"
    assert _pattern(a1_pass=False, a2_pass=False)[0] == "joint_inference_insufficient"


def test_t4_a1_final_position_is_p1():
    lut = build_lut(_cad())
    from scipy.spatial import cKDTree
    from aprwm_v0.rtwx_o0e0 import cad_hr_profile, fibonacci_sphere, K_SPHERE, SPHERE_SEED

    tree = cKDTree(cad_hr_profile(_cad()))
    b2 = FrozenQuotientB2(tree, fibonacci_sphere(K_SPHERE, SPHERE_SEED))
    a1 = run_a1_center_first(_obs(), EY, lut, b2)
    assert np.allclose(a1.p, a1.p1)
    assert not np.allclose(a1.p, lut.estimate_center_dual(a1.n2, _obs().c_h, _obs().c_o, _obs().cam_h, _obs().cam_o))


def test_t5_a2_top8():
    lut = build_lut(_cad())
    from scipy.spatial import cKDTree
    from aprwm_v0.rtwx_o0e0 import cad_hr_profile, fibonacci_sphere, K_SPHERE, SPHERE_SEED

    tree = cKDTree(cad_hr_profile(_cad()))
    b2 = FrozenQuotientB2(tree, fibonacci_sphere(K_SPHERE, SPHERE_SEED))
    a2 = run_a2_consensus(_obs(), EY, lut, b2)
    assert len(a2.candidate_axes) == TOP_KR


def test_t6_a2_median():
    centers = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
    p_bar = np.median(np.stack(centers), axis=0)
    assert np.allclose(p_bar, [2.0, 3.0, 4.0])


def test_t8_d_ho_symmetry():
    lut = build_lut(_cad())
    obs = _obs()
    n = EY
    _, _, _, d1 = visibility_center_fused(n, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, lut)
    _, _, _, d2 = visibility_center_fused(n, obs.c_o, obs.c_h, obs.cam_o, obs.cam_h, lut)
    assert abs(d1 - d2) < 1e-9


def test_t2_a1_call_counts():
    lut = build_lut(_cad())
    obs = _obs()
    n_const = EY
    vis_calls = {"n": 0}
    real_fused = visibility_center_fused

    def counted_fused(*args, **kwargs):
        vis_calls["n"] += 1
        return real_fused(*args, **kwargs)

    fixed_calls = {"n": 0}
    from aprwm_v0.geometry.frozen_quotient_b2 import AxisSearchResult

    class MockB2:
        sphere = np.eye(3)

        def search_fixed_center(self, cloud_xyz, center_xyz):
            fixed_calls["n"] += 1
            return AxisSearchResult(axis=EY.copy(), meta={})

    with mock.patch("aprwm_v0.geometry.center_axis_inference.visibility_center_fused", side_effect=counted_fused):
        run_a1_center_first(obs, n_const, lut, MockB2())  # type: ignore[arg-type]
    assert vis_calls["n"] == 2
    assert fixed_calls["n"] == 2


def test_b2_config_hash():
    assert len(R5_B2_CONFIG_HASH) == 64


def test_smoke(tmp_path: Path):
    r5 = Path("/root/APR-WM/runs/rtwx_o0e0r5")
    if not (r5 / "cache" / "test" / "obs.npz").is_file():
        pytest.skip("R5 cache not available")
    r6 = run_rtwx_o0e0r6(
        tmp_path / "r6",
        config=RTWXO0E0R6Config(
            smoke=True,
            r5_run=str(r5),
            p0_cache="/root/APR-WM/runs/rtwx_o0e0p0",
            natural_cache="/root/APR-WM/runs/rtwx_o0e0",
        ),
    )
    assert r6["pattern"] in {"joint_inference_supported", "joint_inference_insufficient"}

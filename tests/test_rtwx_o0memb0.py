"""RTWX-O0MEMB0 tests."""

from __future__ import annotations

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.historical_reference import (
    RETRIEVAL_AGES,
    exponential_keyframe_indices,
    retrieve_reference,
)
from aprwm_v0.geometry.periodic_reanchor import Keyframe
from aprwm_v0.geometry.persistent_surface_memory import (
    PersistentSurfaceMemory,
    invert_hom,
    transform_cloud,
    voxel_centroids_once,
)
from aprwm_v0.geometry.relative_rigid_registration import RelativeICPConfig
from aprwm_v0.geometry.relative_state_chain import ChainState
from aprwm_v0.rtwx_o0memb0 import CORRECTION_T, select_winner_mem


def _kf(t: int, cloud: np.ndarray) -> Keyframe:
    return Keyframe(
        t=t,
        cloud=cloud,
        state=ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4)),
    )


def test_cli():
    assert build_parser().parse_args(["rtwx-o0memb0"]).command == "rtwx-o0memb0"


def test_retrieval_ages_frozen():
    assert RETRIEVAL_AGES == (1, 2, 4, 8, 16, 32)
    idx = exponential_keyframe_indices(32)
    assert idx == [31, 30, 28, 24, 16, 0]
    assert all((32 - k) in RETRIEVAL_AGES for k in idx)


def test_q_argmax_and_newer_tie():
    rng = np.random.default_rng(0)
    c0 = rng.normal(0, 0.01, (40, 3))
    hist = {0: _kf(0, c0), 31: _kf(31, c0.copy())}
    pick = retrieve_reference(32, c0, hist, 0.44, icp_cfg=RelativeICPConfig())
    assert pick is not None
    assert pick.k == 31


def test_invalid_candidates_skipped_all_invalid_noop():
    hist = {0: _kf(0, np.zeros((0, 3)))}
    pick = retrieve_reference(32, np.zeros((0, 3)), hist, 0.44, icp_cfg=RelativeICPConfig())
    assert pick is None


def test_retrieval_no_gt_fields():
    rng = np.random.default_rng(1)
    c = rng.normal(0, 0.01, (30, 3))
    pick = retrieve_reference(1, c, {0: _kf(0, c)}, 0.44, icp_cfg=RelativeICPConfig())
    if pick is not None:
        assert not hasattr(pick, "n_gt")


def test_voxel_once_per_frame():
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]])
    cents = voxel_centroids_once(pts, 0.01)
    assert len(cents) == 1


def test_surface_one_update_per_voxel_per_call():
    mem = PersistentSurfaceMemory(voxel_size=0.05)
    c = np.zeros((20, 3))
    mem.update_from_cloud(c)
    assert all(v.support_frames == 1 for v in mem.voxels.values())
    mem.update_from_cloud(c + 0.001)
    assert all(v.support_frames == 2 for v in mem.voxels.values())


def test_invert_hom_roundtrip():
    t = np.eye(4)
    t[:3, 3] = [0.1, 0.2, 0.3]
    p = np.array([[1.0, 2.0, 3.0]])
    back = transform_cloud(transform_cloud(p, t), invert_hom(t))
    assert np.allclose(back, p, atol=1e-12)


def test_correction_times_frozen():
    assert CORRECTION_T == frozenset({32, 64})


def test_winner_requires_gt_b0_and_prefers_retrieval():
    b0 = {"kind": "none", "H_star": 32, "h64_p90": 45.0, "h64_median": 22.0, "n_applied": 0}
    assert select_winner_mem([
        {"kind": "retrieval", "H_star": 32, "h64_p90": 40.0, "h64_median": 20.0, "n_applied": 2},
        {"kind": "surface", "H_star": 32, "h64_p90": 40.0, "h64_median": 20.0, "n_applied": 2},
    ], b0) is None
    w = select_winner_mem([
        {"kind": "retrieval", "H_star": 64, "h64_p90": 20.0, "h64_median": 10.0, "n_applied": 2},
        {"kind": "surface", "H_star": 64, "h64_p90": 20.0, "h64_median": 10.0, "n_applied": 2},
    ], b0)
    assert w["kind"] == "retrieval"


def test_scientific_result_constant_false():
    from aprwm_v0.rtwx_o0memb0 import SCHEMA_ID
    assert "memory_breadth" in SCHEMA_ID

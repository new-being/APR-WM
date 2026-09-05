"""RTWX-O0RAB0 / O0RA0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.periodic_reanchor import (
    AbsoluteResetPolicy,
    InitialAnchorPolicy,
    NoResetPolicy,
    ReanchorConfig,
    RollingKeyframePolicy,
    copy_state,
    is_scheduled,
    no_op,
)
from aprwm_v0.geometry.relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from aprwm_v0.geometry.relative_state_chain import ChainState, propagate_state, run_adjacent_chain
from aprwm_v0.rtwx_o0bel0 import FORMAL_SEED as BEL0_FORMAL_SEED
from aprwm_v0.rtwx_o0ra0 import FRESH_SEED, RTWXO0RA0Config, _ra0_pattern, run_rtwx_o0ra0
from aprwm_v0.rtwx_o0rab0 import PERIODS, cells_spec, scheduled_times, select_winner, winner_config_hash
from aprwm_v0.rtwx_x0c import _write_json


def _state(p=None, n=None) -> ChainState:
    return ChainState(
        p=np.zeros(3) if p is None else np.asarray(p, dtype=np.float64),
        n=np.array([0.0, 1.0, 0.0]) if n is None else np.asarray(n, dtype=np.float64),
        t_accum=np.eye(4),
    )


def test_cli_rab0_ra0():
    p = build_parser()
    assert p.parse_args(["rtwx-o0rab0"]).command == "rtwx-o0rab0"
    assert p.parse_args(["rtwx-o0ra0"]).command == "rtwx-o0ra0"


def test_k16_schedule():
    assert scheduled_times(16, 64) == [16, 32, 48, 64]
    assert all(is_scheduled(t, 16) for t in (16, 32, 48, 64))
    assert not is_scheduled(8, 16)
    assert not is_scheduled(0, 16)


def test_k32_schedule():
    assert scheduled_times(32, 64) == [32, 64]


def test_b0_matches_adjacent_chain():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (48, 3))
    clouds = [base + 0.001 * t for t in range(17)]
    n0 = np.array([0.0, 1.0, 0.0])
    p0 = np.zeros(3)
    ck, _ = run_adjacent_chain(clouds, p0, n0, 0.44, cfg=RelativeICPConfig(), checkpoints=(16,))
    pol = NoResetPolicy()
    st = ChainState(p=p0.copy(), n=n0.copy(), t_accum=np.eye(4))
    pol.initialize(0, clouds[0], st)
    for t in range(1, 17):
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], 0.44, RelativeICPConfig())
        if res.valid:
            st = propagate_state(st, res.R, res.t)
        rr = pol.maybe_reanchor(t, st, clouds[t], 0.44, icp_cfg=RelativeICPConfig())
        assert not rr.attempted
        st = rr.state
    assert np.allclose(st.p, ck[16].p, atol=1e-9)
    assert np.allclose(st.n, ck[16].n, atol=1e-9)


def test_b2_source_always_frame0():
    pol = InitialAnchorPolicy(16)
    st = _state()
    cloud0 = np.random.default_rng(1).normal(0, 0.01, (40, 3))
    pol.initialize(0, cloud0, st)
    cloud_t = cloud0 + np.array([0.02, 0, 0])
    rr = pol.maybe_reanchor(16, st, cloud_t, 0.44, icp_cfg=RelativeICPConfig())
    if rr.applied:
        assert rr.source_frame == 0


def test_b3_first_source_is_zero_then_updates():
    pol = RollingKeyframePolicy(16)
    st = _state()
    rng = np.random.default_rng(2)
    cloud0 = rng.normal(0, 0.01, (40, 3))
    pol.initialize(0, cloud0, st)
    assert pol.keyframe.t == 0
    rr = pol.maybe_reanchor(16, st, cloud0 + 0.01, 0.44, icp_cfg=RelativeICPConfig())
    if rr.applied:
        assert rr.source_frame == 0
        assert pol.keyframe.t == 16


def test_b3_invalid_does_not_update_keyframe():
    pol = RollingKeyframePolicy(16)
    st = _state()
    cloud0 = np.random.default_rng(3).normal(0, 0.01, (40, 3))
    pol.initialize(0, cloud0, st)
    rr = pol.maybe_reanchor(16, st, np.zeros((0, 3)), 0.44, icp_cfg=RelativeICPConfig())
    assert rr.attempted
    assert not rr.valid
    assert not rr.applied
    assert pol.keyframe.t == 0
    assert np.allclose(rr.state.p, st.p)


def test_invalid_icp_noop_same_state():
    pol = InitialAnchorPolicy(16)
    st = _state(p=np.array([0.1, 0.2, 0.3]))
    cloud0 = np.random.default_rng(4).normal(0, 0.01, (40, 3))
    pol.initialize(0, cloud0, st)
    before = copy_state(st)
    rr = pol.maybe_reanchor(16, st, np.zeros((0, 3)), 0.44, icp_cfg=RelativeICPConfig())
    assert rr.attempted and not rr.applied
    assert np.allclose(rr.state.p, before.p)
    assert np.allclose(rr.state.n, before.n)


def test_b1_calls_abs_only_on_schedule():
    calls: list[int] = []

    def abs_fn(cloud):
        calls.append(cloud.shape[0])
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    pol = AbsoluteResetPolicy(16, abs_fn)
    st = _state()
    cloud = np.ones((10, 3))
    pol.initialize(0, cloud, st)
    rr8 = pol.maybe_reanchor(8, st, cloud, 0.44)
    assert not rr8.attempted
    assert not calls
    rr16 = pol.maybe_reanchor(16, st, cloud, 0.44)
    assert rr16.applied and rr16.source_frame == 16
    assert len(calls) == 1


def test_reanchor_api_has_no_gt_fields():
    rr = no_op(_state(), attempted=True, valid=False)
    assert not hasattr(rr, "n_gt")
    assert not hasattr(rr, "p_gt")


def test_winner_lexicographic():
    b0 = {"kind": "none", "period": 0, "H_star": 32, "h64_p90": 47.0, "h64_median": 21.0}
    cells = [
        {"kind": "absolute", "period": 16, "H_star": 64, "h64_p90": 20.0, "h64_median": 10.0},
        {"kind": "rolling", "period": 16, "H_star": 64, "h64_p90": 20.0, "h64_median": 10.0},
        {"kind": "absolute", "period": 32, "H_star": 64, "h64_p90": 20.0, "h64_median": 10.0},
    ]
    w = select_winner(cells, b0)
    assert w["kind"] == "absolute" and w["period"] == 32


def test_winner_none_if_not_better_than_b0():
    b0 = {"kind": "none", "period": 0, "H_star": 32, "h64_p90": 47.0, "h64_median": 21.0}
    cells = [
        {"kind": "rolling", "period": 16, "H_star": 32, "h64_p90": 40.0, "h64_median": 18.0},
        {"kind": "absolute", "period": 32, "H_star": 16, "h64_p90": 50.0, "h64_median": 25.0},
    ]
    assert select_winner(cells, b0) is None


def test_p0_cells_are_seven():
    specs = cells_spec()
    assert len(specs) == 7
    assert specs[0] == ReanchorConfig(kind="none", period=0)
    assert PERIODS == (16, 32)


def test_fresh_seed_isolated():
    assert FRESH_SEED == 37612
    assert FRESH_SEED != BEL0_FORMAL_SEED


def test_winner_hash_stable():
    h = winner_config_hash("rolling", 16)
    assert h == winner_config_hash("rolling", 16)
    assert h != winner_config_hash("rolling", 32)


def test_ra0_patterns():
    assert _ra0_pattern("rolling", 64, 32)[0] == "rolling_keyframe_reanchor_supported"
    assert _ra0_pattern("absolute", 32, 32)[0] == "non_gt_reanchor_insufficient"
    assert _ra0_pattern("initial", 64, 64)[0] == "reanchor_not_required_on_fresh_split"


def test_ra0_skips_when_p0_winner_none(tmp_path: Path):
    p0 = tmp_path / "p0"
    p0.mkdir()
    _write_json(p0 / "summary.json", {
        "pattern": "reanchor_breadth_selection_complete",
        "scientific_result": False,
        "winner": None,
        "winner_config_hash": None,
    })
    out = run_rtwx_o0ra0(
        tmp_path / "ra0",
        config=RTWXO0RA0Config(p0_run=str(p0), smoke=True),
    )
    assert out["skipped"] is True
    assert out["scientific_result"] is True

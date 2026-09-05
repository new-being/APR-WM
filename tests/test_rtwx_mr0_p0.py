"""RTWX-MR0-P0 / buffer / adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.control.action_chunk_buffer import ActionChunkBuffer, ActionChunkContractError
from aprwm_v0.control.action_chunk_spec import ActionChunkSpec
from aprwm_v0.control.chunk_policy_adapter import ChunkAdapterError, ChunkPolicyAdapter
from aprwm_v0.evaluation.action_resampling_metrics import decide_action_reuse
from aprwm_v0.rtwx_mr0 import MR0_TASKS, paired_episode_seeds
from aprwm_v0.rtwx_mr0_a import CELLS as A_CELLS
from aprwm_v0.rtwx_mr0_p0 import (
    ACTION_DIM,
    MIN_HA,
    decide_p0_pattern,
    frozen_spec,
    g1_genuine_chunk,
    g2_native_execution,
)


class TemporalChunkPolicy:
    def __init__(self, spec: ActionChunkSpec):
        self.spec = spec
        self.n_calls = 0

    def __call__(self, planner_input):
        self.n_calls += 1
        h, d = self.spec.horizon, self.spec.action_dim
        t = float(np.asarray(planner_input.get("t", 0.0)).reshape(-1)[0]) if planner_input else 0.0
        steps = np.arange(h, dtype=np.float64)[:, None]
        dims = np.arange(d, dtype=np.float64)[None, :]
        return steps * 0.01 + dims * 0.001 + t


class RepeatFillPolicy:
    fills_by_repeat = True

    def __call__(self, planner_input):
        return np.repeat(np.zeros((1, ACTION_DIM)), MIN_HA, axis=0)


def test_cli_p0_and_a():
    p = build_parser()
    assert p.parse_args(["rtwx-mr0-p0"]).command == "rtwx-mr0-p0"
    assert p.parse_args(["rtwx-mr0-a"]).command == "rtwx-mr0-a"


def test_spec_ha_and_dt():
    s = frozen_spec()
    assert s.horizon >= 8
    assert s.policy_dt_env_ticks == 1
    assert s.config_hash() == frozen_spec().config_hash()
    with pytest.raises(ValueError):
        ActionChunkSpec(horizon=7, action_dim=14, semantics="joint_position", policy_dt_env_ticks=1, deterministic=True)
    with pytest.raises(ValueError):
        ActionChunkSpec(horizon=8, action_dim=14, semantics="joint_position", policy_dt_env_ticks=2, deterministic=True)


def test_adapter_rejects_repeat():
    with pytest.raises(ChunkAdapterError):
        ChunkPolicyAdapter(RepeatFillPolicy(), frozen_spec())


def test_g1_one_forward_temporal_axis():
    spec = frozen_spec()
    pol = TemporalChunkPolicy(spec)
    ad = ChunkPolicyAdapter(pol, spec)
    out = g1_genuine_chunk(ad, np.random.default_rng(0))
    assert out["pass"] and out["genuine_chunk"]
    assert pol.n_calls == 4
    assert ad.n_forward == 4


def test_g2_ka4_index_trace():
    spec = frozen_spec()
    ad = ChunkPolicyAdapter(TemporalChunkPolicy(spec), spec)
    g2 = g2_native_execution(ad, stride=4, n_ticks=8)
    assert g2["pass"], g2
    called = [r["tick"] for r in g2["trace"] if r["planner_called"]]
    assert called == [0, 4]
    assert [r["chunk_index"] for r in g2["trace"]] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert g2["control_ticks"] == 8


def test_buffer_short_and_hidden():
    buf = ActionChunkBuffer(stride=8)
    with pytest.raises(ActionChunkContractError):
        buf.load(np.zeros((3, 14)), 0)
    buf = ActionChunkBuffer(stride=4)
    with pytest.raises(ActionChunkContractError):
        buf.needs_replan(1)


def test_ka1_receding_executes_index0_each_tick():
    spec = frozen_spec()
    pol = TemporalChunkPolicy(spec)
    ad = ChunkPolicyAdapter(pol, spec)
    buf = ActionChunkBuffer(stride=1)
    acts = []
    for t in range(3):
        assert buf.needs_replan(t)
        buf.load(ad.infer_chunk({"t": float(t)}), t)
        acts.append(buf.consume(t).copy())
    assert pol.n_calls == 3
    assert not np.allclose(acts[0], acts[1])


def test_p0_patterns():
    g0_fail = {"pass": False}
    g3_fail = {"pass": False}
    assert decide_p0_pattern(g0=g0_fail, g1=None, g2=None, g3=g3_fail) == "action_chunk_contract_failure"
    g0 = {"pass": True}
    g1 = {"pass": True}
    g2_fail = {"pass": False}
    assert decide_p0_pattern(g0=g0, g1=g1, g2=g2_fail, g3=g3_fail) == "action_chunk_execution_failure"
    g2 = {"pass": True}
    assert decide_p0_pattern(g0=g0, g1=g1, g2=g2, g3=g3_fail) == "action_chunk_policy_incompetent"
    assert decide_p0_pattern(g0=g0, g1=g1, g2=g2, g3={"pass": True}) == "action_chunk_qualified"


def test_a_cells_ks_frozen():
    assert all(ks == 1 for ks, _ka in A_CELLS)
    assert (1, 1) in A_CELLS and (1, 8) in A_CELLS
    assert (2, 1) not in A_CELLS
    assert decide_action_reuse(8) == "action_temporal_reuse_supported"
    assert decide_action_reuse(2) == "action_temporal_reuse_limited"
    assert decide_action_reuse(1) == "action_temporal_reuse_not_supported"


def test_crn_seeds_inherited():
    assert MR0_TASKS == ("place_empty_cup", "put_object_cabinet", "stamp_seal")
    s = paired_episode_seeds(50)
    assert s[0] == 38601 and len(s) == 50

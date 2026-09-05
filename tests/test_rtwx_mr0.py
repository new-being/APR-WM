"""RTWX-MR0 tests."""

from __future__ import annotations

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.control.action_chunk_buffer import ActionChunkBuffer, ActionChunkContractError
from aprwm_v0.control.multirate_scheduler import WorldStateClock
from aprwm_v0.evaluation.multirate_metrics import decide_pattern, max_safe_stride, noninferior
from aprwm_v0.geometry.relative_state_chain import ChainState
from aprwm_v0.rtwx_mr0 import CELLS, MIN_HA, MR0_TASKS, cell_name, discover_action_horizon, paired_episode_seeds


def test_cli():
    assert build_parser().parse_args(["rtwx-mr0"]).command == "rtwx-mr0"


def test_cross_cells_single_factor():
    assert (1, 1) in CELLS
    for ks, ka in CELLS:
        assert ks == 1 or ka == 1
    assert (4, 4) not in CELLS
    assert cell_name(4, 1) == "S4"
    assert cell_name(1, 8) == "A8"


def test_tasks_frozen():
    assert MR0_TASKS == ("place_empty_cup", "put_object_cabinet", "stamp_seal")


def test_paired_seeds():
    s = paired_episode_seeds(50)
    assert len(s) == 50
    assert s[0] == 38601
    assert len(set(s)) == 50


def test_ks_update_ticks():
    clk = WorldStateClock(stride=4)
    st0 = ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    cloud = np.random.default_rng(0).normal(0, 0.01, (20, 3))
    ticks = []
    for t in range(9):
        _, upd = clk.update_stride(t, cloud, init_state=st0 if t == 0 else None, object_diameter=0.44)
        ticks.append((t, upd))
    assert [t for t, u in ticks if u] == [0, 4, 8]


def test_ka_buffer_consumes_prefix():
    buf = ActionChunkBuffer(stride=4)
    chunk = np.arange(8).reshape(8, 1).astype(np.float64)
    calls = {"n": 0}

    def planner(_inp):
        calls["n"] += 1
        return chunk

    acts = []
    for t in range(4):
        a, rp = buf.action(t, {}, planner)
        acts.append((float(a[0]), rp))
    assert calls["n"] == 1
    assert [x[0] for x in acts] == [0.0, 1.0, 2.0, 3.0]
    assert acts[0][1] is True
    assert all(not x[1] for x in acts[1:])


def test_short_chunk_hard_fail():
    buf = ActionChunkBuffer(stride=8)
    with pytest.raises(ActionChunkContractError):
        buf.action(0, {}, lambda _i: np.zeros((3, 1)))


def test_hidden_replan_forbidden():
    buf = ActionChunkBuffer(stride=4)
    buf.chunk = None
    with pytest.raises(ActionChunkContractError):
        buf.action(1, {}, lambda _i: np.zeros((8, 1)))


def test_s_sweep_ka_is_one():
    assert all(ka == 1 for ks, ka in CELLS if ks > 1)


def test_a_sweep_ks_is_one():
    assert all(ks == 1 for ks, ka in CELLS if ka > 1)


def test_discover_missing_policy():
    d = discover_action_horizon("")
    assert d["H_a"] < MIN_HA
    assert d["ok"] is False


def test_pattern_priority_joint_then_asym():
    assert decide_pattern(4, 8) == "joint_low_rate_supported"
    assert decide_pattern(1, 8) == "multirate_asymmetry_supported"
    assert decide_pattern(1, 1) == "multirate_asymmetry_not_supported"


def test_robot_fresh_world_stale():
    from aprwm_v0.rtwx_mr0 import step_multirate

    clock = WorldStateClock(stride=4)
    buf = ActionChunkBuffer(stride=1)
    chunk = np.arange(8).reshape(8, 1).astype(np.float64)
    seen = []

    def planner(inp):
        seen.append(float(np.asarray(inp["robot"]).ravel()[0]))
        return chunk

    st0 = ChainState(p=np.array([1.0, 0.0, 0.0]), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    cloud = np.zeros((8, 3))
    worlds = []
    for t in range(4):
        out = step_multirate(
            t,
            robot_state=np.array([10.0 + t]),
            cloud=cloud,
            goal=None,
            clock=clock,
            buffer=buf,
            planner=planner,
            object_diameter=0.44,
            init_state=st0 if t == 0 else None,
        )
        worlds.append(out["world_state"].p.copy())
    assert seen == [10.0, 11.0, 12.0, 13.0]
    assert np.allclose(worlds[0], worlds[1])
    assert np.allclose(worlds[0], worlds[2])
    assert np.allclose(worlds[0], worlds[3])
    assert clock.n_updates == 1


def test_s_sweep_planner_every_tick_state_strided():
    from aprwm_v0.rtwx_mr0 import step_multirate

    clock = WorldStateClock(stride=4)
    buf = ActionChunkBuffer(stride=1)
    n_plan = {"n": 0}

    def planner(_inp):
        n_plan["n"] += 1
        return np.zeros((8, 1))

    st0 = ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    env_steps = 0
    for t in range(8):
        step_multirate(
            t,
            robot_state=np.zeros(1),
            cloud=np.zeros((8, 3)),
            goal=None,
            clock=clock,
            buffer=buf,
            planner=planner,
            object_diameter=0.44,
            init_state=st0 if t == 0 else None,
        )
        env_steps += 1
    assert env_steps == 8
    assert n_plan["n"] == 8
    assert clock.n_updates == 2
    assert buf.n_replans == 8


def test_a_sweep_state_every_tick_planner_strided():
    from aprwm_v0.rtwx_mr0 import step_multirate

    clock = WorldStateClock(stride=1)
    buf = ActionChunkBuffer(stride=4)
    n_plan = {"n": 0}

    def planner(_inp):
        n_plan["n"] += 1
        return np.arange(8).reshape(8, 1).astype(np.float64)

    st0 = ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    for t in range(8):
        step_multirate(
            t,
            robot_state=np.zeros(1),
            cloud=np.zeros((8, 3)),
            goal=None,
            clock=clock,
            buffer=buf,
            planner=planner,
            object_diameter=0.44,
            init_state=st0 if t == 0 else None,
        )
    assert n_plan["n"] == 2
    assert clock.n_updates == 8
    assert buf.n_replans == 2


def test_shadow_does_not_count():
    buf = ActionChunkBuffer(stride=4)
    n = {"n": 0}

    def planner(_i):
        n["n"] += 1
        return np.zeros((8, 1))

    buf.action(0, {}, planner)
    shadow = planner({})
    assert shadow.shape[0] == 8
    assert buf.n_replans == 1
    assert n["n"] == 2


def test_planning_seed_deterministic():
    from aprwm_v0.rtwx_mr0 import planning_seed

    a = planning_seed(48601, "place_empty_cup", 0, 4)
    b = planning_seed(48601, "place_empty_cup", 0, 4)
    c = planning_seed(48601, "place_empty_cup", 0, 8)
    assert a == b
    assert a != c


def test_noninferiority():
    def n_rows(n_ok, n_fail, task="place_empty_cup"):
        ok = [{"task": task, "outcome": {"success": True, "safety_violation": False}}] * n_ok
        bad = [{"task": task, "outcome": {"success": False, "safety_violation": False}}] * n_fail
        return ok + bad

    b0 = n_rows(100, 0)
    assert noninferior(n_rows(100, 0), b0)
    assert noninferior(n_rows(96, 4), b0)
    assert noninferior(n_rows(94, 6), b0) is False
    b0m = n_rows(200, 0, "place_empty_cup") + n_rows(50, 0, "stamp_seal")
    k_task = n_rows(200, 0, "place_empty_cup") + n_rows(44, 6, "stamp_seal")
    assert noninferior(k_task, b0m) is False

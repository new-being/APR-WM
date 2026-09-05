"""RTWX-AC0 unit tests (no RoboTwin)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aprwm_v0.cli import build_parser
from aprwm_v0.policy.chunk_dataset import ActionChunkDataset, split_episode_indices, windows_from_episode
from aprwm_v0.policy.explicit_chunk_mlp import ExplicitChunkPolicy
from aprwm_v0.policy.process_features import ProcessEncoder
from aprwm_v0.policy.task_embedding import STATE_DIM, pack_state
from aprwm_v0.rtwx_ac0 import select_winner


def _frame(q=None):
    z3 = np.zeros(3)
    z4 = np.array([1.0, 0.0, 0.0, 0.0])
    qq = np.zeros(14) if q is None else np.asarray(q, dtype=np.float64).reshape(14)
    return {
        "q": qq,
        "qd": np.zeros(14),
        "obj_p": z3,
        "obj_q": z4,
        "goal_p": np.array([0.1, 0.0, 0.0]),
        "goal_q": z4,
        "ee_left_p": z3,
        "ee_right_p": z3,
        "ee_left_q": z4,
        "ee_right_q": z4,
        "grip_l": np.array([1.0]),
        "grip_r": np.array([1.0]),
        "cabinet_q": np.zeros(1),
    }


def test_cli_ac0():
    assert build_parser().parse_args(["rtwx-ac0"]).command == "rtwx-ac0"


def test_episode_split_disjoint():
    rng = np.random.default_rng(0)
    sp = split_episode_indices(50, rng, 40, 5, 5)
    assert len(sp["train"]) == 40
    assert set(sp["train"]).isdisjoint(sp["val"])
    assert set(sp["val"]).isdisjoint(sp["test"])


def test_windows_after_split_and_discard_tail():
    frames = [_frame() for _ in range(10)]
    ep = {"task": "place_empty_cup", "frames": frames, "q": np.zeros((10, 14)), "a": np.zeros((10, 14))}
    w = windows_from_episode(ep, horizon=8)
    assert len(w) == 3
    assert w[0]["chunk"].shape == (8, 14)


def test_process_no_future():
    enc = ProcessEncoder()
    st = _frame()
    with pytest.raises(AssertionError):
        enc.encode("place_empty_cup", st, future_state={"x": 1})
    with pytest.raises(AssertionError):
        enc.encode("place_empty_cup", st, episode_length=10)
    with pytest.raises(AssertionError):
        enc.encode("place_empty_cup", st, normalized_time=0.5)
    z = enc.encode("place_empty_cup", st)
    assert z.shape == (5,)
    assert z[1] in (0.0, 1.0)


def test_pack_dim_and_mlp_one_forward():
    st = pack_state(_frame())
    assert st.shape == (STATE_DIM,)
    m = ExplicitChunkPolicy(STATE_DIM, 5, use_process=False)
    y = m(torch.zeros(2, STATE_DIM), torch.zeros(2, dtype=torch.long), torch.zeros(2, 5))
    assert y.shape == (2, 8, 14)
    m1 = ExplicitChunkPolicy(STATE_DIM, 5, use_process=True)
    y1 = m1(torch.zeros(2, STATE_DIM), torch.zeros(2, dtype=torch.long), torch.zeros(2, 5))
    assert y1.shape == (2, 8, 14)


def test_bakeoff_jobs_three_tasks():
    from aprwm_v0.rtwx_ac0 import bakeoff_job_list

    jobs = bakeoff_job_list(16)
    assert len(jobs) == 48
    assert jobs[0] == ("place_empty_cup", 50601)
    assert jobs[16][0] == "put_object_cabinet"


def test_winner_complexity_margin():
    def br(p, m, s, l1):
        return {"roll": {"pooled": p, "min_task": m, "safety": s}, "metrics": {"val": {"l1": l1}}}

    sel = select_winner(br(0.50, 0.40, 0.0, 0.02), br(0.53, 0.40, 0.0, 0.01))
    assert sel["winner"] == "B0"
    assert sel["reason"] == "complexity_tie_margin"
    sel2 = select_winner(br(0.40, 0.20, 0.0, 0.05), br(0.55, 0.30, 0.0, 0.04))
    assert sel2["winner"] == "B1"

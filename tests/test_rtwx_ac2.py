"""AC2 history alignment, no future leakage, verdict, CLI."""

from __future__ import annotations

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.data.history_windows import left_pad_past_actions, left_pad_states
from aprwm_v0.policy.history_buffer import PolicyHistory
from aprwm_v0.policy.single_step_mlp import SingleStepMLP
from aprwm_v0.rtwx_ac2 import ac2_verdict, history_alignment_ok, qualified


def test_cli_ac2():
    assert build_parser().parse_args(["rtwx-ac2"]).command == "rtwx-ac2"


def test_g4_history_alignment_no_future():
    states = np.arange(5 * 2, dtype=np.float64).reshape(5, 2)
    actions = np.arange(5 * 3, dtype=np.float64).reshape(5, 3) + 100
    reset = np.array([-1.0, -2.0, -3.0])
    st = left_pad_states(states, 3, length=4)
    pa = left_pad_past_actions(actions, 3, reset, length=3)
    assert np.allclose(st, states[0:4])
    assert np.allclose(pa, actions[0:3])
    assert float(pa.max()) < 100 + 3 * 3  # a3 not in input; a3 first elem is 100+9=109, a2 first is 106
    assert 109 not in pa.reshape(-1)
    assert history_alignment_ok()


def test_action_indices_strictly_past():
    actions = np.ones((6, 14))
    reset = np.zeros(14)
    for t in range(6):
        pa = left_pad_past_actions(actions, t, reset, length=3)
        assert pa.shape == (3, 14)


def test_history_buffer_push_after_execute():
    h = PolicyHistory()
    s0 = np.zeros(4)
    h.reset(s0, np.ones(3) * -1)
    h.update_after_step(np.ones(4), np.array([7.0, 7.0, 7.0]))
    assert np.allclose(h.state_stack()[-1], 1)
    assert np.allclose(h.action_stack()[-1], 7)


def test_mlp_shapes():
    b0 = SingleStepMLP("B0")
    b1 = SingleStepMLP("B1")
    b2 = SingleStepMLP("B2")
    import torch

    s = torch.zeros(2, 4, 57)
    t = torch.tensor([0, 1])
    a = torch.zeros(2, 3, 14)
    assert b0(s, t, None).shape == (2, 14)
    assert b1(s, t, None).shape == (2, 14)
    assert b2(s, t, a).shape == (2, 14)


def test_verdict_patterns():
    z = {"pooled": 0.0, "safety": 0.0, "n_tasks_gt0": 0}
    hi = {"pooled": 0.35, "safety": 0.0, "n_tasks_gt0": 2}
    v = ac2_verdict(hi, {"pooled": 0.36, "safety": 0.0, "n_tasks_gt0": 2}, {"pooled": 0.35, "safety": 0.0, "n_tasks_gt0": 2})
    assert v["pattern"] == "single_step_output_supported"
    assert v["winner"] == "B0"
    v = ac2_verdict(z, hi, {"pooled": 0.37, "safety": 0.0, "n_tasks_gt0": 2})
    assert v["pattern"] == "state_history_utility_supported"
    v = ac2_verdict(z, z, hi)
    assert v["pattern"] == "action_history_utility_supported"
    v = ac2_verdict(z, z, z)
    assert v["pattern"] == "short_memory_policy_insufficient"
    assert not qualified(z)

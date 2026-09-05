"""TASK-XL causal encoder API: prefix-only, no future, R10/X1 refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.task_xl import (
    DIFFUSION_TRAIN,
    SUPERVISION_FIRST_INSTRUMENT,
    UNLOCKS_R10,
    UNLOCKS_TASK_X1,
    CausalHistory,
    CausalLeakError,
    CausalTaskProgressEncoder,
    action_attractor_inputs_ok,
    physics_inputs_ok,
    refuse_locked_output,
)


def test_locks_are_hard():
    assert SUPERVISION_FIRST_INSTRUMENT == "B"
    assert DIFFUSION_TRAIN is False
    assert UNLOCKS_R10 is False
    assert UNLOCKS_TASK_X1 is False


def test_r10_and_x1_paths_refused(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        refuse_locked_output(tmp_path / "runs" / "r10_c0" / "x")
    with pytest.raises(RuntimeError, match="TASK-X1"):
        refuse_locked_output("runs/task_x1/formal")


def test_future_state_rejected():
    enc = CausalTaskProgressEncoder()
    s = np.zeros((8, 3))
    a = np.zeros((7, 2))
    with pytest.raises(CausalLeakError, match="s_"):
        from aprwm_v0.task_xl import assert_causal_prefix

        assert_causal_prefix(2, s, a[:2])
    h = CausalHistory.from_episode(2, s, a)
    z_g = enc.encode_task(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(CausalLeakError, match="future"):
        enc.encode_progress(h, z_g, s_future=s[3:])


def test_future_action_as_input_rejected():
    enc = CausalTaskProgressEncoder()
    s = np.linspace(0, 1, 12).reshape(6, 2)
    a = np.linspace(0, 1, 10).reshape(5, 2)
    h = CausalHistory.from_episode(3, s, a)
    z_g = enc.encode_task(np.array([0.2, 0.8]))
    with pytest.raises(CausalLeakError):
        enc.encode_progress(h, z_g, a_future=a[3:])
    with pytest.raises(CausalLeakError):
        enc.encode_progress(h, z_g, success_future=True)


def test_prefix_recoverable_equals_full_cut():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(10, 5))
    a = rng.normal(size=(9, 3))
    g = np.array([0.1, 0.2, 0.3])
    enc = CausalTaskProgressEncoder()
    assert enc.is_prefix_recoverable(s, a, g)
    z_g = enc.encode_task(g)
    h0 = CausalHistory.from_episode(4, s[:5], a[:4])
    h1 = CausalHistory.from_episode(4, s, a)
    assert np.allclose(enc.encode_progress(h0, z_g), enc.encode_progress(h1, z_g))


def test_task_latent_depends_on_g_not_s_t():
    enc = CausalTaskProgressEncoder()
    z1 = enc.encode_task(np.array([1.0, 0.0]))
    z2 = enc.encode_task(np.array([0.0, 1.0]))
    assert not np.allclose(z1, z2)


def test_state_box_split():
    assert physics_inputs_ok({"s_phy", "theta_E"})
    assert not physics_inputs_ok({"s_phy", "z_g"})
    assert action_attractor_inputs_ok({"s_phy", "z_g", "z_p"})

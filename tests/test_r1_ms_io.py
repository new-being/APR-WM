import numpy as np
import pytest
import torch

from aprwm_v0.r1_ms_io import (
    R1MSIOConfig,
    _branch_actions,
    _clone_tree,
    _finite_difference_rmse,
)


def test_clone_tree_detaches_tensor_storage():
    original = {"a": torch.tensor([1.0]), "b": [torch.tensor([2.0])]}
    cloned = _clone_tree(original)
    original["a"][0] = 9.0
    original["b"][0][0] = 8.0
    assert cloned["a"].item() == 1.0
    assert cloned["b"][0].item() == 2.0


def test_counterfactual_actions_change_only_declared_intervention():
    nominal = np.zeros((4, 3), dtype=np.float32)
    branches = _branch_actions(nominal, 0.5)
    np.testing.assert_array_equal(branches["nominal"], nominal)
    assert branches["positive"][0, 0] == 0.5
    assert branches["negative"][0, 0] == -0.5
    np.testing.assert_array_equal(branches["positive"][1:], nominal[1:])


def test_midpoint_qvel_differentiation_is_exact_for_constant_velocity():
    rollout = {
        "qpos": np.array([[0.0], [0.1], [0.2]], dtype=np.float32),
        "qvel": np.array([[1.0], [1.0], [1.0]], dtype=np.float32),
    }
    assert _finite_difference_rmse(rollout, 0.1) == pytest.approx(0.0, abs=1e-6)


def test_h32_is_not_a_decision_horizon():
    config = R1MSIOConfig()
    assert 32 in config.horizons
    assert 32 not in config.decision_horizons

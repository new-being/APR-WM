from pathlib import Path

import torch

from aprwm_v0.r0 import (
    LaptopStateAdapter,
    _ensure_drag_candidate,
    counterfactual_branch_manifest,
    residual_assimilation_ratio,
)


class _MockHinge:
    def get_qpos(self):
        return [0.4]

    def get_qvel(self):
        return [-0.2]


def test_laptop_adapter_extracts_oracle_local_state():
    state = LaptopStateAdapter.extract(_MockHinge(), 0.7)
    torch.testing.assert_close(state.as_tensor(), torch.tensor((0.4, -0.2, 0.7)))
    torch.testing.assert_close(
        LaptopStateAdapter.model_features(state), torch.tensor((0.4, -0.2, 0.7))
    )


def test_residual_assimilation_ratio_has_expected_endpoints():
    assert residual_assimilation_ratio(1.0, 0.25) == 0.75


def test_counterfactual_manifest_has_replay_branch_product():
    rows = counterfactual_branch_manifest(
        task="open_laptop",
        seeds=(1, 2, 3),
        windows_per_seed=4,
        branches_per_window=5,
        horizons=(1, 4, 8, 16),
    )
    assert len(rows) == 3 * 4 * 5
    assert all(row["replay_prefix"] for row in rows)


def test_controlled_candidate_insertion_only_changes_c1():
    proposed = torch.tensor(((0, 1, 2), (0, 1, 2), (3, 4, 5)))
    regime = torch.tensor((0, 1, 2))
    output = _ensure_drag_candidate(proposed, regime)
    torch.testing.assert_close(output[0], proposed[0])
    assert 4 in output[1].tolist()
    torch.testing.assert_close(output[2], proposed[2])

import pytest
import torch

from aprwm_v0.r1_ms import (
    DRAWER_DEVELOPMENT_ASSETS,
    DRAWER_DYNAMICS_METADATA,
    GateMeasurements,
    SupportSignals,
    SupportState,
    asset_directory_complete,
    all_c0_assets_individually_close,
    classify_support,
    counterfactual_manifest,
    drawer_state_from_articulation,
    drawer_target_state_from_env,
    evaluate_ordered_gates,
    frozen_protocol,
    generalized_residual_power,
    stage_manifest,
)


class _MockDrawer:
    def get_qpos(self):
        return torch.tensor([[0.2, 0.7]])

    def get_qvel(self):
        return torch.tensor([[0.1, -0.3]])

    def get_qacc(self):
        return torch.tensor([[0.4, 0.5]])

    def get_qf(self):
        return torch.tensor([[1.2, -0.8]])


class _MockDrawerNoAcceleration:
    def get_qpos(self):
        return torch.tensor([[0.2]])

    def get_qvel(self):
        return torch.tensor([[0.3]])

    def get_qf(self):
        return torch.tensor([[1.2]])


class _MockMixedCabinets:
    def get_qpos(self):
        return torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def get_qvel(self):
        return torch.tensor([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]])

    def get_qacc(self):
        return torch.tensor([[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]])

    def get_qf(self):
        return torch.tensor([[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]])


class _MockTargetJoint:
    active_index = torch.tensor([2, 0])


class _MockHandleLink:
    joint = _MockTargetJoint()


class _MockDrawerEnv:
    cabinet = _MockMixedCabinets()
    handle_link = _MockHandleLink()


def test_drawer_state_extracts_selected_oracle_dof():
    state = drawer_state_from_articulation(_MockDrawer(), joint_index=1)
    torch.testing.assert_close(
        state.stacked(), torch.tensor([[0.7, -0.3, 0.5, -0.8]])
    )


def test_drawer_acceleration_falls_back_to_finite_difference():
    state = drawer_state_from_articulation(
        _MockDrawerNoAcceleration(), previous_qvel=torch.tensor([0.1]), dt=0.1
    )
    torch.testing.assert_close(state.qacc, torch.tensor([2.0]))


def test_drawer_target_joint_is_gathered_per_asset_not_fixed_index():
    state = drawer_target_state_from_env(_MockDrawerEnv())
    torch.testing.assert_close(state.q, torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(state.qvel, torch.tensor([1.3, 1.4]))
    torch.testing.assert_close(state.qacc, torch.tensor([2.3, 2.4]))
    torch.testing.assert_close(state.qf, torch.tensor([3.3, 3.4]))


def test_generalized_power_supports_multi_dof_and_checks_shape():
    power = generalized_residual_power(
        torch.tensor([[1.0, -2.0]]), torch.tensor([[0.5, 0.25]])
    )
    torch.testing.assert_close(power, torch.tensor([0.0]))
    with pytest.raises(ValueError):
        generalized_residual_power(torch.zeros(2), torch.zeros(3))


def test_three_state_validity_is_not_collapsed_to_boolean():
    assert classify_support(SupportSignals(0.2)) is SupportState.MODELED
    assert classify_support(SupportSignals(0.01, near_joint_limit=True)) is SupportState.BOUNDARY
    assert classify_support(SupportSignals(0.2, grasp_constraint_active=True)) is SupportState.UNSUPPORTED


def test_gate_evaluation_stops_after_first_failure():
    rows = evaluate_ordered_gates(
        GateMeasurements(c0_false_revision_rate=0.02, c0_structured_residual=False)
    )
    assert rows[0]["status"] == "fail"
    assert all(row["status"] == "blocked_by_earlier_gate" for row in rows[1:])


def test_all_preregistered_gates_can_pass():
    rows = evaluate_ordered_gates(
        GateMeasurements(
            c0_false_revision_rate=0.0,
            c0_structured_residual=False,
            c1_exact_recovery_rate=0.95,
            accepted_stability=1.0,
            native_h16_gain=0.1,
            native_h32_catastrophic_reversal=False,
            acceptance_rate=0.3,
            c2_forced_wrong_revision_rate=0.05,
            history_gain_over_memoryless=0.01,
        )
    )
    assert all(row["status"] == "pass" for row in rows)


def test_h32_is_evaluation_only_and_branches_are_interventional():
    protocol = frozen_protocol()
    assert 32 in protocol["evaluation_horizons"]
    assert 32 not in protocol["utility_gate"]["horizons"]
    rows = counterfactual_manifest(seeds=(1,), windows_per_asset=2, branches_per_window=3)
    assert len(rows) == 6
    assert all(row["restore_with_state_dict"] for row in rows)
    assert all(row["simulator_executes_perturbation"] for row in rows)


def test_interrupted_empty_asset_directory_is_not_counted(tmp_path):
    asset = tmp_path / "1000"
    asset.mkdir()
    assert not asset_directory_complete(asset)
    (asset / "mobility.urdf").write_text("<robot/>", encoding="utf-8")
    assert asset_directory_complete(asset)


def test_c0_manifest_requires_three_by_three_per_asset_closure():
    c0 = stage_manifest()[0]
    assert c0["asset_ids"] == list(DRAWER_DEVELOPMENT_ASSETS)
    assert c0["seeds"] == [8101, 8111, 8121]
    assert c0["matrix"] == {
        "assets": 3,
        "seeds_per_asset": 3,
        "total_asset_seed_cells": 9,
    }
    assert c0["closure_granularity"] == "per_asset_before_aggregation"
    assert c0["go_rule"] == "all sampled assets individually close"
    assert set(c0["required_dynamics_metadata"]) == set(DRAWER_DYNAMICS_METADATA)


def test_c0_cannot_unlock_from_aggregate_or_partial_asset_success():
    assert not all_c0_assets_individually_close({"1000": True, "1040": True})
    assert not all_c0_assets_individually_close(
        {"1000": True, "1040": False, "1082": True}
    )
    assert all_c0_assets_individually_close(
        {"1000": True, "1040": True, "1082": True}
    )

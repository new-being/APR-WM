import torch

from aprwm_v0.r02 import (
    V6R02Config,
    _candidate_design,
    _normalize_fallback_features,
    _rollout_gate_metrics,
    _sample_rollout_queries,
)


def test_force_candidate_design_separates_tangent_and_physical_inputs():
    tangent = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    physical = torch.tensor([[[0.4, -0.2], [0.6, 0.3]]])
    proposed = torch.tensor([[4, 0, 1]])
    design = _candidate_design(tangent, physical, proposed)
    torch.testing.assert_close(design[0, :, 0, :2], tangent[0])
    torch.testing.assert_close(
        design[0, :, 0, 2], physical[0, :, 1] * physical[0, :, 1].abs()
    )


def test_fallback_normalization_maps_intervention_domain_to_v3_domain():
    physical = torch.tensor([[[0.18, -1.4], [0.93, 1.4]]])
    normalized = _normalize_fallback_features(physical)
    torch.testing.assert_close(
        normalized, torch.tensor([[[0.012, -0.22], [0.09, 0.22]]])
    )


def test_rollout_gate_requires_all_c1_horizons_to_improve():
    config = V6R02Config(horizons=(1, 4))
    rows = []
    for horizon in config.horizons:
        for model, c0, c1 in (
            ("physics", 0.001, 0.2),
            ("no_revision", 0.0015, 0.1),
            ("frozen_v6", 0.0014, 0.08),
        ):
            rows.extend(
                (
                    {"model": model, "regime": "c0_parameter", "horizon": horizon, "state_rmse": c0},
                    {"model": model, "regime": "c1_drag", "horizon": horizon, "state_rmse": c1},
                )
            )
    metrics = _rollout_gate_metrics(rows, config)
    assert metrics["c0_rollout_noninferiority_pass"]
    assert metrics["c1_rollout_improvement_all_horizons"]


def test_rollout_queries_stay_in_declared_local_window():
    state, _ = _sample_rollout_queries(256, torch.Generator().manual_seed(4))
    assert bool(((state[:, 0] >= 0.35) & (state[:, 0] <= 0.75)).all())
    assert bool((state[:, 1].abs() <= 0.40).all())
    assert bool((state[:, 2].abs() <= 0.20).all())

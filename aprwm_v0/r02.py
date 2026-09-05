from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .r0 import SapienHingeBackend, V6R0Config, _aggregate_seed_rows, _sample_local_points, residual_assimilation_ratio
from .train import resolve_device, seed_everything
from .v06 import _binary_auroc, _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import OPERATOR_COMPLEXITY, operator_library, orthogonal_operator_gain
from .v5 import _action_score, _fit_base_posterior, _update_base_belief, _update_candidate_belief
from .v6 import V6Config, _sequential_decision


DRAG_OPERATOR = 4
REGIMES = ("c0_parameter", "c1_drag")
MODELS = ("physics", "no_revision", "frozen_v6")


@dataclass
class V6R02Config:
    seeds: tuple[int, ...] = (13, 23, 33)
    episodes_per_regime: int = 24
    passive_context: int = 8
    discovery_count: int = 16
    action_candidates: int = 32
    selection_probes: int = 3
    validation_max: int = 32
    query_count: int = 8
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    proposal_topk: int = 3
    force_observation_noise: float = 0.002
    posterior_noise_floor: float = 0.01
    physics_dt: float = 1.0 / 250.0
    macro_steps: int = 4
    density_min: float = 110.0
    density_max: float = 230.0
    damping_min: float = 0.025
    damping_max: float = 0.075
    drag_strength: float = 0.12
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    proposal_complexity_price: float = 2.0e-4
    proposal_temperature: float = 0.10
    c0_false_revision_max: float = 0.01
    c0_noninferiority_epsilon: float = 0.002
    tangent_auroc_min: float = 0.80
    trigger_noise_multiplier: float = 1.35


def _backend_config(config: V6R02Config) -> V6R0Config:
    return V6R0Config(
        physics_dt=config.physics_dt,
        macro_steps=config.macro_steps,
        density_min=config.density_min,
        density_max=config.density_max,
        damping_min=config.damping_min,
        damping_max=config.damping_max,
        drag_strength=config.drag_strength,
    )


def _force_observations(
    backend: SapienHingeBackend,
    state: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physical = state[:, :2]
    tangent_rows = []
    residual_rows = []
    for row, hidden in zip(state, memory):
        residual, density_tangent, _ = backend.generalized_force_sample(
            float(row[0]), float(row[1]), float(row[2]), float(hidden)
        )
        tangent_rows.append((density_tangent, float(row[1])))
        residual_rows.append(residual)
    return (
        physical,
        torch.tensor(tangent_rows, dtype=torch.float32),
        torch.tensor(residual_rows, dtype=torch.float32),
    )


def _sample_rollout_queries(
    count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local counterfactuals that stay away from the hinge-limit constraint.

    Pinocchio supplies unconstrained rigid-body dynamics while PhysX also
    enforces the [-0.1, 1.5] joint limit.  R0.2 tests force-space revision, not
    contact with the limit, so its fixed 256 ms window is preregistered inside
    the constraint-free interior.
    """
    q = 0.35 + 0.40 * torch.rand(count, generator=generator)
    velocity = 0.80 * torch.rand(count, generator=generator) - 0.40
    torque = 0.40 * torch.rand(count, generator=generator) - 0.20
    memory = torch.where(
        torch.rand(count, generator=generator) > 0.5,
        torch.ones(count),
        -torch.ones(count),
    )
    return torch.stack((q, velocity, torque), dim=-1), memory


def _collect_seed(
    config: V6R02Config, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 180_000)
    backend_config = _backend_config(config)
    names = (
        "passive_physical", "passive_tangent", "passive_y",
        "discovery_physical", "discovery_tangent", "discovery_y",
        "action_physical", "action_tangent", "action_y",
        "validation_physical", "validation_tangent", "validation_y",
        "query_state", "query_memory", "rollout_truth",
    )
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    regimes = []
    densities = []
    dampings = []
    for regime in range(2):
        for _ in range(config.episodes_per_regime):
            density = config.density_min + (
                config.density_max - config.density_min
            ) * float(torch.rand((), generator=generator))
            damping = config.damping_min + (
                config.damping_max - config.damping_min
            ) * float(torch.rand((), generator=generator))
            backend = SapienHingeBackend(
                density=density,
                damping=damping,
                regime=regime,
                config=backend_config,
            )
            for prefix, count, sampling_regime in (
                ("passive", config.passive_context, "passive"),
                ("discovery", config.discovery_count, "discovery"),
                ("action", config.action_candidates, "diagnostic"),
                ("validation", config.validation_max, "validation"),
            ):
                state, memory = _sample_local_points(count, sampling_regime, generator)
                physical, tangent, target = _force_observations(backend, state, memory)
                fields[f"{prefix}_physical"].append(physical)
                fields[f"{prefix}_tangent"].append(tangent)
                fields[f"{prefix}_y"].append(target)

            query_state, query_memory = _sample_rollout_queries(
                config.query_count, generator
            )
            truth = []
            for horizon in config.horizons:
                horizon_rows = []
                for row, hidden in zip(query_state, query_memory):
                    q, velocity = backend.transition(
                        float(row[0]), float(row[1]), float(row[2]), float(hidden),
                        horizon=horizon,
                    )
                    horizon_rows.append((q, velocity))
                truth.append(torch.tensor(horizon_rows, dtype=torch.float32))
            fields["query_state"].append(query_state)
            fields["query_memory"].append(query_memory)
            fields["rollout_truth"].append(torch.stack(truth, dim=1))
            regimes.append(regime)
            densities.append(density)
            dampings.append(damping)

    output = {name: torch.stack(values).to(device) for name, values in fields.items()}
    output["regime"] = torch.tensor(regimes, dtype=torch.long, device=device)
    output["density"] = torch.tensor(densities, dtype=torch.float32, device=device)
    output["damping"] = torch.tensor(dampings, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 181_000)
    for name in ("passive_y", "discovery_y", "action_y", "validation_y"):
        output[name] = output[name] + config.force_observation_noise * torch.randn(
            output[name].shape, generator=noise_generator, device=device
        )
    return output


def _candidate_design(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    proposed: torch.Tensor,
) -> torch.Tensor:
    operators = operator_library(physical)
    selected = operators.gather(
        -1, proposed[:, None, :].expand(-1, physical.shape[1], -1)
    )
    base = tangent[:, :, None, :].expand(-1, -1, proposed.shape[1], -1)
    return torch.cat((base, selected.unsqueeze(-1)), dim=-1)


def _fit_candidate_posteriors(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    target: torch.Tensor,
    proposed: torch.Tensor,
    config: V6R02Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = _candidate_design(tangent, physical, proposed).permute(0, 2, 1, 3)
    variance = max(
        config.force_observation_noise, config.posterior_noise_floor
    ) ** 2
    means = []
    covariances = []
    for index in range(proposed.shape[1]):
        candidate = design[:, index]
        scale = candidate.square().mean(dim=1).sqrt().clamp_min(1.0e-8)
        normalized = candidate / scale.unsqueeze(1)
        gram = torch.einsum("bni,bnj->bij", normalized, normalized)
        eye = torch.eye(3, dtype=candidate.dtype, device=candidate.device)
        covariance_normalized = torch.linalg.inv(
            gram / variance + config.ridge * eye
        )
        natural = torch.einsum("bni,bn->bi", normalized, target) / variance
        mean_normalized = torch.einsum(
            "bij,bj->bi", covariance_normalized, natural
        )
        inverse_scale = scale.reciprocal()
        means.append(mean_normalized * inverse_scale)
        covariances.append(
            covariance_normalized
            * inverse_scale.unsqueeze(-1)
            * inverse_scale.unsqueeze(-2)
        )
    return torch.stack(means, dim=1), torch.stack(covariances, dim=1)


def _predictive_moments(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    proposed: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = _candidate_design(tangent, physical, proposed)
    mean = torch.einsum("bank,bnk->ban", design, means)
    variance = torch.einsum(
        "bank,bnkl,banl->ban", design, covariances, design
    ).clamp_min(0.0)
    return mean, variance


def _normalize_fallback_features(physical: torch.Tensor) -> torch.Tensor:
    q = 0.012 + (physical[..., 0] - 0.18) * (0.078 / 0.75)
    velocity = physical[..., 1] * (0.22 / 1.4)
    return torch.stack((q, velocity), dim=-1)


def _model_rollout(
    config: V6R02Config,
    data: dict[str, torch.Tensor],
    force_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    q = data["query_state"][..., 0].clone()
    velocity = data["query_state"][..., 1].clone()
    torque = data["query_state"][..., 2]
    device = q.device
    backend_config = _backend_config(config)
    backends = [
        SapienHingeBackend(
            density=float(density), damping=float(damping), regime=0,
            config=backend_config,
        )
        for density, damping in zip(data["density"], data["damping"])
    ]
    snapshots = []
    stable_snapshots = []
    stable = torch.ones_like(q, dtype=torch.bool)
    horizon_steps = {h * config.macro_steps: h for h in config.horizons}
    for micro_step in range(1, max(config.horizons) * config.macro_steps + 1):
        physical = torch.stack((q, velocity), dim=-1)
        base_acceleration_force = torque - data["damping"][:, None] * velocity
        tangent = torch.stack(
            (
                base_acceleration_force / data["density"][:, None],
                velocity,
            ),
            dim=-1,
        )
        residual_force = force_model(tangent, physical)
        acceleration = torch.empty_like(q)
        for episode, backend in enumerate(backends):
            pinocchio = backend.pinocchio
            damping = float(data["damping"][episode])
            for query in range(q.shape[1]):
                q_value = float(q[episode, query])
                velocity_value = float(velocity[episode, query])
                torque_value = float(torque[episode, query])
                h = float(
                    pinocchio.compute_inverse_dynamics(
                        (q_value,), (velocity_value,), (0.0,)
                    )[0]
                )
                generalized_force = (
                    h + torque_value - damping * velocity_value
                    + float(residual_force[episode, query])
                )
                acceleration[episode, query] = float(
                    pinocchio.compute_forward_dynamics(
                        (q_value,), (velocity_value,), (generalized_force,)
                    )[0]
                )
        q = q + config.physics_dt * velocity
        velocity = velocity + config.physics_dt * acceleration
        step_stable = (
            torch.isfinite(q) & torch.isfinite(velocity)
            & (q.abs() < 10.0) & (velocity.abs() < 50.0)
        )
        stable &= step_stable
        q = torch.nan_to_num(q, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        velocity = torch.nan_to_num(
            velocity, nan=0.0, posinf=50.0, neginf=-50.0
        ).clamp(-50, 50)
        if micro_step in horizon_steps:
            snapshots.append(torch.stack((q, velocity), dim=-1))
            stable_snapshots.append(stable.clone())
    return torch.stack(snapshots, dim=2), torch.stack(stable_snapshots, dim=-1)


def _evaluate_seed(
    config: V6R02Config, seed: int, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _collect_seed(config, seed, device)
    regime = data["regime"]
    c0 = regime == 0
    c1 = regime == 1
    initial_theta = _ridge_fit(
        data["passive_tangent"], data["passive_y"], config.ridge
    )
    discovery_residual = data["discovery_y"] - torch.einsum(
        "bni,bi->bn", data["discovery_tangent"], initial_theta
    )
    _, residual_orthogonal, _ = tangent_decomposition(
        data["discovery_tangent"], discovery_residual, config.ridge
    )
    tangent_score = residual_orthogonal.square().mean(dim=-1).sqrt()
    magnitude_score = discovery_residual.square().mean(dim=-1).sqrt()
    triggered = (
        tangent_score
        > config.trigger_noise_multiplier * config.force_observation_noise
    )
    gain = orthogonal_operator_gain(
        data["discovery_tangent"],
        residual_orthogonal,
        operator_library(data["discovery_physical"]),
        ridge=config.ridge,
    )
    complexity = torch.tensor(
        OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device
    )
    proposal_score = gain - config.proposal_complexity_price * complexity
    proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    proposal_coverage = (proposed[c1] == DRAG_OPERATOR).any(dim=-1)
    correct_position = torch.where(
        c1,
        (proposed == DRAG_OPERATOR).float().argmax(dim=-1),
        torch.full_like(regime, -1),
    )

    train_tangent = torch.cat(
        (data["passive_tangent"], data["discovery_tangent"]), dim=1
    )
    train_physical = torch.cat(
        (data["passive_physical"], data["discovery_physical"]), dim=1
    )
    train_y = torch.cat((data["passive_y"], data["discovery_y"]), dim=1)
    v6 = V6Config(
        episodes=train_y.shape[0],
        passive_context=config.passive_context,
        discovery_count=config.discovery_count,
        action_candidates=config.action_candidates,
        selection_probes=config.selection_probes,
        validation_max=config.validation_max,
        proposal_topk=config.proposal_topk,
        posterior_noise_floor=config.posterior_noise_floor,
        ridge=config.ridge,
        residual_ridge=config.residual_ridge,
    )
    means, covariances = _fit_candidate_posteriors(
        train_tangent, train_physical, train_y, proposed, config
    )
    base_mean, base_covariance = _fit_base_posterior(
        train_tangent, train_y, config.force_observation_noise, v6
    )
    log_probabilities = torch.log_softmax(
        proposal_score.gather(1, proposed) / config.proposal_temperature, dim=-1
    )
    observation_variance = max(
        config.force_observation_noise, config.posterior_noise_floor
    ) ** 2
    used = torch.zeros_like(data["action_y"], dtype=torch.bool)
    active = triggered.clone()
    for _ in range(config.selection_probes):
        pool_means, pool_variances = _predictive_moments(
            data["action_tangent"], data["action_physical"], proposed,
            means, covariances,
        )
        score = _action_score(
            "weighted", pool_means, pool_variances, log_probabilities.exp(),
            observation_variance, correct_position, False,
        ).masked_fill(used, float("-inf"))
        action_index = score.argmax(dim=-1)
        selected_tangent = data["action_tangent"].gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        selected_physical = data["action_physical"].gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        selected_y = data["action_y"].gather(
            1, action_index.unsqueeze(-1)
        ).squeeze(-1)
        design = _candidate_design(
            selected_tangent.unsqueeze(1), selected_physical.unsqueeze(1), proposed
        ).squeeze(1)
        means, covariances, log_probabilities = _update_candidate_belief(
            means, covariances, log_probabilities, design, selected_y,
            observation_variance, active,
        )
        base_mean, base_covariance = _update_base_belief(
            base_mean, base_covariance, selected_tangent, selected_y,
            observation_variance, active,
        )
        used[torch.arange(regime.shape[0], device=device), action_index] = True

    order = log_probabilities.topk(2, dim=-1).indices
    selected_position, competitor_position = order[:, 0], order[:, 1]
    selected_operator = proposed.gather(1, selected_position[:, None]).squeeze(-1)
    selection_correct = selected_operator == DRAG_OPERATOR
    validation_design = _candidate_design(
        data["validation_tangent"], data["validation_physical"], proposed
    )
    validation_all = torch.einsum("bank,bnk->ban", validation_design, means)
    selected_prediction = validation_all.gather(
        -1, selected_position[:, None, None].expand(-1, config.validation_max, 1)
    ).squeeze(-1)
    competitor_prediction = validation_all.gather(
        -1, competitor_position[:, None, None].expand(-1, config.validation_max, 1)
    ).squeeze(-1)
    base_prediction = torch.einsum(
        "bni,bi->bn", data["validation_tangent"], base_mean
    )
    accepted, rejected, deferred, samples = _sequential_decision(
        (data["validation_y"] - base_prediction).square(),
        (data["validation_y"] - competitor_prediction).square(),
        (data["validation_y"] - selected_prediction).square(),
        complexity[selected_operator],
        config.force_observation_noise**2,
        v6,
        method="bf",
    )
    # The frozen V3 trigger precedes proposal/selection/validation. Episodes
    # without structural evidence keep explicit physics and spend no
    # validation budget; fallback is reserved for triggered-but-unassimilated
    # cases.
    accepted &= triggered
    rejected &= triggered
    deferred &= triggered
    samples = torch.where(triggered, samples, torch.zeros_like(samples))
    exact = c1 & selection_correct & accepted
    selected_mean = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)
    train_base_prediction = torch.einsum("bni,bi->bn", train_tangent, base_mean)
    fitted_residual = train_y - train_base_prediction
    normalized_train = _normalize_fallback_features(train_physical)

    def physics_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device)

    def base_force(tangent: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqi,bi->bq", tangent, base_mean)

    def fallback_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return base_force(tangent) + _residual_predict(
            normalized_train,
            fitted_residual,
            _normalize_fallback_features(physical),
            config.residual_ridge,
        ).clamp(-0.4, 0.4)

    def candidate_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        design = _candidate_design(tangent, physical, proposed)
        all_predictions = torch.einsum("bank,bnk->ban", design, means)
        return all_predictions.gather(
            -1, selected_position[:, None, None].expand(-1, physical.shape[1], 1)
        ).squeeze(-1)

    def revised_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.where(
            accepted[:, None],
            candidate_force(tangent, physical),
            torch.where(
                triggered[:, None],
                fallback_force(tangent, physical),
                torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device),
            ),
        )

    rollout_outputs = {
        "physics": _model_rollout(config, data, physics_force),
        "no_revision": _model_rollout(config, data, fallback_force),
        "frozen_v6": _model_rollout(config, data, revised_force),
    }
    horizon_rows: list[dict[str, Any]] = []
    truth = data["rollout_truth"]
    scale = torch.tensor((1.0, 2.0), device=device)
    for model, (prediction, stable) in rollout_outputs.items():
        for index, horizon in enumerate(config.horizons):
            error = (prediction[:, :, index] - truth[:, :, index]) / scale
            for regime_index, regime_name in enumerate(REGIMES):
                mask = regime == regime_index
                horizon_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "regime": regime_name,
                        "horizon": horizon,
                        "state_rmse": float(torch.sqrt(error[mask].square().mean())),
                        "stable_fraction": float(stable[mask, :, index].float().mean()),
                    }
                )

    after = float((~exact)[c1].float().mean())
    summary = {
        "seed": seed,
        "tangent_detection_auroc": _binary_auroc(tangent_score, c1),
        "magnitude_detection_auroc": _binary_auroc(magnitude_score, c1),
        "c0_structural_trigger_rate": float(triggered[c0].float().mean()),
        "c1_structural_trigger_recall": float(triggered[c1].float().mean()),
        "natural_candidate_coverage_c1": float(proposal_coverage.float().mean()),
        "selection_accuracy_c1": float(selection_correct[c1].float().mean()),
        "acceptance_power_c1_correct_selection": float(
            accepted[c1 & selection_correct].float().mean()
        ),
        "exact_operator_recovery_c1": float(exact[c1].float().mean()),
        "false_revision_c0": float(accepted[c0].float().mean()),
        "c0_defer_rate": float(deferred[c0].float().mean()),
        "validation_samples": float(samples.mean()),
        "validation_samples_c1": float(samples[c1].mean()),
        "defer_rate": float(deferred.float().mean()),
        "residual_usage_before_c1": 1.0,
        "residual_usage_after_correct_revision_c1": after,
        "residual_assimilation_ratio_c1": residual_assimilation_ratio(1.0, after),
        "selected_operator_coefficient_c1": float(selected_mean[c1, -1].mean()),
    }
    return summary, horizon_rows


def _rollout_gate_metrics(
    horizon_rows: list[dict[str, Any]], config: V6R02Config
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in horizon_rows:
        key = (str(row["model"]), str(row["regime"]), int(row["horizon"]))
        grouped.setdefault(key, []).append(float(row["state_rmse"]))
    means = {key: statistics.mean(values) for key, values in grouped.items()}
    c0_differences = [
        means[("frozen_v6", "c0_parameter", horizon)]
        - means[("physics", "c0_parameter", horizon)]
        for horizon in config.horizons
    ]
    c1_improvements = [
        means[("no_revision", "c1_drag", horizon)]
        - means[("frozen_v6", "c1_drag", horizon)]
        for horizon in config.horizons
    ]
    return {
        "c0_max_v6_minus_physics_rmse": max(c0_differences),
        "c0_rollout_noninferiority_pass": max(c0_differences)
        <= config.c0_noninferiority_epsilon,
        "c1_v6_minus_no_revision_by_horizon": {
            str(horizon): -improvement
            for horizon, improvement in zip(config.horizons, c1_improvements)
        },
        "c1_rollout_improvement_all_horizons": all(
            improvement > 0.0 for improvement in c1_improvements
        ),
        "c1_h16_improvement": c1_improvements[-1],
    }


def run_v6r02(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R02Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6R02Config()
    device = resolve_device(device_name)
    seed_rows = []
    horizon_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        metrics, rollouts = _evaluate_seed(cfg, seed, device)
        seed_rows.append(metrics)
        horizon_rows.extend(rollouts)
    root = Path(output_dir)
    _write_csv(root / "seed_metrics.csv", seed_rows)
    _write_csv(root / "metrics_aggregate.csv", _aggregate_seed_rows(seed_rows))
    _write_csv(root / "rollout_horizons.csv", horizon_rows)
    rollout_gates = _rollout_gate_metrics(horizon_rows, cfg)
    false_revision = statistics.mean(float(row["false_revision_c0"]) for row in seed_rows)
    tangent_auroc = statistics.mean(float(row["tangent_detection_auroc"]) for row in seed_rows)
    assimilation = statistics.mean(float(row["residual_assimilation_ratio_c1"]) for row in seed_rows)
    gates = {
        "c0_false_revision_pass": false_revision < cfg.c0_false_revision_max,
        "c0_rollout_noninferiority_pass": rollout_gates["c0_rollout_noninferiority_pass"],
        "c1_tangent_separation_pass": tangent_auroc >= cfg.tangent_auroc_min,
        "c1_rollout_improvement_pass": rollout_gates["c1_rollout_improvement_all_horizons"],
        "c1_assimilation_pass": assimilation > 0.0,
    }
    summary = {
        "scope": "V6R0.2 frozen V6 force-space reintegration",
        "regimes": list(REGIMES),
        "residual_coordinate": "generalized_force",
        "operator_library_frozen": True,
        "selector_frozen": True,
        "acceptance_frozen": True,
        "c2_executed": False,
        "robotwin_assets_downloaded": False,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "horizons": list(cfg.horizons)},
        "rollout_gates": rollout_gates,
        "gates": gates,
        "overall_go": all(gates.values()),
        "rows": len(seed_rows),
    }
    _write_json(root / "summary.json", summary)
    return summary

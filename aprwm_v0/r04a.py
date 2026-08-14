from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend, _aggregate_seed_rows, _sample_local_points
from .r02 import (
    _backend_config,
    _candidate_design,
    _fit_candidate_posteriors,
    _force_observations,
    _model_rollout,
    _normalize_fallback_features,
    _predictive_moments,
)
from .r03 import V6R03Config
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import (
    OPERATOR_COMPLEXITY,
    OPERATOR_NAMES,
    operator_library,
    orthogonal_operator_gain,
)
from .v5 import _action_score, _fit_base_posterior, _update_base_belief, _update_candidate_belief
from .v6 import V6Config, _sequential_decision


STRATEGIES = (
    "force_only",
    "rollout",
    "rollout_stability",
    "rollout_stability_parameter",
)


@dataclass
class V6R04AConfig(V6R03Config):
    # 2001--2041 were used for predefined-strategy selection.  Defaults point
    # at the untouched confirmatory cohort for the now-fixed policy.
    seeds: tuple[int, ...] = (4001, 4011, 4021, 4031, 4041)
    short_horizons: tuple[int, ...] = (2, 4, 8)
    short_queries: int = 8
    rollout_margin: float = 1.0e-4
    mahalanobis_max: float = 9.21
    accepted_stability_min: float = 0.99


def _truth_rollouts(
    backend: SapienHingeBackend,
    state: torch.Tensor,
    memory: torch.Tensor,
    horizons: tuple[int, ...],
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    truth = []
    validity = []
    for horizon in horizons:
        horizon_truth = []
        horizon_validity = []
        for row, hidden in zip(state, memory):
            q, velocity, valid = backend.transition_with_validity(
                float(row[0]), float(row[1]), float(row[2]), float(hidden),
                horizon=horizon, limit_margin=margin,
            )
            horizon_truth.append((q, velocity))
            horizon_validity.append(valid)
        truth.append(torch.tensor(horizon_truth, dtype=torch.float32))
        validity.append(torch.tensor(horizon_validity, dtype=torch.bool))
    return torch.stack(truth, dim=1), torch.stack(validity, dim=1)


def _collect_native_seed(
    config: V6R04AConfig, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 210_000)
    backend_config = _backend_config(config)
    names = (
        "passive_physical", "passive_tangent", "passive_y",
        "discovery_physical", "discovery_tangent", "discovery_y",
        "action_physical", "action_tangent", "action_y",
        "validation_physical", "validation_tangent", "validation_y",
        "short_state", "short_truth", "short_validity",
        "query_state", "query_memory", "rollout_truth", "rollout_validity",
    )
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    densities = []
    dampings = []
    for _ in range(config.episodes_per_regime):
        density = config.density_min + (
            config.density_max - config.density_min
        ) * float(torch.rand((), generator=generator))
        damping = config.damping_min + (
            config.damping_max - config.damping_min
        ) * float(torch.rand((), generator=generator))
        backend = SapienHingeBackend(
            density=density, damping=damping, regime=0,
            config=backend_config, preserve_engine_dissipation=True,
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
        short_state, short_memory = _sample_local_points(
            config.short_queries, "intervention", generator
        )
        short_truth, short_validity = _truth_rollouts(
            backend, short_state, short_memory, config.short_horizons,
            config.joint_limit_margin,
        )
        query_state, query_memory = _sample_local_points(
            config.query_count, "intervention", generator
        )
        rollout_truth, rollout_validity = _truth_rollouts(
            backend, query_state, query_memory, config.horizons,
            config.joint_limit_margin,
        )
        fields["short_state"].append(short_state)
        fields["short_truth"].append(short_truth)
        fields["short_validity"].append(short_validity)
        fields["query_state"].append(query_state)
        fields["query_memory"].append(query_memory)
        fields["rollout_truth"].append(rollout_truth)
        fields["rollout_validity"].append(rollout_validity)
        densities.append(density)
        dampings.append(damping)
    output = {name: torch.stack(values).to(device) for name, values in fields.items()}
    output["density"] = torch.tensor(densities, dtype=torch.float32, device=device)
    output["damping"] = torch.tensor(dampings, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 211_000)
    for name in ("passive_y", "discovery_y", "action_y", "validation_y"):
        output[name] += config.force_observation_noise * torch.randn(
            output[name].shape, generator=noise_generator, device=device
        )
    return output


def _episode_rmse(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    scale = prediction.new_tensor((1.0, 2.0))
    query_mse = ((prediction - truth) / scale).square().mean(dim=-1)
    return torch.sqrt(
        (query_mse * validity).sum(dim=1)
        / validity.sum(dim=1).clamp_min(1)
    )


def _candidate_residual_force(
    q: float,
    velocity: float,
    torque: float,
    density: float,
    damping: float,
    coefficient: torch.Tensor,
    operator_index: int,
) -> float:
    physical = coefficient.new_tensor((q, velocity))
    operator = operator_library(physical)[operator_index]
    tangent = coefficient.new_tensor(
        ((torque - damping * velocity) / density, velocity)
    )
    return float((coefficient[:2] * tangent).sum() + coefficient[2] * operator)


def _candidate_macro_map(
    backend: SapienHingeBackend,
    q: float,
    velocity: float,
    torque: float,
    coefficient: torch.Tensor,
    operator_index: int,
    config: V6R04AConfig,
) -> torch.Tensor:
    pinocchio = backend.pinocchio
    for _ in range(config.macro_steps):
        h = float(
            pinocchio.compute_inverse_dynamics((q,), (velocity,), (0.0,))[0]
        )
        residual = _candidate_residual_force(
            q,
            velocity,
            torque,
            backend.density,
            backend.damping,
            coefficient,
            operator_index,
        )
        generalized_force = h + torque - backend.damping * velocity + residual
        acceleration = float(
            pinocchio.compute_forward_dynamics(
                (q,), (velocity,), (generalized_force,)
            )[0]
        )
        q = q + config.physics_dt * velocity
        velocity = velocity + config.physics_dt * acceleration
    return coefficient.new_tensor((q, velocity))


def _dynamical_features(
    backend: SapienHingeBackend,
    diagnostic_state: torch.Tensor,
    train_physical: torch.Tensor,
    coefficient: torch.Tensor,
    operator_index: int,
    config: V6R04AConfig,
) -> dict[str, float]:
    radii = []
    effective_damping = []
    positive_power = []
    equilibrium_force = []
    velocity_epsilon = 1.0e-3
    state_epsilons = (1.0e-4, 1.0e-3)
    for point in diagnostic_state:
        q, velocity, torque = (float(value) for value in point)
        columns = []
        for dimension, epsilon in enumerate(state_epsilons):
            plus = (q, velocity)
            minus = (q, velocity)
            if dimension == 0:
                plus = (q + epsilon, velocity)
                minus = (q - epsilon, velocity)
            else:
                plus = (q, velocity + epsilon)
                minus = (q, velocity - epsilon)
            forward = _candidate_macro_map(
                backend, plus[0], plus[1], torque,
                coefficient, operator_index, config,
            )
            backward = _candidate_macro_map(
                backend, minus[0], minus[1], torque,
                coefficient, operator_index, config,
            )
            columns.append((forward - backward) / (2.0 * epsilon))
        jacobian = torch.stack(columns, dim=-1)
        radii.append(float(torch.linalg.eigvals(jacobian).abs().max()))
        force = _candidate_residual_force(
            q, velocity, torque, backend.density, backend.damping,
            coefficient, operator_index,
        )
        force_plus = _candidate_residual_force(
            q, velocity + velocity_epsilon, torque,
            backend.density, backend.damping, coefficient, operator_index,
        )
        force_minus = _candidate_residual_force(
            q, velocity - velocity_epsilon, torque,
            backend.density, backend.damping, coefficient, operator_index,
        )
        force_velocity_derivative = (
            force_plus - force_minus
        ) / (2.0 * velocity_epsilon)
        effective_damping.append(backend.damping - force_velocity_derivative)
        positive_power.append(max(force * velocity, 0.0))
        equilibrium_force.append(
            _candidate_residual_force(
                q, 0.0, 0.0, backend.density, backend.damping,
                coefficient, operator_index,
            )
        )
    query_physical = diagnostic_state[:, :2]
    query_normalized = torch.stack(
        ((query_physical[:, 0] - 0.55) / 0.75, query_physical[:, 1] / 2.8),
        dim=-1,
    )
    train_normalized = torch.stack(
        ((train_physical[:, 0] - 0.55) / 0.75, train_physical[:, 1] / 2.8),
        dim=-1,
    )
    nearest_support = torch.cdist(
        query_normalized.unsqueeze(0), train_normalized.unsqueeze(0)
    ).squeeze(0).min(dim=-1).values
    return {
        "jacobian_spectral_radius_max": max(radii),
        "jacobian_spectral_radius_mean": statistics.mean(radii),
        "effective_damping_min": min(effective_damping),
        "negative_damping_violation": max(0.0, -min(effective_damping)),
        "positive_revision_power_max": max(positive_power),
        "positive_revision_power_mean": statistics.mean(positive_power),
        "equilibrium_force_rms": float(
            torch.tensor(equilibrium_force).square().mean().sqrt()
        ),
        "support_distance_max": float(nearest_support.max()),
        "support_distance_mean": float(nearest_support.mean()),
    }


def _evaluate_seed(
    config: V6R04AConfig, seed: int, device: torch.device
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = _collect_native_seed(config, seed, device)
    episodes = data["density"].shape[0]
    initial_theta = _ridge_fit(
        data["passive_tangent"], data["passive_y"], config.ridge
    )
    discovery_residual = data["discovery_y"] - torch.einsum(
        "bni,bi->bn", data["discovery_tangent"], initial_theta
    )
    _, residual_orthogonal, _ = tangent_decomposition(
        data["discovery_tangent"], discovery_residual, config.ridge
    )
    triggered = residual_orthogonal.square().mean(dim=-1).sqrt() > (
        config.trigger_noise_multiplier * config.force_observation_noise
    )
    gain = orthogonal_operator_gain(
        data["discovery_tangent"], residual_orthogonal,
        operator_library(data["discovery_physical"]), ridge=config.ridge,
    )
    complexity = torch.tensor(OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device)
    proposal_score = gain - config.proposal_complexity_price * complexity
    proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    train_tangent = torch.cat(
        (data["passive_tangent"], data["discovery_tangent"]), dim=1
    )
    train_physical = torch.cat(
        (data["passive_physical"], data["discovery_physical"]), dim=1
    )
    train_y = torch.cat((data["passive_y"], data["discovery_y"]), dim=1)
    v6 = V6Config(
        episodes=episodes,
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
    correct_position = torch.full((episodes,), -1, dtype=torch.long, device=device)
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
            observation_variance, triggered,
        )
        base_mean, base_covariance = _update_base_belief(
            base_mean, base_covariance, selected_tangent, selected_y,
            observation_variance, triggered,
        )
        used[torch.arange(episodes, device=device), action_index] = True

    order = log_probabilities.topk(2, dim=-1).indices
    selected_position, competitor_position = order[:, 0], order[:, 1]
    selected_operator = proposed.gather(1, selected_position[:, None]).squeeze(-1)
    selected_mean = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)
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
    incumbent_sq = (data["validation_y"] - base_prediction).square()
    candidate_sq = (data["validation_y"] - selected_prediction).square()
    competitor_sq = (data["validation_y"] - competitor_prediction).square()
    force_accepted, _, _, _ = _sequential_decision(
        incumbent_sq, competitor_sq, candidate_sq,
        complexity[selected_operator], config.force_observation_noise**2,
        v6, method="bf",
    )
    force_accepted &= triggered

    train_base = torch.einsum("bni,bi->bn", train_tangent, base_mean)
    fitted_residual = train_y - train_base
    normalized_train = _normalize_fallback_features(train_physical)

    def physics_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device)

    def base_force(tangent: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqi,bi->bq", tangent, base_mean)

    def fallback_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return base_force(tangent) + _residual_predict(
            normalized_train, fitted_residual,
            _normalize_fallback_features(physical), config.residual_ridge,
        ).clamp(-0.4, 0.4)

    def candidate_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        design = _candidate_design(tangent, physical, proposed)
        all_prediction = torch.einsum("bank,bnk->ban", design, means)
        return all_prediction.gather(
            -1, selected_position[:, None, None].expand(-1, physical.shape[1], 1)
        ).squeeze(-1)

    short_config = replace(config, horizons=config.short_horizons)
    short_data = dict(data)
    short_data["query_state"] = data["short_state"]
    physics_short, _ = _model_rollout(short_config, short_data, physics_force)
    fallback_short, _ = _model_rollout(short_config, short_data, fallback_force)
    candidate_short, candidate_short_stable = _model_rollout(
        short_config, short_data, candidate_force
    )
    physics_short_rmse = _episode_rmse(
        physics_short, data["short_truth"], data["short_validity"]
    )
    fallback_short_rmse = _episode_rmse(
        fallback_short, data["short_truth"], data["short_validity"]
    )
    candidate_short_rmse = _episode_rmse(
        candidate_short, data["short_truth"], data["short_validity"]
    )
    rollout_better = (
        (candidate_short_rmse + config.rollout_margin < physics_short_rmse)
        & (candidate_short_rmse + config.rollout_margin < fallback_short_rmse)
    ).all(dim=-1)
    q_short = candidate_short[..., 0]
    velocity_short = candidate_short[..., 1]
    stability_veto_pass = (
        candidate_short_stable.all(dim=(-1, -2))
        & torch.isfinite(candidate_short).all(dim=(-1, -2, -3))
        & (q_short > -0.08).all(dim=(-1, -2))
        & (q_short < 1.48).all(dim=(-1, -2))
        & (velocity_short.abs() < 5.0).all(dim=(-1, -2))
    )
    delta_theta = selected_mean[:, :2] - base_mean
    precision = torch.linalg.inv(base_covariance)
    mahalanobis = torch.einsum(
        "bi,bij,bj->b", delta_theta, precision, delta_theta
    )
    acceptance = {
        "force_only": force_accepted,
        "rollout": force_accepted & rollout_better,
        "rollout_stability": force_accepted & rollout_better & stability_veto_pass,
        "rollout_stability_parameter": (
            force_accepted & rollout_better & stability_veto_pass
            & (mahalanobis <= config.mahalanobis_max)
        ),
    }

    physics_final, physics_stable = _model_rollout(config, data, physics_force)
    fallback_final, fallback_stable = _model_rollout(config, data, fallback_force)
    candidate_final, candidate_stable = _model_rollout(config, data, candidate_force)
    final_predictions = {"physics": physics_final, "no_revision": fallback_final}
    final_stability = {"physics": physics_stable, "no_revision": fallback_stable}
    for strategy, accepted in acceptance.items():
        final_predictions[strategy] = torch.where(
            accepted[:, None, None, None], candidate_final, fallback_final
        )
        final_stability[strategy] = torch.where(
            accepted[:, None, None], candidate_stable, fallback_stable
        )

    rows = []
    seed_rows = []
    truth = data["rollout_truth"]
    validity = data["rollout_validity"]
    scale = truth.new_tensor((1.0, 2.0))
    fallback_episode_h16 = None
    episode_h16: dict[str, torch.Tensor] = {}
    for strategy, prediction in final_predictions.items():
        for horizon_index, horizon in enumerate(config.horizons):
            error = (prediction[:, :, horizon_index] - truth[:, :, horizon_index]) / scale
            valid = validity[:, :, horizon_index]
            rmse = _episode_rmse(
                prediction[:, :, horizon_index : horizon_index + 1],
                truth[:, :, horizon_index : horizon_index + 1],
                valid.unsqueeze(-1),
            ).squeeze(-1)
            if horizon == config.horizons[-1]:
                episode_h16[strategy] = rmse
                if strategy == "no_revision":
                    fallback_episode_h16 = rmse
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "horizon": horizon,
                    "state_rmse": float(
                        torch.sqrt((error.square().mean(dim=-1) * valid).sum() / valid.sum().clamp_min(1))
                    ),
                    "stable_fraction": float(
                        final_stability[strategy][:, :, horizon_index].float().mean()
                    ),
                    "valid_window_fraction": float(valid.float().mean()),
                }
            )
    assert fallback_episode_h16 is not None
    force_gain = incumbent_sq.mean(dim=-1) - candidate_sq.mean(dim=-1)
    for strategy in STRATEGIES:
        accepted = acceptance[strategy]
        accepted_count = int(accepted.sum())
        seed_rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "force_accept_rate": float(force_accepted.float().mean()),
                "revision_rate": float(accepted.float().mean()),
                "short_rollout_gate_rate_given_force": float(
                    rollout_better[force_accepted].float().mean()
                ) if bool(force_accepted.any()) else 0.0,
                "stability_veto_pass_given_force": float(
                    stability_veto_pass[force_accepted].float().mean()
                ) if bool(force_accepted.any()) else 0.0,
                "parameter_gate_pass_given_force": float(
                    (mahalanobis[force_accepted] <= config.mahalanobis_max).float().mean()
                ) if bool(force_accepted.any()) else 0.0,
                "mahalanobis_displacement_force_accepted": float(
                    mahalanobis[force_accepted].mean()
                ) if bool(force_accepted.any()) else 0.0,
                "force_validation_gain_accepted": float(
                    force_gain[accepted].mean()
                ) if accepted_count else 0.0,
                "h16_gain_accepted": float(
                    (fallback_episode_h16 - episode_h16[strategy])[accepted].mean()
                ) if accepted_count else 0.0,
                "h16_stability_accepted": float(
                    final_stability[strategy][accepted, :, -1].float().mean()
                ) if accepted_count else 1.0,
            }
        )
    diagnostic_rows = []
    locally_accepted = acceptance["rollout_stability"]
    short_reference = torch.minimum(physics_short_rmse, fallback_short_rmse)
    short_gain = short_reference - candidate_short_rmse
    candidate_h16_stability = candidate_stable[:, :, -1].float().mean(dim=-1)
    h16_candidate_gain = fallback_episode_h16 - episode_h16["force_only"]
    diagnostic_backends = [
        SapienHingeBackend(
            density=float(density), damping=float(damping), regime=0,
            config=_backend_config(config),
        )
        for density, damping in zip(data["density"], data["damping"])
    ]
    # Export every one-step force-accepted revision.  R0.5 can then analyze
    # both nested populations: locally force-useful revisions and the smaller
    # subset that also passed the R0.4 short-rollout/stability checks.
    for episode in torch.nonzero(force_accepted, as_tuple=False).flatten().tolist():
        torque = data["short_state"][episode, :, 2]
        trajectory_points = [data["short_state"][episode]]
        for horizon_index in range(len(config.short_horizons)):
            trajectory_points.append(
                torch.cat(
                    (
                        candidate_short[episode, :, horizon_index],
                        torque.unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
        diagnostic_state = torch.cat(trajectory_points, dim=0)
        dynamical = _dynamical_features(
            diagnostic_backends[episode],
            diagnostic_state,
            train_physical[episode],
            selected_mean[episode],
            int(selected_operator[episode]),
            config,
        )
        long_gain = float(h16_candidate_gain[episode])
        long_stability = float(candidate_h16_stability[episode])
        diagnostic_rows.append(
            {
                "seed": seed,
                "episode": episode,
                "operator": OPERATOR_NAMES[int(selected_operator[episode])],
                "short_rollout_accepted": int(locally_accepted[episode]),
                "short_rollout_better": int(rollout_better[episode]),
                "short_stability_pass": int(stability_veto_pass[episode]),
                "h16_unsafe": int(long_gain <= 0.0 or long_stability < 1.0),
                "h16_gain": long_gain,
                "h16_stability_fraction": long_stability,
                "force_validation_gain": float(force_gain[episode]),
                "short_rollout_gain_min": float(short_gain[episode].min()),
                "short_rollout_gain_mean": float(short_gain[episode].mean()),
                "mahalanobis_displacement": float(mahalanobis[episode]),
                "parameter_l2_displacement": float(delta_theta[episode].norm()),
                "structural_coefficient": float(selected_mean[episode, 2]),
                "structural_coefficient_abs": float(selected_mean[episode, 2].abs()),
                "operator_abs_v_v": int(
                    OPERATOR_NAMES[int(selected_operator[episode])] == "abs_v_v"
                ),
                **dynamical,
            }
        )
    return seed_rows, rows, diagnostic_rows


def run_v6r04a(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R04AConfig | None = None,
) -> dict[str, Any]:
    cfg = config or V6R04AConfig()
    device = resolve_device(device_name)
    seed_rows = []
    horizon_rows = []
    diagnostic_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        metrics, rollouts, diagnostics = _evaluate_seed(cfg, seed, device)
        seed_rows.extend(metrics)
        horizon_rows.extend(rollouts)
        diagnostic_rows.extend(diagnostics)
    root = Path(output_dir)
    _write_csv(root / "strategy_seed_metrics.csv", seed_rows)
    _write_csv(root / "strategy_horizons.csv", horizon_rows)
    _write_csv(root / "episode_diagnostics.csv", diagnostic_rows)

    # The Mahalanobis threshold is an ablation, not a tuned production gate.
    # In transported force coordinates its chi-square interpretation is not
    # calibrated; selecting it after seeing held-out outcomes would leak the
    # evaluation set.  The preregistered rollout + hard-stability policy is the
    # recommended R0.4-A mechanism, while parameter displacement remains a
    # reported diagnostic until calibrated on separate development data.
    final = "rollout_stability"
    def mean_seed(metric: str, strategy: str) -> float:
        return statistics.mean(
            float(row[metric]) for row in seed_rows if row["strategy"] == strategy
        )
    def mean_h16(strategy: str) -> float:
        return statistics.mean(
            float(row["state_rmse"]) for row in horizon_rows
            if row["strategy"] == strategy and int(row["horizon"]) == cfg.horizons[-1]
        )
    final_revision_rate = mean_seed("revision_rate", final)
    final_gain = mean_seed("h16_gain_accepted", final)
    final_stability = mean_seed("h16_stability_accepted", final)
    gates = {
        "h16_noninferior_to_no_revision": mean_h16(final) <= mean_h16("no_revision"),
        "accepted_stability": final_stability >= cfg.accepted_stability_min,
        "nontrivial_safe_revision": final_revision_rate > 0.0 and final_gain >= 0.0,
    }
    summary = {
        "scope": "V6R0.4-A rollout-aware native revision acceptance",
        "held_out_seeds": list(cfg.seeds),
        "strategies": list(STRATEGIES),
        "short_horizons": list(cfg.short_horizons),
        "final_strategy": final,
        "parameter_gate_status": (
            "diagnostic_only: the uncalibrated hard Mahalanobis veto is an ablation"
        ),
        "h16_rmse": {
            strategy: mean_h16(strategy)
            for strategy in ("physics", "no_revision", *STRATEGIES)
        },
        "final_revision_rate": final_revision_rate,
        "final_h16_gain_accepted": final_gain,
        "final_h16_stability_accepted": final_stability,
        "gates": gates,
        "overall_go": all(gates.values()),
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "horizons": list(cfg.horizons), "short_horizons": list(cfg.short_horizons)},
        "robotwin_assets_downloaded": False,
    }
    _write_json(root / "summary.json", summary)
    return summary

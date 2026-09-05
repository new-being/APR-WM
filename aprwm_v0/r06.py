from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend, _sample_local_points
from .r02 import (
    DRAG_OPERATOR,
    _backend_config,
    _candidate_design,
    _fit_candidate_posteriors,
    _force_observations,
    _model_rollout,
    _normalize_fallback_features,
    _predictive_moments,
)
from .r04a import V6R04AConfig, _episode_rmse, _truth_rollouts
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import (
    OPERATOR_COMPLEXITY,
    OPERATOR_NAMES,
    operator_library,
    orthogonal_operator_gain,
)
from .v5 import (
    _action_score,
    _fit_base_posterior,
    _update_base_belief,
    _update_candidate_belief,
)
from .v6 import V6Config, _sequential_decision


REGIMES = ("c0_parameter", "c1_drag", "r0n_native")
STRATEGIES = (
    "no_revision",
    "original_short",
    "passivity_only",
    "passivity_utility",
    "oracle_safe_useful",
)
# These bases have phi(q,v)*v >= 0 throughout the declared q > 0 support, so
# alpha <= 0 makes their zero-input structural contribution dissipative.
DISSIPATIVE_OPERATORS = (3, 4, 5, 6)


@dataclass
class V6R06Config(V6R04AConfig):
    # 6001--6041 exposed two implementation-semantic mismatches (contaminated
    # posterior anchor and a leftover force-BF gate).  Defaults are the fresh
    # confirmatory cohort for the corrected, preregistered two-layer method.
    seeds: tuple[int, ...] = (7001, 7011, 7021, 7031, 7041)
    parameter_prior_weight: float = 0.05
    utility_weights: tuple[float, ...] = (1.0 / 3.0,) * 3
    useful_revision_recall_min: float = 0.80
    c1_exact_recovery_min: float = 0.95


def is_dissipative_operator(operator_index: int) -> bool:
    return operator_index in DISSIPATIVE_OPERATORS


def _collect_seed(
    config: V6R06Config,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 260_000)
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
    regimes = []
    densities = []
    dampings = []
    for regime in range(len(REGIMES)):
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
                regime=1 if regime == 1 else 0,
                config=backend_config,
                preserve_engine_dissipation=regime == 2,
            )
            for prefix, count, sampling_regime in (
                ("passive", config.passive_context, "passive"),
                ("discovery", config.discovery_count, "discovery"),
                ("action", config.action_candidates, "diagnostic"),
                ("validation", config.validation_max, "validation"),
            ):
                state, memory = _sample_local_points(
                    count, sampling_regime, generator
                )
                physical, tangent, target = _force_observations(
                    backend, state, memory
                )
                fields[f"{prefix}_physical"].append(physical)
                fields[f"{prefix}_tangent"].append(tangent)
                fields[f"{prefix}_y"].append(target)
            short_state, short_memory = _sample_local_points(
                config.short_queries, "intervention", generator
            )
            short_truth, short_validity = _truth_rollouts(
                backend,
                short_state,
                short_memory,
                config.short_horizons,
                config.joint_limit_margin,
            )
            query_state, query_memory = _sample_local_points(
                config.query_count, "intervention", generator
            )
            rollout_truth, rollout_validity = _truth_rollouts(
                backend,
                query_state,
                query_memory,
                config.horizons,
                config.joint_limit_margin,
            )
            fields["short_state"].append(short_state)
            fields["short_truth"].append(short_truth)
            fields["short_validity"].append(short_validity)
            fields["query_state"].append(query_state)
            fields["query_memory"].append(query_memory)
            fields["rollout_truth"].append(rollout_truth)
            fields["rollout_validity"].append(rollout_validity)
            regimes.append(regime)
            densities.append(density)
            dampings.append(damping)
    output = {name: torch.stack(values).to(device) for name, values in fields.items()}
    output["regime"] = torch.tensor(regimes, dtype=torch.long, device=device)
    output["density"] = torch.tensor(densities, dtype=torch.float32, device=device)
    output["damping"] = torch.tensor(dampings, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 261_000)
    for name in ("passive_y", "discovery_y", "action_y", "validation_y"):
        output[name] += config.force_observation_noise * torch.randn(
            output[name].shape,
            generator=noise_generator,
            device=device,
        )
    return output


def constrained_dissipative_map_fit(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    target: torch.Tensor,
    operator_index: torch.Tensor,
    base_mean: torch.Tensor,
    base_covariance: torch.Tensor,
    config: V6R06Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weak-posterior MAP fit with alpha <= 0 for annotated operators."""
    episodes = target.shape[0]
    operators = operator_library(physical).gather(
        -1,
        operator_index[:, None, None].expand(-1, physical.shape[1], 1),
    ).squeeze(-1)
    design = torch.cat((tangent, operators.unsqueeze(-1)), dim=-1)
    variance = max(
        config.force_observation_noise, config.posterior_noise_floor
    ) ** 2
    precision = torch.linalg.inv(base_covariance)
    coefficients = []
    projected = []
    for episode in range(episodes):
        x = design[episode]
        y = target[episode]
        gram = x.T @ x / variance
        natural = x.T @ y / variance
        gram[:2, :2] += config.parameter_prior_weight * precision[episode]
        natural[:2] += (
            config.parameter_prior_weight
            * precision[episode]
            @ base_mean[episode]
        )
        gram += config.ridge * torch.eye(3, dtype=x.dtype, device=x.device)
        coefficient = torch.linalg.solve(gram, natural)
        was_projected = bool(coefficient[2] > 0.0)
        if was_projected:
            x_theta = x[:, :2]
            theta_gram = x_theta.T @ x_theta / variance
            theta_natural = x_theta.T @ y / variance
            theta_gram += config.parameter_prior_weight * precision[episode]
            theta_natural += (
                config.parameter_prior_weight
                * precision[episode]
                @ base_mean[episode]
            )
            theta_gram += config.ridge * torch.eye(
                2, dtype=x.dtype, device=x.device
            )
            theta = torch.linalg.solve(theta_gram, theta_natural)
            coefficient = torch.cat((theta, theta.new_zeros(1)))
        coefficients.append(coefficient)
        projected.append(was_projected)
    return (
        torch.stack(coefficients),
        torch.tensor(projected, dtype=torch.bool, device=target.device),
    )


def _selected_force(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    coefficient: torch.Tensor,
    operator_index: torch.Tensor,
) -> torch.Tensor:
    operator = operator_library(physical).gather(
        -1,
        operator_index[:, None, None].expand(-1, physical.shape[1], 1),
    ).squeeze(-1)
    return (
        torch.einsum("bni,bi->bn", tangent, coefficient[:, :2])
        + coefficient[:, 2, None] * operator
    )


def _overdamping_ratio(
    physical: torch.Tensor,
    alpha: torch.Tensor,
    operator_index: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1.0e-3
    plus = physical.clone()
    minus = physical.clone()
    plus[..., 1] += epsilon
    minus[..., 1] -= epsilon
    library_plus = operator_library(plus)
    library_minus = operator_library(minus)
    selected_plus = library_plus.gather(
        -1, operator_index[:, None, None].expand(-1, physical.shape[1], 1)
    ).squeeze(-1)
    selected_minus = library_minus.gather(
        -1, operator_index[:, None, None].expand(-1, physical.shape[1], 1)
    ).squeeze(-1)
    derivative = (selected_plus - selected_minus) / (2.0 * epsilon)
    structural_damping = -alpha[:, None] * derivative
    return structural_damping.clamp_min(0.0).mean(dim=-1) / damping.clamp_min(1.0e-8)


def _evaluate_seed(
    config: V6R06Config,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = _collect_seed(config, seed, device)
    episodes = data["regime"].numel()
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
        data["discovery_tangent"],
        residual_orthogonal,
        operator_library(data["discovery_physical"]),
        ridge=config.ridge,
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
    parameter_prior_mean, parameter_prior_covariance = _fit_base_posterior(
        data["passive_tangent"],
        data["passive_y"],
        config.force_observation_noise,
        v6,
    )
    log_probabilities = torch.log_softmax(
        proposal_score.gather(1, proposed) / config.proposal_temperature, dim=-1
    )
    observation_variance = max(
        config.force_observation_noise, config.posterior_noise_floor
    ) ** 2
    used = torch.zeros_like(data["action_y"], dtype=torch.bool)
    selected_action_tangent = []
    selected_action_physical = []
    selected_action_y = []
    correct_position = torch.full(
        (episodes,), -1, dtype=torch.long, device=device
    )
    for _ in range(config.selection_probes):
        pool_means, pool_variances = _predictive_moments(
            data["action_tangent"],
            data["action_physical"],
            proposed,
            means,
            covariances,
        )
        score = _action_score(
            "weighted",
            pool_means,
            pool_variances,
            log_probabilities.exp(),
            observation_variance,
            correct_position,
            False,
        ).masked_fill(used, float("-inf"))
        action_index = score.argmax(dim=-1)
        action_tangent = data["action_tangent"].gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        action_physical = data["action_physical"].gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        action_y = data["action_y"].gather(
            1, action_index.unsqueeze(-1)
        ).squeeze(-1)
        selected_action_tangent.append(action_tangent)
        selected_action_physical.append(action_physical)
        selected_action_y.append(action_y)
        design = _candidate_design(
            action_tangent.unsqueeze(1), action_physical.unsqueeze(1), proposed
        ).squeeze(1)
        means, covariances, log_probabilities = _update_candidate_belief(
            means,
            covariances,
            log_probabilities,
            design,
            action_y,
            observation_variance,
            triggered,
        )
        base_mean, base_covariance = _update_base_belief(
            base_mean,
            base_covariance,
            action_tangent,
            action_y,
            observation_variance,
            triggered,
        )
        used[torch.arange(episodes, device=device), action_index] = True

    order = log_probabilities.topk(2, dim=-1).indices
    selected_position, competitor_position = order[:, 0], order[:, 1]
    selected_operator = proposed.gather(
        1, selected_position[:, None]
    ).squeeze(-1)
    original_coefficient = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)
    evidence_tangent = torch.cat(
        (train_tangent, torch.stack(selected_action_tangent, dim=1)), dim=1
    )
    evidence_physical = torch.cat(
        (train_physical, torch.stack(selected_action_physical, dim=1)), dim=1
    )
    evidence_y = torch.cat(
        (train_y, torch.stack(selected_action_y, dim=1)), dim=1
    )
    constrained_coefficient, coefficient_projected = constrained_dissipative_map_fit(
        evidence_tangent,
        evidence_physical,
        evidence_y,
        selected_operator,
        parameter_prior_mean,
        parameter_prior_covariance,
        config,
    )
    dissipative_tensor = torch.tensor(
        DISSIPATIVE_OPERATORS, dtype=selected_operator.dtype, device=device
    )
    feasible = (selected_operator[:, None] == dissipative_tensor).any(dim=-1)
    feasible &= constrained_coefficient[:, 2] <= 0.0

    validation_design = _candidate_design(
        data["validation_tangent"], data["validation_physical"], proposed
    )
    validation_all = torch.einsum("bank,bnk->ban", validation_design, means)
    original_validation = validation_all.gather(
        -1,
        selected_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    competitor_validation = validation_all.gather(
        -1,
        competitor_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    constrained_validation = _selected_force(
        data["validation_tangent"],
        data["validation_physical"],
        constrained_coefficient,
        selected_operator,
    )
    base_validation = torch.einsum(
        "bni,bi->bn", data["validation_tangent"], base_mean
    )
    incumbent_sq = (data["validation_y"] - base_validation).square()
    competitor_sq = (data["validation_y"] - competitor_validation).square()
    original_sq = (data["validation_y"] - original_validation).square()
    constrained_sq = (data["validation_y"] - constrained_validation).square()
    original_force_accepted, _, _, _ = _sequential_decision(
        incumbent_sq,
        competitor_sq,
        original_sq,
        complexity[selected_operator],
        config.force_observation_noise**2,
        v6,
        method="bf",
    )
    constrained_force_accepted, _, _, _ = _sequential_decision(
        incumbent_sq,
        competitor_sq,
        constrained_sq,
        complexity[selected_operator],
        config.force_observation_noise**2,
        v6,
        method="bf",
    )
    original_force_accepted &= triggered
    constrained_force_accepted &= triggered & feasible

    evidence_base = torch.einsum("bni,bi->bn", evidence_tangent, base_mean)
    evidence_residual = evidence_y - evidence_base
    normalized_evidence = _normalize_fallback_features(evidence_physical)

    def physics_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device)

    def base_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqi,bi->bq", tangent, base_mean)

    def fallback_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return base_force(tangent, physical) + _residual_predict(
            normalized_evidence,
            evidence_residual,
            _normalize_fallback_features(physical),
            config.residual_ridge,
        ).clamp(-0.4, 0.4)

    def original_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return _selected_force(
            tangent, physical, original_coefficient, selected_operator
        )

    def constrained_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return _selected_force(
            tangent, physical, constrained_coefficient, selected_operator
        )

    short_config = V6R06Config(**{
        **config.__dict__, "horizons": config.short_horizons
    })
    short_data = dict(data)
    short_data["query_state"] = data["short_state"]
    physics_short, _ = _model_rollout(short_config, short_data, physics_force)
    fallback_short, _ = _model_rollout(short_config, short_data, fallback_force)
    original_short, original_short_stable = _model_rollout(
        short_config, short_data, original_force
    )
    constrained_short, constrained_short_stable = _model_rollout(
        short_config, short_data, constrained_force
    )
    physics_short_rmse = _episode_rmse(
        physics_short, data["short_truth"], data["short_validity"]
    )
    fallback_short_rmse = _episode_rmse(
        fallback_short, data["short_truth"], data["short_validity"]
    )
    original_short_rmse = _episode_rmse(
        original_short, data["short_truth"], data["short_validity"]
    )
    constrained_short_rmse = _episode_rmse(
        constrained_short, data["short_truth"], data["short_validity"]
    )
    original_short_better = (
        (original_short_rmse + config.rollout_margin < physics_short_rmse)
        & (original_short_rmse + config.rollout_margin < fallback_short_rmse)
    ).all(dim=-1)
    original_short_safe = original_short_stable.all(dim=(-1, -2))
    original_accepted = (
        original_force_accepted & original_short_better & original_short_safe
    )
    weights = constrained_short_rmse.new_tensor(config.utility_weights)
    utility = ((fallback_short_rmse - constrained_short_rmse) * weights).sum(dim=-1)
    # R0.6 is deliberately two-layer: proposal/selection supplies the force
    # evidence, physical admissibility defines the candidate space, and the
    # independent rollout utility decides usefulness.  Re-applying the old V6
    # force-BF threshold here would create an unregistered third hard gate.
    passivity_only_accepted = triggered & feasible
    passivity_utility_accepted = passivity_only_accepted & (utility > 0.0)

    fallback_final, fallback_final_stable = _model_rollout(
        config, data, fallback_force
    )
    original_final, original_final_stable = _model_rollout(
        config, data, original_force
    )
    constrained_final, constrained_final_stable = _model_rollout(
        config, data, constrained_force
    )
    fallback_h16 = _episode_rmse(
        fallback_final[:, :, -1:],
        data["rollout_truth"][:, :, -1:],
        data["rollout_validity"][:, :, -1:],
    ).squeeze(-1)
    constrained_h16 = _episode_rmse(
        constrained_final[:, :, -1:],
        data["rollout_truth"][:, :, -1:],
        data["rollout_validity"][:, :, -1:],
    ).squeeze(-1)
    constrained_h16_gain = fallback_h16 - constrained_h16
    constrained_h16_safe = constrained_final_stable[:, :, -1].all(dim=-1)
    oracle_useful = (
        triggered & feasible & constrained_h16_safe & (constrained_h16_gain > 0.0)
    )
    acceptance = {
        "no_revision": torch.zeros_like(triggered),
        "original_short": original_accepted,
        "passivity_only": passivity_only_accepted,
        "passivity_utility": passivity_utility_accepted,
        "oracle_safe_useful": oracle_useful,
    }
    predictions = {"no_revision": fallback_final}
    stability = {"no_revision": fallback_final_stable}
    for strategy in STRATEGIES[1:]:
        accepted = acceptance[strategy]
        candidate = original_final if strategy == "original_short" else constrained_final
        candidate_stable = (
            original_final_stable
            if strategy == "original_short"
            else constrained_final_stable
        )
        predictions[strategy] = torch.where(
            accepted[:, None, None, None], candidate, fallback_final
        )
        stability[strategy] = torch.where(
            accepted[:, None, None], candidate_stable, fallback_final_stable
        )

    scale = data["rollout_truth"].new_tensor((1.0, 2.0))
    horizon_rows = []
    for strategy in STRATEGIES:
        for horizon_index, horizon in enumerate(config.horizons):
            valid = data["rollout_validity"][:, :, horizon_index]
            error = (
                predictions[strategy][:, :, horizon_index]
                - data["rollout_truth"][:, :, horizon_index]
            ) / scale
            for regime_index, regime_name in enumerate(REGIMES):
                regime_mask = data["regime"] == regime_index
                regime_valid = valid[regime_mask]
                regime_error = error[regime_mask]
                horizon_rows.append(
                    {
                        "seed": seed,
                        "regime": regime_name,
                        "strategy": strategy,
                        "horizon": horizon,
                        "state_rmse": float(
                            torch.sqrt(
                                (regime_error.square().mean(dim=-1) * regime_valid).sum()
                                / regime_valid.sum().clamp_min(1)
                            )
                        ),
                        "stable_fraction": float(
                            stability[strategy][regime_mask, :, horizon_index]
                            .float().mean()
                        ),
                        "valid_window_fraction": float(regime_valid.float().mean()),
                    }
                )

    overdamping = _overdamping_ratio(
        data["validation_physical"],
        constrained_coefficient[:, 2],
        selected_operator,
        data["damping"],
    )
    precision = torch.linalg.inv(parameter_prior_covariance)
    delta_theta = constrained_coefficient[:, :2] - parameter_prior_mean
    mahalanobis = torch.einsum("bi,bij,bj->b", delta_theta, precision, delta_theta)
    episode_rows = []
    for episode in range(episodes):
        episode_rows.append(
            {
                "seed": seed,
                "episode": episode,
                "regime": REGIMES[int(data["regime"][episode])],
                "triggered": int(triggered[episode]),
                "operator": OPERATOR_NAMES[int(selected_operator[episode])],
                "dissipative_family": int(feasible[episode]),
                "coefficient_projected": int(coefficient_projected[episode]),
                "original_alpha": float(original_coefficient[episode, 2]),
                "constrained_alpha": float(constrained_coefficient[episode, 2]),
                "original_force_accepted": int(original_force_accepted[episode]),
                "constrained_force_accepted": int(
                    constrained_force_accepted[episode]
                ),
                "utility_h248": float(utility[episode]),
                "original_short_accepted": int(original_accepted[episode]),
                "passivity_only_accepted": int(passivity_only_accepted[episode]),
                "passivity_utility_accepted": int(
                    passivity_utility_accepted[episode]
                ),
                "oracle_safe_useful": int(oracle_useful[episode]),
                "constrained_h16_gain": float(constrained_h16_gain[episode]),
                "constrained_h16_stable": int(constrained_h16_safe[episode]),
                "overdamping_ratio": float(overdamping[episode]),
                "parameter_mahalanobis": float(mahalanobis[episode]),
                "parameter_l2": float(delta_theta[episode].norm()),
            }
        )
    return episode_rows, horizon_rows


def run_v6r06(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R06Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6R06Config()
    device = resolve_device(device_name)
    episode_rows = []
    horizon_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        episodes, horizons = _evaluate_seed(cfg, seed, device)
        episode_rows.extend(episodes)
        horizon_rows.extend(horizons)
    root = Path(output_dir)
    _write_csv(root / "episode_decisions.csv", episode_rows)
    _write_csv(root / "strategy_horizons.csv", horizon_rows)

    def selected_episode_rows(regime: str) -> list[dict[str, Any]]:
        return [row for row in episode_rows if row["regime"] == regime]

    def mean_h16(regime: str, strategy: str) -> float:
        return statistics.mean(
            float(row["state_rmse"])
            for row in horizon_rows
            if row["regime"] == regime
            and row["strategy"] == strategy
            and int(row["horizon"]) == cfg.horizons[-1]
        )

    def acceptance_rate(regime: str, field: str) -> float:
        selected = selected_episode_rows(regime)
        return statistics.mean(int(row[field]) for row in selected)

    native = selected_episode_rows("r0n_native")
    accepted_native = [
        row for row in native if int(row["passivity_utility_accepted"])
    ]
    oracle_native = [row for row in native if int(row["oracle_safe_useful"])]
    useful_recall = (
        sum(int(row["passivity_utility_accepted"]) for row in oracle_native)
        / len(oracle_native)
        if oracle_native else 0.0
    )
    useful_intersection = sum(
        int(row["passivity_utility_accepted"]) for row in oracle_native
    )
    useful_precision = (
        useful_intersection / len(accepted_native) if accepted_native else 0.0
    )
    passivity_only_native = [
        row for row in native if int(row["passivity_only_accepted"])
    ]
    utility_rejected = [
        row for row in native
        if int(row["passivity_only_accepted"])
        and not int(row["passivity_utility_accepted"])
    ]
    accepted_stability = (
        statistics.mean(int(row["constrained_h16_stable"]) for row in accepted_native)
        if accepted_native else 1.0
    )
    c1 = selected_episode_rows("c1_drag")
    c1_exact = statistics.mean(
        int(
            int(row["passivity_utility_accepted"])
            and row["operator"] == OPERATOR_NAMES[DRAG_OPERATOR]
            and abs(float(row["constrained_alpha"]) + cfg.drag_strength) <= 0.02
        )
        for row in c1
    )
    c0_false = acceptance_rate(
        "c0_parameter", "passivity_utility_accepted"
    )
    native_gain = mean_h16("r0n_native", "no_revision") - mean_h16(
        "r0n_native", "passivity_utility"
    )
    gates = {
        "c0_false_revision": c0_false <= cfg.c0_false_revision_max,
        "c1_exact_recovery": c1_exact >= cfg.c1_exact_recovery_min,
        "native_h16_noninferiority": native_gain >= 0.0,
        "accepted_stability": accepted_stability >= cfg.accepted_stability_min,
        "nontrivial_acceptance": acceptance_rate(
            "r0n_native", "passivity_utility_accepted"
        ) > 0.0,
        "useful_revision_recall": useful_recall >= cfg.useful_revision_recall_min,
        "beats_original_short": mean_h16(
            "r0n_native", "passivity_utility"
        ) <= mean_h16("r0n_native", "original_short"),
    }
    summary = {
        "scope": "V6R0.6 passivity-feasible then utility-accepted revision",
        "seeds": list(cfg.seeds),
        "dissipative_operators": [
            OPERATOR_NAMES[index] for index in DISSIPATIVE_OPERATORS
        ],
        "acceptance_architecture": (
            "hard dissipative-family feasibility, then equal-weight H2/H4/H8 utility"
        ),
        "parameter_prior_weight": cfg.parameter_prior_weight,
        "thresholds_tuned_on_confirmatory_labels": False,
        "h16_rmse": {
            regime: {strategy: mean_h16(regime, strategy) for strategy in STRATEGIES}
            for regime in REGIMES
        },
        "native_acceptance_rate": {
            "original_short": acceptance_rate(
                "r0n_native", "original_short_accepted"
            ),
            "passivity_only": acceptance_rate(
                "r0n_native", "passivity_only_accepted"
            ),
            "passivity_utility": acceptance_rate(
                "r0n_native", "passivity_utility_accepted"
            ),
            "oracle_safe_useful": acceptance_rate(
                "r0n_native", "oracle_safe_useful"
            ),
        },
        "native_h16_gain": native_gain,
        "passivity_only_native_h16_gain": (
            mean_h16("r0n_native", "no_revision")
            - mean_h16("r0n_native", "passivity_only")
        ),
        "utility_increment_over_passivity_only": (
            mean_h16("r0n_native", "passivity_only")
            - mean_h16("r0n_native", "passivity_utility")
        ),
        "accepted_h16_stability": accepted_stability,
        "useful_revision_recall": useful_recall,
        "useful_revision_precision": useful_precision,
        "oracle_useful_count": len(oracle_native),
        "accepted_count": len(accepted_native),
        "accepted_h16_negative_utility_count": sum(
            float(row["constrained_h16_gain"]) <= 0.0 for row in accepted_native
        ),
        "passivity_only_count": len(passivity_only_native),
        "passivity_only_h16_stability": (
            statistics.mean(
                int(row["constrained_h16_stable"])
                for row in passivity_only_native
            ) if passivity_only_native else 1.0
        ),
        "utility_rejected_count": len(utility_rejected),
        "utility_rejected_oracle_useful_count": sum(
            int(row["oracle_safe_useful"]) for row in utility_rejected
        ),
        "utility_rejected_not_oracle_count": sum(
            not int(row["oracle_safe_useful"]) for row in utility_rejected
        ),
        "c0_false_revision": c0_false,
        "c1_exact_recovery": c1_exact,
        "c1_constrained_alpha_mean": statistics.mean(
            float(row["constrained_alpha"]) for row in c1
        ),
        "accepted_overdamping_ratio_mean": (
            statistics.mean(float(row["overdamping_ratio"]) for row in accepted_native)
            if accepted_native else 0.0
        ),
        "accepted_parameter_mahalanobis_mean": (
            statistics.mean(float(row["parameter_mahalanobis"]) for row in accepted_native)
            if accepted_native else 0.0
        ),
        "gates": gates,
        "overall_go": all(gates.values()),
        "robotwin_assets_downloaded": False,
    }
    _write_json(root / "summary.json", summary)
    return summary

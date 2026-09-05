from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import (
    COUPLED_OPERATOR,
    CUBIC_OPERATOR,
    OPERATOR_COMPLEXITY,
    _ensure_oracle_proposal,
    _sample_features,
    operator_library,
    orthogonal_operator_gain,
    true_force,
)
from .v5 import (
    V5Config,
    _action_score,
    _candidate_design,
    _fit_base_posterior,
    _fit_posteriors,
    _predictive_moments,
    _update_base_belief,
    _update_candidate_belief,
    _v4_config,
)


STRATEGIES = (
    "always_accept",
    "fixed8_raw",
    "fixed8_calibrated",
    "fixed16_calibrated",
    "fixed32_calibrated",
    "sequential_lcb",
    "sequential_bf",
    "oracle_acceptance",
)


@dataclass
class V6Config:
    seeds: tuple[int, ...] = (401, 411, 421, 431, 441)
    noise_levels: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15)
    episodes: int = 2_048
    passive_context: int = 8
    discovery_count: int = 16
    action_candidates: int = 32
    selection_probes: int = 3
    validation_max: int = 32
    proposal_topk: int = 3
    cubic_alpha: float = 800.0
    coupled_alpha: float = 95.0
    outside_alpha: float = 0.24
    proposal_complexity_price: float = 2.0e-4
    proposal_temperature: float = 0.10
    posterior_noise_floor: float = 0.01
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    acceptance_complexity_price: float = 5.0e-4
    acceptance_margin: float = 1.5e-3
    acceptance_rmse: float = 0.070
    validation_step: int = 4
    validation_min: int = 4
    lcb_z: float = 1.645
    bf_accept: float = 6.90
    bf_reject: float = -2.94
    bf_complexity_log_prior: float = 0.35
    residual_cost: float = 0.01
    action_cost: float = 0.002
    validation_cost: float = 0.0005
    model_complexity_cost: float = 0.0015


def _selection_config(config: V6Config) -> V5Config:
    return V5Config(
        episodes=config.episodes,
        passive_context=config.passive_context,
        discovery_count=config.discovery_count,
        heldout_count=8,
        action_candidates=config.action_candidates,
        max_probes=config.selection_probes,
        proposal_topk=config.proposal_topk,
        cubic_alpha=config.cubic_alpha,
        coupled_alpha=config.coupled_alpha,
        outside_alpha=config.outside_alpha,
        proposal_complexity_price=config.proposal_complexity_price,
        proposal_temperature=config.proposal_temperature,
        posterior_noise_floor=config.posterior_noise_floor,
        ridge=config.ridge,
        residual_ridge=config.residual_ridge,
        acceptance_complexity_price=config.acceptance_complexity_price,
        acceptance_margin=config.acceptance_margin,
        acceptance_rmse=config.acceptance_rmse,
        residual_cost=config.residual_cost,
        action_cost=config.action_cost,
        model_complexity_cost=config.model_complexity_cost,
    )


def noise_calibrated_excess_mse(
    squared_error: torch.Tensor, observation_variance: float
) -> torch.Tensor:
    return (squared_error.mean(dim=-1) - observation_variance).clamp_min(0.0)


def fixed_acceptance(
    incumbent_squared_error: torch.Tensor,
    competitor_squared_error: torch.Tensor,
    candidate_squared_error: torch.Tensor,
    complexity: torch.Tensor,
    observation_variance: float,
    config: V6Config,
    *,
    calibrated: bool,
) -> torch.Tensor:
    improvement_old = (
        incumbent_squared_error.mean(dim=-1)
        - candidate_squared_error.mean(dim=-1)
        - config.acceptance_complexity_price * complexity
    )
    improvement_competitor = (
        competitor_squared_error.mean(dim=-1)
        - candidate_squared_error.mean(dim=-1)
    )
    candidate_mse = candidate_squared_error.mean(dim=-1)
    if calibrated:
        candidate_mse = (candidate_mse - observation_variance).clamp_min(0.0)
    return (
        (improvement_old > config.acceptance_margin)
        & (improvement_competitor > config.acceptance_margin)
        & (candidate_mse.sqrt() < config.acceptance_rmse)
    )


def _sequential_decision(
    incumbent_squared_error: torch.Tensor,
    competitor_squared_error: torch.Tensor,
    candidate_squared_error: torch.Tensor,
    complexity: torch.Tensor,
    observation_variance: float,
    config: V6Config,
    *,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    episodes, maximum = candidate_squared_error.shape
    device = candidate_squared_error.device
    accepted = torch.zeros(episodes, dtype=torch.bool, device=device)
    rejected = torch.zeros_like(accepted)
    samples = torch.zeros(episodes, device=device)
    statistic = torch.zeros(episodes, device=device)
    evidence_variance = max(
        observation_variance, config.posterior_noise_floor**2
    )
    for count in range(config.validation_step, maximum + 1, config.validation_step):
        active = ~(accepted | rejected)
        if not bool(active.any()):
            break
        old_sq = incumbent_squared_error[:, :count]
        new_sq = candidate_squared_error[:, :count]
        improvement_old_samples = old_sq - new_sq
        improvement_competitor_samples = (
            competitor_squared_error[:, :count] - new_sq
        )
        mean_improvement_old = improvement_old_samples.mean(dim=-1)
        mean_improvement_competitor = improvement_competitor_samples.mean(dim=-1)
        candidate_mean = new_sq.mean(dim=-1)
        candidate_excess = (candidate_mean - observation_variance).clamp_min(0.0)
        complexity_cost = config.acceptance_complexity_price * complexity
        if method == "lcb":
            improvement_old_se = improvement_old_samples.std(
                dim=-1, unbiased=True
            ) / math.sqrt(count)
            improvement_competitor_se = improvement_competitor_samples.std(
                dim=-1, unbiased=True
            ) / math.sqrt(count)
            candidate_se = new_sq.std(dim=-1, unbiased=True) / math.sqrt(count)
            lower_old = (
                mean_improvement_old
                - config.lcb_z * improvement_old_se
                - complexity_cost
            )
            upper_old = (
                mean_improvement_old
                + config.lcb_z * improvement_old_se
                - complexity_cost
            )
            lower_competitor = (
                mean_improvement_competitor
                - config.lcb_z * improvement_competitor_se
            )
            upper_competitor = (
                mean_improvement_competitor
                + config.lcb_z * improvement_competitor_se
            )
            upper_excess = (
                candidate_mean
                - observation_variance
                + config.lcb_z * candidate_se
            ).clamp_min(0.0)
            lower_excess = (
                candidate_mean
                - observation_variance
                - config.lcb_z * candidate_se
            ).clamp_min(0.0)
            accept_now = (
                (lower_old > config.acceptance_margin)
                & (lower_competitor > config.acceptance_margin)
                & (upper_excess.sqrt() < config.acceptance_rmse)
            )
            reject_now = (
                (upper_old <= 0.0)
                | (upper_competitor <= 0.0)
                | (lower_excess.sqrt() >= config.acceptance_rmse)
            )
            current_statistic = torch.minimum(lower_old, lower_competitor)
        elif method == "bf":
            log_bf_old = improvement_old_samples.sum(dim=-1) / (
                2.0 * evidence_variance
            )
            log_bf_old -= config.bf_complexity_log_prior * complexity
            log_bf_competitor = improvement_competitor_samples.sum(dim=-1) / (
                2.0 * evidence_variance
            )
            log_bf = torch.minimum(log_bf_old, log_bf_competitor)
            adequacy = candidate_excess.sqrt() < config.acceptance_rmse
            accept_now = (log_bf > config.bf_accept) & adequacy
            reject_now = (log_bf < config.bf_reject) | (
                candidate_excess.sqrt() >= config.acceptance_rmse
            )
            current_statistic = log_bf
        else:
            raise ValueError(f"Unknown sequential method: {method}")
        eligible = active & (count >= config.validation_min)
        accept_now &= eligible
        reject_now &= eligible & ~accept_now
        newly_stopped = accept_now | reject_now
        samples = torch.where(newly_stopped, torch.full_like(samples, float(count)), samples)
        statistic = torch.where(newly_stopped, current_statistic, statistic)
        accepted |= accept_now
        rejected |= reject_now
    unresolved = ~(accepted | rejected)
    samples = torch.where(unresolved, torch.full_like(samples, float(maximum)), samples)
    if method == "bf":
        all_old = incumbent_squared_error - candidate_squared_error
        all_competitor = competitor_squared_error - candidate_squared_error
        final_old = all_old.sum(dim=-1) / (2.0 * evidence_variance)
        final_old -= config.bf_complexity_log_prior * complexity
        final_competitor = all_competitor.sum(dim=-1) / (2.0 * evidence_variance)
        final_statistic = torch.minimum(final_old, final_competitor)
    else:
        all_old = incumbent_squared_error - candidate_squared_error
        all_competitor = competitor_squared_error - candidate_squared_error
        final_old = (
            all_old.mean(dim=-1)
            - config.lcb_z
            * all_old.std(dim=-1, unbiased=True)
            / math.sqrt(maximum)
            - config.acceptance_complexity_price * complexity
        )
        final_competitor = (
            all_competitor.mean(dim=-1)
            - config.lcb_z
            * all_competitor.std(dim=-1, unbiased=True)
            / math.sqrt(maximum)
        )
        final_statistic = torch.minimum(final_old, final_competitor)
    statistic = torch.where(unresolved, final_statistic, statistic)
    return accepted, rejected, unresolved, samples


def _evaluate_noise(
    config: V6Config,
    seed: int,
    noise: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    selection = _selection_config(config)
    v4 = _v4_config(selection, noise)
    generator = torch.Generator(device=device).manual_seed(seed + 100_000)
    noise_generator = torch.Generator(device=device).manual_seed(
        seed + 101_000 + int(round(noise * 10_000))
    )
    episodes = config.episodes
    theta = torch.stack(
        (
            24.0 + 22.0 * torch.rand(episodes, generator=generator, device=device),
            0.7 + torch.rand(episodes, generator=generator, device=device),
        ),
        dim=-1,
    )
    model_class = torch.randint(0, 4, (episodes,), generator=generator, device=device)
    adequate = model_class == 0
    cubic = model_class == 1
    coupled = model_class == 2
    outside = model_class == 3
    in_library = cubic | coupled
    correct_operator = torch.full_like(model_class, -1)
    correct_operator[cubic] = CUBIC_OPERATOR
    correct_operator[coupled] = COUPLED_OPERATOR

    passive_x = _sample_features(
        "passive", (episodes, config.passive_context), generator, device
    )
    discovery_x = _sample_features(
        "discovery", (episodes, config.discovery_count), generator, device
    )
    action_pool = _sample_features(
        "diagnostic", (episodes, config.action_candidates), generator, device
    )
    validation_x = _sample_features(
        "intervention", (episodes, config.validation_max), generator, device
    )
    passive_clean = true_force(passive_x, theta, model_class, v4)
    discovery_clean = true_force(discovery_x, theta, model_class, v4)
    passive_y = passive_clean + noise * torch.randn(
        passive_clean.shape, generator=noise_generator, device=device
    )
    discovery_y = discovery_clean + noise * torch.randn(
        discovery_clean.shape, generator=noise_generator, device=device
    )

    initial_theta = _ridge_fit(passive_x, passive_y, config.ridge)
    discovery_residual = discovery_y - torch.einsum(
        "bni,bi->bn", discovery_x, initial_theta
    )
    _, residual_orthogonal, _ = tangent_decomposition(
        discovery_x, discovery_residual, config.ridge
    )
    gain = orthogonal_operator_gain(
        discovery_x,
        residual_orthogonal,
        operator_library(discovery_x),
        ridge=config.ridge,
    )
    complexity_table = torch.tensor(
        OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device
    )
    proposal_score = gain - config.proposal_complexity_price * complexity_table
    natural_proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    natural_recall = (
        natural_proposed == correct_operator.unsqueeze(-1)
    ).any(dim=-1)
    proposed = _ensure_oracle_proposal(natural_proposed, model_class)
    proposal_recall = (proposed == correct_operator.unsqueeze(-1)).any(dim=-1)
    correct_position = torch.where(
        proposal_recall,
        (proposed == correct_operator.unsqueeze(-1)).float().argmax(dim=-1),
        torch.full_like(model_class, -1),
    )

    train_x = torch.cat((passive_x, discovery_x), dim=1)
    train_y = torch.cat((passive_y, discovery_y), dim=1)
    means, covariances = _fit_posteriors(
        train_x, train_y, proposed, noise, selection
    )
    base_mean, base_covariance = _fit_base_posterior(
        train_x, train_y, noise, selection
    )
    log_probabilities = torch.log_softmax(
        proposal_score.gather(1, proposed) / config.proposal_temperature,
        dim=-1,
    )
    observation_variance = max(noise, config.posterior_noise_floor) ** 2
    used_actions = torch.zeros(
        (episodes, config.action_candidates), dtype=torch.bool, device=device
    )
    for _ in range(config.selection_probes):
        probabilities = log_probabilities.exp()
        pool_means, pool_variances = _predictive_moments(
            action_pool, proposed, means, covariances
        )
        score = _action_score(
            "weighted",
            pool_means,
            pool_variances,
            probabilities,
            observation_variance,
            correct_position,
            False,
        ).masked_fill(used_actions, float("-inf"))
        action_index = score.argmax(dim=-1)
        selected_action = action_pool.gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        selected_design = _candidate_design(
            selected_action.unsqueeze(1), proposed
        ).squeeze(1)
        observation_clean = true_force(
            selected_action.unsqueeze(1), theta, model_class, v4
        ).squeeze(1)
        observation = observation_clean + noise * torch.randn(
            observation_clean.shape, generator=noise_generator, device=device
        )
        active = torch.ones(episodes, dtype=torch.bool, device=device)
        means, covariances, log_probabilities = _update_candidate_belief(
            means,
            covariances,
            log_probabilities,
            selected_design,
            observation,
            observation_variance,
            active,
        )
        base_mean, base_covariance = _update_base_belief(
            base_mean,
            base_covariance,
            selected_action,
            observation,
            observation_variance,
            active,
        )
        used_actions[torch.arange(episodes, device=device), action_index] = True

    final_order = log_probabilities.topk(2, dim=-1).indices
    selected_position = final_order[:, 0]
    competitor_position = final_order[:, 1]
    selected_operator = proposed.gather(
        1, selected_position.unsqueeze(-1)
    ).squeeze(-1)
    selection_correct = selected_operator == correct_operator
    selected_mean = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)
    validation_design = _candidate_design(validation_x, proposed)
    validation_candidate_all = torch.einsum(
        "bank,bnk->ban", validation_design, means
    )
    validation_candidate = validation_candidate_all.gather(
        -1,
        selected_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    validation_competitor = validation_candidate_all.gather(
        -1,
        competitor_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    validation_base = torch.einsum("bni,bi->bn", validation_x, base_mean)
    fitted_residual = train_y - torch.einsum("bni,bi->bn", train_x, base_mean)
    validation_fallback = validation_base + _residual_predict(
        train_x, fitted_residual, validation_x, config.residual_ridge
    ).clamp(-0.6, 0.6)
    validation_clean = true_force(validation_x, theta, model_class, v4)
    validation_y = validation_clean + noise * torch.randn(
        validation_clean.shape, generator=noise_generator, device=device
    )
    # Structural evidence compares the old explicit physics family with the
    # proposed family. The generic residual remains a prediction buffer after
    # rejection/defer, not a competing structural explanation.
    incumbent_sq = (validation_y - validation_base).square()
    candidate_sq = (validation_y - validation_candidate).square()
    competitor_sq = (validation_y - validation_competitor).square()
    selected_complexity = complexity_table[selected_operator]
    correct_revision = in_library & selection_correct
    wrong_revision = ~correct_revision

    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        if strategy == "always_accept":
            accepted = torch.ones(episodes, dtype=torch.bool, device=device)
            rejected = torch.zeros_like(accepted)
            deferred = torch.zeros_like(accepted)
            samples = torch.zeros(episodes, device=device)
        elif strategy.startswith("fixed"):
            count = int(strategy.split("_")[0].replace("fixed", ""))
            calibrated = strategy.endswith("calibrated") and not strategy.endswith("raw")
            accepted = fixed_acceptance(
                incumbent_sq[:, :count],
                competitor_sq[:, :count],
                candidate_sq[:, :count],
                selected_complexity,
                noise**2,
                config,
                calibrated=calibrated,
            )
            rejected = ~accepted
            deferred = torch.zeros_like(accepted)
            samples = torch.full((episodes,), float(count), device=device)
        elif strategy == "sequential_lcb":
            accepted, rejected, deferred, samples = _sequential_decision(
                incumbent_sq,
                competitor_sq,
                candidate_sq,
                selected_complexity,
                noise**2,
                config,
                method="lcb",
            )
        elif strategy == "sequential_bf":
            accepted, rejected, deferred, samples = _sequential_decision(
                incumbent_sq,
                competitor_sq,
                candidate_sq,
                selected_complexity,
                noise**2,
                config,
                method="bf",
            )
        elif strategy == "oracle_acceptance":
            accepted = correct_revision
            rejected = ~accepted
            deferred = torch.zeros_like(accepted)
            samples = torch.full((episodes,), 4.0, device=device)
        else:
            raise AssertionError(strategy)

        exact = accepted & correct_revision
        false = accepted & wrong_revision
        final_prediction = torch.where(
            accepted.unsqueeze(-1), validation_candidate, validation_fallback
        )
        clean_mse = (final_prediction - validation_clean).square().mean()
        final_complexity = torch.where(
            accepted, selected_complexity, torch.zeros_like(selected_complexity)
        )
        fallback = ~accepted
        rows.append(
            {
                "strategy": strategy,
                "noise": noise,
                "natural_topk_proposal_recall": float(
                    natural_recall[in_library].float().mean()
                ),
                "controlled_candidate_coverage": float(
                    proposal_recall[in_library].float().mean()
                ),
                "selection_accuracy": float(
                    selection_correct[in_library].float().mean()
                ),
                "acceptance_power_given_correct_selection": float(
                    accepted[correct_revision].float().mean()
                ),
                "exact_operator_recovery": float(exact[in_library].float().mean()),
                "false_revision_rate": float(false[wrong_revision].float().mean()),
                "wrong_operator_acceptance_in_library": float(
                    accepted[in_library & ~selection_correct].float().mean()
                )
                if bool((in_library & ~selection_correct).any())
                else 0.0,
                "adequate_false_expansion": float(accepted[adequate].float().mean()),
                "outside_false_expansion": float(accepted[outside].float().mean()),
                "defer_rate": float(deferred.float().mean()),
                "validation_samples": float(samples.mean()),
                "validation_samples_correct": float(samples[correct_revision].mean()),
                "validation_samples_wrong": float(samples[wrong_revision].mean()),
                "residual_fallback_rate": float(fallback.float().mean()),
                "clean_validation_rmse": float(torch.sqrt(clean_mse)),
                "model_complexity_cost": float(final_complexity.mean())
                * config.model_complexity_cost,
                "residual_compute_cost": float(fallback.float().mean())
                * config.residual_cost,
                "selection_action_cost": config.selection_probes * config.action_cost,
                "validation_evidence_cost": float(samples.mean())
                * config.validation_cost,
                "total_objective": float(clean_mse)
                + float(final_complexity.mean()) * config.model_complexity_cost
                + float(fallback.float().mean()) * config.residual_cost
                + config.selection_probes * config.action_cost
                + float(samples.mean()) * config.validation_cost,
                "parameter_rmse_k": float(
                    torch.sqrt((selected_mean[:, 0] - theta[:, 0]).square().mean())
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for noise in sorted({float(row["noise"]) for row in rows}):
            members = [
                row
                for row in rows
                if row["strategy"] == strategy and float(row["noise"]) == noise
            ]
            for metric in sorted(set(members[0]) - {"strategy", "seed", "noise"}):
                values = [float(row[metric]) for row in members]
                low, high = _bootstrap_ci(values)
                output.append(
                    {
                        "strategy": strategy,
                        "noise": noise,
                        "metric": metric,
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
    return output


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), float(row["noise"]), row["strategy"]): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    noises = sorted({float(row["noise"]) for row in rows})
    comparisons = (
        ("calibrated8_minus_raw8", "fixed8_calibrated", "fixed8_raw"),
        ("fixed32_minus_calibrated8", "fixed32_calibrated", "fixed8_calibrated"),
        ("sequential_lcb_minus_fixed32", "sequential_lcb", "fixed32_calibrated"),
        ("sequential_bf_minus_fixed32", "sequential_bf", "fixed32_calibrated"),
        ("sequential_bf_minus_lcb", "sequential_bf", "sequential_lcb"),
        ("oracle_minus_sequential_bf", "oracle_acceptance", "sequential_bf"),
    )
    metrics = (
        "acceptance_power_given_correct_selection",
        "exact_operator_recovery",
        "false_revision_rate",
        "wrong_operator_acceptance_in_library",
        "adequate_false_expansion",
        "outside_false_expansion",
        "defer_rate",
        "validation_samples",
        "residual_fallback_rate",
        "clean_validation_rmse",
        "total_objective",
    )
    output: list[dict[str, Any]] = []
    for noise in noises:
        for name, left, right in comparisons:
            for metric in metrics:
                values = [
                    float(indexed[(seed, noise, left)][metric])
                    - float(indexed[(seed, noise, right)][metric])
                    for seed in seeds
                ]
                low, high = _bootstrap_ci(values)
                output.append(
                    {
                        "noise": noise,
                        "comparison": name,
                        "metric": metric,
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
    return output


def run_v6(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V6Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6Config()
    device = resolve_device(device_name)
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        for noise in cfg.noise_levels:
            noise_rows = _evaluate_noise(cfg, seed, noise, device)
            for row in noise_rows:
                row["seed"] = seed
            rows.extend(noise_rows)
    root = Path(output_dir)
    _write_csv(root / "strategies.csv", rows)
    _write_csv(root / "strategies_aggregate.csv", _aggregate(rows))
    _write_csv(root / "paired_differences.csv", _paired(rows))
    summary = {
        "config": {
            **cfg.__dict__,
            "seeds": list(cfg.seeds),
            "noise_levels": list(cfg.noise_levels),
        },
        "rows": len(rows),
        "operator_library_is_frozen": True,
        "selection_policy_is_frozen": "posterior_weighted_fixed3",
        "in_library_trigger_and_candidate_availability_are_controlled": True,
        "noise_calibration_subtracts_known_observation_variance": True,
        "bayes_factor_is_dual_plugin_predictive_likelihood_ratio": True,
    }
    _write_json(root / "summary.json", summary)
    return summary

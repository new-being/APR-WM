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
    OPERATOR_NAMES,
    _sample_features,
    _ensure_oracle_proposal,
    operator_library,
    orthogonal_operator_gain,
    true_force,
    V4Config,
)


STRATEGIES = (
    "raw_one",
    "raw_fixed2",
    "normalized_fixed2",
    "weighted_fixed2",
    "weighted_fixed3",
    "sequential_weighted",
    "sequential_lcb",
    "oracle_action",
    "oracle_selection",
)


@dataclass
class V5Config:
    seeds: tuple[int, ...] = (301, 311, 321, 331, 341)
    noise_levels: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15)
    episodes: int = 2_048
    passive_context: int = 8
    discovery_count: int = 16
    heldout_count: int = 8
    action_candidates: int = 32
    max_probes: int = 3
    queries: int = 16
    proposal_topk: int = 3
    cubic_alpha: float = 800.0
    coupled_alpha: float = 95.0
    outside_alpha: float = 0.24
    tangent_ratio_threshold: float = 0.02
    tangent_noise_floor: float = 0.0675
    proposal_complexity_price: float = 2.0e-4
    proposal_temperature: float = 0.10
    posterior_noise_floor: float = 0.01
    select_probability: float = 0.55
    select_margin: float = 0.10
    acceptance_complexity_price: float = 5.0e-4
    acceptance_margin: float = 1.5e-3
    acceptance_rmse: float = 0.095
    lcb_z: float = 0.67449
    lcb_improvement: float = 0.0
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    residual_cost: float = 0.01
    action_cost: float = 0.002
    model_complexity_cost: float = 0.0015


@dataclass(frozen=True)
class StrategySpec:
    criterion: str
    budget: int
    sequential: bool = False
    lcb_acceptance: bool = False
    oracle_action: bool = False
    oracle_selection: bool = False


SPECS = {
    "raw_one": StrategySpec("raw", 1),
    "raw_fixed2": StrategySpec("raw", 2),
    "normalized_fixed2": StrategySpec("normalized", 2),
    "weighted_fixed2": StrategySpec("weighted", 2),
    "weighted_fixed3": StrategySpec("weighted", 3),
    "sequential_weighted": StrategySpec("weighted", 3, sequential=True),
    "sequential_lcb": StrategySpec(
        "weighted", 3, sequential=True, lcb_acceptance=True
    ),
    "oracle_action": StrategySpec(
        "weighted", 3, sequential=True, oracle_action=True
    ),
    "oracle_selection": StrategySpec(
        "weighted", 2, oracle_selection=True
    ),
}


def _v4_config(config: V5Config, noise: float) -> V4Config:
    return V4Config(
        episodes=config.episodes,
        passive_context=config.passive_context,
        discovery_count=config.discovery_count,
        heldout_count=config.heldout_count,
        action_candidates=config.action_candidates,
        diagnostic_probes=2,
        queries=config.queries,
        proposal_topk=config.proposal_topk,
        observation_noise=noise,
        cubic_alpha=config.cubic_alpha,
        coupled_alpha=config.coupled_alpha,
        outside_alpha=config.outside_alpha,
        tangent_ratio_threshold=config.tangent_ratio_threshold,
        tangent_noise_threshold=1.35,
        proposal_complexity_price=config.proposal_complexity_price,
        acceptance_complexity_price=config.acceptance_complexity_price,
        acceptance_margin=config.acceptance_margin,
        acceptance_rmse=config.acceptance_rmse,
        ridge=config.ridge,
        residual_ridge=config.residual_ridge,
        residual_cost=config.residual_cost,
        action_cost=config.action_cost,
        model_complexity_cost=config.model_complexity_cost,
    )


def _candidate_design(
    features: torch.Tensor, proposed: torch.Tensor
) -> torch.Tensor:
    operators = operator_library(features)
    selected = operators.gather(
        -1, proposed[:, None, :].expand(-1, features.shape[1], -1)
    )
    base = features[:, :, None, :].expand(-1, -1, proposed.shape[1], -1)
    return torch.cat((base, selected.unsqueeze(-1)), dim=-1)


def _fit_posteriors(
    features: torch.Tensor,
    target: torch.Tensor,
    proposed: torch.Tensor,
    noise: float,
    config: V5Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = _candidate_design(features, proposed).permute(0, 2, 1, 3)
    means = []
    covariances = []
    variance = max(noise, config.posterior_noise_floor) ** 2
    for index in range(proposed.shape[1]):
        candidate = design[:, index]
        scale = candidate.square().mean(dim=1).sqrt().clamp_min(1.0e-8)
        normalized = candidate / scale.unsqueeze(1)
        gram = torch.einsum("bni,bnj->bij", normalized, normalized)
        eye = torch.eye(3, dtype=features.dtype, device=features.device)
        precision = gram / variance + config.ridge * eye
        covariance_normalized = torch.linalg.inv(precision)
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


def _fit_base_posterior(
    features: torch.Tensor,
    target: torch.Tensor,
    noise: float,
    config: V5Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    variance = max(noise, config.posterior_noise_floor) ** 2
    scale = features.square().mean(dim=1).sqrt().clamp_min(1.0e-8)
    normalized = features / scale.unsqueeze(1)
    gram = torch.einsum("bni,bnj->bij", normalized, normalized)
    eye = torch.eye(2, dtype=features.dtype, device=features.device)
    covariance_normalized = torch.linalg.inv(
        gram / variance + config.ridge * eye
    )
    natural = torch.einsum("bni,bn->bi", normalized, target) / variance
    mean_normalized = torch.einsum(
        "bij,bj->bi", covariance_normalized, natural
    )
    inverse_scale = scale.reciprocal()
    covariance = (
        covariance_normalized
        * inverse_scale.unsqueeze(-1)
        * inverse_scale.unsqueeze(-2)
    )
    return mean_normalized * inverse_scale, covariance


def _predictive_moments(
    features: torch.Tensor,
    proposed: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = _candidate_design(features, proposed)
    mean = torch.einsum("bank,bnk->ban", design, means)
    variance = torch.einsum(
        "bank,bnkl,banl->ban", design, covariances, design
    ).clamp_min(0.0)
    return mean, variance


def noise_normalized_action_score(
    means: torch.Tensor,
    variances: torch.Tensor,
    probabilities: torch.Tensor,
    observation_variance: float,
    *,
    weighted: bool,
) -> torch.Tensor:
    score = means.new_zeros(means.shape[:2])
    for left in range(means.shape[-1]):
        for right in range(left + 1, means.shape[-1]):
            separation = (means[..., left] - means[..., right]).square()
            uncertainty = (
                variances[..., left]
                + variances[..., right]
                + observation_variance
            ).clamp_min(1.0e-10)
            pair = separation / uncertainty
            if weighted:
                pair = pair * (
                    probabilities[:, left] * probabilities[:, right]
                ).unsqueeze(-1)
            score += pair
    return score


def _action_score(
    criterion: str,
    means: torch.Tensor,
    variances: torch.Tensor,
    probabilities: torch.Tensor,
    observation_variance: float,
    correct_position: torch.Tensor,
    oracle_action: bool,
) -> torch.Tensor:
    if criterion == "raw":
        return means.var(dim=-1, unbiased=False)
    weighted = criterion == "weighted"
    score = noise_normalized_action_score(
        means,
        variances,
        probabilities,
        observation_variance,
        weighted=weighted,
    )
    if oracle_action:
        oracle_score = means.new_zeros(means.shape[:2])
        valid = correct_position >= 0
        for candidate in range(means.shape[-1]):
            correct_index = correct_position.clamp_min(0)
            correct_mean = means.gather(
                -1,
                correct_index[:, None, None].expand(-1, means.shape[1], 1),
            ).squeeze(-1)
            correct_variance = variances.gather(
                -1,
                correct_index[:, None, None].expand(-1, means.shape[1], 1),
            ).squeeze(-1)
            pair = (correct_mean - means[..., candidate]).square() / (
                correct_variance
                + variances[..., candidate]
                + observation_variance
            ).clamp_min(1.0e-10)
            different = candidate != correct_index
            oracle_score += (
                pair
                * different.unsqueeze(-1)
                * probabilities[:, candidate].unsqueeze(-1)
            )
        score = torch.where(valid.unsqueeze(-1), oracle_score, score)
    return score


def _update_candidate_belief(
    means: torch.Tensor,
    covariances: torch.Tensor,
    log_probabilities: torch.Tensor,
    design: torch.Tensor,
    observation: torch.Tensor,
    observation_variance: float,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.einsum("bnk,bnk->bn", design, means)
    covariance_design = torch.einsum("bnkl,bnl->bnk", covariances, design)
    variance = (
        torch.einsum("bnk,bnk->bn", design, covariance_design)
        + observation_variance
    ).clamp_min(1.0e-10)
    residual = observation.unsqueeze(-1) - prediction
    likelihood = -0.5 * (variance.log() + residual.square() / variance)
    updated_log = log_probabilities + likelihood
    updated_log -= torch.logsumexp(updated_log, dim=-1, keepdim=True)
    gain = covariance_design / variance.unsqueeze(-1)
    updated_mean = means + gain * residual.unsqueeze(-1)
    updated_covariance = covariances - torch.einsum(
        "bnk,bnl->bnkl", gain, covariance_design
    )
    means = torch.where(active[:, None, None], updated_mean, means)
    covariances = torch.where(
        active[:, None, None, None], updated_covariance, covariances
    )
    log_probabilities = torch.where(
        active.unsqueeze(-1), updated_log, log_probabilities
    )
    return means, covariances, log_probabilities


def _update_base_belief(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    design: torch.Tensor,
    observation: torch.Tensor,
    observation_variance: float,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    covariance_design = torch.einsum("bij,bj->bi", covariance, design)
    prediction = (design * mean).sum(dim=-1)
    variance = (design * covariance_design).sum(dim=-1) + observation_variance
    gain = covariance_design / variance.clamp_min(1.0e-10).unsqueeze(-1)
    updated_mean = mean + gain * (observation - prediction).unsqueeze(-1)
    updated_covariance = covariance - torch.einsum(
        "bi,bj->bij", gain, covariance_design
    )
    return (
        torch.where(active.unsqueeze(-1), updated_mean, mean),
        torch.where(active[:, None, None], updated_covariance, covariance),
    )


def lower_confidence_acceptance(
    incumbent_error: torch.Tensor,
    candidate_error: torch.Tensor,
    complexity: torch.Tensor,
    config: V5Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    improvement = incumbent_error.square() - candidate_error.square()
    mean = improvement.mean(dim=-1)
    standard_error = improvement.std(dim=-1, unbiased=True) / math.sqrt(
        improvement.shape[-1]
    )
    lcb = (
        mean
        - config.lcb_z * standard_error
        - config.acceptance_complexity_price * complexity
    )
    accepted = (lcb > config.lcb_improvement) & (
        candidate_error.square().mean(dim=-1).sqrt() < config.acceptance_rmse
    )
    return accepted, lcb


def _evaluate_strategy(
    config: V5Config,
    seed: int,
    noise: float,
    strategy: str,
    device: torch.device,
) -> dict[str, Any]:
    spec = SPECS[strategy]
    v4 = _v4_config(config, noise)
    generator = torch.Generator(device=device).manual_seed(seed + 80_000)
    noise_generator = torch.Generator(device=device).manual_seed(
        seed + 81_000 + int(round(noise * 10_000))
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
    structural = ~adequate
    correct_operator = torch.full_like(model_class, -1)
    correct_operator[cubic] = CUBIC_OPERATOR
    correct_operator[coupled] = COUPLED_OPERATOR

    passive_x = _sample_features(
        "passive", (episodes, config.passive_context), generator, device
    )
    discovery_x = _sample_features(
        "discovery", (episodes, config.discovery_count), generator, device
    )
    heldout_x = _sample_features(
        "heldout", (episodes, config.heldout_count), generator, device
    )
    action_pool = _sample_features(
        "diagnostic", (episodes, config.action_candidates), generator, device
    )
    passive_clean = true_force(passive_x, theta, model_class, v4)
    discovery_clean = true_force(discovery_x, theta, model_class, v4)
    heldout_clean = true_force(heldout_x, theta, model_class, v4)
    passive_y = passive_clean + noise * torch.randn(
        passive_clean.shape, generator=noise_generator, device=device
    )
    discovery_y = discovery_clean + noise * torch.randn(
        discovery_clean.shape, generator=noise_generator, device=device
    )
    heldout_y = heldout_clean + noise * torch.randn(
        heldout_clean.shape, generator=noise_generator, device=device
    )

    initial_theta = _ridge_fit(passive_x, passive_y, config.ridge)
    discovery_residual = discovery_y - torch.einsum(
        "bni,bi->bn", discovery_x, initial_theta
    )
    _, residual_orthogonal, tangent_ratio = tangent_decomposition(
        discovery_x, discovery_residual, config.ridge
    )
    orthogonal_rms = residual_orthogonal.square().mean(dim=-1).sqrt()
    natural_trigger = (tangent_ratio > config.tangent_ratio_threshold) & (
        orthogonal_rms > max(config.tangent_noise_floor, 1.35 * noise)
    )
    # V5 is a controlled discrimination experiment: for truth represented in
    # the frozen library, trigger/proposal availability are held fixed so a
    # noise sweep does not silently become another detection/proposal sweep.
    trigger = natural_trigger | in_library

    gain = orthogonal_operator_gain(
        discovery_x,
        discovery_residual,
        operator_library(discovery_x),
        ridge=config.ridge,
    )
    complexity = torch.tensor(
        OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device
    )
    proposal_score = gain - config.proposal_complexity_price * complexity
    proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    natural_proposal_recall = (
        proposed == correct_operator.unsqueeze(-1)
    ).any(dim=-1)
    proposed = _ensure_oracle_proposal(proposed, model_class)
    proposal_recall = (proposed == correct_operator.unsqueeze(-1)).any(dim=-1)
    correct_position = torch.where(
        proposal_recall,
        (proposed == correct_operator.unsqueeze(-1)).float().argmax(dim=-1),
        torch.full_like(model_class, -1),
    )

    train_x = torch.cat((passive_x, discovery_x), dim=1)
    train_y = torch.cat((passive_y, discovery_y), dim=1)
    means, covariances = _fit_posteriors(
        train_x, train_y, proposed, noise, config
    )
    base_mean, base_covariance = _fit_base_posterior(
        train_x, train_y, noise, config
    )
    proposed_scores = proposal_score.gather(1, proposed)
    log_probabilities = torch.log_softmax(
        proposed_scores / config.proposal_temperature, dim=-1
    )
    prior_probabilities = log_probabilities.exp().clone()
    prior_entropy = -(prior_probabilities * log_probabilities).sum(dim=-1)
    observation_variance = max(noise, config.posterior_noise_floor) ** 2
    used_actions = torch.zeros(
        (episodes, config.action_candidates), dtype=torch.bool, device=device
    )
    stopped = ~trigger
    probe_count = torch.zeros(episodes, device=device)

    for step in range(spec.budget):
        active = trigger & (~stopped if spec.sequential else True)
        probabilities = log_probabilities.exp()
        pool_means, pool_variances = _predictive_moments(
            action_pool, proposed, means, covariances
        )
        score = _action_score(
            spec.criterion,
            pool_means,
            pool_variances,
            probabilities,
            observation_variance,
            correct_position,
            spec.oracle_action,
        )
        score = score.masked_fill(used_actions, float("-inf"))
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
        used_actions[
            torch.arange(episodes, device=device)[active], action_index[active]
        ] = True
        probe_count += active.float()
        if spec.sequential:
            probabilities = log_probabilities.exp()
            ordered = probabilities.topk(2, dim=-1).values
            sufficient = (ordered[:, 0] > config.select_probability) & (
                ordered[:, 0] - ordered[:, 1] > config.select_margin
            )
            stopped |= active & sufficient

    probabilities = log_probabilities.exp()
    selected_position = probabilities.argmax(dim=-1)
    selected_operator = proposed.gather(
        1, selected_position.unsqueeze(-1)
    ).squeeze(-1)
    if spec.oracle_selection:
        selected_operator = correct_operator.clamp_min(0)
        selected_position = correct_position.clamp_min(0)
    deferred = (
        trigger & ~stopped if spec.sequential and not spec.oracle_selection else torch.zeros_like(trigger)
    )
    selection_correct = selected_operator == correct_operator

    heldout_design = _candidate_design(heldout_x, proposed)
    heldout_candidate_all = torch.einsum("bank,bnk->ban", heldout_design, means)
    heldout_candidate = heldout_candidate_all.gather(
        -1,
        selected_position[:, None, None].expand(-1, config.heldout_count, 1),
    ).squeeze(-1)
    heldout_base = torch.einsum("bni,bi->bn", heldout_x, base_mean)
    initial_base = _ridge_fit(train_x, train_y, config.ridge)
    fitted_residual = train_y - torch.einsum("bni,bi->bn", train_x, initial_base)
    heldout_fallback = heldout_base + _residual_predict(
        train_x, fitted_residual, heldout_x, config.residual_ridge
    ).clamp(-0.6, 0.6)
    selected_complexity = complexity[selected_operator]
    if spec.lcb_acceptance:
        accepted, acceptance_statistic = lower_confidence_acceptance(
            heldout_y - heldout_fallback,
            heldout_y - heldout_candidate,
            selected_complexity,
            config,
        )
        base_mse = (heldout_y - heldout_base).square().mean(dim=-1)
        candidate_mse = (heldout_y - heldout_candidate).square().mean(dim=-1)
        point_gain = (
            base_mse
            - candidate_mse
            - config.acceptance_complexity_price * selected_complexity
        )
        accepted &= point_gain > config.acceptance_margin
    else:
        base_mse = (heldout_y - heldout_base).square().mean(dim=-1)
        candidate_mse = (heldout_y - heldout_candidate).square().mean(dim=-1)
        acceptance_statistic = (
            base_mse
            - candidate_mse
            - config.acceptance_complexity_price * selected_complexity
        )
        accepted = (
            (acceptance_statistic > config.acceptance_margin)
            & (candidate_mse.sqrt() < config.acceptance_rmse)
        )
    accepted &= trigger & ~deferred
    if spec.oracle_selection:
        accepted = in_library

    final_operator = torch.full_like(model_class, -2)
    final_operator[trigger & ~accepted] = -1
    final_operator[accepted] = selected_operator[accepted]
    if spec.oracle_selection:
        final_operator[adequate] = -2
        final_operator[outside] = -1
    unknown = final_operator == -1
    discovered = final_operator >= 0
    exact_recovery = final_operator == correct_operator

    def predict(query_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_design = _candidate_design(query_x, proposed)
        candidate_all = torch.einsum("bank,bnk->ban", query_design, means)
        candidate = candidate_all.gather(
            -1,
            selected_position[:, None, None].expand(-1, query_x.shape[1], 1),
        ).squeeze(-1)
        base = torch.einsum("bqi,bi->bq", query_x, base_mean)
        fallback = base + _residual_predict(
            train_x, fitted_residual, query_x, config.residual_ridge
        ).clamp(-0.6, 0.6)
        prediction = torch.where(discovered.unsqueeze(-1), candidate, base)
        prediction = torch.where(unknown.unsqueeze(-1), fallback, prediction)
        return prediction, fallback

    test_generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    intervention_x = _sample_features(
        "intervention", (episodes, config.queries), test_generator, device
    )
    extrapolation_x = _sample_features(
        "intervention", (episodes, config.queries), test_generator, device
    )
    extrapolation_x[..., 0] += 0.035
    intervention_y = true_force(intervention_x, theta, model_class, v4)
    extrapolation_y = true_force(extrapolation_x, theta, model_class, v4)
    intervention_prediction, _ = predict(intervention_x)
    extrapolation_prediction, _ = predict(extrapolation_x)
    intervention_error = intervention_prediction - intervention_y
    extrapolation_error = extrapolation_prediction - extrapolation_y

    final_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    has_probe = probe_count > 0
    correct_proposed = proposal_recall & in_library
    prior_correct_probability = prior_probabilities.gather(
        1, correct_position.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    final_correct_probability = probabilities.gather(
        1, correct_position.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    failed_recovery = structural & ~exact_recovery
    final_complexity = torch.where(
        discovered, complexity[selected_operator], torch.zeros_like(probe_count)
    )
    intervention_mse = intervention_error.square().mean()
    selected_mean = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)
    return {
        "strategy": strategy,
        "noise": noise,
        "natural_topk_proposal_recall": float(
            natural_proposal_recall[in_library].float().mean()
        ),
        "natural_trigger_rate_in_library": float(
            natural_trigger[in_library].float().mean()
        ),
        "natural_trigger_rate_outside": float(
            natural_trigger[outside].float().mean()
        ),
        "controlled_candidate_coverage": float(
            proposal_recall[in_library].float().mean()
        ),
        "selection_accuracy_given_proposal": float(
            selection_correct[correct_proposed & ~deferred].float().mean()
        )
        if bool((correct_proposed & ~deferred).any())
        else 0.0,
        "correct_selection_acceptance": float(
            accepted[selection_correct & in_library].float().mean()
        )
        if bool((selection_correct & in_library).any())
        else 0.0,
        "exact_operator_recovery": float(exact_recovery[in_library].float().mean()),
        "selection_defer_rate": float(deferred[trigger].float().mean())
        if bool(trigger.any())
        else 0.0,
        "incorrect_expansion_rate_adequate": float(discovered[adequate].float().mean()),
        "outside_rejection_rate": float(unknown[outside].float().mean()),
        "outside_rejection_given_trigger": float(
            unknown[outside & natural_trigger].float().mean()
        )
        if bool((outside & natural_trigger).any())
        else 0.0,
        "outside_forced_explanation_rate": float(discovered[outside].float().mean()),
        "probe_count": float(probe_count.mean()),
        "probe_count_recovered": float(probe_count[exact_recovery & in_library].mean())
        if bool((exact_recovery & in_library).any())
        else 0.0,
        "probe_count_failed": float(probe_count[failed_recovery].mean())
        if bool(failed_recovery.any())
        else 0.0,
        "entropy_reduction_per_probe": float(
            ((prior_entropy - final_entropy)[has_probe] / probe_count[has_probe]).mean()
        )
        if bool(has_probe.any())
        else 0.0,
        "correct_probability_gain_per_probe": float(
            (
                (final_correct_probability - prior_correct_probability)[correct_proposed]
                / probe_count[correct_proposed].clamp_min(1.0)
            ).mean()
        )
        if bool(correct_proposed.any())
        else 0.0,
        "residual_fallback_before_discovery": float((trigger & structural).float().mean()),
        "residual_fallback_after_discovery": float((unknown & structural).float().mean()),
        "residual_usage_given_correct_recovery": float(unknown[exact_recovery & in_library].float().mean())
        if bool((exact_recovery & in_library).any())
        else 0.0,
        "residual_usage_given_failed_recovery": float(unknown[failed_recovery].float().mean())
        if bool(failed_recovery.any())
        else 0.0,
        "parameter_rmse_k": float(
            torch.sqrt((selected_mean[:, 0] - theta[:, 0]).square().mean())
        ),
        "parameter_rmse_c": float(
            torch.sqrt((selected_mean[:, 1] - theta[:, 1]).square().mean())
        ),
        "intervention_force_rmse": float(torch.sqrt(intervention_mse)),
        "intervention_rmse_in_library": float(
            torch.sqrt(intervention_error[in_library].square().mean())
        ),
        "intervention_rmse_outside": float(
            torch.sqrt(intervention_error[outside].square().mean())
        ),
        "extrapolation_force_rmse": float(
            torch.sqrt(extrapolation_error.square().mean())
        ),
        "extrapolation_rmse_in_library": float(
            torch.sqrt(extrapolation_error[in_library].square().mean())
        ),
        "model_complexity_cost": float(final_complexity.mean())
        * config.model_complexity_cost,
        "residual_compute_cost": float(unknown.float().mean()) * config.residual_cost,
        "probe_cost": float(probe_count.mean()) * config.action_cost,
        "total_objective": float(intervention_mse)
        + float(final_complexity.mean()) * config.model_complexity_cost
        + float(unknown.float().mean()) * config.residual_cost
        + float(probe_count.mean()) * config.action_cost,
    }


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
    metrics = (
        "natural_topk_proposal_recall",
        "natural_trigger_rate_in_library",
        "natural_trigger_rate_outside",
        "controlled_candidate_coverage",
        "selection_accuracy_given_proposal",
        "correct_selection_acceptance",
        "exact_operator_recovery",
        "selection_defer_rate",
        "incorrect_expansion_rate_adequate",
        "outside_rejection_rate",
        "outside_rejection_given_trigger",
        "probe_count",
        "entropy_reduction_per_probe",
        "residual_fallback_after_discovery",
        "intervention_force_rmse",
        "extrapolation_force_rmse",
        "total_objective",
    )
    comparisons = (
        ("normalized2_minus_raw2", "normalized_fixed2", "raw_fixed2"),
        ("weighted2_minus_normalized2", "weighted_fixed2", "normalized_fixed2"),
        ("weighted3_minus_weighted2", "weighted_fixed3", "weighted_fixed2"),
        ("sequential_minus_weighted3", "sequential_weighted", "weighted_fixed3"),
        ("lcb_minus_sequential", "sequential_lcb", "sequential_weighted"),
        ("oracle_action_minus_sequential", "oracle_action", "sequential_weighted"),
        ("oracle_selection_minus_lcb", "oracle_selection", "sequential_lcb"),
    )
    output: list[dict[str, Any]] = []
    for noise in noises:
        for comparison, left, right in comparisons:
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
                        "comparison": comparison,
                        "metric": metric,
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
    return output


def run_v5(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V5Config | None = None,
) -> dict[str, Any]:
    cfg = config or V5Config()
    device = resolve_device(device_name)
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        for noise in cfg.noise_levels:
            for strategy in STRATEGIES:
                row = _evaluate_strategy(cfg, seed, noise, strategy, device)
                row["seed"] = seed
                rows.append(row)
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
        "proposal_topk": cfg.proposal_topk,
        "posterior_weighted_score_is_pairwise_ig_surrogate": True,
    }
    _write_json(root / "summary.json", summary)
    return summary

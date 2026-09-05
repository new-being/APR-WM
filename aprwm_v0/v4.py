from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _write_csv, _write_json
from .v3 import _calibration, _residual_predict, _ridge_fit, tangent_decomposition


STRATEGIES = (
    "residual_only",
    "passive_selection",
    "raw_active_discovery",
    "active_discovery",
    "oracle_proposal",
    "oracle_selection",
)

OPERATOR_NAMES = (
    "x2",
    "x3",
    "signed_v2",
    "xv",
    "abs_v_v",
    "position_damping",
    "velocity_coupled",
    "saturation",
)
OPERATOR_COMPLEXITY = (1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 3.0)
CUBIC_OPERATOR = 1
COUPLED_OPERATOR = 6


@dataclass
class V4Config:
    seeds: tuple[int, ...] = (201, 211, 221, 231, 241)
    episodes: int = 4_096
    passive_context: int = 8
    discovery_count: int = 16
    heldout_count: int = 8
    action_candidates: int = 32
    diagnostic_probes: int = 2
    queries: int = 16
    proposal_topk: int = 3
    observation_noise: float = 0.05
    cubic_alpha: float = 800.0
    coupled_alpha: float = 95.0
    outside_alpha: float = 0.24
    tangent_ratio_threshold: float = 0.02
    tangent_noise_threshold: float = 1.35
    proposal_complexity_price: float = 2.0e-4
    acceptance_complexity_price: float = 5.0e-4
    acceptance_margin: float = 1.5e-3
    acceptance_rmse: float = 0.095
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    residual_cost: float = 0.01
    action_cost: float = 0.002
    model_complexity_cost: float = 0.0015


def operator_library(features: torch.Tensor) -> torch.Tensor:
    x, v = features[..., 0], features[..., 1]
    operators = (
        x.square(),
        x.pow(3),
        v.square(),
        x * v,
        v * v.abs(),
        x * torch.tanh(v / 0.06),
        x.square() * torch.tanh(v / 0.06),
        torch.tanh((x - 0.045) / 0.018),
    )
    return torch.stack(operators, dim=-1)


def true_structural_force(
    features: torch.Tensor,
    model_class: torch.Tensor,
    config: V4Config,
) -> torch.Tensor:
    """Classes: 0 linear, 1 cubic, 2 coupled, 3 outside dictionary."""
    x, v = features[..., 0], features[..., 1]
    while model_class.ndim < x.ndim:
        model_class = model_class.unsqueeze(-1)
    cubic = config.cubic_alpha * x.pow(3)
    coupled = config.coupled_alpha * x.square() * torch.tanh(v / 0.06)
    # A thresholded oscillatory interaction is deliberately absent from Phi.
    outside = config.outside_alpha * (x > 0.05).float() * torch.sin(v / 0.032)
    return torch.where(
        model_class == 1,
        cubic,
        torch.where(
            model_class == 2,
            coupled,
            torch.where(model_class == 3, outside, torch.zeros_like(x)),
        ),
    )


def true_force(
    features: torch.Tensor,
    theta: torch.Tensor,
    model_class: torch.Tensor,
    config: V4Config,
) -> torch.Tensor:
    while theta.ndim < features.ndim:
        theta = theta.unsqueeze(-2)
    return (features * theta).sum(dim=-1) + true_structural_force(
        features, model_class, config
    )


def _sample_features(
    regime: str,
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if regime == "passive":
        x = 0.018 + 0.006 * torch.rand(*shape, generator=generator, device=device)
        v = 0.04 * torch.rand(*shape, generator=generator, device=device) - 0.02
    elif regime == "discovery":
        # Correlation makes observational operator ranking intentionally ambiguous.
        x = 0.014 + 0.072 * torch.rand(*shape, generator=generator, device=device)
        v = 2.5 * (x - 0.050) + 0.040 * torch.randn(
            *shape, generator=generator, device=device
        )
    elif regime in ("diagnostic", "heldout"):
        x = 0.012 + 0.078 * torch.rand(*shape, generator=generator, device=device)
        v = 0.44 * torch.rand(*shape, generator=generator, device=device) - 0.22
    elif regime == "intervention":
        x = 0.060 + 0.030 * torch.rand(*shape, generator=generator, device=device)
        v = 0.44 * torch.rand(*shape, generator=generator, device=device) - 0.22
    else:
        raise ValueError(regime)
    return torch.stack((x, v), dim=-1)


def orthogonal_operator_gain(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    operators: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    """Explained tangent-orthogonal residual energy for every operator."""
    _, residual_orthogonal, _ = tangent_decomposition(jacobian, residual, ridge)
    gains = []
    for index in range(operators.shape[-1]):
        _, operator_orthogonal, _ = tangent_decomposition(
            jacobian, operators[..., index], ridge
        )
        coefficient = (
            (residual_orthogonal * operator_orthogonal).sum(dim=-1)
            / operator_orthogonal.square().sum(dim=-1).clamp_min(1.0e-12)
        )
        remaining = residual_orthogonal - coefficient.unsqueeze(-1) * operator_orthogonal
        gain = (
            residual_orthogonal.square().sum(dim=-1)
            - remaining.square().sum(dim=-1)
        ) / jacobian.shape[-2]
        gains.append(gain)
    return torch.stack(gains, dim=-1)


def raw_operator_gain(
    residual: torch.Tensor, operators: torch.Tensor
) -> torch.Tensor:
    gains = []
    for index in range(operators.shape[-1]):
        operator = operators[..., index]
        coefficient = (residual * operator).sum(dim=-1) / operator.square().sum(
            dim=-1
        ).clamp_min(1.0e-12)
        remaining = residual - coefficient.unsqueeze(-1) * operator
        gains.append(
            (residual.square().sum(dim=-1) - remaining.square().sum(dim=-1))
            / residual.shape[-1]
        )
    return torch.stack(gains, dim=-1)


def _candidate_coefficients(
    features: torch.Tensor,
    target: torch.Tensor,
    config: V4Config,
) -> torch.Tensor:
    operators = operator_library(features)
    coefficients = []
    for index in range(len(OPERATOR_NAMES)):
        design = torch.cat((features, operators[..., index : index + 1]), dim=-1)
        coefficients.append(_ridge_fit(design, target, config.ridge))
    return torch.stack(coefficients, dim=1)


def _candidate_predictions(
    features: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    operators = operator_library(features)
    base = torch.einsum("bni,bki->bnk", features, coefficients[..., :2])
    structural = operators * coefficients[:, None, :, 2]
    return base + structural


def _ensure_oracle_proposal(
    proposed: torch.Tensor, model_class: torch.Tensor
) -> torch.Tensor:
    output = proposed.clone()
    correct = torch.full_like(model_class, -1)
    correct[model_class == 1] = CUBIC_OPERATOR
    correct[model_class == 2] = COUPLED_OPERATOR
    in_library = correct >= 0
    already = (output == correct.unsqueeze(-1)).any(dim=-1)
    replace = in_library & ~already
    output[replace, -1] = correct[replace]
    return output


def _select_diagnostic_actions(
    action_pool: torch.Tensor,
    predictions: torch.Tensor,
    proposed: torch.Tensor,
    probes: int,
) -> torch.Tensor:
    selected_predictions = predictions.gather(
        2, proposed.unsqueeze(1).expand(-1, action_pool.shape[1], -1)
    )
    disagreement = selected_predictions.var(dim=-1, unbiased=False)
    action_indices = disagreement.topk(probes, dim=-1).indices
    return action_pool.gather(
        1, action_indices.unsqueeze(-1).expand(-1, -1, action_pool.shape[-1])
    )


def _gather_candidate(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    shape = [values.shape[0]] + [1] * (values.ndim - 2) + [1]
    expanded = index.view(*shape).expand(*values.shape[:-1], 1)
    return values.gather(-1, expanded).squeeze(-1)


def _earliest_tangent_trigger(
    evidence_x: torch.Tensor,
    evidence_residual: torch.Tensor,
    config: V4Config,
) -> torch.Tensor:
    result = torch.full(
        (evidence_x.shape[0],),
        config.discovery_count + 1,
        dtype=torch.long,
        device=evidence_x.device,
    )
    for count in range(4, config.discovery_count + 1):
        _, orthogonal, ratio = tangent_decomposition(
            evidence_x[:, :count], evidence_residual[:, :count], config.ridge
        )
        trigger = (ratio > config.tangent_ratio_threshold) & (
            orthogonal.square().mean(dim=-1).sqrt()
            > config.tangent_noise_threshold * config.observation_noise
        )
        result[(result > config.discovery_count) & trigger] = count
    return result


def _evaluate_strategy(
    config: V4Config,
    seed: int,
    strategy: str,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + 60_000)
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
    passive_y = true_force(passive_x, theta, model_class, config)
    discovery_y = true_force(discovery_x, theta, model_class, config)
    heldout_clean = true_force(heldout_x, theta, model_class, config)
    passive_y += config.observation_noise * torch.randn(
        passive_y.shape, generator=generator, device=device
    )
    discovery_y += config.observation_noise * torch.randn(
        discovery_y.shape, generator=generator, device=device
    )
    heldout_y = heldout_clean + config.observation_noise * torch.randn(
        heldout_clean.shape, generator=generator, device=device
    )

    initial_theta = _ridge_fit(passive_x, passive_y, config.ridge)
    discovery_residual = discovery_y - torch.einsum(
        "bni,bi->bn", discovery_x, initial_theta
    )
    _, residual_orthogonal, tangent_ratio = tangent_decomposition(
        discovery_x, discovery_residual, config.ridge
    )
    orthogonal_rms = residual_orthogonal.square().mean(dim=-1).sqrt()
    trigger = (tangent_ratio > config.tangent_ratio_threshold) & (
        orthogonal_rms > config.tangent_noise_threshold * config.observation_noise
    )
    tangent_score = orthogonal_rms / (
        orthogonal_rms + config.tangent_noise_threshold * config.observation_noise
    )
    detection_brier, detection_ece = _calibration(tangent_score, structural)

    operators = operator_library(discovery_x)
    if strategy == "raw_active_discovery":
        gain = raw_operator_gain(discovery_residual, operators)
    else:
        gain = orthogonal_operator_gain(
            discovery_x,
            discovery_residual,
            operators,
            ridge=config.ridge,
        )
    complexity = torch.tensor(
        OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device
    )
    proposal_score = gain - config.proposal_complexity_price * complexity
    proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    if strategy == "oracle_proposal":
        proposed = _ensure_oracle_proposal(proposed, model_class)
    proposal_recall = (proposed == correct_operator.unsqueeze(-1)).any(dim=-1)

    train_x = torch.cat((passive_x, discovery_x), dim=1)
    train_y = torch.cat((passive_y, discovery_y), dim=1)
    candidate_coefficients = _candidate_coefficients(train_x, train_y, config)
    pool_predictions = _candidate_predictions(action_pool, candidate_coefficients)

    uses_active_probe = strategy in (
        "raw_active_discovery",
        "active_discovery",
        "oracle_proposal",
        "oracle_selection",
    )
    if uses_active_probe:
        probe_x = _select_diagnostic_actions(
            action_pool,
            pool_predictions,
            proposed,
            config.diagnostic_probes,
        )
        probe_clean = true_force(probe_x, theta, model_class, config)
        probe_y = probe_clean + config.observation_noise * torch.randn(
            probe_clean.shape, generator=generator, device=device
        )
        probe_predictions = _candidate_predictions(probe_x, candidate_coefficients)
        probe_loss = (probe_predictions - probe_y.unsqueeze(-1)).square().mean(dim=1)
        if strategy == "oracle_selection":
            selected_operator = correct_operator.clamp_min(0)
        else:
            proposed_loss = probe_loss.gather(1, proposed)
            selected_position = proposed_loss.argmin(dim=-1)
            selected_operator = proposed.gather(
                1, selected_position.unsqueeze(-1)
            ).squeeze(-1)
        executed_probes = trigger.float() * config.diagnostic_probes
        fit_x = torch.cat((train_x, probe_x), dim=1)
        fit_y = torch.cat((train_y, probe_y), dim=1)
    else:
        selected_operator = proposed[:, 0]
        executed_probes = torch.zeros(episodes, device=device)
        fit_x, fit_y = train_x, train_y

    fitted_candidates = _candidate_coefficients(fit_x, fit_y, config)
    fitted_base = _ridge_fit(fit_x, fit_y, config.ridge)
    if uses_active_probe:
        final_coefficients = torch.where(
            trigger[:, None, None], fitted_candidates, candidate_coefficients
        )
        train_base = _ridge_fit(train_x, train_y, config.ridge)
        base_coefficients = torch.where(
            trigger.unsqueeze(-1), fitted_base, train_base
        )
    else:
        final_coefficients = fitted_candidates
        base_coefficients = fitted_base
    base_heldout = torch.einsum("bni,bi->bn", heldout_x, base_coefficients)
    candidate_heldout_all = _candidate_predictions(heldout_x, final_coefficients)
    selected_heldout = _gather_candidate(candidate_heldout_all, selected_operator)
    base_mse = (heldout_y - base_heldout).square().mean(dim=-1)
    candidate_mse = (heldout_y - selected_heldout).square().mean(dim=-1)
    selected_complexity = complexity[selected_operator]
    acceptance_gain = (
        base_mse
        - candidate_mse
        - config.acceptance_complexity_price * selected_complexity
    )
    accepted = (
        trigger
        & (acceptance_gain > config.acceptance_margin)
        & (candidate_mse.sqrt() < config.acceptance_rmse)
    )

    if strategy == "oracle_selection":
        accepted = in_library
    if strategy == "residual_only":
        accepted = torch.zeros_like(trigger)
    # -2: retain linear, -1: explicit unknown/fallback, >=0: discovered operator.
    final_operator = torch.full_like(model_class, -2)
    final_operator[trigger & ~accepted] = -1
    final_operator[accepted] = selected_operator[accepted]
    if strategy == "oracle_selection":
        final_operator[adequate] = -2
        final_operator[outside] = -1

    test_generator = torch.Generator(device=device).manual_seed(seed + 70_000)
    intervention_x = _sample_features(
        "intervention", (episodes, config.queries), test_generator, device
    )
    intervention_y = true_force(intervention_x, theta, model_class, config)
    base_prediction = torch.einsum(
        "bqi,bi->bq", intervention_x, base_coefficients
    )
    all_operator_predictions = _candidate_predictions(
        intervention_x, final_coefficients
    )
    selected_prediction = _gather_candidate(
        all_operator_predictions, selected_operator
    )
    fitted_residual = fit_y - torch.einsum("bni,bi->bn", fit_x, base_coefficients)
    fallback_correction = _residual_predict(
        fit_x,
        fitted_residual,
        intervention_x,
        config.residual_ridge,
    )
    # The generic fallback is a temporary predictor, not an unrestricted
    # extrapolator. Bound it to the force scale represented by this benchmark.
    fallback_prediction = base_prediction + fallback_correction.clamp(-0.6, 0.6)
    prediction = torch.where(
        (final_operator >= 0).unsqueeze(-1), selected_prediction, base_prediction
    )
    prediction = torch.where(
        (final_operator == -1).unsqueeze(-1), fallback_prediction, prediction
    )

    exact_recovery = final_operator == correct_operator
    selection_correct = selected_operator == correct_operator
    selected_in_proposal = proposal_recall & in_library
    correctly_selected = selection_correct & in_library
    earliest = _earliest_tangent_trigger(
        discovery_x, discovery_residual, config
    )
    recovered = exact_recovery & in_library
    observations_to_recovery = earliest.float() + executed_probes
    unknown = final_operator == -1
    discovered = final_operator >= 0
    intervention_error = prediction - intervention_y
    intervention_mse = intervention_error.square().mean()
    final_complexity = torch.where(
        discovered, complexity[selected_operator], torch.zeros_like(executed_probes)
    )
    residual_before = trigger & structural
    residual_after = unknown & structural
    return {
        "strategy": strategy,
        "detection_brier": detection_brier,
        "detection_ece": detection_ece,
        "topk_proposal_recall": float(proposal_recall[in_library].float().mean()),
        "selection_accuracy_given_proposal": float(
            selection_correct[selected_in_proposal].float().mean()
        )
        if bool(selected_in_proposal.any())
        else 0.0,
        "acceptance_rate_given_correct_selection": float(
            accepted[correctly_selected].float().mean()
        )
        if bool(correctly_selected.any())
        else 0.0,
        "exact_operator_recovery": float(exact_recovery[in_library].float().mean()),
        "incorrect_expansion_rate_adequate": float(discovered[adequate].float().mean()),
        "outside_rejection_rate": float(unknown[outside].float().mean()),
        "outside_forced_explanation_rate": float(discovered[outside].float().mean()),
        "observations_to_recovery": float(
            observations_to_recovery[recovered].mean()
        )
        if bool(recovered.any())
        else float(config.discovery_count + config.diagnostic_probes + 1),
        "diagnostic_probe_count": float(executed_probes.mean()),
        "revision_count": float(discovered.float().mean()),
        "final_model_complexity": float(final_complexity.mean()),
        "unknown_dwell_rate": float(unknown.float().mean()),
        "residual_fallback_before_discovery": float(residual_before.float().mean()),
        "residual_fallback_after_discovery": float(residual_after.float().mean()),
        "intervention_force_rmse": float(torch.sqrt(intervention_mse)),
        "intervention_rmse_adequate": float(
            torch.sqrt(intervention_error[adequate].square().mean())
        ),
        "intervention_rmse_cubic": float(
            torch.sqrt(intervention_error[cubic].square().mean())
        ),
        "intervention_rmse_coupled": float(
            torch.sqrt(intervention_error[coupled].square().mean())
        ),
        "intervention_rmse_in_library": float(
            torch.sqrt(intervention_error[in_library].square().mean())
        ),
        "intervention_rmse_outside": float(
            torch.sqrt(intervention_error[outside].square().mean())
        ),
        "model_complexity_cost": float(final_complexity.mean())
        * config.model_complexity_cost,
        "residual_compute_cost": float(unknown.float().mean()) * config.residual_cost,
        "probe_cost": float(executed_probes.mean()) * config.action_cost,
        "total_objective": float(intervention_mse)
        + float(final_complexity.mean()) * config.model_complexity_cost
        + float(unknown.float().mean()) * config.residual_cost
        + float(executed_probes.mean()) * config.action_cost,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        members = [row for row in rows if row["strategy"] == strategy]
        for metric in sorted(set(members[0]) - {"strategy", "seed"}):
            values = [float(row[metric]) for row in members]
            low, high = _bootstrap_ci(values)
            output.append(
                {
                    "strategy": strategy,
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
    indexed = {(int(row["seed"]), row["strategy"]): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics = (
        "topk_proposal_recall",
        "selection_accuracy_given_proposal",
        "acceptance_rate_given_correct_selection",
        "exact_operator_recovery",
        "incorrect_expansion_rate_adequate",
        "outside_rejection_rate",
        "diagnostic_probe_count",
        "residual_fallback_after_discovery",
        "intervention_force_rmse",
        "intervention_rmse_in_library",
        "total_objective",
    )
    comparisons = (
        ("active_minus_passive", "active_discovery", "passive_selection"),
        ("active_minus_raw", "active_discovery", "raw_active_discovery"),
        ("oracle_proposal_minus_active", "oracle_proposal", "active_discovery"),
        ("oracle_selection_minus_oracle_proposal", "oracle_selection", "oracle_proposal"),
        ("active_minus_residual", "active_discovery", "residual_only"),
    )
    output: list[dict[str, Any]] = []
    for comparison, left, right in comparisons:
        for metric in metrics:
            values = [
                float(indexed[(seed, left)][metric])
                - float(indexed[(seed, right)][metric])
                for seed in seeds
            ]
            low, high = _bootstrap_ci(values)
            output.append(
                {
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


def run_v4(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V4Config | None = None,
) -> dict[str, Any]:
    cfg = config or V4Config()
    device = resolve_device(device_name)
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        for strategy in STRATEGIES:
            row = _evaluate_strategy(cfg, seed, strategy, device)
            row["seed"] = seed
            rows.append(row)
    root = Path(output_dir)
    _write_csv(root / "strategies.csv", rows)
    _write_csv(root / "strategies_aggregate.csv", _aggregate(rows))
    _write_csv(root / "paired_differences.csv", _paired(rows))
    summary = {
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds)},
        "rows": len(rows),
        "operator_library": list(OPERATOR_NAMES),
        "truth_in_library": ["cubic", "velocity_coupled"],
        "truth_outside_library": ["thresholded_velocity_oscillation"],
        "proposal_and_selection_are_separate": True,
    }
    _write_json(root / "summary.json", summary)
    return summary

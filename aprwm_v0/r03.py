from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend, _aggregate_seed_rows, _sample_local_points
from .r02 import (
    DRAG_OPERATOR,
    V6R02Config,
    _backend_config,
    _candidate_design,
    _fit_candidate_posteriors,
    _force_observations,
    _model_rollout,
    _normalize_fallback_features,
    _predictive_moments,
)
from .train import resolve_device, seed_everything
from .v06 import _binary_auroc, _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import OPERATOR_COMPLEXITY, OPERATOR_NAMES, operator_library, orthogonal_operator_gain
from .v5 import _action_score, _fit_base_posterior, _update_base_belief, _update_candidate_belief
from .v6 import V6Config, _sequential_decision


REGIMES = ("c0_parameter", "c1_drag", "c2_history", "r0n_native")


@dataclass
class V6R03Config(V6R02Config):
    seeds: tuple[int, ...] = (1001, 1011, 1021, 1031, 1041)
    c2_unknown_rejection_min: float = 0.90
    c2_wrong_revision_max: float = 0.05
    native_revision_gain_min: float = 0.0
    native_assimilation_min: float = 0.0
    valid_window_fraction_min: float = 0.70
    joint_limit_margin: float = 0.02


def _collect_seed(
    config: V6R03Config, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 190_000)
    backend_config = _backend_config(config)
    names = (
        "passive_physical", "passive_tangent", "passive_y",
        "discovery_physical", "discovery_tangent", "discovery_y",
        "action_physical", "action_tangent", "action_y",
        "validation_physical", "validation_tangent", "validation_y",
        "query_state", "query_memory", "rollout_truth", "rollout_validity",
    )
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    regimes = []
    densities = []
    dampings = []
    for regime in range(4):
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
                regime=regime if regime < 3 else 0,
                config=backend_config,
                preserve_engine_dissipation=regime == 3,
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

            query_state, query_memory = _sample_local_points(
                config.query_count, "intervention", generator
            )
            truth = []
            validity = []
            for horizon in config.horizons:
                horizon_rows = []
                horizon_validity = []
                for row, hidden in zip(query_state, query_memory):
                    q, velocity, valid = backend.transition_with_validity(
                        float(row[0]), float(row[1]), float(row[2]), float(hidden),
                        horizon=horizon,
                        limit_margin=config.joint_limit_margin,
                    )
                    horizon_rows.append((q, velocity))
                    horizon_validity.append(valid)
                truth.append(torch.tensor(horizon_rows, dtype=torch.float32))
                validity.append(torch.tensor(horizon_validity, dtype=torch.bool))
            fields["query_state"].append(query_state)
            fields["query_memory"].append(query_memory)
            fields["rollout_truth"].append(torch.stack(truth, dim=1))
            fields["rollout_validity"].append(torch.stack(validity, dim=1))
            regimes.append(regime)
            densities.append(density)
            dampings.append(damping)

    output = {name: torch.stack(values).to(device) for name, values in fields.items()}
    output["regime"] = torch.tensor(regimes, dtype=torch.long, device=device)
    output["density"] = torch.tensor(densities, dtype=torch.float32, device=device)
    output["damping"] = torch.tensor(dampings, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 191_000)
    for name in ("passive_y", "discovery_y", "action_y", "validation_y"):
        output[name] = output[name] + config.force_observation_noise * torch.randn(
            output[name].shape, generator=noise_generator, device=device
        )
    return output


def _masked_rmse(error: torch.Tensor, mask: torch.Tensor) -> float:
    selected = error[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.square().mean().sqrt())


def _evaluate_seed(
    config: V6R03Config, seed: int, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _collect_seed(config, seed, device)
    regime = data["regime"]
    c0, c1, c2, native = (regime == index for index in range(4))
    structural = ~c0

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
    triggered = tangent_score > (
        config.trigger_noise_multiplier * config.force_observation_noise
    )
    gain = orthogonal_operator_gain(
        data["discovery_tangent"], residual_orthogonal,
        operator_library(data["discovery_physical"]), ridge=config.ridge,
    )
    complexity = torch.tensor(OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device)
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
    incumbent_sq = (data["validation_y"] - base_prediction).square()
    candidate_sq = (data["validation_y"] - selected_prediction).square()
    competitor_sq = (data["validation_y"] - competitor_prediction).square()
    accepted, rejected, deferred, samples = _sequential_decision(
        incumbent_sq, competitor_sq, candidate_sq,
        complexity[selected_operator], config.force_observation_noise**2,
        v6, method="bf",
    )
    accepted &= triggered
    rejected &= triggered
    deferred &= triggered
    samples = torch.where(triggered, samples, torch.zeros_like(samples))
    exact_c1 = c1 & selection_correct & accepted
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
            normalized_train, fitted_residual,
            _normalize_fallback_features(physical), config.residual_ridge,
        ).clamp(-0.4, 0.4)

    def candidate_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        design = _candidate_design(tangent, physical, proposed)
        all_predictions = torch.einsum("bank,bnk->ban", design, means)
        return all_predictions.gather(
            -1, selected_position[:, None, None].expand(-1, physical.shape[1], 1)
        ).squeeze(-1)

    def revised_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.where(
            accepted[:, None], candidate_force(tangent, physical),
            torch.where(
                triggered[:, None], fallback_force(tangent, physical),
                torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device),
            ),
        )

    rollout_outputs = {
        "physics": _model_rollout(config, data, physics_force),
        "no_revision": _model_rollout(config, data, fallback_force),
        "frozen_v6": _model_rollout(config, data, revised_force),
    }
    truth = data["rollout_truth"]
    validity = data["rollout_validity"]
    scale = torch.tensor((1.0, 2.0), device=device)
    horizon_rows: list[dict[str, Any]] = []
    per_episode_h16: dict[str, torch.Tensor] = {}
    frozen_v6_h16_stability = torch.ones_like(regime, dtype=torch.float32)
    for model, (prediction, stable) in rollout_outputs.items():
        for horizon_index, horizon in enumerate(config.horizons):
            error = (prediction[:, :, horizon_index] - truth[:, :, horizon_index]) / scale
            query_mse = error.square().mean(dim=-1)
            valid_query = validity[:, :, horizon_index]
            episode_rmse = torch.sqrt(
                (query_mse * valid_query).sum(dim=-1)
                / valid_query.sum(dim=-1).clamp_min(1)
            )
            if horizon == config.horizons[-1]:
                per_episode_h16[model] = episode_rmse
                if model == "frozen_v6":
                    frozen_v6_h16_stability = stable[:, :, horizon_index].float().mean(dim=-1)
            for regime_index, regime_name in enumerate(REGIMES):
                regime_mask = regime == regime_index
                valid_mask = regime_mask[:, None] & validity[:, :, horizon_index]
                horizon_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "regime": regime_name,
                        "horizon": horizon,
                        "state_rmse": _masked_rmse(error, valid_mask.unsqueeze(-1).expand_as(error)),
                        "stable_fraction": float(stable[regime_mask, :, horizon_index].float().mean()),
                        "valid_window_fraction": float(validity[regime_mask, :, horizon_index].float().mean()),
                    }
                )

    native_accepted = native & accepted
    if bool(native_accepted.any()):
        native_validation_gain = float(
            (incumbent_sq.mean(dim=-1) - candidate_sq.mean(dim=-1))[native_accepted].mean()
        )
        native_h16_gain = float(
            (per_episode_h16["no_revision"] - per_episode_h16["frozen_v6"])[native_accepted].mean()
        )
    else:
        native_validation_gain = 0.0
        native_h16_gain = 0.0
    native_trigger_count = int((native & triggered).sum())
    native_assimilation = (
        float((native & triggered & accepted).sum()) / native_trigger_count
        if native_trigger_count else 0.0
    )
    summary = {
        "seed": seed,
        "tangent_detection_auroc": _binary_auroc(tangent_score, structural),
        "magnitude_detection_auroc": _binary_auroc(magnitude_score, structural),
        "c0_trigger_rate": float(triggered[c0].float().mean()),
        "c0_false_revision": float(accepted[c0].float().mean()),
        "c1_trigger_recall": float(triggered[c1].float().mean()),
        "c1_candidate_coverage": float(proposal_coverage.float().mean()),
        "c1_selection_accuracy": float(selection_correct[c1].float().mean()),
        "c1_exact_recovery": float(exact_c1[c1].float().mean()),
        "c1_assimilation_ratio": float(exact_c1[c1].float().mean()),
        "c1_operator_coefficient": float(selected_mean[c1, -1].mean()),
        "c2_trigger_recall": float(triggered[c2].float().mean()),
        "c2_unknown_rejection": float((~accepted)[c2].float().mean()),
        "c2_forced_wrong_revision": float(accepted[c2].float().mean()),
        "c2_residual_fallback_usage": float((triggered & ~accepted)[c2].float().mean()),
        "native_trigger_rate": float(triggered[native].float().mean()),
        "native_revision_rate": float(accepted[native].float().mean()),
        "native_residual_usage_after": float((triggered & ~accepted)[native].float().mean()),
        "native_assimilation_ratio": native_assimilation,
        "native_validation_gain_accepted": native_validation_gain,
        "native_h16_gain_accepted": native_h16_gain,
        "native_h16_stability_accepted": float(
            frozen_v6_h16_stability[native_accepted].mean()
        ) if bool(native_accepted.any()) else 1.0,
        "native_operator_coefficient_abs_accepted": float(
            selected_mean[native_accepted, -1].abs().mean()
        ) if bool(native_accepted.any()) else 0.0,
        "native_parameter_correction_norm_accepted": float(
            selected_mean[native_accepted, :2].norm(dim=-1).mean()
        ) if bool(native_accepted.any()) else 0.0,
        "validation_samples": float(samples.mean()),
        "defer_rate": float(deferred.float().mean()),
    }
    native_count = int(native.sum())
    c2_count = int(c2.sum())
    for operator_index, operator_name in enumerate(OPERATOR_NAMES):
        summary[f"native_revision_{operator_name}"] = float(
            (native & accepted & (selected_operator == operator_index)).sum()
        ) / native_count
        summary[f"c2_wrong_revision_{operator_name}"] = float(
            (c2 & accepted & (selected_operator == operator_index)).sum()
        ) / c2_count
    return summary, horizon_rows


def _rollout_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["regime"]), int(row["horizon"]))
        grouped.setdefault(key, []).append(float(row["state_rmse"]))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def run_v6r03(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R03Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6R03Config()
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
    index = _rollout_index(horizon_rows)
    h16 = cfg.horizons[-1]
    c0_gap = max(
        index[("frozen_v6", "c0_parameter", horizon)]
        - index[("physics", "c0_parameter", horizon)]
        for horizon in cfg.horizons
    )
    c1_all_better = all(
        index[("frozen_v6", "c1_drag", horizon)]
        < index[("no_revision", "c1_drag", horizon)]
        for horizon in cfg.horizons
    )
    mean_metric = lambda name: statistics.mean(float(row[name]) for row in seed_rows)
    minimum_validity = min(float(row["valid_window_fraction"]) for row in horizon_rows)
    native_h16_gain = (
        index[("no_revision", "r0n_native", h16)]
        - index[("frozen_v6", "r0n_native", h16)]
    )
    gates = {
        "c0_noninferiority": c0_gap <= cfg.c0_noninferiority_epsilon,
        "c0_false_revision": mean_metric("c0_false_revision") < cfg.c0_false_revision_max,
        "c1_rollout_gain": c1_all_better,
        "c1_assimilation": mean_metric("c1_assimilation_ratio") > 0.0,
        "c2_unknown_rejection": mean_metric("c2_unknown_rejection") >= cfg.c2_unknown_rejection_min,
        "c2_wrong_revision": mean_metric("c2_forced_wrong_revision") <= cfg.c2_wrong_revision_max,
        "native_revision_utility": native_h16_gain > cfg.native_revision_gain_min,
        "native_assimilation": mean_metric("native_assimilation_ratio") > cfg.native_assimilation_min,
        "valid_support_coverage": minimum_validity >= cfg.valid_window_fraction_min,
    }
    summary = {
        "scope": "V6R0.3 formal held-out SAPIEN bridge",
        "held_out_seeds": list(cfg.seeds),
        "regimes": list(REGIMES),
        "residual_coordinate": "generalized_force",
        "mechanism_frozen_from_r02": True,
        "validity_mask": "no PhysX joint-limit activation within window",
        "c0_max_v6_minus_physics": c0_gap,
        "native_h16_no_revision_minus_v6": native_h16_gain,
        "minimum_valid_window_fraction": minimum_validity,
        "gates": gates,
        "overall_go": all(gates.values()),
        "c2_executed": True,
        "native_executed": True,
        "robotwin_assets_downloaded": False,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "horizons": list(cfg.horizons)},
    }
    _write_json(root / "summary.json", summary)
    return summary

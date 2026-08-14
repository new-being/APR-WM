from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _binary_auroc, _write_csv, _write_json


STRATEGIES = (
    "linear_refit",
    "magnitude_revision",
    "tangent_revision",
    "always_expand",
    "oracle_class",
)
CLASS_NAMES = ("adequate", "cubic", "unseen")


@dataclass
class V3Config:
    seeds: tuple[int, ...] = (101, 111, 121, 131, 141)
    episodes: int = 4_096
    passive_context: int = 8
    evidence_count: int = 16
    validation_count: int = 4
    queries: int = 16
    observation_noise: float = 0.05
    cubic_alpha: float = 800.0
    unseen_alpha: float = 95.0
    tangent_ratio_threshold: float = 0.02
    tangent_noise_threshold: float = 1.35
    magnitude_noise_threshold: float = 2.20
    revision_improvement_threshold: float = 0.0005
    extension_accept_rmse: float = 0.11
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    residual_cost: float = 0.01
    model_complexity_cost: float = 0.006


def structural_force(
    features: torch.Tensor,
    model_class: torch.Tensor,
    config: V3Config,
) -> torch.Tensor:
    """Structural terms: one discoverable and one absent from the proposal set."""
    x, v = features[..., 0], features[..., 1]
    while model_class.ndim < x.ndim:
        model_class = model_class.unsqueeze(-1)
    cubic = config.cubic_alpha * x.pow(3)
    # This velocity-coupled term is deliberately not in the cubic extension
    # family. It is smooth, changes sign with velocity, and cannot be repaired
    # by changing only k and c.
    unseen = config.unseen_alpha * x.square() * torch.tanh(v / 0.06)
    return torch.where(
        model_class == 1,
        cubic,
        torch.where(model_class == 2, unseen, torch.zeros_like(x)),
    )


def true_force(
    features: torch.Tensor,
    theta: torch.Tensor,
    model_class: torch.Tensor,
    config: V3Config,
) -> torch.Tensor:
    while theta.ndim < features.ndim:
        theta = theta.unsqueeze(-2)
    return (features * theta).sum(dim=-1) + structural_force(
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
    elif regime == "evidence":
        x = 0.014 + 0.072 * torch.rand(*shape, generator=generator, device=device)
        v = 0.40 * torch.rand(*shape, generator=generator, device=device) - 0.20
    elif regime == "intervention":
        x = 0.060 + 0.030 * torch.rand(*shape, generator=generator, device=device)
        v = 0.40 * torch.rand(*shape, generator=generator, device=device) - 0.20
    else:
        raise ValueError(regime)
    return torch.stack((x, v), dim=-1)


def _ridge_fit(
    design: torch.Tensor, target: torch.Tensor, ridge: float
) -> torch.Tensor:
    # Column normalization is essential when a proposed basis (x^3) is orders
    # of magnitude smaller than the native parameter Jacobian columns.
    scale = design.square().mean(dim=-2).sqrt().clamp_min(1.0e-8)
    normalized = design / scale.unsqueeze(-2)
    dimension = design.shape[-1]
    gram = torch.einsum("bni,bnj->bij", normalized, normalized)
    eye = torch.eye(dimension, dtype=design.dtype, device=design.device)
    natural = torch.einsum("bni,bn->bi", normalized, target)
    normalized_coefficient = torch.linalg.solve(
        gram + ridge * eye, natural.unsqueeze(-1)
    ).squeeze(-1)
    return normalized_coefficient / scale


def tangent_decomposition(
    jacobian: torch.Tensor, residual: torch.Tensor, ridge: float = 1.0e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a windowed residual onto col(J_theta) and its complement."""
    delta = _ridge_fit(jacobian, residual, ridge)
    parallel = torch.einsum("bni,bi->bn", jacobian, delta)
    orthogonal = residual - parallel
    score = orthogonal.square().sum(dim=-1) / residual.square().sum(dim=-1).clamp_min(
        1.0e-12
    )
    return parallel, orthogonal, score


def _cubic_design(features: torch.Tensor) -> torch.Tensor:
    return torch.cat((features, features[..., :1].pow(3)), dim=-1)


def _unseen_design(features: torch.Tensor) -> torch.Tensor:
    x, v = features[..., :1], features[..., 1:2]
    return torch.cat((features, x.square() * torch.tanh(v / 0.06)), dim=-1)


def _residual_basis(features: torch.Tensor) -> torch.Tensor:
    """Generic fixed RBF fallback; it does not contain an explicit true formula."""
    x_centers = torch.tensor((0.02, 0.04, 0.06, 0.08), device=features.device)
    v_centers = torch.tensor((-0.15, 0.0, 0.15), device=features.device)
    grid_x, grid_v = torch.meshgrid(x_centers, v_centers, indexing="ij")
    dx = (features[..., 0, None] - grid_x.flatten()) / 0.025
    dv = (features[..., 1, None] - grid_v.flatten()) / 0.11
    return torch.exp(-0.5 * (dx.square() + dv.square()))


def _residual_predict(
    train_x: torch.Tensor,
    train_residual: torch.Tensor,
    query_x: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    train_basis = _residual_basis(train_x)
    query_basis = _residual_basis(query_x)
    coefficient = _ridge_fit(train_basis, train_residual, ridge)
    return torch.einsum("bqi,bi->bq", query_basis, coefficient)


def _classify_revision(
    trigger: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    config: V3Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = _ridge_fit(train_x, train_y, config.ridge)
    extension = _ridge_fit(_cubic_design(train_x), train_y, config.ridge)
    base_prediction = torch.einsum("bni,bi->bn", validation_x, base)
    extension_prediction = torch.einsum(
        "bni,bi->bn", _cubic_design(validation_x), extension
    )
    base_mse = (validation_y - base_prediction).square().mean(dim=-1)
    extension_mse = (validation_y - extension_prediction).square().mean(dim=-1)
    improvement = base_mse - extension_mse
    accepted = (
        trigger
        & (improvement > config.revision_improvement_threshold)
        & (extension_mse.sqrt() < config.extension_accept_rmse)
    )
    unknown = trigger & ~accepted
    state = torch.zeros_like(trigger, dtype=torch.long)
    state[accepted] = 1
    state[unknown] = 2
    return state, improvement, extension_mse.sqrt()


def _earliest_trigger(
    evidence_x: torch.Tensor,
    evidence_residual: torch.Tensor,
    config: V3Config,
    method: str,
) -> torch.Tensor:
    result = torch.full(
        (evidence_x.shape[0],),
        config.evidence_count + 1,
        dtype=torch.long,
        device=evidence_x.device,
    )
    for count in range(4, config.evidence_count + 1):
        x = evidence_x[:, :count]
        residual = evidence_residual[:, :count]
        _, orthogonal, ratio = tangent_decomposition(x, residual, config.ridge)
        if method == "tangent":
            triggered = (ratio > config.tangent_ratio_threshold) & (
                orthogonal.square().mean(dim=-1).sqrt()
                > config.tangent_noise_threshold * config.observation_noise
            )
        else:
            triggered = residual.square().mean(dim=-1).sqrt() > (
                config.magnitude_noise_threshold * config.observation_noise
            )
        result[(result > config.evidence_count) & triggered] = count
    return result


def _calibration(probability: torch.Tensor, label: torch.Tensor) -> tuple[float, float]:
    brier = float((probability - label.float()).square().mean())
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (probability >= low) & (
            probability < high if index < 9 else probability <= high
        )
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(probability[mask].mean()) - float(label[mask].float().mean())
            )
    return brier, ece


def _evaluate_strategy(
    config: V3Config, seed: int, strategy: str, device: torch.device
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + 40_000)
    episodes = config.episodes
    theta = torch.stack(
        (
            24.0 + 22.0 * torch.rand(episodes, generator=generator, device=device),
            0.7 + torch.rand(episodes, generator=generator, device=device),
        ),
        dim=-1,
    )
    model_class = torch.randint(
        0, 3, (episodes,), generator=generator, device=device
    )
    passive_x = _sample_features(
        "passive", (episodes, config.passive_context), generator, device
    )
    evidence_x = _sample_features(
        "evidence", (episodes, config.evidence_count), generator, device
    )
    passive_y = true_force(passive_x, theta, model_class, config)
    evidence_y = true_force(evidence_x, theta, model_class, config)
    passive_y += config.observation_noise * torch.randn(
        passive_y.shape, generator=generator, device=device
    )
    evidence_y += config.observation_noise * torch.randn(
        evidence_y.shape, generator=generator, device=device
    )

    initial_theta = _ridge_fit(passive_x, passive_y, config.ridge)
    evidence_residual = evidence_y - torch.einsum(
        "bni,bi->bn", evidence_x, initial_theta
    )
    _, orthogonal, tangent_score = tangent_decomposition(
        evidence_x, evidence_residual, config.ridge
    )
    orthogonal_rms = orthogonal.square().mean(dim=-1).sqrt()
    residual_rms = evidence_residual.square().mean(dim=-1).sqrt()
    tangent_trigger = (tangent_score > config.tangent_ratio_threshold) & (
        orthogonal_rms > config.tangent_noise_threshold * config.observation_noise
    )
    magnitude_trigger = (
        residual_rms > config.magnitude_noise_threshold * config.observation_noise
    )

    validation = config.validation_count
    train_x = torch.cat((passive_x, evidence_x[:, :-validation]), dim=1)
    train_y = torch.cat((passive_y, evidence_y[:, :-validation]), dim=1)
    validation_x = evidence_x[:, -validation:]
    validation_y = evidence_y[:, -validation:]
    if strategy == "linear_refit":
        state = torch.zeros(episodes, dtype=torch.long, device=device)
    elif strategy == "always_expand":
        state, _, _ = _classify_revision(
            torch.ones(episodes, dtype=torch.bool, device=device),
            train_x,
            train_y,
            validation_x,
            validation_y,
            config,
        )
    elif strategy in ("magnitude_revision", "tangent_revision"):
        state, _, _ = _classify_revision(
            tangent_trigger if strategy == "tangent_revision" else magnitude_trigger,
            train_x,
            train_y,
            validation_x,
            validation_y,
            config,
        )
    elif strategy == "oracle_class":
        state = model_class.clone()
    else:
        raise ValueError(strategy)

    all_x = torch.cat((passive_x, evidence_x), dim=1)
    all_y = torch.cat((passive_y, evidence_y), dim=1)
    linear_theta = _ridge_fit(all_x, all_y, config.ridge)
    cubic_theta = _ridge_fit(_cubic_design(all_x), all_y, config.ridge)
    unseen_theta = _ridge_fit(_unseen_design(all_x), all_y, config.ridge)

    test_generator = torch.Generator(device=device).manual_seed(seed + 50_000)
    id_x = _sample_features("passive", (episodes, config.queries), test_generator, device)
    intervention_x = _sample_features(
        "intervention", (episodes, config.queries), test_generator, device
    )
    id_y = true_force(id_x, theta, model_class, config)
    intervention_y = true_force(intervention_x, theta, model_class, config)
    structural = model_class > 0
    cubic = model_class == 1
    unseen = model_class == 2
    adequate = model_class == 0

    def predict(query_x: torch.Tensor) -> torch.Tensor:
        linear_prediction = torch.einsum("bqi,bi->bq", query_x, linear_theta)
        cubic_prediction = torch.einsum(
            "bqi,bi->bq", _cubic_design(query_x), cubic_theta
        )
        fitted_residual = all_y - torch.einsum(
            "bni,bi->bn", all_x, linear_theta
        )
        residual_prediction = _residual_predict(
            all_x, fitted_residual, query_x, config.residual_ridge
        )
        prediction = torch.where(
            (state == 1).unsqueeze(-1), cubic_prediction, linear_prediction
        )
        prediction = torch.where(
            (state == 2).unsqueeze(-1),
            linear_prediction + residual_prediction,
            prediction,
        )
        if strategy == "oracle_class":
            # Oracle supplies the true family, including the held-out one, but
            # all coefficients (including theta) are still fit from noisy data.
            unseen_prediction = torch.einsum(
                "bqi,bi->bq", _unseen_design(query_x), unseen_theta
            )
            prediction = torch.where(
                adequate.unsqueeze(-1), linear_prediction, prediction
            )
            prediction = torch.where(cubic.unsqueeze(-1), cubic_prediction, prediction)
            prediction = torch.where(unseen.unsqueeze(-1), unseen_prediction, prediction)
        return prediction

    id_prediction = predict(id_x)
    intervention_prediction = predict(intervention_x)
    tangent_probability = (orthogonal_rms / (
        orthogonal_rms + config.tangent_noise_threshold * config.observation_noise
    )).clamp(0, 1)
    magnitude_probability = (residual_rms / (
        residual_rms + config.magnitude_noise_threshold * config.observation_noise
    )).clamp(0, 1)
    detection_score = (
        tangent_probability if strategy != "magnitude_revision" else magnitude_probability
    )
    brier, ece = _calibration(detection_score, structural)
    predicted_structural = state > 0
    correct_state = state == model_class
    earliest = _earliest_trigger(
        evidence_x,
        evidence_residual,
        config,
        "magnitude" if strategy == "magnitude_revision" else "tangent",
    )
    triggered_structural = structural & (earliest <= config.evidence_count)
    intervention_mse = (intervention_prediction - intervention_y).square().mean()
    if strategy == "oracle_class":
        # The oracle reveals and instantiates either structural family, so both
        # non-linear classes pay model-complexity rather than fallback-compute cost.
        complexity = (state > 0).float()
        residual_calls = torch.zeros_like(complexity)
    else:
        complexity = (state == 1).float()
        residual_calls = (state == 2).float()
    selected_theta = torch.where(
        (state == 1).unsqueeze(-1), cubic_theta[:, :2], linear_theta
    )
    if strategy == "oracle_class":
        selected_theta = torch.where(
            unseen.unsqueeze(-1), unseen_theta[:, :2], selected_theta
        )
    return {
        "strategy": strategy,
        "parameter_rmse_k": float(
            torch.sqrt((selected_theta[:, 0] - theta[:, 0]).square().mean())
        ),
        "parameter_rmse_c": float(
            torch.sqrt((selected_theta[:, 1] - theta[:, 1]).square().mean())
        ),
        "structural_detection_auroc": _binary_auroc(detection_score, structural),
        "structural_detection_brier": brier,
        "structural_detection_ece": ece,
        "false_revision_rate_adequate": float(predicted_structural[adequate].float().mean()),
        "missed_revision_rate_cubic": float((state[cubic] != 1).float().mean()),
        "unknown_recall_unseen": float((state[unseen] == 2).float().mean()),
        "model_state_accuracy": float(correct_state.float().mean()),
        "mean_evidence_to_trigger": float(earliest[triggered_structural].float().mean())
        if bool(triggered_structural.any())
        else float(config.evidence_count + 1),
        "id_force_rmse": float(torch.sqrt((id_prediction - id_y).square().mean())),
        "intervention_force_rmse": float(torch.sqrt(intervention_mse)),
        "intervention_rmse_adequate": float(
            torch.sqrt((intervention_prediction[adequate] - intervention_y[adequate]).square().mean())
        ),
        "intervention_rmse_cubic": float(
            torch.sqrt((intervention_prediction[cubic] - intervention_y[cubic]).square().mean())
        ),
        "intervention_rmse_unseen": float(
            torch.sqrt((intervention_prediction[unseen] - intervention_y[unseen]).square().mean())
        ),
        "model_revision_rate": float(complexity.mean()),
        "unknown_residual_rate": float(residual_calls.mean()),
        "model_complexity_cost": float(complexity.mean()) * config.model_complexity_cost,
        "residual_compute_cost": float(residual_calls.mean()) * config.residual_cost,
        "total_objective": float(intervention_mse)
        + float(complexity.mean()) * config.model_complexity_cost
        + float(residual_calls.mean()) * config.residual_cost,
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
        "structural_detection_auroc",
        "false_revision_rate_adequate",
        "missed_revision_rate_cubic",
        "unknown_recall_unseen",
        "model_state_accuracy",
        "intervention_force_rmse",
        "total_objective",
    )
    output: list[dict[str, Any]] = []
    for comparison, left, right in (
        ("tangent_minus_magnitude", "tangent_revision", "magnitude_revision"),
        ("tangent_minus_always_expand", "tangent_revision", "always_expand"),
        ("oracle_minus_tangent", "oracle_class", "tangent_revision"),
        ("tangent_minus_linear", "tangent_revision", "linear_refit"),
    ):
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


def run_v3(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V3Config | None = None,
) -> dict[str, Any]:
    cfg = config or V3Config()
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
        "initial_active_physics_families": ["linear"],
        "dormant_revision_proposals": ["cubic"],
        "includes_truth_outside_revision_proposals": True,
    }
    _write_json(root / "summary.json", summary)
    return summary

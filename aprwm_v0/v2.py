from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _binary_auroc, _write_csv, _write_json


STRATEGIES = (
    "linear_compensation",
    "passive",
    "parameter_ig",
    "router_no_probe",
    "joint_ig",
    "oracle_probe",
    "full_oracle",
)


@dataclass
class V2Config:
    seeds: tuple[int, ...] = (13, 23, 33, 43, 53)
    episodes: int = 4_096
    passive_context: int = 8
    probes: int = 2
    observation_noise: float = 0.08
    nonlinear_alpha: float = 800.0
    model_prior_bad: float = 0.5
    task_beta: float = 10.0
    probe_cost: float = 0.02
    residual_cost: float = 0.04
    residual_utility_threshold: float = 0.5
    model_route_confidence: float = 0.99


def force(
    features: torch.Tensor,
    theta: torch.Tensor,
    model_bad: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    while theta.ndim < features.ndim:
        theta = theta.unsqueeze(-2)
    linear = (features * theta).sum(dim=-1)
    nonlinear = alpha * features[..., 0].pow(3)
    while model_bad.ndim < nonlinear.ndim:
        model_bad = model_bad.unsqueeze(-1)
    return linear + model_bad * nonlinear


def sample_action(
    action: str,
    episodes: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if action == "passive":
        x = 0.018 + 0.006 * torch.rand(
            episodes, generator=generator, device=device
        )
        v = 0.04 * torch.rand(episodes, generator=generator, device=device) - 0.02
    elif action == "velocity_probe":
        x = 0.012 + 0.006 * torch.rand(
            episodes, generator=generator, device=device
        )
        sign = torch.where(
            torch.rand(episodes, generator=generator, device=device) < 0.5,
            -torch.ones(episodes, device=device),
            torch.ones(episodes, device=device),
        )
        v = sign * (
            0.70
            + 0.20
            * torch.rand(episodes, generator=generator, device=device)
        )
    elif action == "amplitude_probe":
        x = 0.065 + 0.015 * torch.rand(
            episodes, generator=generator, device=device
        )
        v = 0.03 * torch.rand(episodes, generator=generator, device=device) - 0.015
    else:
        raise ValueError(f"Unknown action {action!r}")
    return torch.stack((x, v), dim=-1)


class JointBelief:
    """Particle belief over model class and continuous spring parameters."""

    def __init__(self, config: V2Config, device: torch.device):
        k_grid = torch.linspace(18.0, 62.0, 45, device=device)
        c_grid = torch.linspace(0.4, 2.0, 17, device=device)
        models = torch.tensor((0.0, 1.0), device=device)
        model, k, c = torch.meshgrid(models, k_grid, c_grid, indexing="ij")
        self.theta = torch.stack((k.flatten(), c.flatten()), dim=-1)
        self.model_bad = model.flatten()
        self.config = config

    def prior(self, episodes: int) -> torch.Tensor:
        log_model = torch.where(
            self.model_bad.bool(),
            math.log(self.config.model_prior_bad),
            math.log(1.0 - self.config.model_prior_bad),
        )
        log_weights = log_model - math.log(self.theta.shape[0] // 2)
        return log_weights.unsqueeze(0).expand(episodes, -1).clone()

    def update(
        self,
        log_weights: torch.Tensor,
        features: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        prediction = (
            features[:, None, 0] * self.theta[None, :, 0]
            + features[:, None, 1] * self.theta[None, :, 1]
            + self.model_bad[None, :]
            * self.config.nonlinear_alpha
            * features[:, None, 0].pow(3)
        )
        log_likelihood = -0.5 * (
            (observation[:, None] - prediction) / self.config.observation_noise
        ).square()
        updated = log_weights + log_likelihood
        return updated - torch.logsumexp(updated, dim=-1, keepdim=True)

    def summary(self, log_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = log_weights.exp()
        p_bad = (weights * self.model_bad).sum(dim=-1)
        mean = weights @ self.theta
        centered = self.theta.unsqueeze(0) - mean.unsqueeze(1)
        variance = (weights.unsqueeze(-1) * centered.square()).sum(dim=1)
        # Exact conditioning (used by the full oracle) creates zero-weight
        # particles with log weight -inf. Their entropy contribution is the
        # limiting value 0, not the floating-point product 0 * -inf = NaN.
        entropy_terms = torch.where(
            weights > 0,
            weights * log_weights,
            torch.zeros_like(weights),
        )
        entropy = -entropy_terms.sum(dim=-1)
        return {"p_bad": p_bad, "theta_mean": mean, "theta_std": variance.sqrt(), "entropy": entropy}

    def model_entropy(self, log_weights: torch.Tensor) -> torch.Tensor:
        # 1 - 1e-8 rounds to 1 in float32, so use an epsilon that preserves
        # finite binary entropy at exact oracle probabilities.
        p_bad = self.summary(log_weights)["p_bad"].clamp(1e-6, 1 - 1e-6)
        return -(p_bad * p_bad.log() + (1 - p_bad) * (1 - p_bad).log())

    def parameter_entropy(self, log_weights: torch.Tensor) -> torch.Tensor:
        summary = self.summary(log_weights)
        return torch.log(summary["theta_std"].clamp_min(1e-6)).sum(dim=-1)

    def expected_information_gain(
        self,
        log_weights: torch.Tensor,
        candidates: torch.Tensor,
        *,
        target: str,
    ) -> torch.Tensor:
        # Moment-matched Gaussian predictive entropy. For joint IG, between-
        # hypothesis variance includes both theta and model-class disagreement.
        weights = log_weights.exp()
        predictions = (
            candidates[:, :, None, 0] * self.theta[None, None, :, 0]
            + candidates[:, :, None, 1] * self.theta[None, None, :, 1]
            + self.model_bad[None, None, :]
            * self.config.nonlinear_alpha
            * candidates[:, :, None, 0].pow(3)
        )
        if target == "joint":
            parameter = self.expected_information_gain(
                log_weights, candidates, target="parameter"
            )
            model = self.expected_information_gain(
                log_weights, candidates, target="model"
            )
            parameter_scale = parameter.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            model_scale = model.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            return parameter / parameter_scale + self.config.task_beta * model / model_scale
        if target == "model":
            model_probabilities = []
            model_means = []
            model_variances = []
            for model_value in (0.0, 1.0):
                mask = self.model_bad == model_value
                model_weights = weights[:, mask]
                probability = model_weights.sum(dim=-1, keepdim=True)
                normalized = model_weights / probability.clamp_min(1e-12)
                model_prediction = predictions[:, :, mask]
                mean = (normalized[:, None, :] * model_prediction).sum(dim=-1)
                variance = (
                    normalized[:, None, :]
                    * (model_prediction - mean.unsqueeze(-1)).square()
                ).sum(dim=-1)
                model_probabilities.append(probability)
                model_means.append(mean)
                model_variances.append(variance)
            p0, p1 = model_probabilities
            mean0, mean1 = model_means
            within = p0 * model_variances[0] + p1 * model_variances[1]
            between = p0 * p1 * (mean0 - mean1).square()
            return 0.5 * torch.log1p(
                between / (within + self.config.observation_noise**2)
            )
        if target == "predictive_joint":
            mean = (weights[:, None, :] * predictions).sum(dim=-1)
            variance = (
                weights[:, None, :] * (predictions - mean.unsqueeze(-1)).square()
            ).sum(dim=-1)
        elif target == "parameter":
            # Remove between-model disagreement: uncertainty is evaluated
            # within each model and averaged by p(M).
            variance = predictions.new_zeros(predictions.shape[:2])
            for model_value in (0.0, 1.0):
                mask = self.model_bad == model_value
                model_weights = weights[:, mask]
                probability = model_weights.sum(dim=-1, keepdim=True)
                normalized = model_weights / probability.clamp_min(1e-12)
                model_prediction = predictions[:, :, mask]
                model_mean = (
                    normalized[:, None, :] * model_prediction
                ).sum(dim=-1)
                model_variance = (
                    normalized[:, None, :]
                    * (model_prediction - model_mean.unsqueeze(-1)).square()
                ).sum(dim=-1)
                variance += probability * model_variance
        else:
            raise ValueError(target)
        noise_variance = self.config.observation_noise**2
        return 0.5 * torch.log1p(variance / noise_variance)


def _select_action(
    strategy: str,
    belief: JointBelief,
    log_weights: torch.Tensor,
    candidates: torch.Tensor,
    true_model_bad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy in ("linear_compensation", "passive", "router_no_probe"):
        chosen = torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
        ig = chosen.float() * 0.0
    elif strategy == "parameter_ig":
        values = belief.expected_information_gain(log_weights, candidates, target="parameter")
        chosen = values.argmax(dim=-1)
        ig = values.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    elif strategy == "joint_ig":
        values = belief.expected_information_gain(log_weights, candidates, target="joint")
        chosen = values.argmax(dim=-1)
        ig = values.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    elif strategy in ("oracle_probe", "full_oracle"):
        # The privileged diagnostic action maximally separates model classes;
        # it does not condition the action itself on the hidden class.
        chosen = torch.full_like(true_model_bad, 2, dtype=torch.long)
        values = belief.expected_information_gain(log_weights, candidates, target="joint")
        ig = values.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(strategy)
    return candidates[torch.arange(candidates.shape[0], device=candidates.device), chosen], ig


def _calibration(
    probability: torch.Tensor, label: torch.Tensor, bins: int = 10
) -> tuple[float, float]:
    brier = float((probability - label.float()).square().mean())
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probability >= low) & (
            probability < high if index < bins - 1 else probability <= high
        )
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(probability[mask].mean()) - float(label[mask].float().mean())
            )
    return brier, ece


def residual_decision(
    model_bad_probability: torch.Tensor,
    correction: torch.Tensor,
    compute_price: float,
    model_confidence: float,
) -> torch.Tensor:
    # If M is adequate, applying the structural correction creates the same
    # squared error that it removes when M is bad. The net expected benefit is
    # therefore (2p(M_bad)-1)*delta^2 before compute price.
    expected_net_utility = (
        2.0 * model_bad_probability - 1.0
    ) * correction.square()
    return (expected_net_utility > compute_price) & (
        model_bad_probability > model_confidence
    )


def _evaluate_strategy(
    config: V2Config,
    seed: int,
    strategy: str,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10_000)
    episodes = config.episodes
    theta_true = torch.stack(
        (
            24.0 + 22.0 * torch.rand(episodes, generator=generator, device=device),
            0.7 + 1.0 * torch.rand(episodes, generator=generator, device=device),
        ),
        dim=-1,
    )
    model_bad = (
        torch.rand(episodes, generator=generator, device=device) < 0.5
    ).float()
    belief = JointBelief(config, device)
    log_weights = belief.prior(episodes)
    if strategy == "linear_compensation":
        log_weights[:, belief.model_bad.bool()] = float("-inf")
        log_weights -= torch.logsumexp(log_weights, dim=-1, keepdim=True)
    for _ in range(config.passive_context):
        feature = sample_action("passive", episodes, generator, device)
        observation = force(feature, theta_true, model_bad, config.nonlinear_alpha)
        observation += config.observation_noise * torch.randn(
            episodes, generator=generator, device=device
        )
        log_weights = belief.update(log_weights, feature, observation)
    passive_summary = belief.summary(log_weights)
    passive_model_entropy = belief.model_entropy(log_weights)
    passive_parameter_entropy = belief.parameter_entropy(log_weights)
    passive_joint_entropy = passive_summary["entropy"]
    selected_ig: list[torch.Tensor] = []
    probe_counts = torch.zeros(episodes, device=device)
    velocity_probe_counts = torch.zeros(episodes, device=device)
    amplitude_probe_counts = torch.zeros(episodes, device=device)
    for _ in range(config.probes):
        passive = sample_action("passive", episodes, generator, device)
        velocity = sample_action("velocity_probe", episodes, generator, device)
        amplitude = sample_action("amplitude_probe", episodes, generator, device)
        candidates = torch.stack((passive, velocity, amplitude), dim=1)
        action, ig = _select_action(strategy, belief, log_weights, candidates, model_bad)
        is_probe = (action[:, 0] > 0.04) | (action[:, 1].abs() > 0.15)
        probe_counts += is_probe.float()
        velocity_probe_counts += (action[:, 1].abs() > 0.15).float()
        amplitude_probe_counts += (action[:, 0] > 0.04).float()
        observation = force(action, theta_true, model_bad, config.nonlinear_alpha)
        observation += config.observation_noise * torch.randn(
            episodes, generator=generator, device=device
        )
        log_weights = belief.update(log_weights, action, observation)
        selected_ig.append(ig)
    summary = belief.summary(log_weights)
    if strategy == "full_oracle":
        incompatible = belief.model_bad.unsqueeze(0) != model_bad.unsqueeze(1)
        log_weights = log_weights.masked_fill(incompatible, float("-inf"))
        log_weights -= torch.logsumexp(log_weights, dim=-1, keepdim=True)
        summary = belief.summary(log_weights)
    final_model_entropy = belief.model_entropy(log_weights)
    final_parameter_entropy = belief.parameter_entropy(log_weights)

    test_generator = torch.Generator(device=device)
    test_generator.manual_seed(seed + 30_000)
    id_features = sample_action("passive", episodes, test_generator, device)
    intervention_features = sample_action(
        "amplitude_probe", episodes, test_generator, device
    )
    id_true = force(id_features, theta_true, model_bad, config.nonlinear_alpha)
    intervention_true = force(
        intervention_features, theta_true, model_bad, config.nonlinear_alpha
    )
    id_physics = (id_features * summary["theta_mean"]).sum(dim=-1)
    intervention_physics = (
        intervention_features * summary["theta_mean"]
    ).sum(dim=-1)
    residual_id = config.nonlinear_alpha * id_features[:, 0].pow(3)
    residual_intervention = config.nonlinear_alpha * intervention_features[:, 0].pow(3)

    # r_t is a compute decision separate from the chosen environment action.
    # It is variable-budget and cost-aware: execute only if expected structural
    # loss reduction exceeds the residual compute price.
    residual_probability = summary["p_bad"]
    if strategy in ("linear_compensation", "passive"):
        id_route = torch.zeros(episodes, dtype=torch.bool, device=device)
        intervention_route = torch.zeros_like(id_route)
    else:
        compute_price = config.residual_cost * config.residual_utility_threshold
        id_route = residual_decision(
            residual_probability,
            residual_id,
            compute_price,
            config.model_route_confidence,
        )
        intervention_route = residual_decision(
            residual_probability,
            residual_intervention,
            compute_price,
            config.model_route_confidence,
        )
    id_prediction = id_physics + residual_id * id_route
    intervention_prediction = (
        intervention_physics + residual_intervention * intervention_route
    )
    adequate = ~model_bad.bool()
    posterior_miss = (
        (summary["theta_mean"] - theta_true).abs()
        > 1.64485 * summary["theta_std"]
    ).any(dim=-1)
    strict = adequate & posterior_miss
    brier, model_ece = _calibration(summary["p_bad"], model_bad.bool())
    passive_brier, _ = _calibration(passive_summary["p_bad"], model_bad.bool())
    mean_ig = torch.stack(selected_ig).mean(dim=0)
    has_probe = probe_counts > 0
    joint_entropy_reduction = passive_joint_entropy - summary["entropy"]
    model_entropy_reduction = passive_model_entropy - final_model_entropy
    parameter_entropy_reduction = passive_parameter_entropy - final_parameter_entropy
    structural_target = model_bad * residual_intervention
    structural_utility = structural_target.square()
    return {
        "strategy": strategy,
        "parameter_rmse_k": float(
            torch.sqrt((summary["theta_mean"][:, 0] - theta_true[:, 0]).square().mean())
        ),
        "parameter_rmse_c": float(
            torch.sqrt((summary["theta_mean"][:, 1] - theta_true[:, 1]).square().mean())
        ),
        "parameter_rmse_k_adequate": float(
            torch.sqrt(
                (summary["theta_mean"][adequate, 0] - theta_true[adequate, 0])
                .square()
                .mean()
            )
        ),
        "parameter_rmse_k_inadequate": float(
            torch.sqrt(
                (summary["theta_mean"][~adequate, 0] - theta_true[~adequate, 0])
                .square()
                .mean()
            )
        ),
        "parameter_bias_k_adequate": float(
            (summary["theta_mean"][adequate, 0] - theta_true[adequate, 0]).mean()
        ),
        "parameter_bias_k_inadequate": float(
            (summary["theta_mean"][~adequate, 0] - theta_true[~adequate, 0]).mean()
        ),
        "model_auroc": _binary_auroc(summary["p_bad"], model_bad.bool()),
        "model_brier": brier,
        "model_ece": model_ece,
        "passive_model_brier": passive_brier,
        "id_force_rmse": float(torch.sqrt(F_mse(id_prediction, id_true))),
        "intervention_force_rmse": float(
            torch.sqrt(F_mse(intervention_prediction, intervention_true))
        ),
        "intervention_physics_only_rmse": float(
            torch.sqrt(F_mse(intervention_physics, intervention_true))
        ),
        "compensation_brittleness": float(
            torch.sqrt(F_mse(intervention_physics, intervention_true))
            - torch.sqrt(F_mse(id_physics, id_true))
        ),
        "residual_misuse_rate_adequate": float(
            intervention_route[adequate].float().mean()
        ),
        "residual_misuse_rate_strict": float(
            intervention_route[strict].float().mean()
        )
        if bool(strict.any())
        else 0.0,
        "residual_call_rate": float(intervention_route.float().mean()),
        "structural_utility_capture": float(
            (structural_utility * intervention_route).sum()
            / structural_utility.sum().clamp_min(1e-12)
        ),
        "probe_count": float(probe_counts.mean()),
        "velocity_probe_count": float(velocity_probe_counts.mean()),
        "amplitude_probe_count": float(amplitude_probe_counts.mean()),
        "probe_cost": float(probe_counts.mean()) * config.probe_cost,
        "information_gain_per_probe": float(
            (mean_ig * (probe_counts > 0)).sum() / probe_counts.gt(0).sum().clamp_min(1)
        ),
        "joint_entropy_reduction_per_probe": float(
            (joint_entropy_reduction[has_probe] / probe_counts[has_probe]).mean()
        )
        if bool(has_probe.any())
        else 0.0,
        "model_entropy_reduction_per_probe": float(
            (model_entropy_reduction[has_probe] / probe_counts[has_probe]).mean()
        )
        if bool(has_probe.any())
        else 0.0,
        "parameter_entropy_reduction_per_probe": float(
            (parameter_entropy_reduction[has_probe] / probe_counts[has_probe]).mean()
        )
        if bool(has_probe.any())
        else 0.0,
        "total_decision_cost": float(probe_counts.mean()) * config.probe_cost
        + float(intervention_route.float().mean()) * config.residual_cost,
    }


def F_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean()


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        members = [row for row in rows if row["strategy"] == strategy]
        for metric in sorted(set(members[0]) - {"seed", "strategy"}):
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
    output: list[dict[str, Any]] = []
    metrics = (
        "parameter_rmse_k",
        "model_auroc",
        "model_brier",
        "intervention_force_rmse",
        "residual_misuse_rate_adequate",
        "probe_count",
        "total_decision_cost",
    )
    for comparison, left, right in (
        ("joint_ig_minus_parameter_ig", "joint_ig", "parameter_ig"),
        ("joint_ig_minus_router_no_probe", "joint_ig", "router_no_probe"),
        ("oracle_minus_joint_ig", "oracle_probe", "joint_ig"),
        ("full_oracle_minus_joint_ig", "full_oracle", "joint_ig"),
        ("joint_ig_minus_linear_compensation", "joint_ig", "linear_compensation"),
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


def run_v2(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V2Config | None = None,
) -> dict[str, Any]:
    cfg = config or V2Config()
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
    result = {
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds)},
        "rows": len(rows),
        "physical_action_and_compute_decision_are_separate": True,
    }
    _write_json(root / "summary.json", result)
    return result

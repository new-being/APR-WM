from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .models import MLP
from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _binary_auroc, _top_budget, _write_csv, _write_json


STRUCTURAL_ROUTER_VARIANTS = ("state_only", "posterior_mean", "posterior_full")
ROUTER_NAMES = (*STRUCTURAL_ROUTER_VARIANTS, "total_utility")


@dataclass
class V1Config:
    seeds: tuple[int, ...] = (13, 23, 33, 43, 53)
    expert_train_steps: int = 1_200
    router_train_steps: int = 1_000
    train_trajectories: int = 256
    eval_trajectories: int = 2_048
    queries_per_trajectory: int = 32
    max_context: int = 32
    residual_budget: float = 0.1
    structural_probability: float = 0.5
    observation_noise: float = 0.08
    learning_rate: float = 8.0e-4


class V1Data:
    def __init__(self, device: torch.device, seed: int, config: V1Config):
        self.device = device
        self.config = config
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def _interaction_features(self, shape: tuple[int, ...]) -> torch.Tensor:
        penetration = 0.08 * torch.rand(
            (*shape, 1), generator=self.generator, device=self.device
        )
        normal_speed = 0.4 * torch.rand(
            (*shape, 1), generator=self.generator, device=self.device
        ) - 0.2
        return torch.cat((penetration, -normal_speed), dim=-1)

    @staticmethod
    def structural_force(features: torch.Tensor) -> torch.Tensor:
        penetration = features[..., 0]
        closing = features[..., 1]
        excess = (penetration - 0.035).clamp_min(0.0)
        return 900.0 * excess.square() * (
            1.0 + 0.35 * torch.tanh(closing / 0.08)
        )

    def sample(self, trajectories: int) -> dict[str, torch.Tensor]:
        cfg = self.config
        theta = torch.stack(
            (
                20.0
                + 30.0
                * torch.rand(
                    trajectories, generator=self.generator, device=self.device
                ),
                0.6
                + 1.2
                * torch.rand(
                    trajectories, generator=self.generator, device=self.device
                ),
            ),
            dim=-1,
        )
        structural = (
            torch.rand(
                trajectories, generator=self.generator, device=self.device
            )
            < cfg.structural_probability
        )
        choices = torch.tensor((4, 8, 16, 32), device=self.device)
        context_count = choices[
            torch.randint(
                0,
                len(choices),
                (trajectories,),
                generator=self.generator,
                device=self.device,
            )
        ].clamp_max(cfg.max_context)
        context_mask = (
            torch.arange(cfg.max_context, device=self.device).unsqueeze(0)
            < context_count.unsqueeze(1)
        )
        context_x = self._interaction_features(
            (trajectories, cfg.max_context)
        )
        context_structural = self.structural_force(context_x) * structural.unsqueeze(-1)
        context_y = (
            (context_x * theta.unsqueeze(1)).sum(dim=-1)
            + context_structural
            + cfg.observation_noise
            * torch.randn(
                (trajectories, cfg.max_context),
                generator=self.generator,
                device=self.device,
            )
        )
        query_x = self._interaction_features(
            (trajectories, cfg.queries_per_trajectory)
        )
        query_structural = self.structural_force(query_x) * structural.unsqueeze(-1)
        query_true_force = (
            query_x * theta.unsqueeze(1)
        ).sum(dim=-1) + query_structural
        return {
            "theta": theta,
            "structural": structural,
            "context_count": context_count,
            "context_mask": context_mask,
            "context_x": context_x,
            "context_y": context_y,
            "query_x": query_x,
            "query_structural": query_structural,
            "query_true_force": query_true_force,
        }


def parameter_posterior(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    context_mask: torch.Tensor,
    observation_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = context_x.shape[0]
    dtype = context_x.dtype
    device = context_x.device
    prior_mean = torch.tensor((35.0, 1.2), dtype=dtype, device=device)
    prior_std = torch.tensor((15.0, 0.7), dtype=dtype, device=device)
    prior_precision = torch.diag(prior_std.square().reciprocal()).expand(
        batch, -1, -1
    )
    masked_x = context_x * context_mask.unsqueeze(-1)
    xtx = torch.einsum("bci,bcj->bij", masked_x, context_x)
    xty = torch.einsum(
        "bci,bc->bi", masked_x, context_y * context_mask
    )
    noise_variance = observation_noise**2
    precision = prior_precision + xtx / noise_variance
    covariance = torch.linalg.inv(precision)
    natural = (
        torch.einsum("bij,j->bi", prior_precision, prior_mean)
        + xty / noise_variance
    )
    mean = torch.einsum("bij,bj->bi", covariance, natural)
    residual = context_y - (context_x * mean.unsqueeze(1)).sum(dim=-1)
    residual_rms = torch.sqrt(
        (residual.square() * context_mask).sum(dim=-1)
        / context_mask.sum(dim=-1).clamp_min(1)
    )
    return mean, covariance, residual_rms


def posterior_features(
    batch: dict[str, torch.Tensor], observation_noise: float = 0.08
) -> dict[str, torch.Tensor]:
    mean, covariance, residual_rms = parameter_posterior(
        batch["context_x"],
        batch["context_y"],
        batch["context_mask"],
        observation_noise,
    )
    query = batch["query_x"]
    trajectories, queries, _ = query.shape
    std = torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt()
    base = torch.stack(
        (query[..., 0] / 0.08, query[..., 1] / 0.2), dim=-1
    )
    normalized_mean = (mean - mean.new_tensor((35.0, 1.2))) / mean.new_tensor(
        (15.0, 0.7)
    )
    normalized_std = std / std.new_tensor((15.0, 0.7))
    shared_mean = normalized_mean.unsqueeze(1).expand(-1, queries, -1)
    shared_std = normalized_std.unsqueeze(1).expand(-1, queries, -1)
    diagnostics = torch.stack(
        (
            residual_rms / 0.25,
            batch["context_count"].float() / batch["context_x"].shape[1],
        ),
        dim=-1,
    ).unsqueeze(1).expand(-1, queries, -1)
    return {
        "state_only": base.reshape(-1, 2),
        "posterior_mean": torch.cat((base, shared_mean), dim=-1).reshape(-1, 4),
        "posterior_full": torch.cat(
            (base, shared_mean, shared_std, diagnostics), dim=-1
        ).reshape(-1, 8),
        "posterior_mean_raw": mean,
        "posterior_covariance": covariance,
        "context_residual_rms": residual_rms,
    }


class V1Router(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        dimensions = {"state_only": 2, "posterior_mean": 4, "posterior_full": 8}
        self.variant = variant
        self.network = MLP(dimensions[variant], 32, 1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class StructuralExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = MLP(8, 96, 1, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def _train_models(
    config: V1Config, seed: int, device: torch.device
) -> tuple[StructuralExpert, dict[str, V1Router]]:
    data = V1Data(device, seed + 100, config)
    expert = StructuralExpert().to(device)
    expert_optimizer = torch.optim.AdamW(
        expert.parameters(), lr=config.learning_rate
    )
    # Identification-only / expert stage: the structural target is defined
    # against oracle parameters, preventing parameter error from becoming the
    # residual target.
    for _ in range(config.expert_train_steps):
        batch = data.sample(config.train_trajectories)
        features = posterior_features(batch, config.observation_noise)["posterior_full"]
        target = batch["query_structural"].reshape(-1)
        expert_optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(expert(features), target)
        loss.backward()
        expert_optimizer.step()
    expert.eval()
    routers = {
        name: V1Router("posterior_full" if name == "total_utility" else name).to(device)
        for name in ROUTER_NAMES
    }
    optimizers = {
        name: torch.optim.AdamW(router.parameters(), lr=config.learning_rate)
        for name, router in routers.items()
    }
    for _ in range(config.router_train_steps):
        batch = data.sample(config.train_trajectories)
        feature_sets = posterior_features(batch, config.observation_noise)
        with torch.no_grad():
            target = batch["query_structural"].reshape(-1)
            prediction = expert(feature_sets["posterior_full"])
            structural_utility = target.square() - (target - prediction).square()
            query_x = batch["query_x"]
            theta = batch["theta"]
            posterior_mean = feature_sets["posterior_mean_raw"]
            true_force = batch["query_true_force"]
            estimated_physics = (query_x * posterior_mean.unsqueeze(1)).sum(dim=-1)
            total_error = (true_force - estimated_physics).reshape(-1)
            total_utility = total_error.square() - (total_error - prediction).square()
            structural_rank_target = _top_budget(
                structural_utility, config.residual_budget
            ).float()
            total_rank_target = _top_budget(
                total_utility, config.residual_budget
            ).float()
        for name, router in routers.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            inputs = feature_sets[
                "posterior_full" if name == "total_utility" else name
            ]
            rank_target = (
                total_rank_target if name == "total_utility" else structural_rank_target
            )
            loss = F.binary_cross_entropy_with_logits(router(inputs), rank_target)
            loss.backward()
            optimizer.step()
    return expert, routers


def _posterior_metrics(
    theta: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    structural: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    std = torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt()
    difference = theta - mean
    inverse = torch.linalg.inv(covariance)
    mahalanobis = torch.einsum("bi,bij,bj->b", difference, inverse, difference)
    logdet = torch.logdet(covariance)
    nll = 0.5 * (mahalanobis + logdet + 2 * math.log(2 * math.pi))
    quantiles = ((0.50, 0.67449), (0.80, 1.28155), (0.90, 1.64485), (0.95, 1.95996))
    for label, mask in (
        ("adequate", ~structural),
        ("inadequate", structural),
        ("all", torch.ones_like(structural, dtype=torch.bool)),
    ):
        calibration_errors = []
        coverage: dict[str, float] = {}
        for nominal, z_value in quantiles:
            covered = (
                difference.abs() <= z_value * std
            ).float()[mask].mean(dim=0)
            observed = float(covered.mean())
            coverage[f"coverage_{int(nominal * 100)}"] = observed
            calibration_errors.append(abs(observed - nominal))
        rows.append(
            {
                "condition": label,
                "parameter_rmse_k": float(
                    torch.sqrt(difference[mask, 0].square().mean())
                ),
                "parameter_rmse_c": float(
                    torch.sqrt(difference[mask, 1].square().mean())
                ),
                "parameter_nll": float(nll[mask].mean()),
                "parameter_calibration_error": statistics.mean(calibration_errors),
                **coverage,
            }
        )
    return rows


@torch.no_grad()
def _evaluate_seed(
    config: V1Config,
    seed: int,
    expert: StructuralExpert,
    routers: dict[str, V1Router],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = V1Data(device, seed + 20_000, config)
    batch = data.sample(config.eval_trajectories)
    features = posterior_features(batch, config.observation_noise)
    theta = batch["theta"]
    mean = features["posterior_mean_raw"]
    covariance = features["posterior_covariance"]
    query_x = batch["query_x"]
    structural_force = batch["query_structural"]
    true_force = batch["query_true_force"]
    oracle_physics = (query_x * theta.unsqueeze(1)).sum(dim=-1)
    estimated_physics = (query_x * mean.unsqueeze(1)).sum(dim=-1)
    prediction = expert(features["posterior_full"]).reshape_as(structural_force)
    structural_utility = structural_force.square() - (
        structural_force - prediction
    ).square()
    structural_oracle_route = _top_budget(
        structural_utility.flatten(), config.residual_budget
    ).reshape_as(structural_force)
    structural_oracle_selected = float(
        structural_utility[structural_oracle_route].sum()
    )
    total_error = true_force - estimated_physics
    total_utility = total_error.square() - (total_error - prediction).square()
    total_oracle_route = _top_budget(
        total_utility.flatten(), config.residual_budget
    ).reshape_as(structural_force)
    total_oracle_selected = float(total_utility[total_oracle_route].sum())
    posterior_miss = (
        (theta - mean).abs()
        > 1.64485
        * torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt()
    ).any(dim=-1)
    strict_misuse_edges = (~batch["structural"] & posterior_miss).unsqueeze(-1).expand_as(
        structural_force
    )
    adequate_edges = (~batch["structural"]).unsqueeze(-1).expand_as(
        structural_force
    )
    routing_rows: list[dict[str, Any]] = []
    score_sets = {
        name: router(
            features["posterior_full" if name == "total_utility" else name]
        ).reshape_as(structural_force)
        for name, router in routers.items()
    }
    score_sets["structural_oracle"] = structural_utility
    score_sets["total_oracle"] = total_utility
    for name, scores in score_sets.items():
        route = _top_budget(scores.flatten(), config.residual_budget).reshape_as(scores)
        routed_force = estimated_physics + prediction * route
        oracle_parameter_routed_force = oracle_physics + prediction * route
        selected_structural = float(structural_utility[route].sum())
        selected_total = float(total_utility[route].sum())
        calls = int(route.sum())
        adequate_trajectory_mask = (~batch["structural"]).unsqueeze(-1).expand_as(
            true_force
        )
        inadequate_trajectory_mask = batch["structural"].unsqueeze(-1).expand_as(
            true_force
        )
        routing_rows.append(
            {
                "router": name,
                "structural_utility_auroc": _binary_auroc(
                    scores.flatten(), structural_oracle_route.flatten()
                ),
                "total_utility_auroc": _binary_auroc(
                    scores.flatten(), total_oracle_route.flatten()
                ),
                "structural_utility_capture": selected_structural
                / max(structural_oracle_selected, 1e-12),
                "total_utility_capture": selected_total
                / max(total_oracle_selected, 1e-12),
                "force_rmse": float(torch.sqrt(F.mse_loss(routed_force, true_force))),
                "force_rmse_adequate": float(
                    torch.sqrt(
                        F.mse_loss(
                            routed_force[adequate_trajectory_mask],
                            true_force[adequate_trajectory_mask],
                        )
                    )
                ),
                "force_rmse_inadequate": float(
                    torch.sqrt(
                        F.mse_loss(
                            routed_force[inadequate_trajectory_mask],
                            true_force[inadequate_trajectory_mask],
                        )
                    )
                ),
                "oracle_parameter_force_rmse": float(
                    torch.sqrt(F.mse_loss(oracle_parameter_routed_force, true_force))
                ),
                "structural_force_rmse": float(
                    torch.sqrt(F.mse_loss(prediction * route, structural_force))
                ),
                "residual_misuse_rate_adequate": float(route[adequate_edges].float().mean()),
                "residual_misuse_rate_posterior_miss": float(
                    route[strict_misuse_edges].float().mean()
                )
                if bool(strict_misuse_edges.any())
                else 0.0,
                "residual_misuse_share": float((route & strict_misuse_edges).sum())
                / max(calls, 1),
                "route_fraction": float(route.float().mean()),
            }
        )
    condition_rows: list[dict[str, Any]] = []
    for class_name, class_mask in (
        ("adequate", ~batch["structural"]),
        ("inadequate", batch["structural"]),
    ):
        expanded = class_mask.unsqueeze(-1).expand_as(true_force)
        for parameter_mode, physics in (
            ("oracle", oracle_physics),
            ("estimated", estimated_physics),
        ):
            error = physics - true_force
            condition_rows.append(
                {
                    "model_class": class_name,
                    "parameter_mode": parameter_mode,
                    "physics_force_rmse": float(
                        torch.sqrt(error[expanded].square().mean())
                    ),
                    "physics_force_mse": float(error[expanded].square().mean()),
                }
            )
    posterior_rows = _posterior_metrics(
        theta, mean, covariance, batch["structural"]
    )
    return routing_rows, condition_rows, posterior_rows


def _aggregate(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        metric_names = sorted(set(members[0]) - set(keys) - {"seed"})
        for metric in metric_names:
            values = [float(member[metric]) for member in members]
            low, high = _bootstrap_ci(values)
            output.append(
                {
                    **dict(zip(keys, group)),
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                }
            )
    return output


def _decomposition_rows(
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), row["model_class"], row["parameter_mode"]): row
        for row in conditions
    }
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in conditions}):
        adequate_oracle = float(indexed[(seed, "adequate", "oracle")]["physics_force_mse"])
        adequate_estimated = float(
            indexed[(seed, "adequate", "estimated")]["physics_force_mse"]
        )
        inadequate_oracle = float(
            indexed[(seed, "inadequate", "oracle")]["physics_force_mse"]
        )
        inadequate_estimated = float(
            indexed[(seed, "inadequate", "estimated")]["physics_force_mse"]
        )
        output.append(
            {
                "seed": seed,
                "reference_floor": adequate_oracle,
                "parameter_error": adequate_estimated - adequate_oracle,
                "structural_error": inadequate_oracle - adequate_oracle,
                "combined_error": inadequate_estimated - adequate_oracle,
                "interaction_term": inadequate_estimated
                - inadequate_oracle
                - adequate_estimated
                + adequate_oracle,
            }
        )
    return output


def _routing_paired_rows(routing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), row["router"]): row for row in routing
    }
    output: list[dict[str, Any]] = []
    metrics = (
        "structural_utility_capture",
        "total_utility_capture",
        "force_rmse",
        "oracle_parameter_force_rmse",
        "residual_misuse_rate_adequate",
        "residual_misuse_rate_posterior_miss",
    )
    seeds = sorted({int(row["seed"]) for row in routing})
    for comparison, left, right in (
        ("posterior_full_minus_state_only", "posterior_full", "state_only"),
        ("total_utility_minus_posterior_full", "total_utility", "posterior_full"),
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


def run_v1(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V1Config | None = None,
) -> dict[str, Any]:
    cfg = config or V1Config()
    device = resolve_device(device_name)
    root = Path(output_dir)
    routing: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    posterior: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        expert, routers = _train_models(cfg, seed, device)
        for model in (expert, *routers.values()):
            model.eval()
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        torch.save(expert.state_dict(), seed_root / "structural_expert.pt")
        for name, router in routers.items():
            torch.save(router.state_dict(), seed_root / f"router_{name}.pt")
        seed_routing, seed_conditions, seed_posterior = _evaluate_seed(
            cfg, seed, expert, routers, device
        )
        for row in seed_routing:
            row["seed"] = seed
        for row in seed_conditions:
            row["seed"] = seed
        for row in seed_posterior:
            row["seed"] = seed
        _write_csv(seed_root / "routing.csv", seed_routing)
        _write_csv(seed_root / "conditions.csv", seed_conditions)
        _write_csv(seed_root / "posterior.csv", seed_posterior)
        routing.extend(seed_routing)
        conditions.extend(seed_conditions)
        posterior.extend(seed_posterior)
    _write_csv(root / "routing.csv", routing)
    _write_csv(root / "conditions.csv", conditions)
    _write_csv(root / "posterior.csv", posterior)
    _write_csv(root / "routing_aggregate.csv", _aggregate(routing, ("router",)))
    _write_csv(
        root / "conditions_aggregate.csv",
        _aggregate(conditions, ("model_class", "parameter_mode")),
    )
    _write_csv(
        root / "posterior_aggregate.csv",
        _aggregate(posterior, ("condition",)),
    )
    decomposition = _decomposition_rows(conditions)
    _write_csv(root / "decomposition.csv", decomposition)
    _write_csv(
        root / "decomposition_aggregate.csv", _aggregate(decomposition, ())
    )
    paired = _routing_paired_rows(routing)
    _write_csv(root / "routing_paired_differences.csv", paired)
    result = {
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds)},
        "routing_rows": len(routing),
        "condition_rows": len(conditions),
        "posterior_rows": len(posterior),
    }
    _write_json(root / "summary.json", result)
    return result

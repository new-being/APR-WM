from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import (
    Expert,
    RegimeData,
    Router,
    _binary_auroc,
    _make_execution,
    _top_budget,
    _write_csv,
    _write_json,
)


ROUTING_OBJECTIVES = ("magnitude", "necessity", "utility")
EVALUATION_STRATEGIES = ("random", *ROUTING_OBJECTIVES, "oracle")
REPRESENTATIVE_REGIMES = (
    ("slower", 16, "8x"),
    ("crossover", 128, "8x"),
    ("mild_speedup", 512, "1x"),
    ("clear_speedup", 512, "8x"),
)


@dataclass
class V07Config:
    seeds: tuple[int, ...] = (13, 23, 33, 43, 53)
    interaction_density: float = 0.8
    residual_budget: float = 0.1
    expert_train_steps: int = 1_200
    router_train_steps: int = 1_000
    train_edge_batch: int = 8_192
    eval_edge_batch: int = 131_072
    learning_rate: float = 8.0e-4
    latency_iterations: int = 100
    latency_block_size: int = 10
    latency_warmup: int = 30
    n_objects: int = 12


def task_weights(features: torch.Tensor) -> torch.Tensor:
    """Observable anisotropic downstream sensitivity for the two force axes."""

    x_weight = 0.15 + 2.85 * torch.sigmoid(4.0 * features[:, 4])
    y_weight = 0.15 + 2.85 * torch.sigmoid(-4.0 * features[:, 5])
    return torch.stack((x_weight, y_weight), dim=-1)


def residual_utility(
    target: torch.Tensor, prediction: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    before = (weights * target.square()).sum(dim=-1)
    after = (weights * (target - prediction).square()).sum(dim=-1)
    return before - after


def cost_aware_scores(
    utility: torch.Tensor, cost: float, *, mode: str, lambda_cost: float = 1.0
) -> torch.Tensor:
    if mode == "ratio":
        return utility / cost
    if mode == "difference":
        return utility - lambda_cost * cost
    raise ValueError(f"Unknown cost-aware utility mode {mode!r}")


def _target_values(
    objective: str,
    needed: torch.Tensor,
    prediction: torch.Tensor,
    utility: torch.Tensor,
) -> torch.Tensor:
    if objective == "magnitude":
        return torch.linalg.vector_norm(prediction, dim=-1)
    if objective == "necessity":
        return needed.float()
    if objective == "utility":
        return utility
    raise ValueError(f"Unknown routing objective {objective!r}")


def _train_experts(
    config: V07Config, seed: int, device: torch.device
) -> dict[str, Expert]:
    data = RegimeData(device, seed + 100)
    experts = {name: Expert(name).to(device) for name in ("1x", "8x")}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        for name, model in experts.items()
    }
    necessities = (0.05, 0.1, 0.2, 0.4)
    for step in range(config.expert_train_steps):
        necessity = necessities[step % len(necessities)]
        features, _, _, target = data.sample(
            config.train_edge_batch, 1.0, necessity
        )
        for name, expert in experts.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(expert(features), target)
            loss.backward()
            optimizer.step()
    return experts


@torch.no_grad()
def _normalization_stats(
    config: V07Config,
    seed: int,
    expert: Expert,
    device: torch.device,
) -> dict[str, tuple[float, float]]:
    data = RegimeData(device, seed + 2_000)
    features, _, needed, target = data.sample(
        config.eval_edge_batch, 1.0, config.residual_budget
    )
    prediction = expert(features)
    utility = residual_utility(target, prediction, task_weights(features))
    stats: dict[str, tuple[float, float]] = {}
    for objective in ("magnitude", "utility"):
        values = _target_values(objective, needed, prediction, utility)
        stats[objective] = (float(values.mean()), float(values.std().clamp_min(1e-6)))
    return stats


def _train_routers(
    config: V07Config,
    seed: int,
    expert: Expert,
    device: torch.device,
) -> tuple[dict[str, Router], dict[str, tuple[float, float]]]:
    data = RegimeData(device, seed + 3_000)
    routers = {name: Router("current").to(device) for name in ROUTING_OBJECTIVES}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        for name, model in routers.items()
    }
    stats = _normalization_stats(config, seed, expert, device)
    expert.eval()
    for _ in range(config.router_train_steps):
        features, _, needed, target = data.sample(
            config.train_edge_batch, 1.0, config.residual_budget
        )
        with torch.no_grad():
            prediction = expert(features)
            utility = residual_utility(target, prediction, task_weights(features))
        for objective, router in routers.items():
            optimizer = optimizers[objective]
            optimizer.zero_grad(set_to_none=True)
            scores = router(features)
            values = _target_values(objective, needed, prediction, utility)
            if objective == "necessity":
                rank_target = values
            else:
                # The deployment decision is exact-budget top-k allocation.
                # Train every objective with the same ranking-classification
                # loss so only the supervision signal changes.
                rank_target = _top_budget(values, config.residual_budget).float()
            loss = F.binary_cross_entropy_with_logits(scores, rank_target)
            loss.backward()
            optimizer.step()
    return routers, stats


@torch.no_grad()
def _evaluate_seed(
    config: V07Config,
    seed: int,
    experts: dict[str, Expert],
    router_sets: dict[str, dict[str, Router]],
    device: torch.device,
) -> list[dict[str, Any]]:
    data = RegimeData(device, seed + 20_000)
    features, active, needed, target = data.sample(
        config.eval_edge_batch,
        config.interaction_density,
        config.residual_budget,
    )
    features = features[active]
    needed = needed[active]
    target = target[active]
    weights = task_weights(features)
    route_count = max(1, round(config.residual_budget * features.shape[0]))
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(seed + 30_000)
    rows: list[dict[str, Any]] = []
    for expert_name, expert in experts.items():
        prediction = expert(features)
        utility = residual_utility(target, prediction, weights)
        oracle_route = _top_budget(utility, config.residual_budget)
        oracle_selected = float(utility[oracle_route].sum())
        dense_mse = float(F.mse_loss(prediction, target))
        dense_task_loss = float(
            (weights * (target - prediction).square()).sum(dim=-1).mean()
        )
        scores_by_strategy: dict[str, torch.Tensor] = {
            "random": torch.rand(
                features.shape[0], generator=random_generator, device=device
            ),
            "magnitude": router_sets[expert_name]["magnitude"](features),
            "necessity": router_sets[expert_name]["necessity"](features),
            "utility": router_sets[expert_name]["utility"](features),
            "oracle": utility,
        }
        for strategy, scores in scores_by_strategy.items():
            route = _top_budget(scores, config.residual_budget)
            sparse_prediction = torch.zeros_like(target)
            sparse_prediction[route] = prediction[route]
            task_loss = float(
                (weights * (target - sparse_prediction).square())
                .sum(dim=-1)
                .mean()
            )
            selected_utility = float(utility[route].sum())
            overlap = float((route & oracle_route).sum()) / route_count
            rows.append(
                {
                    "seed": seed,
                    "expert": expert_name,
                    "strategy": strategy,
                    "router_auroc_utility_top_budget": _binary_auroc(
                        scores, oracle_route
                    ),
                    "router_auroc_necessity": _binary_auroc(scores, needed),
                    "oracle_top_budget_recall": overlap,
                    "selected_utility_per_active_edge": selected_utility
                    / features.shape[0],
                    "oracle_utility_per_active_edge": oracle_selected
                    / features.shape[0],
                    "utility_capture": selected_utility
                    / max(oracle_selected, 1e-12),
                    "dense_rmse": math.sqrt(dense_mse),
                    "adaptive_rmse": math.sqrt(
                        float(F.mse_loss(sparse_prediction, target))
                    ),
                    "dense_task_loss": dense_task_loss,
                    "adaptive_task_loss": task_loss,
                    "delta_task_loss": task_loss - dense_task_loss,
                    "route_fraction_of_active": float(route.float().mean()),
                }
            )
    return rows


def _time_paired(
    baseline: Any,
    adaptive: Any,
    config: V07Config,
    device: torch.device,
) -> dict[str, float]:
    # Alternate order on every sample so clock boost, thermal drift, and
    # allocator/kernel caches affect both paths symmetrically.
    for iteration in range(config.latency_warmup):
        if iteration % 2:
            adaptive()
            baseline()
        else:
            baseline()
            adaptive()
    torch.cuda.synchronize(device)
    contact_samples: list[float] = []
    adaptive_samples: list[float] = []

    def record(forward: Any) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(config.latency_block_size):
            forward()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / config.latency_block_size

    for iteration in range(config.latency_iterations):
        if iteration % 2:
            adaptive_ms = record(adaptive)
            contact_ms = record(baseline)
        else:
            contact_ms = record(baseline)
            adaptive_ms = record(adaptive)
        contact_samples.append(contact_ms)
        adaptive_samples.append(adaptive_ms)
    ratios = [
        contact / adaptive
        for contact, adaptive in zip(contact_samples, adaptive_samples)
    ]

    torch.cuda.reset_peak_memory_stats(device)
    baseline()
    torch.cuda.synchronize(device)
    contact_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    torch.cuda.reset_peak_memory_stats(device)
    adaptive()
    torch.cuda.synchronize(device)
    adaptive_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "contact_median": statistics.median(contact_samples),
        "contact_mean": statistics.mean(contact_samples),
        "contact_std": statistics.stdev(contact_samples),
        "adaptive_median": statistics.median(adaptive_samples),
        "adaptive_mean": statistics.mean(adaptive_samples),
        "adaptive_std": statistics.stdev(adaptive_samples),
        "speedup": statistics.median(ratios),
        "contact_memory": contact_memory,
        "adaptive_memory": adaptive_memory,
    }


@torch.no_grad()
def _latency_seed(
    config: V07Config,
    seed: int,
    experts: dict[str, Expert],
    router_sets: dict[str, dict[str, Router]],
    device: torch.device,
) -> list[dict[str, Any]]:
    if device.type != "cuda":
        raise RuntimeError("V0.7 latency validation requires CUDA")
    data = RegimeData(device, seed + 40_000)
    rows: list[dict[str, Any]] = []
    edges_per_world = config.n_objects * (config.n_objects - 1)
    for regime, batch_size, expert_name in REPRESENTATIVE_REGIMES:
        features, active, _, _ = data.sample(
            batch_size * edges_per_world,
            config.interaction_density,
            config.residual_budget,
        )
        expert = experts[expert_name]
        baseline = _make_execution(
            expert,
            router_sets[expert_name]["utility"],
            features,
            active,
            "contact_packed",
            config.residual_budget,
        )
        for objective in ROUTING_OBJECTIVES:
            adaptive = _make_execution(
                expert,
                router_sets[expert_name][objective],
                features,
                active,
                "hard_sparse",
                config.residual_budget,
            )
            timing = _time_paired(baseline, adaptive, config, device)
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "batch_size": batch_size,
                    "expert": expert_name,
                    "objective": objective,
                    "contact_latency_ms": timing["contact_median"],
                    "adaptive_latency_ms": timing["adaptive_median"],
                    "speedup_vs_contact": timing["speedup"],
                    "contact_mean_latency_ms": timing["contact_mean"],
                    "adaptive_mean_latency_ms": timing["adaptive_mean"],
                    "contact_std_latency_ms": timing["contact_std"],
                    "adaptive_std_latency_ms": timing["adaptive_std"],
                    "contact_peak_allocated_mb": timing["contact_memory"],
                    "adaptive_peak_allocated_mb": timing["adaptive_memory"],
                }
            )
    return rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items()):
        base = dict(zip(keys, group))
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            low, high = _bootstrap_ci(values)
            output.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                }
            )
    return output


def _paired_router_differences(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), row["expert"], row["strategy"]): row for row in rows
    }
    output: list[dict[str, Any]] = []
    metrics = (
        "router_auroc_utility_top_budget",
        "utility_capture",
        "adaptive_rmse",
        "delta_task_loss",
    )
    for expert in ("1x", "8x"):
        for baseline in ("magnitude", "necessity"):
            for metric in metrics:
                differences = [
                    float(indexed[(seed, expert, "utility")][metric])
                    - float(indexed[(seed, expert, baseline)][metric])
                    for seed in sorted({int(row["seed"]) for row in rows})
                ]
                low, high = _bootstrap_ci(differences)
                output.append(
                    {
                        "expert": expert,
                        "comparison": f"utility_minus_{baseline}",
                        "metric": metric,
                        "n": len(differences),
                        "mean": statistics.mean(differences),
                        "std": statistics.stdev(differences)
                        if len(differences) > 1
                        else 0.0,
                        "bootstrap_95_low": low,
                        "bootstrap_95_high": high,
                    }
                )
    return output


def run_v07(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    config: V07Config | None = None,
    reuse_models: bool = False,
) -> dict[str, Any]:
    cfg = config or V07Config()
    device = resolve_device(device_name)
    root = Path(output_dir)
    evaluation_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        seed_root = root / f"seed_{seed}"
        if reuse_models:
            experts = {name: Expert(name).to(device) for name in ("1x", "8x")}
            router_sets = {
                expert_name: {
                    objective: Router("current").to(device)
                    for objective in ROUTING_OBJECTIVES
                }
                for expert_name in experts
            }
            for expert_name, expert in experts.items():
                expert.load_state_dict(
                    torch.load(
                        seed_root / f"expert_{expert_name}.pt",
                        map_location=device,
                        weights_only=True,
                    )
                )
                for objective, router in router_sets[expert_name].items():
                    router.load_state_dict(
                        torch.load(
                            seed_root / f"router_{expert_name}_{objective}.pt",
                            map_location=device,
                            weights_only=True,
                        )
                    )
            normalization = json.loads(
                (seed_root / "normalization.json").read_text(encoding="utf-8")
            )
        else:
            experts = _train_experts(cfg, seed, device)
            router_sets: dict[str, dict[str, Router]] = {}
            normalization: dict[str, Any] = {}
            for expert_name, expert in experts.items():
                routers, stats = _train_routers(cfg, seed, expert, device)
                router_sets[expert_name] = routers
                normalization[expert_name] = stats
        for module in (
            *experts.values(),
            *(router for routers in router_sets.values() for router in routers.values()),
        ):
            module.eval()
        seed_root.mkdir(parents=True, exist_ok=True)
        for expert_name, expert in experts.items():
            torch.save(expert.state_dict(), seed_root / f"expert_{expert_name}.pt")
            for objective, router in router_sets[expert_name].items():
                torch.save(
                    router.state_dict(),
                    seed_root / f"router_{expert_name}_{objective}.pt",
                )
        _write_json(seed_root / "normalization.json", normalization)
        seed_evaluation = _evaluate_seed(
            cfg, seed, experts, router_sets, device
        )
        seed_latency = _latency_seed(cfg, seed, experts, router_sets, device)
        _write_csv(seed_root / "evaluation.csv", seed_evaluation)
        _write_csv(seed_root / "latency.csv", seed_latency)
        evaluation_rows.extend(seed_evaluation)
        latency_rows.extend(seed_latency)
    _write_csv(root / "evaluation.csv", evaluation_rows)
    _write_json(root / "evaluation.json", evaluation_rows)
    _write_csv(root / "latency.csv", latency_rows)
    _write_json(root / "latency.json", latency_rows)
    evaluation_aggregate = _aggregate_rows(
        evaluation_rows,
        ("expert", "strategy"),
        (
            "router_auroc_utility_top_budget",
            "router_auroc_necessity",
            "oracle_top_budget_recall",
            "selected_utility_per_active_edge",
            "utility_capture",
            "adaptive_rmse",
            "delta_task_loss",
        ),
    )
    latency_aggregate = _aggregate_rows(
        latency_rows,
        ("regime", "batch_size", "expert", "objective"),
        ("speedup_vs_contact",),
    )
    _write_csv(root / "evaluation_aggregate.csv", evaluation_aggregate)
    _write_csv(root / "latency_aggregate.csv", latency_aggregate)
    paired = _paired_router_differences(evaluation_rows)
    _write_csv(root / "paired_differences.csv", paired)
    _write_json(root / "paired_differences.json", paired)
    result = {
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds)},
        "evaluation_rows": len(evaluation_rows),
        "latency_rows": len(latency_rows),
        "reused_models": reuse_models,
        "cost_aware_ranking_equivalent_at_fixed_cost": True,
    }
    _write_json(root / "summary.json", result)
    return result

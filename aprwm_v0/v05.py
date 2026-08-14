from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, load_config
from .graph import FRICTION, MASS, MATERIAL, POS, RADIUS, VEL, interaction_features, scatter_edge_messages
from .models import AdaptiveHybrid, DynamicsModel, integrate_state, make_edge_inputs
from .simulator import SyntheticWorld
from .train import load_checkpoint, resolve_device, run_suite, train_one


STRATEGIES = ("random", "heuristic", "learned", "oracle")
DEFAULT_BUDGETS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0)
DEFAULT_HORIZONS = (1, 5, 10, 20, 50)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _global_budget_route(
    scores: torch.Tensor, contact: torch.Tensor, budget: float
) -> torch.Tensor:
    mask = contact.bool().flatten()
    route = torch.zeros_like(mask)
    available = int(mask.sum())
    count = int(round(budget * available))
    if budget > 0 and available > 0:
        count = max(1, count)
    count = min(count, available)
    if count:
        candidates = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        chosen = torch.topk(scores.flatten()[candidates], count, sorted=False).indices
        route[candidates[chosen]] = True
    return route.reshape_as(contact)


def _router_inputs(
    model: AdaptiveHybrid, state: torch.Tensor, action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    physics = model.world.physics_acceleration(state, action)
    inputs, receiver, _, contact, _ = make_edge_inputs(
        state, action, physics.edge_acceleration
    )
    return inputs, receiver, contact, physics.acceleration


def _strategy_scores(
    model: AdaptiveHybrid,
    state: torch.Tensor,
    action: torch.Tensor,
    strategy: str,
    *,
    random_generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    inputs, receiver, contact, physics_acceleration = _router_inputs(model, state, action)
    all_residual: torch.Tensor | None = None
    if strategy == "random":
        scores = torch.rand(contact.shape, generator=random_generator, device=state.device)
    elif strategy == "learned":
        logits = model.router(inputs).squeeze(-1) / model.model_config.gate_temperature
        scores = torch.sigmoid(logits)
    elif strategy == "heuristic":
        # No material/category input: rank by nonlinear-contact risk estimated
        # from penetration, closing speed, tangential slip, and friction.
        rel_pos = inputs[..., 16:18]
        rel_vel = inputs[..., 18:20]
        distance = inputs[..., 20:21].clamp_min(1.0e-6)
        penetration = inputs[..., 21]
        friction = inputs[..., 25]
        normal = rel_pos / distance
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        closing = (-(rel_vel * normal).sum(dim=-1)).clamp_min(0.0)
        slip = (rel_vel * tangent).sum(dim=-1).abs()
        scores = penetration.square() * (1.0 + closing) + friction * penetration * slip
    elif strategy == "oracle":
        all_residual = model.residual_expert(inputs) * contact.unsqueeze(-1)
        physics_edges = model.world.physics_acceleration(state, action).edge_acceleration
        true_edges = model.world.true_acceleration(state, action).edge_acceleration
        before = (true_edges - physics_edges).square().sum(dim=-1)
        after = (true_edges - physics_edges - all_residual).square().sum(dim=-1)
        scores = before - after
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    return scores, inputs, receiver, contact, physics_acceleration, all_residual


def routed_step(
    model: AdaptiveHybrid,
    state: torch.Tensor,
    action: torch.Tensor,
    strategy: str,
    budget: float,
    *,
    random_generator: torch.Generator,
) -> tuple[torch.Tensor, int, int, int]:
    scores, inputs, receiver, contact, physics_acceleration, all_residual = _strategy_scores(
        model, state, action, strategy, random_generator=random_generator
    )
    route = _global_budget_route(scores, contact, budget)
    if all_residual is None:
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_route = route.reshape(-1)
        flat_residual = inputs.new_zeros((flat_inputs.shape[0], 2))
        if bool(flat_route.any()):
            values = model.residual_expert(flat_inputs[flat_route])
            flat_residual[flat_route] = values.to(flat_residual.dtype)
        residual = flat_residual.reshape(*inputs.shape[:-1], 2)
    else:
        residual = all_residual * route.unsqueeze(-1)
    acceleration = physics_acceleration + scatter_edge_messages(
        residual, receiver, state.shape[1]
    )
    next_state = integrate_state(state, acceleration, model.world_config.dt)
    return next_state, int(route.sum()), int(contact.sum()), contact.numel()


@torch.no_grad()
def pareto_experiment(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    batches: int = 4,
    batch_size: int = 128,
    horizon: int = 20,
    device_name: str = "auto",
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    model, config = load_checkpoint(checkpoint, device)
    if not isinstance(model, AdaptiveHybrid):
        raise TypeError("Pareto evaluation requires an adaptive checkpoint")
    model.eval()
    world = SyntheticWorld(config.world)
    profile = model.compute_profile()
    rows: list[dict[str, Any]] = []
    for ood in (False, True):
        split = "ood" if ood else "id"
        nodes = config.eval.ood_n_objects if ood else config.eval.id_n_objects
        for strategy in STRATEGIES:
            for budget in budgets:
                data_generator = torch.Generator(device=device)
                data_generator.manual_seed(config.seed + (20_000 if ood else 10_000))
                route_generator = torch.Generator(device=device)
                route_generator.manual_seed(config.seed + 40_000)
                teacher_route_generator = torch.Generator(device=device)
                teacher_route_generator.manual_seed(config.seed + 45_000)
                squared_error = 0.0
                teacher_squared_error = 0.0
                endpoint_squared_error = 0.0
                value_count = 0
                endpoint_count = 0
                active = 0
                contacts = 0
                graph_edges = 0
                for _ in range(batches):
                    truth, actions = world.sample_trajectories(
                        batch_size,
                        nodes,
                        horizon,
                        generator=data_generator,
                        device=device,
                        ood=ood,
                    )
                    current = truth[:, 0]
                    for step in range(horizon):
                        teacher_prediction, _, _, _ = routed_step(
                            model,
                            truth[:, step],
                            actions[:, step],
                            strategy,
                            budget,
                            random_generator=teacher_route_generator,
                        )
                        teacher_error = (
                            teacher_prediction[..., 0:4]
                            - truth[:, step + 1, ..., 0:4]
                        )
                        teacher_squared_error += float(teacher_error.square().sum())
                        current, used, contact_count, edge_count = routed_step(
                            model,
                            current,
                            actions[:, step],
                            strategy,
                            budget,
                            random_generator=route_generator,
                        )
                        error = current[..., 0:4] - truth[:, step + 1, ..., 0:4]
                        squared_error += float(error.square().sum())
                        value_count += error.numel()
                        active += used
                        contacts += contact_count
                        graph_edges += edge_count
                    endpoint = current[..., 0:4] - truth[:, -1, ..., 0:4]
                    endpoint_squared_error += float(endpoint.square().sum())
                    endpoint_count += endpoint.numel()
                active_graph = active / max(graph_edges, 1)
                router_flops = (
                    profile["router_flops_per_edge"] if strategy == "learned" else 0
                )
                expert_flops = profile["expert_flops_per_edge"]
                analytical = (
                    router_flops + active_graph * expert_flops
                ) / (profile["router_flops_per_edge"] + expert_flops)
                rows.append(
                    {
                        "split": split,
                        "strategy": strategy,
                        "contact_budget": budget,
                        "rollout_rmse": math.sqrt(squared_error / value_count),
                        "teacher_forced_one_step_rmse": math.sqrt(
                            teacher_squared_error / value_count
                        ),
                        "endpoint_rmse": math.sqrt(
                            endpoint_squared_error / endpoint_count
                        ),
                        "activated_contact_fraction": active / max(contacts, 1),
                        "activated_graph_fraction": active_graph,
                        "analytical_compute_fraction": analytical,
                        "selection_cost_included": strategy == "learned",
                    }
                )
    output = Path(output_dir)
    _write_csv(output / "pareto.csv", rows)
    _write_json(output / "pareto.json", rows)
    return rows


@torch.no_grad()
def horizon_experiment(
    checkpoints: list[str | Path],
    output_dir: str | Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    batch_size: int = 128,
    device_name: str = "auto",
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    max_horizon = max(horizons)
    rows: list[dict[str, Any]] = []
    for ood in (False, True):
        for model, config in loaded:
            model.eval()
            world = SyntheticWorld(config.world)
            nodes = config.eval.ood_n_objects if ood else config.eval.id_n_objects
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed + (60_000 if ood else 50_000))
            truth, actions = world.sample_trajectories(
                batch_size,
                nodes,
                max_horizon,
                generator=generator,
                device=device,
                ood=ood,
            )
            current = truth[:, 0]
            horizon_set = set(horizons)
            for step in range(1, max_horizon + 1):
                output = model(
                    current,
                    actions[:, step - 1],
                    hard_routing=model.is_adaptive,
                )
                current = output["next_state"]
                if step in horizon_set:
                    error = current[..., 0:4] - truth[:, step, ..., 0:4]
                    rows.append(
                        {
                            "split": "ood" if ood else "id",
                            "model": model.model_name,
                            "horizon": step,
                            "endpoint_rmse": float(error.square().mean().sqrt()),
                        }
                    )
    output = Path(output_dir)
    _write_csv(output / "horizon.csv", rows)
    _write_json(output / "horizon.json", rows)
    return rows


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    return float(
        (left_rank * right_rank).sum()
        / (
            torch.linalg.vector_norm(left_rank)
            * torch.linalg.vector_norm(right_rank)
        ).clamp_min(1.0e-8)
    )


@torch.no_grad()
def continuum_experiment(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    grid_size: int = 25,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, config = load_checkpoint(checkpoint, device)
    if not isinstance(model, AdaptiveHybrid):
        raise TypeError("Continuum evaluation requires an adaptive checkpoint")
    model.eval()
    world = SyntheticWorld(config.world)
    penetrations = torch.linspace(0.001, 0.08, grid_size, device=device)
    impact_speeds = torch.linspace(0.0, 0.8, grid_size, device=device)
    penetration_grid, speed_grid = torch.meshgrid(
        penetrations, impact_speeds, indexing="ij"
    )
    batch = grid_size * grid_size
    state = torch.zeros((batch, 2, 8), device=device)
    radius = 0.12
    state[:, :, RADIUS] = radius
    state[:, :, MASS] = 1.0
    state[:, :, FRICTION] = 0.5
    # Fix material identity throughout: only interaction regime changes.
    state[:, :, MATERIAL] = 1.0
    penetration_flat = penetration_grid.flatten()
    speed_flat = speed_grid.flatten()
    state[:, 0, 0] = -0.5 * (2 * radius - penetration_flat)
    state[:, 1, 0] = 0.5 * (2 * radius - penetration_flat)
    state[:, 0, 2] = 0.5 * speed_flat
    state[:, 1, 2] = -0.5 * speed_flat
    action = torch.zeros((batch, 2, 2), device=device)
    inputs, _, contact, _ = _router_inputs(model, state, action)
    gates = torch.sigmoid(
        model.router(inputs).squeeze(-1) / model.model_config.gate_temperature
    )
    physics = world.physics_acceleration(state, action)
    truth = world.true_acceleration(state, action)
    physics_error = torch.linalg.vector_norm(
        truth.edge_acceleration - physics.edge_acceleration, dim=-1
    )
    # One direction per pair is sufficient and avoids duplicate rows.
    gates = gates[:, 0].float()
    physics_error = physics_error[:, 0].float()
    rows = [
        {
            "penetration": float(penetration_flat[index]),
            "impact_speed": float(speed_flat[index]),
            "physics_edge_error": float(physics_error[index]),
            "gate_probability": float(gates[index]),
            "material": 1.0,
        }
        for index in range(batch)
    ]
    summary = {
        "spearman_gate_vs_physics_error": _rank_correlation(gates, physics_error),
        "material_fixed": 1.0,
        "samples": batch,
    }
    output = Path(output_dir)
    _write_csv(output / "continuum.csv", rows)
    _write_json(output / "continuum_summary.json", summary)
    return summary


@torch.no_grad()
def latency_experiment(
    checkpoints: list[str | Path],
    output_dir: str | Path,
    *,
    batch_sizes: tuple[int, ...] = (1, 128),
    warmup: int = 50,
    iterations: int = 200,
    block_size: int = 20,
    device_name: str = "auto",
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Latency experiment requires CUDA")
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        model, config = load_checkpoint(checkpoint, device)
        model.eval()
        world = SyntheticWorld(config.world)
        for batch_size in batch_sizes:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed + batch_size)
            state = world.sample_initial_state(
                batch_size,
                config.eval.id_n_objects,
                generator=generator,
                device=device,
            )
            action = world.sample_actions(
                batch_size,
                config.eval.id_n_objects,
                1,
                generator=generator,
                device=device,
            )[:, 0]
            modes = (
                ("hard", "soft", "contact_all", "heuristic_40")
                if model.is_adaptive
                else ("default",)
            )
            for mode in modes:
                hard = mode == "hard"
                route_generator = torch.Generator(device=device)
                route_generator.manual_seed(config.seed + batch_size + 90_000)

                def forward_once():
                    if mode in ("contact_all", "heuristic_40"):
                        budget = 1.0 if mode == "contact_all" else 0.4
                        return routed_step(
                            model,
                            state,
                            action,
                            "heuristic",
                            budget,
                            random_generator=route_generator,
                        )
                    return model(state, action, hard_routing=hard)

                for _ in range(warmup):
                    forward_once()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                samples: list[float] = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(block_size):
                        output = forward_once()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / block_size)
                if mode in ("contact_all", "heuristic_40"):
                    _, active_count, _, graph_edges = output
                    semantic_used = float(active_count)
                else:
                    semantic_used = float(output["gate_used"].sum())
                    graph_edges = output["gate_used"].numel()
                if model.model_name == "physics":
                    executed_fraction = 0.0
                elif model.is_adaptive and mode in (
                    "hard",
                    "contact_all",
                    "heuristic_40",
                ):
                    executed_fraction = semantic_used / graph_edges
                else:
                    # Dense neural/residual and adaptive-soft all execute the
                    # expert over every graph edge before applying a mask.
                    executed_fraction = 1.0
                rows.append(
                    {
                        "model": model.model_name,
                        "mode": mode,
                        "batch_size": batch_size,
                        "mean_latency_ms": statistics.mean(samples),
                        "median_latency_ms": statistics.median(samples),
                        "std_latency_ms": statistics.stdev(samples),
                        "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                        / (1024**2),
                        "semantic_gate_fraction": semantic_used / graph_edges,
                        "expert_executed_graph_fraction": executed_fraction,
                        "iterations": iterations,
                        "block_size": block_size,
                    }
                )
    output = Path(output_dir)
    _write_csv(output / "latency.csv", rows)
    _write_json(output / "latency.json", rows)
    return rows


def _bootstrap_ci(
    values: list[float], *, seed: int = 123, samples: int = 10_000
) -> tuple[float, float]:
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choices(values, k=len(values))) for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def aggregate_seeds(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    metric_files = sorted(root.glob("seed_*/suite_metrics.json"))
    if len(metric_files) < 2:
        raise ValueError(f"Need at least two completed seeds under {root}")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    paired: dict[tuple[int, str], dict[str, float]] = {}
    for path in metric_files:
        seed = int(path.parent.name.removeprefix("seed_"))
        with path.open("r", encoding="utf-8") as handle:
            suite = json.load(handle)
        for result in suite:
            model = result["model"]
            for split in ("id", "ood"):
                metrics = result[split]
                for metric in (
                    "hard_rollout_state_rmse",
                    "routing_auroc",
                    "learned_compute_fraction",
                ):
                    grouped.setdefault((model, split, metric), []).append(
                        float(metrics[metric])
                    )
                paired.setdefault((seed, split), {})[model] = float(
                    metrics["hard_rollout_state_rmse"]
                )
    rows: list[dict[str, Any]] = []
    for (model, split, metric), values in sorted(grouped.items()):
        low, high = _bootstrap_ci(values)
        rows.append(
            {
                "model": model,
                "split": split,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    paired_rows: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        differences = [
            models["adaptive"] - models["residual"]
            for (seed, row_split), models in paired.items()
            if row_split == split and "adaptive" in models and "residual" in models
        ]
        low, high = _bootstrap_ci(differences)
        paired_rows.append(
            {
                "split": split,
                "comparison": "adaptive_minus_residual_hard_rollout_rmse",
                "n": len(differences),
                "mean": statistics.mean(differences),
                "std": statistics.stdev(differences),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    result = {"aggregate": rows, "paired": paired_rows}
    _write_csv(root / "aggregate.csv", rows)
    _write_csv(root / "paired_differences.csv", paired_rows)
    _write_json(root / "aggregate.json", result)
    return result


def seed_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
) -> dict[str, Any]:
    base = load_config(config_path)
    root = Path(output_dir)
    for seed in seeds:
        config = copy.deepcopy(base)
        config.seed = seed
        config.output_dir = str(root / f"seed_{seed}")
        run_suite(config)
    return aggregate_seeds(root)


def ablation_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 7,
) -> list[dict[str, Any]]:
    base = load_config(config_path)
    variants: dict[str, dict[str, Any]] = {
        "full": {},
        "no_warmup": {"warmup": 1},
        "no_distillation": {"lambda_router": 0.0},
        "soft_gate": {"training_routing": "soft"},
        "budget_40": {"budget": 0.4},
        "budget_80": {"budget": 0.8},
    }
    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for name, overrides in variants.items():
        config = copy.deepcopy(base)
        config.seed = seed
        config.output_dir = str(root / name)
        if "warmup" in overrides:
            config.train.regularizer_warmup_steps = int(overrides["warmup"])
        if "lambda_router" in overrides:
            config.train.lambda_router = float(overrides["lambda_router"])
        if "training_routing" in overrides:
            config.model.training_routing = str(overrides["training_routing"])
        if "budget" in overrides:
            config.model.residual_budget_fraction = float(overrides["budget"])
        result = train_one("adaptive", config, output_root=Path(config.output_dir))
        for split in ("id", "ood"):
            metrics = result[split]
            assert isinstance(metrics, dict)
            rows.append(
                {
                    "variant": name,
                    "split": split,
                    "hard_rollout_rmse": metrics["hard_rollout_state_rmse"],
                    "soft_rollout_rmse": metrics["rollout_state_rmse"],
                    "routing_auroc": metrics["routing_auroc"],
                    "active_graph_fraction": metrics["active_graph_edge_fraction"],
                    "analytical_compute_fraction": metrics["learned_compute_fraction"],
                }
            )
    _write_csv(root / "ablation.csv", rows)
    _write_json(root / "ablation.json", rows)
    return rows

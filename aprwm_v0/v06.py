from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .models import MLP
from .train import resolve_device, seed_everything


FEATURE_DIM = 12
EXPERT_SCALES: dict[str, tuple[int, int]] = {
    # Depth is fixed so the labels denote actual compute multipliers rather
    # than loosely ordered architecture sizes. Approximate FLOPs/edge are
    # 21,120 / 43,120 / 85,600 / 169,264 (1.00x / 2.04x / 4.05x / 8.01x).
    "1x": (96, 3),
    "2x": (140, 3),
    "4x": (200, 3),
    "8x": (284, 3),
}
ROUTER_NAMES = ("linear", "tiny", "current")
EXECUTION_MODES = ("dense_all", "contact_packed", "masked_dense", "hard_sparse")
DEFAULT_DENSITIES = (0.2, 0.5, 0.8, 1.0)
DEFAULT_NECESSITIES = (0.05, 0.1, 0.2, 0.4)
DEFAULT_BATCH_SIZES = (1, 16, 128, 512)


@dataclass
class V06Config:
    seed: int = 13
    n_objects: int = 12
    train_steps: int = 1_500
    train_edge_batch: int = 8_192
    learning_rate: float = 8.0e-4
    eval_edge_batch: int = 131_072
    latency_iterations: int = 30
    latency_block_size: int = 10
    latency_warmup: int = 30


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


class RegimeData:
    """Edge-level dense-coupling benchmark with state-dependent nonlinearity.

    There are no object or edge identity features. The same interaction can be
    simple or nonlinear according to its current extension/velocity regime.
    Stage-0 active interaction existence is sampled independently of residual
    necessity, which prevents contact density from encoding complexity.
    """

    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def sample(
        self,
        edge_count: int,
        interaction_density: float,
        necessity: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.rand(
            (edge_count, FEATURE_DIM - 1),
            generator=self.generator,
            device=self.device,
        )
        raw = 2.0 * raw - 1.0
        extension = raw[:, 0]
        relative_speed = raw[:, 1]
        friction = 0.5 * (raw[:, 2] + 1.0)
        phase = raw[:, 3]
        score = (
            1.15 * extension.abs()
            + 0.75 * relative_speed.abs()
            + 0.25 * friction
            + 0.15 * extension * relative_speed
            + 0.08 * torch.sin(torch.pi * phase)
        )
        # Each requested necessity defines a continuous regime boundary. This
        # scalar is observable physical context, not an edge/category ID.
        threshold = torch.quantile(score, 1.0 - necessity)
        features = torch.cat(
            (raw, threshold.expand(edge_count, 1)), dim=-1
        )
        active = (
            torch.rand(
                edge_count, generator=self.generator, device=self.device
            )
            < interaction_density
        )
        needed = (score > threshold) & active
        excess = (score - threshold).clamp_min(0.0)
        amplitude = 0.12 + 0.9 * excess + 0.7 * excess.square()
        residual = torch.stack(
            (
                -torch.sign(extension) * amplitude,
                -torch.tanh(2.5 * relative_speed) * amplitude * (0.4 + friction),
            ),
            dim=-1,
        ) * needed.unsqueeze(-1)
        return features, active, needed, residual


class Router(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "linear":
            self.network = nn.Linear(FEATURE_DIM, 1)
            self.flops_per_edge = 2 * FEATURE_DIM
        elif name == "tiny":
            self.network = nn.Sequential(
                nn.Linear(FEATURE_DIM, 16), nn.SiLU(), nn.Linear(16, 1)
            )
            self.flops_per_edge = 2 * FEATURE_DIM * 16 + 2 * 16
        elif name == "current":
            self.network = MLP(FEATURE_DIM, 32, 1, 2)
            self.flops_per_edge = self.network.approximate_flops()
        else:
            raise ValueError(f"Unknown router {name!r}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class Expert(nn.Module):
    def __init__(self, scale: str):
        super().__init__()
        if scale not in EXPERT_SCALES:
            raise ValueError(f"Unknown expert scale {scale!r}")
        width, depth = EXPERT_SCALES[scale]
        self.scale = scale
        self.network = MLP(FEATURE_DIM, width, 2, depth)
        self.flops_per_edge = self.network.approximate_flops()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = torch.argsort(scores, descending=True)
    ranked = labels[order]
    true_positive = torch.cat(
        (torch.zeros(1, device=scores.device), torch.cumsum(ranked.float(), 0) / positives)
    )
    false_positive = torch.cat(
        (
            torch.zeros(1, device=scores.device),
            torch.cumsum((~ranked).float(), 0) / negatives,
        )
    )
    return float(torch.trapezoid(true_positive, false_positive))


def _top_budget(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    count = int(round(fraction * scores.numel()))
    if fraction > 0 and scores.numel() > 0:
        count = max(1, count)
    count = min(count, scores.numel())
    route = torch.zeros_like(scores, dtype=torch.bool)
    if count:
        route[torch.topk(scores, count, sorted=False).indices] = True
    return route


def train_models(
    config: V06Config, output_dir: str | Path, device: torch.device
) -> tuple[dict[str, Expert], dict[str, Router], list[dict[str, Any]]]:
    root = Path(output_dir)
    data = RegimeData(device, config.seed + 100)
    experts = {name: Expert(name).to(device) for name in EXPERT_SCALES}
    routers = {name: Router(name).to(device) for name in ROUTER_NAMES}
    modules: dict[str, nn.Module] = {**experts, **routers}
    optimizers = {
        name: torch.optim.AdamW(module.parameters(), lr=config.learning_rate)
        for name, module in modules.items()
    }
    necessities = list(DEFAULT_NECESSITIES)
    history: list[dict[str, Any]] = []
    for step in range(1, config.train_steps + 1):
        necessity = necessities[(step - 1) % len(necessities)]
        features, active, needed, target = data.sample(
            config.train_edge_batch, 1.0, necessity
        )
        active_features = features[active]
        active_target = target[active]
        active_needed = needed[active].float()
        for name, expert in experts.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction = expert(active_features)
            loss = F.mse_loss(prediction, active_target)
            loss.backward()
            optimizer.step()
        for name, router in routers.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits = router(active_features)
            loss = F.binary_cross_entropy_with_logits(logits, active_needed)
            loss.backward()
            optimizer.step()
        if step == 1 or step % 250 == 0 or step == config.train_steps:
            with torch.no_grad():
                row: dict[str, Any] = {"step": step, "necessity": necessity}
                for name, expert in experts.items():
                    row[f"expert_{name}_mse"] = float(
                        F.mse_loss(expert(active_features), active_target)
                    )
                for name, router in routers.items():
                    row[f"router_{name}_auroc"] = _binary_auroc(
                        router(active_features), active_needed.bool()
                    )
                history.append(row)
    root.mkdir(parents=True, exist_ok=True)
    for name, module in modules.items():
        torch.save(module.state_dict(), root / f"{name}.pt")
    _write_json(root / "training_history.json", history)
    return experts, routers, history


def load_models(
    output_dir: str | Path, device: torch.device
) -> tuple[dict[str, Expert], dict[str, Router], list[dict[str, Any]]]:
    root = Path(output_dir)
    experts = {name: Expert(name).to(device) for name in EXPERT_SCALES}
    routers = {name: Router(name).to(device) for name in ROUTER_NAMES}
    for name, module in {**experts, **routers}.items():
        module.load_state_dict(
            torch.load(root / f"{name}.pt", map_location=device, weights_only=True)
        )
    history_path = root / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return experts, routers, history


@torch.no_grad()
def accuracy_sweep(
    config: V06Config,
    experts: dict[str, Expert],
    routers: dict[str, Router],
    output_dir: str | Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    data = RegimeData(device, config.seed + 20_000)
    rows: list[dict[str, Any]] = []
    for density in DEFAULT_DENSITIES:
        for necessity in DEFAULT_NECESSITIES:
            features, active, needed, target = data.sample(
                config.eval_edge_batch, density, necessity
            )
            active_features = features[active]
            active_target = target[active]
            active_needed = needed[active]
            for expert_name, expert in experts.items():
                dense_prediction = expert(active_features)
                dense_mse = float(F.mse_loss(dense_prediction, active_target))
                dense_rmse = math.sqrt(dense_mse)
                for router_name, router in routers.items():
                    logits = router(active_features)
                    route = _top_budget(logits, necessity)
                    sparse_prediction = torch.zeros_like(active_target)
                    if bool(route.any()):
                        sparse_prediction[route] = expert(active_features[route])
                    sparse_mse = float(F.mse_loss(sparse_prediction, active_target))
                    sparse_rmse = math.sqrt(sparse_mse)
                    true_positive = float((route & active_needed).sum())
                    rows.append(
                        {
                            "interaction_density": density,
                            "residual_necessity": necessity,
                            "expert": expert_name,
                            "expert_flops_per_edge": expert.flops_per_edge,
                            "router": router_name,
                            "router_flops_per_edge": router.flops_per_edge,
                            "dense_mse": dense_mse,
                            "dense_rmse": dense_rmse,
                            "adaptive_mse": sparse_mse,
                            "adaptive_rmse": sparse_rmse,
                            "delta_error": sparse_mse - dense_mse,
                            "delta_rmse": sparse_rmse - dense_rmse,
                            "relative_mse_change": (sparse_mse - dense_mse)
                            / max(dense_mse, 1.0e-12),
                            "router_auroc": _binary_auroc(logits, active_needed),
                            "route_fraction_of_active": float(route.float().mean()),
                            "necessity_recall": true_positive
                            / max(float(active_needed.sum()), 1.0),
                            "active_edge_samples": int(active.sum()),
                        }
                    )
    _write_csv(Path(output_dir) / "accuracy_matrix.csv", rows)
    _write_json(Path(output_dir) / "accuracy_matrix.json", rows)
    return rows


def _make_execution(
    expert: Expert,
    router: Router,
    features: torch.Tensor,
    active: torch.Tensor,
    mode: str,
    route_budget: float,
) -> Callable[[], torch.Tensor]:
    if mode == "dense_all":
        return lambda: expert(features)
    # Stage-0 packing is common to both baselines. Keeping it outside the
    # timed closure isolates the incremental learned-routing decision fairly.
    active_features = features[active]
    if mode == "contact_packed":
        return lambda: expert(active_features)
    if mode == "masked_dense":
        def masked_dense() -> torch.Tensor:
            route = _top_budget(router(active_features), route_budget)
            return expert(active_features) * route.unsqueeze(-1)
        return masked_dense
    if mode == "hard_sparse":
        def hard_sparse() -> torch.Tensor:
            route = _top_budget(router(active_features), route_budget)
            return expert(active_features[route])
        return hard_sparse
    raise ValueError(f"Unknown execution mode {mode!r}")


@torch.no_grad()
def latency_sweep(
    config: V06Config,
    experts: dict[str, Expert],
    routers: dict[str, Router],
    output_dir: str | Path,
    device: torch.device,
    *,
    full_matrix: bool,
) -> list[dict[str, Any]]:
    if device.type != "cuda":
        raise RuntimeError("V0.6 latency sweep requires CUDA")
    data = RegimeData(device, config.seed + 30_000)
    densities = DEFAULT_DENSITIES if full_matrix else (0.8,)
    necessities = DEFAULT_NECESSITIES if full_matrix else (0.1,)
    rows: list[dict[str, Any]] = []
    edges_per_world = config.n_objects * (config.n_objects - 1)
    for batch_size in DEFAULT_BATCH_SIZES:
        total_edges = batch_size * edges_per_world
        for density in densities:
            for necessity in necessities:
                features, active, _, _ = data.sample(total_edges, density, necessity)
                for expert_name, expert in experts.items():
                    for router_name, router in routers.items():
                        logits = router(features[active])
                        route_fraction = float(
                            _top_budget(logits, necessity).float().mean()
                        )
                        for mode in EXECUTION_MODES:
                            forward = _make_execution(
                                expert, router, features, active, mode, necessity
                            )
                            for _ in range(config.latency_warmup):
                                forward()
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            samples: list[float] = []
                            for _ in range(config.latency_iterations):
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(config.latency_block_size):
                                    output = forward()
                                end.record()
                                end.synchronize()
                                samples.append(
                                    start.elapsed_time(end) / config.latency_block_size
                                )
                            if mode == "dense_all":
                                expert_fraction = 1.0
                                router_fraction = 0.0
                            elif mode == "contact_packed":
                                expert_fraction = float(active.float().mean())
                                router_fraction = 0.0
                            elif mode == "masked_dense":
                                expert_fraction = float(active.float().mean())
                                router_fraction = float(active.float().mean())
                            else:
                                expert_fraction = (
                                    float(active.float().mean()) * route_fraction
                                )
                                router_fraction = float(active.float().mean())
                            analytical_flops = total_edges * (
                                expert_fraction * expert.flops_per_edge
                                + router_fraction * router.flops_per_edge
                            )
                            rows.append(
                                {
                                    "batch_size": batch_size,
                                    "interaction_density": density,
                                    "residual_necessity": necessity,
                                    "expert": expert_name,
                                    "expert_flops_per_edge": expert.flops_per_edge,
                                    "router": router_name,
                                    "router_flops_per_edge": router.flops_per_edge,
                                    "execution": mode,
                                    "route_fraction_of_active": route_fraction,
                                    "expert_executed_graph_fraction": expert_fraction,
                                    "analytical_flops": analytical_flops,
                                    "median_latency_ms": statistics.median(samples),
                                    "mean_latency_ms": statistics.mean(samples),
                                    "std_latency_ms": statistics.stdev(samples),
                                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                                    / (1024**2),
                                }
                            )
    _write_csv(Path(output_dir) / "latency_matrix.csv", rows)
    _write_json(Path(output_dir) / "latency_matrix.json", rows)
    return rows


def summarize_break_even(
    accuracy_rows: list[dict[str, Any]], latency_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    accuracy_index = {
        (
            row["interaction_density"],
            row["residual_necessity"],
            row["expert"],
            row["router"],
        ): row
        for row in accuracy_rows
    }
    latency_index = {
        (
            row["batch_size"],
            row["interaction_density"],
            row["residual_necessity"],
            row["expert"],
            row["router"],
            row["execution"],
        ): row
        for row in latency_rows
    }
    summaries: list[dict[str, Any]] = []
    for key, sparse in latency_index.items():
        batch, density, necessity, expert, router, execution = key
        if execution != "hard_sparse":
            continue
        contact = latency_index[(
            batch,
            density,
            necessity,
            expert,
            router,
            "contact_packed",
        )]
        accuracy = accuracy_index[(density, necessity, expert, router)]
        speedup = contact["median_latency_ms"] / sparse["median_latency_ms"]
        summaries.append(
            {
                "batch_size": batch,
                "interaction_density": density,
                "residual_necessity": necessity,
                "expert": expert,
                "expert_flops_per_edge": sparse["expert_flops_per_edge"],
                "router": router,
                "router_flops_per_edge": sparse["router_flops_per_edge"],
                "contact_latency_ms": contact["median_latency_ms"],
                "adaptive_latency_ms": sparse["median_latency_ms"],
                "speedup_vs_contact": speedup,
                "is_faster": speedup > 1.0,
                "delta_error": accuracy["delta_error"],
                "delta_rmse": accuracy["delta_rmse"],
                "relative_mse_change": accuracy["relative_mse_change"],
                "router_auroc": accuracy["router_auroc"],
                "route_fraction_of_active": accuracy["route_fraction_of_active"],
            }
        )
    return summaries


def run_v06(
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    full_matrix: bool = False,
    reuse_models: bool = False,
    config: V06Config | None = None,
) -> dict[str, Any]:
    cfg = config or V06Config()
    device = resolve_device(device_name)
    seed_everything(cfg.seed)
    output = Path(output_dir)
    if reuse_models:
        experts, routers, history = load_models(output / "models", device)
    else:
        experts, routers, history = train_models(cfg, output / "models", device)
    for module in (*experts.values(), *routers.values()):
        module.eval()
    accuracy = accuracy_sweep(cfg, experts, routers, output, device)
    latency = latency_sweep(
        cfg,
        experts,
        routers,
        output,
        device,
        full_matrix=full_matrix,
    )
    break_even = summarize_break_even(accuracy, latency)
    _write_csv(output / "break_even.csv", break_even)
    _write_json(output / "break_even.json", break_even)
    result = {
        "config": cfg.__dict__,
        "full_matrix": full_matrix,
        "reused_models": reuse_models,
        "accuracy_rows": len(accuracy),
        "latency_rows": len(latency),
        "break_even_rows": len(break_even),
        "faster_configurations": sum(bool(row["is_faster"]) for row in break_even),
    }
    _write_json(output / "summary.json", result)
    return result

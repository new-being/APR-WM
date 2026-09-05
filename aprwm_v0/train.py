from __future__ import annotations

import csv
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .evaluate import autocast_context, evaluate_model
from .losses import dynamics_loss
from .models import MODEL_NAMES, DynamicsModel, build_model, parameter_count
from .simulator import SyntheticWorld


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _training_batch(
    world: SyntheticWorld,
    config: ExperimentConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, actions = world.sample_trajectories(
        config.train.batch_size,
        config.train.n_objects,
        config.train.trajectory_steps,
        generator=generator,
        device=device,
        ood=False,
    )
    batch, horizon, nodes, _ = actions.shape
    current = states[:, :-1].reshape(batch * horizon, nodes, -1)
    action = actions.reshape(batch * horizon, nodes, -1)
    target = states[:, 1:].reshape(batch * horizon, nodes, -1)
    return current, action, target


def train_one(
    model_name: str,
    config: ExperimentConfig,
    *,
    output_root: Path | None = None,
) -> dict[str, object]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    world = SyntheticWorld(config.world)
    model = build_model(model_name, config.world, config.model).to(device)
    run_dir = (output_root or Path(config.output_dir)) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config.to_dict())

    print(
        json.dumps(
            {
                "event": "start",
                "model": model_name,
                "device": str(device),
                "parameters": parameter_count(model),
                "steps": 0 if model_name == "physics" else config.train.steps,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    if model_name != "physics" and config.train.steps > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        amp_enabled = config.train.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + 101)
        model.train()
        for step in range(1, config.train.steps + 1):
            if model.is_adaptive:
                model.force_open = step <= config.train.regularizer_warmup_steps
            state, action, target = _training_batch(world, config, generator, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.train.amp):
                output = model(state, action, hard_routing=False)
                loss, metrics = dynamics_loss(
                    output,
                    target,
                    config.train,
                    step=step,
                    is_adaptive=model.is_adaptive,
                    use_residual_regularizer=model_name in ("residual", "adaptive"),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % config.train.log_every == 0 or step == config.train.steps:
                record: dict[str, float | int] = {"step": step, **metrics}
                history.append(record)
                print(json.dumps({"event": "train", "model": model_name, **record}), flush=True)

    train_seconds = time.perf_counter() - started
    checkpoint = {
        "format_version": 1,
        "model_name": model_name,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    _write_json(run_dir / "train_history.json", history)
    evaluation = evaluate_model(model, world, config, device)
    result: dict[str, object] = {
        **evaluation,
        "device": str(device),
        "parameters": parameter_count(model),
        "train_seconds": train_seconds,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
    }
    _write_json(run_dir / "metrics.json", result)
    print(json.dumps({"event": "complete", **result}, ensure_ascii=False), flush=True)
    return result


def write_suite_summary(
    results: list[dict[str, object]], output_root: Path
) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        row: dict[str, object] = {
            "model": result["model"],
            "parameters": result["parameters"],
            "peak_cuda_memory_mb": result["peak_cuda_memory_mb"],
        }
        for split in ("id", "ood"):
            metrics = result[split]
            assert isinstance(metrics, dict)
            for key, value in metrics.items():
                row[f"{split}_{key}"] = value
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "suite_metrics.json", results)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "suite_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(config: ExperimentConfig) -> list[dict[str, object]]:
    output_root = Path(config.output_dir)
    results = [train_one(name, config, output_root=output_root) for name in MODEL_NAMES]
    write_suite_summary(results, output_root)
    return results


def summarize_existing(output_root: str | Path) -> list[dict[str, object]]:
    root = Path(output_root)
    results: list[dict[str, object]] = []
    for name in MODEL_NAMES:
        metrics_path = root / name / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for {name}: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    write_suite_summary(results, root)
    return results


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[DynamicsModel, ExperimentConfig]:
    from .config import _merge_dataclass

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    config = ExperimentConfig()
    stored_config = copy.deepcopy(checkpoint["config"])
    # V0 migration: fixed thresholds were replaced by explicit compute budgets.
    stored_config.get("model", {}).pop("hard_threshold", None)
    _merge_dataclass(config, stored_config)
    config.validate()
    model = build_model(checkpoint["model_name"], config.world, config.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorldConfig:
    dt: float = 0.05
    spring_k: float = 35.0
    damping: float = 1.2
    ground_drag: float = 0.12
    nonlinear_k: float = 260.0
    nonlinear_friction: float = 0.55
    tangent_speed: float = 0.08
    contact_fraction: float = 0.7
    position_extent: float = 0.8
    radius_min: float = 0.10
    radius_max: float = 0.16
    train_mass_min: float = 0.7
    train_mass_max: float = 1.5
    train_friction_min: float = 0.05
    train_friction_max: float = 0.7
    ood_mass_min: float = 0.4
    ood_mass_max: float = 2.2
    ood_friction_min: float = 0.0
    ood_friction_max: float = 1.0
    train_complex_probability: float = 0.35
    ood_complex_probability: float = 0.55
    velocity_scale: float = 0.18
    train_action_scale: float = 0.7
    ood_action_scale: float = 1.0
    action_smoothing: float = 0.75


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    router_hidden_dim: int = 32
    depth: int = 3
    gate_temperature: float = 0.7
    residual_budget_fraction: float = 0.6
    training_routing: str = "hard_topk"


@dataclass
class TrainConfig:
    steps: int = 4_000
    batch_size: int = 128
    trajectory_steps: int = 4
    n_objects: int = 6
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    lambda_residual: float = 2.0e-4
    lambda_gate: float = 2.0e-5
    lambda_router: float = 2.0e-3
    residual_need_threshold: float = 0.05
    residual_need_temperature: float = 0.02
    regularizer_warmup_steps: int = 600
    log_every: int = 100
    amp: bool = True


@dataclass
class EvalConfig:
    batches: int = 8
    batch_size: int = 128
    id_n_objects: int = 6
    ood_n_objects: int = 8
    horizon: int = 20
    stability_speed: float = 20.0


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "auto"
    output_dir: str = "runs/v0"
    world: WorldConfig = field(default_factory=WorldConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.train.n_objects < 2 or self.eval.id_n_objects < 2 or self.eval.ood_n_objects < 2:
            raise ValueError("At least two objects are required for interaction graphs.")
        if self.world.dt <= 0:
            raise ValueError("world.dt must be positive.")
        if self.train.steps < 0 or self.train.batch_size <= 0:
            raise ValueError("Invalid training schedule.")
        if self.model.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if not 0 < self.model.residual_budget_fraction <= 1:
            raise ValueError("residual_budget_fraction must be in (0, 1].")
        if self.model.training_routing not in ("hard_topk", "soft"):
            raise ValueError("model.training_routing must be 'hard_topk' or 'soft'.")
        if self.train.residual_need_temperature <= 0:
            raise ValueError("residual_need_temperature must be positive.")


def _merge_dataclass(instance: Any, values: dict[str, Any], prefix: str = "") -> Any:
    fields = instance.__dataclass_fields__
    unknown = set(values) - set(fields)
    if unknown:
        location = prefix or "config"
        raise ValueError(f"Unknown keys in {location}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be an object")
            _merge_dataclass(current, value, f"{prefix}{key}.")
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        _merge_dataclass(config, values)
    config.validate()
    return config

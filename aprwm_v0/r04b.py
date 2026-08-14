from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend
from .r02 import V6R02Config, _backend_config
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _ridge_fit


STRATEGIES = (
    "physics",
    "memoryless",
    "history",
    "utility_memoryless",
    "utility_history",
    "oracle_history",
)


@dataclass
class V6R04BConfig(V6R02Config):
    """Controlled delayed-force test of fallback state sufficiency.

    The C2 force is -alpha * tau[t-1].  The previous command is observable in
    the trajectory but absent from the instantaneous state, so this benchmark
    isolates Markov closure without changing residual network capacity.
    """

    seeds: tuple[int, ...] = (3001, 3011, 3021, 3031, 3041)
    episodes_per_regime: int = 24
    train_count: int = 48
    validation_count: int = 24
    query_count: int = 8
    history_length: int = 3
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    hysteresis_strength: float = 0.10
    fallback_ridge: float = 1.0e-3
    utility_margin: float = 1.0e-5
    joint_limit_margin: float = 0.02
    stability_min: float = 0.99


def _memoryless_design(state: torch.Tensor) -> torch.Tensor:
    """A deliberately generous instantaneous basis (12 features)."""
    q = (state[..., 0] - 0.55) / 0.40
    velocity = state[..., 1] / 1.40
    torque = state[..., 2] / 0.80
    one = torch.ones_like(q)
    return torch.stack(
        (
            one,
            q,
            velocity,
            torque,
            q.square(),
            velocity.square(),
            torque.square(),
            q * velocity,
            q * torque,
            velocity * torque,
            torch.sin(q),
            torch.tanh(velocity),
        ),
        dim=-1,
    )


def _history_design(state: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    return torch.cat((_memoryless_design(state), history / 0.80), dim=-1)


def _sample_force_inputs(
    episodes: int,
    count: int,
    history_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = 0.18 + 0.75 * torch.rand(episodes, count, generator=generator)
    velocity = 2.8 * torch.rand(episodes, count, generator=generator) - 1.4
    torque = 1.6 * torch.rand(episodes, count, generator=generator) - 0.8
    history = 1.6 * torch.rand(
        episodes, count, history_length, generator=generator
    ) - 0.8
    return torch.stack((q, velocity, torque), dim=-1), history


def _force_targets(
    backends: list[SapienHingeBackend],
    state: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for episode, backend in enumerate(backends):
        targets = []
        for sample in range(state.shape[1]):
            point = state[episode, sample]
            residual, _, _ = backend.generalized_force_sample(
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(history[episode, sample, 0]),
            )
            targets.append(residual)
        rows.append(targets)
    return torch.tensor(rows, dtype=torch.float32)


def _predict_force(
    strategy: str,
    state: torch.Tensor,
    history: torch.Tensor,
    memoryless_coefficient: torch.Tensor,
    history_coefficient: torch.Tensor,
    memoryless_useful: torch.Tensor,
    history_useful: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    memoryless = torch.einsum(
        "...d,...d->...",
        _memoryless_design(state),
        memoryless_coefficient,
    )
    history_prediction = torch.einsum(
        "...d,...d->...",
        _history_design(state, history),
        history_coefficient,
    )
    if strategy == "physics":
        return torch.zeros_like(memoryless)
    if strategy == "memoryless":
        return memoryless
    if strategy == "history":
        return history_prediction
    if strategy == "utility_memoryless":
        return torch.where(memoryless_useful, memoryless, torch.zeros_like(memoryless))
    if strategy == "utility_history":
        return torch.where(history_useful, history_prediction, torch.zeros_like(history_prediction))
    if strategy == "oracle_history":
        return -alpha * history[..., 0]
    raise ValueError(strategy)


def _truth_sequences(
    backends: list[SapienHingeBackend],
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    initial_history: torch.Tensor,
    config: V6R04BConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizon_set = set(config.horizons)
    all_truth = []
    all_validity = []
    for episode, backend in enumerate(backends):
        episode_truth = []
        episode_validity = []
        for query in range(initial_state.shape[1]):
            q = float(initial_state[episode, query, 0])
            velocity = float(initial_state[episode, query, 1])
            history = initial_history[episode, query].clone()
            valid = True
            snapshots: dict[int, tuple[float, float]] = {}
            validity: dict[int, bool] = {}
            for step in range(1, max(config.horizons) + 1):
                torque = float(actions[episode, query, step - 1])
                q, velocity, step_valid = backend.transition_with_validity(
                    q,
                    velocity,
                    torque,
                    float(history[0]),
                    horizon=1,
                    limit_margin=config.joint_limit_margin,
                )
                valid &= step_valid
                history = torch.cat((history.new_tensor((torque,)), history[:-1]))
                if step in horizon_set:
                    snapshots[step] = (q, velocity)
                    validity[step] = valid
            episode_truth.append([snapshots[h] for h in config.horizons])
            episode_validity.append([validity[h] for h in config.horizons])
        all_truth.append(episode_truth)
        all_validity.append(episode_validity)
    return (
        torch.tensor(all_truth, dtype=torch.float32),
        torch.tensor(all_validity, dtype=torch.bool),
    )


def _model_sequences(
    strategy: str,
    backends: list[SapienHingeBackend],
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    initial_history: torch.Tensor,
    memoryless_coefficient: torch.Tensor,
    history_coefficient: torch.Tensor,
    memoryless_useful: torch.Tensor,
    history_useful: torch.Tensor,
    config: V6R04BConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    episodes, queries = initial_state.shape[:2]
    q = initial_state[..., 0].clone()
    velocity = initial_state[..., 1].clone()
    history = initial_history.clone()
    stable = torch.ones((episodes, queries), dtype=torch.bool)
    snapshots = []
    stability_snapshots = []
    horizon_set = set(config.horizons)
    for macro_step in range(1, max(config.horizons) + 1):
        torque = actions[..., macro_step - 1]
        for _ in range(config.macro_steps):
            state = torch.stack((q, velocity, torque), dim=-1)
            residual = _predict_force(
                strategy,
                state,
                history,
                memoryless_coefficient[:, None, :].expand(-1, queries, -1),
                history_coefficient[:, None, :].expand(-1, queries, -1),
                memoryless_useful[:, None].expand(-1, queries),
                history_useful[:, None].expand(-1, queries),
                config.hysteresis_strength,
            )
            acceleration = torch.empty_like(q)
            for episode, backend in enumerate(backends):
                pinocchio = backend.pinocchio
                damping = backend.damping
                for query in range(queries):
                    q_value = float(q[episode, query])
                    velocity_value = float(velocity[episode, query])
                    h = float(
                        pinocchio.compute_inverse_dynamics(
                            (q_value,), (velocity_value,), (0.0,)
                        )[0]
                    )
                    generalized_force = (
                        h
                        + float(torque[episode, query])
                        - damping * velocity_value
                        + float(residual[episode, query])
                    )
                    acceleration[episode, query] = float(
                        pinocchio.compute_forward_dynamics(
                            (q_value,), (velocity_value,), (generalized_force,)
                        )[0]
                    )
            q = q + config.physics_dt * velocity
            velocity = velocity + config.physics_dt * acceleration
            stable &= (
                torch.isfinite(q)
                & torch.isfinite(velocity)
                & (q.abs() < 10.0)
                & (velocity.abs() < 50.0)
            )
            q = torch.nan_to_num(q, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            velocity = torch.nan_to_num(
                velocity, nan=0.0, posinf=50.0, neginf=-50.0
            ).clamp(-50, 50)
        history = torch.cat((torque.unsqueeze(-1), history[..., :-1]), dim=-1)
        if macro_step in horizon_set:
            snapshots.append(torch.stack((q, velocity), dim=-1))
            stability_snapshots.append(stable.clone())
    return torch.stack(snapshots, dim=2), torch.stack(stability_snapshots, dim=-1)


def _evaluate_seed(
    config: V6R04BConfig,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(seed + 230_000)
    backend_config = _backend_config(config)
    backend_config.hysteresis_strength = config.hysteresis_strength
    densities = config.density_min + (config.density_max - config.density_min) * torch.rand(
        config.episodes_per_regime, generator=generator
    )
    dampings = config.damping_min + (config.damping_max - config.damping_min) * torch.rand(
        config.episodes_per_regime, generator=generator
    )
    backends = [
        SapienHingeBackend(
            density=float(density),
            damping=float(damping),
            regime=2,
            config=backend_config,
        )
        for density, damping in zip(densities, dampings)
    ]
    train_state, train_history = _sample_force_inputs(
        config.episodes_per_regime,
        config.train_count,
        config.history_length,
        generator,
    )
    validation_state, validation_history = _sample_force_inputs(
        config.episodes_per_regime,
        config.validation_count,
        config.history_length,
        generator,
    )
    train_clean = _force_targets(backends, train_state, train_history)
    validation_clean = _force_targets(backends, validation_state, validation_history)
    noise_generator = torch.Generator().manual_seed(seed + 231_000)
    train_y = train_clean + config.force_observation_noise * torch.randn(
        train_clean.shape, generator=noise_generator
    )
    validation_y = validation_clean + config.force_observation_noise * torch.randn(
        validation_clean.shape, generator=noise_generator
    )
    memoryless_coefficient = _ridge_fit(
        _memoryless_design(train_state), train_y, config.fallback_ridge
    )
    history_coefficient = _ridge_fit(
        _history_design(train_state, train_history), train_y, config.fallback_ridge
    )
    memoryless_validation = torch.einsum(
        "bnd,bd->bn", _memoryless_design(validation_state), memoryless_coefficient
    )
    history_validation = torch.einsum(
        "bnd,bd->bn",
        _history_design(validation_state, validation_history),
        history_coefficient,
    )
    physics_validation_error = validation_y.square().mean(dim=-1)
    memoryless_validation_error = (
        validation_y - memoryless_validation
    ).square().mean(dim=-1)
    history_validation_error = (
        validation_y - history_validation
    ).square().mean(dim=-1)
    memoryless_useful = (
        memoryless_validation_error + config.utility_margin < physics_validation_error
    )
    history_useful = history_validation_error + config.utility_margin < physics_validation_error

    initial_q = 0.35 + 0.40 * torch.rand(
        config.episodes_per_regime, config.query_count, generator=generator
    )
    initial_velocity = 0.80 * torch.rand(
        config.episodes_per_regime, config.query_count, generator=generator
    ) - 0.40
    initial_torque = torch.zeros_like(initial_q)
    initial_state = torch.stack((initial_q, initial_velocity, initial_torque), dim=-1)
    actions = 0.8 * torch.rand(
        config.episodes_per_regime,
        config.query_count,
        max(config.horizons),
        generator=generator,
    ) - 0.4
    initial_history = 0.8 * torch.rand(
        config.episodes_per_regime,
        config.query_count,
        config.history_length,
        generator=generator,
    ) - 0.4
    truth, validity = _truth_sequences(
        backends, initial_state, actions, initial_history, config
    )
    scale = truth.new_tensor((1.0, 2.0))
    horizon_rows = []
    force_rows = []
    clean_predictions = {
        "physics": torch.zeros_like(validation_clean),
        "memoryless": memoryless_validation,
        "history": history_validation,
        "utility_memoryless": torch.where(
            memoryless_useful[:, None], memoryless_validation, torch.zeros_like(memoryless_validation)
        ),
        "utility_history": torch.where(
            history_useful[:, None], history_validation, torch.zeros_like(history_validation)
        ),
        "oracle_history": -config.hysteresis_strength * validation_history[..., 0],
    }
    utility_rates = {
        "physics": 0.0,
        "memoryless": 1.0,
        "history": 1.0,
        "utility_memoryless": float(memoryless_useful.float().mean()),
        "utility_history": float(history_useful.float().mean()),
        "oracle_history": 1.0,
    }
    for strategy in STRATEGIES:
        prediction, stable = _model_sequences(
            strategy,
            backends,
            initial_state,
            actions,
            initial_history,
            memoryless_coefficient,
            history_coefficient,
            memoryless_useful,
            history_useful,
            config,
        )
        force_rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "force_rmse": float(
                    (clean_predictions[strategy] - validation_clean).square().mean().sqrt()
                ),
                "fallback_usage": utility_rates[strategy],
            }
        )
        for horizon_index, horizon in enumerate(config.horizons):
            valid = validity[..., horizon_index]
            state_error = (
                (prediction[..., horizon_index, :] - truth[..., horizon_index, :]) / scale
            ).square().mean(dim=-1)
            horizon_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "horizon": horizon,
                    "state_rmse": float(
                        torch.sqrt((state_error * valid).sum() / valid.sum().clamp_min(1))
                    ),
                    "stable_fraction": float(stable[..., horizon_index].float().mean()),
                    "valid_window_fraction": float(valid.float().mean()),
                }
            )
    return force_rows, horizon_rows


def run_v6r04b(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R04BConfig | None = None,
) -> dict[str, Any]:
    cfg = config or V6R04BConfig()
    device = resolve_device(device_name)
    force_rows = []
    horizon_rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        force, horizons = _evaluate_seed(cfg, seed, device)
        force_rows.extend(force)
        horizon_rows.extend(horizons)
    root = Path(output_dir)
    _write_csv(root / "fallback_force_metrics.csv", force_rows)
    _write_csv(root / "fallback_horizons.csv", horizon_rows)

    def mean_force(strategy: str) -> float:
        return statistics.mean(
            float(row["force_rmse"]) for row in force_rows if row["strategy"] == strategy
        )

    def mean_usage(strategy: str) -> float:
        return statistics.mean(
            float(row["fallback_usage"]) for row in force_rows if row["strategy"] == strategy
        )

    def mean_horizon(strategy: str, horizon: int) -> float:
        return statistics.mean(
            float(row["state_rmse"])
            for row in horizon_rows
            if row["strategy"] == strategy and int(row["horizon"]) == horizon
        )

    def mean_stability(strategy: str) -> float:
        return statistics.mean(
            float(row["stable_fraction"])
            for row in horizon_rows
            if row["strategy"] == strategy and int(row["horizon"]) == cfg.horizons[-1]
        )

    final = "utility_history"
    gates = {
        "history_force_closure": mean_force("history") < mean_force("memoryless"),
        "fallback_positive_utility": mean_horizon(final, cfg.horizons[-1]) < mean_horizon("physics", cfg.horizons[-1]),
        "history_beats_memoryless": mean_horizon(final, cfg.horizons[-1]) < mean_horizon("utility_memoryless", cfg.horizons[-1]),
        "stable": mean_stability(final) >= cfg.stability_min,
    }
    summary = {
        "scope": "V6R0.4-B history-aware C2 fallback",
        "held_out_seeds": list(cfg.seeds),
        "c2_definition": "force residual = -alpha * previous commanded torque",
        "history_length": cfg.history_length,
        "instantaneous_basis_dimension": 12,
        "history_basis_dimension": 12 + cfg.history_length,
        "force_rmse": {strategy: mean_force(strategy) for strategy in STRATEGIES},
        "h16_rmse": {
            strategy: mean_horizon(strategy, cfg.horizons[-1]) for strategy in STRATEGIES
        },
        "fallback_usage": {
            strategy: mean_usage(strategy)
            for strategy in ("utility_memoryless", "utility_history")
        },
        "final_strategy": final,
        "gates": gates,
        "overall_go": all(gates.values()),
        "config": {
            **cfg.__dict__,
            "seeds": list(cfg.seeds),
            "horizons": list(cfg.horizons),
        },
        "robotwin_assets_downloaded": False,
    }
    _write_json(root / "summary.json", summary)
    return summary

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend, V6R0Config, _aggregate_seed_rows, _sample_local_points
from .train import resolve_device, seed_everything
from .v06 import _binary_auroc, _write_csv, _write_json
from .v3 import tangent_decomposition
from .v4 import operator_library, orthogonal_operator_gain


DRAG_OPERATOR = 4


@dataclass
class V6R01Config:
    seeds: tuple[int, ...] = (13, 23, 33)
    episodes: int = 24
    samples_per_episode: int = 48
    physics_dt: float = 1.0 / 250.0
    force_observation_noise: float = 0.002
    density_min: float = 110.0
    density_max: float = 230.0
    damping_min: float = 0.025
    damping_max: float = 0.075
    drag_strength: float = 0.12
    ridge: float = 1.0e-5
    clean_rmse_max: float = 1.0e-5
    bias_z_max: float = 3.5
    variance_ratio_low: float = 0.80
    variance_ratio_high: float = 1.20
    state_correlation_max: float = 0.10
    tangent_auroc_min: float = 0.80
    trigger_noise_multiplier: float = 1.35
    c0_trigger_rate_max: float = 0.01


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x.flatten() - x.mean()
    y_centered = y.flatten() - y.mean()
    denominator = torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    ).clamp_min(1.0e-12)
    return float((x_centered * y_centered).sum() / denominator)


def _backend_config(config: V6R01Config) -> V6R0Config:
    return V6R0Config(
        physics_dt=config.physics_dt,
        macro_steps=1,
        density_min=config.density_min,
        density_max=config.density_max,
        damping_min=config.damping_min,
        damping_max=config.damping_max,
        drag_strength=config.drag_strength,
    )


def _collect_force_regime(
    config: V6R01Config,
    seed: int,
    regime: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(seed + 170_000 + 10_000 * regime)
    noise_generator = torch.Generator(device=device).manual_seed(
        seed + 171_000 + 10_000 * regime
    )
    backend_config = _backend_config(config)
    feature_rows = []
    tangent_rows = []
    clean_rows = []
    sample_rows: list[dict[str, Any]] = []
    for episode in range(config.episodes):
        density = config.density_min + (
            config.density_max - config.density_min
        ) * float(torch.rand((), generator=generator))
        damping = config.damping_min + (
            config.damping_max - config.damping_min
        ) * float(torch.rand((), generator=generator))
        backend = SapienHingeBackend(
            density=density,
            damping=damping,
            regime=regime,
            config=backend_config,
        )
        state, memory = _sample_local_points(
            config.samples_per_episode, "validation", generator
        )
        episode_clean = []
        episode_tangent = []
        for sample, (row, hidden) in enumerate(zip(state, memory)):
            residual, density_tangent, acceleration = (
                backend.generalized_force_sample(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(hidden),
                )
            )
            episode_clean.append(residual)
            # d r_tau / d density and d r_tau / d damping.
            episode_tangent.append((density_tangent, float(row[1])))
            sample_rows.append(
                {
                    "seed": seed,
                    "regime": "c0_parameter" if regime == 0 else "c1_drag",
                    "episode": episode,
                    "sample": sample,
                    "q": float(row[0]),
                    "qvel": float(row[1]),
                    "torque": float(row[2]),
                    "qacc": acceleration,
                    "density": density,
                    "damping": damping,
                    "clean_force_residual": residual,
                    "density_tangent": density_tangent,
                }
            )
        feature_rows.append(state[:, :2])
        tangent_rows.append(torch.tensor(episode_tangent, dtype=torch.float32))
        clean_rows.append(torch.tensor(episode_clean, dtype=torch.float32))
    features = torch.stack(feature_rows).to(device)
    tangent = torch.stack(tangent_rows).to(device)
    clean = torch.stack(clean_rows).to(device)
    observed = clean + config.force_observation_noise * torch.randn(
        clean.shape, generator=noise_generator, device=device
    )
    offset = 0
    for episode in range(config.episodes):
        for sample in range(config.samples_per_episode):
            sample_rows[offset]["observed_force_residual"] = float(
                observed[episode, sample]
            )
            offset += 1
    return {
        "features": features,
        "tangent": tangent,
        "clean": clean,
        "observed": observed,
    }, sample_rows


def _closure_metrics(
    data: dict[str, torch.Tensor], config: V6R01Config
) -> dict[str, Any]:
    clean = data["clean"]
    observed = data["observed"]
    features = data["features"]
    count = observed.numel()
    bias = float(observed.mean())
    bias_standard_error = config.force_observation_noise / count**0.5
    variance = float(observed.var(unbiased=True))
    expected_variance = config.force_observation_noise**2
    variance_ratio = variance / expected_variance
    corr_q = _pearson(observed, features[..., 0])
    corr_qvel = _pearson(observed, features[..., 1])
    clean_rmse = float(clean.square().mean().sqrt())
    bias_z = abs(bias) / bias_standard_error
    max_correlation = max(abs(corr_q), abs(corr_qvel))
    passed = bool(
        clean_rmse <= config.clean_rmse_max
        and bias_z <= config.bias_z_max
        and config.variance_ratio_low <= variance_ratio <= config.variance_ratio_high
        and max_correlation <= config.state_correlation_max
    )
    return {
        "c0_clean_force_rmse": clean_rmse,
        "c0_force_bias": bias,
        "c0_force_bias_z": bias_z,
        "c0_force_variance": variance,
        "c0_noise_variance_ratio": variance_ratio,
        "c0_corr_q": corr_q,
        "c0_corr_qvel": corr_qvel,
        "c0_max_abs_state_corr": max_correlation,
        "c0_closure_pass": float(passed),
    }


def _c1_metrics(
    c0: dict[str, torch.Tensor],
    c1: dict[str, torch.Tensor],
    config: V6R01Config,
) -> dict[str, float]:
    _, c0_orthogonal, _ = tangent_decomposition(
        c0["tangent"], c0["observed"], config.ridge
    )
    _, c1_orthogonal, _ = tangent_decomposition(
        c1["tangent"], c1["observed"], config.ridge
    )
    c0_tangent_score = c0_orthogonal.square().mean(dim=-1).sqrt()
    c1_tangent_score = c1_orthogonal.square().mean(dim=-1).sqrt()
    c0_magnitude_score = c0["observed"].square().mean(dim=-1).sqrt()
    c1_magnitude_score = c1["observed"].square().mean(dim=-1).sqrt()
    labels = torch.cat(
        (torch.zeros_like(c0_tangent_score, dtype=torch.bool),
         torch.ones_like(c1_tangent_score, dtype=torch.bool))
    )
    tangent_scores = torch.cat((c0_tangent_score, c1_tangent_score))
    magnitude_scores = torch.cat((c0_magnitude_score, c1_magnitude_score))
    threshold = config.trigger_noise_multiplier * config.force_observation_noise

    operators = operator_library(c1["features"])
    gain = orthogonal_operator_gain(
        c1["tangent"], c1["observed"], operators, ridge=config.ridge
    )
    ranking = gain.argsort(dim=-1, descending=True)
    drag = operators[..., DRAG_OPERATOR]
    _, drag_orthogonal, _ = tangent_decomposition(
        c1["tangent"], drag, config.ridge
    )
    coefficient = (
        (c1_orthogonal * drag_orthogonal).sum(dim=-1)
        / drag_orthogonal.square().sum(dim=-1).clamp_min(1.0e-12)
    )
    remaining = c1_orthogonal - coefficient.unsqueeze(-1) * drag_orthogonal
    return {
        "tangent_detection_auroc": _binary_auroc(tangent_scores, labels),
        "magnitude_detection_auroc": _binary_auroc(magnitude_scores, labels),
        "c0_structural_trigger_rate": float(
            (c0_tangent_score > threshold).float().mean()
        ),
        "c1_structural_trigger_recall": float(
            (c1_tangent_score > threshold).float().mean()
        ),
        "c1_drag_top1_recovery": float((ranking[:, 0] == DRAG_OPERATOR).float().mean()),
        "c1_drag_top3_recovery": float(
            (ranking[:, :3] == DRAG_OPERATOR).any(dim=-1).float().mean()
        ),
        "c1_drag_coefficient_mean": float(coefficient.mean()),
        "c1_drag_coefficient_mae": float(
            (coefficient + config.drag_strength).abs().mean()
        ),
        "c1_orthogonal_force_rmse_before": float(
            c1_orthogonal.square().mean().sqrt()
        ),
        "c1_orthogonal_force_rmse_after_drag": float(
            remaining.square().mean().sqrt()
        ),
    }


def run_v6r01_closure(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R01Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6R01Config()
    device = resolve_device(device_name)
    c0_by_seed: dict[int, dict[str, torch.Tensor]] = {}
    seed_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    # Phase A is intentionally C0-only.  Phase B is not collected unless every
    # development seed passes the declared closure gate.
    for seed in cfg.seeds:
        seed_everything(seed)
        c0, rows = _collect_force_regime(cfg, seed, 0, device)
        c0_by_seed[seed] = c0
        seed_rows.append({"seed": seed, **_closure_metrics(c0, cfg)})
        sample_rows.extend(rows)
    closure_pass = all(bool(row["c0_closure_pass"]) for row in seed_rows)

    c1_executed = False
    if closure_pass:
        c1_executed = True
        for row in seed_rows:
            seed = int(row["seed"])
            seed_everything(seed)
            c1, rows = _collect_force_regime(cfg, seed, 1, device)
            row.update(_c1_metrics(c0_by_seed[seed], c1, cfg))
            sample_rows.extend(rows)

    root = Path(output_dir)
    _write_csv(root / "force_samples.csv", sample_rows)
    _write_csv(root / "seed_metrics.csv", seed_rows)
    _write_csv(root / "metrics_aggregate.csv", _aggregate_seed_rows(seed_rows))

    if c1_executed:
        tangent_values = [float(row["tangent_detection_auroc"]) for row in seed_rows]
        trigger_values = [float(row["c0_structural_trigger_rate"]) for row in seed_rows]
        c1_go = bool(
            statistics.mean(tangent_values) >= cfg.tangent_auroc_min
            and min(tangent_values) > 0.5
            and statistics.mean(trigger_values) <= cfg.c0_trigger_rate_max
        )
    else:
        c1_go = False
    summary = {
        "scope": "V6R0.1 generalized-force interface validation",
        "residual_coordinate": "generalized_force",
        "equation": "r_tau = M(q) qdd + h(q,qdot) - tau_known",
        "phase_a_c0_closure_pass": closure_pass,
        "phase_b_c1_executed": c1_executed,
        "phase_b_c1_detection_pass": c1_go,
        "full_v6_revision_executed": False,
        "c2_executed": False,
        "robotwin_assets_downloaded": False,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds)},
        "criteria": {
            "clean_rmse_max": cfg.clean_rmse_max,
            "bias_z_max": cfg.bias_z_max,
            "variance_ratio": [cfg.variance_ratio_low, cfg.variance_ratio_high],
            "state_correlation_max": cfg.state_correlation_max,
            "tangent_auroc_min": cfg.tangent_auroc_min,
            "c0_trigger_rate_max": cfg.c0_trigger_rate_max,
        },
        "rows": len(seed_rows),
    }
    _write_json(root / "summary.json", summary)
    return summary

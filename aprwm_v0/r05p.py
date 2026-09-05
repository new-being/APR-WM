from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .r0 import SapienHingeBackend
from .r02 import (
    _backend_config,
    _model_rollout,
    _normalize_fallback_features,
)
from .r04a import (
    V6R04AConfig,
    _collect_native_seed,
    _dynamical_features,
    _episode_rmse,
)
from .r05 import pairwise_auroc
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit
from .v4 import operator_library
from .v5 import _fit_base_posterior


PARAMETERIZATIONS = ("fixed_theta", "refit_theta")


@dataclass
class V6R05PConfig(V6R04AConfig):
    seeds: tuple[int, ...] = (5001, 5011, 5021, 5031, 5041)
    alpha_magnitudes: tuple[float, ...] = (0.15, 0.30, 0.45)
    power_noise_multiplier: float = 3.0
    power_velocity_scale: float = 1.4

    @property
    def power_noise_epsilon(self) -> float:
        # Fixed from the declared C0 generalized-force observation floor and
        # the preregistered validation velocity support, not from H16 labels.
        return (
            self.power_noise_multiplier
            * self.force_observation_noise
            * self.power_velocity_scale
        )


def structural_power(alpha: float, velocity: torch.Tensor) -> torch.Tensor:
    """Zero-input power for r_tau = alpha * |v| * v."""
    return alpha * velocity.abs().pow(3)


def power_violation_metrics(
    alpha: float,
    velocity: torch.Tensor,
    epsilon: float,
) -> tuple[float, float]:
    power = structural_power(alpha, velocity)
    return float(power.max()), float((power > epsilon).float().mean())


def _candidate_coefficients(
    parameterization: str,
    alpha: float,
    base_mean: torch.Tensor,
    train_tangent: torch.Tensor,
    train_physical: torch.Tensor,
    train_y: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    alpha_column = base_mean.new_full((base_mean.shape[0], 1), alpha)
    if parameterization == "fixed_theta":
        return torch.cat((base_mean, alpha_column), dim=-1)
    if parameterization == "refit_theta":
        operator = operator_library(train_physical)[..., 4]
        theta = _ridge_fit(train_tangent, train_y - alpha * operator, ridge)
        return torch.cat((theta, alpha_column), dim=-1)
    raise ValueError(parameterization)


def _evaluate_seed(
    config: V6R05PConfig,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    data = _collect_native_seed(config, seed, device)
    episodes = data["density"].shape[0]
    train_tangent = torch.cat(
        (data["passive_tangent"], data["discovery_tangent"]), dim=1
    )
    train_physical = torch.cat(
        (data["passive_physical"], data["discovery_physical"]), dim=1
    )
    train_y = torch.cat((data["passive_y"], data["discovery_y"]), dim=1)
    base_mean, base_covariance = _fit_base_posterior(
        train_tangent, train_y, config.force_observation_noise, config
    )
    precision = torch.linalg.inv(base_covariance)
    train_base = torch.einsum("bni,bi->bn", train_tangent, base_mean)
    fitted_residual = train_y - train_base
    normalized_train = _normalize_fallback_features(train_physical)

    def physics_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=tangent.device)

    def base_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqi,bi->bq", tangent, base_mean)

    def fallback_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        return base_force(tangent, physical) + _residual_predict(
            normalized_train,
            fitted_residual,
            _normalize_fallback_features(physical),
            config.residual_ridge,
        ).clamp(-0.4, 0.4)

    short_config = replace(config, horizons=config.short_horizons)
    short_data = dict(data)
    short_data["query_state"] = data["short_state"]
    physics_short, _ = _model_rollout(short_config, short_data, physics_force)
    fallback_short, _ = _model_rollout(short_config, short_data, fallback_force)
    fallback_final, _ = _model_rollout(config, data, fallback_force)
    physics_short_rmse = _episode_rmse(
        physics_short, data["short_truth"], data["short_validity"]
    )
    fallback_short_rmse = _episode_rmse(
        fallback_short, data["short_truth"], data["short_validity"]
    )
    fallback_h16_rmse = _episode_rmse(
        fallback_final[:, :, -1:],
        data["rollout_truth"][:, :, -1:],
        data["rollout_validity"][:, :, -1:],
    ).squeeze(-1)
    short_reference = torch.minimum(physics_short_rmse, fallback_short_rmse)
    validation_base = torch.einsum(
        "bni,bi->bn", data["validation_tangent"], base_mean
    )
    validation_base_mse = (
        data["validation_y"] - validation_base
    ).square().mean(dim=-1)
    diagnostic_backends = [
        SapienHingeBackend(
            density=float(density),
            damping=float(damping),
            regime=0,
            config=_backend_config(config),
        )
        for density, damping in zip(data["density"], data["damping"])
    ]

    rows: list[dict[str, Any]] = []
    signed_alphas = tuple(-value for value in config.alpha_magnitudes) + tuple(
        config.alpha_magnitudes
    )
    for parameterization in PARAMETERIZATIONS:
        for alpha in signed_alphas:
            coefficient = _candidate_coefficients(
                parameterization,
                alpha,
                base_mean,
                train_tangent,
                train_physical,
                train_y,
                config.ridge,
            )

            def candidate_force(
                tangent: torch.Tensor,
                physical: torch.Tensor,
                coefficient: torch.Tensor = coefficient,
            ) -> torch.Tensor:
                parameter_force = torch.einsum(
                    "bqi,bi->bq", tangent, coefficient[:, :2]
                )
                structural = operator_library(physical)[..., 4]
                return parameter_force + coefficient[:, 2, None] * structural

            candidate_short, candidate_short_stable = _model_rollout(
                short_config, short_data, candidate_force
            )
            candidate_final, candidate_final_stable = _model_rollout(
                config, data, candidate_force
            )
            candidate_short_rmse = _episode_rmse(
                candidate_short, data["short_truth"], data["short_validity"]
            )
            candidate_h16_rmse = _episode_rmse(
                candidate_final[:, :, -1:],
                data["rollout_truth"][:, :, -1:],
                data["rollout_validity"][:, :, -1:],
            ).squeeze(-1)
            short_gain = short_reference - candidate_short_rmse
            short_stable = candidate_short_stable.all(dim=(-1, -2))
            short_accepted = (
                (short_gain > config.rollout_margin).all(dim=-1) & short_stable
            )
            h16_gain = fallback_h16_rmse - candidate_h16_rmse
            h16_stability = candidate_final_stable[:, :, -1].float().mean(dim=-1)
            validation_candidate = torch.einsum(
                "bni,bi->bn", data["validation_tangent"], coefficient[:, :2]
            ) + coefficient[:, 2, None] * operator_library(
                data["validation_physical"]
            )[..., 4]
            force_gain = validation_base_mse - (
                data["validation_y"] - validation_candidate
            ).square().mean(dim=-1)
            delta_theta = coefficient[:, :2] - base_mean
            mahalanobis = torch.einsum(
                "bi,bij,bj->b", delta_theta, precision, delta_theta
            )
            for episode in range(episodes):
                torque = data["short_state"][episode, :, 2]
                trajectory_points = [data["short_state"][episode]]
                for horizon_index in range(len(config.short_horizons)):
                    trajectory_points.append(
                        torch.cat(
                            (
                                candidate_short[episode, :, horizon_index],
                                torque.unsqueeze(-1),
                            ),
                            dim=-1,
                        )
                    )
                diagnostic_state = torch.cat(trajectory_points, dim=0)
                power_max, violation_fraction = power_violation_metrics(
                    alpha,
                    diagnostic_state[:, 1],
                    config.power_noise_epsilon,
                )
                dynamical = _dynamical_features(
                    diagnostic_backends[episode],
                    diagnostic_state,
                    train_physical[episode],
                    coefficient[episode],
                    4,
                    config,
                )
                negative_utility = bool(h16_gain[episode] <= 0.0)
                unstable = bool(h16_stability[episode] < 1.0)
                rows.append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "parameterization": parameterization,
                        "alpha": alpha,
                        "alpha_magnitude": abs(alpha),
                        "power_direction": "dissipative" if alpha < 0 else "active",
                        "power_noise_epsilon": config.power_noise_epsilon,
                        "power_max": power_max,
                        "power_violation_fraction": violation_fraction,
                        "force_validation_gain": float(force_gain[episode]),
                        "short_rollout_gain_h2": float(short_gain[episode, 0]),
                        "short_rollout_gain_h4": float(short_gain[episode, 1]),
                        "short_rollout_gain_h8": float(short_gain[episode, 2]),
                        "short_rollout_gain_min": float(short_gain[episode].min()),
                        "short_rollout_accepted": int(short_accepted[episode]),
                        "h16_gain": float(h16_gain[episode]),
                        "h16_stability_fraction": float(h16_stability[episode]),
                        "h16_negative_utility": int(negative_utility),
                        "h16_unstable": int(unstable),
                        "h16_unsafe": int(negative_utility or unstable),
                        "mahalanobis_displacement": float(mahalanobis[episode]),
                        "parameter_l2_displacement": float(delta_theta[episode].norm()),
                        "jacobian_spectral_radius_max": dynamical[
                            "jacobian_spectral_radius_max"
                        ],
                        "negative_damping_violation": dynamical[
                            "negative_damping_violation"
                        ],
                        "support_distance_max": dynamical["support_distance_max"],
                    }
                )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for parameterization in PARAMETERIZATIONS:
        selected_parameterization = [
            row for row in rows if row["parameterization"] == parameterization
        ]
        for magnitude in sorted({float(row["alpha_magnitude"]) for row in rows}):
            for direction in ("dissipative", "active"):
                selected = [
                    row for row in selected_parameterization
                    if float(row["alpha_magnitude"]) == magnitude
                    and row["power_direction"] == direction
                ]
                output.append(
                    {
                        "parameterization": parameterization,
                        "alpha_magnitude": magnitude,
                        "power_direction": direction,
                        "n": len(selected),
                        "h16_unsafe_rate": statistics.mean(
                            int(row["h16_unsafe"]) for row in selected
                        ),
                        "h16_negative_utility_rate": statistics.mean(
                            int(row["h16_negative_utility"]) for row in selected
                        ),
                        "h16_unstable_rate": statistics.mean(
                            int(row["h16_unstable"]) for row in selected
                        ),
                        "h16_gain_mean": statistics.mean(
                            float(row["h16_gain"]) for row in selected
                        ),
                        "short_accepted_rate": statistics.mean(
                            int(row["short_rollout_accepted"]) for row in selected
                        ),
                        "short_h8_gain_mean": statistics.mean(
                            float(row["short_rollout_gain_h8"]) for row in selected
                        ),
                        "power_violation_fraction_mean": statistics.mean(
                            float(row["power_violation_fraction"]) for row in selected
                        ),
                    }
                )
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row["seed"], row["episode"], row["parameterization"],
            row["alpha_magnitude"], row["power_direction"],
        ): row
        for row in rows
    }
    output = []
    for seed, episode, parameterization, magnitude, direction in index:
        if direction != "dissipative":
            continue
        passive = index[(seed, episode, parameterization, magnitude, "dissipative")]
        active = index[(seed, episode, parameterization, magnitude, "active")]
        output.append(
            {
                "seed": seed,
                "episode": episode,
                "parameterization": parameterization,
                "alpha_magnitude": magnitude,
                "short_h8_gain_dissipative": passive["short_rollout_gain_h8"],
                "short_h8_gain_active": active["short_rollout_gain_h8"],
                "short_h8_gain_abs_gap": abs(
                    float(passive["short_rollout_gain_h8"])
                    - float(active["short_rollout_gain_h8"])
                ),
                "h16_gain_dissipative": passive["h16_gain"],
                "h16_gain_active": active["h16_gain"],
                "h16_unsafe_dissipative": passive["h16_unsafe"],
                "h16_unsafe_active": active["h16_unsafe"],
                "h16_unstable_dissipative": passive["h16_unstable"],
                "h16_unstable_active": active["h16_unstable"],
                "active_short_accepted": active["short_rollout_accepted"],
                "dissipative_short_accepted": passive["short_rollout_accepted"],
            }
        )
    return output


def run_v6r05p(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R05PConfig | None = None,
) -> dict[str, Any]:
    cfg = config or V6R05PConfig()
    device = resolve_device(device_name)
    rows = []
    for seed in cfg.seeds:
        seed_everything(seed)
        rows.extend(_evaluate_seed(cfg, seed, device))
    aggregate = _aggregate_rows(rows)
    pairs = _paired_rows(rows)
    root = Path(output_dir)
    _write_csv(root / "candidate_outcomes.csv", rows)
    _write_csv(root / "direction_summary.csv", aggregate)
    _write_csv(root / "symmetric_pairs.csv", pairs)

    diagnostics = {
        "power_violation_fraction": 1.0,
        "power_max": 1.0,
        "jacobian_spectral_radius_max": 1.0,
        "negative_damping_violation": 1.0,
        "mahalanobis_displacement": 1.0,
        "short_rollout_gain_min": -1.0,
        "force_validation_gain": -1.0,
    }
    outcomes = ("h16_unsafe", "h16_unstable", "h16_negative_utility")
    auc_rows = []
    for parameterization in PARAMETERIZATIONS:
        selected = [
            row for row in rows if row["parameterization"] == parameterization
        ]
        for outcome in outcomes:
            labels = torch.tensor([bool(row[outcome]) for row in selected])
            for diagnostic, direction in diagnostics.items():
                scores = direction * torch.tensor(
                    [float(row[diagnostic]) for row in selected]
                )
                auc_rows.append(
                    {
                        "parameterization": parameterization,
                        "outcome": outcome,
                        "diagnostic": diagnostic,
                        "n": len(selected),
                        "positive_count": int(labels.sum()),
                        "auroc": pairwise_auroc(scores, labels),
                    }
                )
    _write_csv(root / "diagnostic_auroc.csv", auc_rows)

    seed_auc_rows = []
    for seed in cfg.seeds:
        for parameterization in PARAMETERIZATIONS:
            selected = [
                row for row in rows
                if row["seed"] == seed
                and row["parameterization"] == parameterization
            ]
            for outcome in outcomes:
                labels = torch.tensor([bool(row[outcome]) for row in selected])
                for diagnostic, direction in diagnostics.items():
                    scores = direction * torch.tensor(
                        [float(row[diagnostic]) for row in selected]
                    )
                    seed_auc_rows.append(
                        {
                            "seed": seed,
                            "parameterization": parameterization,
                            "outcome": outcome,
                            "diagnostic": diagnostic,
                            "auroc": pairwise_auroc(scores, labels),
                        }
                    )
    _write_csv(root / "seed_diagnostic_auroc.csv", seed_auc_rows)

    def direction_metrics(parameterization: str, direction: str) -> dict[str, float]:
        selected = [
            row for row in rows
            if row["parameterization"] == parameterization
            and row["power_direction"] == direction
        ]
        return {
            "n": len(selected),
            "h16_unsafe_rate": statistics.mean(
                int(row["h16_unsafe"]) for row in selected
            ),
            "h16_negative_utility_rate": statistics.mean(
                int(row["h16_negative_utility"]) for row in selected
            ),
            "h16_unstable_rate": statistics.mean(
                int(row["h16_unstable"]) for row in selected
            ),
            "short_accepted_rate": statistics.mean(
                int(row["short_rollout_accepted"]) for row in selected
            ),
            "h16_gain_mean": statistics.mean(
                float(row["h16_gain"]) for row in selected
            ),
        }

    auc = {
        parameterization: {
            outcome: {
                row["diagnostic"]: row["auroc"]
                for row in auc_rows
                if row["parameterization"] == parameterization
                and row["outcome"] == outcome
            }
            for outcome in outcomes
        }
        for parameterization in PARAMETERIZATIONS
    }
    seed_power_auc = {
        parameterization: {
            outcome: [
                float(row["auroc"])
                for row in seed_auc_rows
                if row["parameterization"] == parameterization
                and row["outcome"] == outcome
                and row["diagnostic"] == "power_violation_fraction"
            ]
            for outcome in outcomes
        }
        for parameterization in PARAMETERIZATIONS
    }

    def short_direction_metrics(
        parameterization: str, direction: str
    ) -> dict[str, float | int]:
        selected = [
            row for row in rows
            if row["parameterization"] == parameterization
            and row["power_direction"] == direction
            and int(row["short_rollout_accepted"])
        ]
        return {
            "n": len(selected),
            "h16_unsafe_rate": statistics.mean(
                int(row["h16_unsafe"]) for row in selected
            ) if selected else float("nan"),
            "h16_unstable_rate": statistics.mean(
                int(row["h16_unstable"]) for row in selected
            ) if selected else float("nan"),
            "h16_negative_utility_rate": statistics.mean(
                int(row["h16_negative_utility"]) for row in selected
            ) if selected else float("nan"),
        }

    paired_metrics = {}
    for parameterization in PARAMETERIZATIONS:
        selected_pairs = [
            row for row in pairs if row["parameterization"] == parameterization
        ]
        both_short = [
            row for row in selected_pairs
            if int(row["active_short_accepted"])
            and int(row["dissipative_short_accepted"])
        ]
        gaps = [float(row["short_h8_gain_abs_gap"]) for row in selected_pairs]
        paired_metrics[parameterization] = {
            "n": len(selected_pairs),
            "short_h8_gain_gap_mean": statistics.mean(gaps),
            "short_h8_gain_gap_median": statistics.median(gaps),
            "active_unsafe_passive_safe": sum(
                int(row["h16_unsafe_active"])
                and not int(row["h16_unsafe_dissipative"])
                for row in selected_pairs
            ),
            "passive_unsafe_active_safe": sum(
                int(row["h16_unsafe_dissipative"])
                and not int(row["h16_unsafe_active"])
                for row in selected_pairs
            ),
            "active_unstable_passive_stable": sum(
                int(row["h16_unstable_active"])
                and not int(row["h16_unstable_dissipative"])
                for row in selected_pairs
            ),
            "both_short_accepted": len(both_short),
            "both_short_h8_gain_gap_mean": statistics.mean(
                float(row["short_h8_gain_abs_gap"]) for row in both_short
            ) if both_short else float("nan"),
            "both_short_h8_gain_gap_median": statistics.median(
                float(row["short_h8_gain_abs_gap"]) for row in both_short
            ) if both_short else float("nan"),
            "both_short_active_unsafe_passive_safe": sum(
                int(row["h16_unsafe_active"])
                and not int(row["h16_unsafe_dissipative"])
                for row in both_short
            ),
            "both_short_active_unstable_passive_stable": sum(
                int(row["h16_unstable_active"])
                and not int(row["h16_unstable_dissipative"])
                for row in both_short
            ),
        }
    summary = {
        "scope": "V6R0.5P operator-controlled passivity falsification",
        "primary_hypothesis": (
            "within the dissipative abs_v_v family, zero-input positive power "
            "predicts H16-unsafe revision"
        ),
        "operator": "abs_v_v",
        "alpha_magnitudes": list(cfg.alpha_magnitudes),
        "seeds": list(cfg.seeds),
        "power_noise_epsilon": cfg.power_noise_epsilon,
        "power_epsilon_source": (
            "3 * declared C0 force-noise floor * preregistered velocity support"
        ),
        "thresholds_fitted_from_h16_labels": False,
        "direction_metrics": {
            parameterization: {
                direction: direction_metrics(parameterization, direction)
                for direction in ("dissipative", "active")
            }
            for parameterization in PARAMETERIZATIONS
        },
        "diagnostic_auroc": auc,
        "seed_power_auroc": seed_power_auc,
        "short_accepted_direction_metrics": {
            parameterization: {
                direction: short_direction_metrics(parameterization, direction)
                for direction in ("dissipative", "active")
            }
            for parameterization in PARAMETERIZATIONS
        },
        "paired_metrics": paired_metrics,
        "paired_count": len(pairs),
        "active_short_accepted_count": {
            parameterization: sum(
                int(row["active_short_accepted"])
                for row in pairs if row["parameterization"] == parameterization
            )
            for parameterization in PARAMETERIZATIONS
        },
        "interpretation_boundary": (
            "operator-controlled diagnostic only; dissipativity is not asserted "
            "for conservative structural operators and no R0.6 gate is built"
        ),
        "robotwin_assets_downloaded": False,
    }
    _write_json(root / "summary.json", summary)
    return summary

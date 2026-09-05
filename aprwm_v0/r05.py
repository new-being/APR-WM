from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .r04a import V6R04AConfig, _evaluate_seed
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


# Each transform is fixed before inspecting labels and maps a raw diagnostic to
# a risk score for which larger means "more likely H16-unsafe".
RISK_SIGNALS = {
    "mahalanobis_displacement": 1.0,
    "parameter_l2_displacement": 1.0,
    "jacobian_spectral_radius_max": 1.0,
    "negative_damping_violation": 1.0,
    "positive_revision_power_max": 1.0,
    "equilibrium_force_rms": 1.0,
    "support_distance_max": 1.0,
    "operator_abs_v_v": 1.0,
    "structural_coefficient_abs": 1.0,
    "short_rollout_gain_min": -1.0,
    "force_validation_gain": -1.0,
}


@dataclass
class V6R05Config:
    selection_seeds: tuple[int, ...] = (2001, 2011, 2021, 2031, 2041)
    confirmation_seeds: tuple[int, ...] = (4001, 4011, 4021, 4031, 4041)
    episodes_per_seed: int = 24


def pairwise_auroc(scores: torch.Tensor, unsafe: torch.Tensor) -> float:
    """Tie-correct AUROC, interpreted as unsafe-versus-safe ranking."""
    scores = scores.float()
    unsafe = unsafe.bool()
    positive = scores[unsafe]
    negative = scores[~unsafe]
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(
        ((comparison > 0).float() + 0.5 * (comparison == 0).float()).mean()
    )


def _ranking_rows(
    diagnostics: list[dict[str, Any]], cohort: str, stratum: str
) -> list[dict[str, Any]]:
    selected = [row for row in diagnostics if row["cohort"] == cohort]
    if stratum == "short_accepted":
        selected = [row for row in selected if int(row["short_rollout_accepted"])]
    elif stratum != "force_accepted":
        raise ValueError(stratum)
    labels = torch.tensor([bool(int(row["h16_unsafe"])) for row in selected])
    rows = []
    for signal, direction in RISK_SIGNALS.items():
        raw = torch.tensor([float(row[signal]) for row in selected])
        risk = direction * raw
        safe_values = raw[~labels]
        unsafe_values = raw[labels]
        rows.append(
            {
                "cohort": cohort,
                "stratum": stratum,
                "signal": signal,
                "risk_direction": "higher" if direction > 0 else "lower",
                "n": len(selected),
                "safe_count": int((~labels).sum()),
                "unsafe_count": int(labels.sum()),
                "safe_mean": float(safe_values.mean()) if safe_values.numel() else float("nan"),
                "unsafe_mean": float(unsafe_values.mean()) if unsafe_values.numel() else float("nan"),
                "unsafe_auroc": pairwise_auroc(risk, labels),
            }
        )
    return rows


def run_v6r05(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R05Config | None = None,
) -> dict[str, Any]:
    cfg = config or V6R05Config()
    device = resolve_device(device_name)
    diagnostics: list[dict[str, Any]] = []
    for cohort, seeds in (
        ("selection", cfg.selection_seeds),
        ("confirmation", cfg.confirmation_seeds),
    ):
        experiment = V6R04AConfig(
            seeds=seeds,
            episodes_per_regime=cfg.episodes_per_seed,
        )
        for seed in seeds:
            seed_everything(seed)
            _, _, episode_rows = _evaluate_seed(experiment, seed, device)
            for row in episode_rows:
                diagnostics.append({"cohort": cohort, **row})

    ranking_rows = []
    for cohort in ("selection", "confirmation"):
        for stratum in ("force_accepted", "short_accepted"):
            ranking_rows.extend(_ranking_rows(diagnostics, cohort, stratum))
    pooled = [{**row, "cohort": "pooled"} for row in diagnostics]
    for stratum in ("force_accepted", "short_accepted"):
        ranking_rows.extend(_ranking_rows(pooled, "pooled", stratum))

    seed_rows = []
    for cohort, seeds in (
        ("selection", cfg.selection_seeds),
        ("confirmation", cfg.confirmation_seeds),
    ):
        for seed in seeds:
            seed_candidates = [
                row for row in diagnostics
                if row["cohort"] == cohort and int(row["seed"]) == seed
            ]
            for stratum in ("force_accepted", "short_accepted"):
                selected = seed_candidates
                if stratum == "short_accepted":
                    selected = [
                        row for row in selected
                        if int(row["short_rollout_accepted"])
                    ]
                labels = torch.tensor(
                    [bool(int(row["h16_unsafe"])) for row in selected]
                )
                for signal, direction in RISK_SIGNALS.items():
                    raw = torch.tensor([float(row[signal]) for row in selected])
                    auc = (
                        pairwise_auroc(direction * raw, labels)
                        if selected else float("nan")
                    )
                    seed_rows.append(
                        {
                            "cohort": cohort,
                            "stratum": stratum,
                            "seed": seed,
                            "signal": signal,
                            "n": len(selected),
                            "safe_count": int((~labels).sum()) if selected else 0,
                            "unsafe_count": int(labels.sum()) if selected else 0,
                            "unsafe_auroc": auc,
                            "rank_defined": int(not math.isnan(auc)),
                        }
                    )

    root = Path(output_dir)
    _write_csv(root / "revision_episode_diagnostics.csv", diagnostics)
    _write_csv(root / "diagnostic_rankings.csv", ranking_rows)
    _write_csv(root / "seed_diagnostic_rankings.csv", seed_rows)

    def cohort_counts(name: str, stratum: str) -> dict[str, int]:
        selected = [row for row in diagnostics if row["cohort"] == name]
        if stratum == "short_accepted":
            selected = [row for row in selected if int(row["short_rollout_accepted"])]
        unsafe_count = sum(int(row["h16_unsafe"]) for row in selected)
        return {
            "locally_accepted": len(selected),
            "h16_safe": len(selected) - unsafe_count,
            "h16_unsafe": unsafe_count,
        }

    def cohort_auc(name: str, stratum: str) -> dict[str, float]:
        return {
            str(row["signal"]): float(row["unsafe_auroc"])
            for row in ranking_rows
            if row["cohort"] == name and row["stratum"] == stratum
        }

    primary_stratum = "force_accepted"
    confirmation_auc = cohort_auc("confirmation", primary_stratum)
    ordered = sorted(
        confirmation_auc,
        key=lambda signal: (
            -1.0 if math.isnan(confirmation_auc[signal]) else confirmation_auc[signal]
        ),
        reverse=True,
    )
    defined_seed_auc: dict[str, list[float]] = {signal: [] for signal in RISK_SIGNALS}
    for row in seed_rows:
        if (
            row["cohort"] == "confirmation"
            and row["stratum"] == primary_stratum
            and int(row["rank_defined"])
        ):
            defined_seed_auc[str(row["signal"])].append(float(row["unsafe_auroc"]))
    summary = {
        "scope": "V6R0.5 retrospective dynamical-safety diagnostic benchmark",
        "label": "unsafe iff locally accepted candidate has non-positive H16 gain or any H16 instability",
        "thresholds_fitted": False,
        "selection_seeds": list(cfg.selection_seeds),
        "confirmation_seeds": list(cfg.confirmation_seeds),
        "counts": {
            cohort: {
                stratum: cohort_counts(cohort, stratum)
                for stratum in ("force_accepted", "short_accepted")
            }
            for cohort in ("selection", "confirmation")
        },
        "primary_ranking_stratum": primary_stratum,
        "selection_auroc": cohort_auc("selection", primary_stratum),
        "confirmation_auroc": confirmation_auc,
        "pooled_auroc": cohort_auc("pooled", primary_stratum),
        "short_accepted_auroc": {
            cohort: cohort_auc(cohort, "short_accepted")
            for cohort in ("selection", "confirmation", "pooled")
        },
        "confirmation_ranking": ordered,
        "confirmation_seed_auc_mean_where_defined": {
            signal: statistics.mean(values) if values else None
            for signal, values in defined_seed_auc.items()
        },
        "confirmation_seed_auc_defined_count": {
            signal: len(values) for signal, values in defined_seed_auc.items()
        },
        "operator_outcomes": {
            cohort: {
                operator: {
                    "force_accepted": sum(
                        1 for row in diagnostics
                        if row["cohort"] == cohort and row["operator"] == operator
                    ),
                    "h16_unsafe": sum(
                        int(row["h16_unsafe"]) for row in diagnostics
                        if row["cohort"] == cohort and row["operator"] == operator
                    ),
                    "short_accepted": sum(
                        int(row["short_rollout_accepted"]) for row in diagnostics
                        if row["cohort"] == cohort and row["operator"] == operator
                    ),
                }
                for operator in sorted(
                    {row["operator"] for row in diagnostics if row["cohort"] == cohort}
                )
            }
            for cohort in ("selection", "confirmation")
        },
        "short_accepted_failures": [
            {
                key: row[key]
                for key in (
                    "cohort", "seed", "episode", "operator", "h16_gain",
                    "h16_stability_fraction", "mahalanobis_displacement",
                    "jacobian_spectral_radius_max", "negative_damping_violation",
                    "positive_revision_power_max", "support_distance_max",
                )
            }
            for row in diagnostics
            if int(row["short_rollout_accepted"]) and int(row["h16_unsafe"])
        ],
        "interpretation_boundary": (
            "diagnostic ranking only; no threshold or R0.6 acceptance is selected"
        ),
        "robotwin_assets_downloaded": False,
    }
    _write_json(root / "summary.json", summary)
    return summary

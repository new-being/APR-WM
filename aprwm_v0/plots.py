from __future__ import annotations

import csv
import os
from pathlib import Path


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aprwm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_pareto(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "pareto.csv")
    plt = _matplotlib()
    colors = {
        "random": "#8c8c8c",
        "heuristic": "#e69f00",
        "learned": "#0072b2",
        "oracle": "#009e73",
    }
    plot_specs = (
        (
            "activated_graph_fraction",
            "rollout_allocation",
            "Activated graph-edge fraction",
            "rollout_rmse",
            "Rollout state RMSE",
        ),
        (
            "analytical_compute_fraction",
            "rollout_compute",
            "Analytical learned-compute fraction",
            "rollout_rmse",
            "Rollout state RMSE",
        ),
        (
            "activated_graph_fraction",
            "one_step_allocation",
            "Activated graph-edge fraction",
            "teacher_forced_one_step_rmse",
            "Teacher-forced one-step RMSE",
        ),
    )
    for x_key, suffix, x_label, y_key, y_label in plot_specs:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for axis, split in zip(axes, ("id", "ood")):
            for strategy in colors:
                selected = [
                    row
                    for row in rows
                    if row["split"] == split and row["strategy"] == strategy
                ]
                selected.sort(key=lambda row: float(row[x_key]))
                axis.plot(
                    [float(row[x_key]) for row in selected],
                    [float(row[y_key]) for row in selected],
                    marker="o",
                    markersize=3,
                    label=strategy,
                    color=colors[strategy],
                )
            axis.set_title(split.upper())
            axis.set_xlabel(x_label)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(y_label)
        axes[1].legend(frameon=False)
        figure.tight_layout()
        figure.savefig(root / f"pareto_{suffix}.png", dpi=180)
        plt.close(figure)


def plot_horizon(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "horizon.csv")
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, split in zip(axes, ("id", "ood")):
        models = sorted({row["model"] for row in rows})
        for model in models:
            selected = [
                row for row in rows if row["split"] == split and row["model"] == model
            ]
            selected.sort(key=lambda row: int(row["horizon"]))
            axis.plot(
                [int(row["horizon"]) for row in selected],
                [float(row["endpoint_rmse"]) for row in selected],
                marker="o",
                label=model,
            )
        axis.set_title(split.upper())
        axis.set_xlabel("Rollout horizon")
        axis.set_ylabel("Endpoint RMSE")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "horizon.png", dpi=180)
    plt.close(figure)


def plot_continuum(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "continuum.csv")
    plt = _matplotlib()
    error = [float(row["physics_edge_error"]) for row in rows]
    gate = [float(row["gate_probability"]) for row in rows]
    speed = [float(row["impact_speed"]) for row in rows]
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    scatter = axis.scatter(error, gate, c=speed, s=10, alpha=0.65, cmap="viridis")
    axis.set_xscale("symlog", linthresh=1.0e-3)
    axis.set_xlabel("Physics-only edge acceleration error")
    axis.set_ylabel("Router gate probability")
    axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axis, label="Impact speed")
    figure.tight_layout()
    figure.savefig(root / "continuum.png", dpi=180)
    plt.close(figure)


def plot_latency(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "latency.csv")
    plt = _matplotlib()
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    figure, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 4))
    if len(batch_sizes) == 1:
        axes = [axes]
    for axis, batch_size in zip(axes, batch_sizes):
        selected = [row for row in rows if int(row["batch_size"]) == batch_size]
        labels = [f'{row["model"]}-{row["mode"]}' for row in selected]
        values = [float(row["median_latency_ms"]) for row in selected]
        axis.bar(range(len(values)), values, color="#0072b2")
        axis.set_xticks(range(len(values)), labels, rotation=35, ha="right")
        axis.set_ylabel("Median CUDA latency (ms)")
        axis.set_title(f"Batch {batch_size}")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "latency.png", dpi=180)
    plt.close(figure)


def plot_v06(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "break_even.csv")
    plt = _matplotlib()
    colors = {"linear": "#e69f00", "tiny": "#0072b2", "current": "#009e73"}
    # Crossover at the central compute regime used by the core latency sweep.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, batch in zip(axes, (1, 128, 512)):
        selected_batch = [
            row
            for row in rows
            if int(row["batch_size"]) == batch
            and float(row["interaction_density"]) == 0.8
            and float(row["residual_necessity"]) == 0.1
        ]
        for router, color in colors.items():
            selected = [row for row in selected_batch if row["router"] == router]
            selected.sort(key=lambda row: float(row["expert_flops_per_edge"]))
            axis.plot(
                [float(row["expert_flops_per_edge"]) for row in selected],
                [float(row["speedup_vs_contact"]) for row in selected],
                marker="o",
                color=color,
                label=router,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlabel("Residual expert FLOPs / active edge")
        axis.set_ylabel("Speedup vs contact-conditioned residual")
        axis.set_title(f"Batch {batch}")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "expert_crossover.png", dpi=180)
    plt.close(figure)

    accuracy = _read_csv(root / "accuracy_matrix.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, router in zip(axes, colors):
        selected = [
            row
            for row in accuracy
            if row["router"] == router and row["expert"] == "8x"
        ]
        for density in (0.2, 0.5, 0.8, 1.0):
            curve = [
                row
                for row in selected
                if float(row["interaction_density"]) == density
            ]
            curve.sort(key=lambda row: float(row["residual_necessity"]))
            axis.plot(
                [float(row["residual_necessity"]) for row in curve],
                [float(row["delta_error"]) for row in curve],
                marker="o",
                label=f"density={density:g}",
            )
        axis.set_title(router)
        axis.set_xlabel("Residual necessity")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Adaptive MSE − contact-conditioned MSE")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "accuracy_regimes.png", dpi=180)
    plt.close(figure)


def plot_v07(output_dir: str | Path) -> None:
    root = Path(output_dir)
    aggregate = _read_csv(root / "evaluation_aggregate.csv")
    latency = _read_csv(root / "latency_aggregate.csv")
    plt = _matplotlib()
    strategies = ("random", "magnitude", "necessity", "utility", "oracle")
    colors = {
        "random": "#999999",
        "magnitude": "#e69f00",
        "necessity": "#0072b2",
        "utility": "#009e73",
        "oracle": "#cc79a7",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, expert in zip(axes, ("1x", "8x")):
        selected = {
            row["strategy"]: row
            for row in aggregate
            if row["expert"] == expert and row["metric"] == "utility_capture"
        }
        means = [float(selected[name]["mean"]) for name in strategies]
        low = [
            means[index] - float(selected[name]["bootstrap_95_low"])
            for index, name in enumerate(strategies)
        ]
        high = [
            float(selected[name]["bootstrap_95_high"]) - means[index]
            for index, name in enumerate(strategies)
        ]
        axis.bar(
            range(len(strategies)),
            means,
            yerr=[low, high],
            capsize=3,
            color=[colors[name] for name in strategies],
        )
        axis.set_xticks(range(len(strategies)), strategies, rotation=25, ha="right")
        axis.set_title(f"{expert} residual expert")
        axis.set_ylabel("Selected utility / oracle utility")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "utility_capture.png", dpi=180)
    plt.close(figure)

    regimes = ("slower", "crossover", "mild_speedup", "clear_speedup")
    objectives = ("magnitude", "necessity", "utility")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    width = 0.24
    for objective_index, objective in enumerate(objectives):
        selected = {
            row["regime"]: row
            for row in latency
            if row["objective"] == objective
            and row["metric"] == "speedup_vs_contact"
        }
        means = [float(selected[name]["mean"]) for name in regimes]
        low = [
            means[index] - float(selected[name]["bootstrap_95_low"])
            for index, name in enumerate(regimes)
        ]
        high = [
            float(selected[name]["bootstrap_95_high"]) - means[index]
            for index, name in enumerate(regimes)
        ]
        positions = [index + (objective_index - 1) * width for index in range(4)]
        axis.bar(
            positions,
            means,
            width,
            yerr=[low, high],
            capsize=3,
            label=objective,
            color=colors[objective],
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(4), ("B16 / 8x", "B128 / 8x", "B512 / 1x", "B512 / 8x"))
    axis.set_ylabel("Speedup vs contact-conditioned residual")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "multiseed_speedup.png", dpi=180)
    plt.close(figure)


def plot_v1(output_dir: str | Path) -> None:
    root = Path(output_dir)
    posterior = _read_csv(root / "posterior_aggregate.csv")
    routing = _read_csv(root / "routing_aggregate.csv")
    conditions = _read_csv(root / "conditions_aggregate.csv")
    plt = _matplotlib()
    colors = {"adequate": "#0072b2", "inadequate": "#d55e00"}

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    nominals = (50, 80, 90, 95)
    for condition in ("adequate", "inadequate"):
        values = []
        for nominal in nominals:
            row = next(
                item
                for item in posterior
                if item["condition"] == condition
                and item["metric"] == f"coverage_{nominal}"
            )
            values.append(100 * float(row["mean"]))
        axis.plot(
            nominals,
            values,
            marker="o",
            label=condition,
            color=colors[condition],
        )
    axis.plot(nominals, nominals, color="black", linestyle="--", label="ideal")
    axis.set_xlabel("Nominal posterior coverage (%)")
    axis.set_ylabel("Observed marginal coverage (%)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "parameter_calibration.png", dpi=180)
    plt.close(figure)

    routers = (
        "state_only",
        "posterior_mean",
        "posterior_full",
        "total_utility",
        "structural_oracle",
        "total_oracle",
    )
    metrics = {
        (row["router"], row["metric"]): row for row in routing
    }
    figure, axis = plt.subplots(figsize=(7, 4.8))
    for router in routers:
        x = float(metrics[(router, "structural_utility_capture")]["mean"])
        y = float(metrics[(router, "force_rmse")]["mean"])
        misuse = float(
            metrics[(router, "residual_misuse_rate_posterior_miss")]["mean"]
        )
        axis.scatter(x, y, s=80 + 1200 * misuse, label=router)
        axis.annotate(router, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Structural utility capture")
    axis.set_ylabel("Total force RMSE using estimated physics")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "decomposition_prediction_tradeoff.png", dpi=180)
    plt.close(figure)

    labels = (
        ("adequate", "oracle"),
        ("adequate", "estimated"),
        ("inadequate", "oracle"),
        ("inadequate", "estimated"),
    )
    values = []
    for model_class, parameter_mode in labels:
        row = next(
            item
            for item in conditions
            if item["model_class"] == model_class
            and item["parameter_mode"] == parameter_mode
            and item["metric"] == "physics_force_rmse"
        )
        values.append(float(row["mean"]))
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar(
        range(4),
        values,
        color=("#56b4e9", "#0072b2", "#e69f00", "#d55e00"),
    )
    axis.set_xticks(
        range(4),
        ("adequate\noracle θ", "adequate\nestimated θ", "inadequate\noracle θ", "inadequate\nestimated θ"),
    )
    axis.set_ylabel("Physics-only force RMSE")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "error_decomposition.png", dpi=180)
    plt.close(figure)


def plot_v2(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "strategies_aggregate.csv")
    plt = _matplotlib()
    strategies = (
        "linear_compensation",
        "passive",
        "parameter_ig",
        "joint_ig",
        "oracle_probe",
        "full_oracle",
    )
    labels = (
        "Linear compensation",
        "Passive joint belief",
        "Parameter IG",
        "Joint IG",
        "Diagnostic oracle",
        "Model-class oracle",
    )
    colors = ("#d55e00", "#999999", "#e69f00", "#009e73", "#cc79a7", "#000000")
    index = {(row["strategy"], row["metric"]): row for row in rows}

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    width = 0.36
    id_values = [float(index[(name, "id_force_rmse")]["mean"]) for name in strategies]
    intervention_values = [
        float(index[(name, "intervention_force_rmse")]["mean"])
        for name in strategies
    ]
    positions = list(range(len(strategies)))
    axis.bar([x - width / 2 for x in positions], id_values, width, label="ID")
    axis.bar(
        [x + width / 2 for x in positions],
        intervention_values,
        width,
        label="Intervention",
    )
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Force RMSE")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "compensation_brittleness.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.8))
    for strategy, label, color in zip(strategies, labels, colors):
        cost = float(index[(strategy, "total_decision_cost")]["mean"])
        error = float(index[(strategy, "intervention_force_rmse")]["mean"])
        auroc = float(index[(strategy, "model_auroc")]["mean"])
        axis.scatter(cost, error, s=70 + 180 * auroc, color=color)
        offset = {
            "joint_ig": (-58, 8),
            "oracle_probe": (5, 8),
            "full_oracle": (-132, -12),
        }.get(strategy, (5, 4))
        axis.annotate(label, (cost, error), xytext=offset, textcoords="offset points")
    axis.set_xlabel("Probe + residual decision cost")
    axis.set_ylabel("Intervention force RMSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "epistemic_cost_pareto.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    active_strategies = strategies[2:]
    active_labels = labels[2:]
    velocity = [
        float(index[(name, "velocity_probe_count")]["mean"])
        for name in active_strategies
    ]
    amplitude = [
        float(index[(name, "amplitude_probe_count")]["mean"])
        for name in active_strategies
    ]
    x_positions = range(len(active_strategies))
    axes[0].bar(x_positions, velocity, label="velocity probe", color="#0072b2")
    axes[0].bar(
        x_positions,
        amplitude,
        bottom=velocity,
        label="amplitude probe",
        color="#d55e00",
    )
    axes[0].set_xticks(x_positions, active_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Mean probe count")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    auroc_values = [float(index[(name, "model_auroc")]["mean"]) for name in strategies]
    axes[1].bar(range(len(strategies)), auroc_values, color=colors)
    axes[1].set_xticks(range(len(strategies)), labels, rotation=15, ha="right")
    axes[1].set_ylabel("Model-class AUROC")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "probe_allocation.png", dpi=180)
    plt.close(figure)


def plot_v3(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "strategies_aggregate.csv")
    plt = _matplotlib()
    strategies = (
        "linear_refit",
        "magnitude_revision",
        "tangent_revision",
        "always_expand",
        "oracle_class",
    )
    labels = ("Linear", "Magnitude", "Tangent", "Always expand", "Oracle class")
    colors = ("#999999", "#e69f00", "#009e73", "#0072b2", "#000000")
    index = {(row["strategy"], row["metric"]): row for row in rows}

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    metrics = (
        ("intervention_force_rmse", "Intervention force RMSE"),
        ("total_objective", "Prediction + resource objective"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        values = [float(index[(name, metric)]["mean"]) for name in strategies]
        axis.bar(range(len(strategies)), values, color=colors)
        axis.set_xticks(range(len(strategies)), labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "revision_performance.png", dpi=180)
    plt.close(figure)

    selected = ("magnitude_revision", "tangent_revision", "always_expand")
    selected_labels = ("Magnitude", "Tangent", "Always expand")
    metrics = (
        "false_revision_rate_adequate",
        "missed_revision_rate_cubic",
        "unknown_recall_unseen",
    )
    metric_labels = ("False revision", "Missed cubic", "Unknown recall")
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    width = 0.24
    positions = list(range(len(selected)))
    for offset, metric, label in zip((-width, 0.0, width), metrics, metric_labels):
        values = [float(index[(name, metric)]["mean"]) for name in selected]
        axis.bar([x + offset for x in positions], values, width, label=label)
    axis.set_xticks(positions, selected_labels)
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "revision_confusion.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for name, label, color in zip(strategies, labels, colors):
        complexity = float(index[(name, "model_complexity_cost")]["mean"])
        residual = float(index[(name, "residual_compute_cost")]["mean"])
        error = float(index[(name, "intervention_force_rmse")]["mean"])
        axis.scatter(complexity + residual, error, s=75, color=color)
        offset = {
            "magnitude_revision": (-62, 8),
            "always_expand": (-8, -15),
            "oracle_class": (8, 5),
        }.get(name, (5, 4))
        axis.annotate(
            label,
            (complexity + residual, error),
            xytext=offset,
            textcoords="offset points",
        )
    axis.set_xlabel("Model-complexity + residual-compute cost")
    axis.set_ylabel("Intervention force RMSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "revision_cost_pareto.png", dpi=180)
    plt.close(figure)


def plot_v4(output_dir: str | Path) -> None:
    root = Path(output_dir)
    rows = _read_csv(root / "strategies_aggregate.csv")
    plt = _matplotlib()
    strategies = (
        "residual_only",
        "passive_selection",
        "raw_active_discovery",
        "active_discovery",
        "oracle_proposal",
        "oracle_selection",
    )
    labels = (
        "Residual only",
        "Passive selection",
        "Raw active",
        "Orthogonal active",
        "Oracle proposal",
        "Oracle selection",
    )
    colors = ("#999999", "#e69f00", "#d55e00", "#009e73", "#0072b2", "#000000")
    index = {(row["strategy"], row["metric"]): row for row in rows}

    selected = strategies[1:]
    selected_labels = labels[1:]
    metrics = (
        "topk_proposal_recall",
        "selection_accuracy_given_proposal",
        "acceptance_rate_given_correct_selection",
        "exact_operator_recovery",
    )
    metric_labels = (
        "Top-k proposal recall",
        "Selection | proposal",
        "Acceptance | correct",
        "Exact recovery",
    )
    figure, axis = plt.subplots(figsize=(9, 4.7))
    width = 0.19
    positions = list(range(len(selected)))
    for offset, metric, label in zip(
        (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width),
        metrics,
        metric_labels,
    ):
        values = [float(index[(name, metric)]["mean"]) for name in selected]
        axis.bar([x + offset for x in positions], values, width, label=label)
    axis.set_xticks(positions, selected_labels, rotation=15, ha="right")
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "discovery_decomposition.png", dpi=180)
    plt.close(figure)

    class_metrics = (
        "intervention_rmse_adequate",
        "intervention_rmse_cubic",
        "intervention_rmse_coupled",
        "intervention_rmse_outside",
    )
    class_labels = ("Adequate", "Cubic", "Coupled", "Outside")
    figure, axis = plt.subplots(figsize=(10, 4.8))
    width = 0.13
    positions = list(range(len(strategies)))
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, metric, label in zip(offsets, class_metrics, class_labels):
        values = [float(index[(name, metric)]["mean"]) for name in strategies]
        axis.bar([x + offset for x in positions], values, width, label=label)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Intervention force RMSE")
    axis.legend(frameon=False, ncol=4)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "operator_intervention_rmse.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    before = [
        float(index[(name, "residual_fallback_before_discovery")]["mean"])
        for name in strategies
    ]
    after = [
        float(index[(name, "residual_fallback_after_discovery")]["mean"])
        for name in strategies
    ]
    width = 0.36
    axis.bar([x - width / 2 for x in positions], before, width, label="Before discovery")
    axis.bar([x + width / 2 for x in positions], after, width, label="After discovery")
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Residual fallback rate")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "residual_transition.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.7))
    for name, label, color in zip(strategies, labels, colors):
        cost = sum(
            float(index[(name, metric)]["mean"])
            for metric in ("model_complexity_cost", "residual_compute_cost", "probe_cost")
        )
        error = float(index[(name, "intervention_force_rmse")]["mean"])
        axis.scatter(cost, error, s=75, color=color)
        offset = {
            "raw_active_discovery": (6, 5),
            "active_discovery": (-112, 9),
            "oracle_proposal": (-108, -12),
            "oracle_selection": (6, 5),
        }.get(name, (5, 4))
        axis.annotate(label, (cost, error), xytext=offset, textcoords="offset points")
    axis.set_xlabel("Model + residual + diagnostic cost")
    axis.set_ylabel("Intervention force RMSE")
    axis.margins(x=0.15, y=0.12)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "discovery_cost_pareto.png", dpi=180)
    plt.close(figure)

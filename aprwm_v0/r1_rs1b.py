"""R1-RS1B: long-horizon admissibility on revise-worthy revisions only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import (
    ProbeSpec,
    R1RS1Config,
    _evaluate_episode,
    _mean_ci,
    _require_rs0_unlock,
    _sample_door_points,
)
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a4 import A0, FREQ_HZ, _compute_episode
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v4 import OPERATOR_NAMES


PREREG_PATH = "REPORT/R1_RS1B_PREREG.md"
FORMAL_SEEDS = (9901, 9911, 9921, 9931, 9941)
SMOKE_SEED = 9011
ALPHAS = (-0.12, -0.18, -0.24)
AMP_SCALES = (1.5, 2.0)
H32 = 32
MACRO_STEPS = 4
DAMPING_TOL = 1.0e-8
POWER_TOL = 1.0e-8


@dataclass
class R1RS1BConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    alphas: tuple[float, ...] = ALPHAS
    amp_scales: tuple[float, ...] = AMP_SCALES
    h32: int = H32
    macro_steps: int = MACRO_STEPS
    query_count: int = 8
    diagnostic_points: int = 48
    jacobian_epsilon: float = 1.0e-4
    damping_tolerance: float = DAMPING_TOL
    power_tolerance: float = POWER_TOL
    stable_rate_min: float = 0.90
    joint_limit_margin: float = 0.05
    friction: float = 0.10
    damping: float = 0.10


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_rs1a5_unlock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R1-RS1B locked until R1-RS1A.5 passes; missing summary")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("rs1a5_go"):
        raise RuntimeError("R1-RS1B locked: rs1a5_go is false")
    aggregate = payload.get("aggregate", {})
    cells = aggregate.get("cells", {})
    if not cells or not math.isfinite(float(aggregate.get("gates", {}).get("C_tol", math.nan))):
        raise RuntimeError("R1-RS1B locked: RS1A.5 summary lacks frozen intake policy")
    return payload


def _operator_value(index: int, q: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    qv = np.asarray(q, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    values = (
        qv**2,
        qv**3,
        vv**2,
        qv * vv,
        vv * np.abs(vv),
        qv * np.tanh(vv / 0.06),
        qv**2 * np.tanh(vv / 0.06),
        np.tanh((qv - 0.045) / 0.018),
    )
    return np.asarray(values[index], dtype=np.float64)


def _revision_force(alpha_hat: float, operator_index: int) -> Callable[[float, float], float]:
    def force(q: float, velocity: float) -> float:
        return float(alpha_hat * _operator_value(operator_index, q, velocity))

    return force


def _support_box(fit_physical: np.ndarray) -> dict[str, float]:
    states = np.asarray(fit_physical, dtype=np.float64)
    return {
        "q_min": float(np.min(states[:, 0])),
        "q_max": float(np.max(states[:, 0])),
        "v_min": float(np.min(states[:, 1])),
        "v_max": float(np.max(states[:, 1])),
    }


def _outside_fit_support(q: np.ndarray, v: np.ndarray, box: dict[str, float]) -> np.ndarray:
    return (
        (q < box["q_min"])
        | (q > box["q_max"])
        | (v < box["v_min"])
        | (v > box["v_max"])
    )


def _normalized_support_excursion(
    q: np.ndarray, v: np.ndarray, box: dict[str, float]
) -> float:
    q_scale = max(box["q_max"] - box["q_min"], 1.0e-9)
    v_scale = max(box["v_max"] - box["v_min"], 1.0e-9)
    q_excess = np.maximum(box["q_min"] - q, 0.0) + np.maximum(q - box["q_max"], 0.0)
    v_excess = np.maximum(box["v_min"] - v, 0.0) + np.maximum(v - box["v_max"], 0.0)
    if not len(q):
        return 0.0
    return float(np.max(np.maximum(q_excess / q_scale, v_excess / v_scale)))


def _jacobian_expansion(
    backend: DoorModeABackend,
    q: float,
    velocity: float,
    residual_fn: Callable[[float, float], float] | None,
    epsilon: float,
) -> float:
    def field(q0: float, v0: float) -> np.ndarray:
        residual = 0.0 if residual_fn is None else residual_fn(q0, v0)
        return backend.instantaneous_vector_field(
            q0, v0, 0.0, residual_force=residual, include_hidden=False
        )

    jq = (field(q + epsilon, velocity) - field(q - epsilon, velocity)) / (2.0 * epsilon)
    jv = (field(q, velocity + epsilon) - field(q, velocity - epsilon)) / (2.0 * epsilon)
    jacobian = np.column_stack((jq, jv))
    symmetric = 0.5 * (jacobian + jacobian.T)
    return float(np.linalg.eigvalsh(symmetric)[-1])


def _local_diagnostics(
    config: R1RS1BConfig,
    *,
    seed: int,
    alpha: float,
    candidate: dict[str, Any],
    probe_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    coefficient = np.asarray(candidate["coefficient"], dtype=np.float64)
    operator_index = int(candidate["operator_index"])
    alpha_hat = float(coefficient[2])
    fit = np.asarray(candidate["fit_physical"], dtype=np.float64)
    if len(fit) > config.diagnostic_points:
        indices = np.linspace(0, len(fit) - 1, config.diagnostic_points).astype(int)
        states = fit[indices]
    else:
        states = fit
    force = _revision_force(alpha_hat, operator_index)
    epsilon = config.jacobian_epsilon

    q = states[:, 0]
    v = states[:, 1]
    dr_dv = np.asarray(
        [
            (force(float(qi), float(vi + epsilon)) - force(float(qi), float(vi - epsilon)))
            / (2.0 * epsilon)
            for qi, vi in states
        ],
        dtype=np.float64,
    )
    effective_damping = -dr_dv
    revision = np.asarray([force(float(qi), float(vi)) for qi, vi in states])
    support_power = v * revision
    probe_q = np.asarray(probe_arrays["q"], dtype=np.float64)
    probe_v = np.asarray(probe_arrays["qvel"], dtype=np.float64)
    probe_revision = np.asarray(
        [force(float(qi), float(vi)) for qi, vi in zip(probe_q, probe_v)]
    )
    probe_power = probe_v * probe_revision

    backend = DoorModeABackend(
        regime="C1-H",
        friction=config.friction,
        damping=config.damping,
        seed=seed,
        alpha=alpha,
    )
    try:
        expansion = []
        for qi, vi in states:
            incumbent = _jacobian_expansion(
                backend, float(qi), float(vi), None, epsilon
            )
            revised = _jacobian_expansion(
                backend, float(qi), float(vi), force, epsilon
            )
            expansion.append(revised - incumbent)
    finally:
        backend.close()

    expansion_array = np.asarray(expansion, dtype=np.float64)
    min_damping = float(np.min(effective_damping)) if len(effective_damping) else math.nan
    power_parts = [part for part in (support_power, probe_power) if len(part)]
    max_power = (
        float(max(np.max(part) for part in power_parts)) if power_parts else math.nan
    )
    return {
        "operator": OPERATOR_NAMES[operator_index],
        "operator_index": operator_index,
        "coefficient": [float(x) for x in coefficient],
        "structural_alpha_hat": alpha_hat,
        "effective_damping_min": min_damping,
        "effective_damping_mean": float(np.mean(effective_damping)),
        "effective_damping_pass": bool(min_damping >= -config.damping_tolerance),
        "power_max": max_power,
        "positive_power_integral": float(
            np.sum(np.maximum(probe_power, 0.0)) * 0.002
        ),
        "passivity_violation": bool(max_power > config.power_tolerance),
        "expansion_excess_mean": float(np.mean(expansion_array)),
        "expansion_excess_max": float(np.max(expansion_array)),
        "fit_support": _support_box(fit),
        "n_diagnostic_states": int(len(states)),
    }


def _blind_h32(
    config: R1RS1BConfig,
    *,
    seed: int,
    alpha: float,
    candidate: dict[str, Any],
    fit_support: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coefficient = np.asarray(candidate["coefficient"], dtype=np.float64)
    operator_index = int(candidate["operator_index"])
    force = _revision_force(float(coefficient[2]), operator_index)
    generator = torch.Generator().manual_seed(
        seed + 59_000 + int(abs(alpha) * 10_000)
    )
    queries = _sample_door_points(config.query_count, "intervention", generator)
    horizon = config.h32 * config.macro_steps

    backend = DoorModeABackend(
        regime="C1-H",
        friction=config.friction,
        damping=config.damping,
        seed=seed,
        alpha=alpha,
    )
    truth_traces: list[dict[str, Any]] = []
    no_revision_traces: list[dict[str, Any]] = []
    revised_traces: list[dict[str, Any]] = []
    no_errors: list[float] = []
    revised_errors: list[float] = []
    exits: list[np.ndarray] = []
    excursions: list[float] = []
    scale = np.asarray([1.0, 2.0], dtype=np.float64)
    try:
        for state in queries:
            q0, v0, torque = [float(x) for x in state]
            truth = backend.rollout_trace(
                q0,
                v0,
                torque,
                horizon=horizon,
                margin=config.joint_limit_margin,
                include_hidden=True,
            )
            no_revision = backend.rollout_trace(
                q0,
                v0,
                torque,
                horizon=horizon,
                margin=config.joint_limit_margin,
                include_hidden=False,
            )
            revised = backend.rollout_trace(
                q0,
                v0,
                torque,
                horizon=horizon,
                residual_fn=force,
                margin=config.joint_limit_margin,
                include_hidden=False,
            )
            truth_traces.append(truth)
            no_revision_traces.append(no_revision)
            revised_traces.append(revised)
            if truth["finite"] and no_revision["finite"] and revised["finite"]:
                truth_final = np.asarray([truth["q"][-1], truth["qvel"][-1]])
                no_final = np.asarray([no_revision["q"][-1], no_revision["qvel"][-1]])
                revised_final = np.asarray([revised["q"][-1], revised["qvel"][-1]])
                no_errors.append(float(np.sqrt(np.mean(((no_final - truth_final) / scale) ** 2))))
                revised_errors.append(
                    float(np.sqrt(np.mean(((revised_final - truth_final) / scale) ** 2)))
                )
            outside = _outside_fit_support(
                revised["q"], revised["qvel"], fit_support
            )
            exits.append(outside)
            excursions.append(
                _normalized_support_excursion(
                    revised["q"], revised["qvel"], fit_support
                )
            )
    finally:
        backend.close()

    rmse_no = float(np.mean(no_errors)) if no_errors else math.nan
    rmse_revised = float(np.mean(revised_errors)) if revised_errors else math.nan
    support_samples = sum(len(x) for x in exits)
    support_exits = sum(int(np.sum(x)) for x in exits)
    stable = bool(
        len(revised_traces) == config.query_count
        and all(trace["stable"] for trace in revised_traces)
    )

    def stack(field: str, traces: list[dict[str, Any]]) -> np.ndarray:
        # An unstable rollout may terminate early. Pad rather than losing the
        # failure episode while serializing the blind evaluation artifact.
        output = np.full((len(traces), horizon), np.nan, dtype=np.float64)
        for index, trace in enumerate(traces):
            values = np.asarray(trace[field], dtype=np.float64)
            output[index, : len(values)] = values
        return output

    arrays = {
        "query_state": queries.detach().cpu().numpy(),
        "truth_q": stack("q", truth_traces),
        "truth_qvel": stack("qvel", truth_traces),
        "no_revision_q": stack("q", no_revision_traces),
        "no_revision_qvel": stack("qvel", no_revision_traces),
        "revised_q": stack("q", revised_traces),
        "revised_qvel": stack("qvel", revised_traces),
    }
    return (
        {
            "rmse_h32_no_revision": rmse_no,
            "rmse_h32_revised": rmse_revised,
            "gain_h32": (
                float(rmse_no - rmse_revised)
                if math.isfinite(rmse_no) and math.isfinite(rmse_revised)
                else math.nan
            ),
            "stable_h32": stable,
            "support_exit_fraction": (
                float(support_exits / support_samples) if support_samples else math.nan
            ),
            "support_excursion_max": float(max(excursions, default=0.0)),
            "h32_blind": True,
        },
        arrays,
    )


def _binary_auroc(scores: list[float], labels: list[int]) -> float:
    if not scores or len(set(labels)) < 2:
        return math.nan
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return float(wins / (len(positives) * len(negatives)))


def _aggregate(rows: list[dict[str, Any]], config: R1RS1BConfig) -> dict[str, Any]:
    intake = [row for row in rows if row["revise_worthy"]]
    pipeline = [row for row in intake if row.get("pipeline_accepted")]
    accepted = [row for row in intake if row.get("accepted")]
    stable_rate = (
        float(np.mean([row["stable_h32"] for row in accepted])) if accepted else 0.0
    )
    passivity_violations = int(
        sum(bool(row["passivity_violation"]) for row in accepted)
    )

    seed_gains = []
    for seed in config.seeds:
        values = [
            float(row["gain_h32"])
            for row in accepted
            if row["seed"] == seed and math.isfinite(float(row["gain_h32"]))
        ]
        if values:
            seed_gains.append(float(np.mean(values)))
    gain_stats = _mean_ci(seed_gains)

    labels = [int(not row["stable_h32"]) for row in accepted]
    expansion_scores = [float(row["expansion_excess_max"]) for row in accepted]
    expansion_auroc = _binary_auroc(expansion_scores, labels)

    gates = {
        "nontrivial_acceptance": {
            "value": len(accepted),
            "pass": len(accepted) > 0,
        },
        "passivity_violations": {
            "value": passivity_violations,
            "max": 0,
            "pass": passivity_violations == 0,
        },
        "seed_mean_h32_gain": {
            **gain_stats,
            "pass": bool(gain_stats["n"] > 0 and gain_stats["mean"] > 0.0),
        },
        "accepted_h32_stability": {
            "value": stable_rate,
            "min": config.stable_rate_min,
            "pass": stable_rate >= config.stable_rate_min,
        },
    }
    go = bool(all(item["pass"] for item in gates.values()))
    return {
        "n_episodes": len(rows),
        "n_revise_worthy": len(intake),
        "n_pipeline_accepted": len(pipeline),
        "n_accepted": len(accepted),
        "intake_rate": float(len(intake) / len(rows)) if rows else 0.0,
        "acceptance_rate_given_intake": (
            float(len(accepted) / len(intake)) if intake else 0.0
        ),
        "gates": gates,
        "expansion_instability_auroc": expansion_auroc,
        "mechanism_hit": bool(
            math.isfinite(expansion_auroc) and expansion_auroc >= 0.75
        ),
        "rs1b_go": go,
    }


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = row.pop("_probe_arrays")
    h32 = row.pop("_h32_arrays", None)
    try:
        with h5py.File(path, "w") as handle:
            handle.create_group("metadata").attrs["json"] = json.dumps(
                row, sort_keys=True, default=str
            )
            raw = handle.create_group("raw_truth")
            for key in (
                "q",
                "qvel",
                "qacc",
                "tau_command",
                "tau_hidden",
                "residual",
                "validity",
                "density_tangent",
                "r_perp",
            ):
                if key in probe:
                    raw.create_dataset(key, data=probe[key], compression="gzip")
            learner = handle.create_group("learner_visible")
            for key in ("q", "qvel", "qacc", "tau_command", "residual", "validity"):
                if key in probe:
                    learner.create_dataset(key, data=probe[key], compression="gzip")
            learner.attrs["excludes_tau_hidden"] = True
            learner.attrs["excludes_truth_qfrc_passive"] = True
            decision = handle.create_group("decision")
            decision.attrs["revise_worthy"] = bool(row["revise_worthy"])
            decision.attrs["pipeline_accepted"] = bool(row.get("pipeline_accepted", False))
            decision.attrs["accepted"] = bool(row.get("accepted", False))
            decision.attrs["h32_blind"] = True
            diagnostics = handle.create_group("diagnostics")
            for key in (
                "effective_damping_min",
                "effective_damping_mean",
                "power_max",
                "positive_power_integral",
                "expansion_excess_mean",
                "expansion_excess_max",
                "support_exit_fraction",
                "support_excursion_max",
            ):
                if key in row:
                    diagnostics.attrs[key] = row[key]
            if h32 is not None:
                evaluation = handle.create_group("evaluation")
                for key, value in h32.items():
                    evaluation.create_dataset(key, data=value, compression="gzip")
    finally:
        row["_probe_arrays"] = probe
        if h32 is not None:
            row["_h32_arrays"] = h32


def run_r1_rs1b(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS1BConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1BConfig()
    _require_rs0_unlock(Path(rs0_summary))
    policy = _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")

    policy_aggregate = policy["aggregate"]
    c_tol = float(policy_aggregate["gates"]["C_tol"])
    thresholds = {
        float(amplitude): float(cell["threshold"])
        for amplitude, cell in policy_aggregate["cells"].items()
    }
    if smoke:
        jobs = [(SMOKE_SEED, -0.24, 1.5)]
        scientific = False
    else:
        jobs = [
            (seed, alpha, amplitude)
            for seed in cfg.seeds
            for alpha in cfg.alphas
            for amplitude in cfg.amp_scales
        ]
        scientific = True

    intake_config = R1RS1AConfig()
    revision_config = R1RS1Config()
    rows: list[dict[str, Any]] = []
    for index, (seed, alpha, amplitude) in enumerate(jobs, start=1):
        print(
            f"[RS1B {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amplitude}*A0",
            flush=True,
        )
        seed_everything(seed)
        intake = _compute_episode(
            intake_config,
            seed=seed,
            alpha=float(alpha),
            amp_scale=float(amplitude),
            freq_hz=FREQ_HZ,
            a0=A0,
        )
        threshold = thresholds[float(amplitude)]
        detected = bool(intake["D0"] >= threshold)
        consequential = bool(intake["C"] >= c_tol)
        revise_worthy = bool(detected and consequential)
        row: dict[str, Any] = {
            "seed": seed,
            "alpha": float(alpha),
            "amp_scale": float(amplitude),
            "D0": float(intake["D0"]),
            "C": float(intake["C"]),
            "X_phi": float(intake["X_phi"]),
            "detect_threshold": threshold,
            "C_tol": c_tol,
            "detected": detected,
            "consequential": consequential,
            "revise_worthy": revise_worthy,
            "pipeline_executed": False,
            "pipeline_accepted": False,
            "accepted": False,
            "_probe_arrays": intake["probe_arrays"],
        }

        if revise_worthy:
            regime = "C1-L" if abs(alpha) <= 0.12 + 1.0e-12 else "C1-H"
            probe = ProbeSpec(
                name=f"RS1B-A{amplitude:.1f}",
                kind="sine",
                amplitude=A0 * float(amplitude),
                freq_hz=FREQ_HZ,
            )
            proposal = _evaluate_episode(
                revision_config,
                seed=seed,
                regime=regime,
                probe_name=probe.name,
                device=device,
                scientific=scientific,
                alpha=float(alpha),
                probe_spec=probe,
                return_candidate=True,
            )
            candidate = proposal.pop("_candidate")
            row.update(
                {
                    "pipeline_executed": True,
                    "pipeline_accepted": bool(proposal["accepted"]),
                    "triggered": bool(proposal["triggered"]),
                    "selected_operator": proposal["selected_operator"],
                    "alpha_hat": float(proposal["alpha_hat"]),
                    "short_utility": float(proposal["utility"]),
                    "proposal_force_accepted_diagnostic": bool(candidate["force_accepted"]),
                    "proposal_short_safe": bool(candidate["short_safe"]),
                }
            )
            local = _local_diagnostics(
                cfg,
                seed=seed,
                alpha=float(alpha),
                candidate=candidate,
                probe_arrays=intake["probe_arrays"],
            )
            row.update(local)
            row["accepted"] = bool(
                row["pipeline_accepted"]
                and row["effective_damping_pass"]
                and not row["passivity_violation"]
            )
            # The accepted set is frozen before this blind evaluation is called.
            if row["accepted"]:
                blind, arrays = _blind_h32(
                    cfg,
                    seed=seed,
                    alpha=float(alpha),
                    candidate=candidate,
                    fit_support=local["fit_support"],
                )
                row.update(blind)
                row["_h32_arrays"] = arrays

        rel = f"seed_{seed}/alpha_{alpha:+.2f}/A{amplitude:.1f}.hdf5"
        _write_episode_h5(root / rel, row)
        row["path"] = rel
        rows.append(row)

    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs1b_go": False,
            "n_episodes": len(rows),
            "n_revise_worthy": sum(row["revise_worthy"] for row in rows),
            "n_accepted": sum(row["accepted"] for row in rows),
            "note": "plumbing smoke only; confirmatory gates not evaluated",
        }
    )

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: row.get(key)
                for key in (
                    "seed",
                    "alpha",
                    "amp_scale",
                    "D0",
                    "C",
                    "detected",
                    "consequential",
                    "revise_worthy",
                    "triggered",
                    "selected_operator",
                    "alpha_hat",
                    "short_utility",
                    "pipeline_accepted",
                    "effective_damping_min",
                    "power_max",
                    "expansion_excess_max",
                    "accepted",
                    "gain_h32",
                    "stable_h32",
                    "support_exit_fraction",
                )
            }
        )
    _write_csv(root / "episodes.csv", csv_rows)

    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {"unlocked": True, "passed": True}
    status["R1-RS1"] = {"unlocked": True, "passed": False, "informative": True}
    status["R1-RS1A.5"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS1B"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1b_go")),
        "smoke_only": smoke,
        "population": "revise_worthy_only",
    }
    status["R1-RS1C"] = {
        "locked": not bool(aggregate.get("rs1b_go")),
        "unlocked": bool(aggregate.get("rs1b_go")),
    }

    slim_rows = []
    for row in rows:
        slim_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"_probe_arrays", "_h32_arrays"}
            }
        )
    summary = {
        "stage": "R1-RS1B",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen_intake": {
            "source": str(Path(rs1a5_summary)),
            "C_tol": c_tol,
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "population": "C >= C_tol and D0 >= frozen cell threshold",
            "A_star": 1.5,
            "sensitivity_A": 2.0,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "acceptance_order": (
            "revise_worthy -> frozen R0.6 proposal/selection/H2-H8 "
            "-> effective damping/passivity -> freeze -> blind H32"
        ),
        "episodes": slim_rows,
        "aggregate": aggregate,
        "rs1b_go": bool(aggregate.get("rs1b_go")),
        "unlocks_rs1c": bool(aggregate.get("rs1b_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    return summary

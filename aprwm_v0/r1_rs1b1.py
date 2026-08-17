"""R1-RS1B.1: calibrate modeled-support departure against H32 prediction risk."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from .r1_rs1b import (
    AMP_SCALES,
    ALPHAS,
    DAMPING_TOL,
    H32,
    MACRO_STEPS,
    POWER_TOL,
    R1RS1BConfig,
    _local_diagnostics,
    _normalized_support_excursion,
    _require_rs1a5_unlock,
    _revision_force,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1_RS1B1_PREREG.md"
FORMAL_SEEDS = (9951, 9961, 9971, 9981, 9991)
SMOKE_SEED = 9021
REGION_MIN_QUERIES = 3


@dataclass
class R1RS1B1Config:
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
    joint_limit_margin: float = 0.05
    friction: float = 0.10
    damping: float = 0.10


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_rs1b_config(config: R1RS1B1Config) -> R1RS1BConfig:
    return R1RS1BConfig(
        seeds=config.seeds,
        alphas=config.alphas,
        amp_scales=config.amp_scales,
        h32=config.h32,
        macro_steps=config.macro_steps,
        query_count=config.query_count,
        diagnostic_points=config.diagnostic_points,
        jacobian_epsilon=config.jacobian_epsilon,
        damping_tolerance=config.damping_tolerance,
        power_tolerance=config.power_tolerance,
        joint_limit_margin=config.joint_limit_margin,
        friction=config.friction,
        damping=config.damping,
    )


def modeled_distance(
    q: np.ndarray | float,
    lo: float,
    hi: float,
    margin: float,
) -> np.ndarray:
    qv = np.asarray(q, dtype=np.float64)
    width = max(hi - lo, 1.0e-9)
    interior_lo = lo + margin
    interior_hi = hi - margin
    if interior_hi <= interior_lo:
        return np.abs(qv - 0.5 * (lo + hi)) / width
    below = np.maximum(interior_lo - qv, 0.0)
    above = np.maximum(qv - interior_hi, 0.0)
    return (below + above) / width


def fit_box_distance(q: np.ndarray, v: np.ndarray, box: dict[str, float]) -> np.ndarray:
    q_scale = max(box["q_max"] - box["q_min"], 1.0e-9)
    v_scale = max(box["v_max"] - box["v_min"], 1.0e-9)
    q_excess = np.maximum(box["q_min"] - q, 0.0) + np.maximum(q - box["q_max"], 0.0)
    v_excess = np.maximum(box["v_min"] - v, 0.0) + np.maximum(v - box["v_max"], 0.0)
    return np.maximum(q_excess / q_scale, v_excess / v_scale)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ties.
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 3 or float(np.std(xv)) == 0.0 or float(np.std(yv)) == 0.0:
        return math.nan
    rx = _rankdata(xv)
    ry = _rankdata(yv)
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(rx * ry) / denom)


def _terminal_rmse(
    truth: dict[str, Any],
    other: dict[str, Any],
    scale: np.ndarray,
) -> float:
    if not (truth["finite"] and other["finite"]):
        return math.nan
    truth_final = np.asarray([truth["q"][-1], truth["qvel"][-1]])
    other_final = np.asarray([other["q"][-1], other["qvel"][-1]])
    return float(np.sqrt(np.mean(((other_final - truth_final) / scale) ** 2)))


def _blind_h32_calibrated(
    config: R1RS1B1Config,
    *,
    seed: int,
    alpha: float,
    candidate: dict[str, Any],
    fit_support: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    coefficient = np.asarray(candidate["coefficient"], dtype=np.float64)
    operator_index = int(candidate["operator_index"])
    force = _revision_force(float(coefficient[2]), operator_index)
    generator = torch.Generator().manual_seed(
        seed + 59_000 + int(abs(alpha) * 10_000)
    )
    queries = _sample_door_points(config.query_count, "intervention", generator)
    horizon = config.h32 * config.macro_steps
    scale = np.asarray([1.0, 2.0], dtype=np.float64)
    backend = DoorModeABackend(
        regime="C1-H",
        friction=config.friction,
        damping=config.damping,
        seed=seed,
        alpha=alpha,
    )
    query_rows: list[dict[str, Any]] = []
    truth_traces: list[dict[str, Any]] = []
    no_revision_traces: list[dict[str, Any]] = []
    revised_traces: list[dict[str, Any]] = []
    try:
        lo, hi = float(backend.lo), float(backend.hi)
        for index, state in enumerate(queries):
            q0, v0, torque = [float(x) for x in state]
            truth = backend.rollout_trace(
                q0, v0, torque, horizon=horizon, margin=config.joint_limit_margin, include_hidden=True
            )
            no_revision = backend.rollout_trace(
                q0, v0, torque, horizon=horizon, margin=config.joint_limit_margin, include_hidden=False
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
            d_modeled = modeled_distance(
                revised["q"], lo, hi, config.joint_limit_margin
            )
            d_fit = fit_box_distance(
                np.asarray(revised["q"], dtype=np.float64),
                np.asarray(revised["qvel"], dtype=np.float64),
                fit_support,
            )
            rmse_no = _terminal_rmse(truth, no_revision, scale)
            rmse_rev = _terminal_rmse(truth, revised, scale)
            gain = (
                float(rmse_no - rmse_rev)
                if math.isfinite(rmse_no) and math.isfinite(rmse_rev)
                else math.nan
            )
            d_exit = float(np.max(d_modeled)) if len(d_modeled) else math.nan
            i_exit = float(np.mean(d_modeled > 0.0)) if len(d_modeled) else math.nan
            query_rows.append(
                {
                    "query_index": index,
                    "finite": bool(truth["finite"] and no_revision["finite"] and revised["finite"]),
                    "binary_modeled_stable": bool(
                        revised["finite"] and all(int(c) == 0 for c in revised["support_code"])
                    ),
                    "D_exit": d_exit,
                    "I_exit": i_exit,
                    "D_exit_fit": float(np.max(d_fit)) if len(d_fit) else math.nan,
                    "I_exit_fit": float(np.mean(d_fit > 0.0)) if len(d_fit) else math.nan,
                    "rmse_h32_no_revision": rmse_no,
                    "rmse_h32_revised": rmse_rev,
                    "gain_h32": gain,
                    "harmful": bool(
                        (not (truth["finite"] and no_revision["finite"] and revised["finite"]))
                        or (math.isfinite(gain) and gain < 0.0)
                    ),
                }
            )
    finally:
        backend.close()

    def stack(field: str, traces: list[dict[str, Any]]) -> np.ndarray:
        output = np.full((len(traces), horizon), np.nan, dtype=np.float64)
        for index, trace in enumerate(traces):
            values = np.asarray(trace[field], dtype=np.float64)
            output[index, : len(values)] = values
        return output

    gains = [row["gain_h32"] for row in query_rows if math.isfinite(row["gain_h32"])]
    episode = {
        "joint_lo": lo,
        "joint_hi": hi,
        "D_exit": float(np.mean([row["D_exit"] for row in query_rows])),
        "I_exit": float(np.mean([row["I_exit"] for row in query_rows])),
        "D_exit_max": float(np.max([row["D_exit"] for row in query_rows])),
        "rmse_h32_no_revision": float(
            np.nanmean([row["rmse_h32_no_revision"] for row in query_rows])
        ),
        "rmse_h32_revised": float(
            np.nanmean([row["rmse_h32_revised"] for row in query_rows])
        ),
        "gain_h32": float(np.mean(gains)) if gains else math.nan,
        "stable_h32": bool(all(row["binary_modeled_stable"] for row in query_rows)),
        "h32_blind": True,
        "n_harmful_queries": int(sum(row["harmful"] for row in query_rows)),
    }
    arrays = {
        "query_state": queries.detach().cpu().numpy(),
        "truth_q": stack("q", truth_traces),
        "revised_q": stack("q", revised_traces),
        "revised_qvel": stack("qvel", revised_traces),
        "D_exit_query": np.asarray([row["D_exit"] for row in query_rows]),
        "I_exit_query": np.asarray([row["I_exit"] for row in query_rows]),
        "rmse_revised_query": np.asarray(
            [row["rmse_h32_revised"] for row in query_rows]
        ),
        "gain_query": np.asarray([row["gain_h32"] for row in query_rows]),
    }
    return episode, arrays, query_rows


def _d_tol(d_exit: np.ndarray, i_exit: np.ndarray) -> float:
    zeros = d_exit[i_exit <= 0.0]
    positives = d_exit[d_exit > 0.0]
    zero_med = float(np.median(zeros)) if len(zeros) else 0.0
    pos_med = float(np.median(positives)) if len(positives) else math.nan
    if not math.isfinite(pos_med):
        return math.nan
    return zero_med + 0.5 * pos_med


def _region(d_exit: float, d_tol: float) -> str:
    if not math.isfinite(d_exit):
        return "unknown"
    if d_exit <= 0.0:
        return "supported"
    if math.isfinite(d_tol) and d_exit <= d_tol:
        return "extrapolative"
    return "risky"


def _aggregate(rows: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    accepted_q = [q for q in queries if q.get("accepted")]
    if not accepted_q:
        return {
            "rs1b1_go": False,
            "n_accepted": 0,
            "n_accepted_queries": 0,
            "note": "no accepted revise-worthy revisions; calibration vacuous",
        }

    d_exit = np.asarray([q["D_exit"] for q in accepted_q], dtype=np.float64)
    i_exit = np.asarray([q["I_exit"] for q in accepted_q], dtype=np.float64)
    rmse = np.asarray([q["rmse_h32_revised"] for q in accepted_q], dtype=np.float64)
    gain = np.asarray([q["gain_h32"] for q in accepted_q], dtype=np.float64)
    d_tol = _d_tol(d_exit, i_exit)
    for query in accepted_q:
        query["region"] = _region(float(query["D_exit"]), d_tol)

    exited = [q for q in accepted_q if float(q["I_exit"]) > 0.0]
    exit_gains = [float(q["gain_h32"]) for q in exited if math.isfinite(q["gain_h32"])]
    mean_exit_gain = float(np.mean(exit_gains)) if exit_gains else math.nan
    p_positive_given_exit = (
        float(np.mean([g > 0.0 for g in exit_gains])) if exit_gains else math.nan
    )
    nonimplication = bool(
        math.isfinite(mean_exit_gain)
        and mean_exit_gain > 0.0
        and math.isfinite(p_positive_given_exit)
        and p_positive_given_exit >= 0.5
    )

    seed_rho_d = []
    seed_rho_i = []
    for seed in sorted({int(q["seed"]) for q in accepted_q}):
        qs = [q for q in accepted_q if int(q["seed"]) == seed]
        seed_rho_d.append(
            spearman([q["D_exit"] for q in qs], [q["rmse_h32_revised"] for q in qs])
        )
        seed_rho_i.append(
            spearman([q["I_exit"] for q in qs], [q["rmse_h32_revised"] for q in qs])
        )
    rho_d_stats = _mean_ci([x for x in seed_rho_d if math.isfinite(x)])
    rho_i_stats = _mean_ci([x for x in seed_rho_i if math.isfinite(x)])
    rho_d_mean = rho_d_stats["mean"]
    rho_i_mean = rho_i_stats["mean"]
    distance_assoc = bool(rho_d_stats["n"] > 0 and rho_d_mean > 0.0)
    if rho_i_stats["n"] == 0 or not math.isfinite(rho_i_mean):
        continuous_better = bool(distance_assoc)
    else:
        continuous_better = bool(
            math.isfinite(rho_d_mean) and abs(rho_d_mean) > abs(rho_i_mean)
        )

    regions = {}
    for name in ("supported", "extrapolative", "risky"):
        xs = [q for q in accepted_q if q["region"] == name]
        rmses = [float(q["rmse_h32_revised"]) for q in xs if math.isfinite(q["rmse_h32_revised"])]
        gains = [float(q["gain_h32"]) for q in xs if math.isfinite(q["gain_h32"])]
        regions[name] = {
            "n": len(xs),
            "mean_rmse_revised": float(np.mean(rmses)) if rmses else math.nan,
            "mean_gain": float(np.mean(gains)) if gains else math.nan,
        }
    region_gate_applicable = (
        regions["supported"]["n"] >= REGION_MIN_QUERIES
        and regions["risky"]["n"] >= REGION_MIN_QUERIES
        and math.isfinite(regions["supported"]["mean_rmse_revised"])
        and math.isfinite(regions["risky"]["mean_rmse_revised"])
    )
    region_pass = (not region_gate_applicable) or (
        regions["risky"]["mean_rmse_revised"] > regions["supported"]["mean_rmse_revised"]
    )

    gates = {
        "nontrivial_acceptance": {"value": len(accepted), "pass": len(accepted) > 0},
        "nonimplication": {
            "mean_gain_given_exit": mean_exit_gain,
            "p_positive_gain_given_exit": p_positive_given_exit,
            "n_exit_queries": len(exited),
            "pass": nonimplication,
        },
        "seed_mean_spearman_D_rmse": {**rho_d_stats, "pass": distance_assoc},
        "continuous_beats_binary": {
            "rho_D": rho_d_mean,
            "rho_I": rho_i_mean,
            "pass": continuous_better,
        },
        "region_rmse_risky_gt_supported": {
            "applicable": region_gate_applicable,
            "pass": region_pass,
            "supported": regions["supported"],
            "risky": regions["risky"],
        },
    }
    go = bool(
        gates["nontrivial_acceptance"]["pass"]
        and gates["nonimplication"]["pass"]
        and gates["seed_mean_spearman_D_rmse"]["pass"]
        and gates["continuous_beats_binary"]["pass"]
        and gates["region_rmse_risky_gt_supported"]["pass"]
    )
    return {
        "n_episodes": len(rows),
        "n_revise_worthy": sum(bool(row.get("revise_worthy")) for row in rows),
        "n_accepted": len(accepted),
        "n_accepted_queries": len(accepted_q),
        "D_tol": d_tol,
        "pooled_spearman_D_rmse": spearman(d_exit, rmse),
        "pooled_spearman_I_rmse": spearman(i_exit, rmse),
        "pooled_spearman_D_gain": spearman(d_exit, gain),
        "regions": regions,
        "gates": gates,
        "rs1b1_go": go,
        "installs_support_exit_hard_filter": False,
    }


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = row.pop("_probe_arrays")
    h32 = row.pop("_h32_arrays", None)
    queries = row.pop("_query_rows", [])
    try:
        with h5py.File(path, "w") as handle:
            handle.create_group("metadata").attrs["json"] = json.dumps(
                row, sort_keys=True, default=str
            )
            raw = handle.create_group("raw_truth")
            for key in ("q", "qvel", "qacc", "tau_command", "tau_hidden", "residual"):
                if key in probe:
                    raw.create_dataset(key, data=probe[key], compression="gzip")
            learner = handle.create_group("learner_visible")
            for key in ("q", "qvel", "qacc", "tau_command", "residual"):
                if key in probe:
                    learner.create_dataset(key, data=probe[key], compression="gzip")
            learner.attrs["excludes_tau_hidden"] = True
            decision = handle.create_group("decision")
            decision.attrs["accepted"] = bool(row.get("accepted"))
            decision.attrs["accept_frozen_before_h32"] = True
            evaluation = handle.create_group("evaluation")
            evaluation.attrs["h32_blind"] = True
            evaluation.create_dataset(
                "queries_json",
                data=np.bytes_(json.dumps(queries, default=str)),
            )
            if h32 is not None:
                for key, value in h32.items():
                    evaluation.create_dataset(key, data=value, compression="gzip")
    finally:
        row["_probe_arrays"] = probe
        if h32 is not None:
            row["_h32_arrays"] = h32
        row["_query_rows"] = queries


def run_r1_rs1b1(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS1B1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1B1Config()
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
    rs1b_cfg = _as_rs1b_config(cfg)
    rows: list[dict[str, Any]] = []
    query_catalog: list[dict[str, Any]] = []
    for index, (seed, alpha, amplitude) in enumerate(jobs, start=1):
        print(
            f"[RS1B.1 {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amplitude}*A0",
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
            "detect_threshold": threshold,
            "C_tol": c_tol,
            "detected": detected,
            "consequential": consequential,
            "revise_worthy": revise_worthy,
            "pipeline_executed": False,
            "pipeline_accepted": False,
            "accepted": False,
            "_probe_arrays": intake["probe_arrays"],
            "_query_rows": [],
        }
        if revise_worthy:
            regime = "C1-L" if abs(alpha) <= 0.12 + 1.0e-12 else "C1-H"
            probe = ProbeSpec(
                name=f"RS1B1-A{amplitude:.1f}",
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
                }
            )
            local = _local_diagnostics(
                rs1b_cfg,
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
            if row["accepted"]:
                blind, arrays, query_rows = _blind_h32_calibrated(
                    cfg,
                    seed=seed,
                    alpha=float(alpha),
                    candidate=candidate,
                    fit_support=local["fit_support"],
                )
                row.update(blind)
                row["_h32_arrays"] = arrays
                for query in query_rows:
                    query.update(
                        {
                            "seed": seed,
                            "alpha": float(alpha),
                            "amp_scale": float(amplitude),
                            "accepted": True,
                        }
                    )
                row["_query_rows"] = query_rows
                query_catalog.extend(query_rows)

        rel = f"seed_{seed}/alpha_{alpha:+.2f}/A{amplitude:.1f}.hdf5"
        _write_episode_h5(root / rel, row)
        row["path"] = rel
        rows.append(row)

    aggregate = (
        _aggregate(rows, query_catalog)
        if scientific
        else {
            "rs1b1_go": False,
            "n_episodes": len(rows),
            "n_accepted": sum(row["accepted"] for row in rows),
            "note": "plumbing smoke only; confirmatory gates not evaluated",
            "installs_support_exit_hard_filter": False,
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
                    "revise_worthy",
                    "accepted",
                    "D_exit",
                    "I_exit",
                    "gain_h32",
                    "rmse_h32_revised",
                    "stable_h32",
                    "n_harmful_queries",
                )
            }
        )
    _write_csv(root / "episodes.csv", csv_rows)
    if query_catalog:
        _write_csv(root / "queries.csv", query_catalog)

    status = stage_status()
    status["R1-RS1A.5"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS1B"] = {
        "unlocked": True,
        "passed": False,
        "formal": True,
        "go": False,
    }
    status["R1-RS1B.1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1b1_go")),
        "smoke_only": smoke,
        "hard_filter": False,
    }
    status["R1-RS1C"] = {"locked": True, "unlocked": False}
    status["R1-RS2"] = {"locked": True}

    slim_rows = []
    for row in rows:
        slim_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"_probe_arrays", "_h32_arrays", "_query_rows"}
            }
        )
    summary = {
        "stage": "R1-RS1B.1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen_intake": {
            "source": str(Path(rs1a5_summary)),
            "C_tol": c_tol,
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "population": "revise_worthy accepted only",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "primary_distance": "modeled_joint_interior",
        "does_not_install_support_exit_hard_filter": True,
        "episodes": slim_rows,
        "aggregate": aggregate,
        "rs1b1_go": bool(aggregate.get("rs1b1_go")),
        "unlocks_rs1c": False,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    return summary

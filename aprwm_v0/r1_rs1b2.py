"""R1-RS1B.2: targeted queries to identify support-departure bands."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import ProbeSpec, R1RS1Config, _evaluate_episode, _require_rs0_unlock
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a4 import A0, FREQ_HZ, _compute_episode
from .r1_rs1b import AMP_SCALES, ALPHAS, _local_diagnostics, _require_rs1a5_unlock, _revision_force
from .r1_rs1b1 import (
    R1RS1B1Config,
    _as_rs1b_config,
    _terminal_rmse,
    _write_episode_h5,
    fit_box_distance,
    modeled_distance,
    spearman,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1/R1_RS1B2_PREREG.md"
FORMAL_SEEDS = (10001, 10011, 10021, 10031, 10041)
SMOKE_SEED = 9031
OCCUPANCY_MIN = 12
BANDS = (
    ("B0_inside", 0.0, 0.0),
    ("B1_mild", 0.0, 0.04),
    ("B2_mid", 0.04, 0.08),
    ("B3_far", 0.08, 0.16),
)
QUERY_EXCITATION = 0.02
INTERIOR_PAD = 0.08


@dataclass
class R1RS1B2Config(R1RS1B1Config):
    seeds: tuple[int, ...] = FORMAL_SEEDS
    queries_per_band: int = 4


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def realized_band(d_exit: float) -> str:
    if not math.isfinite(d_exit) or d_exit <= 0.0:
        return "B0_inside"
    if d_exit <= 0.04:
        return "B1_mild"
    if d_exit <= 0.08:
        return "B2_mid"
    return "B3_far"


def sample_band_queries(
    *,
    lo: float,
    hi: float,
    margin: float,
    n_per_band: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[str]]:
    width = max(hi - lo, 1.0e-9)
    interior_lo = lo + margin
    interior_hi = hi - margin
    pad = INTERIOR_PAD * width
    rows: list[list[float]] = []
    labels: list[str] = []
    for name, d_lo, d_hi in BANDS:
        for index in range(n_per_band):
            lower_side = index % 2 == 0
            if name == "B0_inside":
                span = max(interior_hi - interior_lo - 2.0 * pad, 1.0e-6)
                q = interior_lo + pad + span * float(torch.rand(1, generator=generator))
                velocity = QUERY_EXCITATION * (float(torch.rand(1, generator=generator)) - 0.5)
                torque = QUERY_EXCITATION * (float(torch.rand(1, generator=generator)) - 0.5)
            else:
                d = d_lo + (d_hi - d_lo) * float(torch.rand(1, generator=generator))
                d = max(d, 1.0e-6)
                if lower_side:
                    q = interior_lo - d * width
                    velocity = -QUERY_EXCITATION * float(torch.rand(1, generator=generator))
                else:
                    q = interior_hi + d * width
                    velocity = QUERY_EXCITATION * float(torch.rand(1, generator=generator))
                q = float(np.clip(q, lo + 1.0e-4, hi - 1.0e-4))
                torque = QUERY_EXCITATION * (float(torch.rand(1, generator=generator)) - 0.5)
            rows.append([float(q), float(velocity), float(torque)])
            labels.append(name)
    return torch.tensor(rows, dtype=torch.float32), labels


def _blind_h32_banded(
    config: R1RS1B2Config,
    *,
    seed: int,
    alpha: float,
    candidate: dict[str, Any],
    fit_support: dict[str, float],
    n_per_band: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    coefficient = np.asarray(candidate["coefficient"], dtype=np.float64)
    force = _revision_force(float(coefficient[2]), int(candidate["operator_index"]))
    generator = torch.Generator().manual_seed(
        seed + 61_000 + int(abs(alpha) * 10_000)
    )
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
    try:
        lo, hi = float(backend.lo), float(backend.hi)
        queries, intended = sample_band_queries(
            lo=lo,
            hi=hi,
            margin=config.joint_limit_margin,
            n_per_band=n_per_band,
            generator=generator,
        )
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
            d_modeled = modeled_distance(revised["q"], lo, hi, config.joint_limit_margin)
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
            query_rows.append(
                {
                    "query_index": index,
                    "intended_band": intended[index],
                    "realized_band": realized_band(d_exit),
                    "finite": bool(truth["finite"] and no_revision["finite"] and revised["finite"]),
                    "binary_modeled_stable": bool(
                        revised["finite"] and all(int(c) == 0 for c in revised["support_code"])
                    ),
                    "D_exit": d_exit,
                    "I_exit": float(np.mean(d_modeled > 0.0)) if len(d_modeled) else math.nan,
                    "D_exit_fit": float(np.max(d_fit)) if len(d_fit) else math.nan,
                    "rmse_h32_no_revision": rmse_no,
                    "rmse_h32_revised": rmse_rev,
                    "gain_h32": gain,
                    "harmful": bool(
                        (not (truth["finite"] and no_revision["finite"] and revised["finite"]))
                        or (math.isfinite(gain) and gain < 0.0)
                    ),
                    "q0": q0,
                    "v0": v0,
                    "torque": torque,
                }
            )
    finally:
        backend.close()

    gains = [row["gain_h32"] for row in query_rows if math.isfinite(row["gain_h32"])]
    episode = {
        "joint_lo": lo,
        "joint_hi": hi,
        "D_exit": float(np.mean([row["D_exit"] for row in query_rows])),
        "I_exit": float(np.mean([row["I_exit"] for row in query_rows])),
        "rmse_h32_revised": float(np.nanmean([row["rmse_h32_revised"] for row in query_rows])),
        "gain_h32": float(np.mean(gains)) if gains else math.nan,
        "stable_h32": bool(all(row["binary_modeled_stable"] for row in query_rows)),
        "h32_blind": True,
        "n_harmful_queries": int(sum(row["harmful"] for row in query_rows)),
        "n_queries": len(query_rows),
    }
    arrays = {
        "query_state": queries.detach().cpu().numpy(),
        "D_exit_query": np.asarray([row["D_exit"] for row in query_rows]),
        "rmse_revised_query": np.asarray([row["rmse_h32_revised"] for row in query_rows]),
        "gain_query": np.asarray([row["gain_h32"] for row in query_rows]),
    }
    return episode, arrays, query_rows


def _band_stats(queries: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats = {}
    for name, _, _ in BANDS:
        xs = [q for q in queries if q["realized_band"] == name]
        rmses = [float(q["rmse_h32_revised"]) for q in xs if math.isfinite(q["rmse_h32_revised"])]
        gains = [float(q["gain_h32"]) for q in xs if math.isfinite(q["gain_h32"])]
        stats[name] = {
            "n": len(xs),
            "mean_rmse_revised": float(np.mean(rmses)) if rmses else math.nan,
            "mean_gain": float(np.mean(gains)) if gains else math.nan,
        }
    return stats


def _aggregate(rows: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    accepted_q = [q for q in queries if q.get("accepted")]
    bands = _band_stats(accepted_q)
    occupancy = {
        name: {"n": int(bands[name]["n"]), "min": OCCUPANCY_MIN, "pass": int(bands[name]["n"]) >= OCCUPANCY_MIN}
        for name, _, _ in BANDS
    }
    occupancy_go = bool(all(item["pass"] for item in occupancy.values()))

    exited = [q for q in accepted_q if float(q["D_exit"]) > 0.0]
    rho_exit = spearman(
        [q["D_exit"] for q in exited],
        [q["rmse_h32_revised"] for q in exited],
    )
    b1 = bands["B1_mild"]
    b3 = bands["B3_far"]
    continuous = bool(
        occupancy_go
        and math.isfinite(rho_exit)
        and rho_exit > 0.0
        and math.isfinite(float(b1["mean_rmse_revised"]))
        and math.isfinite(float(b3["mean_rmse_revised"]))
        and float(b1["mean_rmse_revised"]) < float(b3["mean_rmse_revised"])
    )
    exited_means = [
        float(bands[name]["mean_rmse_revised"])
        for name, _, _ in BANDS[1:]
        if int(bands[name]["n"]) > 0 and math.isfinite(float(bands[name]["mean_rmse_revised"]))
    ]
    jump = bool(
        occupancy_go
        and (not continuous)
        and math.isfinite(float(bands["B0_inside"]["mean_rmse_revised"]))
        and exited_means
        and float(bands["B0_inside"]["mean_rmse_revised"]) < min(exited_means)
    )
    if occupancy_go:
        mechanism = "continuous" if continuous else ("jump" if jump else "unresolved")
    elif int(bands["B1_mild"]["n"]) < OCCUPANCY_MIN:
        mechanism = "unfillable_intermediate"
    else:
        mechanism = "occupancy_fail"

    exit_gains = [float(q["gain_h32"]) for q in accepted_q if float(q["I_exit"]) > 0 and math.isfinite(q["gain_h32"])]
    confusion: dict[str, dict[str, int]] = {}
    for intended, _, _ in BANDS:
        confusion[intended] = {}
        for realized, _, _ in BANDS:
            confusion[intended][realized] = int(
                sum(
                    q["intended_band"] == intended and q["realized_band"] == realized
                    for q in accepted_q
                )
            )

    return {
        "n_episodes": len(rows),
        "n_revise_worthy": sum(bool(row.get("revise_worthy")) for row in rows),
        "n_accepted": len(accepted),
        "n_accepted_queries": len(accepted_q),
        "bands": bands,
        "occupancy": occupancy,
        "confusion_intended_to_realized": confusion,
        "spearman_D_rmse_given_exit": rho_exit,
        "nonimplication_p_positive_gain_given_exit": (
            float(np.mean([g > 0.0 for g in exit_gains])) if exit_gains else math.nan
        ),
        "mechanism": mechanism,
        "continuous_candidate": continuous,
        "jump_candidate": jump,
        "gates": {
            "occupancy": occupancy,
            "occupancy_all": occupancy_go,
        },
        "rs1b2_go": occupancy_go,
        "installs_support_exit_hard_filter": False,
    }


def run_r1_rs1b2(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS1B2Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1B2Config()
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
    n_per_band = 1 if smoke else cfg.queries_per_band
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
            f"[RS1B.2 {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amplitude}*A0",
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
                name=f"RS1B2-A{amplitude:.1f}",
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
                    "selected_operator": proposal["selected_operator"],
                    "alpha_hat": float(proposal["alpha_hat"]),
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
                blind, arrays, query_rows = _blind_h32_banded(
                    cfg,
                    seed=seed,
                    alpha=float(alpha),
                    candidate=candidate,
                    fit_support=local["fit_support"],
                    n_per_band=n_per_band,
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
            "rs1b2_go": False,
            "n_episodes": len(rows),
            "n_accepted": sum(row["accepted"] for row in rows),
            "note": "plumbing smoke only; confirmatory gates not evaluated",
            "installs_support_exit_hard_filter": False,
        }
    )
    _write_csv(
        root / "episodes.csv",
        [
            {key: row.get(key) for key in ("seed", "alpha", "amp_scale", "accepted", "D_exit", "gain_h32")}
            for row in rows
        ],
    )
    if query_catalog:
        _write_csv(root / "queries.csv", query_catalog)

    status = stage_status()
    status["R1-RS1B"] = {"passed": False, "go": False}
    status["R1-RS1B.1"] = {"passed": False, "go": False, "frozen": True}
    status["R1-RS1B.2"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1b2_go")),
        "smoke_only": smoke,
        "hard_filter": False,
    }
    status["R1-RS1C"] = {"locked": True}
    status["R1-RS2"] = {"locked": True}

    summary = {
        "stage": "R1-RS1B.2",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "does_not_install_support_exit_hard_filter": True,
        "query_design": {
            "bands": [list(item) for item in BANDS],
            "queries_per_band": n_per_band,
            "excitation": QUERY_EXCITATION,
            "occupancy_min": OCCUPANCY_MIN,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": [
            {k: v for k, v in row.items() if k not in {"_probe_arrays", "_h32_arrays", "_query_rows"}}
            for row in rows
        ],
        "aggregate": aggregate,
        "rs1b2_go": bool(aggregate.get("rs1b2_go")),
        "unlocks_rs1c": False,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    return summary

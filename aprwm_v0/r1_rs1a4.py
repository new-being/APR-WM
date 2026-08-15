"""R1-RS1A.4: detectability vs future consequence (decision-theoretic)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import _observe, _require_rs0_unlock, _sample_door_points, _torque_series
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import EPS, R1RS1AConfig, _operating_point, _recall_at_threshold
from .r1_rs1a3 import _exposure, _probe_spec, _rollout_with_tangent
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/R1_RS1A4_PREREG.md"
FORMAL_SEEDS = (9701, 9711, 9721, 9731, 9741)
SMOKE_SEED = 8991
A0 = 0.12
AMP_SCALES = (0.5, 1.0, 1.5, 2.0)
FREQ_HZ = 0.20
ALPHAS = (0.0, -0.03, -0.06, -0.09, -0.12, -0.18, -0.24)
TARGET_C0_FPR = 0.01
QUERY_COUNT = 8
HORIZON_MACRO = 32
MACRO_STEPS = 4
MISS_TOLERATED_MIN = 0.70


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_episode(
    config: R1RS1AConfig,
    *,
    seed: int,
    alpha: float,
    amp_scale: float,
    freq_hz: float,
    a0: float,
) -> dict[str, Any]:
    amp = float(a0) * float(amp_scale)
    regime = "C0" if abs(alpha) < 1e-15 else "C1-L"
    mix = int(
        hashlib.md5(f"rs1a4:{alpha}:{amp_scale}:{freq_hz}".encode()).hexdigest()[:8], 16
    )
    generator = torch.Generator().manual_seed(seed + 23_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime=regime,
        friction=config.friction,
        damping=config.damping,
        seed=seed,
        alpha=float(alpha),
    )
    try:
        state = _sample_door_points(config.passive_context, "passive", generator)
        _physical, tangent, target, _ = _observe(backend, state)
        passive_y = target + config.force_observation_noise * torch.randn(
            target.shape, generator=generator
        )
        v5 = V5Config()
        v5.ridge = config.ridge
        v5.posterior_noise_floor = config.posterior_noise_floor
        mean, _cov = _fit_base_posterior(
            tangent.unsqueeze(0),
            passive_y.unsqueeze(0),
            config.force_observation_noise,
            v5,
        )

        n_steps = int(round(config.duration_s / config.timestep))
        spec = _probe_spec(amp, freq_hz)
        torques = _torque_series(spec, n_steps, config.timestep, seed=seed + 91)
        probe = _rollout_with_tangent(backend, torques, config.joint_limit_margin)

        tang = np.stack([probe["density_tangent"], probe["qvel"]], axis=1)
        y = probe["residual"].astype(np.float64)
        rng = np.random.default_rng(seed + 77)
        y_obs = y + config.force_observation_noise * rng.normal(size=y.shape)
        tang_t = torch.tensor(tang, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(0)
        pred = torch.einsum("bni,bi->bn", tang_t, mean)
        resid = y_t - pred
        _, r_perp, _ = tangent_decomposition(tang_t, resid, config.ridge)
        r = r_perp[0].detach().cpu().numpy()
        d0 = float(np.sqrt(np.mean(r * r)))
        exp = _exposure(probe["qvel"])

        # Consequence queries depend only on (seed, alpha) — not excitation —
        # so C measures mismatch harm, not probe-path confounding.
        q_gen = torch.Generator().manual_seed(
            seed + 41_000 + int(abs(alpha) * 10_000) % 10_000
        )
        query = _sample_door_points(QUERY_COUNT, "intervention", q_gen)
        horizon = HORIZON_MACRO * MACRO_STEPS
        cs = []
        l_nom = []
        l_ora = []
        for row in query:
            out = backend.forecast_pair(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                horizon=horizon,
                margin=config.joint_limit_margin,
            )
            c = float(out["rmse_nominal"]) - float(out["rmse_oracle"])
            cs.append(c)
            l_nom.append(float(out["rmse_nominal"]))
            l_ora.append(float(out["rmse_oracle"]))

        return {
            "seed": seed,
            "alpha": float(alpha),
            "amp_scale": float(amp_scale),
            "freq_hz": float(freq_hz),
            "amplitude": amp,
            "regime": regime,
            "D0": d0,
            "C": float(np.mean(cs)),
            "L_no_rev": float(np.mean(l_nom)),
            "L_adeq": float(np.mean(l_ora)),
            **exp,
            "probe_arrays": {**probe, "r_perp": r},
        }
    finally:
        backend.close()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Per excitation cell thresholds using alpha=0 as negatives, alpha!=0 as pool?
    # Prereg: within each excitation cell, tau from alpha=0 vs each alpha separately
    # For episode-level detect, use cell tau from alpha=0 vs all alpha!=0 in that cell
    cells = []
    for amp in AMP_SCALES:
        cell_rows = [r for r in rows if abs(r["amp_scale"] - amp) < 1e-12]
        c0 = [r for r in cell_rows if abs(r["alpha"]) < 1e-15]
        pos = [r for r in cell_rows if abs(r["alpha"]) >= 1e-15]
        scores = np.asarray([r["D0"] for r in c0] + [r["D0"] for r in pos], dtype=np.float64)
        labels = np.asarray([0] * len(c0) + [1] * len(pos), dtype=np.int32)
        op = _operating_point(scores, labels, target_fpr=TARGET_C0_FPR)
        cells.append(
            {
                "amp_scale": float(amp),
                "threshold": op["threshold"],
                "c0_fpr": op["fpr"],
                "n_c0": len(c0),
                "n_pos": len(pos),
            }
        )

    def thr_for(amp: float) -> float:
        return next(c["threshold"] for c in cells if abs(c["amp_scale"] - amp) < 1e-12)

    annotated = []
    for r in rows:
        thr = thr_for(r["amp_scale"])
        detect = bool(r["D0"] >= thr) if abs(r["alpha"]) >= 1e-15 else False
        # alpha=0 never counted as detect for miss analysis
        annotated.append({**r, "threshold": thr, "detect": detect})

    c0_cs = [r["C"] for r in annotated if abs(r["alpha"]) < 1e-15]
    strong = [
        r["C"]
        for r in annotated
        if abs(r["alpha"] + 0.24) < 1e-12 and abs(r["amp_scale"] - 2.0) < 1e-12
    ]
    c_tol = float(np.median(c0_cs) + 0.5 * np.median(strong)) if c0_cs and strong else float("nan")

    nonzero = [r for r in annotated if abs(r["alpha"]) >= 1e-15]
    for r in nonzero:
        r["consequential"] = bool(r["C"] >= c_tol)
        r["policy"] = (
            "tolerate"
            if not r["consequential"]
            else ("revise_worthy" if r["detect"] else "probe")
        )

    misses = [r for r in nonzero if not r["detect"]]
    detects = [r for r in nonzero if r["detect"]]
    cons = [r for r in nonzero if r["consequential"]]
    tol = [r for r in nonzero if not r["consequential"]]

    p_miss_tolerated = (
        float(np.mean([not r["consequential"] for r in misses])) if misses else float("nan")
    )
    p_detect_cons = float(np.mean([r["detect"] for r in cons])) if cons else float("nan")
    p_detect_tol = float(np.mean([r["detect"] for r in tol])) if tol else float("nan")
    mean_c_miss = float(np.mean([r["C"] for r in misses])) if misses else float("nan")
    mean_c_det = float(np.mean([r["C"] for r in detects])) if detects else float("nan")

    # Alpha × amp summaries
    grid = []
    for alpha in ALPHAS:
        for amp in AMP_SCALES:
            xs = [
                r
                for r in annotated
                if abs(r["alpha"] - alpha) < 1e-12 and abs(r["amp_scale"] - amp) < 1e-12
            ]
            if not xs:
                continue
            grid.append(
                {
                    "alpha": float(alpha),
                    "amp_scale": float(amp),
                    "mean_D0": float(np.mean([r["D0"] for r in xs])),
                    "mean_C": float(np.mean([r["C"] for r in xs])),
                    "mean_X_phi": float(np.mean([r["X_phi"] for r in xs])),
                    "detect_rate": float(np.mean([r["detect"] for r in xs]))
                    if abs(alpha) >= 1e-15
                    else 0.0,
                    "mean_consequential": float(np.mean([r.get("consequential", False) for r in xs]))
                    if abs(alpha) >= 1e-15
                    else 0.0,
                }
            )

    from collections import Counter

    policy_counts = dict(Counter(r["policy"] for r in nonzero))

    gates = {
        "C_tol": c_tol,
        "p_miss_tolerated": p_miss_tolerated,
        "p_miss_tolerated_pass": bool(p_miss_tolerated >= MISS_TOLERATED_MIN),
        "p_detect_consequential": p_detect_cons,
        "p_detect_tolerated": p_detect_tol,
        "detect_enriched_in_consequence": bool(p_detect_cons > p_detect_tol),
        "mean_C_miss": mean_c_miss,
        "mean_C_detect": mean_c_det,
        "mean_C_miss_lt_detect": bool(mean_c_miss < mean_c_det),
        "n_miss": len(misses),
        "n_detect": len(detects),
        "n_consequential": len(cons),
        "n_tolerated": len(tol),
        "policy_counts": policy_counts,
    }
    gates["rs1a4_go"] = bool(
        gates["p_miss_tolerated_pass"]
        and gates["detect_enriched_in_consequence"]
        and gates["mean_C_miss_lt_detect"]
    )

    return {
        "cells": cells,
        "grid": grid,
        "gates": gates,
        "rs1a4_go": gates["rs1a4_go"],
        "episodes_annotated": [
            {k: v for k, v in r.items() if k != "probe_arrays"} for r in annotated
        ],
        "n_episodes": len(rows),
    }


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = row["probe_arrays"]
    meta = {k: v for k, v in row.items() if k != "probe_arrays"}
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["json"] = json.dumps(
            meta, sort_keys=True, default=str
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
            if key in arrays:
                raw.create_dataset(key, data=arrays[key], compression="gzip")
        scores = handle.create_group("scores")
        scores.attrs["D0"] = float(row["D0"])
        scores.attrs["C"] = float(row["C"])
        scores.attrs["X_phi"] = float(row["X_phi"])
        scores.attrs["revision_pipeline_executed"] = False


def run_r1_rs1a4(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    base = R1RS1AConfig()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _ = resolve_device("cpu")

    if smoke:
        base = replace(base, duration_s=2.0, passive_context=8)
        jobs = [(SMOKE_SEED, 0.0, 1.0), (SMOKE_SEED, -0.12, 1.0)]
        scientific = False
    else:
        jobs = [
            (seed, alpha, amp)
            for seed in FORMAL_SEEDS
            for alpha in ALPHAS
            for amp in AMP_SCALES
        ]
        scientific = True

    rows: list[dict[str, Any]] = []
    for index, (seed, alpha, amp) in enumerate(jobs, start=1):
        print(
            f"[RS1A.4 {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amp}*A0",
            flush=True,
        )
        seed_everything(seed)
        row = _compute_episode(
            base,
            seed=seed,
            alpha=float(alpha),
            amp_scale=float(amp),
            freq_hz=FREQ_HZ,
            a0=A0,
        )
        rel = f"seed_{seed}/alpha_{alpha:+.2f}/A{amp:.1f}.hdf5"
        _write_episode_h5(root / rel, row)
        slim = {k: v for k, v in row.items() if k != "probe_arrays"}
        slim["path"] = rel
        rows.append(slim)

    aggregate = (
        _aggregate(rows)
        if scientific
        else {"rs1a4_go": False, "n_episodes": len(rows), "note": "smoke only"}
    )

    _write_csv(
        root / "detect_consequence.csv",
        [
            {
                "seed": r["seed"],
                "alpha": r["alpha"],
                "amp_scale": r["amp_scale"],
                "D0": r["D0"],
                "C": r["C"],
                "L_no_rev": r["L_no_rev"],
                "L_adeq": r["L_adeq"],
                "X_phi": r["X_phi"],
                "rms_v": r["rms_v"],
            }
            for r in rows
        ],
    )
    if scientific:
        _write_csv(root / "grid.csv", aggregate["grid"])

    status = stage_status()
    status["R1-RS1A.3"] = {"unlocked": True, "passed": True}
    status["R1-RS1A.4"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a4_go")),
        "smoke_only": smoke,
    }
    status["R1-RS1B"] = {
        "locked": not bool(aggregate.get("rs1a4_go")),
        "population": "revise_worthy_only" if aggregate.get("rs1a4_go") else "locked",
    }
    status["R1-RS2"] = {"locked": True}

    summary = {
        "stage": "R1-RS1A.4",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "E_known": "D0",
            "E_unknown": "support_offline",
            "typed_channels": True,
            "revision_pipeline": False,
            "freq_hz": FREQ_HZ,
            "alphas": list(ALPHAS),
            "amp_scales": list(AMP_SCALES),
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1a4_go": bool(aggregate.get("rs1a4_go")),
        "unlocks_rs1b_revise_worthy": bool(aggregate.get("rs1a4_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

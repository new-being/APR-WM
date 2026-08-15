"""R1-RS1A.5: value of epistemic excitation on the probe population."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .r1_mj import stage_status
from .r1_rs1 import _require_rs0_unlock
from .r1_rs1a import R1RS1AConfig, _operating_point
from .r1_rs1a4 import (
    A0,
    FREQ_HZ,
    HORIZON_MACRO,
    MACRO_STEPS,
    QUERY_COUNT,
    TARGET_C0_FPR,
    _compute_episode,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/R1_RS1A5_PREREG.md"
FORMAL_SEEDS = (9801, 9811, 9821, 9831, 9841)
SMOKE_SEED = 9001
ALPHAS = (0.0, -0.18, -0.24)
AMP_SCALES = (0.5, 1.0, 1.5, 2.0)
BASE_AMPS = (0.5, 1.0)
LAMBDA_COST = 0.0015


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_cost(amp_scale: float) -> float:
    return float(amp_scale) ** 2


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Cell thresholds from alpha=0
    cells = {}
    for amp in AMP_SCALES:
        c0 = [
            r
            for r in rows
            if abs(r["alpha"]) < 1e-15 and abs(r["amp_scale"] - amp) < 1e-12
        ]
        pos = [
            r
            for r in rows
            if abs(r["alpha"]) >= 1e-15 and abs(r["amp_scale"] - amp) < 1e-12
        ]
        scores = np.asarray([r["D0"] for r in c0] + [r["D0"] for r in pos], dtype=np.float64)
        labels = np.asarray([0] * len(c0) + [1] * len(pos), dtype=np.int32)
        op = _operating_point(scores, labels, target_fpr=TARGET_C0_FPR)
        cells[float(amp)] = {
            "threshold": op["threshold"],
            "c0_fpr": op["fpr"],
            "n_c0": len(c0),
            "n_pos": len(pos),
        }

    def thr(amp: float) -> float:
        return float(cells[float(amp)]["threshold"])

    annotated = []
    for r in rows:
        detect = bool(r["D0"] >= thr(r["amp_scale"])) if abs(r["alpha"]) >= 1e-15 else False
        annotated.append({**r, "detect": detect, "threshold": thr(r["amp_scale"])})

    c0_cs = [r["C"] for r in annotated if abs(r["alpha"]) < 1e-15]
    strong = [
        r["C"]
        for r in annotated
        if abs(r["alpha"] + 0.24) < 1e-12 and abs(r["amp_scale"] - 2.0) < 1e-12
    ]
    c_tol = float(np.median(c0_cs) + 0.5 * np.median(strong)) if c0_cs and strong else float("nan")

    index = {
        (r["seed"], float(r["alpha"]), float(r["amp_scale"])): r for r in annotated
    }

    probe_instances = []
    voi_rows = []
    for r in annotated:
        if abs(r["alpha"]) < 1e-15:
            continue
        if r["amp_scale"] not in BASE_AMPS:
            continue
        consequential = bool(r["C"] >= c_tol)
        is_probe = consequential and (not r["detect"])
        if not is_probe:
            continue
        probe_instances.append(r)
        for a_act in AMP_SCALES:
            if a_act <= r["amp_scale"] + 1e-12:
                continue
            key = (r["seed"], float(r["alpha"]), float(a_act))
            if key not in index:
                continue
            alt = index[key]
            flip = bool(alt["detect"])
            c_act = _action_cost(a_act)
            c_base = _action_cost(r["amp_scale"])
            v = (1.0 if flip else 0.0) * float(r["C"]) - LAMBDA_COST * c_act
            v_delta = (1.0 if flip else 0.0) * float(r["C"]) - LAMBDA_COST * (
                c_act - c_base
            )
            voi_rows.append(
                {
                    "seed": r["seed"],
                    "alpha": r["alpha"],
                    "A_base": r["amp_scale"],
                    "A_act": float(a_act),
                    "C": r["C"],
                    "X_phi_base": r["X_phi"],
                    "X_phi_act": alt["X_phi"],
                    "flip": int(flip),
                    "c_act": c_act,
                    "V": v,
                    "V_delta": v_delta,
                    "D0_base": r["D0"],
                    "D0_act": alt["D0"],
                }
            )

    # Aggregate VoI by (A_base, A_act)
    pairs = []
    best_from_half = None
    for a_b in BASE_AMPS:
        best = None
        for a_a in AMP_SCALES:
            if a_a <= a_b + 1e-12:
                continue
            xs = [
                v
                for v in voi_rows
                if abs(v["A_base"] - a_b) < 1e-12 and abs(v["A_act"] - a_a) < 1e-12
            ]
            if not xs:
                continue
            seeds = sorted({v["seed"] for v in xs})
            seed_vs = []
            seed_flips = []
            seed_vd = []
            for seed in seeds:
                ys = [v for v in xs if v["seed"] == seed]
                seed_vs.append(float(np.mean([v["V"] for v in ys])))
                seed_flips.append(float(np.mean([v["flip"] for v in ys])))
                seed_vd.append(float(np.mean([v["V_delta"] for v in ys])))
            entry = {
                "A_base": float(a_b),
                "A_act": float(a_a),
                "n": len(xs),
                "seed_mean_V": float(np.mean(seed_vs)),
                "seed_mean_V_delta": float(np.mean(seed_vd)),
                "seed_mean_flip": float(np.mean(seed_flips)),
                "seed_Vs": seed_vs,
            }
            pairs.append(entry)
            if best is None or entry["seed_mean_V"] > best["seed_mean_V"]:
                best = entry
        if abs(a_b - 0.5) < 1e-12:
            best_from_half = best

    gates = {
        "C_tol": c_tol,
        "lambda": LAMBDA_COST,
        "n_probe_instances": len(probe_instances),
        "n_voi_evals": len(voi_rows),
        "exists_positive_V_from_0.5A": bool(
            any(p["seed_mean_V"] > 0 and abs(p["A_base"] - 0.5) < 1e-12 for p in pairs)
        ),
        "A_star_from_0.5A": (
            None
            if best_from_half is None
            else {
                "A_act": best_from_half["A_act"],
                "seed_mean_V": best_from_half["seed_mean_V"],
                "seed_mean_flip": best_from_half["seed_mean_flip"],
            }
        ),
        "A_star_positive": bool(
            best_from_half is not None and best_from_half["seed_mean_V"] > 0
        ),
        "A_star_is_max_A": bool(
            best_from_half is not None and abs(best_from_half["A_act"] - 2.0) < 1e-12
        ),
    }
    gates["rs1a5_go"] = bool(
        gates["exists_positive_V_from_0.5A"] and gates["A_star_positive"]
    )

    return {
        "cells": {str(k): v for k, v in cells.items()},
        "pairs": pairs,
        "voi_rows": voi_rows,
        "gates": gates,
        "rs1a5_go": gates["rs1a5_go"],
        "n_episodes": len(rows),
        "policy_freeze": {
            "tolerate": "C < C_tol",
            "probe": "C >= C_tol and detect=0",
            "revise_worthy": "C >= C_tol and detect=1",
            "RS1B_population": "revise_worthy_only",
        },
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


def run_r1_rs1a5(
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
        jobs = [
            (SMOKE_SEED, 0.0, 0.5),
            (SMOKE_SEED, 0.0, 2.0),
            (SMOKE_SEED, -0.24, 0.5),
            (SMOKE_SEED, -0.24, 2.0),
        ]
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
            f"[RS1A.5 {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amp}*A0",
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
        else {"rs1a5_go": False, "n_episodes": len(rows), "note": "smoke only"}
    )

    _write_csv(
        root / "episodes.csv",
        [
            {
                "seed": r["seed"],
                "alpha": r["alpha"],
                "amp_scale": r["amp_scale"],
                "D0": r["D0"],
                "C": r["C"],
                "X_phi": r["X_phi"],
            }
            for r in rows
        ],
    )
    if scientific:
        _write_csv(root / "voi_pairs.csv", aggregate["voi_rows"])
        _write_csv(root / "voi_summary.csv", aggregate["pairs"])

    status = stage_status()
    status["R1-RS1A.4"] = {"unlocked": True, "passed": False, "informative": True}
    status["R1-RS1A.5"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a5_go")),
        "smoke_only": smoke,
    }
    status["R1-RS1B"] = {
        "locked": True,
        "population": "revise_worthy_only",
        "note": "locked until explicitly opened after VoI policy",
    }

    summary = {
        "stage": "R1-RS1A.5",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "lambda": LAMBDA_COST,
            "c_A": "(A/A0)^2",
            "E_known": "D0",
            "policy_populations": "tolerate/probe/revise_worthy",
            "RS1B_population": "revise_worthy_only",
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1a5_go": bool(aggregate.get("rs1a5_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

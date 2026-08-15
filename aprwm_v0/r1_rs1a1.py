"""R1-RS1A.1: compositional inadequacy evidence (D3' / Dg)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .r1_mj import stage_status
from .r1_rs1 import _require_rs0_unlock
from .r1_rs1a import (
    ALL_REGIMES,
    EPS,
    PRIMARY_REGIMES,
    PROBES,
    R1RS1AConfig,
    _auprc,
    _auroc,
    _compute_detector_scores,
    _operating_point,
    _recall_at_threshold,
    _task_mask,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/R1_RS1A1_PREREG.md"
DEV_SEEDS = (9301, 9311, 9321)
FORMAL_SEEDS = (9401, 9411, 9421, 9431, 9441)
SMOKE_SEED = 8961
GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
D3_REFERENCE_GAMMA = 1.0
AUROC_DELTA = 0.02
TARGET_C0_FPR = 0.01
PRIMARY_DETECTORS = ("D0", "D1", "D3", "D3p", "Dg")
SECONDARY_DETECTORS = ("D2",)


@dataclass
class R1RS1A1Config:
    base: R1RS1AConfig | None = None
    gamma_grid: tuple[float, ...] = GAMMA_GRID
    auroc_delta: float = AUROC_DELTA
    target_c0_fpr: float = TARGET_C0_FPR
    d3_reference_gamma: float = D3_REFERENCE_GAMMA
    dev_seeds: tuple[int, ...] = DEV_SEEDS
    formal_seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = ALL_REGIMES
    probes: tuple[str, ...] = PROBES

    def resolved_base(self) -> R1RS1AConfig:
        return self.base or R1RS1AConfig()


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compose_scores(row: dict[str, Any], gamma: float, d3_ref_gamma: float) -> dict[str, float]:
    d0 = float(row["scores"]["D0"])
    d1 = float(row["scores"]["D1"])
    d2 = float(row["scores"]["D2"])
    s_u = float(row["support_term"])
    u = float(row["u_unsupported"])
    d3 = d2 + float(d3_ref_gamma) * s_u
    d3p = d1 + float(gamma) * s_u
    dg = d1 if u <= 0.0 else d1 + float(gamma) * s_u
    return {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D3p": d3p, "Dg": dg}


def _collect_rows(
    cfg: R1RS1AConfig,
    *,
    seeds: tuple[int, ...],
    regimes: tuple[str, ...],
    probes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jobs = [
        (seed, regime, probe)
        for seed in seeds
        for regime in regimes
        for probe in probes
    ]
    for index, (seed, regime, probe) in enumerate(jobs, start=1):
        print(
            f"[RS1A.1 collect {index}/{len(jobs)}] "
            f"seed={seed} regime={regime} probe={probe}",
            flush=True,
        )
        seed_everything(seed)
        row = _compute_detector_scores(
            cfg, seed=seed, regime=regime, probe_name=probe
        )
        slim = {k: v for k, v in row.items() if k != "probe_arrays"}
        slim["probe_arrays"] = row["probe_arrays"]
        rows.append(slim)
    return rows


def _regime_recalls_at_op(
    rows: list[dict[str, Any]],
    *,
    detector: str,
    gamma: float,
    d3_ref_gamma: float,
    target_fpr: float,
) -> dict[str, float]:
    primary = [r for r in rows if r["regime"] in PRIMARY_REGIMES]
    composed = [_compose_scores(r, gamma, d3_ref_gamma) for r in primary]
    # Operating point on pooled structural vs C0
    idx, labels = _task_mask(primary, "pooled_structural")
    scores = np.asarray([composed[i][detector] for i in idx], dtype=np.float64)
    op = _operating_point(scores, labels, target_fpr=target_fpr)
    thr = op["threshold"]

    def recall(regime: str) -> float:
        xs = [
            composed[i][detector]
            for i, r in enumerate(primary)
            if r["regime"] == regime
        ]
        if not xs:
            return float("nan")
        return _recall_at_threshold(
            np.asarray(xs), np.ones(len(xs), dtype=np.int32), thr
        )

    return {
        "threshold": thr,
        "c0_fpr": op["fpr"],
        "recall_c1_l": recall("C1-L"),
        "recall_c1_h": recall("C1-H"),
        "recall_c2": recall("C2-latch"),
        "recall_pooled": op["tpr"],
    }


def _select_gamma(
    rows: list[dict[str, Any]], config: R1RS1A1Config
) -> dict[str, Any]:
    grid_rows = []
    best = None
    for gamma in config.gamma_grid:
        op = _regime_recalls_at_op(
            rows,
            detector="D3p",
            gamma=float(gamma),
            d3_ref_gamma=config.d3_reference_gamma,
            target_fpr=config.target_c0_fpr,
        )
        score = min(op["recall_c1_l"], op["recall_c2"])
        feasible = op["c0_fpr"] <= config.target_c0_fpr + 1e-12
        entry = {
            "gamma": float(gamma),
            "objective_min_recall": float(score),
            "feasible": bool(feasible),
            **op,
        }
        grid_rows.append(entry)
        if not feasible:
            continue
        if best is None or entry["objective_min_recall"] > best["objective_min_recall"]:
            best = entry
        elif (
            best is not None
            and entry["objective_min_recall"] == best["objective_min_recall"]
            and entry["gamma"] < best["gamma"]
        ):
            # tie-break: smaller gamma (less support weight)
            best = entry
    if best is None:
        # fall back to gamma maximizing objective ignoring feasibility note
        best = max(grid_rows, key=lambda e: e["objective_min_recall"])
        best = {**best, "fallback_no_feasible": True}
    return {"selected": best, "grid": grid_rows}


def _evaluate_detectors(
    rows: list[dict[str, Any]],
    *,
    gamma: float,
    config: R1RS1A1Config,
) -> dict[str, Any]:
    primary = [r for r in rows if r["regime"] in PRIMARY_REGIMES]
    composed_rows = []
    for r in primary:
        scores = _compose_scores(r, gamma, config.d3_reference_gamma)
        composed_rows.append({**r, "scores": scores})

    tasks = ("known_structural", "outside_library", "pooled_structural")
    metrics: dict[str, Any] = {"tasks": {}, "operating_points": {}}
    for task in tasks:
        idx, labels = _task_mask(composed_rows, task)
        task_rows = [composed_rows[i] for i in idx]
        metrics["tasks"][task] = {}
        for det in PRIMARY_DETECTORS + SECONDARY_DETECTORS:
            scores = np.asarray([r["scores"][det] for r in task_rows], dtype=np.float64)
            metrics["tasks"][task][det] = {
                "auroc": _auroc(scores, labels),
                "auprc": _auprc(scores, labels),
                "n": int(len(labels)),
                "n_pos": int(labels.sum()),
                "n_neg": int((1 - labels).sum()),
            }

    for det in PRIMARY_DETECTORS + SECONDARY_DETECTORS:
        metrics["operating_points"][det] = _regime_recalls_at_op(
            primary,
            detector=det,
            gamma=gamma,
            d3_ref_gamma=config.d3_reference_gamma,
            target_fpr=config.target_c0_fpr,
        )

    op0 = metrics["operating_points"]["D0"]
    op3p = metrics["operating_points"]["D3p"]
    auroc_c1_d0 = metrics["tasks"]["known_structural"]["D0"]["auroc"]
    auroc_c1_d3p = metrics["tasks"]["known_structural"]["D3p"]["auroc"]
    gates = {
        "c1l_recall_d3p_gt_d0": bool(op3p["recall_c1_l"] > op0["recall_c1_l"]),
        "c2_recall_d3p_gt_d0": bool(op3p["recall_c2"] > op0["recall_c2"]),
        "c1_auroc_non_degradation": bool(
            auroc_c1_d3p >= auroc_c1_d0 - config.auroc_delta
        ),
        "recall_c1_l_d0": op0["recall_c1_l"],
        "recall_c1_l_d3p": op3p["recall_c1_l"],
        "recall_c2_d0": op0["recall_c2"],
        "recall_c2_d3p": op3p["recall_c2"],
        "auroc_c1_d0": auroc_c1_d0,
        "auroc_c1_d3p": auroc_c1_d3p,
        "auroc_delta": config.auroc_delta,
        "gamma": gamma,
    }
    gates["rs1a1_go"] = bool(
        gates["c1l_recall_d3p_gt_d0"]
        and gates["c2_recall_d3p_gt_d0"]
        and gates["c1_auroc_non_degradation"]
    )
    metrics["gates"] = gates
    metrics["rs1a1_go"] = gates["rs1a1_go"]

    # Dg vs D3p diagnostic: identical when non-C2 have u=0
    diffs = []
    for r in composed_rows:
        diffs.append(abs(r["scores"]["Dg"] - r["scores"]["D3p"]))
    metrics["dg_vs_d3p"] = {
        "max_abs_diff": float(max(diffs)) if diffs else float("nan"),
        "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "note": "identical when u=0 on all non-support regimes",
    }
    return metrics


def _write_episode_h5(
    path: Path, row: dict[str, Any], scores: dict[str, float], gamma: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = row["probe_arrays"]
    meta = {
        k: v
        for k, v in row.items()
        if k not in {"probe_arrays", "scores"}
    }
    meta["scores"] = scores
    meta["gamma"] = gamma
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["json"] = json.dumps(
            meta, sort_keys=True, default=str
        )
        raw = handle.create_group("raw_truth")
        learner = handle.create_group("learner_visible")
        for key in (
            "q",
            "qvel",
            "qacc",
            "tau_command",
            "tau_hidden",
            "residual",
            "validity",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        for key in ("q", "qvel", "qacc", "tau_command", "residual", "validity"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_tau_hidden"] = True
        group = handle.create_group("detector_scores")
        for det, value in scores.items():
            group.attrs[det] = float(value)
        group.attrs["gamma"] = float(gamma)
        group.attrs["revision_pipeline_executed"] = False


def run_r1_rs1a1(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    config: R1RS1A1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1A1Config()
    base = cfg.resolved_base()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _ = resolve_device("cpu")

    if smoke:
        smoke_base = replace(base, duration_s=2.0, discovery_count=8)
        rows = _collect_rows(
            smoke_base,
            seeds=(SMOKE_SEED,),
            regimes=ALL_REGIMES,
            probes=("P1",),
        )
        # smoke uses gamma=1 placeholder; no selection
        gamma = 1.0
        gamma_selection = {
            "note": "smoke only; gamma not selected",
            "selected": {"gamma": 1.0},
        }
        scientific = False
        aggregate = {
            "rs1a1_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only",
        }
        phase_rows = {"smoke": rows}
    else:
        scientific = True
        # ---- Stage 1: dev gamma selection ----
        print("=== RS1A.1 Stage 1: dev gamma selection ===", flush=True)
        dev_rows = _collect_rows(
            base,
            seeds=cfg.dev_seeds,
            regimes=PRIMARY_REGIMES,  # no CNEG on selection
            probes=cfg.probes,
        )
        gamma_selection = _select_gamma(dev_rows, cfg)
        gamma = float(gamma_selection["selected"]["gamma"])
        print(f"Selected gamma={gamma}", flush=True)
        _write_json(root / "gamma_selection.json", gamma_selection)

        # ---- Stage 2: formal held-out ----
        print("=== RS1A.1 Stage 2: formal held-out ===", flush=True)
        formal_rows = _collect_rows(
            base,
            seeds=cfg.formal_seeds,
            regimes=cfg.regimes,
            probes=cfg.probes,
        )
        aggregate = _evaluate_detectors(formal_rows, gamma=gamma, config=cfg)
        aggregate["gamma_selection"] = gamma_selection
        phase_rows = {"dev": dev_rows, "formal": formal_rows}
        rows = formal_rows

    # Write formal/smoke episodes with composed scores
    csv_rows = []
    out_split = "smoke" if smoke else "formal"
    for r in rows:
        scores = _compose_scores(r, gamma, cfg.d3_reference_gamma)
        rel = f"{out_split}/seed_{r['seed']}/{r['regime']}/{r['probe']}.hdf5"
        _write_episode_h5(root / rel, r, scores, gamma)
        csv_rows.append(
            {
                "seed": r["seed"],
                "regime": r["regime"],
                "probe": r["probe"],
                "gamma": gamma,
                "u_unsupported": r["u_unsupported"],
                "support_term": r["support_term"],
                **scores,
            }
        )
    _write_csv(root / f"detector_scores_{out_split}.csv", csv_rows)

    if not smoke and "dev" in phase_rows:
        dev_csv = []
        for r in phase_rows["dev"]:
            scores = _compose_scores(r, gamma, cfg.d3_reference_gamma)
            rel = f"dev/seed_{r['seed']}/{r['regime']}/{r['probe']}.hdf5"
            _write_episode_h5(root / rel, r, scores, gamma)
            dev_csv.append(
                {
                    "seed": r["seed"],
                    "regime": r["regime"],
                    "probe": r["probe"],
                    "gamma": gamma,
                    "u_unsupported": r["u_unsupported"],
                    **scores,
                }
            )
        _write_csv(root / "detector_scores_dev.csv", dev_csv)

    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {"unlocked": True, "passed": True}
    status["R1-RS1"] = {"unlocked": True, "passed": False}
    status["R1-RS1A"] = {"unlocked": True, "passed": False, "informative": True}
    status["R1-RS1A.1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a1_go")),
        "smoke_only": smoke,
        "gamma": gamma,
    }
    status["R1-RS1B"] = {
        "locked_until_RS1A1": True,
        "unlocked": bool(aggregate.get("rs1a1_go")),
    }
    status["R1-RS2"] = {"locked": True}

    summary = {
        "stage": "R1-RS1A.1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen": {
            "revision_pipeline": False,
            "gamma_grid": list(cfg.gamma_grid),
            "auroc_delta": cfg.auroc_delta,
            "d3_reference_gamma": cfg.d3_reference_gamma,
            "selected_gamma": gamma,
        },
        "gamma_selection": gamma_selection,
        "aggregate": aggregate,
        "rs1a1_go": bool(aggregate.get("rs1a1_go")),
        "unlocks_rs1b": bool(aggregate.get("rs1a1_go")),
        "stage_status": status,
        "output": str(root.resolve()),
        "n_episodes_written": len(rows),
    }
    _write_json(root / "summary.json", summary)
    return summary

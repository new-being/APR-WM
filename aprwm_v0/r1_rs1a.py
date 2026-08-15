"""R1-RS1A: multi-evidence model-inadequacy detection (trigger-only)."""

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
from .r1_rs import RS0_NOMINAL_DAMPING, RS0_NOMINAL_FRICTION
from .r1_rs1 import (
    PROBE_BANK,
    PROBES,
    _observe,
    _prereg_sha256 as _rs1_prereg_sha256,
    _require_rs0_unlock,
    _sample_door_points,
    _torque_series,
)
from .r1_rs1_backend import DoorModeABackend
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/R1_RS1A_PREREG.md"
PRIMARY_REGIMES = ("C0", "C1-L", "C1-H", "C2-latch")
DIAGNOSTIC_REGIMES = ("CNEG",)
ALL_REGIMES = PRIMARY_REGIMES + DIAGNOSTIC_REGIMES
FORMAL_SEEDS = (9201, 9211, 9221, 9231, 9241)
SMOKE_SEED = 8951
DETECTORS = ("D0", "D1", "D2", "D3")
EPS = 1.0e-8
SUPPORT_GAMMA = 1.0
SUPPORT_AUDIT_COUNT = 32
SUPPORT_AUDIT_MARGIN = 0.05
TARGET_C0_FPR = 0.01


@dataclass
class R1RS1AConfig:
    passive_context: int = 8
    discovery_count: int = 16
    force_observation_noise: float = 0.002
    posterior_noise_floor: float = 0.01
    ridge: float = 1.0e-5
    support_gamma: float = SUPPORT_GAMMA
    support_audit_count: int = SUPPORT_AUDIT_COUNT
    support_audit_margin: float = SUPPORT_AUDIT_MARGIN
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = ALL_REGIMES
    probes: tuple[str, ...] = PROBES
    duration_s: float = 10.0
    timestep: float = 0.002
    friction: float = RS0_NOMINAL_FRICTION
    damping: float = RS0_NOMINAL_DAMPING
    joint_limit_margin: float = 0.05


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return correct / (len(pos) * len(neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(-scores)
    y = labels[order]
    tp = 0
    fp = 0
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    precs = []
    recalls = []
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precs.append(tp / max(tp + fp, 1))
        recalls.append(tp / n_pos)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precs, recalls):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    thresholds = np.unique(scores)
    cand = np.concatenate(
        ([float(scores.max()) + 1.0], thresholds[::-1], [float(scores.min()) - 1.0])
    )
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((1 - labels).sum()), 1)
    curve = []
    for thr in cand:
        pred = scores >= thr
        tpr = float(np.sum(pred & (labels == 1)) / n_pos)
        fpr = float(np.sum(pred & (labels == 0)) / n_neg)
        curve.append({"threshold": float(thr), "tpr": tpr, "fpr": fpr})
    return curve


def _operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_fpr: float,
) -> dict[str, float]:
    curve = _roc_curve(scores, labels)
    feasible = [p for p in curve if p["fpr"] <= target_fpr + 1e-12]
    if not feasible:
        best = min(curve, key=lambda p: (p["fpr"], -p["tpr"]))
    else:
        best = max(feasible, key=lambda p: p["tpr"])
    return {
        "threshold": best["threshold"],
        "fpr": best["fpr"],
        "tpr": best["tpr"],
        "target_fpr": target_fpr,
    }


def _recall_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    mask = labels == 1
    if not mask.any():
        return float("nan")
    return float(np.mean(scores[mask] >= threshold))


def _compute_detector_scores(
    config: R1RS1AConfig,
    *,
    seed: int,
    regime: str,
    probe_name: str,
) -> dict[str, Any]:
    mix = int(hashlib.md5(f"rs1a:{regime}:{probe_name}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 17_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime=regime,
        friction=config.friction,
        damping=config.damping,
        seed=seed,
    )
    try:
        n_steps = int(round(config.duration_s / config.timestep))
        torques = _torque_series(
            PROBE_BANK[probe_name], n_steps, config.timestep, seed=seed + 91
        )
        probe = backend.rollout_probe(torques, margin=config.joint_limit_margin)

        # Frozen hinge-range support audit (independent of residual magnitude).
        lo, hi = float(backend.lo), float(backend.hi)
        delta = float(config.support_audit_margin)
        if hi > lo + 2.0 * delta:
            q_grid = np.linspace(
                lo + delta, hi - delta, int(config.support_audit_count), dtype=np.float64
            )
        else:
            q_grid = np.linspace(lo, hi, int(config.support_audit_count), dtype=np.float64)
        support_counts = {"modeled": 0, "boundary": 0, "unsupported": 0}
        for q in q_grid:
            _r, _t, _a, _h, support = backend.force_sample(float(q), 0.0, 0.0)
            support_counts[support] += 1
        n_obs = max(sum(support_counts.values()), 1)
        u = support_counts["unsupported"] / n_obs

        pools: dict[str, torch.Tensor] = {}
        for prefix, count, sampling in (
            ("passive", config.passive_context, "passive"),
            ("discovery", config.discovery_count, "discovery"),
        ):
            state = _sample_door_points(count, sampling, generator)
            physical, tangent, target, _supports = _observe(backend, state)
            pools[f"{prefix}_physical"] = physical
            pools[f"{prefix}_tangent"] = tangent
            pools[f"{prefix}_y"] = target + config.force_observation_noise * torch.randn(
                target.shape, generator=generator
            )

        passive_tangent = pools["passive_tangent"].unsqueeze(0)
        passive_y = pools["passive_y"].unsqueeze(0)
        discovery_tangent = pools["discovery_tangent"].unsqueeze(0)
        discovery_y = pools["discovery_y"].unsqueeze(0)

        v5 = V5Config()
        v5.ridge = config.ridge
        v5.posterior_noise_floor = config.posterior_noise_floor
        mean, cov = _fit_base_posterior(
            passive_tangent, passive_y, config.force_observation_noise, v5
        )
        pred_mean = torch.einsum("bni,bi->bn", discovery_tangent, mean)
        pred_var = torch.einsum(
            "bni,bij,bnj->bn", discovery_tangent, cov, discovery_tangent
        ).clamp_min(0.0)
        obs_var = max(config.force_observation_noise, config.posterior_noise_floor) ** 2
        sigma = (pred_var + obs_var).sqrt()

        discovery_residual = discovery_y - pred_mean
        _, residual_orthogonal, _ = tangent_decomposition(
            discovery_tangent, discovery_residual, config.ridge
        )
        r_perp = residual_orthogonal[0]
        sig = sigma[0].clamp_min(EPS)
        z = r_perp.abs() / sig

        d0 = float(r_perp.square().mean().sqrt())
        d1 = float(z.square().mean().sqrt())
        ell = 0.5 * (z.square() - 1.0)
        cum = torch.cumsum(ell, dim=0)
        d2 = float(cum.max())
        support_term = -math.log(max(1.0 - u, 0.0) + EPS)
        d3 = d2 + float(config.support_gamma) * support_term

        rs1_threshold = 1.35 * config.force_observation_noise
        rs1_triggered = d0 > rs1_threshold

        return {
            "seed": seed,
            "regime": regime,
            "probe": probe_name,
            "scores": {"D0": d0, "D1": d1, "D2": d2, "D3": d3},
            "support_counts": support_counts,
            "u_unsupported": u,
            "support_term": support_term,
            "rs1_baseline_triggered": bool(rs1_triggered),
            "rs1_baseline_threshold": rs1_threshold,
            "n_steps": n_steps,
            "probe_arrays": probe,
        }
    finally:
        backend.close()


def _task_mask(
    rows: list[dict[str, Any]], task: str
) -> tuple[np.ndarray, np.ndarray]:
    if task == "known_structural":
        pos = {"C1-L", "C1-H"}
        neg = {"C0"}
    elif task == "outside_library":
        pos = {"C2-latch"}
        neg = {"C0"}
    elif task == "pooled_structural":
        pos = {"C1-L", "C1-H", "C2-latch"}
        neg = {"C0"}
    else:
        raise ValueError(task)
    indices = []
    labels = []
    for i, row in enumerate(rows):
        if row["regime"] in pos:
            indices.append(i)
            labels.append(1)
        elif row["regime"] in neg:
            indices.append(i)
            labels.append(0)
    return np.asarray(indices), np.asarray(labels, dtype=np.int32)


def _aggregate(rows: list[dict[str, Any]], config: R1RS1AConfig) -> dict[str, Any]:
    primary = [r for r in rows if r["regime"] in PRIMARY_REGIMES]
    tasks = ("known_structural", "outside_library", "pooled_structural")
    metrics: dict[str, Any] = {"tasks": {}, "operating_points": {}, "hypothesis": {}}

    for task in tasks:
        idx, labels = _task_mask(primary, task)
        task_rows = [primary[i] for i in idx]
        metrics["tasks"][task] = {}
        for det in DETECTORS:
            scores = np.asarray([r["scores"][det] for r in task_rows], dtype=np.float64)
            metrics["tasks"][task][det] = {
                "auroc": _auroc(scores, labels),
                "auprc": _auprc(scores, labels),
                "n": int(len(labels)),
                "n_pos": int(labels.sum()),
                "n_neg": int((1 - labels).sum()),
            }

    idx_p, labels_p = _task_mask(primary, "pooled_structural")
    pooled_rows = [primary[i] for i in idx_p]
    c1l = [r for r in primary if r["regime"] == "C1-L"]
    c1h = [r for r in primary if r["regime"] == "C1-H"]
    c2 = [r for r in primary if r["regime"] == "C2-latch"]

    for det in DETECTORS:
        scores_p = np.asarray([r["scores"][det] for r in pooled_rows], dtype=np.float64)
        op = _operating_point(scores_p, labels_p, target_fpr=TARGET_C0_FPR)
        thr = op["threshold"]
        metrics["operating_points"][det] = {
            **op,
            "recall_c1_l": (
                _recall_at_threshold(
                    np.asarray([r["scores"][det] for r in c1l]),
                    np.ones(len(c1l), dtype=np.int32),
                    thr,
                )
                if c1l
                else float("nan")
            ),
            "recall_c1_h": (
                _recall_at_threshold(
                    np.asarray([r["scores"][det] for r in c1h]),
                    np.ones(len(c1h), dtype=np.int32),
                    thr,
                )
                if c1h
                else float("nan")
            ),
            "recall_c2": (
                _recall_at_threshold(
                    np.asarray([r["scores"][det] for r in c2]),
                    np.ones(len(c2), dtype=np.int32),
                    thr,
                )
                if c2
                else float("nan")
            ),
            "recall_pooled": op["tpr"],
            "c0_false_trigger": op["fpr"],
        }

    seed_deltas: dict[str, list[float]] = {
        "outside_library": [],
        "pooled_structural": [],
    }
    for seed in sorted({r["seed"] for r in primary}):
        seed_rows = [r for r in primary if r["seed"] == seed]
        for task in seed_deltas:
            idx, labels = _task_mask(seed_rows, task)
            if len(labels) == 0 or labels.sum() == 0 or (1 - labels).sum() == 0:
                continue
            task_rows = [seed_rows[i] for i in idx]
            s0 = np.asarray([r["scores"]["D0"] for r in task_rows])
            s3 = np.asarray([r["scores"]["D3"] for r in task_rows])
            seed_deltas[task].append(_auroc(s3, labels) - _auroc(s0, labels))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    op0 = metrics["operating_points"]["D0"]
    op3 = metrics["operating_points"]["D3"]
    auroc_out_d0 = metrics["tasks"]["outside_library"]["D0"]["auroc"]
    auroc_out_d3 = metrics["tasks"]["outside_library"]["D3"]["auroc"]
    auroc_pool_d0 = metrics["tasks"]["pooled_structural"]["D0"]["auroc"]
    auroc_pool_d3 = metrics["tasks"]["pooled_structural"]["D3"]["auroc"]

    hyp = {
        "auroc_outside_d3_gt_d0": bool(auroc_out_d3 > auroc_out_d0),
        "auroc_pooled_d3_gt_d0": bool(auroc_pool_d3 > auroc_pool_d0),
        "recall_c1l_d3_gt_d0_at_c0_fpr": bool(op3["recall_c1_l"] > op0["recall_c1_l"]),
        "recall_c2_d3_gt_d0_at_c0_fpr": bool(op3["recall_c2"] > op0["recall_c2"]),
        "seed_mean_delta_auroc_outside": _mean(seed_deltas["outside_library"]),
        "seed_mean_delta_auroc_pooled": _mean(seed_deltas["pooled_structural"]),
        "seed_deltas": seed_deltas,
    }
    hyp["rs1a_go"] = bool(
        hyp["auroc_outside_d3_gt_d0"]
        and hyp["auroc_pooled_d3_gt_d0"]
        and hyp["recall_c1l_d3_gt_d0_at_c0_fpr"]
        and hyp["recall_c2_d3_gt_d0_at_c0_fpr"]
    )
    metrics["hypothesis"] = hyp
    metrics["ablations"] = {
        "d1_vs_d0_pooled_auroc": metrics["tasks"]["pooled_structural"]["D1"]["auroc"]
        - metrics["tasks"]["pooled_structural"]["D0"]["auroc"],
        "d2_vs_d1_pooled_auroc": metrics["tasks"]["pooled_structural"]["D2"]["auroc"]
        - metrics["tasks"]["pooled_structural"]["D1"]["auroc"],
        "d3_vs_d2_outside_auroc": metrics["tasks"]["outside_library"]["D3"]["auroc"]
        - metrics["tasks"]["outside_library"]["D2"]["auroc"],
    }

    cneg = [r for r in rows if r["regime"] == "CNEG"]
    metrics["cneg_diagnostic"] = {
        "n": len(cneg),
        "mean_scores": {
            det: (
                float(np.mean([r["scores"][det] for r in cneg]))
                if cneg
                else float("nan")
            )
            for det in DETECTORS
        },
        "rs1_baseline_trigger_rate": (
            float(np.mean([r["rs1_baseline_triggered"] for r in cneg]))
            if cneg
            else float("nan")
        ),
    }
    metrics["rs1a_go"] = hyp["rs1a_go"]
    metrics["n_primary"] = len(primary)
    metrics["n_total"] = len(rows)
    metrics["config_gamma"] = config.support_gamma
    return metrics


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = row["probe_arrays"]
    meta = {k: v for k, v in row.items() if k != "probe_arrays"}
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
        scores = handle.create_group("detector_scores")
        for det, value in row["scores"].items():
            scores.attrs[det] = float(value)
        scores.attrs["revision_pipeline_executed"] = False


def run_r1_rs1a(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    config: R1RS1AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1AConfig()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _ = resolve_device("cpu")

    if smoke:
        jobs = [(SMOKE_SEED, regime, "P1") for regime in ALL_REGIMES]
        scientific = False
        cfg = R1RS1AConfig(
            **{
                **asdict(cfg),
                "duration_s": 2.0,
                "discovery_count": 8,
            }
        )
    else:
        jobs = [
            (seed, regime, probe)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for probe in cfg.probes
        ]
        scientific = True

    rows: list[dict[str, Any]] = []
    for index, (seed, regime, probe) in enumerate(jobs, start=1):
        print(
            f"[RS1A {index}/{len(jobs)}] seed={seed} regime={regime} probe={probe}",
            flush=True,
        )
        seed_everything(seed)
        row = _compute_detector_scores(
            cfg, seed=seed, regime=regime, probe_name=probe
        )
        rel = f"seed_{seed}/{regime}/{probe}.hdf5"
        _write_episode_h5(root / rel, row)
        slim = {k: v for k, v in row.items() if k != "probe_arrays"}
        slim["path"] = rel
        rows.append(slim)

    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs1a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; scientific ROC/PR not evaluated",
        }
    )

    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {"unlocked": True, "passed": True}
    status["R1-RS1"] = {
        "unlocked": True,
        "passed": False,
        "note": "scientific fail; split into RS1A/RS1B",
    }
    status["R1-RS1A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a_go")),
        "smoke_only": smoke,
    }
    status["R1-RS1B"] = {"locked_until_RS1A": True}
    status["R1-RS1C"] = {"locked": True}
    status["R1-RS2"] = {"locked": True}

    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "seed": r["seed"],
                "regime": r["regime"],
                "probe": r["probe"],
                "primary": int(r["regime"] in PRIMARY_REGIMES),
                "label_known": int(r["regime"] in {"C1-L", "C1-H"}),
                "label_outside": int(r["regime"] == "C2-latch"),
                "label_pooled": int(r["regime"] in {"C1-L", "C1-H", "C2-latch"}),
                "D0": r["scores"]["D0"],
                "D1": r["scores"]["D1"],
                "D2": r["scores"]["D2"],
                "D3": r["scores"]["D3"],
                "u_unsupported": r["u_unsupported"],
                "rs1_baseline_triggered": int(r["rs1_baseline_triggered"]),
            }
        )
    _write_csv(root / "detector_scores.csv", csv_rows)

    summary = {
        "stage": "R1-RS1A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "rs1_prereg_sha256_ref": _rs1_prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen": {
            "revision_pipeline": False,
            "acceptance": False,
            "passivity": False,
            "h32": False,
            "support_gamma": cfg.support_gamma,
            "detectors": list(DETECTORS),
        },
        "config": {
            **{
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(cfg).items()
            }
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1a_go": bool(aggregate.get("rs1a_go")),
        "unlocks_rs1b": bool(aggregate.get("rs1a_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

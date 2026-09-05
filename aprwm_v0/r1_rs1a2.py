"""R1-RS1A.2: weak structural signal detection (statistic ablation)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import (
    PROBE_BANK,
    PROBES,
    _observe,
    _require_rs0_unlock,
    _sample_door_points,
    _torque_series,
)
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import (
    ALL_REGIMES,
    EPS,
    PRIMARY_REGIMES,
    R1RS1AConfig,
    SUPPORT_AUDIT_COUNT,
    SUPPORT_AUDIT_MARGIN,
    _auprc,
    _auroc,
    _operating_point,
    _recall_at_threshold,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v4 import OPERATOR_NAMES, operator_library
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/REG/R1/R1_RS1A2_PREREG.md"
FORMAL_SEEDS = (9501, 9511, 9521, 9531, 9541)
SMOKE_SEED = 8971
RESIDUAL_DETECTORS = ("D0", "D_SNR", "D_dir", "D_corr", "D_lib")
CHALLENGER_DETECTORS = ("D_SNR", "D_dir", "D_corr", "D_lib")
TARGET_C0_FPR = 0.01
AUROC_DELTA = 0.02


@dataclass
class R1RS1A2Config:
    base: R1RS1AConfig | None = None
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = ALL_REGIMES
    probes: tuple[str, ...] = PROBES
    target_c0_fpr: float = TARGET_C0_FPR
    auroc_delta: float = AUROC_DELTA
    support_audit_count: int = SUPPORT_AUDIT_COUNT
    support_audit_margin: float = SUPPORT_AUDIT_MARGIN

    def resolved_base(self) -> R1RS1AConfig:
        return self.base or R1RS1AConfig()


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_statistics(
    r_perp: np.ndarray,
    physical: np.ndarray,
    *,
    u: float,
) -> dict[str, float]:
    r = np.asarray(r_perp, dtype=np.float64).reshape(-1)
    abs_r = np.abs(r)
    d0 = float(np.sqrt(np.mean(r * r)))
    d_snr = float(np.mean(abs_r) / (np.std(abs_r) + EPS))
    unit = r / (abs_r + EPS)
    d_dir = float(np.abs(np.mean(unit)))
    if r.size >= 2:
        num = r[:-1] * r[1:]
        den = (np.abs(r[:-1]) * np.abs(r[1:])) + EPS
        d_corr = float(np.mean(num / den))
    else:
        d_corr = 0.0

    # Library-matched evidence (no true-operator privilege).
    feats = torch.tensor(physical, dtype=torch.float32)
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    ops = operator_library(feats).detach().cpu().numpy()  # (T, K)
    matches = []
    for k in range(ops.shape[1]):
        phi = ops[:, k]
        denom = math.sqrt(float(np.sum(phi * phi)) + EPS)
        matches.append(abs(float(np.sum(r * phi))) / denom)
    d_lib = float(max(matches)) if matches else 0.0
    best_k = int(np.argmax(matches)) if matches else -1

    s_u = -math.log(max(1.0 - float(u), 0.0) + EPS)
    return {
        "D0": d0,
        "D_SNR": d_snr,
        "D_dir": d_dir,
        "D_corr": d_corr,
        "D_lib": d_lib,
        "D_support": s_u,
        "lib_best_operator": OPERATOR_NAMES[best_k] if best_k >= 0 else "none",
        "lib_matches": {
            OPERATOR_NAMES[i]: float(matches[i]) for i in range(len(matches))
        },
    }


def _compute_episode(
    config: R1RS1AConfig,
    *,
    seed: int,
    regime: str,
    probe_name: str,
    support_audit_count: int,
    support_audit_margin: float,
) -> dict[str, Any]:
    mix = int(hashlib.md5(f"rs1a2:{regime}:{probe_name}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 19_000 + (mix % 10_000))
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

        lo, hi = float(backend.lo), float(backend.hi)
        delta = float(support_audit_margin)
        if hi > lo + 2.0 * delta:
            q_grid = np.linspace(lo + delta, hi - delta, support_audit_count)
        else:
            q_grid = np.linspace(lo, hi, support_audit_count)
        support_counts = {"modeled": 0, "boundary": 0, "unsupported": 0}
        for q in q_grid:
            *_rest, support = backend.force_sample(float(q), 0.0, 0.0)
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
        discovery_physical = pools["discovery_physical"].numpy()

        v5 = V5Config()
        v5.ridge = config.ridge
        v5.posterior_noise_floor = config.posterior_noise_floor
        mean, _cov = _fit_base_posterior(
            passive_tangent, passive_y, config.force_observation_noise, v5
        )
        pred_mean = torch.einsum("bni,bi->bn", discovery_tangent, mean)
        discovery_residual = discovery_y - pred_mean
        _, residual_orthogonal, _ = tangent_decomposition(
            discovery_tangent, discovery_residual, config.ridge
        )
        r_perp = residual_orthogonal[0].detach().cpu().numpy()
        stats = _compute_statistics(r_perp, discovery_physical, u=u)

        return {
            "seed": seed,
            "regime": regime,
            "probe": probe_name,
            "scores": {k: stats[k] for k in list(RESIDUAL_DETECTORS) + ["D_support"]},
            "lib_best_operator": stats["lib_best_operator"],
            "lib_matches": stats["lib_matches"],
            "support_counts": support_counts,
            "u_unsupported": u,
            "n_steps": n_steps,
            "probe_arrays": probe,
        }
    finally:
        backend.close()


def _subset_labels(
    rows: list[dict[str, Any]], positives: set[str], negatives: set[str]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    kept = []
    labels = []
    for row in rows:
        if row["regime"] in positives:
            kept.append(row)
            labels.append(1)
        elif row["regime"] in negatives:
            kept.append(row)
            labels.append(0)
    return kept, np.asarray(labels, dtype=np.int32)


def _op_recalls(
    rows: list[dict[str, Any]],
    *,
    detector: str,
    target_fpr: float,
) -> dict[str, float]:
    # Operating point from C0 vs C1-L (primary weak-signal task).
    task_rows, labels = _subset_labels(rows, {"C1-L"}, {"C0"})
    scores = np.asarray([r["scores"][detector] for r in task_rows], dtype=np.float64)
    op = _operating_point(scores, labels, target_fpr=target_fpr)
    thr = op["threshold"]

    def recall(regime: str) -> float:
        xs = [r["scores"][detector] for r in rows if r["regime"] == regime]
        if not xs:
            return float("nan")
        return _recall_at_threshold(
            np.asarray(xs), np.ones(len(xs), dtype=np.int32), thr
        )

    c0 = [r["scores"][detector] for r in rows if r["regime"] == "C0"]
    fpr = (
        float(np.mean(np.asarray(c0) >= thr)) if c0 else float("nan")
    )
    return {
        "threshold": thr,
        "c0_fpr": fpr,
        "recall_c1_l": recall("C1-L"),
        "recall_c1_h": recall("C1-H"),
        "recall_c2": recall("C2-latch"),
        "tpr_c1_l": op["tpr"],
    }


def _typed_or_eval(
    rows: list[dict[str, Any]],
    *,
    known: str,
    target_fpr: float,
) -> dict[str, Any]:
    primary = [r for r in rows if r["regime"] in PRIMARY_REGIMES]
    # tau_u: any positive support on C0 should be impossible; use eps floor.
    c0_support = [r["scores"]["D_support"] for r in primary if r["regime"] == "C0"]
    tau_u = max(EPS, float(np.max(c0_support) + EPS) if c0_support else EPS)

    # tau_k from known channel C0 vs C1-L at FPR<=1%
    task_rows, labels = _subset_labels(primary, {"C1-L"}, {"C0"})
    scores_k = np.asarray([r["scores"][known] for r in task_rows], dtype=np.float64)
    op = _operating_point(scores_k, labels, target_fpr=target_fpr)
    tau_k = op["threshold"]

    def triggered(row: dict[str, Any]) -> bool:
        return bool(
            row["scores"][known] >= tau_k or row["scores"]["D_support"] > tau_u
        )

    def rate(regime: str) -> float:
        xs = [r for r in primary if r["regime"] == regime]
        if not xs:
            return float("nan")
        return float(np.mean([triggered(r) for r in xs]))

    return {
        "known_channel": known,
        "tau_k": float(tau_k),
        "tau_u": float(tau_u),
        "c0_false_trigger": rate("C0"),
        "recall_c1_l": rate("C1-L"),
        "recall_c1_h": rate("C1-H"),
        "recall_c2": rate("C2-latch"),
    }


def _aggregate(rows: list[dict[str, Any]], config: R1RS1A2Config) -> dict[str, Any]:
    primary = [r for r in rows if r["regime"] in PRIMARY_REGIMES]
    metrics: dict[str, Any] = {
        "tasks": {},
        "operating_points": {},
        "typed": {},
        "gates": {},
    }

    tasks = {
        "c1_l": ({"C1-L"}, {"C0"}),
        "c1_h": ({"C1-H"}, {"C0"}),
        "c1_pooled": ({"C1-L", "C1-H"}, {"C0"}),
    }
    for task_name, (pos, neg) in tasks.items():
        task_rows, labels = _subset_labels(primary, pos, neg)
        metrics["tasks"][task_name] = {}
        for det in RESIDUAL_DETECTORS:
            scores = np.asarray([r["scores"][det] for r in task_rows], dtype=np.float64)
            metrics["tasks"][task_name][det] = {
                "auroc": _auroc(scores, labels),
                "auprc": _auprc(scores, labels),
                "n": int(len(labels)),
                "n_pos": int(labels.sum()),
                "n_neg": int((1 - labels).sum()),
            }

    for det in RESIDUAL_DETECTORS:
        metrics["operating_points"][det] = _op_recalls(
            primary, detector=det, target_fpr=config.target_c0_fpr
        )

    metrics["typed"]["D_lib_or_support"] = _typed_or_eval(
        primary, known="D_lib", target_fpr=config.target_c0_fpr
    )
    metrics["typed"]["D0_or_support"] = _typed_or_eval(
        primary, known="D0", target_fpr=config.target_c0_fpr
    )

    # GO: challenger beats D0 on C1-L recall at matched FPR, with AUROC non-deg.
    base_op = metrics["operating_points"]["D0"]
    base_auroc = metrics["tasks"]["c1_l"]["D0"]["auroc"]
    qualifiers = []
    for det in CHALLENGER_DETECTORS:
        op = metrics["operating_points"][det]
        auroc = metrics["tasks"]["c1_l"][det]["auroc"]
        ok = bool(
            op["recall_c1_l"] > base_op["recall_c1_l"]
            and auroc >= base_auroc - config.auroc_delta
        )
        entry = {
            "detector": det,
            "recall_c1_l": op["recall_c1_l"],
            "auroc_c1_l": auroc,
            "qualifies": ok,
        }
        if ok:
            qualifiers.append(entry)
    if qualifiers:
        winner = max(
            qualifiers,
            key=lambda e: (e["recall_c1_l"], e["auroc_c1_l"]),
        )
    else:
        winner = None

    # Lib best-op distribution on C1-L
    c1l = [r for r in primary if r["regime"] == "C1-L"]
    from collections import Counter

    metrics["c1_l_lib_best"] = dict(Counter(r["lib_best_operator"] for r in c1l))

    metrics["gates"] = {
        "d0_recall_c1_l": base_op["recall_c1_l"],
        "d0_auroc_c1_l": base_auroc,
        "qualifiers": qualifiers,
        "winner": winner,
        "rs1a2_go": bool(winner is not None),
        "auroc_delta": config.auroc_delta,
    }
    metrics["rs1a2_go"] = bool(winner is not None)
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
        scores.attrs["lib_best_operator"] = row["lib_best_operator"]


def run_r1_rs1a2(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    config: R1RS1A2Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1A2Config()
    base = cfg.resolved_base()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _ = resolve_device("cpu")

    if smoke:
        base = replace(base, duration_s=2.0, discovery_count=8)
        jobs = [(SMOKE_SEED, regime, "P1") for regime in ALL_REGIMES]
        scientific = False
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
            f"[RS1A.2 {index}/{len(jobs)}] seed={seed} regime={regime} probe={probe}",
            flush=True,
        )
        seed_everything(seed)
        row = _compute_episode(
            base,
            seed=seed,
            regime=regime,
            probe_name=probe,
            support_audit_count=cfg.support_audit_count,
            support_audit_margin=cfg.support_audit_margin,
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
            "rs1a2_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only",
        }
    )

    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "seed": r["seed"],
                "regime": r["regime"],
                "probe": r["probe"],
                "lib_best_operator": r["lib_best_operator"],
                "u_unsupported": r["u_unsupported"],
                **r["scores"],
            }
        )
    _write_csv(root / "detector_scores.csv", csv_rows)

    status = stage_status()
    status["R1-RS1A"] = {"unlocked": True, "passed": False, "informative": True}
    status["R1-RS1A.1"] = {"unlocked": True, "passed": False, "informative": True}
    status["R1-RS1A.2"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a2_go")),
        "smoke_only": smoke,
        "winner": (aggregate.get("gates") or {}).get("winner"),
    }
    status["R1-RS1B"] = {
        "locked_until_weak_signal_resolved": True,
        "unlocked": bool(aggregate.get("rs1a2_go")),
    }
    status["R1-RS2"] = {"locked": True}

    summary = {
        "stage": "R1-RS1A.2",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen": {
            "revision_pipeline": False,
            "gamma_composition": False,
            "residual_detectors": list(RESIDUAL_DETECTORS),
            "operator_library": list(OPERATOR_NAMES),
        },
        "config": {
            "seeds": list(cfg.seeds),
            "regimes": list(cfg.regimes),
            "probes": list(cfg.probes),
            "target_c0_fpr": cfg.target_c0_fpr,
            "auroc_delta": cfg.auroc_delta,
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1a2_go": bool(aggregate.get("rs1a2_go")),
        "unlocks_rs1b": bool(aggregate.get("rs1a2_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

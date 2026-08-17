"""R1-RS5A: intervention-indexed calibration with abstention."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs1a import R1RS1AConfig
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import R1RS2C0Config
from .r1_rs3a import _jsonable
from .r1_rs4a import (
    MODE_A_AMPS,
    PULL_JOBS,
    PUSH_JOBS,
    REGIMES,
    _brier,
    _contact_episode,
    _fit_logit,
    _mode_a_episode,
    _predict_logit,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1_RS5A_PREREG.md"
SEEDS = (15101, 15111, 15121, 15131, 15141)
DEV_SEEDS = SEEDS[:3]
HELD_SEEDS = SEEDS[3:]
SUPPORT_KEYS = ("sign_coverage", "log_n", "q_span")
SMOKE_JOBS = (
    ("mode_a", 15101, "C0", 1.5, None, 1.0),
    ("contact", 15101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 15101, "C0", math.nan, "pull_push", 1.0),
)
SUPPORT_RIDGE = 1.0e-4
LOGIT_RIDGE = 1.0
CONTACT_FRAC_CUT = 0.5


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS5AConfig:
    seeds: tuple[int, ...] = SEEDS
    logit_ridge: float = LOGIT_RIDGE
    known_fpr_max: float = 0.20
    unseen_fcr_max: float = 0.10
    calibrated_abstain_max: float = 0.20


def _require_rs4a(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS5A locked until RS4A has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS5A locked: RS4A scientific matrix has not been run")
    return payload


def coarse_index(row: dict[str, Any]) -> str:
    """Learner-visible coarse intervention. Does not see pull_push family label."""
    frac = float(row.get("contact_frac", 0.0))
    return "mode_a" if frac < CONTACT_FRAC_CUT else "contact"


def _support_vec(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in SUPPORT_KEYS], dtype=np.float64)


def _fit_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cloud = np.stack([_support_vec(row) for row in rows], axis=0)
    mean = np.mean(cloud, axis=0)
    cov = np.cov(cloud, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    cov = cov + SUPPORT_RIDGE * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov)
    distances = [
        float(np.sqrt(max((vec - mean) @ precision @ (vec - mean), 0.0)))
        for vec in cloud
    ]
    return {
        "mean": mean,
        "precision": precision,
        "tau": float(np.quantile(distances, 0.95)) if distances else math.nan,
        "n": len(rows),
    }


def _distance(row: dict[str, Any], support: dict[str, Any]) -> float:
    vec = _support_vec(row)
    delta = vec - support["mean"]
    return float(np.sqrt(max(delta @ support["precision"] @ delta, 0.0)))


def _in_support(row: dict[str, Any], support: dict[str, Any]) -> bool:
    return _distance(row, support) <= float(support["tau"]) + 1.0e-12


def _fit_calibration(rows: list[dict[str, Any]], ridge: float) -> np.ndarray:
    features = np.asarray([[float(row["log_d0"])] for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["y_inadequate"]) for row in rows], dtype=np.float64)
    return _fit_logit(features, labels, ridge)


def _prob(row: dict[str, Any], weights: np.ndarray) -> float:
    features = np.asarray([[float(row["log_d0"])]], dtype=np.float64)
    return float(_predict_logit(features, weights)[0])


def apply_policies(
    row: dict[str, Any],
    *,
    global_w: np.ndarray,
    calib: dict[str, np.ndarray],
    supports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    index = coarse_index(row)
    p_global = _prob(row, global_w)
    p_indexed = _prob(row, calib[index])
    valid = _in_support(row, supports[index])
    return {
        "coarse_I": index,
        "valid_support": bool(valid),
        "p_global": p_global,
        "p_indexed": p_indexed,
        "global_confident": True,
        "indexed_confident": True,
        "abstain_confident": bool(valid),
        "global_yhat": int(p_global > 0.5),
        "indexed_yhat": int(p_indexed > 0.5),
        "abstain_yhat": (int(p_indexed > 0.5) if valid else None),
    }


def _ece(probs: list[float], labels: list[int], bins: int = 5) -> float:
    if not probs:
        return math.nan
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & ((p < right) if right < 1.0 else (p <= right))
        if not np.any(mask):
            continue
        total += abs(float(np.mean(p[mask])) - float(np.mean(y[mask]))) * float(np.mean(mask))
    return float(total)


def _aggregate(rows: list[dict[str, Any]], config: R1RS5AConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev = [row for row in rows if int(row["seed"]) in DEV_SEEDS]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    mode_a_dev = [row for row in dev if row["family"] == "mode_a"]
    pull_dev = [row for row in dev if row["family"] == "contact_pull"]
    known_dev = mode_a_dev + pull_dev
    global_w = _fit_calibration(known_dev, config.logit_ridge)
    calib = {
        "mode_a": _fit_calibration(mode_a_dev, config.logit_ridge),
        "contact": _fit_calibration(pull_dev, config.logit_ridge),
    }
    supports = {
        "mode_a": _fit_support(mode_a_dev),
        "contact": _fit_support(pull_dev),
    }
    scored = []
    for row in rows:
        applied = apply_policies(row, global_w=global_w, calib=calib, supports=supports)
        scored.append({**row, **applied})

    held_known = [
        row
        for row in scored
        if int(row["seed"]) in HELD_SEEDS and row["family"] in {"mode_a", "contact_pull"}
    ]
    brier_g = _brier(
        np.asarray([row["p_global"] for row in held_known]),
        np.asarray([row["y_inadequate"] for row in held_known], dtype=np.float64),
    )
    brier_i = _brier(
        np.asarray([row["p_indexed"] for row in held_known]),
        np.asarray([row["y_inadequate"] for row in held_known], dtype=np.float64),
    )
    h1 = bool(math.isfinite(brier_i) and math.isfinite(brier_g) and brier_i < brier_g)

    def fpr_indexed(index: str) -> float:
        source = mode_a_dev if index == "mode_a" else pull_dev
        c0_dev = [_prob(row, calib[index]) for row in source if row["regime"] == "C0"]
        tau = float(np.quantile(c0_dev, 0.95)) if c0_dev else math.nan
        c0_held = [
            row for row in held_known if row["coarse_I"] == index and row["regime"] == "C0"
        ]
        if not c0_held or not math.isfinite(tau):
            return math.nan
        return float(np.mean([row["p_indexed"] > tau for row in c0_held]))

    fpr_ma = fpr_indexed("mode_a")
    fpr_c = fpr_indexed("contact")
    h2 = bool(
        math.isfinite(fpr_ma)
        and math.isfinite(fpr_c)
        and fpr_ma <= config.known_fpr_max + 1.0e-12
        and fpr_c <= config.known_fpr_max + 1.0e-12
    )

    push = [row for row in scored if row["family"] == "contact_push"]
    fcr = {
        "global": float(np.mean([row["global_confident"] for row in push])) if push else math.nan,
        "indexed": float(np.mean([row["indexed_confident"] for row in push])) if push else math.nan,
        "abstain": float(np.mean([row["abstain_confident"] for row in push])) if push else math.nan,
    }
    h3 = bool(
        math.isfinite(fcr["abstain"])
        and fcr["abstain"] <= config.unseen_fcr_max + 1.0e-12
        and abs(fcr["global"] - 1.0) < 1.0e-12
        and abs(fcr["indexed"] - 1.0) < 1.0e-12
    )

    held_pull = [
        row
        for row in scored
        if int(row["seed"]) in HELD_SEEDS and row["family"] == "contact_pull"
    ]
    pull_abstain = (
        float(np.mean([not row["abstain_confident"] for row in held_pull]))
        if held_pull
        else math.nan
    )
    h4 = bool(
        math.isfinite(pull_abstain) and pull_abstain <= config.calibrated_abstain_max + 1.0e-12
    )
    del held

    return {
        "n_episodes": len(rows),
        "n_dev": len(dev),
        "n_held": len([row for row in rows if int(row["seed"]) in HELD_SEEDS]),
        "h1_indexed_brier": {
            "brier_global": brier_g,
            "brier_indexed": brier_i,
            "ece_global": _ece(
                [row["p_global"] for row in held_known],
                [int(row["y_inadequate"]) for row in held_known],
            ),
            "ece_indexed": _ece(
                [row["p_indexed"] for row in held_known],
                [int(row["y_inadequate"]) for row in held_known],
            ),
            "pass": h1,
        },
        "h2_known_c0_fpr": {
            "fpr_mode_a": fpr_ma,
            "fpr_contact": fpr_c,
            "max": config.known_fpr_max,
            "pass": h2,
        },
        "h3_unseen_false_confidence": {
            "n_push": len(push),
            "fcr": fcr,
            "pass": h3,
        },
        "h4_calibrated_contact_abstain": {
            "heldout_contact_pull_abstain_rate": pull_abstain,
            "max": config.calibrated_abstain_max,
            "pass": h4,
        },
        "support_tau": {
            "mode_a": float(supports["mode_a"]["tau"]),
            "contact": float(supports["contact"]["tau"]),
        },
        "rs5a_go": bool(h1 and h2 and h3 and h4),
        "rewrites_rs4a_go": False,
        "unlocks_rs5b": False,
        "unlocks_revision": False,
        "uses_pull_push_label_in_policy": False,
        "one_hot_closed_world": False,
    }, scored


def run_r1_rs5a(
    output: str | Path,
    *,
    rs4a_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS5AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS5AConfig()
    _require_rs4a(Path(rs4a_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    if smoke:
        jobs: list[tuple[Any, ...]] = list(SMOKE_JOBS)
        scientific = False
    else:
        jobs = [
            ("mode_a", seed, regime, amp, None, 1.0)
            for seed in cfg.seeds
            for regime in REGIMES
            for amp in MODE_A_AMPS
        ] + [
            ("contact", seed, regime, math.nan, script, scale)
            for seed in cfg.seeds
            for regime in REGIMES
            for script, scale in PULL_JOBS + PUSH_JOBS
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[RS5A {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed))
        if domain == "mode_a":
            row = _mode_a_episode(
                intake, seed=int(seed), regime=str(regime), amp_scale=float(amp)
            )
        else:
            row = _contact_episode(
                intake,
                c0,
                seed=int(seed),
                regime=str(regime),
                script=str(script),
                scale=float(scale),
            )
        rows.append(row)
    if scientific:
        aggregate, scored = _aggregate(rows, cfg)
    else:
        aggregate = {
            "rs5a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; H1–H4 not evaluated",
        }
        scored = rows
    status = stage_status()
    status["R1-RS4A"] = {"frozen": True, "passed": False, "zero_shot_transport": "abandoned"}
    status["R1-RS5A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs5a_go")),
        "smoke_only": smoke,
    }
    status["R1-RS5B"] = {"locked": True, "opened": False}
    status["R1-RS5C"] = {"locked": True, "opened": False}
    status["R1-RS3B"] = {"locked": True, "opened": False}
    status["R1-RS4B"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R1-RS5A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "calibrated": ["mode_a", "contact_pull"],
            "unseen": "pull_push",
            "coarse_index": "contact_frac<0.5",
            "support": list(SUPPORT_KEYS),
            "rs5b": "locked",
            "zero_shot_transport": "abandoned",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(scored),
        "aggregate": _jsonable(aggregate),
        "rs5a_go": bool(aggregate.get("rs5a_go")),
        "rs4a_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "family",
        "seed",
        "regime",
        "script",
        "y_inadequate",
        "D0",
        "coarse_I",
        "valid_support",
        "p_global",
        "p_indexed",
        "abstain_confident",
        "sign_coverage",
        "contact_frac",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in scored],
    )
    return summary

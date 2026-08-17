"""R3-V7A: mixture-trained contextual structural-inadequacy belief."""

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
    FEATURE_NAMES,
    MODE_A_AMPS,
    PULL_JOBS,
    PUSH_JOBS,
    REGIMES,
    _auroc,
    _brier,
    _contact_episode,
    _feature_matrix,
    _fit_logit,
    _mode_a_episode,
    _predict_logit,
    _standardize,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3_V7A_PREREG.md"
SEEDS = (16101, 16111, 16121, 16131, 16141)
DEV_SEEDS = SEEDS[:3]
HELD_SEEDS = SEEDS[3:]
FAMILIES = ("mode_a", "contact_pull", "contact_push")
SMOKE_JOBS = (
    ("mode_a", 16101, "C0", 1.5, None, 1.0),
    ("contact", 16101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 16101, "C0", math.nan, "pull_push", 1.0),
)
LOGIT_RIDGE = 1.0
MIN_FAMILY_AUROC = 0.60
HELD_OUT_FPR_MAX = 0.20


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7AConfig:
    seeds: tuple[int, ...] = SEEDS
    logit_ridge: float = LOGIT_RIDGE
    min_family_auroc: float = MIN_FAMILY_AUROC
    held_out_fpr_max: float = HELD_OUT_FPR_MAX


def _require_rs5a(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7A locked until RS5A has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7A locked: RS5A scientific matrix has not been run")
    return payload


def _fit_scored(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    names: tuple[str, ...],
    ridge: float,
) -> np.ndarray:
    y_train = np.asarray([row["y_inadequate"] for row in train], dtype=np.float64)
    x_train = _feature_matrix(train, names)
    x_test = _feature_matrix(test, names)
    train_std, test_std = _standardize(x_train, x_test)
    weights = _fit_logit(train_std, y_train, ridge)
    return _predict_logit(test_std, weights)


def _c0_tau(train: list[dict[str, Any]], p_train: np.ndarray) -> float:
    scores = [
        float(p_train[index])
        for index, row in enumerate(train)
        if row["regime"] == "C0"
    ]
    return float(np.quantile(scores, 0.95)) if scores else math.nan


def _family_auroc(rows: list[dict[str, Any]], probs: np.ndarray, family: str) -> float:
    mask = [row["family"] == family for row in rows]
    if not any(mask):
        return math.nan
    scores = np.asarray([probs[i] for i, keep in enumerate(mask) if keep], dtype=np.float64)
    labels = np.asarray(
        [rows[i]["y_inadequate"] for i, keep in enumerate(mask) if keep],
        dtype=np.float64,
    )
    return _auroc(scores, labels)


def _aggregate(rows: list[dict[str, Any]], config: R3V7AConfig) -> dict[str, Any]:
    dev = [row for row in rows if int(row["seed"]) in DEV_SEEDS]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    y_held = np.asarray([row["y_inadequate"] for row in held], dtype=np.float64)
    p_ctx = _fit_scored(dev, held, FEATURE_NAMES, config.logit_ridge)
    p_d0 = _fit_scored(dev, held, ("log_d0",), config.logit_ridge)
    p_ctx_dev = _fit_scored(dev, dev, FEATURE_NAMES, config.logit_ridge)
    tau = _c0_tau(dev, p_ctx_dev)
    held_c0 = [float(p_ctx[i]) for i, row in enumerate(held) if row["regime"] == "C0"]
    fpr = (
        float(np.mean([score > tau for score in held_c0]))
        if held_c0 and math.isfinite(tau)
        else math.nan
    )
    brier_ctx = _brier(p_ctx, y_held)
    brier_d0 = _brier(p_d0, y_held)
    family_auroc = {family: _family_auroc(held, p_ctx, family) for family in FAMILIES}
    h1 = bool(math.isfinite(brier_ctx) and math.isfinite(brier_d0) and brier_ctx < brier_d0)
    h2 = bool(math.isfinite(fpr) and fpr <= config.held_out_fpr_max)
    h3 = bool(
        all(
            math.isfinite(score) and score >= config.min_family_auroc
            for score in family_auroc.values()
        )
    )
    loio_train = [row for row in dev if row["family"] != "contact_push"]
    loio_test = [row for row in held if row["family"] == "contact_push"]
    if loio_train and loio_test:
        p_loio = _fit_scored(loio_train, loio_test, FEATURE_NAMES, config.logit_ridge)
        y_loio = np.asarray([row["y_inadequate"] for row in loio_test], dtype=np.float64)
        mix_on_push = np.asarray(
            [p_ctx[i] for i, row in enumerate(held) if row["family"] == "contact_push"],
            dtype=np.float64,
        )
        diagnostic = {
            "n_train": len(loio_train),
            "n_test": len(loio_test),
            "brier_loio": _brier(p_loio, y_loio),
            "brier_mixture_on_push": _brier(mix_on_push, y_loio),
            "auroc_loio": _auroc(p_loio, y_loio),
            "auroc_mixture_on_push": _auroc(mix_on_push, y_loio),
            "note": "diagnostic only; not a GO",
        }
    else:
        diagnostic = {"note": "insufficient rows"}
    return {
        "n_dev": len(dev),
        "n_held": len(held),
        "tau_mix_dev_c0": tau,
        "h1_mixture_brier": {
            "brier_context": brier_ctx,
            "brier_d0": brier_d0,
            "pass": h1,
        },
        "h2_held_c0_fpr": {
            "fpr": fpr,
            "max": config.held_out_fpr_max,
            "pass": h2,
        },
        "h3_within_family_auroc": {
            "min": config.min_family_auroc,
            "auroc": family_auroc,
            "pass": h3,
        },
        "diagnostic_loio_push": diagnostic,
        "v7a_go": bool(h1 and h2 and h3),
        "uses_family_label_in_policy": False,
        "rewrites_rs5a_go": False,
        "unlocks_rs5b": False,
        "unlocks_revision": False,
        "is_old_v07_routing": False,
    }


def run_r3_v7a(
    output: str | Path,
    *,
    rs5a_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7AConfig()
    _require_rs5a(Path(rs5a_summary))
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
            f"[V7A {index}/{len(jobs)}] {domain} seed={seed} {regime} "
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
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "v7a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; mixture GO not evaluated",
        }
    )
    status = stage_status()
    status["R1-transport"] = {"frozen": True, "mainline": "closed"}
    status["R1-RS5A"] = {"frozen": True, "passed": True, "mainline": False}
    status["R1-RS5B"] = {"locked": True, "opened": False}
    status["R3-V7A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7a_go")),
        "smoke_only": smoke,
    }
    status["R3-V7B"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R3-V7A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "score": "p_struct(h_summary) on embodied mixture",
            "features": list(FEATURE_NAMES),
            "no_family_label": True,
            "transport_mainline": "closed",
            "rs5b": "locked",
            "v7b_sequential_belief": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(rows),
        "aggregate": _jsonable(aggregate),
        "v7a_go": bool(aggregate.get("v7a_go")),
        "rs5a_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "family",
        "domain",
        "seed",
        "regime",
        "script",
        "scale",
        "y_inadequate",
        "D0",
        *FEATURE_NAMES,
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

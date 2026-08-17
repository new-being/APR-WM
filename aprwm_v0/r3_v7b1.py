"""R3-V7B.1: same GRU, integrated-Brier objective vs time-weighted BCE."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .r1_mj import stage_status
from .r1_rs1a import R1RS1AConfig
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import R1RS2C0Config
from .r1_rs3a import _jsonable
from .r1_rs4a import FEATURE_NAMES, MODE_A_AMPS, PULL_JOBS, PUSH_JOBS, REGIMES, _brier
from .r3_v7b import (
    EpiGRU,
    R3V7BConfig,
    STEP_DIM,
    _contact_trace,
    _gru_probs,
    _mode_a_trace,
    _pad,
    mean_positive_dwell,
    recovery_time,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7B1_PREREG.md"
SEEDS = (18101, 18111, 18121, 18131, 18141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
DELTA = 0.02
C0_STEP_FPR_MAX = 0.20
BRIER_FRACS = (0.10, 0.25, 0.50, 0.75, 1.00)
EVENT_PHASES = (
    ("first_contact", 2.0),
    ("release", 3.0),
    ("recontact", 2.5),
    ("quiet", 5.0),
)
SMOKE_JOBS = (
    ("mode_a", 18101, "C0", 1.5, None, 1.0),
    ("contact", 18101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 18101, "C0", math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7B1Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    target_hz: float = 20.0


def _require_v7b(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7B.1 locked until R3-V7B has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7B.1 locked: V7B scientific matrix has not been run")
    return payload


def integrated_brier(series_list: list[np.ndarray], labels: list[int]) -> float:
    if not series_list:
        return math.nan
    scores = []
    for probs, label in zip(series_list, labels):
        target = float(label)
        scores.append(float(np.mean((probs - target) ** 2)))
    return float(np.mean(scores))


def c0_brier_mass(series_list: list[np.ndarray]) -> tuple[float, float]:
    if not series_list:
        return math.nan, math.nan
    brier = float(np.mean([float(np.mean(probs * probs)) for probs in series_list]))
    mass = float(np.mean([float(np.mean(probs)) for probs in series_list]))
    return brier, mass


def brier_at_frac(series_list: list[np.ndarray], labels: list[int], frac: float) -> float:
    preds = []
    for probs in series_list:
        index = int(round(frac * (len(probs) - 1)))
        index = min(max(index, 0), len(probs) - 1)
        preds.append(float(probs[index]))
    return _brier(np.asarray(preds), np.asarray(labels, dtype=np.float64))


def first_phase_index(phase: np.ndarray, value: float) -> int:
    hits = np.where(np.asarray(phase, dtype=np.float64) == value)[0]
    return int(hits[0]) if len(hits) else -1


def event_mean_p(probs: np.ndarray, phase: np.ndarray, value: float, half: int = 2) -> float:
    index = first_phase_index(phase, value)
    if index < 0:
        return math.nan
    lo = max(0, index - half)
    hi = min(len(probs), index + half + 1)
    return float(np.mean(probs[lo:hi]))


def _trace_config(config: R3V7B1Config) -> R3V7BConfig:
    return R3V7BConfig(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
    )


def _train_gru(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7B1Config,
    *,
    objective: str,
    seed: int,
) -> EpiGRU:
    torch.manual_seed(seed)
    model = EpiGRU(STEP_DIM, config.hidden)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(row["steps"] - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            pred = model(padded, lengths)
            losses = []
            for index, row in enumerate(chunk):
                t_len = int(lengths[index])
                target = torch.full((t_len,), float(row["y_inadequate"]))
                p_t = pred[index, :t_len]
                if objective == "ibs":
                    losses.append(torch.mean((p_t - target) ** 2))
                else:
                    time = torch.linspace(0.0, 1.0, t_len)
                    weight = 0.3 + 0.7 * time
                    bce = nn.functional.binary_cross_entropy(p_t, target, reduction="none")
                    losses.append(torch.sum(weight * bce) / float(t_len))
            torch.stack(losses).mean().backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-config.lr)
                        param.grad.zero_()
        model.eval()
        with torch.no_grad():
            series = [_gru_probs(model, row, mean, std) for row in val]
            labels = [int(row["y_inadequate"]) for row in val]
            if objective == "ibs":
                score = integrated_brier(series, labels)
            else:
                score = _brier(
                    np.asarray([float(item[-1]) for item in series]),
                    np.asarray(labels, dtype=np.float64),
                )
        if score + 1.0e-6 < best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _series_for(
    model: EpiGRU, rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray]:
    return [_gru_probs(model, row, mean, std) for row in rows]


def _aggregate(rows: list[dict[str, Any]], config: R3V7B1Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    step_stack = np.concatenate([row["steps"] for row in train], axis=0)
    step_mean = np.mean(step_stack, axis=0)
    step_std = np.where(np.std(step_stack, axis=0) < 1.0e-12, 1.0, np.std(step_stack, axis=0))
    b2 = _train_gru(train, val, step_mean, step_std, config, objective="bce", seed=11)
    b3 = _train_gru(train, val, step_mean, step_std, config, objective="ibs", seed=13)
    s2 = _series_for(b2, held, step_mean, step_std)
    s3 = _series_for(b3, held, step_mean, step_std)
    tau = float(
        np.quantile(
            [
                float(_gru_probs(b2, row, step_mean, step_std)[-1])
                for row in train + val
                if row["regime"] == "C0"
            ],
            0.95,
        )
    )
    ibs2 = integrated_brier(s2, labels)
    ibs3 = integrated_brier(s3, labels)
    c0_2 = [s2[i] for i, row in enumerate(held) if row["regime"] == "C0"]
    c0_3 = [s3[i] for i, row in enumerate(held) if row["regime"] == "C0"]
    b_c0_2, m_c0_2 = c0_brier_mass(c0_2)
    b_c0_3, m_c0_3 = c0_brier_mass(c0_3)
    flags3 = np.concatenate([series > tau for series in c0_3]) if c0_3 else np.zeros(0)
    fpr3 = float(np.mean(flags3)) if len(flags3) else math.nan
    mid2 = brier_at_frac(s2, labels, 0.5)
    mid3 = brier_at_frac(s3, labels, 0.5)
    fin2 = brier_at_frac(s2, labels, 1.0)
    fin3 = brier_at_frac(s3, labels, 1.0)
    h1 = bool(math.isfinite(ibs3) and math.isfinite(ibs2) and ibs3 < ibs2)
    h2 = bool(
        math.isfinite(b_c0_3)
        and math.isfinite(b_c0_2)
        and b_c0_3 < b_c0_2
        and math.isfinite(fpr3)
        and fpr3 <= config.c0_step_fpr_max
    )
    h3 = bool(
        math.isfinite(mid3)
        and math.isfinite(fin3)
        and mid3 <= mid2 + config.delta
        and fin3 <= fin2 + config.delta
    )
    events = {}
    for name, value in EVENT_PHASES:
        vals = [
            event_mean_p(s3[i], held[i]["phase"], value)
            for i, row in enumerate(held)
            if row["regime"] == "C0" and row["family"] == "contact_switch"
        ]
        events[name] = float(np.nanmean(vals)) if vals else math.nan
    dwells = [mean_positive_dwell(series > tau) for series in c0_3]
    recoveries = [
        recovery_time(s3[i], tau, int(held[i]["quiet_start"]))
        for i, row in enumerate(held)
        if row["regime"] == "C0" and row["family"] == "contact_switch"
    ]
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_b2_c0_final": tau,
        "delta": config.delta,
        "h1_ibs": {"ibs_b2": ibs2, "ibs_b3": ibs3, "pass": h1},
        "h2_c0_occupancy": {
            "b_c0_b2": b_c0_2,
            "b_c0_b3": b_c0_3,
            "m_c0_b2": m_c0_2,
            "m_c0_b3": m_c0_3,
            "fpr_b3": fpr3,
            "max_fpr": config.c0_step_fpr_max,
            "pass": h2,
        },
        "h3_noninferior_accumulation": {
            "brier_mid_b2": mid2,
            "brier_mid_b3": mid3,
            "brier_final_b2": fin2,
            "brier_final_b3": fin3,
            "delta": config.delta,
            "pass": h3,
        },
        "diagnostic": {
            "brier_frac_b2": {str(frac): brier_at_frac(s2, labels, frac) for frac in BRIER_FRACS},
            "brier_frac_b3": {str(frac): brier_at_frac(s3, labels, frac) for frac in BRIER_FRACS},
            "event_aligned_c0_switch_b3": events,
            "mean_fp_dwell_c0_b3": float(np.mean(dwells)) if dwells else math.nan,
            "median_recovery_c0_switch_b3": float(np.nanmedian(recoveries)) if recoveries else math.nan,
            "tdet_abandoned": True,
        },
        "v7b1_go": bool(h1 and h2 and h3),
        "same_architecture": True,
        "forgetting_penalty": False,
        "uses_phase_in_model": False,
        "unlocks_v7c": False,
        "unlocks_rs5b": False,
    }


def run_r3_v7b1(
    output: str | Path,
    *,
    v7b_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7B1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7B1Config()
    _require_v7b(Path(v7b_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    trace_cfg = _trace_config(cfg)
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
            for script, scale in PULL_JOBS + PUSH_JOBS + (("switch_cycle", 1.0),)
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[V7B.1 {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 9)
        torch.manual_seed(int(seed) + 9)
        if domain == "mode_a":
            row = _mode_a_trace(
                intake, seed=int(seed), regime=str(regime), amp_scale=float(amp), config=trace_cfg
            )
        else:
            row = _contact_trace(
                intake,
                c0,
                seed=int(seed),
                regime=str(regime),
                script=str(script),
                scale=float(scale),
                config=trace_cfg,
            )
        rows.append(row)
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "v7b1_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; IBS GO not evaluated",
        }
    )
    status = stage_status()
    status["R3-V7B"] = {"frozen": True, "passed": False, "memory_helps": True}
    status["R3-V7B.1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7b1_go")),
        "smoke_only": smoke,
    }
    status["R3-V7C"] = {"locked": True, "opened": False}
    status["R3-V7D"] = {"locked": True, "opened": False}
    slim = [
        {
            key: row[key]
            for key in (
                "family",
                "domain",
                "seed",
                "regime",
                "script",
                "scale",
                "y_inadequate",
                "D0",
                "T",
                "quiet_start",
                *FEATURE_NAMES,
            )
            if key in row
        }
        for row in rows
    ]
    summary = {
        "stage": "R3-V7B.1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "architecture": "GRU32 identical to V7B B2",
            "objective_b2": "time-weighted BCE",
            "objective_b3": "episode-balanced IBS",
            "delta": cfg.delta,
            "no_forgetting_penalty": True,
            "no_phase_in_model": True,
            "v7c": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7b1_go": bool(aggregate.get("v7b1_go")),
        "v7b_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

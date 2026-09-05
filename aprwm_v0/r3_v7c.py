"""R3-V7C: evidence-warranted GRU with oracle / sensorized / no contact."""

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
from .r1_rs2 import LEARNER_SENSOR_KEYS, R1RS2C0Config
from .r1_rs3a import _jsonable
from .r1_rs4a import FEATURE_NAMES, MODE_A_AMPS, PULL_JOBS, PUSH_JOBS
from .r3_v7b import EpiGRU, _gru_probs
from .r3_v7b1 import brier_at_frac, c0_brier_mass
from .r3_v7b3 import (
    C0_STEP_FPR_MAX,
    DELTA,
    W_HIGH,
    W_LOW,
    R3V7B3Config,
    _contact_pair,
    _mode_a_pair,
    _train,
    spearman,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7C_PREREG.md"
SEEDS = (21101, 21111, 21121, 21131, 21141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
CONTACT_INSTANT_COL = 5
CONTACT_WINDOW_COL = 6
SMOKE_JOBS = (
    ("mode_a", 21101, 1.5, None, 1.0),
    ("contact", 21101, math.nan, "fast_pull", 1.0),
    ("contact", 21101, math.nan, "switch_cycle", 1.0),
)
CHANNELS = ("oracle", "sensor", "none")


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7CConfig:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    w_low: float = W_LOW
    w_high: float = W_HIGH
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    target_hz: float = 20.0


def _require_v7b3(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C locked until R3-V7B.3 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C locked: V7B.3 scientific matrix has not been run")
    if not payload.get("v7b3_go"):
        raise RuntimeError("R3-V7C locked: V7B.3_GO is false")
    return payload


def zero_contact_slots(steps: np.ndarray) -> np.ndarray:
    out = np.asarray(steps, dtype=np.float64).copy()
    if out.size == 0:
        return out
    out[:, CONTACT_INSTANT_COL] = 0.0
    out[:, CONTACT_WINDOW_COL] = 0.0
    return out


def causal_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    for index in range(len(values)):
        lo = max(0, index + 1 - int(window))
        out[index] = float(np.mean(values[lo : index + 1]))
    return out


def sensor_contact_slots(
    steps: np.ndarray,
    proxy: np.ndarray,
    resid_rms: np.ndarray,
    window: int,
) -> np.ndarray:
    out = np.asarray(steps, dtype=np.float64).copy()
    n = len(out)
    if n == 0:
        return out
    proxy = np.asarray(proxy, dtype=np.float64)
    resid_rms = np.asarray(resid_rms, dtype=np.float64)
    if len(proxy) < n or len(resid_rms) < n:
        out[:, CONTACT_INSTANT_COL] = 0.0
        out[:, CONTACT_WINDOW_COL] = 0.0
        return out
    out[:, CONTACT_INSTANT_COL] = proxy[:n]
    out[:, CONTACT_WINDOW_COL] = causal_window_mean(resid_rms[:n], window)
    return out


def h4_sensor_useful(
    *,
    brier_mid_s: float,
    brier_mid_n: float,
    brier_final_s: float,
    brier_final_n: float,
    b_c0_s: float,
    b_c0_n: float,
    brier_c1_s: float,
    brier_c1_n: float,
) -> bool:
    mid_final = bool(
        math.isfinite(brier_mid_s)
        and math.isfinite(brier_mid_n)
        and math.isfinite(brier_final_s)
        and math.isfinite(brier_final_n)
        and brier_mid_s < brier_mid_n
        and brier_final_s < brier_final_n
    )
    occupancy_c1 = bool(
        math.isfinite(b_c0_s)
        and math.isfinite(b_c0_n)
        and math.isfinite(brier_c1_s)
        and math.isfinite(brier_c1_n)
        and b_c0_s < b_c0_n
        and brier_c1_s < brier_c1_n
    )
    return mid_final or occupancy_c1


def _attach_channels(rows: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    for row in rows:
        steps = np.asarray(row["steps"], dtype=np.float64)
        row["steps_oracle"] = steps
        row["steps_none"] = zero_contact_slots(steps)
        proxy = row.get("contact_proxy")
        resid = row.get("arm_tau_resid_rms")
        if proxy is None or resid is None:
            row["steps_sensor"] = row["steps_none"].copy()
        else:
            row["steps_sensor"] = sensor_contact_slots(steps, proxy, resid, window)
    return rows


def _contact_pair_sensors(
    intake: R1RS1AConfig,
    c0: R1RS2C0Config,
    *,
    seed: int,
    script: str,
    scale: float,
    config: R3V7CConfig,
) -> list[dict[str, Any]]:
    inner = R3V7B3Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
    )
    rows = _contact_pair(
        intake,
        c0,
        seed=seed,
        script=script,
        scale=scale,
        config=inner,
        include_learner_sensors=True,
    )
    return _attach_channels(rows, config.window)


def _mode_a_channels(
    intake: R1RS1AConfig, *, seed: int, amp_scale: float, config: R3V7CConfig
) -> list[dict[str, Any]]:
    inner = R3V7B3Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
    )
    rows = _mode_a_pair(intake, seed=seed, amp_scale=amp_scale, config=inner)
    return _attach_channels(rows, config.window)


def _with_steps(rows: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    key = f"steps_{channel}"
    out = []
    for row in rows:
        copy = dict(row)
        copy["steps"] = row[key]
        out.append(copy)
    return out


def _fpr(series_list: list[np.ndarray], tau: float) -> float:
    flags = np.concatenate([probs > tau for probs in series_list]) if series_list else np.zeros(0)
    return float(np.mean(flags)) if len(flags) else math.nan


def _aggregate(rows: list[dict[str, Any]], config: R3V7CConfig) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    models: dict[str, EpiGRU] = {}
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    series: dict[str, list[np.ndarray]] = {}
    for channel, seed in zip(CHANNELS, (41, 42, 43), strict=True):
        train_c = _with_steps(train, channel)
        val_c = _with_steps(val, channel)
        held_c = _with_steps(held, channel)
        stack = np.concatenate([row["steps"] for row in train_c], axis=0)
        mean = np.mean(stack, axis=0)
        std = np.where(np.std(stack, axis=0) < 1.0e-12, 1.0, np.std(stack, axis=0))
        inner = R3V7B3Config(
            hidden=config.hidden,
            window=config.window,
            epochs=config.epochs,
            lr=config.lr,
        )
        model = _train(train_c, val_c, mean, std, inner, warranted=True, seed=seed)
        models[channel] = model
        means[channel] = mean
        stds[channel] = std
        series[channel] = [_gru_probs(model, row, mean, std) for row in held_c]
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    c1_idx = [i for i, row in enumerate(held) if row["regime"] != "C0"]
    tau = float(
        np.quantile(
            [
                float(
                    _gru_probs(
                        models["sensor"],
                        {**row, "steps": row["steps_sensor"]},
                        means["sensor"],
                        stds["sensor"],
                    )[-1]
                )
                for row in train + val
                if row["regime"] == "C0"
            ],
            0.95,
        )
    )
    mid = {ch: brier_at_frac(series[ch], labels, 0.5) for ch in CHANNELS}
    fin = {ch: brier_at_frac(series[ch], labels, 1.0) for ch in CHANNELS}
    b_c0 = {}
    m_c0 = {}
    fpr = {}
    for ch in CHANNELS:
        b_c0[ch], m_c0[ch] = c0_brier_mass([series[ch][i] for i in c0_idx])
        fpr[ch] = _fpr([series[ch][i] for i in c0_idx], tau)
    c1_labels = [1] * len(c1_idx)
    c1_fin = {
        ch: brier_at_frac([series[ch][i] for i in c1_idx], c1_labels, 1.0) for ch in CHANNELS
    }
    corrs, high_p, low_p = [], [], []
    for index in c1_idx:
        w = held[index]["w_evid"][: len(series["sensor"][index])]
        p = series["sensor"][index][: len(w)]
        corrs.append(spearman(p, w))
        high = p[w > config.w_high]
        low = p[w < config.w_low]
        if len(high):
            high_p.append(float(np.mean(high)))
        if len(low):
            low_p.append(float(np.mean(low)))
    mean_corr = float(np.mean(corrs)) if corrs else math.nan
    e_high = float(np.mean(high_p)) if high_p else math.nan
    e_low = float(np.mean(low_p)) if low_p else math.nan
    h1 = bool(
        math.isfinite(mid["sensor"])
        and math.isfinite(fin["sensor"])
        and mid["sensor"] <= mid["oracle"] + config.delta
        and fin["sensor"] <= fin["oracle"] + config.delta
    )
    h2 = bool(math.isfinite(fpr["sensor"]) and fpr["sensor"] <= config.c0_step_fpr_max)
    h3 = bool(
        math.isfinite(mean_corr)
        and mean_corr > 0.0
        and math.isfinite(e_high)
        and math.isfinite(e_low)
        and e_high > e_low
    )
    h4 = h4_sensor_useful(
        brier_mid_s=mid["sensor"],
        brier_mid_n=mid["none"],
        brier_final_s=fin["sensor"],
        brier_final_n=fin["none"],
        b_c0_s=b_c0["sensor"],
        b_c0_n=b_c0["none"],
        brier_c1_s=c1_fin["sensor"],
        brier_c1_n=c1_fin["none"],
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_b5s_c0_final": tau,
        "h1_vs_oracle": {
            "brier_mid": mid,
            "brier_final": fin,
            "delta": config.delta,
            "pass": h1,
        },
        "h2_c0_occupancy": {
            "b_c0": b_c0,
            "m_c0": m_c0,
            "fpr": fpr,
            "max_fpr": config.c0_step_fpr_max,
            "pass": h2,
        },
        "h3_tracks_evidence": {
            "mean_spearman_c1": mean_corr,
            "e_p_w_high": e_high,
            "e_p_w_low": e_low,
            "pass": h3,
        },
        "h4_sensor_useful": {
            "brier_c1_final": c1_fin,
            "pass": h4,
        },
        "v7c_go": bool(h1 and h2 and h3 and h4),
        "uses_oracle_contact_at_runtime_s": False,
        "uses_jtf_at_runtime_s": False,
        "uses_phase_in_model": False,
        "uses_w_at_runtime": False,
        "same_gru_as_b5": True,
        "unlocks_v7d": False,
    }


def run_r3_v7c(
    output: str | Path,
    *,
    v7b3_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7CConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7CConfig()
    _require_v7b3(Path(v7b3_summary))
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
            ("mode_a", seed, amp, None, 1.0)
            for seed in cfg.seeds
            for amp in MODE_A_AMPS
        ] + [
            ("contact", seed, math.nan, script, scale)
            for seed in cfg.seeds
            for script, scale in PULL_JOBS + PUSH_JOBS + (("switch_cycle", 1.0),)
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, amp, script, scale = job
        print(
            f"[V7C {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 17)
        if domain == "mode_a":
            rows.extend(
                _mode_a_channels(intake, seed=int(seed), amp_scale=float(amp), config=cfg)
            )
        else:
            rows.extend(
                _contact_pair_sensors(
                    intake,
                    c0,
                    seed=int(seed),
                    script=str(script),
                    scale=float(scale),
                    config=cfg,
                )
            )
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "v7c_go": False,
            "n_episodes": len(rows),
            "n_pairs": len({row["pair_id"] for row in rows}),
            "note": "plumbing smoke only; V7C GO not evaluated",
            "sensor_keys": list(LEARNER_SENSOR_KEYS),
        }
    )
    status = stage_status()
    status["R3-V7B.3"] = {"frozen": True, "passed": True}
    status["R3-V7C"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7c_go")),
        "smoke_only": smoke,
    }
    status["R3-V7D"] = {"locked": True, "opened": False}
    slim = []
    for row in rows:
        item = {
            key: row[key]
            for key in (
                "family",
                "domain",
                "seed",
                "regime",
                "script",
                "scale",
                "pair_id",
                "y_inadequate",
                "D0",
                "T",
                *FEATURE_NAMES,
            )
            if key in row
        }
        item["w_final"] = float(row["w_evid"][-1]) if len(row["w_evid"]) else math.nan
        if row.get("contact_proxy") is not None:
            item["sensor_proxy_mean"] = float(np.mean(row["contact_proxy"]))
        slim.append(item)
    summary = {
        "stage": "R3-V7C",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "architecture": "B5 GRU32",
            "supervision": "y * w_evid",
            "variable": "oracle contact -> noisy proprioceptive contact",
            "runtime_sees_w": False,
            "runtime_sees_jtf_s": False,
            "v7d": "locked",
            "delta": cfg.delta,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7c_go": bool(aggregate.get("v7c_go")),
        "v7b3_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

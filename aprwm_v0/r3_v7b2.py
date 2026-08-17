"""R3-V7B.2: fast/slow epistemic state with persistence gating."""

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
from .r3_v7b1 import (
    BRIER_FRACS,
    brier_at_frac,
    c0_brier_mass,
    first_phase_index,
    integrated_brier,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7B2_PREREG.md"
SEEDS = (19101, 19111, 19121, 19131, 19141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
DELTA = 0.02
C0_STEP_FPR_MAX = 0.20
B2_HIDDEN = 32
B4_HIDDEN = 32
WIDE_HIDDEN = 40
EVENT_RISE = (("first_contact", 2.0), ("release", 3.0), ("recontact", 2.5))
EVENT_ALL = EVENT_RISE + (("quiet", 5.0),)
SMOKE_JOBS = (
    ("mode_a", 19101, "C0", 1.5, None, 1.0),
    ("contact", 19101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 19101, "C0", math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7B2Config:
    seeds: tuple[int, ...] = SEEDS
    hidden_b2: int = B2_HIDDEN
    hidden_b4: int = B4_HIDDEN
    hidden_wide: int = WIDE_HIDDEN
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    target_hz: float = 20.0


def _require_v7b1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7B.2 locked until R3-V7B.1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7B.2 locked: V7B.1 scientific matrix has not been run")
    return payload


def n_params_gru(dim: int, hidden: int) -> int:
    # PyTorch GRU: two bias vectors per gate.
    return 3 * (dim * hidden + hidden * hidden + 2 * hidden) + hidden + 1


def n_params_fastslow(dim: int, hidden: int) -> int:
    gru = 3 * (dim * hidden + hidden * hidden + 2 * hidden)
    gate = 3 * hidden + 1
    cand = 2 * hidden * hidden + hidden
    head = hidden + 1
    return gru + gate + cand + head


def event_delta_p(probs: np.ndarray, phase: np.ndarray, value: float) -> float:
    index = first_phase_index(phase, value)
    if index < 0:
        return math.nan
    pre_lo = max(0, index - 4)
    pre = probs[pre_lo:index]
    post = probs[index : min(len(probs), index + 3)]
    if len(pre) == 0 or len(post) == 0:
        return math.nan
    return float(max(0.0, float(np.mean(post)) - float(np.mean(pre))))


def max_event_rise(probs: np.ndarray, phase: np.ndarray) -> float:
    rises = [event_delta_p(probs, phase, value) for _, value in EVENT_RISE]
    finite = [value for value in rises if math.isfinite(value)]
    return float(max(finite)) if finite else math.nan


class FastSlowEpi(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fast = nn.GRU(dim, hidden, batch_first=True)
        self.gate = nn.Linear(3 * hidden, 1)
        self.candidate = nn.Linear(2 * hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(
        self, padded: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        fast_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
            self.fast(packed)[0], batch_first=True
        )
        batch, steps, hidden = fast_seq.shape
        slow = padded.new_zeros(batch, hidden)
        fast_prev = padded.new_zeros(batch, hidden)
        probs = []
        lengths = lengths.to(padded.device)
        for time in range(steps):
            live = time < lengths
            u_fast = fast_seq[:, time]
            gate = torch.sigmoid(
                self.gate(torch.cat((u_fast, fast_prev, slow), dim=-1))
            )
            cand = torch.tanh(self.candidate(torch.cat((slow, u_fast), dim=-1)))
            new_slow = (1.0 - gate) * slow + gate * cand
            slow = torch.where(live.unsqueeze(-1), new_slow, slow)
            probs.append(torch.sigmoid(self.head(slow)).squeeze(-1))
            fast_prev = torch.where(live.unsqueeze(-1), u_fast, fast_prev)
        return torch.stack(probs, dim=1), fast_seq


def count_module(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def _bce_train(
    model: nn.Module,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7B2Config,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(row["steps"] - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            out = model(padded, lengths)
            pred = out[0] if isinstance(out, tuple) else out
            losses = []
            for index, row in enumerate(chunk):
                t_len = int(lengths[index])
                time = torch.linspace(0.0, 1.0, t_len)
                weight = 0.3 + 0.7 * time
                target = torch.full((t_len,), float(row["y_inadequate"]))
                bce = nn.functional.binary_cross_entropy(
                    pred[index, :t_len], target, reduction="none"
                )
                losses.append(torch.sum(weight * bce) / float(t_len))
            torch.stack(losses).mean().backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-config.lr)
                        param.grad.zero_()
        model.eval()
        with torch.no_grad():
            finals = []
            labels = []
            for row in val:
                probs = _predict_series(model, row, mean, std)
                finals.append(float(probs[-1]))
                labels.append(int(row["y_inadequate"]))
            score = _brier(np.asarray(finals), np.asarray(labels, dtype=np.float64))
        if score + 1.0e-6 < best:
            best = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_series(
    model: nn.Module, row: dict[str, Any], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    feat = (row["steps"] - mean) / std
    padded, lengths = _pad([feat])
    with torch.no_grad():
        out = model(padded, lengths)
        pred = out[0] if isinstance(out, tuple) else out
        return pred[0, : int(lengths[0])].numpy().astype(np.float64)


def _fast_norm_series(
    model: FastSlowEpi, row: dict[str, Any], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    feat = (row["steps"] - mean) / std
    padded, lengths = _pad([feat])
    with torch.no_grad():
        _probs, fast = model(padded, lengths)
        norms = torch.linalg.vector_norm(fast[0, : int(lengths[0])], dim=-1)
        return norms.numpy().astype(np.float64)


def _series_list(
    model: nn.Module, rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray
) -> list[np.ndarray]:
    return [_predict_series(model, row, mean, std) for row in rows]


def _c0_switch(rows: list[dict[str, Any]], series: list[np.ndarray]) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [
        (row, series[index])
        for index, row in enumerate(rows)
        if row["regime"] == "C0" and row["family"] == "contact_switch"
    ]


def _aggregate(rows: list[dict[str, Any]], config: R3V7B2Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    step_stack = np.concatenate([row["steps"] for row in train], axis=0)
    step_mean = np.mean(step_stack, axis=0)
    step_std = np.where(np.std(step_stack, axis=0) < 1.0e-12, 1.0, np.std(step_stack, axis=0))
    b2 = _bce_train(EpiGRU(STEP_DIM, config.hidden_b2), train, val, step_mean, step_std, config, 21)
    wide = _bce_train(
        EpiGRU(STEP_DIM, config.hidden_wide), train, val, step_mean, step_std, config, 22
    )
    b4 = _bce_train(
        FastSlowEpi(STEP_DIM, config.hidden_b4), train, val, step_mean, step_std, config, 23
    )
    counts = {
        "b2": count_module(b2),
        "b2_wide": count_module(wide),
        "b4": count_module(b4),
        "formula_b2": n_params_gru(STEP_DIM, config.hidden_b2),
        "formula_wide": n_params_gru(STEP_DIM, config.hidden_wide),
        "formula_b4": n_params_fastslow(STEP_DIM, config.hidden_b4),
    }
    s2 = _series_list(b2, held, step_mean, step_std)
    sw = _series_list(wide, held, step_mean, step_std)
    s4 = _series_list(b4, held, step_mean, step_std)
    dev = train + val
    tau = float(
        np.quantile(
            [float(_predict_series(b2, row, step_mean, step_std)[-1]) for row in dev if row["regime"] == "C0"],
            0.95,
        )
    )
    s2_dev = _series_list(b2, dev, step_mean, step_std)
    eta_vals = [max_event_rise(ser, row["phase"]) for row, ser in _c0_switch(dev, s2_dev)]
    eta_vals = [value for value in eta_vals if math.isfinite(value)]
    eta = float(np.median(eta_vals)) if eta_vals else math.nan
    mid2, mid4 = brier_at_frac(s2, labels, 0.5), brier_at_frac(s4, labels, 0.5)
    fin2, fin4 = brier_at_frac(s2, labels, 1.0), brier_at_frac(s4, labels, 1.0)
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    b_c0_2, m_c0_2 = c0_brier_mass([s2[i] for i in c0_idx])
    b_c0_w, m_c0_w = c0_brier_mass([sw[i] for i in c0_idx])
    b_c0_4, m_c0_4 = c0_brier_mass([s4[i] for i in c0_idx])
    flags4 = np.concatenate([s4[i] > tau for i in c0_idx]) if c0_idx else np.zeros(0)
    fpr4 = float(np.mean(flags4)) if len(flags4) else math.nan
    held_switch = _c0_switch(held, s4)
    delta4 = [max_event_rise(ser, row["phase"]) for row, ser in held_switch]
    delta4 = [value for value in delta4 if math.isfinite(value)]
    med_delta4 = float(np.median(delta4)) if delta4 else math.nan
    h1 = bool(
        math.isfinite(mid4)
        and math.isfinite(fin4)
        and mid4 <= mid2 + config.delta
        and fin4 <= fin2 + config.delta
        and (mid4 < mid2 or fin4 < fin2)
    )
    h2 = bool(
        math.isfinite(b_c0_4)
        and math.isfinite(b_c0_2)
        and b_c0_4 < b_c0_2
        and math.isfinite(fpr4)
        and fpr4 <= config.c0_step_fpr_max
    )
    h3 = bool(math.isfinite(med_delta4) and math.isfinite(eta) and med_delta4 < eta)
    h4 = bool(math.isfinite(b_c0_4) and math.isfinite(b_c0_w) and b_c0_4 < b_c0_w)
    events = {}
    for name, value in EVENT_ALL:
        pvals, fnorms = [], []
        for row, ser in held_switch:
            idx = first_phase_index(row["phase"], value)
            pvals.append(float(ser[idx]) if idx >= 0 else math.nan)
            if isinstance(b4, FastSlowEpi) and idx >= 0:
                norms = _fast_norm_series(b4, row, step_mean, step_std)
                fnorms.append(float(norms[idx]))
        events[name] = {
            "mean_p_slow": float(np.nanmean(pvals)) if pvals else math.nan,
            "mean_fast_norm": float(np.nanmean(fnorms)) if fnorms else math.nan,
        }
    dwells = [mean_positive_dwell(s4[i] > tau) for i in c0_idx]
    recoveries = [
        recovery_time(s4[i], tau, int(held[i]["quiet_start"]))
        for i in c0_idx
        if held[i]["family"] == "contact_switch"
    ]
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "param_counts": counts,
        "tau_dev_b2_c0_final": tau,
        "eta_dev_b2_median_event_rise": eta,
        "delta": config.delta,
        "h1_accumulation_vs_b2": {
            "brier_mid_b2": mid2,
            "brier_mid_b4": mid4,
            "brier_final_b2": fin2,
            "brier_final_b4": fin4,
            "brier_final_wide": brier_at_frac(sw, labels, 1.0),
            "pass": h1,
        },
        "h2_c0_occupancy_vs_b2": {
            "b_c0_b2": b_c0_2,
            "b_c0_b4": b_c0_4,
            "m_c0_b2": m_c0_2,
            "m_c0_b4": m_c0_4,
            "fpr_b4": fpr4,
            "max_fpr": config.c0_step_fpr_max,
            "pass": h2,
        },
        "h3_event_bounded_slow": {
            "eta": eta,
            "median_max_rise_b4": med_delta4,
            "pass": h3,
        },
        "h4_capacity_control": {
            "b_c0_wide": b_c0_w,
            "b_c0_b4": b_c0_4,
            "m_c0_wide": m_c0_w,
            "pass": h4,
        },
        "diagnostic": {
            "ibs_b2": integrated_brier(s2, labels),
            "ibs_wide": integrated_brier(sw, labels),
            "ibs_b4": integrated_brier(s4, labels),
            "brier_frac_b4": {str(frac): brier_at_frac(s4, labels, frac) for frac in BRIER_FRACS},
            "event_aligned_c0_switch_b4": events,
            "mean_fp_dwell_c0_b4": float(np.mean(dwells)) if dwells else math.nan,
            "median_recovery_c0_switch_b4": float(np.nanmedian(recoveries)) if recoveries else math.nan,
            "n_dev_eta_episodes": len(eta_vals),
        },
        "v7b2_go": bool(h1 and h2 and h3 and h4),
        "uses_phase_in_model": False,
        "same_bce_objective": True,
        "unlocks_v7c": False,
        "unlocks_rs5b": False,
    }


def run_r3_v7b2(
    output: str | Path,
    *,
    v7b1_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7B2Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7B2Config()
    _require_v7b1(Path(v7b1_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    trace_cfg = R3V7BConfig(
        hidden=cfg.hidden_b2, window=cfg.window, epochs=cfg.epochs, lr=cfg.lr, target_hz=cfg.target_hz
    )
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
            f"[V7B.2 {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 11)
        torch.manual_seed(int(seed) + 11)
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
            "v7b2_go": False,
            "n_episodes": len(rows),
            "param_formulas": {
                "b2": n_params_gru(STEP_DIM, cfg.hidden_b2),
                "wide": n_params_gru(STEP_DIM, cfg.hidden_wide),
                "b4": n_params_fastslow(STEP_DIM, cfg.hidden_b4),
            },
            "note": "plumbing smoke only; fast/slow GO not evaluated",
        }
    )
    status = stage_status()
    status["R3-V7B.1"] = {"frozen": True, "passed": False, "ibs_insufficient": True}
    status["R3-V7B.2"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7b2_go")),
        "smoke_only": smoke,
    }
    status["R3-V7C"] = {"locked": True, "opened": False}
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
        "stage": "R3-V7B.2",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "p_from_slow_only": True,
            "persistence_gate": True,
            "objective": "time-weighted BCE (same as V7B B2)",
            "no_phase_in_model": True,
            "delta": cfg.delta,
            "capacity_control": "B2-wide hidden 40",
            "v7c": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7b2_go": bool(aggregate.get("v7b2_go")),
        "v7b1_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

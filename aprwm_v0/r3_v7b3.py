"""R3-V7B.3: evidence-warranted supervision on matched C0/C1 pairs."""

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
from .r1_rs1 import _observe, _sample_door_points, _torque_series
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a3 import _probe_spec, _rollout_with_tangent
from .r1_rs1a4 import A0, FREQ_HZ
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import LEARNER_SENSOR_KEYS, REGIME_ALPHA, DoorContactC0Backend, R1RS2C0Config
from .r1_rs3a import _jsonable
from .r1_rs4a import FEATURE_NAMES, MODE_A_AMPS, PULL_JOBS, PUSH_JOBS, REGIMES, _brier
from .r3_v7b import (
    EpiGRU,
    R3V7BConfig,
    STEP_DIM,
    _build_trace,
    _downsample,
    _gru_probs,
    _pad,
)
from .r3_v7b1 import brier_at_frac, c0_brier_mass
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3_V7B3_PREREG.md"
SEEDS = (20101, 20111, 20121, 20131, 20141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
DELTA = 0.02
GAMMA = 0.05
W_LOW = 0.25
W_HIGH = 0.75
C0_STEP_FPR_MAX = 0.20
C1_REGIMES = ("C1-L", "C1-H")
SMOKE_JOBS = (
    ("mode_a", 20101, 1.5, None, 1.0),
    ("contact", 20101, math.nan, "fast_pull", 1.0),
    ("contact", 20101, math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7B3Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    gamma: float = GAMMA
    w_low: float = W_LOW
    w_high: float = W_HIGH
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    target_hz: float = 20.0


def _require_v7b2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7B.3 locked until R3-V7B.2 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7B.3 locked: V7B.2 scientific matrix has not been run")
    return payload


def evidence_weights(residual_c0: np.ndarray, residual_c1: np.ndarray) -> np.ndarray:
    a = np.asarray(residual_c0, dtype=np.float64)
    b = np.asarray(residual_c1, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    delta = (b[:n] - a[:n]) ** 2
    cumul = np.cumsum(delta)
    denom = float(cumul[-1]) + 1.0e-12
    return cumul / denom


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1])


def _family(script: str | None) -> str:
    if script is None:
        return "mode_a"
    if script == "pull_push":
        return "contact_push"
    if script == "switch_cycle":
        return "contact_switch"
    return "contact_pull"


def _pack_arrays(
    *,
    q: np.ndarray,
    velocity: np.ndarray,
    residual: np.ndarray,
    density: np.ndarray,
    contact: np.ndarray,
    phase: np.ndarray,
    action: np.ndarray,
    intake: R1RS1AConfig,
    window: int,
    script: str | None,
    seed: int,
    regime: str,
    scale: float,
    pair_id: str,
    residual_raw: np.ndarray,
) -> dict[str, Any]:
    n = min(len(q), len(velocity), len(residual), len(density), len(contact), len(phase), len(action))
    trace = _build_trace(
        q=q[:n],
        velocity=velocity[:n],
        residual=residual[:n],
        density=density[:n],
        contact=contact[:n],
        phase=phase[:n],
        action=action[:n],
        intake=intake,
        window=window,
        script=script,
    )
    return {
        "domain": "mode_a" if script is None else "contact",
        "family": _family(script),
        "seed": int(seed),
        "regime": regime,
        "script": script,
        "scale": float(scale),
        "pair_id": pair_id,
        "y_inadequate": int(regime != "C0"),
        "residual_raw": np.asarray(residual_raw[:n], dtype=np.float64),
        "w_evid": np.zeros(n, dtype=np.float64),
        **trace,
        **trace["geom"],
    }


def _mode_a_pair(
    intake: R1RS1AConfig,
    *,
    seed: int,
    amp_scale: float,
    config: R3V7B3Config,
    include_rgb: bool = False,
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed + 81_000 + int(10 * amp_scale))
    n_steps = int(round(intake.duration_s / intake.timestep))
    torques = _torque_series(
        _probe_spec(A0 * float(amp_scale), FREQ_HZ), n_steps, intake.timestep, seed=seed + 91
    )
    stride = max(1, int(round((1.0 / config.target_hz) / intake.timestep)))
    pair_id = f"mode_a:{seed}:{amp_scale}"
    rows = []
    state = None
    for regime in REGIMES:
        alpha = float(REGIME_ALPHA[regime])
        backend = DoorModeABackend(
            regime="C0" if abs(alpha) < 1.0e-15 else "C1-L",
            friction=intake.friction,
            damping=intake.damping,
            seed=seed,
            alpha=alpha,
            include_rgb=include_rgb,
        )
        try:
            if state is None:
                state = _sample_door_points(intake.passive_context, "passive", generator)
            _observe(backend, state)
            probe = _rollout_with_tangent(backend, torques, intake.joint_limit_margin)
            q = _downsample(probe["q"], stride)
            velocity = _downsample(probe["qvel"], stride)
            residual = _downsample(probe["residual"], stride)
            density = _downsample(probe["density_tangent"], stride)
            action = _downsample(np.asarray(torques, dtype=np.float64), stride)
            n = min(len(q), len(action))
            packed = _pack_arrays(
                q=q[:n],
                velocity=velocity[:n],
                residual=residual[:n],
                density=density[:n],
                contact=np.zeros(n),
                phase=np.full(n, 2.0),
                action=action[:n],
                intake=intake,
                window=config.window,
                script=None,
                seed=seed,
                regime=regime,
                scale=amp_scale,
                pair_id=pair_id,
                residual_raw=residual[:n],
            )
            if include_rgb:
                packed["rgb"] = backend.rgb_along_trace(q[:n], velocity[:n])
            rows.append(packed)
        finally:
            backend.close()
    return _attach_weights(rows)


def _contact_pair(
    intake: R1RS1AConfig,
    c0: R1RS2C0Config,
    *,
    seed: int,
    script: str,
    scale: float,
    config: R3V7B3Config,
    include_learner_sensors: bool = False,
    include_tactile: bool = False,
    include_rgb: bool = False,
) -> list[dict[str, Any]]:
    pair_id = f"contact:{seed}:{script}:{scale}"
    backend = DoorContactC0Backend(c0, seed=int(seed), regime="C0", include_rgb=include_rgb)
    try:
        backend.prepare()
        recorded = backend.rollout(
            script,
            scale=float(scale),
            include_learner_sensors=include_learner_sensors,
            include_tactile=include_tactile,
            include_rgb=include_rgb,
        )
        actions = np.asarray(recorded["actions"], dtype=np.float64)
        n_ctrl = max(len(actions), 1)
        stride = max(1, int(len(recorded["arrays"]["q"]) // n_ctrl))
        phases = _downsample(recorded["arrays"]["phase"], stride)[:n_ctrl]
        packed = [_arrays_to_row(recorded, intake, config, seed, "C0", script, scale, pair_id)]
        for regime in C1_REGIMES:
            backend.alpha = float(REGIME_ALPHA[regime])
            backend.regime = regime
            replayed = backend.replay_rollout(
                actions,
                phases=phases,
                include_learner_sensors=include_learner_sensors,
                include_tactile=include_tactile,
                include_rgb=include_rgb,
            )
            packed.append(
                _arrays_to_row(replayed, intake, config, seed, regime, script, scale, pair_id)
            )
    finally:
        backend.close()
    return _attach_weights(packed)


def _arrays_to_row(
    result: dict[str, Any],
    intake: R1RS1AConfig,
    config: R3V7B3Config,
    seed: int,
    regime: str,
    script: str,
    scale: float,
    pair_id: str,
) -> dict[str, Any]:
    arrays = result["arrays"]
    actions = np.asarray(result["actions"], dtype=np.float64)
    n_ctrl = max(len(actions), 1)
    stride = max(1, int(len(arrays["q"]) // n_ctrl))
    q = _downsample(arrays["q"], stride)
    velocity = _downsample(arrays["qvel"], stride)
    residual = _downsample(arrays["residual"], stride)
    density = _downsample(arrays["density_tangent"], stride)
    contact = _downsample(arrays["contact_count"], stride)
    phase = _downsample(arrays["phase"], stride)
    n = min(len(q), n_ctrl)
    signed = np.sign(velocity[:n])
    amp = np.linalg.norm(actions[:n, :3], axis=1) if actions.ndim == 2 else np.zeros(n)
    packed = _pack_arrays(
        q=q[:n],
        velocity=velocity[:n],
        residual=residual[:n],
        density=density[:n],
        contact=contact[:n],
        phase=phase[:n],
        action=signed * amp[:n],
        intake=intake,
        window=config.window,
        script=script,
        seed=seed,
        regime=regime,
        scale=scale,
        pair_id=pair_id,
        residual_raw=residual[:n],
    )
    for key in LEARNER_SENSOR_KEYS:
        if key in arrays and len(np.asarray(arrays[key])) >= n:
            packed[key] = _downsample(arrays[key], stride)[:n]
    if "taxel" in arrays:
        taxel = _downsample(arrays["taxel"], stride)[:n]
        packed["taxel"] = np.asarray(taxel, dtype=np.float64)
    for key in ("taxel_shear", "tactile_geom", "tactile_surf"):
        if key in arrays:
            packed[key] = _downsample(arrays[key], stride)[:n]
    if "rgb" in arrays:
        rgb = np.asarray(arrays["rgb"])
        packed["rgb"] = rgb[:n] if len(rgb) >= n else rgb
    return packed


def _attach_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_reg = {row["regime"]: row for row in rows}
    c0 = by_reg["C0"]
    for row in rows:
        if row["regime"] == "C0":
            row["w_evid"] = np.zeros(int(row["T"]), dtype=np.float64)
            row["target"] = np.zeros(int(row["T"]), dtype=np.float64)
        else:
            weights = evidence_weights(c0["residual_raw"], row["residual_raw"])
            n = min(int(row["T"]), len(weights))
            w = np.zeros(int(row["T"]), dtype=np.float64)
            w[:n] = weights[:n]
            row["w_evid"] = w
            row["target"] = w.copy()
    return rows


def _train(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7B3Config,
    *,
    warranted: bool,
    seed: int,
) -> EpiGRU:
    torch.manual_seed(seed)
    model = EpiGRU(STEP_DIM, config.hidden)
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
            pred = model(padded, lengths)
            losses = []
            for index, row in enumerate(chunk):
                t_len = int(lengths[index])
                p_t = pred[index, :t_len]
                if warranted:
                    target = torch.tensor(row["target"][:t_len], dtype=torch.float32)
                    losses.append(nn.functional.binary_cross_entropy(p_t, target))
                else:
                    time = torch.linspace(0.0, 1.0, t_len)
                    weight = 0.3 + 0.7 * time
                    y = torch.full((t_len,), float(row["y_inadequate"]))
                    bce = nn.functional.binary_cross_entropy(p_t, y, reduction="none")
                    losses.append(torch.sum(weight * bce) / float(t_len))
            torch.stack(losses).mean().backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-config.lr)
                        param.grad.zero_()
        model.eval()
        with torch.no_grad():
            scores = []
            for row in val:
                probs = _gru_probs(model, row, mean, std)
                if warranted:
                    target = row["target"][: len(probs)]
                    scores.append(float(np.mean((probs - target) ** 2)))
                else:
                    scores.append(float((probs[-1] - row["y_inadequate"]) ** 2))
            score = float(np.mean(scores)) if scores else math.inf
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


def _aggregate(rows: list[dict[str, Any]], config: R3V7B3Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    step_stack = np.concatenate([row["steps"] for row in train], axis=0)
    step_mean = np.mean(step_stack, axis=0)
    step_std = np.where(np.std(step_stack, axis=0) < 1.0e-12, 1.0, np.std(step_stack, axis=0))
    b2 = _train(train, val, step_mean, step_std, config, warranted=False, seed=31)
    b5 = _train(train, val, step_mean, step_std, config, warranted=True, seed=32)
    s2 = [_gru_probs(b2, row, step_mean, step_std) for row in held]
    s5 = [_gru_probs(b5, row, step_mean, step_std) for row in held]
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
    mid2, mid5 = brier_at_frac(s2, labels, 0.5), brier_at_frac(s5, labels, 0.5)
    fin2, fin5 = brier_at_frac(s2, labels, 1.0), brier_at_frac(s5, labels, 1.0)
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    c1_idx = [i for i, row in enumerate(held) if row["regime"] != "C0"]
    b_c0_2, m_c0_2 = c0_brier_mass([s2[i] for i in c0_idx])
    b_c0_5, m_c0_5 = c0_brier_mass([s5[i] for i in c0_idx])
    flags = np.concatenate([s5[i] > tau for i in c0_idx]) if c0_idx else np.zeros(0)
    fpr = float(np.mean(flags)) if len(flags) else math.nan
    corrs = []
    high_p, low_p = [], []
    for index in c1_idx:
        w = held[index]["w_evid"][: len(s5[index])]
        p = s5[index][: len(w)]
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
    early_abs, late_diff, early_diff = [], [], []
    by_pair: dict[str, dict[str, tuple[dict[str, Any], np.ndarray]]] = {}
    for index, row in enumerate(held):
        by_pair.setdefault(row["pair_id"], {})[row["regime"]] = (row, s5[index])
    for pair in by_pair.values():
        if "C0" not in pair:
            continue
        p0 = pair["C0"][1]
        for regime in C1_REGIMES:
            if regime not in pair:
                continue
            row1, p1 = pair[regime]
            n = min(len(p0), len(p1), len(row1["w_evid"]))
            w = row1["w_evid"][:n]
            d = p1[:n] - p0[:n]
            early = w < config.w_low
            late = w > config.w_high
            if np.any(early):
                early_abs.append(float(np.mean(np.abs(d[early]))))
                early_diff.append(float(np.mean(d[early])))
            if np.any(late):
                late_diff.append(float(np.mean(d[late])))
    mean_early_abs = float(np.mean(early_abs)) if early_abs else math.nan
    mean_early_d = float(np.mean(early_diff)) if early_diff else math.nan
    mean_late_d = float(np.mean(late_diff)) if late_diff else math.nan
    h1 = bool(
        math.isfinite(mid5)
        and math.isfinite(fin5)
        and mid5 <= mid2 + config.delta
        and fin5 <= fin2 + config.delta
    )
    h2 = bool(
        math.isfinite(b_c0_5)
        and math.isfinite(b_c0_2)
        and b_c0_5 < b_c0_2
        and math.isfinite(fpr)
        and fpr <= config.c0_step_fpr_max
    )
    h3 = bool(math.isfinite(mean_corr) and mean_corr > 0.0 and math.isfinite(e_high) and math.isfinite(e_low) and e_high > e_low)
    h4 = bool(
        math.isfinite(mean_early_abs)
        and mean_early_abs <= 0.20
        and math.isfinite(mean_late_d)
        and math.isfinite(mean_early_d)
        and mean_late_d > mean_early_d + config.gamma
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_b2_c0_final": tau,
        "h1_vs_b2": {
            "brier_mid_b2": mid2,
            "brier_mid_b5": mid5,
            "brier_final_b2": fin2,
            "brier_final_b5": fin5,
            "pass": h1,
        },
        "h2_c0_occupancy": {
            "b_c0_b2": b_c0_2,
            "b_c0_b5": b_c0_5,
            "m_c0_b2": m_c0_2,
            "m_c0_b5": m_c0_5,
            "fpr_b5": fpr,
            "max_fpr": config.c0_step_fpr_max,
            "pass": h2,
        },
        "h3_tracks_evidence": {
            "mean_spearman_c1": mean_corr,
            "e_p_w_high": e_high,
            "e_p_w_low": e_low,
            "pass": h3,
        },
        "h4_matched_gap": {
            "mean_abs_early": mean_early_abs,
            "mean_diff_early": mean_early_d,
            "mean_diff_late": mean_late_d,
            "gamma": config.gamma,
            "pass": h4,
        },
        "v7b3_go": bool(h1 and h2 and h3 and h4),
        "uses_counterfactual_at_runtime": False,
        "uses_phase_in_model": False,
        "same_gru_as_b2": True,
        "unlocks_v7c": False,
        "stop_oracle_door_recurrent_if_h2_fails": True,
    }


def run_r3_v7b3(
    output: str | Path,
    *,
    v7b2_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7B3Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7B3Config()
    _require_v7b2(Path(v7b2_summary))
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
            f"[V7B.3 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 13)
        if domain == "mode_a":
            rows.extend(
                _mode_a_pair(intake, seed=int(seed), amp_scale=float(amp), config=cfg)
            )
        else:
            rows.extend(
                _contact_pair(
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
            "v7b3_go": False,
            "n_episodes": len(rows),
            "n_pairs": len({row["pair_id"] for row in rows}),
            "note": "plumbing smoke only; warranted GO not evaluated",
        }
    )
    status = stage_status()
    status["R3-V7B.2"] = {"frozen": True, "passed": False}
    status["R3-V7B.3"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7b3_go")),
        "smoke_only": smoke,
    }
    status["R3-V7C"] = {"locked": True, "opened": False}
    slim = []
    for row in rows:
        slim.append(
            {
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
        )
        slim[-1]["w_final"] = float(row["w_evid"][-1]) if len(row["w_evid"]) else math.nan
    summary = {
        "stage": "R3-V7B.3",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "architecture": "B2 GRU32",
            "supervision_b5": "y * w_evid from matched residual divergence",
            "runtime_sees_counterfactual": False,
            "delta": cfg.delta,
            "gamma": cfg.gamma,
            "v7c": "locked",
            "stop_if_h2_fails": True,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7b3_go": bool(aggregate.get("v7b3_go")),
        "v7b2_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

"""R3-V7B: persistent contextual epistemic belief vs V7A static summary."""

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .r1_mj import stage_status
from .r1_rs1 import _observe, _sample_door_points, _torque_series
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a3 import _probe_spec, _rollout_with_tangent
from .r1_rs1a4 import A0, FREQ_HZ
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import REGIME_ALPHA, DoorContactC0Backend, R1RS2C0Config
from .r1_rs3a import _after_theta, _fit_theta, _jsonable, _perp
from .r1_rs4a import (
    FEATURE_NAMES,
    MODE_A_AMPS,
    PULL_JOBS,
    PUSH_JOBS,
    REGIMES,
    _brier,
    _d0_on_window,
    _feature_matrix,
    _fit_logit,
    _predict_logit,
    _probe_mask,
    _standardize,
    geometry_features,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7B_PREREG.md"
SEEDS = (17101, 17111, 17121, 17131, 17141)
DEV_SEEDS = SEEDS[:3]
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
TARGET_HZ = 20.0
WINDOW = 16
HIDDEN = 32
LOGIT_RIDGE = 1.0
C0_STEP_FPR_MAX = 0.20
SMOKE_JOBS = (
    ("mode_a", 17101, "C0", 1.5, None, 1.0),
    ("contact", 17101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 17101, "C0", math.nan, "switch_cycle", 1.0),
)
STEP_DIM = 12


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7BConfig:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = HIDDEN
    window: int = WINDOW
    logit_ridge: float = LOGIT_RIDGE
    epochs: int = 60
    lr: float = 1.0e-2
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    target_hz: float = TARGET_HZ


def _require_v7a(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7B locked until R3-V7A has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7B locked: V7A scientific matrix has not been run")
    if not payload.get("v7a_go"):
        raise RuntimeError("R3-V7B locked: V7A_GO is not true")
    return payload


def detection_time(probs: np.ndarray, tau: float) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    hits = np.where(probs > tau)[0]
    return int(hits[0]) if len(hits) else int(len(probs))


def mean_positive_dwell(flags: np.ndarray) -> float:
    flags = np.asarray(flags, dtype=bool)
    if not np.any(flags):
        return 0.0
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def recovery_time(probs: np.ndarray, tau: float, quiet_start: int) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    if quiet_start <= 0 or quiet_start >= len(probs):
        return math.nan
    if not np.any(probs[:quiet_start] > tau):
        return 0.0
    rest = probs[quiet_start:]
    hits = np.where(rest <= tau)[0]
    return float(hits[0] if len(hits) else len(rest))


def _family(script: str | None) -> str:
    if script is None:
        return "mode_a"
    if script == "pull_push":
        return "contact_push"
    if script == "switch_cycle":
        return "contact_switch"
    return "contact_pull"


def _downsample(values: np.ndarray, stride: int) -> np.ndarray:
    values = np.asarray(values)
    if stride <= 1:
        return values
    return values[stride - 1 :: stride]


def _local_stats(
    q: np.ndarray,
    velocity: np.ndarray,
    r_perp: np.ndarray,
    contact: np.ndarray,
    tangent: np.ndarray,
    index: int,
    window: int,
) -> np.ndarray:
    lo = max(0, index + 1 - window)
    q_w = q[lo : index + 1]
    v_w = velocity[lo : index + 1]
    r_w = r_perp[lo : index + 1]
    c_w = contact[lo : index + 1]
    tang_w = tangent[lo : index + 1]
    pos = float(np.mean(v_w > 0.0))
    neg = float(np.mean(v_w < 0.0))
    sv_min = float(np.linalg.svd(tang_w, compute_uv=False).min()) if len(tang_w) else 0.0
    return np.asarray(
        [
            float(np.mean(c_w > 0.0)),
            min(pos, neg),
            float(np.max(q_w) - np.min(q_w)) if len(q_w) else 0.0,
            math.log(float(np.sqrt(np.mean(r_w * r_w))) + 1.0e-12),
            math.log(sv_min + 1.0e-12),
            math.log(float(len(q_w)) + 1.0e-12),
        ],
        dtype=np.float64,
    )


def _step_features(
    *,
    q: np.ndarray,
    velocity: np.ndarray,
    action: np.ndarray,
    r_perp: np.ndarray,
    contact: np.ndarray,
    tangent: np.ndarray,
    window: int,
) -> np.ndarray:
    rows = []
    for index in range(len(q)):
        local = _local_stats(q, velocity, r_perp, contact, tangent, index, window)
        rows.append(
            np.concatenate(
                [
                    np.asarray(
                        [
                            float(q[index]),
                            float(velocity[index]),
                            float(action[index]),
                            float(r_perp[index]),
                            math.log1p(abs(float(r_perp[index]))),
                            float(contact[index] > 0.0),
                        ],
                        dtype=np.float64,
                    ),
                    local,
                ]
            )
        )
    return np.stack(rows, axis=0)


def _causal_summaries(
    *,
    q: np.ndarray,
    velocity: np.ndarray,
    r_perp: np.ndarray,
    contact: np.ndarray,
    tangent: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index in range(len(q)):
        sl = slice(0, index + 1)
        d0 = float(np.sqrt(np.mean(r_perp[sl] * r_perp[sl])))
        rows.append(
            geometry_features(
                tangent=tangent[sl],
                q=q[sl],
                velocity=velocity[sl],
                d0=d0,
                contact_frac=float(np.mean(contact[sl] > 0.0)),
            )
        )
    return rows


def _build_trace(
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
) -> dict[str, Any]:
    tangent = np.stack((density, velocity), axis=1)
    if script is None:
        h0 = np.zeros(len(q), dtype=bool)
        h0[: max(4, int(0.2 * len(q)))] = True
        probe = np.ones(len(q), dtype=bool)
    else:
        h0 = phase < 2.0
        probe = _probe_mask(phase, script)
        if not np.any(probe):
            probe = ~h0
    if not np.any(h0):
        h0[: max(4, len(q) // 5)] = True
    mean = _fit_theta(tangent[h0], residual[h0], intake)
    r_after = _after_theta(tangent, residual, mean)
    r_perp = _perp(tangent, r_after, intake.ridge)
    d0 = _d0_on_window(tangent[probe], residual[probe], tangent[h0], residual[h0], intake)
    geom = geometry_features(
        tangent=tangent[probe],
        q=q[probe],
        velocity=velocity[probe],
        d0=d0,
        contact_frac=float(np.mean(contact[probe] > 0.0)),
    )
    steps = _step_features(
        q=q,
        velocity=velocity,
        action=action,
        r_perp=r_perp,
        contact=contact,
        tangent=tangent,
        window=window,
    )
    prefixes = _causal_summaries(
        q=q, velocity=velocity, r_perp=r_perp, contact=contact, tangent=tangent
    )
    quiet = int(np.argmax(phase >= 5.0)) if np.any(phase >= 5.0) else -1
    return {
        "D0": d0,
        "geom": geom,
        "steps": steps,
        "prefixes": prefixes,
        "phase": phase,
        "quiet_start": quiet,
        "T": int(len(q)),
    }


def _mode_a_trace(
    intake: R1RS1AConfig, *, seed: int, regime: str, amp_scale: float, config: R3V7BConfig
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    mix = int(hashlib.md5(f"v7b:{alpha}:{amp_scale}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 71_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime="C0" if abs(alpha) < 1.0e-15 else "C1-L",
        friction=intake.friction,
        damping=intake.damping,
        seed=seed,
        alpha=alpha,
    )
    try:
        state = _sample_door_points(intake.passive_context, "passive", generator)
        _physical, tangent_passive, target, _ = _observe(backend, state)
        n_steps = int(round(intake.duration_s / intake.timestep))
        torques = _torque_series(
            _probe_spec(A0 * float(amp_scale), FREQ_HZ), n_steps, intake.timestep, seed=seed + 91
        )
        probe = _rollout_with_tangent(backend, torques, intake.joint_limit_margin)
        stride = max(1, int(round((1.0 / config.target_hz) / intake.timestep)))
        q = _downsample(probe["q"], stride)
        velocity = _downsample(probe["qvel"], stride)
        residual = _downsample(probe["residual"], stride)
        density = _downsample(probe["density_tangent"], stride)
        action = _downsample(np.asarray(torques, dtype=np.float64), stride)
        n = min(len(q), len(action))
        rng = np.random.default_rng(seed + 77)
        residual = residual[:n] + intake.force_observation_noise * rng.normal(size=n)
        contact = np.zeros(n, dtype=np.float64)
        phase = np.full(n, 2.0, dtype=np.float64)
        trace = _build_trace(
            q=q[:n],
            velocity=velocity[:n],
            residual=residual,
            density=density[:n],
            contact=contact,
            phase=phase,
            action=action[:n],
            intake=intake,
            window=config.window,
            script=None,
        )
    finally:
        backend.close()
    return {
        "domain": "mode_a",
        "family": "mode_a",
        "seed": int(seed),
        "regime": regime,
        "script": None,
        "scale": float(amp_scale),
        "y_inadequate": int(regime != "C0"),
        **trace,
        **trace["geom"],
    }


def _contact_trace(
    intake: R1RS1AConfig,
    c0: R1RS2C0Config,
    *,
    seed: int,
    regime: str,
    script: str,
    scale: float,
    config: R3V7BConfig,
) -> dict[str, Any]:
    backend = DoorContactC0Backend(c0, seed=int(seed), regime=regime)
    try:
        backend.prepare()
        result = backend.rollout(script, scale=float(scale))
    finally:
        backend.close()
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
    action = signed * amp[:n]
    trace = _build_trace(
        q=q[:n],
        velocity=velocity[:n],
        residual=residual[:n],
        density=density[:n],
        contact=contact[:n],
        phase=phase[:n],
        action=action,
        intake=intake,
        window=config.window,
        script=script,
    )
    return {
        "domain": "contact",
        "family": _family(script),
        "seed": int(seed),
        "regime": regime,
        "script": script,
        "scale": float(scale),
        "y_inadequate": int(regime != "C0"),
        **trace,
        **trace["geom"],
    }


class EpiGRU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        hidden, _ = self.gru(packed)
        seq, _ = pad_packed_sequence(hidden, batch_first=True)
        return torch.sigmoid(self.head(seq).squeeze(-1))


def _pad(batch: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(item) for item in batch], dtype=torch.long)
    dim = batch[0].shape[1]
    padded = torch.zeros(len(batch), int(lengths.max()), dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, : len(item)] = torch.tensor(item, dtype=torch.float32)
    return padded, lengths


def _train_gru(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7BConfig,
) -> EpiGRU:
    model = EpiGRU(STEP_DIM, config.hidden)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_brier = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + 3).permutation(len(train))
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(row["steps"] - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            pred = model(padded, lengths)
            loss = padded.new_zeros(())
            count = 0.0
            for index, row in enumerate(chunk):
                t_len = int(lengths[index])
                time = torch.linspace(0.0, 1.0, t_len)
                weight = 0.3 + 0.7 * time
                target = torch.full((t_len,), float(row["y_inadequate"]))
                bce = nn.functional.binary_cross_entropy(
                    pred[index, :t_len], target, reduction="none"
                )
                loss = loss + torch.sum(weight * bce)
                count += float(t_len)
            (loss / max(count, 1.0)).backward()
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
                feat = (row["steps"] - mean) / std
                padded, lengths = _pad([feat])
                probs = model(padded, lengths)[0, : int(lengths[0])].numpy()
                finals.append(float(probs[-1]))
                labels.append(int(row["y_inadequate"]))
            brier = _brier(np.asarray(finals), np.asarray(labels, dtype=np.float64))
        if brier + 1.0e-6 < best_brier:
            best_brier = brier
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _gru_probs(model: EpiGRU, row: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    feat = (row["steps"] - mean) / std
    padded, lengths = _pad([feat])
    with torch.no_grad():
        return model(padded, lengths)[0, : int(lengths[0])].numpy().astype(np.float64)


def _static_probs(row: dict[str, Any], weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = _feature_matrix(row["prefixes"], FEATURE_NAMES)
    normed = (matrix - mean) / std
    return _predict_logit(normed, weights)


def _aggregate(rows: list[dict[str, Any]], config: R3V7BConfig) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    y_held = np.asarray([row["y_inadequate"] for row in held], dtype=np.float64)
    x_train = _feature_matrix(train + val, FEATURE_NAMES)
    x_held = _feature_matrix(held, FEATURE_NAMES)
    y_dev = np.asarray([row["y_inadequate"] for row in train + val], dtype=np.float64)
    z_tr, z_te = _standardize(x_train, x_held)
    w_ctx = _fit_logit(z_tr, y_dev, config.logit_ridge)
    d_tr, d_te = _standardize(
        _feature_matrix(train + val, ("log_d0",)),
        _feature_matrix(held, ("log_d0",)),
    )
    w_d0 = _fit_logit(d_tr, y_dev, config.logit_ridge)
    p_b1 = _predict_logit(z_te, w_ctx)
    p_b0 = _predict_logit(d_te, w_d0)
    prefix_mean = np.mean(x_train, axis=0)
    prefix_std = np.where(np.std(x_train, axis=0) < 1.0e-12, 1.0, np.std(x_train, axis=0))
    step_stack = np.concatenate([row["steps"] for row in train], axis=0)
    step_mean = np.mean(step_stack, axis=0)
    step_std = np.where(np.std(step_stack, axis=0) < 1.0e-12, 1.0, np.std(step_stack, axis=0))
    gru = _train_gru(train, val, step_mean, step_std, config)
    p_b2 = []
    p_b2_mid = []
    p_b1_mid = []
    p_b2_series: list[np.ndarray] = []
    for row in held:
        series = _gru_probs(gru, row, step_mean, step_std)
        static_t = _static_probs(row, w_ctx, prefix_mean, prefix_std)
        mid = max(0, (row["T"] - 1) // 2)
        p_b2.append(float(series[-1]))
        p_b2_mid.append(float(series[mid]))
        p_b1_mid.append(float(static_t[mid]))
        p_b2_series.append(series)
        row["p_b2_series"] = series
        row["p_b1_series"] = static_t
    p_b2_arr = np.asarray(p_b2)
    tau = float(
        np.quantile(
            [
                float(_gru_probs(gru, row, step_mean, step_std)[-1])
                for row in train + val
                if row["regime"] == "C0"
            ],
            0.95,
        )
    )
    c0_flags = []
    dwells = []
    det_b2 = []
    det_b1 = []
    recoveries = []
    for row in held:
        series = row["p_b2_series"]
        static_t = row["p_b1_series"]
        if row["regime"] == "C0":
            flags = series > tau
            c0_flags.extend(flags.tolist())
            dwells.append(mean_positive_dwell(flags))
            if row["family"] == "contact_switch":
                recoveries.append(recovery_time(series, tau, int(row["quiet_start"])))
        else:
            det_b2.append(detection_time(series, tau) / max(row["T"], 1))
            det_b1.append(detection_time(static_t, tau) / max(row["T"], 1))
    step_fpr = float(np.mean(c0_flags)) if c0_flags else math.nan
    h1 = bool(_brier(p_b2_arr, y_held) < _brier(p_b1, y_held))
    h2 = bool(_brier(np.asarray(p_b2_mid), y_held) < _brier(np.asarray(p_b1_mid), y_held))
    h3 = bool(math.isfinite(step_fpr) and step_fpr <= config.c0_step_fpr_max)
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_c0_final": tau,
        "h1_final_brier": {
            "brier_b0": _brier(p_b0, y_held),
            "brier_b1": _brier(p_b1, y_held),
            "brier_b2": _brier(p_b2_arr, y_held),
            "pass": h1,
        },
        "h2_mid_brier": {
            "brier_b1": _brier(np.asarray(p_b1_mid), y_held),
            "brier_b2": _brier(np.asarray(p_b2_mid), y_held),
            "pass": h2,
        },
        "h3_c0_step_fpr": {
            "fpr": step_fpr,
            "max": config.c0_step_fpr_max,
            "pass": h3,
        },
        "diagnostic": {
            "median_tdet_frac_c1_b1": float(np.median(det_b1)) if det_b1 else math.nan,
            "median_tdet_frac_c1_b2": float(np.median(det_b2)) if det_b2 else math.nan,
            "mean_fp_dwell_c0": float(np.mean(dwells)) if dwells else math.nan,
            "median_recovery_c0_switch": float(np.nanmedian(recoveries)) if recoveries else math.nan,
        },
        "v7b_go": bool(h1 and h2 and h3),
        "uses_family_label": False,
        "unlocks_rs5b": False,
        "unlocks_revision": False,
        "unlocks_v7c": False,
        "predicts_only_p_struct": True,
    }


def run_r3_v7b(
    output: str | Path,
    *,
    v7a_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7BConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7BConfig()
    _require_v7a(Path(v7a_summary))
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
            for script, scale in PULL_JOBS + PUSH_JOBS + (("switch_cycle", 1.0),)
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[V7B {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 7)
        torch.manual_seed(int(seed) + 7)
        if domain == "mode_a":
            row = _mode_a_trace(
                intake, seed=int(seed), regime=str(regime), amp_scale=float(amp), config=cfg
            )
        else:
            row = _contact_trace(
                intake,
                c0,
                seed=int(seed),
                regime=str(regime),
                script=str(script),
                scale=float(scale),
                config=cfg,
            )
        rows.append(row)
    if scientific:
        aggregate = _aggregate(rows, cfg)
    else:
        aggregate = {
            "v7b_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; persistent GO not evaluated",
            "switch_phases": sorted({float(p) for row in rows if row["script"] == "switch_cycle" for p in row["phase"]}),
        }
    status = stage_status()
    status["R3-V7A"] = {"frozen": True, "passed": True}
    status["R3-V7B"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7b_go")),
        "smoke_only": smoke,
    }
    status["R3-V7C"] = {"locked": True, "opened": False}
    status["R3-V7D"] = {"locked": True, "opened": False}
    status["R1-RS5B"] = {"locked": True, "opened": False}
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
                    "y_inadequate",
                    "D0",
                    "T",
                    "quiet_start",
                    *FEATURE_NAMES,
                )
                if key in row
            }
        )
    summary = {
        "stage": "R3-V7B",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "primary_baseline": "V7A-style static contextual (B1)",
            "model": "GRU32 p_struct only",
            "no_family_label": True,
            "no_rgb": True,
            "no_tactile": True,
            "rs5b": "locked",
            "v7c": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7b_go": bool(aggregate.get("v7b_go")),
        "v7a_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

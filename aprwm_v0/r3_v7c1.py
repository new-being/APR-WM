"""R3-V7C.1: learned tactile representation vs B5-S warranted belief."""

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
from .r1_rs2 import TAXEL_RES, R1RS2C0Config
from .r1_rs3a import _jsonable
from .r1_rs4a import FEATURE_NAMES, MODE_A_AMPS, PULL_JOBS, PUSH_JOBS
from .r3_v7b import EpiGRU, STEP_DIM, _gru_probs, _pad
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
from .r3_v7c import (
    CONTACT_INSTANT_COL,
    CONTACT_WINDOW_COL,
    _attach_channels,
    _fpr,
    _with_steps,
    h4_sensor_useful,
    zero_contact_slots,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3_V7C1_PREREG.md"
SEEDS = (22101, 22111, 22121, 22131, 22141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
SMOKE_JOBS = (
    ("mode_a", 22101, 1.5, None, 1.0),
    ("contact", 22101, math.nan, "fast_pull", 1.0),
    ("contact", 22101, math.nan, "switch_cycle", 1.0),
)
Z_DIM = 2


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7C1Config:
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
    taxel_res: int = TAXEL_RES


def _require_v7c(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C.1 locked until R3-V7C has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C.1 locked: V7C scientific matrix has not been run")
    if not payload.get("v7c_go"):
        raise RuntimeError("R3-V7C.1 locked: V7C_GO is false")
    return payload


class TactileEpi(nn.Module):
    """Same GRU32 as B5; only the contact slots come from a learned encoder."""

    def __init__(self, hidden: int = 32, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32, z_dim),
            nn.Tanh(),
        )
        self.gru = EpiGRU(STEP_DIM, hidden)

    def encode(self, taxel: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width = taxel.shape
        flat = taxel.reshape(batch * steps, 1, height, width)
        return self.encoder(flat).reshape(batch, steps, Z_DIM)

    def fuse(self, steps: torch.Tensor, taxel: torch.Tensor) -> torch.Tensor:
        fused = steps.clone()
        fused[:, :, CONTACT_INSTANT_COL : CONTACT_WINDOW_COL + 1] = self.encode(taxel)
        return fused

    def forward(
        self, steps: torch.Tensor, taxel: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.gru(self.fuse(steps, taxel), lengths)


def _pad_taxel(batch: list[np.ndarray], res: int) -> torch.Tensor:
    lengths = [len(item) for item in batch]
    padded = torch.zeros(len(batch), max(lengths), res, res, dtype=torch.float32)
    for index, item in enumerate(batch):
        tensor = torch.tensor(np.asarray(item, dtype=np.float64), dtype=torch.float32)
        padded[index, : len(item)] = tensor
    return padded


def _ensure_taxel(row: dict[str, Any], res: int) -> np.ndarray:
    if row.get("taxel") is not None:
        taxel = np.asarray(row["taxel"], dtype=np.float64)
        if taxel.ndim == 3:
            return taxel
    return np.zeros((int(row["T"]), res, res), dtype=np.float64)


def _tactile_probs(
    model: TactileEpi, row: dict[str, Any], mean: np.ndarray, std: np.ndarray, res: int
) -> np.ndarray:
    model.eval()
    steps = (zero_contact_slots(row["steps"]) - mean) / std
    taxel = _ensure_taxel(row, res)
    padded, lengths = _pad([steps])
    maps = _pad_taxel([taxel], res)
    with torch.no_grad():
        probs = model(padded, maps, lengths)[0, : int(lengths[0])]
    return probs.cpu().numpy()


def _train_tactile(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7C1Config,
    seed: int,
) -> TactileEpi:
    torch.manual_seed(seed)
    model = TactileEpi(config.hidden)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = math.inf
    stall = 0
    inner = R3V7B3Config(hidden=config.hidden, epochs=config.epochs, lr=config.lr)
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(zero_contact_slots(row["steps"]) - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            maps = _pad_taxel([_ensure_taxel(row, config.taxel_res) for row in chunk], config.taxel_res)
            pred = model(padded, maps, lengths)
            losses = []
            for index, row in enumerate(chunk):
                t_len = int(lengths[index])
                target = torch.tensor(row["target"][:t_len], dtype=torch.float32)
                losses.append(nn.functional.binary_cross_entropy(pred[index, :t_len], target))
            torch.stack(losses).mean().backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-inner.lr)
                        param.grad.zero_()
        model.eval()
        with torch.no_grad():
            scores = []
            for row in val:
                probs = _tactile_probs(model, row, mean, std, config.taxel_res)
                target = row["target"][: len(probs)]
                scores.append(float(np.mean((probs - target) ** 2)))
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


def _recon_mse(
    model: TactileEpi, rows: list[dict[str, Any]], res: int
) -> float:
    zs, ys = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            taxel = _ensure_taxel(row, res)
            if len(taxel) == 0:
                continue
            maps = torch.tensor(taxel[None], dtype=torch.float32)
            z = model.encode(maps)[0].cpu().numpy()
            pressure = np.mean(taxel, axis=(1, 2))
            zs.append(z)
            ys.append(pressure)
    if not zs:
        return math.nan
    z = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    try:
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return math.nan
    pred = design @ weights
    return float(np.mean((pred - y) ** 2))


def _inner(config: R3V7C1Config) -> R3V7B3Config:
    return R3V7B3Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
    )


def _collect_contact(
    intake: R1RS1AConfig,
    c0: R1RS2C0Config,
    *,
    seed: int,
    script: str,
    scale: float,
    config: R3V7C1Config,
) -> list[dict[str, Any]]:
    rows = _contact_pair(
        intake,
        c0,
        seed=seed,
        script=script,
        scale=scale,
        config=_inner(config),
        include_learner_sensors=True,
        include_tactile=True,
    )
    return _attach_channels(rows, config.window)


def _collect_mode_a(
    intake: R1RS1AConfig, *, seed: int, amp_scale: float, config: R3V7C1Config
) -> list[dict[str, Any]]:
    rows = _mode_a_pair(intake, seed=seed, amp_scale=amp_scale, config=_inner(config))
    for row in rows:
        row["taxel"] = np.zeros((int(row["T"]), config.taxel_res, config.taxel_res))
    return _attach_channels(rows, config.window)


def _aggregate(rows: list[dict[str, Any]], config: R3V7C1Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    train_s = _with_steps(train, "sensor")
    val_s = _with_steps(val, "sensor")
    held_s = _with_steps(held, "sensor")
    train_n = _with_steps(train, "none")
    val_n = _with_steps(val, "none")
    held_n = _with_steps(held, "none")
    stack_s = np.concatenate([row["steps"] for row in train_s], axis=0)
    mean_s = np.mean(stack_s, axis=0)
    std_s = np.where(np.std(stack_s, axis=0) < 1.0e-12, 1.0, np.std(stack_s, axis=0))
    stack_n = np.concatenate([row["steps"] for row in train_n], axis=0)
    mean_n = np.mean(stack_n, axis=0)
    std_n = np.where(np.std(stack_n, axis=0) < 1.0e-12, 1.0, np.std(stack_n, axis=0))
    inner = _inner(config)
    sensor = _train(train_s, val_s, mean_s, std_s, inner, warranted=True, seed=51)
    none = _train(train_n, val_n, mean_n, std_n, inner, warranted=True, seed=52)
    tactile = _train_tactile(train, val, mean_n, std_n, config, seed=53)
    series = {
        "sensor": [_gru_probs(sensor, row, mean_s, std_s) for row in held_s],
        "none": [_gru_probs(none, row, mean_n, std_n) for row in held_n],
        "tactile": [_tactile_probs(tactile, row, mean_n, std_n, config.taxel_res) for row in held],
    }
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    c1_idx = [i for i, row in enumerate(held) if row["regime"] != "C0"]
    tau = float(
        np.quantile(
            [
                float(_tactile_probs(tactile, row, mean_n, std_n, config.taxel_res)[-1])
                for row in train + val
                if row["regime"] == "C0"
            ],
            0.95,
        )
    )
    mid = {ch: brier_at_frac(series[ch], labels, 0.5) for ch in series}
    fin = {ch: brier_at_frac(series[ch], labels, 1.0) for ch in series}
    b_c0, m_c0, fpr = {}, {}, {}
    for ch in series:
        b_c0[ch], m_c0[ch] = c0_brier_mass([series[ch][i] for i in c0_idx])
        fpr[ch] = _fpr([series[ch][i] for i in c0_idx], tau)
    c1_labels = [1] * len(c1_idx)
    c1_fin = {
        ch: brier_at_frac([series[ch][i] for i in c1_idx], c1_labels, 1.0) for ch in series
    }
    corrs, high_p, low_p = [], [], []
    for index in c1_idx:
        w = held[index]["w_evid"][: len(series["tactile"][index])]
        p = series["tactile"][index][: len(w)]
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
    field_corrs = []
    for row in held:
        if row["domain"] != "contact":
            continue
        taxel = _ensure_taxel(row, config.taxel_res)
        pressure = np.mean(taxel, axis=(1, 2))
        contact = np.asarray(row["steps_oracle"][:, CONTACT_INSTANT_COL], dtype=np.float64)
        n = min(len(pressure), len(contact))
        field_corrs.append(spearman(pressure[:n], contact[:n]))
    recon = _recon_mse(tactile, held, config.taxel_res)
    h1 = bool(
        math.isfinite(mid["tactile"])
        and math.isfinite(fin["tactile"])
        and mid["tactile"] <= mid["sensor"] + config.delta
        and fin["tactile"] <= fin["sensor"] + config.delta
    )
    h2 = bool(math.isfinite(fpr["tactile"]) and fpr["tactile"] <= config.c0_step_fpr_max)
    h3 = bool(
        math.isfinite(mean_corr)
        and mean_corr > 0.0
        and math.isfinite(e_high)
        and math.isfinite(e_low)
        and e_high > e_low
    )
    h4 = h4_sensor_useful(
        brier_mid_s=mid["tactile"],
        brier_mid_n=mid["none"],
        brier_final_s=fin["tactile"],
        brier_final_n=fin["none"],
        b_c0_s=b_c0["tactile"],
        b_c0_n=b_c0["none"],
        brier_c1_s=c1_fin["tactile"],
        brier_c1_n=c1_fin["none"],
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_tactile_c0_final": tau,
        "h1_vs_b5s": {
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
        "h4_not_empty_vs_n": {
            "brier_c1_final": c1_fin,
            "pass": h4,
        },
        "diagnostic_not_go": {
            "spearman_taxel_mean_vs_oracle_contact": float(np.mean(field_corrs))
            if field_corrs
            else math.nan,
            "linear_probe_mse_z_to_taxel_mean": recon,
            "note": "reconstruction is diagnostic only; H3 failure still fails the stage",
        },
        "v7c1_go": bool(h1 and h2 and h3 and h4),
        "uses_jtf_as_model_input": False,
        "uses_phase_in_model": False,
        "uses_w_at_runtime": False,
        "same_gru_as_b5": True,
        "unlocks_v7d": False,
        "encoder_trained_for_reconstruction": False,
    }


def run_r3_v7c1(
    output: str | Path,
    *,
    v7c_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7C1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7C1Config()
    _require_v7c(Path(v7c_summary))
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
            f"[V7C.1 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 19)
        if domain == "mode_a":
            rows.extend(
                _collect_mode_a(intake, seed=int(seed), amp_scale=float(amp), config=cfg)
            )
        else:
            rows.extend(
                _collect_contact(
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
            "v7c1_go": False,
            "n_episodes": len(rows),
            "n_pairs": len({row["pair_id"] for row in rows}),
            "taxel_nonzero": bool(
                any(
                    float(np.mean(_ensure_taxel(row, cfg.taxel_res))) > 0.0
                    for row in rows
                    if row["domain"] == "contact"
                )
            ),
            "note": "plumbing smoke only; V7C.1 GO not evaluated",
        }
    )
    status = stage_status()
    status["R3-V7C"] = {"frozen": True, "passed": True}
    status["R3-V7C.1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("v7c1_go")),
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
        item["taxel_mean"] = float(np.mean(_ensure_taxel(row, cfg.taxel_res)))
        slim.append(item)
    summary = {
        "stage": "R3-V7C.1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "architecture": "B5 GRU32 + tactile encoder into contact slots",
            "supervision": "y * w_evid",
            "variable": "hand-crafted B5-S proxy -> learned tactile z",
            "runtime_sees_w": False,
            "v7d": "locked",
            "delta": cfg.delta,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7c1_go": bool(aggregate.get("v7c1_go")),
        "v7c_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

"""R3-V7C.2: locate field vs encoder vs fusion tactile bottleneck."""

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
    C1_REGIMES,
    DELTA,
    W_HIGH,
    W_LOW,
    R3V7B3Config,
    _train,
    spearman,
)
from .r3_v7c import CONTACT_INSTANT_COL, CONTACT_WINDOW_COL, _fpr, _with_steps, h4_sensor_useful, zero_contact_slots
from .r3_v7c1 import (
    R3V7C1Config,
    TactileEpi,
    Z_DIM,
    _collect_contact,
    _collect_mode_a,
    _ensure_taxel,
    _pad_taxel,
    _tactile_probs,
    _train_tactile,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7C2_PREREG.md"
SEEDS = (23101, 23111, 23121, 23131, 23141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
EPS_INFO = 0.005
SMOKE_JOBS = (
    ("mode_a", 23101, 1.5, None, 1.0),
    ("contact", 23101, math.nan, "fast_pull", 1.0),
    ("contact", 23101, math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7C2Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    w_low: float = W_LOW
    w_high: float = W_HIGH
    eps_info: float = EPS_INFO
    target_hz: float = 20.0
    taxel_res: int = TAXEL_RES


def _require_v7c1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C.2 locked until R3-V7C.1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C.2 locked: V7C.1 scientific matrix has not been run")
    return payload


class LinearFieldEpi(nn.Module):
    """Raw flattened taxel linear readout into the same 2 contact slots."""

    def __init__(self, hidden: int = 32, res: int = TAXEL_RES) -> None:
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(int(res) * int(res), Z_DIM), nn.Tanh())
        self.gru = EpiGRU(STEP_DIM, hidden)

    def encode(self, taxel: torch.Tensor) -> torch.Tensor:
        batch, steps, height, width = taxel.shape
        return self.readout(taxel.reshape(batch * steps, height * width)).reshape(batch, steps, Z_DIM)

    def fuse(self, steps: torch.Tensor, taxel: torch.Tensor) -> torch.Tensor:
        fused = steps.clone()
        fused[:, :, CONTACT_INSTANT_COL : CONTACT_WINDOW_COL + 1] = self.encode(taxel)
        return fused

    def forward(
        self, steps: torch.Tensor, taxel: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.gru(self.fuse(steps, taxel), lengths)


def classify_locus(
    delta_x: float,
    delta_z: float,
    *,
    tactile_useful_vs_n: bool,
    eps: float = EPS_INFO,
) -> str:
    if not math.isfinite(delta_x) or delta_x <= eps:
        return "FIELD"
    if not math.isfinite(delta_z) or delta_z <= eps:
        return "ENCODER"
    if not tactile_useful_vs_n:
        return "FUSION"
    return "UNRESOLVED"


def bce_vs_target(series: list[np.ndarray], rows: list[dict[str, Any]]) -> float:
    scores = []
    for probs, row in zip(series, rows, strict=True):
        target = np.asarray(row["target"][: len(probs)], dtype=np.float64)
        pred = np.clip(np.asarray(probs[: len(target)], dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
        scores.append(float(np.mean(-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))))
        )
    return float(np.mean(scores)) if scores else math.nan


def shuffle_taxel(taxel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    taxel = np.asarray(taxel, dtype=np.float64)
    if len(taxel) == 0:
        return taxel
    return taxel[rng.permutation(len(taxel))]


def _c1_cfg(config: R3V7C2Config) -> R3V7C1Config:
    return R3V7C1Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
        taxel_res=config.taxel_res,
    )


def _train_linear_field(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7C2Config,
    seed: int,
) -> LinearFieldEpi:
    torch.manual_seed(seed)
    model = LinearFieldEpi(config.hidden, config.taxel_res)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(zero_contact_slots(row["steps"]) - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            maps = _pad_taxel(
                [_ensure_taxel(row, config.taxel_res) for row in chunk], config.taxel_res
            )
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
                        param.add_(param.grad, alpha=-config.lr)
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


def _shuffled_probs(
    model: nn.Module,
    rows: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    res: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        copy = dict(row)
        copy["taxel"] = shuffle_taxel(_ensure_taxel(row, res), rng)
        out.append(_tactile_probs(model, copy, mean, std, res))
    return out


def _pair_taxel_gap(held: list[dict[str, Any]], config: R3V7C2Config) -> dict[str, float]:
    early, late = [], []
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in held:
        by_pair.setdefault(row["pair_id"], {})[row["regime"]] = row
    for pair in by_pair.values():
        if "C0" not in pair:
            continue
        x0 = _ensure_taxel(pair["C0"], config.taxel_res)
        for regime in C1_REGIMES:
            if regime not in pair:
                continue
            x1 = _ensure_taxel(pair[regime], config.taxel_res)
            w = np.asarray(pair[regime]["w_evid"], dtype=np.float64)
            n = min(len(x0), len(x1), len(w))
            dist = np.mean((x1[:n] - x0[:n]) ** 2, axis=(1, 2))
            if np.any(w[:n] < config.w_low):
                early.append(float(np.mean(dist[w[:n] < config.w_low])))
            if np.any(w[:n] > config.w_high):
                late.append(float(np.mean(dist[w[:n] > config.w_high])))
    return {
        "mean_l2_early": float(np.mean(early)) if early else math.nan,
        "mean_l2_late": float(np.mean(late)) if late else math.nan,
    }


def _aggregate(rows: list[dict[str, Any]], config: R3V7C2Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    train_n = _with_steps(train, "none")
    val_n = _with_steps(val, "none")
    held_n = _with_steps(held, "none")
    stack = np.concatenate([row["steps"] for row in train_n], axis=0)
    mean_n = np.mean(stack, axis=0)
    std_n = np.where(np.std(stack, axis=0) < 1.0e-12, 1.0, np.std(stack, axis=0))
    inner = R3V7B3Config(hidden=config.hidden, epochs=config.epochs, lr=config.lr)
    p0 = _train(train_n, val_n, mean_n, std_n, inner, warranted=True, seed=61)
    px = _train_linear_field(train, val, mean_n, std_n, config, seed=62)
    pz = _train_tactile(train, val, mean_n, std_n, _c1_cfg(config), seed=63)
    s0 = [_gru_probs(p0, row, mean_n, std_n) for row in held_n]
    sx = [_tactile_probs(px, row, mean_n, std_n, config.taxel_res) for row in held]
    sz = [_tactile_probs(pz, row, mean_n, std_n, config.taxel_res) for row in held]
    sh = _shuffled_probs(pz, held, mean_n, std_n, config.taxel_res, seed=77)
    l0, lx, lz, lsh = (
        bce_vs_target(s0, held),
        bce_vs_target(sx, held),
        bce_vs_target(sz, held),
        bce_vs_target(sh, held),
    )
    delta_x = float(l0 - lx) if math.isfinite(l0) and math.isfinite(lx) else math.nan
    delta_z = float(l0 - lz) if math.isfinite(l0) and math.isfinite(lz) else math.nan
    delta_sh = float(lsh - lz) if math.isfinite(lsh) and math.isfinite(lz) else math.nan
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    c1_idx = [i for i, row in enumerate(held) if row["regime"] != "C0"]
    b_c0_n, _ = c0_brier_mass([s0[i] for i in c0_idx])
    b_c0_z, _ = c0_brier_mass([sz[i] for i in c0_idx])
    c1_labels = [1] * len(c1_idx)
    c1_n = brier_at_frac([s0[i] for i in c1_idx], c1_labels, 1.0)
    c1_z = brier_at_frac([sz[i] for i in c1_idx], c1_labels, 1.0)
    mid_n, mid_z = brier_at_frac(s0, labels, 0.5), brier_at_frac(sz, labels, 0.5)
    fin_n, fin_z = brier_at_frac(s0, labels, 1.0), brier_at_frac(sz, labels, 1.0)
    useful = h4_sensor_useful(
        brier_mid_s=mid_z,
        brier_mid_n=mid_n,
        brier_final_s=fin_z,
        brier_final_n=fin_n,
        b_c0_s=b_c0_z,
        b_c0_n=b_c0_n,
        brier_c1_s=c1_z,
        brier_c1_n=c1_n,
    )
    locus = classify_locus(
        delta_x, delta_z, tactile_useful_vs_n=useful, eps=config.eps_info
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "probes": {
            "L_p0": l0,
            "L_px": lx,
            "L_pz": lz,
            "L_shuffle": lsh,
            "delta_x": delta_x,
            "delta_z": delta_z,
            "delta_shuffle": delta_sh,
            "eps": config.eps_info,
        },
        "h4_style_pz_vs_n": {
            "brier_mid_n": mid_n,
            "brier_mid_z": mid_z,
            "brier_final_n": fin_n,
            "brier_final_z": fin_z,
            "b_c0_n": b_c0_n,
            "b_c0_z": b_c0_z,
            "brier_c1_n": c1_n,
            "brier_c1_z": c1_z,
            "useful": useful,
        },
        "matched_taxel_l2": _pair_taxel_gap(held, config),
        "locus": locus,
        "primary_target": "e_t = y * w_t",
        "oracle_contact_not_primary": True,
        "v7d_locked": True,
        "method_go": None,
    }


def run_r3_v7c2(
    output: str | Path,
    *,
    v7c1_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7C2Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7C2Config()
    _require_v7c1(Path(v7c1_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    inner = _c1_cfg(cfg)
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
            f"[V7C.2 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 23)
        if domain == "mode_a":
            rows.extend(
                _collect_mode_a(intake, seed=int(seed), amp_scale=float(amp), config=inner)
            )
        else:
            rows.extend(
                _collect_contact(
                    intake,
                    c0,
                    seed=int(seed),
                    script=str(script),
                    scale=float(scale),
                    config=inner,
                )
            )
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "locus": None,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; locus not classified",
        }
    )
    status = stage_status()
    status["R3-V7C.1"] = {"frozen": True, "passed": False}
    status["R3-V7C.2"] = {
        "unlocked": True,
        "diagnostic": True,
        "locus": aggregate.get("locus"),
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
                "T",
                *FEATURE_NAMES,
            )
            if key in row
        }
        item["w_final"] = float(row["w_evid"][-1]) if len(row["w_evid"]) else math.nan
        slim.append(item)
    summary = {
        "stage": "R3-V7C.2",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "kind": "diagnostic locus, not method GO",
            "target": "e_t = y * w_t",
            "eps_info": cfg.eps_info,
            "v7d": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "locus": aggregate.get("locus"),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

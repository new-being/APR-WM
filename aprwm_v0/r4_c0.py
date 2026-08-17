"""R4-C0: I(X_tactile; e_t | h^S) > 0 on frozen I0 v2 / I1 h^S.

No I0 retune. No tactile CNN. GRU32 + raw/linear X only.
"""

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
from .r1_rs3a import _jsonable
from .r3_v7b import EpiGRU, _pad
from .r3_v7b3 import evidence_weights
from .r3_v7c2 import bce_vs_target, shuffle_taxel
from .r4_i0 import R4I0Config, simulate_pattern
from .r4_i1 import HS_KEYS, _hs, _tactile
from .train import seed_everything
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R4_C0_PREREG.md"
EPS_INFO = 0.005
REL_EVID_MIN = 1.0e-4
HIDDEN = 32
EPOCHS = 60
LR = 1.0e-2
TRAIN_Q0 = (-0.0008, -0.0006, -0.0004, -0.0002, 0.0, 0.0002, 0.0004, 0.0006, 0.0008)
VAL_Q0 = (-0.0010, 0.0010)
HELD_Q0 = (-0.0012, 0.0015, -0.0015, 0.0012)
SMOKE_TRAIN_Q0 = (0.0,)
SMOKE_VAL_Q0 = (0.0004,)
SMOKE_HELD_Q0 = (0.0012,)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R4C0Config:
    hidden: int = HIDDEN
    epochs: int = EPOCHS
    lr: float = LR
    eps_info: float = EPS_INFO
    rel_evid_min: float = REL_EVID_MIN


def warrant_from_macro(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    *,
    rel_min: float = REL_EVID_MIN,
) -> tuple[np.ndarray, float, bool]:
    """B.3-style w_t from noiseless macro residuals. Null if A/B barely diverge."""
    a = np.asarray(residual_a, dtype=np.float64)
    b = np.asarray(residual_b, dtype=np.float64)
    n = min(len(a), len(b))
    delta2 = np.sum((b[:n] - a[:n]) ** 2, axis=1)
    energy = float(np.sum(delta2))
    scale = 0.5 * float(np.sum(a[:n] ** 2) + np.sum(b[:n] ** 2))
    relative = energy / (scale + 1.0e-12)
    if relative < rel_min:
        return np.zeros(n, dtype=np.float64), relative, False
    diff = np.sqrt(np.maximum(delta2, 0.0))
    return evidence_weights(np.zeros_like(diff), diff), relative, True


def _require_i0_i1(i0_path: Path, i1_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not i0_path.is_file():
        raise RuntimeError("R4-C0 locked until R4-I0 has been run")
    if not i1_path.is_file():
        raise RuntimeError("R4-C0 locked until R4-I1 has been run")
    i0 = json.loads(i0_path.read_text(encoding="utf-8"))
    i1 = json.loads(i1_path.read_text(encoding="utf-8"))
    if i0.get("stage") != "R4-I0" or not i0.get("r4_i0_pass"):
        raise RuntimeError("R4-C0 locked: R4_I0_PASS is false")
    if i0.get("protocol") != "stiffness_swap_v2":
        raise RuntimeError("R4-C0 locked: I0 protocol is not stiffness_swap_v2")
    if i1.get("stage") != "R4-I1" or not i1.get("r4_i1_pass"):
        raise RuntimeError("R4-C0 locked: R4_I1_PASS is false")
    return i0, i1


def _pair_episodes(q0: float, config: R4I0Config, rel_min: float) -> list[dict[str, Any]]:
    left = simulate_pattern("stiff_left", config, q0_block=q0)
    right = simulate_pattern("stiff_right", config, q0_block=q0)
    hs_a = np.stack([_hs(row) for row in left["rows"]])
    hs_b = np.stack([_hs(row) for row in right["rows"]])
    x_a = np.stack([_tactile(row) for row in left["rows"]])
    x_b = np.stack([_tactile(row) for row in right["rows"]])
    weights, relative, warranted = warrant_from_macro(hs_a, hs_b, rel_min=rel_min)
    n = len(weights)
    return [
        {
            "q0": float(q0),
            "regime": "A",
            "y": 0.0,
            "steps": hs_a[:n],
            "taxel": x_a[:n],
            "w_evid": np.zeros(n, dtype=np.float64),
            "target": np.zeros(n, dtype=np.float64),
            "relative_macro_energy": relative,
            "warrant_active": warranted,
            "T": n,
        },
        {
            "q0": float(q0),
            "regime": "B",
            "y": 1.0,
            "steps": hs_b[:n],
            "taxel": x_b[:n],
            "w_evid": weights,
            "target": weights.copy(),
            "relative_macro_energy": relative,
            "warrant_active": warranted,
            "T": n,
        },
    ]


def _collect(q0s: tuple[float, ...], phys: R4I0Config, rel_min: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q0 in q0s:
        rows.extend(_pair_episodes(q0, phys, rel_min))
    return rows


def _feat_stats(rows: list[dict[str, Any]], *, with_x: bool) -> tuple[np.ndarray, np.ndarray]:
    mats = []
    for row in rows:
        steps = np.asarray(row["steps"], dtype=np.float64)
        if with_x:
            steps = np.concatenate([steps, np.asarray(row["taxel"], dtype=np.float64)], axis=1)
        mats.append(steps)
    stacked = np.concatenate(mats, axis=0)
    mean = stacked.mean(axis=0)
    std = np.where(stacked.std(axis=0) < 1.0e-8, 1.0, stacked.std(axis=0))
    return mean, std


def _features(row: dict[str, Any], *, with_x: bool, shuffle: bool, rng: np.random.Generator) -> np.ndarray:
    steps = np.asarray(row["steps"], dtype=np.float64)
    if not with_x:
        return steps
    taxel = np.asarray(row["taxel"], dtype=np.float64)
    if shuffle:
        taxel = shuffle_taxel(taxel, rng)
    return np.concatenate([steps, taxel], axis=1)


def _probs(
    model: EpiGRU,
    row: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    with_x: bool,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    generator = rng if rng is not None else np.random.default_rng(0)
    feats = (_features(row, with_x=with_x, shuffle=shuffle, rng=generator) - mean) / std
    padded, lengths = _pad([feats])
    model.eval()
    with torch.no_grad():
        pred = model(padded, lengths)[0, : int(lengths[0])]
    return pred.cpu().numpy()


def _train(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R4C0Config,
    *,
    with_x: bool,
    shuffle: bool,
    seed: int,
) -> EpiGRU:
    dim = int(mean.shape[0])
    torch.manual_seed(seed)
    model = EpiGRU(dim, config.hidden)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        epoch_rng = np.random.default_rng(epoch + seed + 17)
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [
                (
                    _features(row, with_x=with_x, shuffle=shuffle, rng=epoch_rng) - mean
                )
                / std
                for row in chunk
            ]
            padded, lengths = _pad(feats)
            pred = model(padded, lengths)
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
                probs = _probs(model, row, mean, std, with_x=with_x, shuffle=shuffle)
                target = np.asarray(row["target"][: len(probs)], dtype=np.float64)
                scores.append(float(np.mean((probs - target) ** 2)))
            score = float(np.mean(scores)) if scores else math.inf
        if score + 1.0e-6 < best:
            best = score
            stall = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stall += 1
            if stall >= 12:
                break
    model.load_state_dict(best_state)
    return model


def _eval_model(
    model: EpiGRU,
    rows: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    with_x: bool,
    shuffle: bool = False,
) -> tuple[list[np.ndarray], float]:
    rng = np.random.default_rng(0)
    series = [
        _probs(model, row, mean, std, with_x=with_x, shuffle=shuffle, rng=rng) for row in rows
    ]
    return series, bce_vs_target(series, rows)


def run_r4_c0(
    output: str | Path,
    *,
    i0_summary: str | Path,
    i1_summary: str | Path,
    config: R4C0Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R4C0Config()
    if smoke:
        cfg = R4C0Config(hidden=cfg.hidden, epochs=2, lr=cfg.lr, eps_info=cfg.eps_info)
    i0, i1 = _require_i0_i1(Path(i0_summary), Path(i1_summary))
    seed_everything(71)
    phys = R4I0Config()
    train_q0 = SMOKE_TRAIN_Q0 if smoke else TRAIN_Q0
    val_q0 = SMOKE_VAL_Q0 if smoke else VAL_Q0
    held_q0 = SMOKE_HELD_Q0 if smoke else HELD_Q0
    train = _collect(train_q0, phys, cfg.rel_evid_min)
    val = _collect(val_q0, phys, cfg.rel_evid_min)
    held = _collect(held_q0, phys, cfg.rel_evid_min)
    mean0, std0 = _feat_stats(train, with_x=False)
    meanx, stdx = _feat_stats(train, with_x=True)
    p0 = _train(train, val, mean0, std0, cfg, with_x=False, shuffle=False, seed=81)
    px = _train(train, val, meanx, stdx, cfg, with_x=True, shuffle=False, seed=82)
    pshuf = _train(train, val, meanx, stdx, cfg, with_x=True, shuffle=True, seed=83)
    _, loss0 = _eval_model(p0, held, mean0, std0, with_x=False)
    _, lossx = _eval_model(px, held, meanx, stdx, with_x=True)
    _, losss = _eval_model(pshuf, held, meanx, stdx, with_x=True, shuffle=True)
    delta = (
        float(loss0 - lossx) if math.isfinite(loss0) and math.isfinite(lossx) else math.nan
    )
    delta_shuf = (
        float(loss0 - losss) if math.isfinite(loss0) and math.isfinite(losss) else math.nan
    )
    passed = bool(not smoke and math.isfinite(delta) and delta > cfg.eps_info)
    mean_rel = float(np.mean([row["relative_macro_energy"] for row in held if row["regime"] == "B"]))
    warrant_frac = float(np.mean([float(row["warrant_active"]) for row in held if row["regime"] == "B"]))
    mean_w = float(np.mean([float(np.mean(row["w_evid"])) for row in held if row["regime"] == "B"]))
    status = stage_status()
    status["R4-I0"] = {"protocol": "stiffness_swap_v2", "r4_i0_pass": True, "frozen": True}
    status["R4-I1"] = {"r4_i1_pass": True, "frozen": True}
    status["R4-C0"] = {"r4_c0_go": passed, "delta_x": delta, "smoke": smoke}
    payload = {
        "stage": "R4-C0",
        "scientific_result": not smoke,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "i0_protocol": i0.get("protocol"),
        "i1_pass": bool(i1.get("r4_i1_pass")),
        "i0_frozen": True,
        "hs_keys": list(HS_KEYS),
        "split": "trajectory_q0",
        "train_q0": list(train_q0),
        "val_q0": list(val_q0),
        "held_q0": list(held_q0),
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "hidden": cfg.hidden,
        "epochs": cfg.epochs,
        "encoder": False,
        "loss_hs": loss0,
        "loss_hs_x": lossx,
        "loss_hs_x_shuffle": losss,
        "delta_x": delta,
        "delta_x_shuffle": delta_shuf,
        "eps_info": cfg.eps_info,
        "rel_evid_min": cfg.rel_evid_min,
        "held_mean_relative_macro_energy": mean_rel,
        "held_warrant_active_frac": warrant_frac,
        "held_mean_w_evid_b": mean_w,
        "r4_c0_go": passed,
        "config": asdict(cfg),
        "note": (
            "e_t=y w_t with y=A/B and w_t from noiseless macro residual divergence; "
            "raw X concatenated (no CNN); not Door weights"
        ),
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload

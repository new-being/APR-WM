"""R3-V7C.3: observation-sufficiency ablation of tactile measurements."""

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
from .r3_v7b3 import C1_REGIMES, W_HIGH, W_LOW, R3V7B3Config, _train
from .r3_v7c import CONTACT_INSTANT_COL, CONTACT_WINDOW_COL, _with_steps, zero_contact_slots
from .r3_v7c1 import R3V7C1Config, Z_DIM, _collect_contact, _collect_mode_a, _ensure_taxel
from .r3_v7c2 import EPS_INFO, bce_vs_target
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3_V7C3_PREREG.md"
SEEDS = (24101, 24111, 24121, 24131, 24141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
CANDIDATES = ("N", "NS", "NSG", "NSGM")
SMOKE_JOBS = (
    ("mode_a", 24101, 1.5, None, 1.0),
    ("contact", 24101, math.nan, "fast_pull", 1.0),
    ("contact", 24101, math.nan, "switch_cycle", 1.0),
)
GEOM_DIM = 7
SURF_DIM = 9


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7C3Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    w_low: float = W_LOW
    w_high: float = W_HIGH
    eps_info: float = EPS_INFO
    target_hz: float = 20.0
    taxel_res: int = TAXEL_RES


def _require_v7c2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C.3 locked until R3-V7C.2 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C.3 locked: V7C.2 scientific matrix has not been run")
    return payload


class LinearObsEpi(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(int(in_dim), Z_DIM), nn.Tanh())
        self.gru = EpiGRU(STEP_DIM, hidden)

    def forward(
        self, steps: torch.Tensor, obs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch, time, dim = obs.shape
        fused = steps.clone()
        fused[:, :, CONTACT_INSTANT_COL : CONTACT_WINDOW_COL + 1] = self.readout(
            obs.reshape(batch * time, dim)
        ).reshape(batch, time, Z_DIM)
        return self.gru(fused, lengths)


def observation_vector(row: dict[str, Any], name: str, res: int = TAXEL_RES) -> np.ndarray:
    taxel = _ensure_taxel(row, res).reshape(int(row["T"]), -1)
    n_steps = len(taxel)
    shear = row.get("taxel_shear")
    if shear is None:
        shear = np.zeros((n_steps, res, res), dtype=np.float64)
    shear = np.asarray(shear, dtype=np.float64).reshape(n_steps, -1)[:n_steps]
    geom = row.get("tactile_geom")
    if geom is None:
        geom = np.zeros((n_steps, GEOM_DIM), dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)[:n_steps]
    if geom.ndim == 1:
        geom = np.repeat(geom[None, :], n_steps, axis=0)
    surf = row.get("tactile_surf")
    if surf is None:
        surf = np.zeros((n_steps, SURF_DIM), dtype=np.float64)
    surf = np.asarray(surf, dtype=np.float64)[:n_steps]
    if surf.ndim == 1:
        surf = np.repeat(surf[None, :], n_steps, axis=0)
    if name == "N":
        return taxel
    if name == "NS":
        return np.concatenate([taxel, shear], axis=1)
    if name == "NSG":
        return np.concatenate([taxel, shear, geom], axis=1)
    if name == "NSGM":
        return np.concatenate([taxel, shear, geom, surf], axis=1)
    raise ValueError(name)


def sufficient_candidates(deltas: dict[str, float], eps: float) -> list[str]:
    return [name for name, value in deltas.items() if math.isfinite(value) and value > eps]


def stop_tactile_branch(deltas: dict[str, float], eps: float) -> bool:
    finite = [value for value in deltas.values() if math.isfinite(value)]
    if not finite:
        return True
    return float(max(finite)) <= eps


def _pad_obs(batch: list[np.ndarray]) -> torch.Tensor:
    dim = batch[0].shape[1]
    length = max(len(item) for item in batch)
    padded = torch.zeros(len(batch), length, dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, : len(item)] = torch.tensor(item, dtype=torch.float32)
    return padded


def _obs_probs(
    model: LinearObsEpi,
    row: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    name: str,
    res: int,
) -> np.ndarray:
    model.eval()
    steps = (zero_contact_slots(row["steps"]) - mean) / std
    obs = observation_vector(row, name, res)
    padded, lengths = _pad([steps])
    with torch.no_grad():
        pred = model(padded, _pad_obs([obs]), lengths)[0, : int(lengths[0])]
    return pred.cpu().numpy()


def _train_obs(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    name: str,
    config: R3V7C3Config,
    seed: int,
) -> LinearObsEpi:
    torch.manual_seed(seed)
    in_dim = observation_vector(train[0], name, config.taxel_res).shape[1]
    model = LinearObsEpi(in_dim, config.hidden)
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
            obs = _pad_obs([observation_vector(row, name, config.taxel_res) for row in chunk])
            pred = model(padded, obs, lengths)
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
                probs = _obs_probs(model, row, mean, std, name, config.taxel_res)
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


def _pair_obs_l2(held: list[dict[str, Any]], name: str, config: R3V7C3Config) -> dict[str, float]:
    early, late = [], []
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in held:
        by_pair.setdefault(row["pair_id"], {})[row["regime"]] = row
    for pair in by_pair.values():
        if "C0" not in pair:
            continue
        x0 = observation_vector(pair["C0"], name, config.taxel_res)
        for regime in C1_REGIMES:
            if regime not in pair:
                continue
            x1 = observation_vector(pair[regime], name, config.taxel_res)
            w = np.asarray(pair[regime]["w_evid"], dtype=np.float64)
            n = min(len(x0), len(x1), len(w))
            dist = np.mean((x1[:n] - x0[:n]) ** 2, axis=1)
            if np.any(w[:n] < config.w_low):
                early.append(float(np.mean(dist[w[:n] < config.w_low])))
            if np.any(w[:n] > config.w_high):
                late.append(float(np.mean(dist[w[:n] > config.w_high])))
    return {
        "mean_l2_early": float(np.mean(early)) if early else math.nan,
        "mean_l2_late": float(np.mean(late)) if late else math.nan,
    }


def _c1_cfg(config: R3V7C3Config) -> R3V7C1Config:
    return R3V7C1Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
        taxel_res=config.taxel_res,
    )


def _attach_mode_a_fields(rows: list[dict[str, Any]], res: int) -> list[dict[str, Any]]:
    for row in rows:
        n = int(row["T"])
        row.setdefault("taxel", np.zeros((n, res, res)))
        row["taxel_shear"] = np.zeros((n, res, res))
        row["tactile_geom"] = np.zeros((n, GEOM_DIM))
        row["tactile_surf"] = np.zeros((n, SURF_DIM))
    return rows


def _aggregate(rows: list[dict[str, Any]], config: R3V7C3Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    train_n = _with_steps(train, "none")
    val_n = _with_steps(val, "none")
    held_n = _with_steps(held, "none")
    stack = np.concatenate([row["steps"] for row in train_n], axis=0)
    mean_n = np.mean(stack, axis=0)
    std_n = np.where(np.std(stack, axis=0) < 1.0e-12, 1.0, np.std(stack, axis=0))
    inner = R3V7B3Config(hidden=config.hidden, epochs=config.epochs, lr=config.lr)
    p0 = _train(train_n, val_n, mean_n, std_n, inner, warranted=True, seed=71)
    s0 = [_gru_probs(p0, row, mean_n, std_n) for row in held_n]
    l0 = bce_vs_target(s0, held)
    losses = {"N": math.nan, "NS": math.nan, "NSG": math.nan, "NSGM": math.nan}
    deltas = dict(losses)
    pair_l2 = {}
    for offset, name in enumerate(CANDIDATES):
        model = _train_obs(train, val, mean_n, std_n, name, config, seed=72 + offset)
        series = [_obs_probs(model, row, mean_n, std_n, name, config.taxel_res) for row in held]
        losses[name] = bce_vs_target(series, held)
        deltas[name] = (
            float(l0 - losses[name]) if math.isfinite(l0) and math.isfinite(losses[name]) else math.nan
        )
        pair_l2[name] = _pair_obs_l2(held, name, config)
    hits = sufficient_candidates(deltas, config.eps_info)
    stop = stop_tactile_branch(deltas, config.eps_info)
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "L_p0": l0,
        "L": losses,
        "delta": deltas,
        "eps": config.eps_info,
        "sufficient": hits,
        "stop_tactile_branch": stop,
        "matched_l2": pair_l2,
        "encoder_not_trained": True,
        "b5_not_integrated": True,
        "v7d_locked": True,
    }


def run_r3_v7c3(
    output: str | Path,
    *,
    v7c2_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7C3Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7C3Config()
    _require_v7c2(Path(v7c2_summary))
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
            f"[V7C.3 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 29)
        if domain == "mode_a":
            rows.extend(
                _attach_mode_a_fields(
                    _collect_mode_a(intake, seed=int(seed), amp_scale=float(amp), config=inner),
                    cfg.taxel_res,
                )
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
            "stop_tactile_branch": None,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; deltas not evaluated",
            "has_shear": bool(
                any(row.get("taxel_shear") is not None for row in rows if row["domain"] == "contact")
            ),
        }
    )
    status = stage_status()
    status["R3-V7C.2"] = {"frozen": True, "locus": "FIELD"}
    status["R3-V7C.3"] = {
        "unlocked": True,
        "observation_sufficiency": True,
        "stop_tactile_branch": aggregate.get("stop_tactile_branch"),
        "sufficient": aggregate.get("sufficient"),
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
        "stage": "R3-V7C.3",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "kind": "observation sufficiency ablation",
            "no_cnn": True,
            "no_b5_integration": True,
            "eps_info": cfg.eps_info,
            "v7d": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "stop_tactile_branch": aggregate.get("stop_tactile_branch"),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

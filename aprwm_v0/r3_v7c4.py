"""R3-V7C.4: representation sufficiency of frozen X_NSG."""

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
from .r3_v7b3 import W_HIGH, W_LOW, R3V7B3Config, _train
from .r3_v7c import CONTACT_INSTANT_COL, CONTACT_WINDOW_COL, _with_steps, zero_contact_slots
from .r3_v7c1 import R3V7C1Config, Z_DIM, _collect_contact, _collect_mode_a, _ensure_taxel
from .r3_v7c2 import EPS_INFO, bce_vs_target
from .r3_v7c3 import (
    GEOM_DIM,
    _attach_mode_a_fields,
    _obs_probs,
    _train_obs,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7C4_PREREG.md"
SEEDS = (25101, 25111, 25121, 25131, 25141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
OBS_NAME = "NSG"
EPS_RETAIN = 0.02
C3_DELTA_NSG_REF = 0.057
SMOKE_JOBS = (
    ("mode_a", 25101, 1.5, None, 1.0),
    ("contact", 25101, math.nan, "fast_pull", 1.0),
    ("contact", 25101, math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7C4Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    w_low: float = W_LOW
    w_high: float = W_HIGH
    eps_info: float = EPS_INFO
    eps_retain: float = EPS_RETAIN
    target_hz: float = 20.0
    taxel_res: int = TAXEL_RES


def _require_v7c3(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C.4 locked until R3-V7C.3 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C.4 locked: V7C.3 scientific matrix has not been run")
    if payload.get("stop_tactile_branch"):
        raise RuntimeError("R3-V7C.4 locked: V7C.3 stop_tactile_branch is true")
    sufficient = payload.get("aggregate", {}).get("sufficient") or payload.get("sufficient") or []
    if OBS_NAME not in sufficient:
        raise RuntimeError("R3-V7C.4 locked: X_NSG was not observation-sufficient in V7C.3")
    return payload


def classify_representation(
    delta_x: float,
    delta_z: float,
    *,
    eps: float = EPS_INFO,
    eps_retain: float = EPS_RETAIN,
) -> str:
    if not math.isfinite(delta_x) or delta_x <= eps:
        return "OBSERVATION_NOT_REPLICATED"
    if not math.isfinite(delta_z) or delta_z <= eps:
        return "DESTROYS"
    if delta_z < delta_x - eps_retain:
        return "LOSES"
    return "SUFFICIENT"


class NSGRepEpi(nn.Module):
    """2-channel CNN + geometry MLP into the two GRU contact slots."""

    def __init__(self, hidden: int = 32, geom_dim: int = GEOM_DIM) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )
        self.geom = nn.Sequential(nn.Linear(int(geom_dim), 8), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(40, Z_DIM), nn.Tanh())
        self.gru = EpiGRU(STEP_DIM, hidden)

    def encode(self, maps: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = maps.shape
        visual = self.cnn(maps.reshape(batch * steps, channels, height, width))
        extra = self.geom(geom.reshape(batch * steps, geom.shape[-1]))
        return self.readout(torch.cat([visual, extra], dim=-1)).reshape(batch, steps, Z_DIM)

    def fuse(self, steps: torch.Tensor, maps: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        fused = steps.clone()
        fused[:, :, CONTACT_INSTANT_COL : CONTACT_WINDOW_COL + 1] = self.encode(maps, geom)
        return fused

    def forward(
        self,
        steps: torch.Tensor,
        maps: torch.Tensor,
        geom: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.gru(self.fuse(steps, maps, geom), lengths)


def nsg_maps(row: dict[str, Any], res: int = TAXEL_RES) -> np.ndarray:
    taxel = _ensure_taxel(row, res)
    n_steps = len(taxel)
    shear = row.get("taxel_shear")
    if shear is None:
        shear = np.zeros((n_steps, res, res), dtype=np.float64)
    shear = np.asarray(shear, dtype=np.float64)
    if shear.ndim == 2:
        shear = np.repeat(shear[None, :, :], n_steps, axis=0)
    shear = shear[:n_steps]
    if shear.shape != taxel.shape:
        shear = np.zeros_like(taxel)
    return np.stack([taxel, shear], axis=1)


def nsg_geom(row: dict[str, Any], n_steps: int | None = None) -> np.ndarray:
    if n_steps is None:
        n_steps = int(row["T"])
    geom = row.get("tactile_geom")
    if geom is None:
        return np.zeros((n_steps, GEOM_DIM), dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)
    if geom.ndim == 1:
        geom = np.repeat(geom[None, :], n_steps, axis=0)
    if len(geom) < n_steps:
        pad = np.zeros((n_steps - len(geom), GEOM_DIM), dtype=np.float64)
        geom = np.concatenate([geom, pad], axis=0)
    return geom[:n_steps]


def _pad_maps(batch: list[np.ndarray], res: int) -> torch.Tensor:
    length = max(len(item) for item in batch)
    padded = torch.zeros(len(batch), length, 2, res, res, dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, : len(item)] = torch.tensor(item, dtype=torch.float32)
    return padded


def _pad_geom(batch: list[np.ndarray]) -> torch.Tensor:
    length = max(len(item) for item in batch)
    padded = torch.zeros(len(batch), length, GEOM_DIM, dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, : len(item)] = torch.tensor(item, dtype=torch.float32)
    return padded


def _z_probs(
    model: NSGRepEpi,
    row: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    res: int,
    *,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    model.eval()
    steps = (zero_contact_slots(row["steps"]) - mean) / std
    maps = nsg_maps(row, res)
    geom = nsg_geom(row, len(maps))
    if shuffle and len(maps) > 1:
        generator = rng if rng is not None else np.random.default_rng(0)
        order = generator.permutation(len(maps))
        maps = maps[order]
        geom = geom[order]
    padded, lengths = _pad([steps])
    with torch.no_grad():
        pred = model(padded, _pad_maps([maps], res), _pad_geom([geom]), lengths)[0, : int(lengths[0])]
    return pred.cpu().numpy()


def _train_z(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7C4Config,
    seed: int,
) -> NSGRepEpi:
    torch.manual_seed(seed)
    model = NSGRepEpi(config.hidden)
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
            maps = _pad_maps([nsg_maps(row, config.taxel_res) for row in chunk], config.taxel_res)
            geom = _pad_geom([nsg_geom(row, int(row["T"])) for row in chunk])
            pred = model(padded, maps, geom, lengths)
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
                probs = _z_probs(model, row, mean, std, config.taxel_res)
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


def _c1_cfg(config: R3V7C4Config) -> R3V7C1Config:
    return R3V7C1Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
        taxel_res=config.taxel_res,
    )


def _aggregate(rows: list[dict[str, Any]], config: R3V7C4Config) -> dict[str, Any]:
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
    p0 = _train(train_n, val_n, mean_n, std_n, inner, warranted=True, seed=81)
    px = _train_obs(train, val, mean_n, std_n, OBS_NAME, config, seed=82)  # type: ignore[arg-type]
    pz = _train_z(train, val, mean_n, std_n, config, seed=83)
    s0 = [_gru_probs(p0, row, mean_n, std_n) for row in held_n]
    sx = [_obs_probs(px, row, mean_n, std_n, OBS_NAME, config.taxel_res) for row in held]
    sz = [_z_probs(pz, row, mean_n, std_n, config.taxel_res) for row in held]
    rng = np.random.default_rng(251)
    sz_shuf = [
        _z_probs(pz, row, mean_n, std_n, config.taxel_res, shuffle=True, rng=rng) for row in held
    ]
    l0 = bce_vs_target(s0, held)
    lx = bce_vs_target(sx, held)
    lz = bce_vs_target(sz, held)
    lz_shuf = bce_vs_target(sz_shuf, held)
    delta_x = float(l0 - lx) if math.isfinite(l0) and math.isfinite(lx) else math.nan
    delta_z = float(l0 - lz) if math.isfinite(l0) and math.isfinite(lz) else math.nan
    delta_shuf = float(l0 - lz_shuf) if math.isfinite(l0) and math.isfinite(lz_shuf) else math.nan
    verdict = classify_representation(
        delta_x, delta_z, eps=config.eps_info, eps_retain=config.eps_retain
    )
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "observation": OBS_NAME,
        "L_p0": l0,
        "L_px": lx,
        "L_pz": lz,
        "L_pz_shuffle": lz_shuf,
        "delta_x": delta_x,
        "delta_z": delta_z,
        "delta_z_shuffle": delta_shuf,
        "delta_nsg_c3_reference": C3_DELTA_NSG_REF,
        "eps": config.eps_info,
        "eps_retain": config.eps_retain,
        "verdict": verdict,
        "opens_belief_integration": verdict == "SUFFICIENT",
        "b5_not_integrated": True,
        "v7d_locked": True,
        "encoder_search": False,
    }


def run_r3_v7c4(
    output: str | Path,
    *,
    v7c3_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7C4Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7C4Config()
    _require_v7c3(Path(v7c3_summary))
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
            f"[V7C.4 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 31)
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
            "verdict": None,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; representation trichotomy not evaluated",
            "has_nsg": bool(
                any(row.get("tactile_geom") is not None for row in rows if row["domain"] == "contact")
            ),
        }
    )
    status = stage_status()
    status["R3-V7C.2"] = {"frozen": True, "locus": "FIELD"}
    status["R3-V7C.3"] = {"frozen": True, "observation": OBS_NAME, "stop_tactile_branch": False}
    status["R3-V7C.4"] = {
        "unlocked": True,
        "verdict": aggregate.get("verdict"),
        "opens_belief_integration": aggregate.get("opens_belief_integration"),
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
        "stage": "R3-V7C.4",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "kind": "representation sufficiency",
            "observation": OBS_NAME,
            "no_b5_integration": True,
            "no_encoder_search": True,
            "eps_info": cfg.eps_info,
            "eps_retain": cfg.eps_retain,
            "v7d": "locked",
            "v7c2_field_retained": True,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "verdict": aggregate.get("verdict"),
        "opens_belief_integration": aggregate.get("opens_belief_integration"),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

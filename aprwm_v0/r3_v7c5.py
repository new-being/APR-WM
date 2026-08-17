"""R3-V7C.5: belief integration of Z_NSG into frozen B5-S."""

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
from .r3_v7b3 import C0_STEP_FPR_MAX, DELTA, W_HIGH, W_LOW, R3V7B3Config, _train, spearman
from .r3_v7c import _fpr, _with_steps, h4_sensor_useful
from .r3_v7c1 import R3V7C1Config, Z_DIM, _collect_contact, _collect_mode_a
from .r3_v7c2 import EPS_INFO, bce_vs_target
from .r3_v7c3 import GEOM_DIM, _attach_mode_a_fields
from .r3_v7c4 import NSGRepEpi, _pad_geom, _pad_maps, nsg_geom, nsg_maps
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R3/R3_V7C5_PREREG.md"
SEEDS = (26101, 26111, 26121, 26131, 26141)
TRAIN_SEEDS = SEEDS[:2]
VAL_SEED = SEEDS[2]
HELD_SEEDS = SEEDS[3:]
EPS_TEMPORAL = 0.005
SMOKE_JOBS = (
    ("mode_a", 26101, 1.5, None, 1.0),
    ("contact", 26101, math.nan, "fast_pull", 1.0),
    ("contact", 26101, math.nan, "switch_cycle", 1.0),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R3V7C5Config:
    seeds: tuple[int, ...] = SEEDS
    hidden: int = 32
    window: int = 16
    epochs: int = 60
    lr: float = 1.0e-2
    delta: float = DELTA
    w_low: float = W_LOW
    w_high: float = W_HIGH
    c0_step_fpr_max: float = C0_STEP_FPR_MAX
    eps_info: float = EPS_INFO
    eps_temporal: float = EPS_TEMPORAL
    target_hz: float = 20.0
    taxel_res: int = TAXEL_RES


def _require_v7c4(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R3-V7C.5 locked until R3-V7C.4 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("R3-V7C.5 locked: V7C.4 scientific matrix has not been run")
    verdict = payload.get("verdict") or payload.get("aggregate", {}).get("verdict")
    if verdict != "SUFFICIENT":
        raise RuntimeError("R3-V7C.5 locked: V7C.4 verdict is not SUFFICIENT")
    return payload


def temporal_gate(delta_real: float, delta_shuf: float, eps: float = EPS_TEMPORAL) -> bool:
    if not math.isfinite(delta_real) or not math.isfinite(delta_shuf):
        return False
    return (delta_real - delta_shuf) > eps


class B5PlusNSG(nn.Module):
    """Frozen B5-S 12-D steps plus C.4 encoder z concatenated (14-D GRU)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.backbone = NSGRepEpi(hidden)
        self.gru = EpiGRU(STEP_DIM + Z_DIM, hidden)

    def encode(self, maps: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(maps, geom)

    def forward(
        self,
        steps: torch.Tensor,
        maps: torch.Tensor,
        geom: torch.Tensor,
        lengths: torch.Tensor,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        encoded = self.encode(maps, geom)
        if shuffle:
            encoded = _shuffle_time(encoded, lengths, rng)
        fused = torch.cat([steps, encoded], dim=-1)
        return self.gru(fused, lengths)


def _shuffle_time(
    encoded: torch.Tensor, lengths: torch.Tensor, rng: np.random.Generator | None
) -> torch.Tensor:
    generator = rng if rng is not None else np.random.default_rng(0)
    out = encoded.clone()
    for index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length > 1:
            order = generator.permutation(length)
            out[index, :length] = encoded[index, order]
    return out


def _fused_probs(
    model: B5PlusNSG,
    row: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    res: int,
    *,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    model.eval()
    steps = (np.asarray(row["steps"], dtype=np.float64) - mean) / std
    maps = nsg_maps(row, res)
    geom = nsg_geom(row, len(maps))
    padded, lengths = _pad([steps])
    with torch.no_grad():
        pred = model(
            padded,
            _pad_maps([maps], res),
            _pad_geom([geom]),
            lengths,
            shuffle=shuffle,
            rng=rng,
        )[0, : int(lengths[0])]
    return pred.cpu().numpy()


def _train_fused(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    config: R3V7C5Config,
    seed: int,
    *,
    shuffle: bool,
) -> B5PlusNSG:
    torch.manual_seed(seed)
    model = B5PlusNSG(config.hidden)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = math.inf
    stall = 0
    for epoch in range(config.epochs):
        model.train()
        order = np.random.default_rng(epoch + seed).permutation(len(train))
        epoch_rng = np.random.default_rng(epoch + seed + 17)
        for start in range(0, len(train), 8):
            chunk = [train[int(i)] for i in order[start : start + 8]]
            feats = [(np.asarray(row["steps"], dtype=np.float64) - mean) / std for row in chunk]
            padded, lengths = _pad(feats)
            maps = _pad_maps([nsg_maps(row, config.taxel_res) for row in chunk], config.taxel_res)
            geom = _pad_geom([nsg_geom(row, int(row["T"])) for row in chunk])
            pred = model(padded, maps, geom, lengths, shuffle=shuffle, rng=epoch_rng)
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
            val_rng = np.random.default_rng(seed + 99)
            for row in val:
                probs = _fused_probs(
                    model, row, mean, std, config.taxel_res, shuffle=shuffle, rng=val_rng
                )
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


def _c1_cfg(config: R3V7C5Config) -> R3V7C1Config:
    return R3V7C1Config(
        seeds=config.seeds,
        hidden=config.hidden,
        window=config.window,
        epochs=config.epochs,
        lr=config.lr,
        target_hz=config.target_hz,
        taxel_res=config.taxel_res,
    )


def _aggregate(rows: list[dict[str, Any]], config: R3V7C5Config) -> dict[str, Any]:
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS]
    val = [row for row in rows if int(row["seed"]) == VAL_SEED]
    held = [row for row in rows if int(row["seed"]) in HELD_SEEDS]
    labels = [int(row["y_inadequate"]) for row in held]
    train_s = _with_steps(train, "sensor")
    val_s = _with_steps(val, "sensor")
    held_s = _with_steps(held, "sensor")
    stack = np.concatenate([row["steps"] for row in train_s], axis=0)
    mean_s = np.mean(stack, axis=0)
    std_s = np.where(np.std(stack, axis=0) < 1.0e-12, 1.0, np.std(stack, axis=0))
    inner = R3V7B3Config(hidden=config.hidden, epochs=config.epochs, lr=config.lr)
    sensor = _train(train_s, val_s, mean_s, std_s, inner, warranted=True, seed=91)
    real = _train_fused(train_s, val_s, mean_s, std_s, config, seed=92, shuffle=False)
    shuf = _train_fused(train_s, val_s, mean_s, std_s, config, seed=93, shuffle=True)
    eval_rng = np.random.default_rng(261)
    series = {
        "sensor": [_gru_probs(sensor, row, mean_s, std_s) for row in held_s],
        "sz": [_fused_probs(real, row, mean_s, std_s, config.taxel_res) for row in held_s],
        "sz_shuf": [
            _fused_probs(shuf, row, mean_s, std_s, config.taxel_res, shuffle=True, rng=eval_rng)
            for row in held_s
        ],
    }
    c0_idx = [i for i, row in enumerate(held) if row["regime"] == "C0"]
    c1_idx = [i for i, row in enumerate(held) if row["regime"] != "C0"]
    tau = float(
        np.quantile(
            [
                float(_fused_probs(real, {**row, "steps": row["steps_sensor"]}, mean_s, std_s, config.taxel_res)[-1])
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
    c1_fin = {ch: brier_at_frac([series[ch][i] for i in c1_idx], c1_labels, 1.0) for ch in series}
    corrs, high_p, low_p = [], [], []
    for index in c1_idx:
        w = held[index]["w_evid"][: len(series["sz"][index])]
        p = series["sz"][index][: len(w)]
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
    loss = {ch: bce_vs_target(series[ch], held) for ch in series}
    delta_real = (
        float(loss["sensor"] - loss["sz"])
        if math.isfinite(loss["sensor"]) and math.isfinite(loss["sz"])
        else math.nan
    )
    delta_shuf = (
        float(loss["sensor"] - loss["sz_shuf"])
        if math.isfinite(loss["sensor"]) and math.isfinite(loss["sz_shuf"])
        else math.nan
    )
    h1 = bool(
        math.isfinite(mid["sz"])
        and math.isfinite(fin["sz"])
        and mid["sz"] <= mid["sensor"] + config.delta
        and fin["sz"] <= fin["sensor"] + config.delta
    )
    h2 = bool(math.isfinite(fpr["sz"]) and fpr["sz"] <= config.c0_step_fpr_max)
    h3 = bool(
        math.isfinite(mean_corr)
        and mean_corr > 0.0
        and math.isfinite(e_high)
        and math.isfinite(e_low)
        and e_high > e_low
    )
    h4 = h4_sensor_useful(
        brier_mid_s=mid["sz"],
        brier_mid_n=mid["sensor"],
        brier_final_s=fin["sz"],
        brier_final_n=fin["sensor"],
        b_c0_s=b_c0["sz"],
        b_c0_n=b_c0["sensor"],
        brier_c1_s=c1_fin["sz"],
        brier_c1_n=c1_fin["sensor"],
    )
    h5 = temporal_gate(delta_real, delta_shuf, config.eps_temporal)
    go = bool(h1 and h2 and h3 and h4 and h5)
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_held": len(held),
        "tau_dev_sz_c0_final": tau,
        "L": loss,
        "delta_real": delta_real,
        "delta_shuf": delta_shuf,
        "delta_temporal": (
            float(delta_real - delta_shuf)
            if math.isfinite(delta_real) and math.isfinite(delta_shuf)
            else math.nan
        ),
        "eps_temporal": config.eps_temporal,
        "h1_noninferior_vs_s": {
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
        "h4_useful_vs_s": {
            "brier_c1_final": c1_fin,
            "pass": h4,
        },
        "h5_temporal": {
            "delta_real": delta_real,
            "delta_shuf": delta_shuf,
            "gap": (
                float(delta_real - delta_shuf)
                if math.isfinite(delta_real) and math.isfinite(delta_shuf)
                else math.nan
            ),
            "eps": config.eps_temporal,
            "pass": h5,
        },
        "v7c5_go": go,
        "epistemic_usefulness": go,
        "representation_sufficiency": "PASS (V7C.4, frozen)",
        "encoder_search": False,
        "b5_s_kept": True,
        "unlocks_v7d": False,
    }


def run_r3_v7c5(
    output: str | Path,
    *,
    v7c4_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R3V7C5Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R3V7C5Config()
    _require_v7c4(Path(v7c4_summary))
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
            f"[V7C.5 {index}/{len(jobs)}] {domain} seed={seed} amp={amp} "
            f"script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed) + 33)
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
            "v7c5_go": None,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; belief-integration GO not evaluated",
            "has_nsg": bool(
                any(row.get("tactile_geom") is not None for row in rows if row["domain"] == "contact")
            ),
            "has_sensor_steps": bool(all("steps_sensor" in row for row in rows)),
        }
    )
    status = stage_status()
    status["R3-V7C.2"] = {"frozen": True, "locus": "FIELD"}
    status["R3-V7C.4"] = {"frozen": True, "verdict": "SUFFICIENT"}
    status["R3-V7C.5"] = {
        "unlocked": True,
        "v7c5_go": aggregate.get("v7c5_go"),
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
        "stage": "R3-V7C.5",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "kind": "belief integration",
            "b5_s_kept": True,
            "encoder_architecture": "V7C.4 NSG CNN+geom",
            "no_encoder_search": True,
            "eps_temporal": cfg.eps_temporal,
            "v7d": "locked",
            "representation_sufficiency": "PASS",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "v7c5_go": aggregate.get("v7c5_go"),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    _write_csv(root / "episodes.csv", slim)
    return summary

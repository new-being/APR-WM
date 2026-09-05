"""RTWX-O0RA0: fresh confirmation of the frozen RAB0 winner (scientific)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH
from .rtwx_o0bel0 import FORMAL_SEED as BEL0_FORMAL_SEED
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_o0rab0 import (
    PERIODS,
    RTWXO0RAB0Config,
    eval_cell,
    winner_config_hash,
    _load_assets,
    _train_s0,
)
from .rtwx_o0seq0 import (
    RTWXO0SEQ0Config,
    collect_sequences_numpy,
    collect_sequences_robotwin,
)
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0RA0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0ra0.winner_confirmation.v1"
FRESH_SEED = 37612
N_SEQ = 100
T_TRANS = 64
FAMILY_PATTERN = {
    "absolute": "periodic_absolute_reanchor_supported",
    "initial": "initial_anchor_relocalization_supported",
    "rolling": "rolling_keyframe_reanchor_supported",
}


@dataclass(frozen=True)
class RTWXO0RA0Config:
    output: str = "runs/rtwx_o0ra0"
    p0_run: str = "runs/rtwx_o0rab0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = 47604
    seq_seed: int = FRESH_SEED
    n_sequences: int = N_SEQ
    n_transitions: int = T_TRANS
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0RA0Config) -> RTWXO0RA0Config:
    if cfg.smoke:
        return replace(
            cfg,
            n_sequences=4,
            n_transitions=16,
            epochs_unet=4,
            backend="numpy",
        )
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0RA0 must not write there")


def _ra0_pattern(winner_kind: str, h_star_w: int, h_star_b0: int) -> tuple[str, list[str]]:
    tags: list[str] = []
    if h_star_b0 >= 64 and h_star_w >= 64:
        return "reanchor_not_required_on_fresh_split", tags
    if h_star_w >= 64 and h_star_w > h_star_b0:
        return FAMILY_PATTERN[winner_kind], tags
    return "non_gt_reanchor_insufficient", tags


def run_rtwx_o0ra0(output: str | Path, config: RTWXO0RA0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0RA0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    if int(cfg.seq_seed) == int(BEL0_FORMAL_SEED):
        raise RuntimeError("RA0 fresh seed must not equal BEL0/P0 seed 37611")

    p0 = json.loads((_resolve_apr(cfg.p0_run) / "summary.json").read_text())
    winner = p0.get("winner")
    if not winner:
        summary = {
            "pattern": "non_gt_reanchor_insufficient",
            "tags": ["p0_winner_none"],
            "scientific_result": True,
            "skipped": True,
        }
        _write_json(root / "summary.json", summary)
        return summary
    kind = str(winner["kind"])
    period = int(winner["period"])
    if kind not in FAMILY_PATTERN or period not in PERIODS:
        raise RuntimeError("P0 winner not in frozen RAB0 matrix")
    expected_hash = winner_config_hash(kind, period)
    if p0.get("winner_config_hash") != expected_hash:
        raise RuntimeError("RA0 winner config hash != P0 frozen winner")
    if p0.get("scientific_result") is not False:
        raise RuntimeError("P0 must be marked scientific_result=false")

    cache_obs = root / "cache" / "observations" / "obs.npz"
    cache_gt = root / "cache" / "gt" / "gt.npz"
    seq_cfg = RTWXO0SEQ0Config(
        output=str(root),
        natural_cache=cfg.natural_cache,
        p0_cache=cfg.p0_cache,
        r5_run=cfg.r5_run,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        seed=cfg.seed,
        seq_seed=cfg.seq_seed,
        n_seq=cfg.n_sequences,
        t_trans=cfg.n_transitions,
        epochs_unet=cfg.epochs_unet,
        smoke=False,
    )
    if not cache_obs.is_file():
        print(f"[rtwx-o0ra0] collect n={cfg.n_sequences} T={cfg.n_transitions} seed={cfg.seq_seed}", flush=True)
        if cfg.backend == "numpy":
            obs, gt = collect_sequences_numpy(seq_cfg)
        else:
            obs, gt = collect_sequences_robotwin(seq_cfg)
        cache_obs.parent.mkdir(parents=True, exist_ok=True)
        cache_gt.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_obs, **obs)
        np.savez_compressed(cache_gt, **gt)
    obs = dict(np.load(cache_obs))
    gt = dict(np.load(cache_gt))

    assets_cfg = RTWXO0RAB0Config(
        output=cfg.output,
        natural_cache=cfg.natural_cache,
        p0_cache=cfg.p0_cache,
        r5_run=cfg.r5_run,
        robotwin_repo=cfg.robotwin_repo,
        epochs_unet=cfg.epochs_unet,
        smoke=cfg.smoke,
    )
    d_o, lut, b2, n_const, _ = _load_assets(assets_cfg)
    s0 = _train_s0(assets_cfg)
    rng = np.random.default_rng(cfg.seed)

    print("[rtwx-o0ra0] eval B0", flush=True)
    _, b0_row = eval_cell(
        obs, gt, kind="none", period=0,
        s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o,
        n_seq=cfg.n_sequences, rng=rng,
    )
    print(f"[rtwx-o0ra0] eval winner {kind} K={period}", flush=True)
    per_seq_w, w_row = eval_cell(
        obs, gt, kind=kind, period=period,
        s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o,
        n_seq=cfg.n_sequences, rng=rng,
    )
    pattern, tags = _ra0_pattern(kind, int(w_row["H_star"]), int(b0_row["H_star"]))
    summary = {
        "pattern": pattern,
        "tags": tags,
        "scientific_result": True,
        "winner": {"kind": kind, "period": period},
        "winner_config_hash": expected_hash,
        "H_star_B0": int(b0_row["H_star"]),
        "H_star_winner": int(w_row["H_star"]),
        "B0": {k: v for k, v in b0_row.items() if k != "horizon"},
        "winner_row": {k: v for k, v in w_row.items() if k != "horizon"},
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "fresh_seed": cfg.seq_seed,
        "unlocks_o0e1": False,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH, "seq_seed": cfg.seq_seed})
    _write_json(root / "per_sequence_winner.json", per_seq_w)
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0ra0] pattern={pattern}", flush=True)
    return summary

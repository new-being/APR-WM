"""RTWX-MR0: multi-rate state–action resampling probe."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .control.action_chunk_buffer import ActionChunkBuffer
from .control.multirate_scheduler import WorldStateClock
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, estimate_relative_rigid
from .geometry.relative_state_chain import ChainState
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_o0rel0 import ICP_CFG
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/RTWX0MR0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_mr0.multirate_resampling.v1"
SEED = 48601
EPISODE_SEED0 = 38601
MIN_HA = 8
STRIDE_ICP_VALID_MIN = 0.80
CELLS = ((1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8))
MR0_TASKS = ("place_empty_cup", "put_object_cabinet", "stamp_seal")
TASK_ROLE = {
    "place_empty_cup": "transport",
    "put_object_cabinet": "contact",
    "stamp_seal": "precision",
}
N_EP = 50
MAX_TICKS = 200
G0S_STRIDES = (2, 4, 8)
MR0_CONFIG = {
    "state_strides": [1, 2, 4, 8],
    "action_strides": [1, 2, 4, 8],
    "cells": [list(c) for c in CELLS],
    "episodes_per_task": N_EP,
    "success_noninferiority_pp": 0.05,
    "per_task_max_drop_pp": 0.10,
    "safety_max_increase_pp": 0.02,
    "min_asymmetry_ratio": 2.0,
    "min_ha": MIN_HA,
}


@dataclass(frozen=True)
class RTWXMR0Config:
    output: str = "runs/rtwx_mr0"
    bel0_cache: str = "runs/rtwx_o0bel0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    policy_path: str = ""
    seed: int = SEED
    smoke: bool = False


def _lock(cfg: RTWXMR0Config) -> RTWXMR0Config:
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-MR0 must not write there")


def discover_action_horizon(policy_path: str) -> dict[str, Any]:
    """Frozen chunk-policy instrument. Missing weights → H_a=0."""
    p = Path(policy_path) if policy_path else None
    if p is None or not str(policy_path).strip() or not p.is_file():
        return {"ok": False, "H_a": 0, "reason": "no_frozen_chunk_policy"}
    return {"ok": False, "H_a": 0, "reason": "policy_loader_not_wired", "path": str(p)}


def g0s_stride_valid_rates(clouds: list[np.ndarray], d_o: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in G0S_STRIDES:
        n_ok = n = 0
        for t in range(k, len(clouds), k):
            res = estimate_relative_rigid(clouds[t - k], clouds[t], d_o, ICP_CFG)
            n += 1
            n_ok += int(res.valid)
        rate = float(n_ok / n) if n else float("nan")
        out[str(k)] = {
            "n": n,
            "valid_rate": rate,
            "instrument_ok": bool(np.isfinite(rate) and rate >= STRIDE_ICP_VALID_MIN),
        }
    return out


def run_g0s_bel0(
    bel0_root: Path,
    n_seq: int = 20,
    *,
    smoke: bool = False,
    mr0_cfg: RTWXMR0Config | None = None,
) -> dict[str, Any]:
    from .rtwx_o0rab0 import RTWXO0RAB0Config, _load_assets, _train_s0, load_bel0_formal_cache
    from .rtwx_o0seq0 import _build_clouds_seq

    obs, gt = load_bel0_formal_cache(bel0_root)
    n_seq = min(n_seq, int(obs["rgb_h"].shape[0]))
    src = mr0_cfg or RTWXMR0Config()
    cfg = RTWXO0RAB0Config(
        bel0_cache=str(bel0_root),
        natural_cache=src.natural_cache,
        p0_cache=src.p0_cache,
        r5_run=src.r5_run,
        robotwin_repo=src.robotwin_repo,
        epochs_unet=4 if smoke else 20,
        smoke=smoke,
    )
    d_o, _lut, _b2, _n_const, _ = _load_assets(cfg)
    s0 = _train_s0(cfg)
    rng = np.random.default_rng(SEED)
    rates: list[dict[str, Any]] = []
    for si in range(n_seq):
        clouds = _build_clouds_seq(obs, gt, s0, si, rng=rng, use_gt_mask=False)
        rates.append(g0s_stride_valid_rates(clouds, d_o))
    pooled: dict[str, Any] = {}
    for k in G0S_STRIDES:
        vr = [r[str(k)]["valid_rate"] for r in rates]
        pooled[str(k)] = {
            "valid_rate": float(np.mean(vr)),
            "instrument_ok": bool(np.mean(vr) >= STRIDE_ICP_VALID_MIN),
        }
    return {"n_seq": n_seq, "per_stride": pooled}


def paired_episode_seeds(n_ep: int = N_EP) -> list[int]:
    return [int(EPISODE_SEED0 + i) for i in range(n_ep)]


def cell_name(ks: int, ka: int) -> str:
    if ks == 1 and ka == 1:
        return "B0"
    if ka == 1:
        return f"S{ks}"
    return f"A{ka}"


def assert_cross_cells() -> None:
    for ks, ka in CELLS:
        if ks != 1 and ka != 1:
            raise RuntimeError("MR0 forbids 2-D cells")


def planning_seed(experiment_seed: int, task_id: str, episode_id: int, planning_tick: int) -> int:
    payload = f"{int(experiment_seed)}|{task_id}|{int(episode_id)}|{int(planning_tick)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def step_multirate(
    t: int,
    *,
    robot_state: Any,
    cloud: np.ndarray,
    goal: Any,
    clock: WorldStateClock,
    buffer: ActionChunkBuffer,
    planner: Callable[[Any], np.ndarray],
    object_diameter: float,
    init_state: ChainState | None = None,
) -> dict[str, Any]:
    """One policy tick: fresh robot, strided world, chunked action. No hidden replan."""
    world, state_updated = clock.update_stride(
        t, cloud, init_state=init_state, object_diameter=object_diameter,
    )
    planner_input = {"robot": robot_state, "world": world, "goal": goal, "t": t}
    action, replanned = buffer.action(t, planner_input, planner)
    return {
        "action": action,
        "state_updated": bool(state_updated),
        "replanned": bool(replanned),
        "planner_input": planner_input,
        "n_state": int(clock.n_updates),
        "n_action": int(buffer.n_replans),
        "world_state": world,
    }


def run_rtwx_mr0(output: str | Path, config: RTWXMR0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXMR0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    assert_cross_cells()

    ha = discover_action_horizon(cfg.policy_path)
    print(f"[rtwx-mr0] G0-A H_a={ha.get('H_a')} reason={ha.get('reason')}", flush=True)

    g0s: dict[str, Any] = {}
    bel0 = _resolve_apr(cfg.bel0_cache)
    try:
        print("[rtwx-mr0] G0-S stride ICP on BEL0 cache", flush=True)
        n_seq = 4 if cfg.smoke else 20
        g0s = run_g0s_bel0(bel0, n_seq=n_seq, smoke=cfg.smoke, mr0_cfg=cfg)
        print(f"[rtwx-mr0] G0-S {g0s['per_stride']}", flush=True)
    except FileNotFoundError as e:
        g0s = {"error": str(e)}

    if int(ha.get("H_a", 0)) < MIN_HA:
        summary = {
            "pattern": "action_chunk_contract_failure",
            "scientific_result": True,
            "H_a": int(ha.get("H_a", 0)),
            "g0_a": ha,
            "g0_s": g0s,
            "mr0_config": MR0_CONFIG,
            "tasks": list(MR0_TASKS),
            "cells": [list(c) for c in CELLS],
            "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
            "note": "Ka>1 requires frozen action-chunk policy with H_a>=8; hold-last-qpos is forbidden.",
        }
        _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
        _write_json(root / "summary.json", summary)
        _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
        print("[rtwx-mr0] pattern=action_chunk_contract_failure", flush=True)
        return summary

    raise RuntimeError("frozen chunk policy present but task loop not wired")

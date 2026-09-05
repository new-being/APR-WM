"""RTWX-TASK-X1-I0: prove sampler can invert known noise. Same X1 checkpoint. No clip, no Stage B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation.ac0_offline_metrics import chunk_diversity
from .evaluation.task_x1_i0 import c0_oracle_reverse, c1_oracle_x0, c2_model_x0
from .policy.chunk_dataset import ActionChunkDataset
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.diffusion_denoiser_mlp import DiffusionChunkDenoiser
from .policy.diffusion_schedule import TRAIN_STEPS, CosineSchedule
from .policy.task_embedding import STATE_DIM, TASK_NAMES
from .rtwx_ac0 import load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0TASKX1I0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_task_x1_i0.sampler_closure.v1"
C0_MAX_ABS = 1e-3
C1_MAX_ABS_LOW = 1e-5
C1_MAX_ABS_T99 = 1e-3
C2_MAX_L1_NORM = 0.50
C1_TS = (10, 50, 99)


@dataclass(frozen=True)
class RTWXTASKX1I0Config:
    output: str = "runs/rtwx_task_x1_i0"
    x1_run: str = "runs/rtwx_task_x1"
    ac0_run: str = "runs/rtwx_ac0"


def _val_loader(ac0: Path) -> DataLoader:
    man = json.loads((ac0 / "split_manifest.json").read_text())
    val = []
    for task in MR0_TASKS:
        for i in man[task]["val"]:
            val.append(load_cached_episode(ac0 / "cache" / task / f"ep_{int(i):03d}.npz", task))
    return DataLoader(ActionChunkDataset(val), batch_size=32, shuffle=False)


def run_rtwx_task_x1_i0(output: str | Path, config: RTWXTASKX1I0Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXTASKX1I0Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    x1 = _resolve_apr(cfg.x1_run)
    ckpt = torch.load(str(x1 / "B1" / "best.pt"), map_location="cpu", weights_only=False)
    assert ckpt.get("schema") == "aprwm.task_x1.diffusion_chunk.v1"
    assert bool(ckpt.get("use_process", True)) is False
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    ac0_b0 = torch.load(str(ac0 / "B0" / "best.pt"), map_location="cpu", weights_only=False)
    same_norm = np.allclose(norm.mu_a, np.asarray(ac0_b0["normalizer"]["mu_a"]))
    den = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES))
    den.load_state_dict(ckpt["state_dict"])
    den.eval()
    sched = CosineSchedule(TRAIN_STEPS, "cpu")
    loader = _val_loader(ac0)
    batch = next(iter(loader))
    x0 = torch.from_numpy(norm.n_act(batch["chunk"].numpy()).astype(np.float32))
    st = torch.from_numpy(norm.n_state(batch["state"].numpy()).astype(np.float32))
    tid = batch["task_id"]
    g = torch.Generator()
    g.manual_seed(0)
    eps = torch.randn(x0.shape, generator=g)

    x0_c0 = c0_oracle_reverse(x0, eps, sched)
    c0_err = float((x0_c0 - x0).abs().max())
    c0_ok = c0_err < C0_MAX_ABS

    c1 = {}
    c1_ok = True
    for t in C1_TS:
        hat = c1_oracle_x0(x0, eps, t, sched)
        err = float((hat - x0).abs().max())
        lim = C1_MAX_ABS_T99 if t == 99 else C1_MAX_ABS_LOW
        c1[str(t)] = {"max_abs": err, "ok": err < lim}
        c1_ok = c1_ok and err < lim

    c2 = {}
    c2_ok = True
    g = torch.Generator()
    g.manual_seed(0)
    eps = torch.randn(x0.shape, generator=g)
    for t in C1_TS:
        x0_c2, eps_hat = c2_model_x0(den, x0, st, tid, t, sched, eps)
        l1 = float((x0_c2 - x0).abs().mean())
        rec = {
            "l1_norm": l1,
            "max_abs_norm": float((x0_c2 - x0).abs().max()),
            "eps_mse": float(((eps_hat - eps) ** 2).mean()),
            "l1_raw": float(np.abs(norm.denorm_act(x0_c2.numpy()) - norm.denorm_act(x0.numpy())).mean()),
            "d_pred": chunk_diversity(norm.denorm_act(x0_c2.numpy())),
            "d_demo": chunk_diversity(norm.denorm_act(x0.numpy())),
        }
        rec["ok"] = l1 < C2_MAX_L1_NORM
        c2[str(t)] = rec
        c2_ok = c2_ok and rec["ok"]
    c2_primary = c2["99"]

    if (not c0_ok) or (not c1_ok):
        pattern = "diffusion_sampler_contract_failure"
    elif not c2_ok:
        pattern = "diffusion_denoiser_insufficient"
    else:
        pattern = "diffusion_sampler_closed"

    space = {
        "clean_action_space": "zscore_from_ac0_train",
        "mu_a": [float(x) for x in np.asarray(norm.mu_a).reshape(-1)[:4]],
        "sig_a": [float(x) for x in np.asarray(norm.sig_a).reshape(-1)[:4]],
        "matches_ac0_b0_normalizer": bool(same_norm),
        "denorm_once_after_sample": True,
        "prediction_type": "epsilon",
        "abar_0": float(sched.abar[0]),
        "abar_99": float(sched.abar[99]),
        "no_clip": True,
    }
    summary = {
        "pattern": pattern,
        "scientific_result": True,
        "purpose": "sampler_closure",
        "same_checkpoint": str(x1 / "B1" / "best.pt"),
        "retrained": False,
        "clipped": False,
        "stage_b_run": False,
        "space": space,
        "C0": {"ok": c0_ok, "max_abs_norm": c0_err, "threshold": C0_MAX_ABS},
        "C1": {"ok": c1_ok, "per_t": c1},
        "C2": {
            "ok": c2_ok,
            "primary_t": 99,
            "l1_norm": c2_primary["l1_norm"],
            "max_abs_norm": c2_primary["max_abs_norm"],
            "eps_mse_t99": c2_primary["eps_mse"],
            "l1_raw": c2_primary["l1_raw"],
            "d_pred": c2_primary["d_pred"],
            "d_demo": c2_primary["d_demo"],
            "threshold_l1_norm": C2_MAX_L1_NORM,
            "per_t": c2,
        },
        "unlocks_stage_b": False,
        "unlocks_retrain": pattern == "diffusion_denoiser_insufficient",
        "fix_sampler_then_restage_a": pattern == "diffusion_sampler_contract_failure",
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-task-x1-i0] pattern={pattern} C0={c0_ok} C1={c1_ok} C2={c2_ok} c0_err={c0_err:.2e} c2_l1={c2_primary['l1_norm']:.3f}", flush=True)
    return summary

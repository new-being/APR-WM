"""RTWX-TASK-X1-I1: freeze everything except prediction_type. epsilon vs v vs x0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluation.task_x1_i1 import AUDIT_TS, rec_gates, rec_metrics, reconstruct_x0
from .evaluation.task_x1_offline import denoise_val_chunks, stage_a_gates
from .policy.chunk_dataset import ActionChunkDataset
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.diffusion_chunk_policy import DiffusionChunkPolicy
from .policy.diffusion_denoiser_mlp import DiffusionChunkDenoiser
from .policy.diffusion_parameterization import diffusion_target
from .policy.diffusion_schedule import INFER_STEPS, TRAIN_STEPS, CosineSchedule
from .policy.task_embedding import STATE_DIM, TASK_NAMES
from .rtwx_ac0 import load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_task_x1 import TASK_X1_TRAIN, _load_ac0_episodes
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0TASKX1I1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_task_x1_i1.parameterization.v1"
CKPT_SCHEMA = "aprwm.task_x1_i1.diffusion_chunk.v1"
TIE_REL = 0.10
BRANCHES = (
    ("B0", "epsilon", False),
    ("B1", "v_prediction", True),
    ("B2", "sample", True),
)


@dataclass(frozen=True)
class RTWXTASKX1I1Config:
    output: str = "runs/rtwx_task_x1_i1"
    x1_run: str = "runs/rtwx_task_x1"
    ac0_run: str = "runs/rtwx_ac0"
    smoke: bool = False


def _val_ldiff(model, sched, loader, norm, device: str, ptype: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    idx0 = 0
    with torch.no_grad():
        for batch in loader:
            s = torch.from_numpy(norm.n_state(batch["state"].numpy()).astype(np.float32)).to(device)
            a = torch.from_numpy(norm.n_act(batch["chunk"].numpy()).astype(np.float32)).to(device)
            tid = batch["task_id"].to(device)
            b = int(s.shape[0])
            t = torch.tensor([((idx0 + i) * 7) % TRAIN_STEPS for i in range(b)], device=device, dtype=torch.long)
            gen = torch.Generator()
            gen.manual_seed(10007 + idx0)
            noise = torch.randn(a.shape, generator=gen).to(device)
            noisy = sched.q_sample(a, t, noise)
            pred = model(noisy_action=noisy, diffusion_t=t, state=s, task_id=tid)
            target = diffusion_target(a, noise, t, sched, ptype)
            tot += float(F.mse_loss(pred, target, reduction="sum").item())
            n += int(np.prod(a.shape))
            idx0 += b
    return tot / max(n, 1)


def train_ptype(train_eps, val_eps, norm, device: str, out_dir: Path, smoke: bool, ptype: str) -> dict[str, Any]:
    tr_ds = ActionChunkDataset(train_eps)
    va_ds = ActionChunkDataset(val_eps)
    model = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
    sched = CosineSchedule(TRAIN_STEPS, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=TASK_X1_TRAIN["lr"], weight_decay=TASK_X1_TRAIN["weight_decay"])
    bs = min(int(TASK_X1_TRAIN["batch_size"]), len(tr_ds))
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=min(256, len(va_ds)), shuffle=False)
    max_ep = 3 if smoke else int(TASK_X1_TRAIN["max_epochs"])
    best = 1e9
    best_state = None
    curve = []
    n_param = int(sum(p.numel() for p in model.parameters()))
    for epoch in range(max_ep):
        model.train()
        tot, n = 0.0, 0
        for batch in tr_ld:
            s = torch.from_numpy(norm.n_state(batch["state"].numpy()).astype(np.float32)).to(device)
            a = torch.from_numpy(norm.n_act(batch["chunk"].numpy()).astype(np.float32)).to(device)
            tid = batch["task_id"].to(device)
            t = torch.randint(0, TRAIN_STEPS, (s.shape[0],), device=device)
            noise = torch.randn_like(a)
            noisy = sched.q_sample(a, t, noise)
            pred = model(noisy_action=noisy, diffusion_t=t, state=s, task_id=tid)
            loss = F.mse_loss(pred, diffusion_target(a, noise, t, sched, ptype))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * s.shape[0]
            n += int(s.shape[0])
        vl = _val_ldiff(model, sched, va_ld, norm, device, ptype)
        curve.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": vl})
        print(f"[rtwx-task-x1-i1] ptype={ptype} ep={epoch} val={vl:.5f}", flush=True)
        if vl + 1e-8 < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_meta = dict(TASK_X1_TRAIN)
    train_meta["prediction_type"] = ptype
    train_meta["early_stop_patience"] = 0
    ckpt = {
        "schema": CKPT_SCHEMA,
        "state_dict": model.state_dict(),
        "normalizer": norm.as_dict(),
        "use_process": False,
        "state_dim": STATE_DIM,
        "horizon": 8,
        "action_dim": 14,
        "train_steps": TRAIN_STEPS,
        "infer_steps": INFER_STEPS,
        "ddim_eta": 0.0,
        "n_param": n_param,
        "val_loss": best,
        "prediction_type": ptype,
        "train": train_meta,
    }
    pt = out_dir / "best.pt"
    torch.save(ckpt, pt)
    _write_json(out_dir / "metrics.json", {"n_param": n_param, "val_loss": best, "prediction_type": ptype, "curve": curve})
    return {"model": model, "sched": sched, "ckpt_path": str(pt), "n_param": n_param, "val_loss": best, "curve": curve, "prediction_type": ptype}


def _load_x1_epsilon(x1: Path, device: str):
    ck = torch.load(str(x1 / "B1" / "best.pt"), map_location=device, weights_only=False)
    den = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
    den.load_state_dict(ck["state_dict"])
    den.eval()
    return den, CosineSchedule(TRAIN_STEPS, device=device), ck


def _audit_branch(model, sched, x0, st, tid, eps, norm, ptype: str) -> dict[str, Any]:
    per_t = {}
    for t in AUDIT_TS:
        hat = reconstruct_x0(model, x0, st, tid, int(t), sched, eps, ptype)
        per_t[str(t)] = rec_metrics(hat, x0, norm)
    return rec_gates(per_t)


def _pick_winner(rec: dict[str, Any], stage_a: dict[str, Any]) -> str | None:
    passers = [k for k in ("B1", "B2") if stage_a.get(k, {}).get("pass")]
    if not passers:
        return None
    if len(passers) == 1:
        return passers[0]
    l1a = rec["B1"]["l1_norm"]["99"]
    l1b = rec["B2"]["l1_norm"]["99"]
    scale = max(min(l1a, l1b), 1e-8)
    if abs(l1a - l1b) / scale >= TIE_REL:
        return "B1" if l1a < l1b else "B2"
    r1 = rec["B1"]["d_pred"]["99"] / max(rec["B1"]["d_demo"]["99"], 1e-12)
    r2 = rec["B2"]["d_pred"]["99"] / max(rec["B2"]["d_demo"]["99"], 1e-12)
    rscale = max(min(r1, r2), 1e-8)
    if abs(r1 - r2) / rscale >= TIE_REL:
        return "B1" if r1 < r2 else "B2"
    return "B1"


def _pattern(rec: dict[str, Any]) -> str:
    p0, p1, p2 = rec["B0"]["pass"], rec["B1"]["pass"], rec["B2"]["pass"]
    if (not p1) and (not p2):
        return "diffusion_high_noise_learning_failure"
    if p0 and p1 and p2:
        return "parameterization_not_the_bottleneck"
    if p1 and p2:
        return "epsilon_parameterization_pathology_supported"
    if p1 and (not p2):
        return "v_parameterization_supported"
    return "direct_sample_prediction_supported"


def run_rtwx_task_x1_i1(output: str | Path, config: RTWXTASKX1I1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXTASKX1I1Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    x1 = _resolve_apr(cfg.x1_run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac0_b0 = torch.load(str(ac0 / "B0" / "best.pt"), map_location="cpu", weights_only=False)
    norm = ChunkNormalizer.from_dict(ac0_b0["normalizer"])
    train_eps, val_eps, _ = _load_ac0_episodes(ac0)
    models: dict[str, Any] = {}

    den0, sched0, ck0 = _load_x1_epsilon(x1, device)
    models["B0"] = {"model": den0, "sched": sched0, "prediction_type": "epsilon", "retrained": False, "ckpt_path": str(x1 / "B1" / "best.pt"), "n_param": int(ck0.get("n_param", 0)), "val_loss": ck0.get("val_ldiff")}
    print(f"[rtwx-task-x1-i1] device={device} B0 frozen epsilon from X1", flush=True)

    for bid, ptype, train in BRANCHES:
        if not train:
            continue
        bdir = root / bid
        if (bdir / "best.pt").is_file() and (bdir / "metrics.json").is_file():
            print(f"[rtwx-task-x1-i1] resume {bid}", flush=True)
            ck = torch.load(str(bdir / "best.pt"), map_location=device, weights_only=False)
            den = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
            den.load_state_dict(ck["state_dict"])
            den.eval()
            models[bid] = {
                "model": den,
                "sched": CosineSchedule(TRAIN_STEPS, device=device),
                "prediction_type": ptype,
                "retrained": True,
                "ckpt_path": str(bdir / "best.pt"),
                "n_param": int(ck.get("n_param", 0)),
                "val_loss": ck.get("val_loss"),
            }
        else:
            packed = train_ptype(train_eps, val_eps, norm, device, bdir, cfg.smoke, ptype)
            packed["model"].eval()
            packed["retrained"] = True
            models[bid] = packed

    loader = DataLoader(ActionChunkDataset(val_eps), batch_size=32, shuffle=False)
    batch = next(iter(loader))
    x0 = torch.from_numpy(norm.n_act(batch["chunk"].numpy()).astype(np.float32)).to(device)
    st = torch.from_numpy(norm.n_state(batch["state"].numpy()).astype(np.float32)).to(device)
    tid = batch["task_id"].to(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    eps = torch.randn(x0.shape, generator=g).to(device)

    rec = {}
    for bid, ptype, _ in BRANCHES:
        (root / bid).mkdir(parents=True, exist_ok=True)
        rec[bid] = _audit_branch(models[bid]["model"], models[bid]["sched"], x0, st, tid, eps, norm, ptype)
        rec[bid]["prediction_type"] = ptype
        rec[bid]["retrained"] = models[bid]["retrained"]
        rec[bid]["val_loss"] = models[bid]["val_loss"]
        rec[bid]["n_param"] = models[bid]["n_param"]
        print(f"[rtwx-task-x1-i1] rec {bid} pass={rec[bid]['pass']} l1_99={rec[bid]['l1_norm']['99']:.4f}", flush=True)
        _write_json(root / bid / "reconstruction.json", rec[bid])

    va_ld = DataLoader(ActionChunkDataset(val_eps), batch_size=64, shuffle=False)
    dummy = np.asarray(norm.n_state(ActionChunkDataset(val_eps)[0]["state"].numpy()[None])[0], dtype=np.float64)
    stage_a: dict[str, Any] = {}
    for bid, ptype, _ in BRANCHES:
        if not rec[bid]["pass"]:
            stage_a[bid] = {"pass": False, "skipped": True, "reason": "reconstruction_gate_fail"}
            continue
        pol = DiffusionChunkPolicy(models[bid]["model"], models[bid]["sched"], device, prediction_type=ptype).eval()
        pred, tgt = denoise_val_chunks(pol, va_ld, norm, device)
        gates = stage_a_gates(pred, tgt, pol, dummy)
        stage_a[bid] = gates
        print(f"[rtwx-task-x1-i1] Stage A {bid} pass={gates['pass']} l1={gates['l1']:.4f}", flush=True)
        _write_json(root / bid / "stage_a.json", gates)

    winner = _pick_winner(rec, stage_a)
    pattern = _pattern(rec)
    frozen = {
        "split": "AC0",
        "state": "s_t,z_g",
        "H_a": 8,
        "d_a": 14,
        "action_norm": "zscore",
        "denoiser": "256x3_mlp",
        "schedule": "cosine",
        "K_train": TRAIN_STEPS,
        "N_infer": INFER_STEPS,
        "eta": 0.0,
        "epochs": 3 if cfg.smoke else 100,
        "early_stop": False,
        "clip": False,
        "process": False,
        "history": False,
        "closed_loop": False,
        "min_snr": False,
        "truncated_schedule": False,
        "only_delta": "prediction_type",
    }
    summary = {
        "pattern": pattern,
        "scientific_result": True,
        "purpose": "parameterization_breadth_probe",
        "winner": winner,
        "hypothesis": "stage_a_explosion_from_epsilon_low_snr_amplification",
        "not_claimed": "v_prediction_always_better",
        "selection_metric": "x0_space_reconstruction_then_stage_a",
        "losses_not_comparable": True,
        "frozen": frozen,
        "reconstruction": {k: {kk: rec[k][kk] for kk in ("pass", "G0_mid_noise", "G1_t99_l1", "G2_t99_diversity", "l1_norm", "d_pred", "d_demo", "prediction_type", "retrained")} for k in rec},
        "stage_a": {k: {"pass": stage_a[k].get("pass"), "skipped": stage_a[k].get("skipped", False), "l1": stage_a[k].get("l1"), "d_pred": stage_a[k].get("d_pred"), "d_demo": stage_a[k].get("d_demo")} for k in stage_a},
        "unlocks_stage_b": False,
        "qualified_for_x1_stage_b": winner is not None,
        "stage_b_seeds_if_restart": 52601,
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-task-x1-i1] winner={winner} pattern={pattern}", flush=True)
    return summary

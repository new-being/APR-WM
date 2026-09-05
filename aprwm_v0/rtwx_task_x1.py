"""RTWX-TASK-X1: conditional diffusion vs frozen AC0-B0. Same data/state/chunk/Ka=1. Does not write MR0-P0."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .policy.task_embedding import STATE_DIM, TASK_NAMES, TASK_TO_ID, pack_state
from .evaluation.ac0_rollout_eval import predict_chunk, rollout_episode, wrap_topp_large_jump
from .evaluation.task_x1_offline import denoise_val_chunks, stage_a_gates
from .policy.chunk_dataset import ActionChunkDataset
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.diffusion_chunk_policy import DiffusionChunkPolicy, policy_seed
from .policy.diffusion_denoiser_mlp import DiffusionChunkDenoiser
from .policy.diffusion_schedule import INFER_STEPS, TRAIN_STEPS, CosineSchedule
from .policy.explicit_chunk_mlp import ExplicitChunkPolicy
from .rtwx_ac0 import PROCESS_DIM, TASK_EMB, TASK_STEP_LIM, load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json
from .task_x0 import _install_mplib_fallback, _load_task_args, _setup_env

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0TASKX1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_task_x1.diffusion_chunk.v1"
CKPT_SCHEMA = "aprwm.task_x1.diffusion_chunk.v1"
BAKEOFF_SEED0 = 52601
P0_FROZEN = "runs/rtwx_mr0_p0/frozen_chunk_policy.pt"
TASK_X1_TRAIN = {
    "action_horizon": 8,
    "action_dim": 14,
    "diffusion_train_steps": TRAIN_STEPS,
    "diffusion_infer_steps": INFER_STEPS,
    "ddim_eta": 0.0,
    "hidden_dim": 256,
    "hidden_layers": 3,
    "condition_dim": 128,
    "time_embed_dim": 64,
    "batch_size": 512,
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "early_stop_patience": 10,
    "prediction_type": "epsilon",
}


I1_B2_CKPT = "runs/rtwx_task_x1_i1/B2/best.pt"
STAGE_B_JOB_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class RTWXTASKX1Config:
    output: str = "runs/rtwx_task_x1"
    ac0_run: str = "runs/rtwx_ac0"
    robotwin_repo: str = "/root/RoboTwin"
    smoke: bool = False
    workers: int = 8
    stage_b_only: bool = False
    b1_ckpt: str = I1_B2_CKPT


def _refuse_p0_write() -> None:
    _ = _resolve_apr(P0_FROZEN)


def _load_ac0_episodes(ac0: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    man = json.loads((ac0 / "split_manifest.json").read_text())
    train, val, test = [], [], []
    for task in MR0_TASKS:
        sp = man[task]
        for split, bucket in (("train", train), ("val", val), ("test", test)):
            for i in sp[split]:
                bucket.append(load_cached_episode(ac0 / "cache" / task / f"ep_{int(i):03d}.npz", task))
    return train, val, test


def _val_ldiff(model, sched: CosineSchedule, loader, norm: ChunkNormalizer, device: str) -> float:
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
            tot += float(F.mse_loss(pred, noise, reduction="sum").item())
            n += int(np.prod(a.shape))
            idx0 += b
    return tot / max(n, 1)


def train_diffusion(train_eps, val_eps, norm: ChunkNormalizer, device: str, out_dir: Path, smoke: bool) -> dict[str, Any]:
    tr_ds = ActionChunkDataset(train_eps)
    va_ds = ActionChunkDataset(val_eps)
    model = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
    sched = CosineSchedule(TRAIN_STEPS, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=TASK_X1_TRAIN["lr"], weight_decay=TASK_X1_TRAIN["weight_decay"])
    bs = min(int(TASK_X1_TRAIN["batch_size"]), len(tr_ds))
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=min(256, len(va_ds)), shuffle=False)
    max_ep = 3 if smoke else int(TASK_X1_TRAIN["max_epochs"])
    patience = 2 if smoke else int(TASK_X1_TRAIN["early_stop_patience"])
    best = 1e9
    best_state = None
    bad = 0
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
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * s.shape[0]
            n += int(s.shape[0])
        vl = _val_ldiff(model, sched, va_ld, norm, device)
        curve.append({"epoch": epoch, "train_ldiff": tot / max(n, 1), "val_ldiff": vl})
        print(f"[rtwx-task-x1] ep={epoch} val_Ldiff={vl:.5f}", flush=True)
        if vl + 1e-8 < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "val_ldiff": best,
        "train": TASK_X1_TRAIN,
    }
    pt = out_dir / "best.pt"
    torch.save(ckpt, pt)
    _write_json(out_dir / "metrics.json", {"n_param": n_param, "val_ldiff": best, "curve": curve})
    return {"model": model, "sched": sched, "ckpt_path": str(pt), "n_param": n_param, "val_ldiff": best, "curve": curve}


def _x1_worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    repo = Path(job["robotwin_repo"])
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    device = str(job["device"])
    ckpt = torch.load(str(job["ckpt_path"]), map_location=device, weights_only=False)
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    task = str(job["task"])
    seed = int(job["seed"])
    kind = str(job["kind"])
    if kind == "B0":
        model = ExplicitChunkPolicy(
            STATE_DIM, PROCESS_DIM, use_process=False, n_tasks=len(TASK_NAMES), task_emb=TASK_EMB
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        def chunk_fn(frame, t, ep_seed):
            return predict_chunk(model, norm, frame, task, device, False)

    else:
        den = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
        den.load_state_dict(ckpt["state_dict"])
        ptype = str(ckpt.get("prediction_type", "epsilon"))
        pol = DiffusionChunkPolicy(den, CosineSchedule(TRAIN_STEPS, device=device), device, prediction_type=ptype).eval()

        def chunk_fn(frame, t, ep_seed):
            st = pack_state(frame)[None]
            sn = norm.n_state(st)[0]
            yn = pol.predict_chunk(sn, TASK_TO_ID[task], policy_seed(TASK_TO_ID[task], ep_seed, t))
            return norm.denorm_act(yn[None])[0]

    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    env, used, err = _setup_env(repo, task, seed, 8, args)
    if env is None:
        return {"task": task, "seed": seed, "success": False, "safety_violation": True, "err": str(err), "kind": kind}
    try:
        wrap_topp_large_jump(env)
        lim = int(getattr(env, "step_lim", None) or TASK_STEP_LIM[task])
        rec = rollout_episode(
            env,
            task,
            model=None,
            normalizer=norm,
            device=device,
            use_process=False,
            max_ticks=min(lim, 40 if job["smoke"] else lim),
            chunk_fn=chunk_fn,
            episode_seed=seed,
        )
        rec["seed"] = seed
        rec["used_seed"] = int(used)
        rec["kind"] = kind
        return rec
    except Exception as exc:
        return {"task": task, "seed": seed, "success": False, "safety_violation": True, "err": repr(exc), "kind": kind}
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def _job_key(r: dict[str, Any]) -> tuple[str, str, int]:
    return str(r["kind"]), str(r["task"]), int(r["seed"])


def _load_kind_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    blob = json.loads(path.read_text())
    if isinstance(blob, list):
        return blob
    return list(blob.get("rows") or [])


def stage_b_verdict(s0: dict[str, Any], s1: dict[str, Any]) -> tuple[str, str | None, float, float]:
    delta = float(s1["pooled"]) - float(s0["pooled"])
    dsafety = float(s1["safety"]) - float(s0["safety"])
    if s1["pooled"] >= 0.20 and delta >= 0.15 and dsafety <= 0.02 and int(s1["n_tasks_gt0"]) >= 2:
        return "diffusion_chunk_breadth_supported", "B1", delta, dsafety
    return "diffusion_chunk_breadth_insufficient", None, delta, dsafety


def _run_jobs(jobs: list[dict[str, Any]], workers: int, commit=None) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def _take(rec: dict[str, Any]) -> None:
        recs.append(rec)
        if commit is not None:
            commit(rec)

    if workers <= 1:
        for j in jobs:
            _take(_x1_worker(j))
        recs.sort(key=lambda r: (r.get("kind", ""), r.get("task", ""), int(r.get("seed", 0))))
        return recs
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        pending = {pool.submit(_x1_worker, j): j for j in jobs}
        while pending:
            done, _ = wait(list(pending), timeout=STAGE_B_JOB_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done:
                for fut, j in list(pending.items()):
                    fut.cancel()
                    _take(
                        {
                            "kind": j["kind"],
                            "task": j["task"],
                            "seed": int(j["seed"]),
                            "success": False,
                            "safety_violation": True,
                            "n_plan": 0,
                            "ticks": 0,
                            "err": "bakeoff_worker_timeout",
                        }
                    )
                break
            for fut in done:
                j = pending.pop(fut)
                try:
                    rec = fut.result(timeout=1)
                except Exception as exc:
                    rec = {
                        "kind": j["kind"],
                        "task": j["task"],
                        "seed": int(j["seed"]),
                        "success": False,
                        "safety_violation": True,
                        "err": repr(exc),
                    }
                _take(rec)
    recs.sort(key=lambda r: (r.get("kind", ""), r.get("task", ""), int(r.get("seed", 0))))
    return recs


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pooled": 0.0, "safety": 1.0, "per_task": {}, "n_tasks_gt0": 0}
    pooled = float(np.mean([1.0 if r.get("success") else 0.0 for r in rows]))
    safety = float(np.mean([1.0 if r.get("safety_violation") else 0.0 for r in rows]))
    per = {}
    for t in MR0_TASKS:
        sub = [r for r in rows if r.get("task") == t]
        per[t] = float(np.mean([1.0 if r.get("success") else 0.0 for r in sub])) if sub else 0.0
    n_gt0 = int(sum(1 for t in MR0_TASKS if per[t] > 0.0))
    return {"pooled": pooled, "safety": safety, "per_task": per, "n_tasks_gt0": n_gt0}


def run_rtwx_task_x1_stage_b(output: str | Path, config: RTWXTASKX1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXTASKX1Config(output=str(output), stage_b_only=True)
    _refuse_p0_write()
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    p0 = _resolve_apr(P0_FROZEN)
    b1_ckpt = _resolve_apr(cfg.b1_ckpt)
    ck1 = torch.load(str(b1_ckpt), map_location="cpu", weights_only=False)
    ptype = str(ck1.get("prediction_type", ""))
    if ptype != "sample":
        raise RuntimeError(f"Stage B frozen ckpt must be I1-B2 sample/x0, got prediction_type={ptype!r} path={b1_ckpt}")
    if bool(ck1.get("use_process", True)):
        raise RuntimeError("Stage B forbids process")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_ep = 2 if cfg.smoke else 16
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    ckpts = {"B0": str(ac0 / "B0" / "best.pt"), "B1": str(b1_ckpt)}
    planned = []
    for kind in ("B0", "B1"):
        for task in MR0_TASKS:
            for j in range(n_ep):
                planned.append((kind, task, BAKEOFF_SEED0 + j))
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for kind in ("B0", "B1"):
        for r in _load_kind_rows(root / kind / "rollout.json"):
            by_key[_job_key(r)] = r
    pending_keys = [k for k in planned if k not in by_key]
    jobs = [
        {
            "kind": kind,
            "task": task,
            "seed": seed,
            "ckpt_path": ckpts[kind],
            "device": device,
            "robotwin_repo": cfg.robotwin_repo,
            "smoke": cfg.smoke,
        }
        for kind, task, seed in pending_keys
    ]
    print(
        f"[rtwx-task-x1] Stage B pending={len(jobs)}/{len(planned)} workers={workers} "
        f"b1={b1_ckpt} ptype={ptype} device={device}",
        flush=True,
    )

    def _commit(rec: dict[str, Any]) -> None:
        by_key[_job_key(rec)] = rec
        for kind in ("B0", "B1"):
            rows_k = [by_key[k] for k in planned if k[0] == kind and k in by_key]
            (root / kind).mkdir(parents=True, exist_ok=True)
            _write_json(
                root / kind / "rollout.json",
                {"rows": rows_k, "summary": _summarize(rows_k), "partial": len(rows_k) < n_ep * len(MR0_TASKS)},
            )
        print(
            f"[rtwx-task-x1] {rec.get('kind')} {rec.get('task')} seed={rec.get('seed')} "
            f"success={rec.get('success')} safety={rec.get('safety_violation')} "
            f"ticks={rec.get('ticks')} done={len(by_key)}/{len(planned)}",
            flush=True,
        )

    if jobs:
        _run_jobs(jobs, workers, commit=_commit)
    b0_rows = [by_key[k] for k in planned if k[0] == "B0" and k in by_key]
    b1_rows = [by_key[k] for k in planned if k[0] == "B1" and k in by_key]
    s0, s1 = _summarize(b0_rows), _summarize(b1_rows)
    pattern, winner, delta, dsafety = stage_b_verdict(s0, s1)
    i1_sa = json.loads((_resolve_apr("runs/rtwx_task_x1_i1") / "B2" / "stage_a.json").read_text())
    summary = {
        "pattern": pattern,
        "scientific_result": False,
        "purpose": "model_family_selection",
        "winner": winner,
        "stage": "B",
        "instrument": "i1_b2_sample_x0",
        "b1_ckpt": str(b1_ckpt),
        "prediction_type": ptype,
        "stage_a_from_i1_b2": i1_sa,
        "B0": s0,
        "B1": s1,
        "delta_pooled": delta,
        "delta_safety": dsafety,
        "n_param_b1": int(ck1.get("n_param", 0)),
        "n_ep": n_ep,
        "seed0": BAKEOFF_SEED0,
        "ka": 1,
        "unlocks_c0": winner == "B1",
        "unlocks_mr0_p0": False,
        "wrote_mr0_p0": False,
        "p0_path_untouched": str(p0),
        "hypothesis": "generative_chunk_maybe_better_than_point_regression",
        "answers": "action_model_family_closed_loop_utility",
        "not_claimed": [
            "actions_are_multimodal",
            "multimodal_diffusion_beats_regression",
            "why_b2_would_succeed",
        ],
        "stop_diffusion_local_dfs": pattern == "diffusion_chunk_breadth_insufficient",
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH, "stage": "B"})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-task-x1] Stage B winner={winner} pattern={pattern} S1={s1['pooled']:.3f} S0={s0['pooled']:.3f}", flush=True)
    return summary


def run_rtwx_task_x1(output: str | Path, config: RTWXTASKX1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXTASKX1Config(output=str(output))
    if cfg.stage_b_only:
        return run_rtwx_task_x1_stage_b(output, cfg)
    _refuse_p0_write()
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    p0 = _resolve_apr(P0_FROZEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac0_b0 = torch.load(str(ac0 / "B0" / "best.pt"), map_location="cpu", weights_only=False)
    norm = ChunkNormalizer.from_dict(ac0_b0["normalizer"])
    train_eps, val_eps, _test_eps = _load_ac0_episodes(ac0)
    print(f"[rtwx-task-x1] device={device} reuse AC0 split/normalizer", flush=True)

    b1_dir = root / "B1"
    if (b1_dir / "best.pt").is_file() and (b1_dir / "metrics.json").is_file():
        print("[rtwx-task-x1] resume B1 ckpt", flush=True)
        packed = json.loads((b1_dir / "metrics.json").read_text())
        den = DiffusionChunkDenoiser(STATE_DIM, num_tasks=len(TASK_NAMES)).to(device)
        ck = torch.load(str(b1_dir / "best.pt"), map_location=device, weights_only=False)
        den.load_state_dict(ck["state_dict"])
        packed.update(
            {"model": den, "sched": CosineSchedule(TRAIN_STEPS, device=device), "ckpt_path": str(b1_dir / "best.pt")}
        )
    else:
        packed = train_diffusion(train_eps, val_eps, norm, device, b1_dir, cfg.smoke)

    policy = DiffusionChunkPolicy(packed["model"], packed["sched"], device).eval()
    va_ld = DataLoader(ActionChunkDataset(val_eps), batch_size=64, shuffle=False)
    pred, tgt = denoise_val_chunks(policy, va_ld, norm, device)
    dummy = np.asarray(norm.n_state(ActionChunkDataset(val_eps)[0]["state"].numpy()[None])[0], dtype=np.float64)
    gates = stage_a_gates(pred, tgt, policy, dummy)
    _write_json(root / "stage_a.json", gates)
    print(f"[rtwx-task-x1] Stage A pass={gates['pass']}", flush=True)
    if not gates["pass"]:
        summary = {
            "pattern": "diffusion_chunk_instrument_failure",
            "scientific_result": False,
            "purpose": "model_family_selection",
            "stage_a": gates,
            "wrote_mr0_p0": False,
            "p0_path_untouched": str(p0),
        }
        _write_json(root / "summary.json", summary)
        return summary

    n_ep = 2 if cfg.smoke else 16
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    jobs = []
    for kind, ckpt in (("B0", str(ac0 / "B0" / "best.pt")), ("B1", packed["ckpt_path"])):
        for task in MR0_TASKS:
            for j in range(n_ep):
                jobs.append(
                    {
                        "kind": kind,
                        "task": task,
                        "seed": BAKEOFF_SEED0 + j,
                        "ckpt_path": ckpt,
                        "device": device,
                        "robotwin_repo": cfg.robotwin_repo,
                        "smoke": cfg.smoke,
                    }
                )
    rows = _run_jobs(jobs, workers)
    b0_rows = [r for r in rows if r.get("kind") == "B0"]
    b1_rows = [r for r in rows if r.get("kind") == "B1"]
    s0, s1 = _summarize(b0_rows), _summarize(b1_rows)
    (root / "B0").mkdir(parents=True, exist_ok=True)
    _write_json(root / "B0" / "rollout.json", {"rows": b0_rows, "summary": s0})
    _write_json(root / "B1" / "rollout.json", {"rows": b1_rows, "summary": s1})
    pattern, winner, delta, dsafety = stage_b_verdict(s0, s1)
    summary = {
        "pattern": pattern,
        "scientific_result": False,
        "purpose": "model_family_selection",
        "winner": winner,
        "stage_a": gates,
        "B0": s0,
        "B1": s1,
        "delta_pooled": delta,
        "delta_safety": dsafety,
        "n_param_b1": packed["n_param"],
        "unlocks_c0": winner == "B1",
        "wrote_mr0_p0": False,
        "p0_path_untouched": str(p0),
        "hypothesis": "generative_chunk_maybe_better_than_point_regression",
        "not_claimed": "actions_are_multimodal",
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-task-x1] winner={winner} pattern={pattern} S1={s1['pooled']:.3f} S0={s0['pooled']:.3f}", flush=True)
    return summary

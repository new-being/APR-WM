"""RTWX-AC0: explicit-state action-chunk provisioning. B0 vs B1, no RGB, no diffusion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .control.action_chunk_spec import ActionChunkSpec
from .evaluation.ac0_offline_metrics import eval_offline
from .evaluation.ac0_rollout_eval import oracle_frame, rollout_episode, wrap_topp_large_jump
from .policy.chunk_dataset import ActionChunkDataset, split_episode_indices
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.explicit_chunk_mlp import ExplicitChunkPolicy
from .policy.process_features import PROCESS_SCHEMA
from .policy.task_embedding import STATE_DIM, STATE_SCHEMA, TASK_NAMES, TASK_SCHEMA
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json
from .task_x0 import _discover_hdf5, _install_mplib_fallback, _load_hdf5_qpos, _load_task_args, _setup_env

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0AC0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ac0.explicit_chunk.v1"
CKPT_SCHEMA = "aprwm.ac0.chunk_mlp.v1"
SEED = 50600
BAKEOFF_SEED0 = 50601
SPLIT_N = (40, 5, 5)
HORIZON = 8
ACTION_DIM = 14
HIDDEN = 256
PROCESS_DIM = 5
TASK_EMB = 16
TIE_PP = 0.05
TASK_STEP_LIM = {
    "place_empty_cup": 500,
    "put_object_cabinet": 700,
    "stamp_seal": 400,
}
AC0_TRAIN = {
    "horizon": HORIZON,
    "action_dim": ACTION_DIM,
    "hidden_dim": HIDDEN,
    "hidden_layers": 3,
    "batch_size": 512,
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "early_stop_patience": 10,
    "loss": "L1",
}
P0_FROZEN_REL = "runs/rtwx_mr0_p0/frozen_chunk_policy.pt"


@dataclass(frozen=True)
class RTWXAC0Config:
    output: str = "runs/rtwx_ac0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    smoke: bool = False
    workers: int = 8


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-AC0 must not write there")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_spec() -> ActionChunkSpec:
    return ActionChunkSpec(
        horizon=HORIZON,
        action_dim=ACTION_DIM,
        semantics="joint_position",
        policy_dt_env_ticks=1,
        deterministic=True,
    )


def config_hash() -> str:
    payload = json.dumps(
        {"train": AC0_TRAIN, "state": STATE_SCHEMA, "process": PROCESS_SCHEMA, "task": TASK_SCHEMA},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _n_split(smoke: bool) -> tuple[int, int, int, int]:
    if smoke:
        return 4, 2, 1, 1
    return 50, SPLIT_N[0], SPLIT_N[1], SPLIT_N[2]


def collect_episode(env: Any, task: str, qa: np.ndarray) -> dict[str, Any]:
    frames: list[dict[str, np.ndarray]] = []
    prev_q = None
    for t in range(int(qa.shape[0])):
        fr = oracle_frame(env, task, prev_q)
        prev_q = fr["q"].copy()
        frames.append(fr)
        env.take_action(np.asarray(qa[t], dtype=np.float64), action_type="qpos")
    a = np.stack([np.asarray(qa[t], dtype=np.float64).reshape(14) for t in range(len(frames))])
    q = np.stack([f["q"] for f in frames])
    return {"task": task, "q": q, "a": a, "frames": frames}


def load_cached_episode(path: Path, task: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    frames = list(z["frames"])
    return {"task": task, "q": np.asarray(z["q"]), "a": np.asarray(z["a"]), "frames": frames}


def save_cached_episode(path: Path, ep: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q=ep["q"], a=ep["a"], frames=np.array(ep["frames"], dtype=object))


def collect_all(cfg: RTWXAC0Config, root: Path, n_demo: int) -> list[dict[str, Any]]:
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    cache_root = root / "cache"
    episodes: list[dict[str, Any]] = []
    for task in MR0_TASKS:
        hdf5s = _discover_hdf5(repo, task)[:n_demo]
        args = _load_task_args(repo, task, "demo_clean")
        args["need_plan"] = False
        args["render_freq"] = 0
        seeds_path = repo / "data" / "demo_clean" / task / "aloha_agilex" / "seed.txt"
        seed_list = [int(x) for x in seeds_path.read_text().split()] if seeds_path.is_file() else list(range(n_demo))
        for i, path in enumerate(hdf5s):
            cpath = cache_root / task / f"ep_{i:03d}.npz"
            if cpath.is_file():
                print(f"[rtwx-ac0] cache {task} {i}", flush=True)
                episodes.append(load_cached_episode(cpath, task))
                continue
            qpack = _load_hdf5_qpos(path)
            if qpack is None:
                print(f"[rtwx-ac0] skip unreadable {path}", flush=True)
                continue
            seed = seed_list[i] if i < len(seed_list) else i
            print(f"[rtwx-ac0] collect {task} ep={i} seed={seed} T={qpack['qa'].shape[0]}", flush=True)
            env, used, err = _setup_env(repo, task, int(seed), 8, args)
            if env is None:
                print(f"[rtwx-ac0] setup fail {err}", flush=True)
                continue
            try:
                qa = qpack["qa"]
                if cfg.smoke:
                    qa = qa[:24]
                ep = collect_episode(env, task, qa)
                ep["seed"] = int(used)
                save_cached_episode(cpath, ep)
                episodes.append(ep)
            finally:
                try:
                    env.close_env()
                except Exception:
                    pass
    return episodes


def _stack_split(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.stack([s["state"] for s in samples])
    pr = np.stack([s["process"] for s in samples])
    ch = np.stack([s["chunk"] for s in samples])
    return st, pr, ch


def train_branch(
    branch: str,
    train_eps: list[dict[str, Any]],
    val_eps: list[dict[str, Any]],
    *,
    device: str,
    out_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    use_p = branch == "B1"
    tr_ds = ActionChunkDataset(train_eps, horizon=HORIZON)
    va_ds = ActionChunkDataset(val_eps, horizon=HORIZON)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError(f"{branch} empty windows")
    st, pr, ch = _stack_split(tr_ds.samples)
    norm = ChunkNormalizer.fit(st, ch, pr)
    model = ExplicitChunkPolicy(
        STATE_DIM, PROCESS_DIM, use_process=use_p, n_tasks=len(TASK_NAMES), task_emb=TASK_EMB
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=AC0_TRAIN["lr"], weight_decay=AC0_TRAIN["weight_decay"])
    bs = min(int(AC0_TRAIN["batch_size"]), len(tr_ds))
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=min(256, len(va_ds)), shuffle=False)
    max_ep = 8 if smoke else int(AC0_TRAIN["max_epochs"])
    patience = 3 if smoke else int(AC0_TRAIN["early_stop_patience"])
    best_l1 = 1e9
    best_state = None
    bad = 0
    curve = []
    n_param = int(sum(p.numel() for p in model.parameters()))
    for epoch in range(max_ep):
        model.train()
        tot = 0.0
        n = 0
        for batch in tr_ld:
            s = torch.from_numpy(norm.n_state(batch["state"].numpy()).astype(np.float32)).to(device)
            p = torch.from_numpy(norm.n_proc(batch["process"].numpy()).astype(np.float32)).to(device)
            y = torch.from_numpy(norm.n_act(batch["chunk"].numpy()).astype(np.float32)).to(device)
            tid = batch["task_id"].to(device)
            pred = model(s, tid, p)
            loss = (pred - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * s.shape[0]
            n += int(s.shape[0])
        off = eval_offline(model, va_ld, norm, device, use_p)
        curve.append({"epoch": epoch, "train_l1_norm": tot / max(n, 1), "val_l1": off["l1"]})
        print(f"[rtwx-ac0] {branch} ep={epoch} val_L1={off['l1']:.4f} e0={off['e0']:.4f} e7={off['e7']:.4f}", flush=True)
        if off["l1"] + 1e-6 < best_l1:
            best_l1 = off["l1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    off_va = eval_offline(model, va_ld, norm, device, use_p)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "schema": CKPT_SCHEMA,
        "branch": branch,
        "state_dict": model.state_dict(),
        "normalizer": norm.as_dict(),
        "use_process": use_p,
        "state_dim": STATE_DIM,
        "process_dim": PROCESS_DIM,
        "horizon": HORIZON,
        "action_dim": ACTION_DIM,
        "state_schema": STATE_SCHEMA,
        "process_schema": PROCESS_SCHEMA,
        "task_schema": TASK_SCHEMA,
        "config_hash": config_hash(),
        "spec": frozen_spec().as_dict(),
        "n_param": n_param,
    }
    pt = out_dir / "best.pt"
    torch.save(ckpt, pt)
    metrics = {
        "branch": branch,
        "n_param": n_param,
        "val": off_va,
        "curve": curve,
        "n_train_win": len(tr_ds),
        "n_val_win": len(va_ds),
    }
    _write_json(out_dir / "metrics.json", metrics)
    return {"model": model, "norm": norm, "metrics": metrics, "ckpt_path": str(pt), "n_param": n_param}


def bakeoff_job_list(n_ep: int) -> list[tuple[str, int]]:
    return [(task, BAKEOFF_SEED0 + j) for task in MR0_TASKS for j in range(n_ep)]


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(r["task"]), int(r["seed"])): r for r in rows}


def _ordered_rows(n_ep: int, by_key: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_key[k] for k in bakeoff_job_list(n_ep) if k in by_key]


def load_bakeoff_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    blob = json.loads(path.read_text())
    if isinstance(blob, list):
        return blob
    return list(blob.get("rows") or [])


def _bakeoff_worker(job: dict[str, Any]) -> dict[str, Any]:
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
    model = ExplicitChunkPolicy(
        STATE_DIM,
        PROCESS_DIM,
        use_process=bool(job["use_process"]),
        n_tasks=len(TASK_NAMES),
        task_emb=TASK_EMB,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    task = str(job["task"])
    seed = int(job["seed"])
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    env, used, err = _setup_env(repo, task, seed, 8, args)
    if env is None:
        return {
            "task": task,
            "success": False,
            "safety_violation": True,
            "n_plan": 0,
            "ticks": 0,
            "seed": seed,
            "err": str(err),
        }
    try:
        wrap_topp_large_jump(env)
        lim = int(getattr(env, "step_lim", None) or TASK_STEP_LIM[task])
        rec = rollout_episode(
            env,
            task,
            model=model,
            normalizer=norm,
            device=device,
            use_process=bool(job["use_process"]),
            max_ticks=min(lim, 80 if job["smoke"] else lim),
        )
        rec["seed"] = int(used)
        return rec
    except Exception as exc:
        return {
            "task": task,
            "success": False,
            "safety_violation": True,
            "n_plan": 0,
            "ticks": 0,
            "seed": seed,
            "err": repr(exc),
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def bakeoff_branch(
    ckpt_path: str,
    use_process: bool,
    cfg: RTWXAC0Config,
    n_ep: int,
    device: str,
    out_json: Path,
) -> list[dict[str, Any]]:
    jobs = bakeoff_job_list(n_ep)
    by_key = _rows_by_key(load_bakeoff_rows(out_json))
    pending = [k for k in jobs if k not in by_key]
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    print(
        f"[rtwx-ac0] bakeoff pending={len(pending)}/{len(jobs)} workers={workers} "
        f"use_process={use_process}",
        flush=True,
    )
    payloads = [
        {
            "task": task,
            "seed": seed,
            "ckpt_path": ckpt_path,
            "use_process": use_process,
            "device": device,
            "robotwin_repo": cfg.robotwin_repo,
            "smoke": cfg.smoke,
        }
        for task, seed in pending
    ]

    def _commit(rec: dict[str, Any]) -> None:
        by_key[(str(rec["task"]), int(rec["seed"]))] = rec
        rows = _ordered_rows(n_ep, by_key)
        _write_json(out_json, {"rows": rows, "summary": summarize_rows(rows), "partial": len(rows) < len(jobs)})
        print(
            f"[rtwx-ac0] bakeoff {rec['task']} seed={rec['seed']} success={rec.get('success')} "
            f"safety={rec.get('safety_violation')} ticks={rec.get('ticks')} done={len(rows)}/{len(jobs)}",
            flush=True,
        )

    if not payloads:
        return _ordered_rows(n_ep, by_key)
    if workers == 1:
        for payload in payloads:
            _commit(_bakeoff_worker(payload))
        return _ordered_rows(n_ep, by_key)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    from concurrent.futures import wait, FIRST_COMPLETED

    job_timeout_s = 720.0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_bakeoff_worker, payload): payload for payload in payloads}
        pending_f = dict(futs)
        while pending_f:
            done, _ = wait(list(pending_f), timeout=job_timeout_s, return_when=FIRST_COMPLETED)
            if not done:
                for fut, payload in list(pending_f.items()):
                    fut.cancel()
                    _commit(
                        {
                            "task": payload["task"],
                            "seed": int(payload["seed"]),
                            "success": False,
                            "safety_violation": True,
                            "n_plan": 0,
                            "ticks": 0,
                            "err": "bakeoff_worker_timeout",
                        }
                    )
                break
            for fut in done:
                payload = pending_f.pop(fut)
                try:
                    rec = fut.result(timeout=1)
                except Exception as exc:
                    rec = {
                        "task": payload["task"],
                        "seed": int(payload["seed"]),
                        "success": False,
                        "safety_violation": True,
                        "n_plan": 0,
                        "ticks": 0,
                        "err": repr(exc),
                    }
                rec["task"] = payload["task"]
                rec["seed"] = int(payload["seed"])
                _commit(rec)
    return _ordered_rows(n_ep, by_key)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pooled": 0.0, "min_task": 0.0, "safety": 1.0}
    pooled = float(np.mean([1.0 if r["success"] else 0.0 for r in rows]))
    safety = float(np.mean([1.0 if r["safety_violation"] else 0.0 for r in rows]))
    mins = []
    for task in MR0_TASKS:
        sub = [r for r in rows if r["task"] == task]
        mins.append(float(np.mean([1.0 if r["success"] else 0.0 for r in sub])) if sub else 0.0)
    return {"pooled": pooled, "min_task": float(min(mins)), "per_task": {t: m for t, m in zip(MR0_TASKS, mins)}, "safety": safety}


def select_winner(b0: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    def key(b: dict[str, Any]) -> tuple:
        return (b["roll"]["pooled"], b["roll"]["min_task"], -b["roll"]["safety"], -b["metrics"]["val"]["l1"])

    top = "B1" if key(b1) > key(b0) else "B0"
    reason = "lexicographic"
    if top == "B1":
        dp = b1["roll"]["pooled"] - b0["roll"]["pooled"]
        if dp < TIE_PP and b1["roll"]["min_task"] <= b0["roll"]["min_task"]:
            top = "B0"
            reason = "complexity_tie_margin"
    return {"winner": top, "reason": reason, "tie_pp": TIE_PP}


def interpret_pattern(b0: dict[str, Any], b1: dict[str, Any], winner: str) -> str:
    l1 = min(b0["metrics"]["val"]["l1"], b1["metrics"]["val"]["l1"])
    e7 = min(b0["metrics"]["val"]["e7"], b1["metrics"]["val"]["e7"])
    pool = max(b0["roll"]["pooled"], b1["roll"]["pooled"])
    if l1 > 0.20 and e7 > 0.25:
        return "deterministic_chunk_insufficient"
    if l1 < 0.08 and pool < 0.15:
        return "covariate_shift_suspected"
    if winner == "B1":
        return "progress_bias_supported"
    if b0["roll"]["pooled"] >= 0.25:
        return "explicit_state_chunk_sufficient"
    return "explicit_chunk_mapping_weak"


def run_rtwx_ac0(output: str | Path, config: RTWXAC0Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXAC0Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    n_demo, n_tr, n_va, n_te = _n_split(cfg.smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rtwx-ac0] device={device} n_demo={n_demo} split={n_tr}/{n_va}/{n_te}", flush=True)
    episodes = collect_all(cfg, root, n_demo)
    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in MR0_TASKS}
    for ep in episodes:
        by_task[ep["task"]].append(ep)
    rng = np.random.default_rng(cfg.seed)
    manifest: dict[str, Any] = {
        "state_source": "oracle_structured",
        "scientific_scope": "action_representation_only",
        "split_unit": "episode",
    }
    train_eps: list[dict[str, Any]] = []
    val_eps: list[dict[str, Any]] = []
    test_eps: list[dict[str, Any]] = []
    for task in MR0_TASKS:
        eps = by_task[task]
        n = min(len(eps), n_demo)
        if n < n_tr + n_va + n_te:
            raise RuntimeError(f"{task} has {n} episodes, need {n_tr + n_va + n_te}")
        sp = split_episode_indices(n_tr + n_va + n_te, rng, n_tr, n_va, n_te)
        manifest[task] = sp
        train_eps.extend(eps[i] for i in sp["train"])
        val_eps.extend(eps[i] for i in sp["val"])
        test_eps.extend(eps[i] for i in sp["test"])
    _write_json(root / "split_manifest.json", manifest)

    packed = {}
    for br in ("B0", "B1"):
        pt = root / br / "best.pt"
        met = root / br / "metrics.json"
        if pt.is_file() and met.is_file():
            print(f"[rtwx-ac0] resume {br} from {pt}", flush=True)
            ckpt = torch.load(str(pt), map_location=device, weights_only=False)
            model = ExplicitChunkPolicy(
                STATE_DIM, PROCESS_DIM, use_process=br == "B1", n_tasks=len(TASK_NAMES), task_emb=TASK_EMB
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            packed[br] = {
                "model": model,
                "norm": ChunkNormalizer.from_dict(ckpt["normalizer"]),
                "metrics": json.loads(met.read_text()),
                "ckpt_path": str(pt),
                "n_param": int(ckpt.get("n_param", 0)),
            }
        else:
            packed[br] = train_branch(br, train_eps, val_eps, device=device, out_dir=root / br, smoke=cfg.smoke)

    te_ds = ActionChunkDataset(test_eps, horizon=HORIZON)
    te_ld = DataLoader(te_ds, batch_size=min(256, max(len(te_ds), 1)), shuffle=False) if len(te_ds) else None
    for br in ("B0", "B1"):
        if te_ld is not None:
            packed[br]["metrics"]["test"] = eval_offline(
                packed[br]["model"], te_ld, packed[br]["norm"], device, br == "B1"
            )

    n_bake = 2 if cfg.smoke else 16
    for br in ("B0", "B1"):
        print(f"[rtwx-ac0] bakeoff {br} N={n_bake}/task", flush=True)
        rows = bakeoff_branch(
            packed[br]["ckpt_path"],
            br == "B1",
            cfg,
            n_bake,
            device,
            root / br / "rollout.json",
        )
        packed[br]["roll_rows"] = rows
        packed[br]["roll"] = summarize_rows(rows)

    sel = select_winner(packed["B0"], packed["B1"])
    pattern = interpret_pattern(packed["B0"], packed["B1"], sel["winner"])
    w = sel["winner"]
    wpt = Path(packed[w]["ckpt_path"])
    selection = {
        "winner": w,
        "reason": sel["reason"],
        "pattern": pattern,
        "checkpoint": str(wpt),
        "checkpoint_sha256": _file_sha256(wpt),
        "config_hash": config_hash(),
        "normalization_hash": packed[w]["norm"].hash(),
        "task_schema_hash": hashlib.sha256(TASK_SCHEMA.encode()).hexdigest(),
        "process_schema_hash": hashlib.sha256(PROCESS_SCHEMA.encode()).hexdigest(),
        "B0": {"val": packed["B0"]["metrics"]["val"], "roll": packed["B0"]["roll"], "n_param": packed["B0"]["n_param"]},
        "B1": {"val": packed["B1"]["metrics"]["val"], "roll": packed["B1"]["roll"], "n_param": packed["B1"]["n_param"]},
        "smoke": cfg.smoke,
    }
    _write_json(root / "selection.json", selection)
    copied = False
    if not cfg.smoke:
        dest = _resolve_apr(P0_FROZEN_REL)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wpt, dest)
        copied = True
        print(f"[rtwx-ac0] froze winner → {dest}", flush=True)
    summary = {
        "pattern": pattern,
        "scientific_result": True,
        "winner": w,
        "selection": selection,
        "copied_to_p0": copied,
        "state_source": "oracle_structured",
        "scientific_scope": "action_representation_only",
        "n_demo": n_demo,
        "n_bakeoff": n_bake,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-ac0] winner={w} pattern={pattern}", flush=True)
    return summary

"""RTWX-AC2: policy-state sufficiency. Ha=1. P0 expert replay then B0/B1/B2 MLP."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.history_windows import HS, PAST_A, left_pad_past_actions, left_pad_states
from .data.single_step_dataset import HistoryWindowDataset
from .evaluation.ac0_rollout_eval import wrap_topp_large_jump
from .evaluation.ac2_offline import eval_l1, stage_a_gates
from .evaluation.ac2_rollout import rollout_ac2
from .evaluation.expert_replay import replay_episode
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.single_step_mlp import ACTION_DIM, TASK_EMB, SingleStepMLP
from .policy.task_embedding import STATE_DIM, TASK_NAMES, pack_state
from .rtwx_ac0 import TASK_STEP_LIM, load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json
from .task_x0 import _install_mplib_fallback, _load_task_args, _setup_env

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0AC2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ac2.policy_state.v1"
CKPT_SCHEMA = "aprwm.ac2.single_step_mlp.v1"
BAKEOFF_SEED0 = 53601
P0_N = 8
P0_MIN = 0.875
TIE_PP = 0.05
UTIL_PP = 0.10
JOB_TIMEOUT_S = 900.0
AC2_TRAIN = {
    "H_a": 1,
    "H_s": HS,
    "hidden_dim": 256,
    "hidden_layers": 3,
    "batch_size": 512,
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "loss": "L1",
    "early_stop": False,
}


@dataclass(frozen=True)
class RTWXAC2Config:
    output: str = "runs/rtwx_ac2"
    ac0_run: str = "runs/rtwx_ac0"
    robotwin_repo: str = "/root/RoboTwin"
    smoke: bool = False
    workers: int = 8


def _load_split_eps(ac0: Path) -> tuple[list, list, list]:
    man = json.loads((ac0 / "split_manifest.json").read_text())
    train, val, test = [], [], []
    for task in MR0_TASKS:
        for split, bucket in (("train", train), ("val", val), ("test", test)):
            for i in man[task][split]:
                bucket.append(load_cached_episode(ac0 / "cache" / task / f"ep_{int(i):03d}.npz", task))
    return train, val, test


def heldout_indices(man: dict, task: str, n: int) -> list[int]:
    ids = [int(i) for i in list(man[task]["val"]) + list(man[task]["test"])]
    return ids[:n]


def _p0_worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = Path(job["robotwin_repo"])
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    return replay_episode(repo, job["task"], int(job["ep_index"]), bool(job["smoke"]))


def _run_mp(jobs: list[dict[str, Any]], workers: int, fn, commit=None) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def _take(rec):
        recs.append(rec)
        if commit is not None:
            commit(rec)

    if workers <= 1:
        for j in jobs:
            _take(fn(j))
        return recs
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        pending = {pool.submit(fn, j): j for j in jobs}
        while pending:
            done, _ = wait(list(pending), timeout=JOB_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done:
                for fut, j in list(pending.items()):
                    fut.cancel()
                    _take({**j, "success": False, "safety_violation": True, "err": "timeout"})
                break
            for fut in done:
                j = pending.pop(fut)
                try:
                    _take(fut.result(timeout=1))
                except Exception as exc:
                    _take({**j, "success": False, "safety_violation": True, "err": repr(exc)})
    return recs


def summarize_rows(rows: list[dict[str, Any]], tasks=MR0_TASKS) -> dict[str, Any]:
    if not rows:
        return {"pooled": 0.0, "safety": 1.0, "per_task": {}, "n_tasks_gt0": 0}
    pooled = float(np.mean([1.0 if r.get("success") else 0.0 for r in rows]))
    safety = float(np.mean([1.0 if r.get("safety_violation") else 0.0 for r in rows]))
    per = {}
    for t in tasks:
        sub = [r for r in rows if r.get("task") == t]
        per[t] = float(np.mean([1.0 if r.get("success") else 0.0 for r in sub])) if sub else 0.0
    n_gt0 = int(sum(1 for t in tasks if per[t] > 0.0))
    return {"pooled": pooled, "safety": safety, "per_task": per, "n_tasks_gt0": n_gt0}


def qualified(s: dict[str, Any]) -> bool:
    return float(s["pooled"]) >= 0.20 and int(s["n_tasks_gt0"]) >= 2 and float(s["safety"]) <= 0.20


def ac2_verdict(s0: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any]) -> dict[str, Any]:
    q0, q1, q2 = qualified(s0), qualified(s1), qualified(s2)
    d10 = float(s1["pooled"]) - float(s0["pooled"])
    d21 = float(s2["pooled"]) - float(s1["pooled"])
    d20 = float(s2["pooled"]) - float(s0["pooled"])
    out = {"d10": d10, "d21": d21, "d20": d20, "q0": q0, "q1": q1, "q2": q2}
    if not (q0 or q1 or q2):
        out.update({"winner": None, "pattern": "short_memory_policy_insufficient"})
        return out
    if d21 >= UTIL_PP and q2:
        out.update({"winner": "B2", "pattern": "action_history_utility_supported"})
        return out
    if d10 >= UTIL_PP and q1:
        out.update({"winner": "B1", "pattern": "state_history_utility_supported"})
        return out
    if q0 and d10 < TIE_PP and d20 < TIE_PP:
        out.update({"winner": "B0", "pattern": "single_step_output_supported"})
        return out
    if TIE_PP <= max(d10, d21, d20) < UTIL_PP:
        cands = [k for k, q in (("B0", q0), ("B1", q1), ("B2", q2)) if q]
        winner = cands[0] if cands else None
        out.update({"winner": winner, "pattern": "inconclusive_small_gain"})
        return out
    scores = [("B0", s0, q0), ("B1", s1, q1), ("B2", s2, q2)]
    good = [(k, s) for k, s, q in scores if q]
    good.sort(key=lambda kv: (-kv[1]["pooled"], {"B0": 0, "B1": 1, "B2": 2}[kv[0]]))
    winner = good[0][0]
    pattern = "single_step_output_supported" if winner == "B0" else (
        "state_history_utility_supported" if winner == "B1" else "action_history_utility_supported"
    )
    out.update({"winner": winner, "pattern": pattern})
    return out


def history_alignment_ok() -> bool:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(5, STATE_DIM))
    actions = rng.normal(size=(5, ACTION_DIM))
    reset_a = rng.normal(size=(ACTION_DIM,))
    st = left_pad_states(states, 3, length=4)
    pa = left_pad_past_actions(actions, 3, reset_a, length=3)
    ok = np.allclose(st, states[0:4]) and np.allclose(pa, actions[0:3])
    st0 = left_pad_states(states, 0, length=4)
    pa0 = left_pad_past_actions(actions, 0, reset_a, length=3)
    ok = ok and np.allclose(st0, np.stack([states[0]] * 4)) and np.allclose(pa0, np.stack([reset_a] * 3))
    return bool(ok)


def train_branch(kind: str, train_eps, val_eps, norm: ChunkNormalizer, device: str, out_dir: Path, smoke: bool) -> dict[str, Any]:
    use_a = kind == "B2"
    tr = HistoryWindowDataset(train_eps, include_action_history=use_a)
    va = HistoryWindowDataset(val_eps, include_action_history=use_a)
    model = SingleStepMLP(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=AC2_TRAIN["lr"], weight_decay=AC2_TRAIN["weight_decay"])
    bs = min(int(AC2_TRAIN["batch_size"]), len(tr))
    tr_ld = DataLoader(tr, batch_size=bs, shuffle=True)
    va_ld = DataLoader(va, batch_size=min(256, len(va)), shuffle=False)
    max_ep = 2 if smoke else int(AC2_TRAIN["max_epochs"])
    best = 1e9
    best_state = None
    curve = []
    n_param = int(sum(p.numel() for p in model.parameters()))
    for epoch in range(max_ep):
        model.train()
        tot, n = 0.0, 0
        for batch in tr_ld:
            st = torch.from_numpy(norm.n_state(batch["states"].numpy()).astype(np.float32)).to(device)
            y = torch.from_numpy(norm.n_act(batch["target_action"].numpy()).astype(np.float32)).to(device)
            tid = batch["task_id"].to(device)
            pa = None
            if kind == "B2":
                pa = torch.from_numpy(norm.n_act(batch["past_actions"].numpy()).astype(np.float32)).to(device)
            pred = model(st, tid, pa)
            loss = (pred - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * st.shape[0]
            n += int(st.shape[0])
        off = eval_l1(model, va_ld, norm, device, kind)
        curve.append({"epoch": epoch, "train_l1": tot / max(n, 1), "val_l1": off["l1"]})
        print(f"[rtwx-ac2] {kind} ep={epoch} val_L1={off['l1']:.4f}", flush=True)
        if off["l1"] + 1e-8 < best:
            best = off["l1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    off = eval_l1(model, va_ld, norm, device, kind)
    dummy_s = torch.from_numpy(norm.n_state(va[0]["states"].numpy()[None]).astype(np.float32)).to(device)
    dummy_t = torch.tensor([int(va[0]["task_id"])], device=device)
    dummy_a = None
    if kind == "B2":
        dummy_a = torch.from_numpy(norm.n_act(va[0]["past_actions"].numpy()[None]).astype(np.float32)).to(device)
    gates = stage_a_gates(off["pred"], off["tgt"], model, (dummy_s, dummy_t, dummy_a))
    gates["G4_history_alignment"] = history_alignment_ok()
    gates["pass"] = bool(gates["pass"] and gates["G4_history_alignment"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "schema": CKPT_SCHEMA,
        "kind": kind,
        "state_dict": model.state_dict(),
        "normalizer": norm.as_dict(),
        "H_a": 1,
        "H_s": HS,
        "n_param": n_param,
        "val_l1": off["l1"],
        "train": AC2_TRAIN,
    }
    torch.save(ckpt, out_dir / "best.pt")
    _write_json(out_dir / "metrics.json", {"n_param": n_param, "val_l1": off["l1"], "curve": curve, "stage_a": gates})
    return {"model": model, "n_param": n_param, "val_l1": off["l1"], "stage_a": gates, "ckpt_path": str(out_dir / "best.pt")}


def _b_worker(job: dict[str, Any]) -> dict[str, Any]:
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
    model = SingleStepMLP(str(job["kind"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    task = str(job["task"])
    seed = int(job["seed"])
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    env, used, err = _setup_env(repo, task, seed, 8, args)
    rec = {"kind": job["kind"], "task": task, "seed": seed}
    if env is None:
        rec.update({"success": False, "safety_violation": True, "err": str(err)})
        return rec
    try:
        wrap_topp_large_jump(env)
        lim = int(getattr(env, "step_lim", None) or TASK_STEP_LIM[task])
        out = rollout_ac2(env, task, model, norm, device, max_ticks=min(lim, 40 if job["smoke"] else lim))
        rec.update(out)
        rec["used_seed"] = int(used)
        rec["seed"] = seed
        rec["kind"] = job["kind"]
        return rec
    except Exception as exc:
        rec.update({"success": False, "safety_violation": True, "err": repr(exc)})
        return rec
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def run_rtwx_ac2(output: str | Path, config: RTWXAC2Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXAC2Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    repo = Path(cfg.robotwin_repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.loads((ac0 / "split_manifest.json").read_text())
    n_rep = 1 if cfg.smoke else P0_N
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    p0_path = root / "P0" / "replay.json"
    p0_path.parent.mkdir(parents=True, exist_ok=True)
    p0_rows = []
    if p0_path.is_file():
        p0_rows = json.loads(p0_path.read_text()).get("rows") or []
    have = {(r["task"], int(r["ep_index"])) for r in p0_rows}
    jobs = []
    for task in MR0_TASKS:
        for idx in heldout_indices(man, task, n_rep):
            if (task, idx) not in have:
                jobs.append({"task": task, "ep_index": idx, "robotwin_repo": str(repo), "smoke": cfg.smoke})
    print(f"[rtwx-ac2] P0 pending={len(jobs)} workers={workers}", flush=True)

    def _commit_p0(rec):
        p0_rows.append(rec)
        summ = summarize_rows(p0_rows)
        _write_json(p0_path, {"rows": p0_rows, "summary": summ})
        print(f"[rtwx-ac2] P0 {rec.get('task')} ep={rec.get('ep_index')} success={rec.get('success')}", flush=True)

    if jobs:
        _run_mp(jobs, workers, _p0_worker, commit=_commit_p0)
    p0_sum = summarize_rows(p0_rows)
    p0_ok = all(float(p0_sum["per_task"].get(t, 0.0)) >= (1.0 if cfg.smoke else P0_MIN) for t in MR0_TASKS)
    p0_blob = {"ok": p0_ok, "summary": p0_sum, "threshold": P0_MIN, "n_per_task": n_rep}
    _write_json(root / "P0" / "summary.json", p0_blob)
    if not p0_ok:
        summary = {
            "pattern": "expert_action_replay_failure",
            "scientific_result": False,
            "purpose": "policy_state_sufficiency",
            "winner": None,
            "P0": p0_blob,
            "training_run": False,
            "unlocks_mr0": False,
        }
        _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
        _write_json(root / "summary.json", summary)
        print("[rtwx-ac2] P0 FAIL stop training", flush=True)
        return summary

    ac0_b0 = torch.load(str(ac0 / "B0" / "best.pt"), map_location="cpu", weights_only=False)
    norm = ChunkNormalizer.from_dict(ac0_b0["normalizer"])
    train_eps, val_eps, _ = _load_split_eps(ac0)
    packed = {}
    for kind in ("B0", "B1", "B2"):
        bdir = root / kind
        if (bdir / "best.pt").is_file() and (bdir / "metrics.json").is_file():
            print(f"[rtwx-ac2] resume {kind}", flush=True)
            met = json.loads((bdir / "metrics.json").read_text())
            model = SingleStepMLP(kind).to(device)
            ck = torch.load(str(bdir / "best.pt"), map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"])
            packed[kind] = {"model": model, "n_param": met.get("n_param"), "val_l1": met.get("val_l1"), "stage_a": met.get("stage_a"), "ckpt_path": str(bdir / "best.pt")}
        else:
            packed[kind] = train_branch(kind, train_eps, val_eps, norm, device, bdir, cfg.smoke)

    if not all(packed[k]["stage_a"]["pass"] for k in packed):
        summary = {
            "pattern": "ac2_instrument_failure",
            "scientific_result": False,
            "purpose": "policy_state_sufficiency",
            "winner": None,
            "P0": p0_blob,
            "stage_a": {k: packed[k]["stage_a"] for k in packed},
            "offline_l1": {k: packed[k]["val_l1"] for k in packed},
            "unlocks_mr0": False,
        }
        _write_json(root / "summary.json", summary)
        return summary

    # optional diagnostic, not a gate
    va = HistoryWindowDataset(val_eps, include_action_history=True)
    n_d = min(256, len(va))
    sts, tid, pas = [], [], []
    for i in range(n_d):
        sts.append(va[i]["states"].numpy())
        tid.append(int(va[i]["task_id"]))
        pas.append(va[i]["past_actions"].numpy())
    st = torch.from_numpy(norm.n_state(np.stack(sts)).astype(np.float32)).to(device)
    tt = torch.tensor(tid, device=device)
    pa = torch.from_numpy(norm.n_act(np.stack(pas)).astype(np.float32)).to(device)
    with torch.no_grad():
        y0 = packed["B0"]["model"](st, tt, None).cpu().numpy()
        y1 = packed["B1"]["model"](st, tt, None).cpu().numpy()
        y2 = packed["B2"]["model"](st, tt, pa).cpu().numpy()
    diag = {
        "delta_a_hist": float(np.abs(y1 - y0).mean()),
        "delta_a_acthist": float(np.abs(y2 - y1).mean()),
        "not_a_gate": True,
    }
    _write_json(root / "diagnostic.json", diag)

    n_ep = 1 if cfg.smoke else 16
    planned = [(k, t, BAKEOFF_SEED0 + j) for k in ("B0", "B1", "B2") for t in MR0_TASKS for j in range(n_ep)]
    by_key = {}
    for k in ("B0", "B1", "B2"):
        rp = root / k / "rollout.json"
        if rp.is_file():
            for r in json.loads(rp.read_text()).get("rows") or []:
                by_key[(r["kind"], r["task"], int(r["seed"]))] = r
    pending = [p for p in planned if p not in by_key]
    bjobs = [
        {
            "kind": k,
            "task": t,
            "seed": s,
            "ckpt_path": packed[k]["ckpt_path"],
            "device": device,
            "robotwin_repo": str(repo),
            "smoke": cfg.smoke,
        }
        for k, t, s in pending
    ]
    print(f"[rtwx-ac2] Stage B pending={len(bjobs)}/{len(planned)}", flush=True)

    def _commit_b(rec):
        by_key[(rec["kind"], rec["task"], int(rec["seed"]))] = rec
        for k in ("B0", "B1", "B2"):
            rows_k = [by_key[p] for p in planned if p[0] == k and p in by_key]
            _write_json(root / k / "rollout.json", {"rows": rows_k, "summary": summarize_rows(rows_k), "partial": len(rows_k) < n_ep * 3})
        print(
            f"[rtwx-ac2] {rec.get('kind')} {rec.get('task')} seed={rec.get('seed')} success={rec.get('success')} "
            f"done={len(by_key)}/{len(planned)}",
            flush=True,
        )

    if bjobs:
        _run_mp(bjobs, workers, _b_worker, commit=_commit_b)
    s0 = summarize_rows([by_key[p] for p in planned if p[0] == "B0" and p in by_key])
    s1 = summarize_rows([by_key[p] for p in planned if p[0] == "B1" and p in by_key])
    s2 = summarize_rows([by_key[p] for p in planned if p[0] == "B2" and p in by_key])
    verd = ac2_verdict(s0, s1, s2)
    summary = {
        "pattern": verd["pattern"],
        "scientific_result": False,
        "purpose": "policy_state_sufficiency",
        "winner": verd["winner"],
        "not_claimed": ["world_is_markov", "memory_architecture"],
        "P0": p0_blob,
        "stage_a": {k: packed[k]["stage_a"] for k in packed},
        "offline_l1": {k: packed[k]["val_l1"] for k in packed},
        "offline_not_winner": True,
        "n_param": {k: packed[k]["n_param"] for k in packed},
        "B0": s0,
        "B1": s1,
        "B2": s2,
        "deltas": {"B1-B0": verd["d10"], "B2-B1": verd["d21"], "B2-B0": verd["d20"]},
        "diagnostic": diag,
        "seed0": BAKEOFF_SEED0,
        "H_a": 1,
        "unlocks_mr0": False,
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-ac2] winner={verd['winner']} pattern={verd['pattern']} S0={s0['pooled']:.3f} S1={s1['pooled']:.3f} S2={s2['pooled']:.3f}", flush=True)
    return summary

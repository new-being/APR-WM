"""RTWX-AC3: demonstration execution contract audit. No policy training."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation.ac3_native import native_replay_once, traj_pkl
from .evaluation.ac3_provenance import audit_episode
from .evaluation.ac3_replay_decomp import replay_dense, replay_sparse_hold
from .evaluation.expert_replay import demo_seed_list
from .rtwx_ac0 import load_cached_episode
from .rtwx_ac2 import P0_MIN, P0_N, heldout_indices, summarize_rows
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json
from .task_x0 import _install_mplib_fallback

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0AC3_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ac3.execution_contract.v1"
JOB_TIMEOUT_S = 1200.0


@dataclass(frozen=True)
class RTWXAC3Config:
    output: str = "runs/rtwx_ac3"
    ac0_run: str = "runs/rtwx_ac0"
    ac2_run: str = "runs/rtwx_ac2"
    robotwin_repo: str = "/root/RoboTwin"
    smoke: bool = False
    workers: int = 8


def _gate(summ: dict[str, Any], smoke: bool) -> bool:
    thr = 1.0 if smoke else P0_MIN
    return all(float(summ["per_task"].get(t, 0.0)) >= thr for t in MR0_TASKS)


def _mp(jobs, workers, fn, commit=None):
    recs = []

    def _take(rec):
        recs.append(rec)
        if commit:
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


def _native_worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = Path(job["robotwin_repo"])
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    return native_replay_once(
        repo,
        job["task"],
        int(job["ep_index"]),
        int(job["seed"]),
        record=bool(job.get("record")),
        trace_path=Path(job["trace_path"]) if job.get("trace_path") else None,
    )


def _r0_worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ["OMP_NUM_THREADS"] = "1"
    repo = Path(job["robotwin_repo"])
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    kind = job["kind"]
    if kind == "B1":
        rec = replay_dense(repo, job["task"], int(job["seed"]), Path(job["trace_path"]), zero_vel=False)
    elif kind == "B2":
        rec = replay_dense(repo, job["task"], int(job["seed"]), Path(job["trace_path"]), zero_vel=True)
    else:
        rec = replay_sparse_hold(repo, job["task"], int(job["seed"]), int(job["ep_index"]), Path(job["trace_path"]))
    rec.update({"kind": kind, "task": job["task"], "ep_index": job["ep_index"]})
    return rec


def r0_pattern(p0_ok: bool, b1: dict, b2: dict, b3: dict, b0_fail: bool) -> str:
    if not p0_ok:
        return "native_expert_replay_failure"
    g1, g2, g3 = _gate(b1, False), _gate(b2, False), _gate(b3, False)
    if g1 and (not g2):
        return "action_state_missing_velocity_target"
    if g2 and g3 and b0_fail:
        return "action_timestep_contract_failure"
    if g1 and g2 and (not g3) and b0_fail:
        return "sparse_action_insufficient"
    return "execution_contract_unresolved"


def run_rtwx_ac3(output: str | Path, config: RTWXAC3Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXAC3Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    ac2 = _resolve_apr(cfg.ac2_run)
    repo = Path(cfg.robotwin_repo)
    man = json.loads((ac0 / "split_manifest.json").read_text())
    n = 1 if cfg.smoke else P0_N
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    missing_traj = {}
    for task in MR0_TASKS:
        missing_traj[task] = [i for i in heldout_indices(man, task, n) if not traj_pkl(repo, task, i).is_file()]

    p0_path = root / "P0" / "native.json"
    p0_path.parent.mkdir(parents=True, exist_ok=True)
    p0_rows = []
    if p0_path.is_file():
        p0_rows = json.loads(p0_path.read_text()).get("rows") or []
    have = {(r["task"], int(r["ep_index"])) for r in p0_rows}
    jobs = []
    for task in MR0_TASKS:
        seeds = demo_seed_list(repo, task, 80)
        for idx in heldout_indices(man, task, n):
            if (task, idx) in have:
                continue
            seed = seeds[idx] if idx < len(seeds) else idx
            tpath = root / "traces" / task / f"ep_{idx:03d}.h5"
            jobs.append(
                {
                    "task": task,
                    "ep_index": idx,
                    "seed": seed,
                    "robotwin_repo": str(repo),
                    "record": True,
                    "trace_path": str(tpath),
                }
            )
    print(f"[rtwx-ac3] P0 native pending={len(jobs)} missing_traj={ {t: len(v) for t,v in missing_traj.items()} }", flush=True)

    def _c0(rec):
        p0_rows.append(rec)
        _write_json(p0_path, {"rows": p0_rows, "summary": summarize_rows(p0_rows)})
        print(f"[rtwx-ac3] P0 {rec.get('task')} ep={rec.get('ep_index')} success={rec.get('success')} err={rec.get('err')}", flush=True)

    if jobs:
        _mp(jobs, workers, _native_worker, commit=_c0)
    p0_sum = summarize_rows(p0_rows)
    p0_ok = _gate(p0_sum, cfg.smoke)
    _write_json(root / "P0" / "summary.json", {"ok": p0_ok, "summary": p0_sum, "missing_traj": missing_traj, "threshold": P0_MIN})

    # D0: always run file audit; Q1/Q2 need traces
    d0 = {}
    for task in MR0_TASKS:
        d0[task] = []
        for idx in heldout_indices(man, task, n):
            ac0_a = None
            cpath = ac0 / "cache" / task / f"ep_{int(idx):03d}.npz"
            if cpath.is_file():
                ep = load_cached_episode(cpath, task)
                ac0_a = np.asarray(ep["a"], dtype=np.float64)
            tpath = root / "traces" / task / f"ep_{idx:03d}.h5"
            trace = None
            if tpath.is_file():
                import h5py

                with h5py.File(tpath, "r") as f:
                    trace = {k: np.asarray(f[k]) for k in f.keys()}
            d0[task].append(audit_episode(repo, task, idx, ac0_a, trace))
    _write_json(root / "D0" / "provenance.json", d0)

    q3_absent = all(rec.get("Q3", {}).get("velocity_absent", True) for t in d0 for rec in d0[t])
    q4_votes = [rec.get("Q4_converted_state_vs_action", {}).get("inferred") for t in d0 for rec in d0[t]]
    d0_roll = {
        "Q3_velocity_absent_all": q3_absent,
        "Q4_inferred_mode": max(set(q4_votes), key=q4_votes.count) if q4_votes else None,
        "raw_hdf5_any": any(rec.get("raw_hdf5") for t in d0 for rec in d0[t]),
        "per_task_first": {t: d0[t][0] if d0[t] else {} for t in MR0_TASKS},
    }
    ac2_p0 = None
    ac2s = ac2 / "P0" / "summary.json"
    if ac2s.is_file():
        ac2_p0 = json.loads(ac2s.read_text())
    b0_fail = True if ac2_p0 is None else (not bool(ac2_p0.get("ok")))

    if not p0_ok:
        missing_all = all(len(missing_traj[t]) == n for t in MR0_TASKS)
        summary = {
            "pattern": "native_expert_replay_failure",
            "environment_replay_contract_failure": missing_all,
            "scientific_result": True,
            "purpose": "demonstration_execution_contract",
            "P0": {"ok": False, "summary": p0_sum, "missing_traj": missing_traj},
            "D0": d0_roll,
            "R0_run": False,
            "AC2_B0_sparse_hdf5": ac2_p0,
            "not_claimed": "policy_state_insufficiency",
        }
        _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
        _write_json(root / "summary.json", summary)
        print(f"[rtwx-ac3] STOP pattern=native_expert_replay_failure pooled={p0_sum['pooled']}", flush=True)
        return summary

    # R0 only after native PASS
    r0_jobs = []
    for task in MR0_TASKS:
        seeds = demo_seed_list(repo, task, 80)
        for idx in heldout_indices(man, task, n):
            tpath = root / "traces" / task / f"ep_{idx:03d}.h5"
            if not tpath.is_file():
                continue
            seed = seeds[idx] if idx < len(seeds) else idx
            for kind in ("B1", "B2", "B3"):
                r0_jobs.append({"kind": kind, "task": task, "ep_index": idx, "seed": seed, "trace_path": str(tpath), "robotwin_repo": str(repo)})
    r0_rows: list[dict[str, Any]] = []
    r0p = root / "R0" / "rows.json"
    r0p.parent.mkdir(parents=True, exist_ok=True)

    def _cr(rec):
        r0_rows.append(rec)
        _write_json(r0p, {"rows": r0_rows})
        print(f"[rtwx-ac3] R0 {rec.get('kind')} {rec.get('task')} ep={rec.get('ep_index')} success={rec.get('success')}", flush=True)

    print(f"[rtwx-ac3] R0 pending={len(r0_jobs)}", flush=True)
    if r0_jobs:
        _mp(r0_jobs, workers, _r0_worker, commit=_cr)
    def _sum_kind(k):
        return summarize_rows([r for r in r0_rows if r.get("kind") == k])

    s1, s2, s3 = _sum_kind("B1"), _sum_kind("B2"), _sum_kind("B3")
    pat = r0_pattern(True, s1, s2, s3, b0_fail)
    summary = {
        "pattern": pat,
        "scientific_result": True,
        "purpose": "demonstration_execution_contract",
        "P0": {"ok": True, "summary": p0_sum},
        "R0": {"B0_sparse_hdf5_fail": b0_fail, "B1_dense_q_qd": s1, "B2_dense_q_only": s2, "B3_sparse_hold": s3},
        "D0": d0_roll,
        "unlocks_policy": False,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    print(f"[rtwx-ac3] pattern={pat}", flush=True)
    return summary

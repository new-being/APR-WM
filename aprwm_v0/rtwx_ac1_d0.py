"""RTWX-AC1-D0: closed-loop state-coverage diagnosis. No training. Shadow metrics only."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation.ac0_rollout_eval import rollout_episode, wrap_topp_large_jump
from .evaluation.state_coverage_metrics import auroc_higher_is_policy, first_persistent_exit, ood_occupancy, tau_95
from .evaluation.state_support import OBJECT_SLICE, ROBOT_SLICE, TaskConditionedStateSupport, split_robot_object
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.explicit_chunk_mlp import ExplicitChunkPolicy
from .policy.task_embedding import STATE_DIM, TASK_NAMES, pack_state
from .rtwx_ac0 import PROCESS_DIM, TASK_EMB, TASK_STEP_LIM, load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json
from .task_x0 import _install_mplib_fallback, _load_task_args, _setup_env

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0AC1D0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ac1_d0.state_coverage.v1"
SEED0 = 51101
K_NN = 5
PERSIST = 3
G1_MAX_EXPERT_OOD = 0.10
G2_MIN_AUROC = 0.80
G3_EARLY_FRAC = 0.70
G3_EARLY_RATIO = 0.5


@dataclass(frozen=True)
class RTWXAC1D0Config:
    output: str = "runs/rtwx_ac1_d0"
    ac0_run: str = "runs/rtwx_ac0"
    robotwin_repo: str = "/root/RoboTwin"
    smoke: bool = False
    workers: int = 8


def _n_ep(smoke: bool) -> int:
    return 2 if smoke else 16


def _episode_states(ep: dict[str, Any]) -> np.ndarray:
    return np.stack([pack_state(fr) for fr in ep["frames"]]).astype(np.float64)


def _load_ac0_splits(ac0_root: Path, smoke: bool) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    _ = smoke
    man = json.loads((ac0_root / "split_manifest.json").read_text())
    train: dict[str, list[np.ndarray]] = {t: [] for t in MR0_TASKS}
    val: dict[str, list[np.ndarray]] = {t: [] for t in MR0_TASKS}
    test: dict[str, list[np.ndarray]] = {t: [] for t in MR0_TASKS}
    for task in MR0_TASKS:
        sp = man[task]
        for split, bucket in (("train", train), ("val", val), ("test", test)):
            for i in sp[split]:
                cpath = ac0_root / "cache" / task / f"ep_{int(i):03d}.npz"
                if not cpath.is_file():
                    continue
                ep = load_cached_episode(cpath, task)
                bucket[task].append(_episode_states(ep))
        if not train[task]:
            raise RuntimeError(f"no AC0 train cache for {task}")
    return train, val, test


def _stack(xs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(xs, axis=0) if xs else np.zeros((0, STATE_DIM))


def _supports_for_task(
    train_s: np.ndarray, mu: np.ndarray, sig: np.ndarray
) -> dict[str, TaskConditionedStateSupport]:
    tr_r, tr_o = split_robot_object(train_s)
    return {
        "all": TaskConditionedStateSupport(train_s, mu, sig, k=K_NN),
        "robot": TaskConditionedStateSupport(tr_r, mu[ROBOT_SLICE], sig[ROBOT_SLICE], k=K_NN),
        "object": TaskConditionedStateSupport(tr_o, mu[OBJECT_SLICE], sig[OBJECT_SLICE], k=K_NN),
    }


def _d0_worker(job: dict[str, Any]) -> dict[str, Any]:
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
        use_process=bool(ckpt.get("use_process", False)),
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
        return {"task": task, "seed": seed, "success": False, "safety_violation": True, "err": str(err), "states": []}
    try:
        wrap_topp_large_jump(env)
        lim = int(getattr(env, "step_lim", None) or TASK_STEP_LIM[task])
        rec = rollout_episode(
            env,
            task,
            model=model,
            normalizer=norm,
            device=device,
            use_process=bool(ckpt.get("use_process", False)),
            max_ticks=min(lim, 40 if job["smoke"] else lim),
            log_states=True,
        )
        rec["seed"] = int(seed)
        rec["used_seed"] = int(used)
        st = rec.pop("states")
        rec["states"] = np.asarray(st, dtype=np.float32)
        return rec
    except Exception as exc:
        return {"task": task, "seed": seed, "success": False, "safety_violation": True, "err": repr(exc), "states": []}
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def _episode_diag(states: np.ndarray, sup: dict[str, TaskConditionedStateSupport], tau: float, t_max: int) -> dict[str, Any]:
    if states.size == 0:
        return {
            "r_ood": 0.0,
            "t_exit": None,
            "t_exit_bar": None,
            "d_max": 0.0,
            "d_all": [],
            "d_robot_mean": 0.0,
            "d_object_mean": 0.0,
            "early_exit": False,
        }
    d_all = sup["all"].distances(states)
    d_r = sup["robot"].distances(states[:, ROBOT_SLICE])
    d_o = sup["object"].distances(states[:, OBJECT_SLICE])
    t_exit = first_persistent_exit(d_all, tau, persist=PERSIST)
    t_max = max(int(t_max), 1)
    return {
        "r_ood": ood_occupancy(d_all, tau),
        "t_exit": t_exit,
        "t_exit_bar": None if t_exit is None else float(t_exit) / float(t_max),
        "d_max": float(np.max(d_all)),
        "d_all_mean": float(np.mean(d_all)),
        "d_robot_mean": float(np.mean(d_r)),
        "d_object_mean": float(np.mean(d_o)),
        "early_exit": bool(t_exit is not None and t_exit < G3_EARLY_RATIO * t_max),
        "n_ticks": int(d_all.size),
        "d_all": [float(x) for x in d_all[:: max(1, d_all.size // 64)]],
    }


def run_rtwx_ac1_d0(output: str | Path, config: RTWXAC1D0Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXAC1D0Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    sel = json.loads((ac0 / "selection.json").read_text())
    winner = str(sel["winner"])
    ckpt_path = ac0 / winner / "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rtwx-ac1-d0] winner={winner} device={device}", flush=True)

    train, val, test = _load_ac0_splits(ac0, cfg.smoke)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    mu, sig = np.asarray(norm.mu_s), np.asarray(norm.sig_s)

    supports = {t: _supports_for_task(_stack(train[t]), mu, sig) for t in MR0_TASKS}
    tau: dict[str, float] = {}
    g1_task: dict[str, float] = {}
    expert_test_d: list[float] = []
    expert_val_d: list[float] = []
    for task in MR0_TASKS:
        d_val = supports[task]["all"].distances(_stack(val[task])) if val[task] else np.zeros(0)
        tau[task] = tau_95(d_val)
        expert_val_d.extend(float(x) for x in d_val)
        d_te = supports[task]["all"].distances(_stack(test[task])) if test[task] else np.zeros(0)
        g1_task[task] = ood_occupancy(d_te, tau[task])
        expert_test_d.extend(float(x) for x in d_te)

    r_ood_expert = float(np.mean([g1_task[t] for t in MR0_TASKS]))
    g1 = bool(r_ood_expert <= G1_MAX_EXPERT_OOD and all(g1_task[t] <= G1_MAX_EXPERT_OOD for t in MR0_TASKS))

    n_ep = _n_ep(cfg.smoke)
    jobs = [
        {
            "task": task,
            "seed": SEED0 + j,
            "ckpt_path": str(ckpt_path),
            "device": device,
            "robotwin_repo": cfg.robotwin_repo,
            "smoke": cfg.smoke,
        }
        for task in MR0_TASKS
        for j in range(n_ep)
    ]
    rows: list[dict[str, Any]] = []
    workers = 1 if cfg.smoke else max(1, int(cfg.workers))
    if workers == 1:
        recs = [_d0_worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        recs = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [pool.submit(_d0_worker, j) for j in jobs]
            for fut in as_completed(futs):
                recs.append(fut.result())

    policy_d: list[float] = []
    failed_early = []
    per_task_rows: dict[str, list[dict[str, Any]]] = {t: [] for t in MR0_TASKS}
    for rec in recs:
        task = str(rec["task"])
        st = np.asarray(rec.get("states", []), dtype=np.float64)
        if st.ndim == 1:
            st = st.reshape(0, STATE_DIM)
        t_max = TASK_STEP_LIM[task] if not cfg.smoke else 40
        diag = _episode_diag(st, supports[task], tau[task], t_max)
        if st.size:
            policy_d.extend(supports[task]["all"].distances(st).tolist())
        slim = {k: rec[k] for k in rec if k != "states"}
        slim.update(diag)
        rows.append(slim)
        per_task_rows[task].append(slim)
        if not slim.get("success", False):
            failed_early.append(bool(slim["early_exit"]))
        print(
            f"[rtwx-ac1-d0] {task} seed={slim.get('seed')} success={slim.get('success')} "
            f"r_ood={slim['r_ood']:.3f} t_exit={slim['t_exit']}",
            flush=True,
        )

    auroc = auroc_higher_is_policy(np.asarray(policy_d), np.asarray(expert_test_d))
    g2 = bool(auroc >= G2_MIN_AUROC)
    p_early = float(np.mean(failed_early)) if failed_early else 0.0
    g3 = bool(p_early >= G3_EARLY_FRAC)
    if not g1:
        pattern = "state_support_metric_invalid"
    elif g1 and g2 and g3:
        pattern = "state_coverage_failure_supported"
    else:
        pattern = "covariate_shift_not_localized"

    summary = {
        "pattern": pattern,
        "scientific_result": True,
        "unlocks_ac1_r0": pattern == "state_coverage_failure_supported",
        "winner_policy": winner,
        "k": K_NN,
        "g1": {"ok": g1, "r_ood_expert_test": r_ood_expert, "per_task": g1_task, "tau_95": tau},
        "g2": {"ok": g2, "auroc": auroc},
        "g3": {"ok": g3, "p_early_exit": p_early, "n_failed": len(failed_early)},
        "pooled_success": float(np.mean([1.0 if r.get("success") else 0.0 for r in rows])) if rows else 0.0,
        "mean_d_robot": float(np.mean([r["d_robot_mean"] for r in rows])) if rows else 0.0,
        "mean_d_object": float(np.mean([r["d_object_mean"] for r in rows])) if rows else 0.0,
        "n_ep": n_ep,
        "seed0": SEED0,
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "rows.json", {"rows": rows})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-ac1-d0] pattern={pattern} g1={g1} g2={g2} auroc={auroc:.3f} g3={g3} p_early={p_early:.3f}", flush=True)
    return summary

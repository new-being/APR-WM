"""AC3-R0 replay decomposition. No policy. Dense q+qd vs dense q vs cadence-correct sparse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .ac3_native import converted_hdf5
from .ac3_provenance import load_converted_q14
from ..task_x0 import _load_task_args, _setup_env


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]) for k in f.keys()}


def _apply_des(env, q14: np.ndarray, qd14: np.ndarray) -> None:
    q = np.asarray(q14, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd14, dtype=np.float64).reshape(-1)
    env.robot.set_arm_joints(q[:6], qd[:6], "left")
    env.robot.set_gripper(float(q[6]), "left")
    env.robot.set_arm_joints(q[7:13], qd[7:13], "right")
    env.robot.set_gripper(float(q[13]), "right")


def replay_dense(repo: Path, task: str, seed: int, trace_path: Path, zero_vel: bool) -> dict[str, Any]:
    tr = _load_trace(trace_path)
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    env, used, err = _setup_env(repo, task, int(seed), 8, args)
    if env is None:
        return {"success": False, "safety_violation": True, "err": str(err), "seed": seed}
    try:
        q = tr["q_des"]
        qd = np.zeros_like(q) if zero_vel else tr["qd_des"]
        for k in range(len(q)):
            _apply_des(env, q[k], qd[k])
            env.scene.step()
        success = bool(env.check_success()) or bool(getattr(env, "eval_success", False))
        return {"success": bool(success), "safety_violation": False, "seed": int(used), "n_steps": int(len(q)), "zero_vel": zero_vel}
    except Exception as exc:
        return {"success": False, "safety_violation": True, "err": repr(exc), "seed": seed}
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def replay_sparse_hold(repo: Path, task: str, seed: int, ep_index: int, trace_path: Path) -> dict[str, Any]:
    st, act = load_converted_q14(converted_hdf5(repo, task, ep_index))
    if act is None:
        return {"success": False, "safety_violation": False, "err": "missing_converted_hdf5"}
    tr = _load_trace(trace_path)
    idx = np.asarray(tr["saved_frame_idx"], dtype=np.int64)
    if idx.size >= 2:
        gaps = np.diff(idx)
        gaps = np.clip(gaps, 1, None)
        if len(gaps) < len(act):
            med = int(np.median(gaps)) if gaps.size else 15
            gaps = np.concatenate([gaps, np.full(len(act) - len(gaps), med, dtype=np.int64)])
        holds = gaps[: len(act)]
    else:
        holds = np.full(len(act), 15, dtype=np.int64)
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    env, used, err = _setup_env(repo, task, int(seed), 8, args)
    if env is None:
        return {"success": False, "safety_violation": True, "err": str(err)}
    try:
        z = np.zeros(14)
        for j in range(len(act)):
            for _ in range(int(holds[j])):
                _apply_des(env, act[j], z)
                env.scene.step()
        success = bool(env.check_success()) or bool(getattr(env, "eval_success", False))
        return {"success": bool(success), "safety_violation": False, "seed": int(used), "n_cmds": int(len(act))}
    except Exception as exc:
        return {"success": False, "safety_violation": True, "err": repr(exc)}
    finally:
        try:
            env.close_env()
        except Exception:
            pass

"""P0: replay hdf5 expert qpos actions from original demo seed. No policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..evaluation.ac0_rollout_eval import wrap_topp_large_jump
from ..task_x0 import _discover_hdf5, _load_hdf5_qpos, _load_task_args, _setup_env


def demo_seed_list(repo, task: str, n: int) -> list[int]:
    seeds_path = repo / "data" / "demo_clean" / task / "aloha_agilex" / "seed.txt"
    if seeds_path.is_file():
        vals = [int(x) for x in seeds_path.read_text().split()]
        return vals
    return list(range(n))


def replay_episode(repo, task: str, ep_index: int, smoke: bool) -> dict[str, Any]:
    hdf5s = _discover_hdf5(repo, task)
    if ep_index >= len(hdf5s):
        return {"task": task, "ep_index": ep_index, "success": False, "safety_violation": True, "err": "missing_hdf5"}
    qpack = _load_hdf5_qpos(hdf5s[ep_index])
    if qpack is None:
        return {"task": task, "ep_index": ep_index, "success": False, "safety_violation": True, "err": "unreadable_hdf5"}
    seeds = demo_seed_list(repo, task, len(hdf5s))
    seed = seeds[ep_index] if ep_index < len(seeds) else ep_index
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    env, used, err = _setup_env(repo, task, int(seed), 8, args)
    if env is None:
        return {"task": task, "ep_index": ep_index, "success": False, "safety_violation": True, "err": str(err), "seed": seed}
    qa = np.asarray(qpack["qa"], dtype=np.float64)
    if smoke:
        qa = qa[: min(24, len(qa))]
    safety = False
    try:
        wrap_topp_large_jump(env)
        for t in range(int(qa.shape[0])):
            if bool(getattr(env, "eval_success", False)):
                break
            env.take_action(qa[t], action_type="qpos")
        success = bool(env.check_success()) or bool(getattr(env, "eval_success", False))
        return {
            "task": task,
            "ep_index": int(ep_index),
            "seed": int(used),
            "success": bool(success),
            "safety_violation": False,
            "n_actions": int(qa.shape[0]),
        }
    except Exception as exc:
        safety = True
        return {
            "task": task,
            "ep_index": int(ep_index),
            "seed": int(used),
            "success": False,
            "safety_violation": True,
            "err": repr(exc),
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass

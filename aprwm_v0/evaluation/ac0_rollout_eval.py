"""Closed-loop Ka=1 bakeoff / P0-G3. Oracle structured state. No adaptive replan."""

from __future__ import annotations

import signal
from typing import Any

import numpy as np
import torch

TAKE_ACTION_TIMEOUT_S = 20


class TakeActionTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise TakeActionTimeout()

from ..policy.process_features import ProcessEncoder
from ..policy.task_embedding import TASK_TO_ID, frame_from_oracle_extract, pack_state
from ..task_x0 import TASK_META, _pose_q, extract_oracle_frame


def _q14(env: Any) -> np.ndarray:
    ql = np.asarray(env.robot.get_left_arm_real_jointState(), dtype=np.float64).reshape(-1)
    qr = np.asarray(env.robot.get_right_arm_real_jointState(), dtype=np.float64).reshape(-1)
    return np.concatenate([ql, qr]).reshape(14)


def wrap_topp_large_jump(env: Any, max_joint_l2: float = 2.0) -> None:
    """Treat huge qpos jumps as TOPP failure (RoboTwin already falls back to 50 steps)."""
    robot = getattr(env, "robot", None)
    if robot is None:
        return
    for name in ("left_mplib_planner", "right_mplib_planner"):
        pl = getattr(robot, name, None)
        if pl is None or not hasattr(pl, "TOPP"):
            continue
        orig = pl.TOPP

        def _make(o):
            def TOPP(path, dt, verbose=True):
                p = np.asarray(path, dtype=np.float64)
                if p.ndim >= 2 and np.linalg.norm(p[-1] - p[0]) > max_joint_l2:
                    raise RuntimeError("ac0_large_joint_jump")
                return o(path, dt, verbose=False)

            return TOPP

        pl.TOPP = _make(orig)


def oracle_frame(env: Any, task: str, prev_q: np.ndarray | None) -> dict[str, np.ndarray]:
    fr = extract_oracle_frame(env, task)
    q = _q14(env)
    qd = np.zeros(14) if prev_q is None else (q - prev_q)
    packed = frame_from_oracle_extract(fr, q, qd)
    actor = getattr(env, TASK_META[task]["actor"])
    packed["obj_q"] = _pose_q(actor)
    if hasattr(env, "cabinet"):
        packed["cabinet_q"] = np.asarray(env.cabinet.get_qpos(), dtype=np.float64).reshape(-1)[:1]
    return packed


@torch.no_grad()
def predict_chunk(
    model,
    normalizer,
    frame: dict[str, np.ndarray],
    task: str,
    device: str,
    use_process: bool,
) -> np.ndarray:
    model.eval()
    enc = ProcessEncoder()
    st = pack_state(frame)[None]
    pr = enc.encode(task, frame)[None]
    s = torch.from_numpy(normalizer.n_state(st).astype(np.float32)).to(device)
    p = torch.from_numpy(normalizer.n_proc(pr).astype(np.float32)).to(device)
    tid = torch.tensor([TASK_TO_ID[task]], dtype=torch.long, device=device)
    y = model(s, tid, p)
    return np.asarray(normalizer.denorm_act(y.cpu().numpy())[0], dtype=np.float64)


def rollout_episode(
    env: Any,
    task: str,
    *,
    model,
    normalizer,
    device: str,
    use_process: bool,
    max_ticks: int,
    log_states: bool = False,
    chunk_fn=None,
    episode_seed: int = 0,
) -> dict[str, Any]:
    prev_q = None
    n_plan = 0
    safety = False
    t = -1
    states: list[np.ndarray] = []
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    for t in range(max_ticks):
        try:
            if bool(getattr(env, "eval_success", False)):
                break
            if getattr(env, "take_action_cnt", 0) >= int(getattr(env, "step_lim", None) or 10**9):
                break
            frame = oracle_frame(env, task, prev_q)
            prev_q = frame["q"].copy()
            if log_states:
                states.append(pack_state(frame).astype(np.float32))
            if chunk_fn is not None:
                chunk = chunk_fn(frame, t, episode_seed)
            else:
                chunk = predict_chunk(model, normalizer, frame, task, device, use_process)
            n_plan += 1
            if t % 25 == 0:
                print(f"[rtwx-ac0] tick={t} task={task}", flush=True)
            signal.alarm(TAKE_ACTION_TIMEOUT_S)
            env.take_action(chunk[0], action_type="qpos")
            signal.alarm(0)
        except TakeActionTimeout:
            safety = True
            break
        except Exception:
            signal.alarm(0)
            safety = True
            break
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)
    success = False
    try:
        success = bool(env.check_success()) or bool(getattr(env, "eval_success", False))
    except Exception:
        success = False
    out = {
        "task": task,
        "success": bool(success),
        "safety_violation": bool(safety),
        "n_plan": int(n_plan),
        "ticks": int(t + 1),
    }
    if log_states:
        out["states"] = np.stack(states) if states else np.zeros((0, 57), dtype=np.float32)
    return out

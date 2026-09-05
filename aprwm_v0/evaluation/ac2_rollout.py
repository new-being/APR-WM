"""AC2 closed-loop next-action policy. Ka=1. Past actions = sent commands."""

from __future__ import annotations

import signal
from typing import Any

import numpy as np
import torch

from .ac0_rollout_eval import TAKE_ACTION_TIMEOUT_S, TakeActionTimeout, _alarm, _q14, oracle_frame
from ..policy.history_buffer import PolicyHistory
from ..policy.task_embedding import TASK_TO_ID, pack_state


@torch.no_grad()
def predict_action(model, hist: PolicyHistory, norm, task: str, device: str) -> np.ndarray:
    model.eval()
    st = torch.from_numpy(norm.n_state(hist.state_stack()[None]).astype(np.float32)).to(device)
    tid = torch.tensor([TASK_TO_ID[task]], dtype=torch.long, device=device)
    pa = None
    if model.kind == "B2":
        pa = torch.from_numpy(norm.n_act(hist.action_stack()[None]).astype(np.float32)).to(device)
    y = model(st, tid, pa)
    return np.asarray(norm.denorm_act(y.detach().cpu().numpy())[0], dtype=np.float64)


def rollout_ac2(env: Any, task: str, model, norm, device: str, max_ticks: int) -> dict[str, Any]:
    prev_q = None
    frame = oracle_frame(env, task, prev_q)
    prev_q = frame["q"].copy()
    hist = PolicyHistory()
    hist.reset(pack_state(frame), _q14(env))
    safety = False
    t = -1
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    for t in range(max_ticks):
        try:
            if bool(getattr(env, "eval_success", False)):
                break
            if getattr(env, "take_action_cnt", 0) >= int(getattr(env, "step_lim", None) or 10**9):
                break
            action = predict_action(model, hist, norm, task, device)
            signal.alarm(TAKE_ACTION_TIMEOUT_S)
            env.take_action(action, action_type="qpos")
            signal.alarm(0)
            frame = oracle_frame(env, task, prev_q)
            prev_q = frame["q"].copy()
            hist.update_after_step(pack_state(frame), action)
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
    return {"task": task, "success": bool(success), "safety_violation": bool(safety), "ticks": int(t + 1)}

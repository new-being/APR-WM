"""Causal left-padded state/action windows. No future leakage."""

from __future__ import annotations

import numpy as np


HS = 4
PAST_A = 3


def left_pad_states(states: np.ndarray, t: int, length: int = HS) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64)
    out = []
    for k in range(t - length + 1, t + 1):
        out.append(s[0] if k < 0 else s[k])
    return np.stack(out, 0)


def left_pad_past_actions(actions: np.ndarray, t: int, reset_action: np.ndarray, length: int = PAST_A) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    ar = np.asarray(reset_action, dtype=np.float64).reshape(-1)
    idx = []
    out = []
    for k in range(t - length, t):
        idx.append(int(k))
        if k < 0:
            out.append(ar)
        else:
            out.append(a[k])
    if idx:
        assert max(idx) < int(t), (idx, t)
    return np.stack(out, 0)

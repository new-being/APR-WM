"""Episode-first single-step windows. Split episodes before this."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .history_windows import HS, PAST_A, left_pad_past_actions, left_pad_states
from ..policy.task_embedding import TASK_TO_ID, pack_state


def episode_states_actions(ep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ep["a"], dtype=np.float64)
    states = np.stack([pack_state(ep["frames"][t]) for t in range(int(a.shape[0]))], 0)
    reset_a = np.asarray(ep["q"][0], dtype=np.float64).reshape(-1)
    return states, a, reset_a


class HistoryWindowDataset(Dataset):
    def __init__(self, episodes: list[dict[str, Any]], *, include_action_history: bool, state_history: int = HS):
        self.include_action_history = bool(include_action_history)
        self.samples: list[dict[str, Any]] = []
        for ep in episodes:
            states, actions, reset_a = episode_states_actions(ep)
            tid = np.int64(TASK_TO_ID[ep["task"]])
            t_len = int(actions.shape[0])
            for t in range(t_len):
                st = left_pad_states(states, t, length=state_history)
                rec = {
                    "states": st.astype(np.float32),
                    "task_id": tid,
                    "target_action": actions[t].astype(np.float32),
                    "t": np.int64(t),
                }
                if self.include_action_history:
                    rec["past_actions"] = left_pad_past_actions(actions, t, reset_a, length=PAST_A).astype(np.float32)
                self.samples.append(rec)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = self.samples[i]
        out = {
            "states": torch.from_numpy(s["states"]),
            "task_id": torch.tensor(int(s["task_id"]), dtype=torch.long),
            "target_action": torch.from_numpy(s["target_action"]),
        }
        if self.include_action_history:
            out["past_actions"] = torch.from_numpy(s["past_actions"])
        return out

"""Episode-first action-chunk windows. Discard tails shorter than Ha."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .process_features import ProcessEncoder
from .task_embedding import TASK_TO_ID, pack_state


HORIZON = 8


def windows_from_episode(ep: dict[str, Any], horizon: int = HORIZON) -> list[dict[str, Any]]:
    q = np.asarray(ep["q"], dtype=np.float64)
    a = np.asarray(ep["a"], dtype=np.float64)
    t_len = int(a.shape[0])
    enc = ProcessEncoder()
    out: list[dict[str, Any]] = []
    for t in range(0, t_len - horizon + 1):
        frame = {k: ep["frames"][t][k] for k in ep["frames"][t]}
        proc = enc.encode(ep["task"], frame, previous_action=None if t == 0 else a[t - 1])
        st = pack_state(frame)
        chunk = a[t : t + horizon]
        if chunk.shape[0] < horizon:
            continue
        out.append(
            {
                "state": st.astype(np.float32),
                "process": proc.astype(np.float32),
                "task_id": np.int64(TASK_TO_ID[ep["task"]]),
                "chunk": chunk.astype(np.float32),
            }
        )
    return out


def split_episode_indices(n: int, rng: np.random.Generator, n_train: int = 40, n_val: int = 5, n_test: int = 5) -> dict[str, list[int]]:
    if n_train + n_val + n_test != n:
        raise ValueError(f"split {n_train}/{n_val}/{n_test} != n={n}")
    perm = rng.permutation(n)
    return {
        "train": [int(i) for i in perm[:n_train]],
        "val": [int(i) for i in perm[n_train : n_train + n_val]],
        "test": [int(i) for i in perm[n_train + n_val :]],
    }


class ActionChunkDataset(Dataset):
    def __init__(self, episodes: list[dict[str, Any]], horizon: int = HORIZON):
        self.samples: list[dict[str, Any]] = []
        for ep in episodes:
            self.samples.extend(windows_from_episode(ep, horizon=horizon))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = self.samples[i]
        return {
            "state": torch.from_numpy(s["state"]),
            "process": torch.from_numpy(s["process"]),
            "task_id": torch.tensor(int(s["task_id"]), dtype=torch.long),
            "chunk": torch.from_numpy(s["chunk"]),
        }

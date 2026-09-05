"""Action chunk buffer: replan only on Ka ticks; never hidden replan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class ActionChunkContractError(RuntimeError):
    pass


@dataclass
class ActionChunkBuffer:
    stride: int
    chunk: np.ndarray | None = None
    chunk_start: int | None = None
    n_replans: int = 0

    def action(
        self,
        t: int,
        planner_input: Any,
        planner: Callable[[Any], np.ndarray],
    ) -> tuple[np.ndarray, bool]:
        s = max(int(self.stride), 1)
        replanned = False
        if t == 0 or (self.chunk is None) or (t % s == 0 and t != self.chunk_start):
            if t != 0 and t % s != 0:
                raise ActionChunkContractError(f"hidden replan at t={t} stride={s}")
            raw = planner(planner_input)
            self.chunk = np.asarray(raw)
            if self.chunk.ndim == 1:
                self.chunk = self.chunk.reshape(1, -1)
            if self.chunk.shape[0] < s:
                raise ActionChunkContractError(
                    f"H_a={self.chunk.shape[0]} < stride={s}"
                )
            self.chunk_start = int(t)
            self.n_replans += 1
            replanned = True
        idx = t - int(self.chunk_start)
        if idx < 0 or idx >= len(self.chunk):
            raise ActionChunkContractError(f"chunk index {idx} out of range len={len(self.chunk)}")
        return np.asarray(self.chunk[idx]).copy(), replanned

    def needs_replan(self, t: int) -> bool:
        s = max(int(self.stride), 1)
        if self.chunk is None or self.chunk_start is None:
            if t != 0 and t % s != 0:
                raise ActionChunkContractError(f"hidden replan at t={t} stride={s}")
            return True
        return bool(t % s == 0 and t != self.chunk_start)

    def load(self, chunk: np.ndarray, start_tick: int) -> None:
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] < max(int(self.stride), 1):
            raise ActionChunkContractError(f"H_a={arr.shape[0]} < stride={self.stride}")
        self.chunk = arr
        self.chunk_start = int(start_tick)
        self.n_replans += 1

    def consume(self, t: int) -> np.ndarray:
        if self.chunk is None or self.chunk_start is None:
            raise ActionChunkContractError("consume without load")
        idx = t - int(self.chunk_start)
        if idx < 0 or idx >= len(self.chunk):
            raise ActionChunkContractError(f"chunk index {idx} out of range len={len(self.chunk)}")
        return np.asarray(self.chunk[idx]).copy()

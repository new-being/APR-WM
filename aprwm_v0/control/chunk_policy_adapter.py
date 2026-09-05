"""Unify native chunk-policy output. Forbidden: repeat / rollout / interpolate / extra calls."""

from __future__ import annotations

from typing import Any

import numpy as np

from .action_chunk_spec import ActionChunkSpec


class ChunkAdapterError(RuntimeError):
    pass


def normalize_native_output(raw: Any) -> np.ndarray:
    if isinstance(raw, dict):
        if "actions" in raw:
            raw = raw["actions"]
        elif "action" in raw:
            raw = raw["action"]
        else:
            raise ChunkAdapterError("dict output missing actions")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ChunkAdapterError(f"batch dim {arr.shape[0]} != 1")
        arr = arr[0]
    if arr.ndim != 2:
        raise ChunkAdapterError(f"expected (H, d), got shape {arr.shape}")
    return arr


class ChunkPolicyAdapter:
    def __init__(self, policy: Any, spec: ActionChunkSpec):
        self.policy = policy
        self.spec = spec
        self.n_forward = 0
        self._assert_not_filler()

    def _assert_not_filler(self) -> None:
        p = self.policy
        if getattr(p, "fills_by_repeat", False):
            raise ChunkAdapterError("repeat-fill policy forbidden")
        if getattr(p, "fills_by_rollout", False):
            raise ChunkAdapterError("multi-call rollout fill forbidden")
        if getattr(p, "fills_by_interpolation", False):
            raise ChunkAdapterError("action interpolation forbidden")

    def infer_chunk(self, planner_input: Any) -> np.ndarray:
        n0 = self.n_forward
        raw = self.policy(planner_input) if callable(self.policy) else self.policy.forward(planner_input)
        self.n_forward += 1
        if self.n_forward != n0 + 1:
            raise ChunkAdapterError("hidden extra policy call")
        chunk = normalize_native_output(raw)
        if chunk.shape[0] < self.spec.horizon:
            raise ChunkAdapterError(f"H_a={chunk.shape[0]} < {self.spec.horizon}")
        if chunk.shape[1] != self.spec.action_dim:
            raise ChunkAdapterError(f"d_a={chunk.shape[1]} != {self.spec.action_dim}")
        return np.asarray(chunk[: self.spec.horizon], dtype=np.float64)

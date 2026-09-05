"""Frozen B2 quotient evaluator — thin wrapper over R1/R5 search (no duplicated logic)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..rtwx_o0e0 import K_SPHERE, REFINE_DEGS, SPHERE_SEED, TOP_KR, fibonacci_sphere
from ..rtwx_o0e0r1 import S_at_center, estimate_axis_b2_at_center
from ..rtwx_o0e0r5 import estimate_axis_b2_center_fn

B2_FROZEN_CONFIG = {
    "K_SPHERE": K_SPHERE,
    "TOP_KR": TOP_KR,
    "REFINE_DEGS": list(REFINE_DEGS),
    "SPHERE_SEED": SPHERE_SEED,
    "score": "mean_min_sq_hr_2d_kdtree",
}
R5_B2_CONFIG_HASH = hashlib.sha256(
    json.dumps(B2_FROZEN_CONFIG, sort_keys=True).encode()
).hexdigest()
CONSISTENCY_WEIGHT = 0.0


@dataclass(frozen=True)
class AxisSearchResult:
    axis: np.ndarray
    meta: dict[str, float]


@dataclass(frozen=True)
class AxisCandidate:
    axis: np.ndarray
    score: float


class FrozenQuotientB2:
    """Frozen Fibonacci-162 → Top-8 → 5°/3°/1° B2 at fixed or joint center."""

    def __init__(self, tree: Any, sphere: np.ndarray | None = None) -> None:
        self.tree = tree
        self.sphere = np.asarray(
            sphere if sphere is not None else fibonacci_sphere(K_SPHERE, SPHERE_SEED),
            dtype=np.float64,
        )
        assert CONSISTENCY_WEIGHT == 0.0

    @property
    def config_hash(self) -> str:
        return R5_B2_CONFIG_HASH

    def assert_frozen(self) -> None:
        assert self.config_hash == R5_B2_CONFIG_HASH

    def score_axis(
        self,
        cloud_xyz: np.ndarray,
        center_xyz: np.ndarray,
        axis_n: np.ndarray,
    ) -> float:
        return S_at_center(cloud_xyz, center_xyz, axis_n, self.tree)

    def search_fixed_center(
        self,
        cloud_xyz: np.ndarray,
        center_xyz: np.ndarray,
    ) -> AxisSearchResult:
        n_hat, meta = estimate_axis_b2_at_center(
            cloud_xyz, center_xyz, self.tree, sphere=self.sphere,
        )
        return AxisSearchResult(axis=n_hat, meta=meta)

    def coarse_candidates_fixed_center(
        self,
        cloud_xyz: np.ndarray,
        center_xyz: np.ndarray,
        top_k: int = TOP_KR,
    ) -> list[AxisCandidate]:
        p_center = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        scores = np.asarray(
            [S_at_center(cloud_xyz, p_center, n, self.tree) for n in self.sphere],
            dtype=np.float64,
        )
        order = np.argsort(scores)
        k = min(top_k, len(order))
        return [
            AxisCandidate(axis=self.sphere[i].copy(), score=float(scores[i]))
            for i in order[:k]
        ]

    def search_joint_center_fn(
        self,
        cloud_xyz: np.ndarray,
        center_fn: Callable[[np.ndarray], np.ndarray],
    ) -> AxisSearchResult:
        n_hat, meta = estimate_axis_b2_center_fn(
            cloud_xyz, center_fn, self.tree, sphere=self.sphere,
        )
        return AxisSearchResult(axis=n_hat, meta=meta)

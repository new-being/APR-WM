"""Multi-view voxel consensus surface (MEM-B0 B2). One centroid per voxel per frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from .relative_state_chain import ChainState, propagate_state


def invert_hom(t4: np.ndarray) -> np.ndarray:
    t = np.asarray(t4, dtype=np.float64).reshape(4, 4)
    r = t[:3, :3]
    tr = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ tr
    return out


def transform_cloud(cloud: np.ndarray, t4: np.ndarray) -> np.ndarray:
    p = np.asarray(cloud, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    r = np.asarray(t4, dtype=np.float64).reshape(4, 4)[:3, :3]
    tr = np.asarray(t4, dtype=np.float64).reshape(4, 4)[:3, 3]
    return (r @ p.T).T + tr.reshape(1, 3)


def voxel_centroids_once(cloud: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], np.ndarray]:
    p = np.asarray(cloud, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return {}
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] == 0:
        return {}
    vs = max(float(voxel_size), 1e-6)
    keys = np.floor(p / vs).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    acc = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    cnt = np.zeros(uniq.shape[0], dtype=np.float64)
    for i in range(p.shape[0]):
        j = inv[i]
        acc[j] += p[i]
        cnt[j] += 1.0
    acc /= np.maximum(cnt[:, None], 1.0)
    return {tuple(int(x) for x in uniq[j]): acc[j].copy() for j in range(uniq.shape[0])}


@dataclass
class SurfaceVoxel:
    mean_xyz: np.ndarray
    support_frames: int = 0
    scatter_m2: float = 0.0


@dataclass
class PersistentSurfaceMemory:
    voxel_size: float
    voxels: dict[tuple[int, int, int], SurfaceVoxel] = field(default_factory=dict)

    def update_from_cloud(self, cloud_in_memory_frame: np.ndarray) -> None:
        cents = voxel_centroids_once(cloud_in_memory_frame, self.voxel_size)
        for key, centroid in cents.items():
            self._update_one(key, centroid)

    def _update_one(self, key: tuple[int, int, int], centroid: np.ndarray) -> None:
        c = np.asarray(centroid, dtype=np.float64).reshape(3)
        cell = self.voxels.get(key)
        if cell is None:
            self.voxels[key] = SurfaceVoxel(mean_xyz=c.copy(), support_frames=1, scatter_m2=0.0)
            return
        n0 = float(cell.support_frames)
        n1 = n0 + 1.0
        delta = c - cell.mean_xyz
        cell.mean_xyz = cell.mean_xyz + delta / n1
        delta2 = c - cell.mean_xyz
        cell.scatter_m2 = float(cell.scatter_m2 + float(delta @ delta2))
        cell.support_frames = int(n1)

    def surface_points(self) -> np.ndarray:
        if not self.voxels:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([v.mean_xyz for v in self.voxels.values()], axis=0)

    def contamination_median(self) -> float:
        sigmas: list[float] = []
        for v in self.voxels.values():
            if v.support_frames < 2:
                continue
            var = v.scatter_m2 / max(v.support_frames, 1)
            sigmas.append(float(np.sqrt(max(var, 0.0))))
        if not sigmas:
            return float("nan")
        return float(np.median(sigmas))


def register_surface_to_cloud(
    memory: PersistentSurfaceMemory,
    current_cloud_base: np.ndarray,
    s0: ChainState,
    object_diameter: float,
    *,
    icp_cfg: RelativeICPConfig | None = None,
) -> tuple[ChainState | None, dict]:
    src = memory.surface_points()
    res = estimate_relative_rigid(src, current_cloud_base, object_diameter, icp_cfg)
    meta = {"valid": bool(res.valid), "n_corr": int(res.n_corr), "rmse": float(res.rmse)}
    if not res.valid:
        return None, meta
    new = propagate_state(s0, res.R, res.t)
    return new, meta

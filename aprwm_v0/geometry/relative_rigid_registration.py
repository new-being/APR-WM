"""Fixed mutual-NN point-to-point ICP for relative rigid motion (REL0)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

REL_ICP_CONFIG_HASH = hashlib.sha256(
    b"rel_icp.v1.voxel_frac=0.02.corr_frac=0.30.n_iters=10.min_corr=16"
).hexdigest()


@dataclass(frozen=True)
class RelativeICPConfig:
    voxel_frac: float = 0.02
    corr_frac: float = 0.30
    n_iters: int = 10
    min_corr: int = 16


@dataclass
class RelativeRigidResult:
    R: np.ndarray
    t: np.ndarray
    valid: bool
    n_corr: int
    rmse: float


def voxel_reduce(points: np.ndarray, voxel_size: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vs = max(float(voxel_size), 1e-6)
    keys = np.floor(p / vs).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    cnt = np.zeros(uniq.shape[0], dtype=np.float64)
    for i in range(p.shape[0]):
        j = inv[i]
        out[j] += p[i]
        cnt[j] += 1.0
    out /= np.maximum(cnt[:, None], 1.0)
    return out


def fuse_voxel_clouds(
    cloud_h: np.ndarray,
    cloud_o: np.ndarray,
    object_diameter: float,
    *,
    cfg: RelativeICPConfig | None = None,
) -> np.ndarray:
    """Per-view voxel → concat → same-scale voxel (REL0 front-end contract)."""
    cfg = cfg or RelativeICPConfig()
    h = float(cfg.voxel_frac * object_diameter)
    ch = voxel_reduce(cloud_h, h)
    co = voxel_reduce(cloud_o, h)
    if ch.size and co.size:
        return voxel_reduce(np.concatenate([ch, co], axis=0), h)
    return ch if ch.size else co


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    return r, t


def _mutual_pairs(
    src: np.ndarray,
    dst: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    if src.shape[0] < 4 or dst.shape[0] < 4:
        return np.zeros((0, 3)), np.zeros((0, 3))
    from scipy.spatial import cKDTree

    tree_d = cKDTree(dst)
    tree_s = cKDTree(src)
    d_s, j_s = tree_d.query(src, k=1)
    d_d, j_d = tree_s.query(dst, k=1)
    idx_s = np.arange(src.shape[0])
    mutual = (d_s <= max_dist) & (j_d[j_s] == idx_s)
    if not np.any(mutual):
        return np.zeros((0, 3)), np.zeros((0, 3))
    ii = idx_s[mutual]
    jj = j_s[mutual]
    return src[ii], dst[jj]


def estimate_relative_rigid(
    cloud0_B: np.ndarray,
    cloud1_B: np.ndarray,
    object_diameter: float,
    cfg: RelativeICPConfig | None = None,
) -> RelativeRigidResult:
    cfg = cfg or RelativeICPConfig()
    p0 = np.asarray(cloud0_B, dtype=np.float64)
    p1 = np.asarray(cloud1_B, dtype=np.float64)
    if p0.shape[0] < 8 or p1.shape[0] < 8:
        return RelativeRigidResult(
            R=np.eye(3), t=np.zeros(3), valid=False, n_corr=0, rmse=float("inf"),
        )
    c0 = p0.mean(0)
    c1 = p1.mean(0)
    r_acc = np.eye(3, dtype=np.float64)
    t_acc = (c1 - c0).astype(np.float64)
    max_d = float(cfg.corr_frac * object_diameter)
    best_rmse = float("inf")
    best_n = 0
    for _ in range(int(cfg.n_iters)):
        src = (r_acc @ p0.T).T + t_acc
        xs, ys = _mutual_pairs(src, p1, max_d)
        if xs.shape[0] < cfg.min_corr:
            return RelativeRigidResult(
                R=r_acc, t=t_acc, valid=False, n_corr=int(xs.shape[0]), rmse=float("inf"),
            )
        dr, dt = _kabsch(xs, ys)
        r_acc = dr @ r_acc
        t_acc = dr @ t_acc + dt
        err = np.linalg.norm((dr @ xs.T).T + dt - ys, axis=1)
        best_rmse = float(np.sqrt(np.mean(err ** 2)))
        best_n = int(xs.shape[0])
    return RelativeRigidResult(
        R=r_acc, t=t_acc, valid=True, n_corr=best_n, rmse=best_rmse,
    )


def delta_T_from_poses(R0: np.ndarray, p0: np.ndarray, R1: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GT relative rigid: X_1 = R_delta X_0 + t_delta (base frame)."""
    r0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    r1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    r_d = r1 @ r0.T
    t_d = p1 - r_d @ p0
    return r_d, t_d

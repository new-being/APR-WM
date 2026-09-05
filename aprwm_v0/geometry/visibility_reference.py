"""CAD-only axial visibility reference LUT (SO(2) quotient yaw-marginalized)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..rtwx_o0q0 import _R_y

EY_CANON = np.array([0.0, 1.0, 0.0], dtype=np.float64)
ALPHA_GRID_DEG = np.arange(0.0, 181.0, 2.0, dtype=np.float64)
DISTANCE_GRID_M = np.array([0.45, 0.60, 0.75, 0.90, 1.05, 1.20], dtype=np.float64)
YAW_MARGINAL_DEG = tuple(float(g) for g in range(0, 360, 45))
LUT_BUILD_SEED = 43601


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def camera_center_world(extrinsic: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (3, 4):
        t = np.eye(4, dtype=np.float64)
        t[:3, :] = e
        e = t
    t_wc = np.linalg.inv(e)
    return t_wc[:3, 3].astype(np.float64)


def view_geometry(
    axis_n: np.ndarray,
    camera_center: np.ndarray,
    observed_centroid: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return (alpha_rad, distance_m, radial_dir_r) for candidate axis n."""
    n = _unit(axis_n)
    c = np.asarray(observed_centroid, dtype=np.float64).reshape(3)
    C = np.asarray(camera_center, dtype=np.float64).reshape(3)
    v = _unit(C - c)
    cos_a = float(np.clip(np.dot(n, v), -1.0, 1.0))
    alpha = float(np.arccos(cos_a))
    d = float(np.linalg.norm(C - c))
    tang = v - np.dot(n, v) * n
    r = _unit(tang)
    return alpha, d, r


def _cup_outward_normals(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    norms = np.zeros_like(p)
    xz = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    side = xz > 1e-4
    norms[side, 0] = p[side, 0] / xz[side]
    norms[side, 2] = p[side, 2] / xz[side]
    top = ~side
    if top.any():
        sy = np.sign(p[top, 1] - np.median(p[:, 1]))
        sy = np.where(np.abs(sy) < 1e-6, 1.0, sy)
        norms[top, 1] = sy
    nrm = np.linalg.norm(norms, axis=1, keepdims=True)
    return norms / np.maximum(nrm, 1e-12)


def _visible_centroid_object(
    cad_pts: np.ndarray,
    normals: np.ndarray,
    camera_pos: np.ndarray,
) -> np.ndarray | None:
    cam = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    pts = np.asarray(cad_pts, dtype=np.float64)
    view = cam - pts
    vis = np.sum(normals * view, axis=1) > 0.0
    if int(vis.sum()) < 8:
        return None
    return np.mean(pts[vis], axis=0)


def _synthetic_camera_pos(alpha_deg: float, dist_m: float) -> np.ndarray:
    a = np.radians(float(alpha_deg))
    u = np.array([np.sin(a), np.cos(a), 0.0], dtype=np.float64)
    return dist_m * _unit(u)


def _decompose_axial_radial(delta_o: np.ndarray, axis: np.ndarray, radial: np.ndarray) -> tuple[float, float]:
    d = np.asarray(delta_o, dtype=np.float64).reshape(3)
    n = _unit(axis)
    r = _unit(radial)
    a = float(np.dot(d, n))
    b = float(np.dot(d, r))
    return a, b


def _bilinear_lut(grid_a: np.ndarray, grid_b: np.ndarray, alpha: float, dist: float) -> tuple[float, float]:
    ag = np.asarray(grid_a, dtype=np.float64)
    bg = np.asarray(grid_b, dtype=np.float64)
    alphas = ALPHA_GRID_DEG
    dists = DISTANCE_GRID_M
    a_deg = np.clip(np.degrees(alpha), float(alphas[0]), float(alphas[-1]))
    d_m = np.clip(dist, float(dists[0]), float(dists[-1]))
    ia = int(np.searchsorted(alphas, a_deg, side="right") - 1)
    ib = int(np.searchsorted(dists, d_m, side="right") - 1)
    ia = int(np.clip(ia, 0, len(alphas) - 2))
    ib = int(np.clip(ib, 0, len(dists) - 2))
    ta = (a_deg - alphas[ia]) / max(alphas[ia + 1] - alphas[ia], 1e-12)
    tb = (d_m - dists[ib]) / max(dists[ib + 1] - dists[ib], 1e-12)
    a00, a10 = ag[ia, ib], ag[ia + 1, ib]
    a01, a11 = ag[ia, ib + 1], ag[ia + 1, ib + 1]
    b00, b10 = bg[ia, ib], bg[ia + 1, ib]
    b01, b11 = bg[ia, ib + 1], bg[ia + 1, ib + 1]
    a0 = (1.0 - tb) * a00 + tb * a01
    a1 = (1.0 - tb) * a10 + tb * a11
    b0 = (1.0 - tb) * b00 + tb * b01
    b1 = (1.0 - tb) * b10 + tb * b11
    a = (1.0 - ta) * a0 + ta * a1
    b = (1.0 - ta) * b0 + ta * b1
    return float(a), float(b)


@dataclass
class AxialVisibilityReference:
    alpha_grid_deg: np.ndarray
    distance_grid_m: np.ndarray
    axial_bias_a: np.ndarray
    radial_bias_b: np.ndarray
    cad_sha256: str
    yaw_marginalization: int = 8
    meta: dict[str, Any] | None = None

    def predict_bias(
        self,
        axis_n: np.ndarray,
        camera_center: np.ndarray,
        observed_centroid: np.ndarray,
    ) -> np.ndarray:
        alpha, dist, r = view_geometry(axis_n, camera_center, observed_centroid)
        if float(np.linalg.norm(r)) < 1e-6:
            r = _unit(np.cross(axis_n, EY_CANON))
            if float(np.linalg.norm(r)) < 1e-6:
                r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        a, b = _bilinear_lut(self.axial_bias_a, self.radial_bias_b, alpha, dist)
        n = _unit(axis_n)
        return a * n + b * r

    def estimate_center(
        self,
        axis_n: np.ndarray,
        observed_centroid: np.ndarray,
        camera_center: np.ndarray,
    ) -> np.ndarray:
        c = np.asarray(observed_centroid, dtype=np.float64).reshape(3)
        return c - self.predict_bias(axis_n, camera_center, c)

    def estimate_center_dual(
        self,
        axis_n: np.ndarray,
        c_h: np.ndarray,
        c_o: np.ndarray,
        cam_h: np.ndarray,
        cam_o: np.ndarray,
    ) -> np.ndarray:
        p_h = self.estimate_center(axis_n, c_h, cam_h)
        p_o = self.estimate_center(axis_n, c_o, cam_o)
        return 0.5 * (p_h + p_o)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        np.savez_compressed(
            p,
            alpha_grid_deg=self.alpha_grid_deg,
            distance_grid_m=self.distance_grid_m,
            axial_bias_a=self.axial_bias_a,
            radial_bias_b=self.radial_bias_b,
            cad_sha256=np.array(self.cad_sha256),
            yaw_marginalization=np.array(self.yaw_marginalization),
        )

    @classmethod
    def load(cls, path: str | Path) -> AxialVisibilityReference:
        z = np.load(path, allow_pickle=True)
        cad = z["cad_sha256"]
        if cad.ndim == 0:
            cad_s = str(cad.item())
        else:
            cad_s = str(cad)
        return cls(
            alpha_grid_deg=np.asarray(z["alpha_grid_deg"]),
            distance_grid_m=np.asarray(z["distance_grid_m"]),
            axial_bias_a=np.asarray(z["axial_bias_a"]),
            radial_bias_b=np.asarray(z["radial_bias_b"]),
            cad_sha256=cad_s,
            yaw_marginalization=int(z.get("yaw_marginalization", 8)),
        )


def build_lut(
    cad_pts: np.ndarray,
    *,
    cad_path: str | Path | None = None,
    alpha_grid_deg: np.ndarray | None = None,
    distance_grid_m: np.ndarray | None = None,
) -> AxialVisibilityReference:
    pts0 = np.asarray(cad_pts, dtype=np.float64)
    normals0 = _cup_outward_normals(pts0)
    alphas = np.asarray(alpha_grid_deg if alpha_grid_deg is not None else ALPHA_GRID_DEG, dtype=np.float64)
    dists = np.asarray(distance_grid_m if distance_grid_m is not None else DISTANCE_GRID_M, dtype=np.float64)
    na, nd = len(alphas), len(dists)
    grid_a = np.zeros((na, nd), dtype=np.float64)
    grid_b = np.zeros((na, nd), dtype=np.float64)
    p_origin = np.zeros(3, dtype=np.float64)
    for ia, alpha_deg in enumerate(alphas):
        for ib, dist_m in enumerate(dists):
            deltas = []
            cam = _synthetic_camera_pos(alpha_deg, dist_m)
            v = _unit(cam)
            r_dir = _unit(v - np.dot(EY_CANON, v) * EY_CANON)
            if float(np.linalg.norm(r_dir)) < 1e-6:
                r_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            for gamma in YAW_MARGINAL_DEG:
                Rg = _R_y(gamma)
                pts = (Rg @ pts0.T).T
                norms = _cup_outward_normals(pts)
                c_vis = _visible_centroid_object(pts, norms, cam)
                if c_vis is None:
                    continue
                delta = c_vis - p_origin
                deltas.append(delta)
            if not deltas:
                grid_a[ia, ib] = 0.0
                grid_b[ia, ib] = 0.0
                continue
            dbar = np.mean(np.stack(deltas), axis=0)
            a, b = _decompose_axial_radial(dbar, EY_CANON, r_dir)
            grid_a[ia, ib] = a
            grid_b[ia, ib] = b
    cad_sha = "unknown"
    if cad_path is not None and Path(cad_path).is_file():
        cad_sha = hashlib.sha256(Path(cad_path).read_bytes()).hexdigest()
    return AxialVisibilityReference(
        alpha_grid_deg=alphas,
        distance_grid_m=dists,
        axial_bias_a=grid_a,
        radial_bias_b=grid_b,
        cad_sha256=cad_sha,
        yaw_marginalization=len(YAW_MARGINAL_DEG),
        meta={"build_seed": LUT_BUILD_SEED},
    )


def cloud_centroid(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1:
        return np.full(3, np.nan, dtype=np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] < 1:
        return np.full(3, np.nan, dtype=np.float64)
    return np.median(p, axis=0)

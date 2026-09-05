"""Shared O0G3R cache loaders for O0G3A/O0G3B instrument cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g3 import N_MIN
from .rtwx_o0g3r import SEEDS, SPLIT_KEYS, _load_split, load_D_O

__all__ = [
    "SEEDS",
    "SPLIT_KEYS",
    "N_MIN",
    "load_D_O",
    "load_g3r_pools",
    "xo_grid",
    "pack_pixel_samples",
]


def xo_grid(xyz: np.ndarray, mask: np.ndarray, p: np.ndarray, quat: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(xyz).shape, np.nan, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & np.isfinite(xyz).all(axis=-1)
    if not m.any():
        return out
    R = _quat_to_R(quat)
    out[m] = ((xyz[m].astype(np.float64) - np.asarray(p, dtype=np.float64).reshape(1, 3)) @ R).astype(np.float32)
    return out


def load_g3r_pools(cache_root: str | Path, seeds: tuple[int, ...] = SEEDS, split: str = "train") -> dict[str, Any]:
    root = Path(cache_root)
    pools = []
    for seed in seeds:
        path = root / f"cache_o0g3r_s{seed}_{split}.npz"
        hit = _load_split(path, split)
        if hit is None:
            raise FileNotFoundError(f"missing cache {path}")
        pools.append(hit)
    out = {k: np.concatenate([p[k] for p in pools], axis=0) for k in SPLIT_KEYS}
    out["n_ep"] = sum(p["n_ep"] for p in pools)
    out["n_steps"] = pools[0]["n_steps"]
    return out


def pack_pixel_samples(
    pool: dict[str, Any],
    *,
    D_O: float,
    rng: np.random.Generator,
    max_samples: int | None = None,
    views: tuple[str, ...] = ("h", "o"),
) -> dict[str, np.ndarray]:
    """Flatten masked valid pixels across frames/views into training rows."""
    rows: dict[str, list[np.ndarray]] = {k: [] for k in ("uvd", "xo", "rgb", "frame", "view", "area", "depth_frac")}
    h, w = pool["rgb_h"].shape[1:3]
    for fi in range(pool["p"].shape[0]):
        for vi, view in enumerate(views):
            rgb = pool[f"rgb_{view}"][fi]
            xyz = pool[f"xyz_{view}"][fi]
            mask = pool[f"mask_{view}"][fi]
            xo = xo_grid(xyz, mask, pool["p"][fi], pool["quat"][fi])
            dep_ok = np.isfinite(xyz[..., 2])
            m = np.asarray(mask, bool) & dep_ok & np.isfinite(xo).all(-1)
            if not m.any():
                continue
            ys, xs = np.where(m)
            area = float(m.sum())
            depth_frac = float(dep_ok.sum()) / dep_ok.size
            u = xs.astype(np.float32) / max(w - 1, 1)
            v = ys.astype(np.float32) / max(h - 1, 1)
            d = xyz[ys, xs, 2].astype(np.float32)
            uvd = np.stack([u, v, d], axis=1)
            rows["uvd"].append(uvd)
            rows["xo"].append(xo[ys, xs] / D_O)
            rows["rgb"].append(rgb[ys, xs].astype(np.float32) / 255.0)
            rows["frame"].append(np.full(ys.shape[0], fi, dtype=np.int32))
            rows["view"].append(np.full(ys.shape[0], vi, dtype=np.int8))
            rows["area"].append(np.full(ys.shape[0], area, dtype=np.float32))
            rows["depth_frac"].append(np.full(ys.shape[0], depth_frac, dtype=np.float32))
    if not rows["uvd"]:
        raise RuntimeError("pack_pixel_samples: no valid pixels")
    out = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
    if max_samples is not None and out["uvd"].shape[0] > max_samples:
        idx = rng.choice(out["uvd"].shape[0], max_samples, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    return out


def count_corr_frame(pool: dict[str, Any], fi: int, *, n_min: int = N_MIN) -> dict[str, int]:
    p, q = pool["p"][fi], pool["quat"][fi]
    nh = int((np.asarray(pool["mask_h"][fi], bool) & np.isfinite(pool["xyz_h"][fi]).all(-1)).sum())
    no = int((np.asarray(pool["mask_o"][fi], bool) & np.isfinite(pool["xyz_o"][fi]).all(-1)).sum())
    xoh = xo_grid(pool["xyz_h"][fi], pool["mask_h"][fi], p, q)
    xoo = xo_grid(pool["xyz_o"][fi], pool["mask_o"][fi], p, q)
    mh = np.asarray(pool["mask_h"][fi], bool) & np.isfinite(xoh).all(-1)
    mo = np.asarray(pool["mask_o"][fi], bool) & np.isfinite(xoo).all(-1)
    nu = int((mh | mo).sum())
    return {"head": nh, "observer": no, "union": nu, "enough": int(nu >= n_min)}

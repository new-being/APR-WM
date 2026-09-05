"""RTWX-O0G4 G0: oracle sparse CAD keypoints → Kabsch orientation ceiling.

No training. Reuses O0G3R test cache. Unlocks O0G4R prereg on PASS. O1 locked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import _e_R_deg, _quat_fix
from .rtwx_o0c import _R_to_quat
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g3 import MED_ER_MAX, P90_ER_MAX, _kabsch, _score_ori_arr
from .rtwx_o0g3_cache import SEEDS as G3R_SEEDS
from .rtwx_o0g3_cache import load_g3r_pools
from .rtwx_o0g3r import load_D_O
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, P_ANY_SEED
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G4_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g4.oracle_sparse_keypoints.v1"
SEEDS = G3R_SEEDS  # 29601/02/03 — O0G3R test episode reuse
N_GEOM = 3
VIS_RADIUS_M = 0.015
P_ENOUGH_AGG = 0.90
P_ENOUGH_SEED = 0.85
DEGEN_FRAC = 0.10  # 2nd singular value / D_O
KEYPOINT_NAMES = ("bottom", "rim_px", "rim_pz", "rim_nx")


@dataclass(frozen=True)
class RTWXO0G4Config:
    output: str = "runs/rtwx_o0g4"
    g3r_cache: str = "runs/rtwx_o0g3r"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    smoke: bool = False
    vis_radius_m: float = VIS_RADIUS_M
    n_geom: int = N_GEOM


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G4 must not write there")


def _lock(cfg: RTWXO0G4Config) -> RTWXO0G4Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, seeds=SEEDS, vis_radius_m=VIS_RADIUS_M, n_geom=N_GEOM)


def load_cad_keypoints(robotwin_repo: str | Path) -> dict[str, Any]:
    path = Path(robotwin_repo) / "assets/objects/021_cup/model_data0.json"
    data = json.loads(path.read_text())
    c = np.asarray(data["center"], dtype=np.float64) * np.asarray(data["scale"], dtype=np.float64)
    h = np.asarray(data["extents"], dtype=np.float64) * np.asarray(data["scale"], dtype=np.float64) / 2.0
    k = np.stack(
        [
            np.array([c[0], c[1] - h[1], c[2]]),
            np.array([c[0] + h[0], c[1] + h[1], c[2]]),
            np.array([c[0], c[1] + h[1], c[2] + h[2]]),
            np.array([c[0] - h[0], c[1] + h[1], c[2]]),
        ]
    )
    return {"K_O": k, "names": list(KEYPOINT_NAMES), "source": str(path), "aabb_center_m": c.tolist(), "aabb_half_m": h.tolist()}


def _visible_kp(xyz: np.ndarray, mask: np.ndarray, kB: np.ndarray, radius: float) -> np.ndarray:
    m = np.asarray(mask, dtype=bool) & np.isfinite(xyz).all(axis=-1)
    out = np.zeros(kB.shape[0], dtype=bool)
    if not m.any():
        return out
    pts = xyz[m].astype(np.float64)
    for j, k in enumerate(kB):
        out[j] = bool(np.min(np.linalg.norm(pts - k.reshape(1, 3), axis=1)) < radius)
    return out


def _nondegenerate(X_O: np.ndarray, D_O: float) -> bool:
    if X_O.shape[0] < 3:
        return False
    Xc = X_O - X_O.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return bool(s.size >= 2 and s[1] > DEGEN_FRAC * D_O)


def _numpy_pool(n: int, K_O: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    p, quat, xyz_h, xyz_o, mask_h, mask_o, vis_h, vis_o = [], [], [], [], [], [], [], []
    h = w = 16
    for i in range(n):
        ang = 0.3 * np.sin(i / 9.0)
        q = _quat_fix(np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0]))[0]
        R = _quat_to_R(q)
        pi = np.array([-0.2, -0.05, 0.74])
        kB = (R @ K_O.T).T + pi
        xyz = np.zeros((h, w, 3), dtype=np.float32)
        m = np.zeros((h, w), dtype=bool)
        for j, kb in enumerate(kB):
            u, v = 2 + j * 3, 4
            xyz[v, u] = kb
            m[v, u] = True
        p.append(pi)
        quat.append(q)
        xyz_h.append(xyz)
        xyz_o.append(xyz.copy())
        mask_h.append(m)
        mask_o.append(m.copy())
        vis_h.append(True)
        vis_o.append(True)
    return {
        "p": np.stack(p),
        "quat": np.stack(quat),
        "xyz_h": np.stack(xyz_h),
        "xyz_o": np.stack(xyz_o),
        "mask_h": np.stack(mask_h),
        "mask_o": np.stack(mask_o),
        "vis_h": np.array(vis_h),
        "vis_o": np.array(vis_o),
    }


def _eval_pool(pool: dict[str, Any], K_O: np.ndarray, *, D_O: float, radius: float, n_geom: int) -> dict[str, Any]:
    n = pool["p"].shape[0]
    K = K_O.shape[0]
    er = np.full(n, np.nan)
    n_vis = np.zeros(n, dtype=np.int32)
    vis_any = pool["vis_h"] | pool["vis_o"]
    per_kp = np.zeros((n, K), dtype=bool)
    degen = np.zeros(n, dtype=bool)
    enough = np.zeros(n, dtype=bool)
    rms = np.full(n, np.nan)
    for i in range(n):
        R = _quat_to_R(pool["quat"][i])
        p = np.asarray(pool["p"][i], dtype=np.float64)
        kB = (R @ K_O.T).T + p
        vh = _visible_kp(pool["xyz_h"][i], pool["mask_h"][i], kB, radius)
        vo = _visible_kp(pool["xyz_o"][i], pool["mask_o"][i], kB, radius)
        v = vh | vo
        per_kp[i] = v
        n_vis[i] = int(v.sum())
        Xo, Xb = K_O[v], kB[v]
        nd = _nondegenerate(Xo, D_O)
        degen[i] = bool(n_vis[i] >= n_geom and not nd)
        enough[i] = bool(n_vis[i] >= n_geom and nd)
        if not enough[i]:
            continue
        Rh, _, rm = _kabsch(Xo, Xb)
        qh = _R_to_quat(Rh) if np.isfinite(Rh).all() else np.full(4, np.nan)
        er[i] = float(_e_R_deg(qh.reshape(1, 4), pool["quat"][i].reshape(1, 4))[0]) if np.isfinite(qh).all() else float("nan")
        rms[i] = rm
    vis_ok = vis_any
    p_any = float(np.mean(vis_any))
    p_enough = float(np.mean(enough[vis_ok])) if vis_ok.any() else 0.0
    return {
        "P_visible_any": p_any,
        "P_enough": p_enough,
        "n_frames": n,
        "median_n_vis": float(np.median(n_vis[vis_ok])) if vis_ok.any() else float("nan"),
        "per_keypoint_visibility": {KEYPOINT_NAMES[j]: float(np.mean(per_kp[vis_ok, j])) if vis_ok.any() else float("nan") for j in range(K)},
        "frac_degenerate": float(np.mean(degen[vis_ok])) if vis_ok.any() else float("nan"),
        "ori": _score_ori_arr(er),
        "median_rms": float(np.nanmedian(rms)),
        "n_vis": n_vis,
        "enough": enough,
        "vis_any": vis_any,
        "e_R": er,
    }


def _pattern(*, g0_cov: bool, g0_ori: bool) -> str:
    if not g0_cov:
        return "sparse_keypoints_unobservable"
    if not g0_ori:
        return "sparse_geometry_insufficient"
    return "sparse_oracle_orientation_supported"


def run_rtwx_o0g4(output: str | Path, config: RTWXO0G4Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G4Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    kp = load_cad_keypoints(repo)
    K_O = np.asarray(kp["K_O"], dtype=np.float64)
    D_O = load_D_O(repo)

    header = {
        "stage": "RTWX-O0G4",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "method": "oracle_sparse_CAD_keypoints_Kabsch",
        "seeds": list(cfg.seeds),
        "keypoints": kp,
        "N_geom": cfg.n_geom,
        "vis_radius_m": cfg.vis_radius_m,
        "unlocks_o1": False,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    per_seed = []
    all_er, all_enough, all_vis = [], [], []
    rng = np.random.default_rng(cfg.seeds[0] if cfg.seeds else 0)
    for seed in cfg.seeds:
        if cfg.backend == "numpy" or cfg.smoke:
            n = 48 if cfg.smoke else 120
            pool = _numpy_pool(n, K_O, rng)
        else:
            pool = load_g3r_pools(cfg.g3r_cache, (int(seed),), "test")
        aud = _eval_pool(pool, K_O, D_O=D_O, radius=cfg.vis_radius_m, n_geom=cfg.n_geom)
        cov_ok = bool(aud["P_visible_any"] >= P_ANY_SEED and aud["P_enough"] >= P_ENOUGH_SEED)
        print(
            f"[rtwx-o0g4] seed={seed} P_any={aud['P_visible_any']:.3f} P_enough={aud['P_enough']:.3f} "
            f"med={aud['ori']['median_e_R_deg']:.4f} p90={aud['ori']['p90_e_R_deg']:.4f}",
            flush=True,
        )
        per_seed.append({"seed": int(seed), "cov_ok": cov_ok, **{k: aud[k] for k in ("P_visible_any", "P_enough", "median_n_vis", "per_keypoint_visibility", "ori")}})
        all_er.append(aud["e_R"])
        all_enough.append(aud["enough"])
        all_vis.append(aud["vis_any"])

    er = np.concatenate(all_er)
    enough = np.concatenate(all_enough)
    vis = np.concatenate(all_vis)
    agg = _score_ori_arr(er[enough])
    p_any = float(np.mean(vis))
    p_enough = float(np.mean(enough[vis])) if vis.any() else 0.0
    g0_cov = bool(p_any >= P_ANY_AGG and p_enough >= P_ENOUGH_AGG and all(r["cov_ok"] for r in per_seed))
    g0_ori = bool(np.isfinite(agg["median_e_R_deg"]) and agg["median_e_R_deg"] <= MED_ER_MAX and agg["p90_e_R_deg"] <= P90_ER_MAX)
    pattern = _pattern(g0_cov=g0_cov, g0_ori=g0_ori)
    print(f"[rtwx-o0g4] pattern={pattern} P_enough={p_enough:.3f} med={agg['median_e_R_deg']:.4f} p90={agg['p90_e_R_deg']:.4f}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_coverage": {
            "ok": g0_cov,
            "P_visible_any_agg": p_any,
            "P_enough_agg": p_enough,
            "gate_any": P_ANY_AGG,
            "gate_enough": P_ENOUGH_AGG,
            "N_geom": cfg.n_geom,
            "vis_radius_m": cfg.vis_radius_m,
        },
        "G0_orientation": {"ok": g0_ori, **agg, "gate": {"med": MED_ER_MAX, "p90": P90_ER_MAX}},
        "mode_mass": {"P_eR_in_75_105": agg["frac_near_90"]},
        "per_seed": per_seed,
        "unlocks_o0g4r_prereg": pattern == "sparse_oracle_orientation_supported",
        "unlocks_o0c2": False,
        "unlocks_o1": False,
        "icp_not_primary": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "G0_coverage", "G0_orientation", "mode_mass", "unlocks_o0g4r_prereg")})
    return summary

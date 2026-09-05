"""RTWX-O0G5A: CAD descriptor observability (frozen FPFH, no training, no RANSAC)."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g2r import RTWXO0G2RConfig, collect_split
from .rtwx_o0g3_cache import xo_grid
from .rtwx_o0g3r import load_D_O
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG
from .rtwx_x0c import FORMAL_N_STEPS, _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G5A_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g5a.cad_descriptor_observability.v1"
SEEDS = (31601, 31602, 31603)
RGB_SIZE = 128
N_TEST_EP = 12
N_DESC = 64
N_CAD = 8192
MAX_OBS = 256
TOP_K = 5
FPFH_RADIUS = 0.02
FPFH_MAX_NN = 30
E_MED_MAX = 0.10
E_P90_MAX = 0.25
TOPK_RECALL_MIN = 0.75
HIT_THRESH = 0.05
AMBIG_FRAC_MAX = 0.15
PHI_TAU = 0.05
XO_TAU = 0.15
P_SUPPORT_AGG = 0.90
P_SUPPORT_SEED = 0.85
SPLIT_KEYS = (
    "p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o",
)


@dataclass(frozen=True)
class RTWXO0G5AConfig:
    output: str = "runs/rtwx_o0g5a"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_test_ep: int = N_TEST_EP
    n_steps: int = FORMAL_N_STEPS
    rgb_size: int = RGB_SIZE
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False


def _lock(cfg: RTWXO0G5AConfig) -> RTWXO0G5AConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, seeds=SEEDS, n_test_ep=N_TEST_EP, n_steps=FORMAL_N_STEPS, rgb_size=RGB_SIZE)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G5A must not write there")


def load_cad_cloud(robotwin_repo: str | Path, n_pts: int = N_CAD) -> np.ndarray:
    import open3d as o3d

    mesh_path = Path(robotwin_repo) / "assets/objects/021_cup/visual/base0.glb"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"empty CAD mesh {mesh_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_pts))
    return np.asarray(pcd.points, dtype=np.float64)


def _fpfh(pts: np.ndarray, radius: float = FPFH_RADIUS, max_nn: int = FPFH_MAX_NN) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )
    return np.asarray(feat.data, dtype=np.float64).T  # (n, 33)


def _nn_desc(query: np.ndarray, bank: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx k,), (dist k,) for each query row."""
    q = query.astype(np.float64)
    b = bank.astype(np.float64)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = qn @ bn.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    dist = 1.0 - np.take_along_axis(sim, idx, axis=1)
    return idx, dist


def _frame_obs(pool: dict[str, Any], fi: int, rng: np.random.Generator, max_obs: int) -> tuple[np.ndarray, np.ndarray, int]:
    p, q = pool["p"][fi], pool["quat"][fi]
    chunks_b, chunks_o = [], []
    for view in ("h", "o"):
        xyz = pool[f"xyz_{view}"][fi]
        mask = pool[f"mask_{view}"][fi]
        xo = xo_grid(xyz, mask, p, q)
        m = np.asarray(mask, bool) & np.isfinite(xyz).all(-1) & np.isfinite(xo).all(-1)
        if m.any():
            chunks_b.append(xyz[m].astype(np.float64))
            chunks_o.append(xo[m].astype(np.float64))
    if not chunks_b:
        return np.zeros((0, 3)), np.zeros((0, 3)), 0
    Xb = np.concatenate(chunks_b, 0)
    Xo = np.concatenate(chunks_o, 0)
    n_usable = int(Xb.shape[0])
    if n_usable > max_obs:
        idx = rng.choice(n_usable, max_obs, replace=False)
        Xb, Xo = Xb[idx], Xo[idx]
    return Xb, Xo, n_usable


def _spread_lambda2(xo: np.ndarray) -> float:
    if xo.shape[0] < 3:
        return float("nan")
    xc = xo - xo.mean(0)
    s = np.linalg.svd(xc, compute_uv=False)
    return float(s[1] ** 2) if s.size >= 2 else float("nan")


def _eval_frame(
    Xb: np.ndarray, Xo_gt: np.ndarray, cad_pts: np.ndarray, cad_fpfh: np.ndarray, *, D_O: float, rng: np.random.Generator,
) -> dict[str, Any] | None:
    if Xb.shape[0] < 3:
        return None
    try:
        obs_f = _fpfh(Xb)
    except Exception:
        return None
    idx1, _ = _nn_desc(obs_f, cad_fpfh, 1)
    idxk, _ = _nn_desc(obs_f, cad_fpfh, TOP_K)
    x_hat = cad_pts[idx1[:, 0]]
    err = np.linalg.norm(x_hat - Xo_gt, axis=1) / D_O
    topk_hit = np.array([
        float(np.min(np.linalg.norm(cad_pts[idxk[i]] - Xo_gt[i], axis=1)) / D_O < HIT_THRESH)
        for i in range(Xo_gt.shape[0])
    ])
    idx2, _ = _nn_desc(obs_f, cad_fpfh, 2)
    amb = []
    for i in range(Xo_gt.shape[0]):
        j1, j2 = int(idx2[i, 0]), int(idx2[i, 1])
        dphi = float(np.linalg.norm(cad_fpfh[j1] - cad_fpfh[j2]))
        dxo = float(np.linalg.norm(cad_pts[j1] - cad_pts[j2]) / D_O)
        amb.append(bool(dphi < PHI_TAU and dxo > XO_TAU))
    return {
        "e_norm": err,
        "topk_hit": topk_hit,
        "ambiguous": np.asarray(amb, bool),
        "lambda2_matched": _spread_lambda2(x_hat),
        "n_obs": int(Xb.shape[0]),
    }


def _g2r_cfg(cfg: RTWXO0G5AConfig, seed: int) -> RTWXO0G2RConfig:
    return RTWXO0G2RConfig(
        output=cfg.output,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        n_train_ep=0,
        n_val_ep=0,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        seed=seed,
        seed_attempts=cfg.seed_attempts,
        max_resample=cfg.max_resample,
        smoke=cfg.smoke,
        rgb_size=cfg.rgb_size,
        epochs=4,
    )


def _save_split(path: Path, name: str, pool: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": pool[k] for k in SPLIT_KEYS}
    payload[f"{name}_n_ep"] = np.array([pool["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([pool["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_split(path: Path, name: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in SPLIT_KEYS}
    p["rgb_h"] = p["rgb_h"].astype(np.uint8)
    p["rgb_o"] = p["rgb_o"].astype(np.uint8)
    for k in ("mask_h", "mask_o", "vis_h", "vis_o"):
        p[k] = p[k].astype(bool)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect_seed(cfg: RTWXO0G5AConfig, root: Path, seed: int, stop: list[str], *, cache_tag: str = "o0g5a") -> dict[str, Any]:
    path = root / f"cache_{cache_tag}_s{seed}_test.npz"
    hit = _load_split(path, "test")
    if hit is not None:
        print(f"[rtwx-{cache_tag}] seed={seed} test: cache", flush=True)
        return hit
    g2r = _g2r_cfg(cfg, seed)
    rng = np.random.default_rng(seed + 20_000)
    if cfg.backend == "numpy":
        from .rtwx_o0g1b import load_delta_O
        from .rtwx_o0g2r import _numpy_split

        delta_O = np.asarray(load_delta_O(cfg.robotwin_repo)["delta_O"], dtype=np.float64)
        pool = _numpy_split(g2r, "test", rng, delta_O)
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_raster_shader()
        _patch_curobo_planner(repo)
        pool = collect_split(g2r, split="test", rng=rng, stop=stop)
    _save_split(path, "test", pool)
    print(f"[rtwx-{cache_tag}] seed={seed} test: collected n={pool['p'].shape[0]}", flush=True)
    return pool


def _numpy_cad_pool(n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Cylinder-like surface for smoke."""
    z = rng.uniform(-0.04, 0.04, n_pts)
    ang = rng.uniform(0, 2 * np.pi, n_pts)
    r = 0.04 + rng.normal(0, 0.002, n_pts)
    x = r * np.cos(ang)
    y = 0.04 + r * np.sin(ang) * 0.2
    return np.stack([x, y, z], axis=1)


def _pattern(*, g0: bool, g1: bool, g2: bool) -> str:
    if not g0:
        return "cad_surface_support_failure"
    if not g1 or not g2:
        return "local_descriptor_ambiguous"
    return "cad_descriptor_observable"


def run_rtwx_o0g5a(output: str | Path, config: RTWXO0G5AConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G5AConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    D_O = load_D_O(repo)
    rng = np.random.default_rng(cfg.seeds[0] + 501)

    if cfg.backend == "numpy" or cfg.smoke:
        cad_pts = _numpy_cad_pool(min(N_CAD, 2048), np.random.default_rng(0))
    else:
        cad_pts = load_cad_cloud(repo, N_CAD)
    print(f"[rtwx-o0g5a] CAD points={cad_pts.shape[0]} computing FPFH...", flush=True)
    cad_fpfh = _fpfh(cad_pts)

    header = {
        "stage": "RTWX-O0G5A",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "method": "frozen_FPFH_CAD_retrieval",
        "rgb_size": cfg.rgb_size,
        "seeds": list(cfg.seeds),
        "D_O_m": D_O,
        "no_training": True,
        "no_ransac": True,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    stop: list[str] = []
    per_seed = []
    all_err, all_topk, all_amb = [], [], []
    support_flags = []
    for seed in cfg.seeds:
        pool = _collect_seed(cfg, root, int(seed), stop)
        if stop and cfg.backend != "numpy":
            raise RuntimeError(f"O0G5A collect stop: {stop}")
        n = pool["p"].shape[0]
        seed_err, seed_topk, seed_amb = [], [], []
        usable = []
        fr_rng = np.random.default_rng(int(seed) + 77)
        for fi in range(n):
            if not (pool["vis_h"][fi] or pool["vis_o"][fi]):
                continue
            Xb, Xo, n_usable = _frame_obs(pool, fi, fr_rng, MAX_OBS if not cfg.smoke else 64)
            usable.append(n_usable >= N_DESC)
            if n_usable < N_DESC:
                continue
            ev = _eval_frame(Xb, Xo, cad_pts, cad_fpfh, D_O=D_O, rng=fr_rng)
            if ev is None:
                continue
            seed_err.append(ev["e_norm"])
            seed_topk.append(ev["topk_hit"])
            seed_amb.append(ev["ambiguous"])
        vis_n = max(len(usable), 1)
        p_sup = float(np.mean(usable)) if usable else 0.0
        err = np.concatenate(seed_err) if seed_err else np.array([])
        topk = np.concatenate(seed_topk) if seed_topk else np.array([])
        amb = np.concatenate(seed_amb) if seed_amb else np.array([])
        med_e = float(np.median(err)) if err.size else float("nan")
        p90_e = float(np.percentile(err, 90)) if err.size else float("nan")
        top5 = float(np.mean(topk)) if topk.size else float("nan")
        f_amb = float(np.mean(amb)) if amb.size else float("nan")
        print(
            f"[rtwx-o0g5a] seed={seed} P_support={p_sup:.3f} med={med_e:.4f} p90={p90_e:.4f} top5={top5:.3f} amb={f_amb:.3f}",
            flush=True,
        )
        per_seed.append({
            "seed": int(seed),
            "P_support": p_sup,
            "support_ok": bool(p_sup >= P_SUPPORT_SEED),
            "median_e_norm": med_e,
            "p90_e_norm": p90_e,
            "top5_recall": top5,
            "frac_ambiguous": f_amb,
            "n_match_points": int(err.size),
        })
        support_flags.extend(usable)
        if err.size:
            all_err.append(err)
            all_topk.append(topk)
            all_amb.append(amb)

    err_all = np.concatenate(all_err) if all_err else np.array([])
    topk_all = np.concatenate(all_topk) if all_topk else np.array([])
    amb_all = np.concatenate(all_amb) if all_amb else np.array([])
    p_support = float(np.mean(support_flags)) if support_flags else 0.0
    med_e = float(np.median(err_all)) if err_all.size else float("nan")
    p90_e = float(np.percentile(err_all, 90)) if err_all.size else float("nan")
    top5 = float(np.mean(topk_all)) if topk_all.size else float("nan")
    f_amb = float(np.mean(amb_all)) if amb_all.size else float("nan")

    g0 = bool(p_support >= P_SUPPORT_AGG and all(s["support_ok"] for s in per_seed))
    g1 = bool(np.isfinite(f_amb) and f_amb < AMBIG_FRAC_MAX)
    g2 = bool(
        np.isfinite(med_e) and med_e <= E_MED_MAX and np.isfinite(p90_e) and p90_e <= E_P90_MAX and top5 >= TOPK_RECALL_MIN
    )
    pattern = _pattern(g0=g0, g1=g1, g2=g2)
    print(f"[rtwx-o0g5a] pattern={pattern} P_support={p_support:.3f} med={med_e:.4f} top5={top5:.3f}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_support": {
            "ok": g0,
            "P_support": p_support,
            "N_desc": N_DESC,
            "gate_agg": P_SUPPORT_AGG,
            "gate_seed": P_SUPPORT_SEED,
        },
        "G1_ambiguity": {"ok": g1, "frac_ambiguous": f_amb, "gate": AMBIG_FRAC_MAX},
        "G2_matching": {
            "ok": g2,
            "median_e_norm": med_e,
            "p90_e_norm": p90_e,
            "top5_recall": top5,
            "gate": {"med": E_MED_MAX, "p90": E_P90_MAX, "top5": TOPK_RECALL_MIN, "hit": HIT_THRESH},
        },
        "per_seed": per_seed,
        "unlocks_o0g5r_prereg": pattern == "cad_descriptor_observable",
        "unlocks_o0c2": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

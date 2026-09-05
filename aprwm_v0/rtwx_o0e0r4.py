"""RTWX-O0E0R4: multi-observation reference test + center-lambda diagnostic."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    N_OBS,
    SEED as O0E0_SEED,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    estimate_effective_pose,
    fibonacci_sphere,
    fuse_points,
    p_surf_from_cloud,
)
from .rtwx_o0e0p0 import (
    RTWXO0E0P0Config,
    RGB_SIZE,
    _capture_one,
    _set_static_pose,
    _stratified_betas,
    sample_pose,
)
from .rtwx_o0e0r1 import _axis_diag_pass, _axis_stats, _fused_clouds, estimate_axis_b2_at_center
from .rtwx_o0e0r2 import _cuda_gc, _load_p0_split, _load_seg_split
from .rtwx_o0e0r3 import RTWXO0E0R3Config, _collect_fresh_test
from .rtwx_o0g1b import load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0q0 import _robot_q
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R4_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r4.multiview_reference.v1"
SEED = 42601
MV_TEST_SEED = 37604
R3_DIAG_SEED = 37603
MV_Q_SEED = 42602
N_TEST_MV = 200
K_VALUES = (1, 2, 4, 8)
K_MAX = 8
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
S0_TRAIN_SEED = O0E0_SEED
Q_PERTURB_STD = 0.08


@dataclass(frozen=True)
class RTWXO0E0R4Config:
    output: str = "runs/rtwx_o0e0r4"
    r3_cache: str = "runs/rtwx_o0e0r3"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    mv_test_seed: int = MV_TEST_SEED
    n_test_mv: int = N_TEST_MV
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0E0R4Config) -> RTWXO0E0R4Config:
    if cfg.smoke:
        return replace(cfg, n_test_mv=40, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R4 must not write there")


def _set_robot_q(env: Any, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64)
    n_l = q.size // 2
    env.robot.set_arm_joints(q[: n_l - 1], np.zeros(n_l - 1), "left")
    env.robot.set_arm_joints(q[n_l : q.size - 1], np.zeros(n_l - 1), "right")
    env.robot.set_gripper(float(q[n_l - 1]), "left")
    env.robot.set_gripper(float(q[-1]), "right")


def fuse_multiview(view_clouds: list[np.ndarray], k: int, *, rng: np.random.Generator) -> np.ndarray:
    chunks: list[np.ndarray] = []
    per = max(N_OBS // max(k, 1), 8)
    for P in view_clouds[:k]:
        P = np.asarray(P, dtype=np.float64)
        if P.shape[0] == 0:
            continue
        if P.shape[0] > per:
            P = P[rng.choice(P.shape[0], per, replace=False)]
        chunks.append(P)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    P = np.concatenate(chunks, 0)
    if P.shape[0] > N_OBS:
        P = P[rng.choice(P.shape[0], N_OBS, replace=False)]
    return P


def _make_q_table(q0: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.asarray(q0 + rng.normal(0.0, Q_PERTURB_STD, size=q0.shape), dtype=np.float64)
        for _ in range(K_MAX)
    ]


def _numpy_view_stack(n: int, seed: int, size: int = RGB_SIZE) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    betas = _stratified_betas(n)
    rng.shuffle(betas)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.28) ** 2
    rgb[disk] = (180, 90, 40)
    rows: dict[str, list] = {
        "rgb_h": [], "rgb_o": [], "xyz_h": [], "xyz_o": [],
        "p": [], "quat": [], "n": [],
    }
    for beta in betas:
        pose = sample_pose(rng, float(beta))
        base = pose["p"]
        v_rgb_h, v_rgb_o, v_xyz_h, v_xyz_o = [], [], [], []
        for vi in range(K_MAX):
            off = np.array([0.012 * vi, 0.006 * vi, 0.0])
            xyz = np.full((size, size, 3), np.nan, np.float32)
            pts = base + off + rng.normal(0, 0.002, size=(48, 3))
            for t, pt in enumerate(pts):
                r = int(size / 2 + (t % 8 - 4) * 3)
                c = int(size / 2 + (t // 8 - 3) * 3)
                if 0 <= r < size and 0 <= c < size:
                    xyz[r, c] = pt.astype(np.float32)
            v_rgb_h.append(rgb.copy())
            v_rgb_o.append(rgb.copy())
            v_xyz_h.append(xyz)
            v_xyz_o.append(xyz)
        rows["rgb_h"].append(np.stack(v_rgb_h))
        rows["rgb_o"].append(np.stack(v_rgb_o))
        rows["xyz_h"].append(np.stack(v_xyz_h))
        rows["xyz_o"].append(np.stack(v_xyz_o))
        rows["p"].append(pose["p"])
        rows["quat"].append(pose["quat"])
        rows["n"].append(pose["n"])
    return {
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "xyz_h": np.stack(rows["xyz_h"]),
        "xyz_o": np.stack(rows["xyz_o"]),
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "n": np.stack(rows["n"]),
    }


def collect_multiview_robotwin(cfg: RTWXO0E0R4Config, n: int, seed: int, stop: list[str]) -> dict[str, Any]:
    from .rtwx_o0 import TASK, _o0_args
    from .rtwx_o0d import _cup_ids
    from .rtwx_x0 import _setup_env

    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    rng = np.random.default_rng(seed)
    betas = _stratified_betas(n)
    rng.shuffle(betas)
    p0cfg = RTWXO0E0P0Config(robotwin_repo=str(repo))
    env = None
    rows: dict[str, list] = {
        "rgb_h": [], "rgb_o": [], "xyz_h": [], "xyz_o": [],
        "p": [], "quat": [], "n": [],
    }
    try:
        env = _setup_env(repo, TASK, seed, 32, args)
        env.check_success = lambda *a, **k: False
        cup_ids = _cup_ids(env)
        q0, _ = _robot_q(env)
        q_table = _make_q_table(q0, MV_Q_SEED)
        rep_h = rep_o = None
        for i, beta in enumerate(betas):
            if i % 25 == 0 or i + 1 == n:
                print(f"[rtwx-o0e0r4] mv collect {i}/{n}", flush=True)
            pose = sample_pose(rng, float(beta))
            _set_static_pose(env, pose["p"], pose["quat"])
            v_rgb_h, v_rgb_o, v_xyz_h, v_xyz_o = [], [], [], []
            for q in q_table:
                _set_robot_q(env, q)
                cap = _capture_one(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
                rep_h, rep_o = cap["rep_h"], cap["rep_o"]
                v_rgb_h.append(cap["rgb_h"])
                v_rgb_o.append(cap["rgb_o"])
                v_xyz_h.append(cap["xyz_h"])
                v_xyz_o.append(cap["xyz_o"])
            rows["rgb_h"].append(np.stack(v_rgb_h))
            rows["rgb_o"].append(np.stack(v_rgb_o))
            rows["xyz_h"].append(np.stack(v_xyz_h))
            rows["xyz_o"].append(np.stack(v_xyz_o))
            rows["p"].append(pose["p"])
            rows["quat"].append(pose["quat"])
            rows["n"].append(pose["n"])
        env.close()
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    return {
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "xyz_h": np.stack(rows["xyz_h"]),
        "xyz_o": np.stack(rows["xyz_o"]),
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "n": np.stack(rows["n"]),
        "q_table_seed": MV_Q_SEED,
    }


def _write_mv_cache(root: Path, pool: dict[str, Any]) -> None:
    d = root / "cache" / "mv_test"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "obs.npz",
        rgb_h=pool["rgb_h"],
        rgb_o=pool["rgb_o"],
        xyz_h=pool["xyz_h"],
        xyz_o=pool["xyz_o"],
    )
    np.savez_compressed(
        d / "gt.npz",
        p_gt=pool["p"],
        quat_gt=pool["quat"],
        n_gt=pool["n"],
    )


def _load_mv_cache(root: Path) -> dict[str, Any]:
    obs = np.load(root / "cache" / "mv_test" / "obs.npz")
    gt = np.load(root / "cache" / "mv_test" / "gt.npz")
    return {
        "rgb_h": obs["rgb_h"],
        "rgb_o": obs["rgb_o"],
        "xyz_h": obs["xyz_h"],
        "xyz_o": obs["xyz_o"],
        "p_gt": np.asarray(gt["p_gt"]),
        "quat_gt": np.asarray(gt["quat_gt"]),
        "n_gt": np.asarray(gt["n_gt"]),
    }


def _collect_mv(cfg: RTWXO0E0R4Config, root: Path) -> dict[str, Any]:
    cache_obs = root / "cache" / "mv_test" / "obs.npz"
    if cache_obs.is_file():
        return _load_mv_cache(root)
    if cfg.smoke:
        pool = _numpy_view_stack(cfg.n_test_mv, cfg.mv_test_seed)
    else:
        stop: list[str] = []
        pool = collect_multiview_robotwin(cfg, cfg.n_test_mv, cfg.mv_test_seed, stop)
        if stop:
            raise RuntimeError(f"mv collect stop: {stop}")
    _write_mv_cache(root, pool)
    return _load_mv_cache(root)


def _s0_per_view_clouds(mv: dict[str, Any], s0: Any, *, seed: int) -> list[list[np.ndarray]]:
    n, kmax = mv["rgb_h"].shape[0], mv["rgb_h"].shape[1]
    rh = mv["rgb_h"].reshape(n * kmax, *mv["rgb_h"].shape[2:])
    ro = mv["rgb_o"].reshape(n * kmax, *mv["rgb_o"].shape[2:])
    mh = _predict_masks(s0, rh).reshape(n, kmax, *rh.shape[1:3])
    mo = _predict_masks(s0, ro).reshape(n, kmax, *ro.shape[1:3])
    rng = np.random.default_rng(seed)
    out: list[list[np.ndarray]] = []
    for i in range(n):
        views = [
            fuse_points(mv["xyz_h"][i, vi], mh[i, vi], mv["xyz_o"][i, vi], mo[i, vi], rng=rng)
            for vi in range(kmax)
        ]
        out.append(views)
    return out


def _center_lambda_diagnostic(
    clouds: list[np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    by_lam: dict[str, list[float]] = {str(lam): [] for lam in LAMBDA_GRID}
    e_center: list[float] = []
    for i in range(n_frames):
        obs_i = {"fused_points_B": clouds[i], "mask_h": np.ones((8, 8), bool), "mask_o": np.ones((8, 8), bool)}
        _, n_hat, _ = estimate_effective_pose(
            obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2", n_const=n_const,
        )
        p_surf = p_surf_from_cloud(clouds[i])
        p_ref = p_surf - float(delta_y) * n_hat if np.isfinite(p_surf).all() else p_surf
        for lam in LAMBDA_GRID:
            p_lam = (1.0 - lam) * p_ref + lam * gt_te["p_gt"][i]
            n_l, _ = estimate_axis_b2_at_center(clouds[i], p_lam, tree, sphere=sphere)
            by_lam[str(lam)].append(e_axis_deg(n_l, gt_te["n_gt"][i]))
            if lam == 0.0:
                e_center.append(float(np.linalg.norm(p_ref - gt_te["p_gt"][i])))
    out_lam: dict[str, Any] = {}
    for lam in LAMBDA_GRID:
        e = np.asarray(by_lam[str(lam)], dtype=np.float64)
        out_lam[str(lam)] = {**_axis_stats(e), "lambda": lam}
    e_c = np.asarray(e_center, dtype=np.float64)
    return {
        "diagnostic_only": True,
        "r3_seed": R3_DIAG_SEED,
        "e_center_ref_median_cm": float(np.nanmedian(e_c) * 100.0),
        "e_center_ref_p90_cm": float(np.nanpercentile(e_c, 90) * 100.0),
        "by_lambda": out_lam,
    }


def _formal_k(
    fused: list[np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_ax, p_hats = [], []
    for i in range(n_frames):
        obs_i = {"fused_points_B": fused[i], "mask_h": np.ones((8, 8), bool), "mask_o": np.ones((8, 8), bool)}
        p_hat, n_hat, _ = estimate_effective_pose(
            obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2", n_const=n_const,
        )
        e_ax.append(e_axis_deg(n_hat, gt_te["n_gt"][i]))
        p_hats.append(p_hat)
    e_ax = np.asarray(e_ax, dtype=np.float64)
    pos = _score_pose(np.stack(p_hats), gt_te["p_gt"])
    ax = _axis_stats(e_ax)
    g1 = _axis_diag_pass(ax["median_e_axis_deg"], ax["p90_e_axis_deg"])
    g2 = bool(pos["E_p"] <= EP_MAX and pos["median_ep_m"] <= MED_MAX)
    return {
        "axis": ax,
        "G1": g1,
        "G2": g2,
        "E_p": pos["E_p"],
        "median_ep_m": pos["median_ep_m"],
        "median_ep_cm": float(pos["median_ep_m"] * 100.0),
        "formal_pass": bool(g1 and g2),
    }


def _pattern(*, any_k_pass: bool, k8_g1: bool, k1_g1: bool) -> tuple[str, list[str]]:
    tags: list[str] = []
    if any_k_pass:
        return "multiview_reference_supported", tags
    if not k8_g1:
        tags.append("multiview_reference_insufficient")
    if not k1_g1:
        tags.append("single_observation_reference_coupled")
    return "multiview_reference_insufficient", tags


def run_rtwx_o0e0r4(output: str | Path, config: RTWXO0E0R4Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0R4Config(output=str(output)))
    root = Path(output).resolve()
    r3_root = Path(cfg.r3_cache).resolve()
    natural_root = Path(cfg.natural_cache).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r4] mv_seed={cfg.mv_test_seed} r3_diag_seed={R3_DIAG_SEED}", flush=True)

    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_y = float(np.asarray(prior["delta_O"], dtype=np.float64)[1])

    repo = Path(cfg.robotwin_repo)
    cad_glb = repo / "assets/objects/021_cup/visual/base0.glb"
    if not cad_glb.is_file():
        hs = np.linspace(0, 0.088, 64)
        cad_pts = np.array(
            [[0.02 + 0.22 * (h / 0.088) * np.cos(th), h, 0.02 + 0.22 * (h / 0.088) * np.sin(th)]
             for h in hs for th in np.linspace(0, 2 * np.pi, 32, endpoint=False)],
            dtype=np.float64,
        )
    else:
        cad_pts = _fps(load_scaled_cad(str(repo), 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
    from scipy.spatial import cKDTree

    tree = cKDTree(cad_hr_profile(cad_pts))
    sphere = fibonacci_sphere(K_SPHERE, SPHERE_SEED)

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "mv_test_seed": cfg.mv_test_seed,
        "r3_cache_diagnostic_only": str(r3_root),
        "r3_diagnostic_seed": R3_DIAG_SEED,
        "K_values": list(K_VALUES),
        "b2_frozen": True,
        "S0_natural": True,
    }
    _write_json(root / "header.json", header)

    print("[rtwx-o0e0r4] train S0", flush=True)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )
    _cuda_gc()

    _, gt_tr = _load_p0_split(p0_root, "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    print("[rtwx-o0e0r4] center-lambda diagnostic on R3 cache", flush=True)
    r3_cfg = RTWXO0E0R3Config(
        output=str(r3_root),
        p0_cache=str(p0_root),
        natural_cache=str(natural_root),
        robotwin_repo=cfg.robotwin_repo,
        fresh_test_seed=R3_DIAG_SEED,
        n_test_fresh=cfg.n_test_mv,
        smoke=cfg.smoke,
    )
    r3_pool = _collect_fresh_test(r3_cfg, r3_root)
    r3_obs = {k: r3_pool[k] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o")}
    r3_masks = {
        "mask_h": _predict_masks(s0, r3_obs["rgb_h"]),
        "mask_o": _predict_masks(s0, r3_obs["rgb_o"]),
    }
    r3_gt = {"p_gt": np.asarray(r3_pool["p"]), "n_gt": np.asarray(r3_pool["n"])}
    r3_clouds = _fused_clouds(r3_obs, r3_masks, seed=cfg.seed)
    center_diag = _center_lambda_diagnostic(
        r3_clouds, r3_gt, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const,
    )
    print(
        f"[rtwx-o0e0r4] lambda0 med={center_diag['by_lambda']['0.0']['median_e_axis_deg']:.1f}° "
        f"lam1={center_diag['by_lambda']['1.0']['median_e_axis_deg']:.2f}° "
        f"e_center_med={center_diag['e_center_ref_median_cm']:.2f}cm",
        flush=True,
    )

    print("[rtwx-o0e0r4] collect/load multiview test", flush=True)
    mv = _collect_mv(cfg, root)
    gt_mv = {"p_gt": mv["p_gt"], "n_gt": mv["n_gt"]}
    n_inst = gt_mv["p_gt"].shape[0]
    view_clouds = _s0_per_view_clouds(mv, s0, seed=cfg.seed + 3)
    rng = np.random.default_rng(cfg.seed + 7)
    per_k: dict[str, Any] = {}
    for k in K_VALUES:
        fused = [fuse_multiview(view_clouds[i], k, rng=rng) for i in range(n_inst)]
        per_k[str(k)] = _formal_k(fused, gt_mv, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const)
        print(
            f"[rtwx-o0e0r4] K={k} med_axis={per_k[str(k)]['axis']['median_e_axis_deg']:.2f}° "
            f"med_ep={per_k[str(k)]['median_ep_cm']:.2f}cm pass={per_k[str(k)]['formal_pass']}",
            flush=True,
        )

    any_pass = any(per_k[str(k)]["formal_pass"] for k in K_VALUES)
    pattern, tags = _pattern(
        any_k_pass=any_pass,
        k8_g1=per_k["8"]["G1"],
        k1_g1=per_k["1"]["G1"],
    )

    result = {
        "header": header,
        "pattern": pattern,
        "patterns": tags,
        "center_lambda_diagnostic": center_diag,
        "multiview_formal": per_k,
        "unlocks_o0e1_prereg": pattern == "multiview_reference_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0r4] pattern={pattern}", flush=True)
    return result

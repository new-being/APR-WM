"""RTWX-O0HYB0: prior-anchored hybrid one-step state update (absolute + relative router)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry.center_axis_inference import VisObs, run_a1_center_first, visibility_center_fused
from .geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, estimate_relative_rigid
from .geometry.relative_validity import (
    calibrate_threshold_youden,
    relative_safe_label,
    symmetric_support_score,
)
from .geometry.visibility_reference import AxialVisibilityReference, camera_center_world
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
    fuse_points,
)
from .rtwx_o0e0p0 import (
    RGB_SIZE,
    RTWXO0E0P0Config,
    P_XY,
    P_Z,
    R0_from_n,
    _capture_one,
    _set_static_pose,
    n_from_tilt,
    sample_pose,
)
from .rtwx_o0e0r1 import _axis_stats, _ep_stats
from .rtwx_o0e0r2 import _load_p0_split, _load_seg_split
from .rtwx_o0e0r4 import _set_robot_q
from .rtwx_o0e0r5 import S0_TRAIN_SEED
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import EP_MAX, MED_MAX, _predict_masks, _train_unet
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0q0 import _R_to_quat, _robot_q
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0rel0 import (
    DELTA_BETA_DEG,
    DELTA_TRANS_M,
    ICP_CFG,
    TRANS_DIR_CATALOG,
    _apply_object_motion,
    _resolve_apr,
    _yaw_octant_gamma,
    assert_obs_no_gt,
    voxel_once,
)
from .rtwx_o0v import CAM_HEAD, CAM_OBS, _get_cam
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0HYB0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0hyb0.hybrid_state_update.v1"
SEED = 46601
CAL_SEED = 37607
FORMAL_SEED = 37608
CAM_SMALL_Q_SEED = 37609
CAM_LARGE_Q_SEED = 37610
N_PER_REGIME = 100
BETA0_DEG = (0.0, 2.0, 5.0)
REGIMES = ("h00", "h01", "h10", "h11")
SPLITS = ("cal", "formal")
PRIOR_N_MED = 5.0
PRIOR_N_P90 = 10.0
ROUTER_AUROC_MIN = 0.55
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n"})
_APR_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RTWXO0HYB0Config:
    output: str = "runs/rtwx_o0hyb0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    cal_seed: int = CAL_SEED
    formal_seed: int = FORMAL_SEED
    n_per_regime: int = N_PER_REGIME
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0HYB0Config) -> RTWXO0HYB0Config:
    if cfg.smoke:
        return replace(cfg, n_per_regime=8, epochs_unet=4, backend="numpy")
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0HYB0 must not write there")


def _sample_upright_pose(rng: np.random.Generator, beta_deg: float) -> dict[str, np.ndarray]:
    from .rtwx_o0e0 import n_from_R
    from .rtwx_o0q0 import _R_y

    alpha = float(rng.uniform(0.0, 2.0 * np.pi))
    gamma = _yaw_octant_gamma(rng)
    n = n_from_tilt(beta_deg, alpha)
    r = R0_from_n(n) @ _R_y(np.degrees(gamma))
    p = np.array([rng.uniform(*P_XY[0]), rng.uniform(*P_XY[1]), P_Z], dtype=np.float64)
    return {"p": p, "R": r, "n": n_from_R(r), "quat": _R_to_quat(r), "beta_deg": float(beta_deg)}


def _regime_flags(regime: str) -> tuple[bool, bool]:
    cam_large = regime in ("h10", "h11")
    obj_move = regime in ("h01", "h11")
    return cam_large, obj_move


def _camera_q1(q0: np.ndarray, catalog_idx: int, *, large: bool) -> np.ndarray:
    base = CAM_LARGE_Q_SEED if large else CAM_SMALL_Q_SEED
    rng = np.random.default_rng(base + int(catalog_idx))
    delta = np.zeros_like(q0)
    if large:
        theta = np.radians([20.0, 30.0][catalog_idx % 2])
        t_scale = [4.0, 6.0][(catalog_idx // 2) % 2]
    else:
        theta = np.radians([0.0, 5.0][catalog_idx % 2])
        t_scale = [1.0, 2.0][(catalog_idx // 2) % 2]
    joint = catalog_idx % max(q0.size - 2, 1)
    delta[joint] = theta
    if joint + 1 < q0.size:
        delta[joint + 1] = t_scale * 2.0
    delta += rng.normal(0.0, 0.008, size=q0.shape)
    return np.asarray(q0 + delta, dtype=np.float64)


def generate_regime_specs(regime: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + {"h00": 0, "h01": 1, "h10": 2, "h11": 3}[regime])
    betas = np.array(list(BETA0_DEG) * (n // len(BETA0_DEG) + 1), dtype=np.float64)[:n]
    rng.shuffle(betas)
    _, obj_move = _regime_flags(regime)
    specs: list[dict[str, Any]] = []
    for j, beta in enumerate(betas):
        pose0 = _sample_upright_pose(rng, float(beta))
        r0 = np.asarray(pose0["R"], dtype=np.float64)
        p0 = np.asarray(pose0["p"], dtype=np.float64)
        n0 = np.asarray(pose0["n"], dtype=np.float64)
        if obj_move:
            db = float(rng.choice(DELTA_BETA_DEG))
            tm = float(rng.choice(DELTA_TRANS_M))
            td = TRANS_DIR_CATALOG[j % len(TRANS_DIR_CATALOG)]
            dg = _yaw_octant_gamma(rng)
            r1, p1, n1, q1 = _apply_object_motion(
                r0, p0, n0, delta_beta_deg=db, delta_gamma=dg, trans_dir=td, trans_mag=tm, rng=rng,
            )
        else:
            r1, p1, n1, q1 = r0.copy(), p0.copy(), n0.copy(), pose0["quat"].copy()
            db, tm = 0.0, 0.0
        specs.append({
            "pair_id": j,
            "regime": regime,
            "beta0_deg": float(beta),
            "delta_beta_deg": db,
            "delta_trans_m": tm,
            "cam_catalog_idx": j % 12,
            "p0": p0,
            "quat0": pose0["quat"],
            "R0": r0,
            "n0": n0,
            "p1": p1,
            "quat1": q1,
            "R1": r1,
            "n1": n1,
        })
    return specs


def _extrinsic_pack(cam: Any) -> np.ndarray:
    return np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)


def _capture_frame(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
    cap = _capture_one(env, size, cup_ids, rep_h, rep_o)
    cap["E_h"] = _extrinsic_pack(_get_cam(env, CAM_HEAD))
    cap["E_o"] = _extrinsic_pack(_get_cam(env, CAM_OBS))
    cap["cam_h"] = camera_center_world(cap["E_h"])
    cap["cam_o"] = camera_center_world(cap["E_o"])
    return cap


def collect_regime_robotwin(
    cfg: RTWXO0HYB0Config,
    regime: str,
    specs: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rtwx_o0 import TASK, _o0_args
    from .rtwx_o0d import _cup_ids
    from .rtwx_x0 import _setup_env

    cam_large, _ = _regime_flags(regime)
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    p0cfg = RTWXO0E0P0Config(robotwin_repo=str(repo), rgb_size=RGB_SIZE)
    env = None
    obs_rows: dict[str, list] = {k: [] for k in (
        "rgb0_h", "rgb0_o", "xyz0_h", "xyz0_o", "rgb1_h", "rgb1_o", "xyz1_h", "xyz1_o",
        "vis0_h", "vis0_o", "vis1_h", "vis1_o",
    )}
    gt_rows: dict[str, list] = {k: [] for k in (
        "p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1",
        "mask0_h", "mask0_o", "mask1_h", "mask1_o", "pair_id",
    )}
    try:
        env = _setup_env(repo, TASK, seed, 32, args)
        env.check_success = lambda *a, **k: False
        cup_ids = _cup_ids(env)
        q0, _ = _robot_q(env)
        rep_h = rep_o = None
        for i, sp in enumerate(specs):
            if i % 25 == 0 or i + 1 == len(specs):
                print(f"[rtwx-o0hyb0] {regime} collect {i}/{len(specs)}", flush=True)
            q1 = _camera_q1(q0, int(sp["cam_catalog_idx"]), large=cam_large)
            _set_static_pose(env, sp["p0"], sp["quat0"])
            _set_robot_q(env, q0)
            cap0 = _capture_frame(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap0["rep_h"], cap0["rep_o"]
            _set_static_pose(env, sp["p1"], sp["quat1"])
            _set_robot_q(env, q1)
            cap1 = _capture_frame(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap1["rep_h"], cap1["rep_o"]
            for prefix, cap in (("0", cap0), ("1", cap1)):
                obs_rows[f"rgb{prefix}_h"].append(cap["rgb_h"])
                obs_rows[f"rgb{prefix}_o"].append(cap["rgb_o"])
                obs_rows[f"xyz{prefix}_h"].append(cap["xyz_h"])
                obs_rows[f"xyz{prefix}_o"].append(cap["xyz_o"])
                obs_rows[f"vis{prefix}_h"].append(cap["vis_h"])
                obs_rows[f"vis{prefix}_o"].append(cap["vis_o"])
            for k in ("p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1"):
                gt_rows[k].append(sp[k])
            gt_rows["mask0_h"].append(cap0["mask_h"])
            gt_rows["mask0_o"].append(cap0["mask_o"])
            gt_rows["mask1_h"].append(cap1["mask_h"])
            gt_rows["mask1_o"].append(cap1["mask_o"])
            gt_rows["pair_id"].append(sp["pair_id"])
        env.close()
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    obs = {k: (np.stack(v) if k.startswith(("rgb", "xyz")) else np.asarray(v, bool)) for k, v in obs_rows.items()}
    gt: dict[str, Any] = {}
    for k, v in gt_rows.items():
        if k == "pair_id":
            gt[k] = np.asarray(v, np.int32)
        elif k.startswith(("mask",)):
            gt[k] = np.stack(v)
        else:
            gt[k] = np.stack(v)
    assert_obs_no_gt(obs)
    return obs, gt


def collect_regime_numpy(regime: str, specs: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rtwx_o0rel0 import _numpy_disk_cloud

    cam_large, obj_move = _regime_flags(regime)
    cam_shift = np.array([0.04, 0.0, 0.0]) if cam_large else np.array([0.008, 0.0, 0.0])
    rng = np.random.default_rng(seed + hash(regime) % 10000)
    obs_rows: dict[str, list] = {k: [] for k in (
        "rgb0_h", "rgb0_o", "xyz0_h", "xyz0_o", "rgb1_h", "rgb1_o", "xyz1_h", "xyz1_o",
        "vis0_h", "vis0_o", "vis1_h", "vis1_o",
    )}
    gt_rows: dict[str, list] = {k: [] for k in (
        "p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1",
        "mask0_h", "mask0_o", "mask1_h", "mask1_o", "pair_id",
    )}
    for sp in specs:
        rgb0_h, m0h, xyz0_h = _numpy_disk_cloud(sp["p0"], RGB_SIZE, rng)
        rgb0_o, m0o, xyz0_o = _numpy_disk_cloud(sp["p0"], RGB_SIZE, rng, shift=np.array([0.01, 0.0, 0.0]))
        p1_vis = sp["p1"] if obj_move else sp["p0"]
        rgb1_h, m1h, xyz1_h = _numpy_disk_cloud(p1_vis, RGB_SIZE, rng, shift=cam_shift)
        rgb1_o, m1o, xyz1_o = _numpy_disk_cloud(p1_vis, RGB_SIZE, rng, shift=cam_shift + np.array([0.01, 0.0, 0.0]))
        for k, v in (
            ("rgb0_h", rgb0_h), ("rgb0_o", rgb0_o), ("xyz0_h", xyz0_h), ("xyz0_o", xyz0_o),
            ("rgb1_h", rgb1_h), ("rgb1_o", rgb1_o), ("xyz1_h", xyz1_h), ("xyz1_o", xyz1_o),
            ("vis0_h", True), ("vis0_o", True), ("vis1_h", True), ("vis1_o", True),
        ):
            obs_rows[k].append(v)
        gt_rows["mask0_h"].append(m0h)
        gt_rows["mask0_o"].append(m0o)
        gt_rows["mask1_h"].append(m1h)
        gt_rows["mask1_o"].append(m1o)
        for k in ("p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1"):
            gt_rows[k].append(sp[k])
        gt_rows["pair_id"].append(sp["pair_id"])
    obs = {k: np.stack(v) if k.startswith(("rgb", "xyz")) else np.asarray(v, bool) for k, v in obs_rows.items()}
    gt: dict[str, Any] = {}
    for k, v in gt_rows.items():
        gt[k] = np.asarray(v, np.int32) if k == "pair_id" else (np.stack(v) if k.startswith("mask") else np.stack(v))
    assert_obs_no_gt(obs)
    return obs, gt


def _write_cache(root: Path, split: str, regime: str, obs: dict[str, Any], gt: dict[str, Any]) -> None:
    d = root / "cache" / split / regime
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "obs.npz", **obs)
    np.savez_compressed(d / "gt.npz", **gt)


def _load_cache(root: Path, split: str, regime: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = dict(np.load(root / "cache" / split / regime / "obs.npz"))
    gt = dict(np.load(root / "cache" / split / regime / "gt.npz"))
    return obs, gt


def _build_s0_cloud(obs: dict[str, Any], s0: Any, pair_i: int, frame_t: int, *, rng: np.random.Generator) -> np.ndarray:
    rh = obs[f"rgb{frame_t}_h"][pair_i]
    ro = obs[f"rgb{frame_t}_o"][pair_i]
    mh = _predict_masks(s0, rh[None])[0] > 0.5
    mo = _predict_masks(s0, ro[None])[0] > 0.5
    return fuse_points(obs[f"xyz{frame_t}_h"][pair_i], mh, obs[f"xyz{frame_t}_o"][pair_i], mo, rng=rng)


def _vis_obs_frame(obs: dict[str, Any], cloud: np.ndarray, pair_i: int, frame_t: int) -> VisObs:
    if f"E{frame_t}_h" in obs:
        return VisObs(
            fused_cloud=cloud,
            c_h=cloud.mean(0) if cloud.size else np.zeros(3),
            c_o=cloud.mean(0) if cloud.size else np.zeros(3),
            cam_h=camera_center_world(obs[f"E{frame_t}_h"][pair_i]),
            cam_o=camera_center_world(obs[f"E{frame_t}_o"][pair_i]),
        )
    return VisObs(
        fused_cloud=cloud,
        c_h=cloud.mean(0) if cloud.size else np.zeros(3),
        c_o=cloud.mean(0) if cloud.size else np.zeros(3),
        cam_h=np.array([0.0, -0.3, 1.0]),
        cam_o=np.array([0.4, -0.2, 0.9]),
    )


def _science_pass(axis_deg: np.ndarray, pos_m: np.ndarray) -> bool:
    ax = _axis_stats(axis_deg)
    ps = _ep_stats(pos_m)
    ep = float(np.nanmean(pos_m))
    return bool(
        ax["median_e_axis_deg"] <= MED_ER_MAX
        and ax["p90_e_axis_deg"] <= P90_ER_MAX
        and ep <= EP_MAX
        and ps["median_ep_m"] <= MED_MAX
    )


def _oracle_pick(
    rel_safe: bool,
    abs_safe: bool,
    e_rel: float,
    e_abs: float,
) -> Literal["rel", "abs"]:
    if rel_safe and not abs_safe:
        return "rel"
    if abs_safe and not rel_safe:
        return "abs"
    return "rel" if e_rel <= e_abs else "abs"


def _eval_split_regime(
    obs: dict[str, Any],
    gt: dict[str, Any],
    *,
    s0: Any,
    lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
    n_const: np.ndarray,
    d_o: float,
    rng: np.random.Generator,
    tau: float | None = None,
) -> dict[str, Any]:
    n_inst = gt["p0"].shape[0]
    rows: list[dict[str, Any]] = []
    for i in range(n_inst):
        c0 = _build_s0_cloud(obs, s0, i, 0, rng=rng)
        c1 = _build_s0_cloud(obs, s0, i, 1, rng=rng)
        vo0 = _vis_obs_frame(obs, c0, i, 0)
        vo1 = _vis_obs_frame(obs, c1, i, 1)
        p0_hat, _, _, _ = visibility_center_fused(
            n_const, vo0.c_h, vo0.c_o, vo0.cam_h, vo0.cam_o, lut,
        )
        n0_hat = n_const.copy()
        a1 = run_a1_center_first(vo1, n_const, lut, b2)
        p1_abs, n1_abs = a1.p1, a1.n1
        e_abs_n = float(e_axis_deg(n1_abs, gt["n1"][i]))
        e_abs_p = float(np.linalg.norm(p1_abs - gt["p1"][i]))
        icp = estimate_relative_rigid(voxel_once(c0, d_o), voxel_once(c1, d_o), d_o, ICP_CFG)
        r_hat, t_hat = icp.R, icp.t
        n1_rel = r_hat @ n0_hat
        n1_rel = n1_rel / max(np.linalg.norm(n1_rel), 1e-12)
        p1_rel = r_hat @ p0_hat + t_hat
        e_rel_n = float(e_axis_deg(n1_rel, gt["n1"][i]))
        e_rel_p = float(np.linalg.norm(p1_rel - gt["p1"][i]))
        e_n0 = float(e_axis_deg(n0_hat, gt["n0"][i]))
        e_p0 = float(np.linalg.norm(p0_hat - gt["p0"][i]))
        q_rel = symmetric_support_score(c0, c1, r_hat, t_hat, d_o)
        rel_safe = relative_safe_label(e_rel_n, e_rel_p)
        abs_safe = relative_safe_label(e_abs_n, e_abs_p)
        pick = _oracle_pick(rel_safe, abs_safe, e_rel_n, e_abs_n)
        if pick == "rel":
            o_n, o_p = e_rel_n, e_rel_p
        else:
            o_n, o_p = e_abs_n, e_abs_p
        if tau is not None:
            use_rel = bool(q_rel >= tau)
            h_n = e_rel_n if use_rel else e_abs_n
            h_p = e_rel_p if use_rel else e_abs_p
        else:
            h_n, h_p = float("nan"), float("nan")
            use_rel = False
        rows.append({
            "pair_id": int(gt["pair_id"][i]),
            "e_n0_deg": e_n0,
            "e_p0_m": e_p0,
            "e_delta_n_deg": float(e_axis_deg(r_hat @ gt["n0"][i], gt["n1"][i])),
            "e_rel_n_deg": e_rel_n,
            "e_rel_p_m": e_rel_p,
            "e_abs_n_deg": e_abs_n,
            "e_abs_p_m": e_abs_p,
            "e_oracle_n_deg": o_n,
            "e_oracle_p_m": o_p,
            "e_hyb_n_deg": h_n,
            "e_hyb_p_m": h_p,
            "q_rel": q_rel,
            "rel_safe": rel_safe,
            "abs_safe": abs_safe,
            "hyb_used_rel": use_rel,
            "icp_valid": bool(icp.valid),
        })
    return {"per_pair": rows}


def _regime_table(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    e_n = np.asarray([r[f"e_{prefix}_n_deg"] for r in rows], dtype=np.float64)
    e_p = np.asarray([r[f"e_{prefix}_p_m"] for r in rows], dtype=np.float64)
    return {
        "axis": _axis_stats(e_n),
        "position": {**_ep_stats(e_p), "E_p": float(np.nanmean(e_p))},
        "pass": _science_pass(e_n, e_p),
    }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    tpr = tps / max(y.sum(), 1)
    fpr = fps / max((~y).sum(), 1)
    return float(np.trapz(tpr, fpr))


def _pattern(
    *,
    l1_ok: bool,
    l2_ok: bool,
    l3_ok: bool,
    router_auroc: float,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    if not l1_ok:
        return "prior_anchor_failure", tags
    if not l2_ok:
        return "hybrid_complementarity_insufficient", tags
    if l3_ok:
        return "hybrid_state_update_supported", tags
    if np.isfinite(router_auroc) and router_auroc < ROUTER_AUROC_MIN:
        return "hybrid_router_failure", tags
    return "hybrid_state_update_insufficient", tags


def run_rtwx_o0hyb0(output: str | Path, config: RTWXO0HYB0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0HYB0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    natural_root = _resolve_apr(cfg.natural_cache)
    p0_root = _resolve_apr(cfg.p0_cache)
    r5_root = _resolve_apr(cfg.r5_run)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    split_seeds = {"cal": cfg.cal_seed, "formal": cfg.formal_seed}
    for split in SPLITS:
        for regime in REGIMES:
            cache_obs = root / "cache" / split / regime / "obs.npz"
            if not cache_obs.is_file():
                specs = generate_regime_specs(regime, cfg.n_per_regime, split_seeds[split])
                print(f"[rtwx-o0hyb0] collect {split}/{regime} backend={cfg.backend}", flush=True)
                if cfg.backend == "numpy":
                    obs, gt = collect_regime_numpy(regime, specs, seed=split_seeds[split])
                else:
                    obs, gt = collect_regime_robotwin(cfg, regime, specs, seed=split_seeds[split] + hash(regime) % 997)
                _write_cache(root, split, regime, obs, gt)

    repo = Path(cfg.robotwin_repo)
    if not repo.is_absolute():
        repo = repo.resolve()
    d_o = load_D_O(str(repo) if repo.is_dir() else "/root/RoboTwin")

    lut_path = r5_root / "visibility_lut.npz"
    if not lut_path.is_file():
        raise FileNotFoundError(f"R5 visibility_lut missing: {lut_path}")
    lut = AxialVisibilityReference.load(lut_path)
    cad_glb = repo / "assets/objects/021_cup/visual/base0.glb"
    if cad_glb.is_file():
        cad_pts = _fps(load_scaled_cad(str(repo), 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
    else:
        hs = np.linspace(0, 0.088, 64)
        cad_pts = np.array(
            [[0.02 + 0.22 * (h / 0.088) * np.cos(th), h, 0.02 + 0.22 * (h / 0.088) * np.sin(th)]
             for h in hs for th in np.linspace(0, 2 * np.pi, 32, endpoint=False)],
            dtype=np.float64,
        )
    from scipy.spatial import cKDTree

    tree = cKDTree(cad_hr_profile(cad_pts))
    b2 = FrozenQuotientB2(tree, fibonacci_sphere(K_SPHERE, SPHERE_SEED))
    b2.assert_frozen()
    _, gt_tr = _load_p0_split(p0_root, "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    print("[rtwx-o0hyb0] train S0", flush=True)
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )

    rng = np.random.default_rng(cfg.seed)
    cal_rows: list[dict[str, Any]] = []
    formal_by_regime: dict[str, dict[str, Any]] = {}

    for regime in REGIMES:
        obs_c, gt_c = _load_cache(root, "cal", regime)
        res_c = _eval_split_regime(obs_c, gt_c, s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o, rng=rng)
        cal_rows.extend([{**r, "regime": regime} for r in res_c["per_pair"]])

    e_n0 = np.asarray([r["e_n0_deg"] for r in cal_rows], dtype=np.float64)
    e_p0 = np.asarray([r["e_p0_m"] for r in cal_rows], dtype=np.float64)
    l1_axis = _axis_stats(e_n0)
    l1_pos = _ep_stats(e_p0)
    l1_ok = bool(l1_axis["median_e_axis_deg"] <= PRIOR_N_MED and l1_axis["p90_e_axis_deg"] <= PRIOR_N_P90)
    l1_ok = l1_ok and bool(l1_pos["median_ep_m"] <= MED_MAX and float(np.nanmean(e_p0)) <= EP_MAX)

    tau_star = calibrate_threshold_youden(
        np.asarray([r["q_rel"] for r in cal_rows], dtype=np.float64),
        np.asarray([r["rel_safe"] for r in cal_rows], dtype=bool),
    )

    l2_ok = True
    l3_ok = True
    router_scores: list[float] = []
    router_labels: list[bool] = []
    methods_table: dict[str, dict[str, dict[str, Any]]] = {
        m: {} for m in ("absolute", "relative", "oracle", "hybrid")
    }

    for regime in REGIMES:
        obs_f, gt_f = _load_cache(root, "formal", regime)
        res_f = _eval_split_regime(
            obs_f, gt_f, s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o, rng=rng, tau=tau_star,
        )
        rows = res_f["per_pair"]
        formal_by_regime[regime] = res_f
        for m, key in (("absolute", "abs"), ("relative", "rel"), ("oracle", "oracle"), ("hybrid", "hyb")):
            methods_table[m][regime] = _regime_table(rows, key)
        l2_ok = l2_ok and methods_table["oracle"][regime]["pass"]
        l3_ok = l3_ok and methods_table["hybrid"][regime]["pass"]
        router_scores.extend([r["q_rel"] for r in rows])
        router_labels.extend([r["rel_safe"] for r in rows])

    router_auroc = _auroc(np.asarray(router_scores), np.asarray(router_labels, dtype=bool))
    pattern, tags = _pattern(l1_ok=l1_ok, l2_ok=l2_ok, l3_ok=l3_ok, router_auroc=router_auroc)

    summary = {
        "pattern": pattern,
        "tags": tags,
        "unlocks_hybrid_world_state_branch": pattern == "hybrid_state_update_supported",
        "unlocks_o0e1": False,
        "unlocks_o1": False,
        "tau_star": float(tau_star),
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "b2_config_hash": R5_B2_CONFIG_HASH,
        "G_L1_prior_anchor": {"ok": l1_ok, "axis_n0": l1_axis, "position_p0": {**l1_pos, "E_p": float(np.nanmean(e_p0))}},
        "G_L2_oracle_hybrid": {"ok": l2_ok, "by_regime": {r: methods_table["oracle"][r] for r in REGIMES}},
        "G_L3_formal_hybrid": {"ok": l3_ok, "by_regime": {r: methods_table["hybrid"][r] for r in REGIMES}},
        "methods_table": methods_table,
        "router": {
            "auroc": router_auroc,
            "tau_star": float(tau_star),
            "formal_rel_selection_rate": float(np.mean([r["hyb_used_rel"] for reg in formal_by_regime for r in formal_by_regime[reg]["per_pair"]])),
            "by_regime_selection_rate": {
                r: float(np.mean([x["hyb_used_rel"] for x in formal_by_regime[r]["per_pair"]])) for r in REGIMES
            },
        },
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0hyb0] pattern={pattern} tau*={tau_star:.4f}", flush=True)
    return summary

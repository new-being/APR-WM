"""RTWX-O0E0R2: controlled-pose segmentation repair + contingent science."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    SEED as O0E0_SEED,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    estimate_effective_pose,
    fibonacci_sphere,
    fuse_points,
    p_surf_from_cloud,
)
from .rtwx_o0e0p0 import RTWXO0E0P0Config, collect_split_numpy, collect_split_robotwin
from .rtwx_o0e0r1 import (
    _axis_diag_pass,
    _axis_stats,
    _ep_stats,
    _fused_clouds,
    _tilt_bucket,
    estimate_axis_b2_at_center,
)
from .rtwx_o0g1b import load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, EP_MAX, _iou, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r2.segmentation_repair_contingent_science.v1"
SEED = 40601
FRESH_TEST_SEED = 37602
N_TEST_FRESH = 200
S0_TRAIN_SEED = O0E0_SEED
S1_TRAIN_SEED = O0E0_SEED
AXIS_IMPROVE_DEG = 5.0


@dataclass(frozen=True)
class RTWXO0E0R2Config:
    output: str = "runs/rtwx_o0e0r2"
    p0_cache: str = "runs/rtwx_o0e0p0"
    natural_cache: str = "runs/rtwx_o0e0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    fresh_test_seed: int = FRESH_TEST_SEED
    n_test_fresh: int = N_TEST_FRESH
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0E0R2Config) -> RTWXO0E0R2Config:
    if cfg.smoke:
        return replace(cfg, n_test_fresh=40, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R2 must not write there")


def _cuda_gc() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_seg_split(cache_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(cache_root / "cache" / split / "seg_gt.npz")
    return np.asarray(z["rgb_h"]), np.asarray(z["rgb_o"]), np.asarray(z["mask_h"]), np.asarray(z["mask_o"])


def _load_p0_split(p0_root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = np.load(p0_root / "cache" / split / "obs.npz")
    gt = np.load(p0_root / "cache" / split / "gt.npz")
    return {k: obs[k] for k in obs.files}, {k: np.asarray(gt[k]) for k in gt.files}


def _write_fresh_cache(root: Path, pool: dict[str, Any]) -> None:
    d = root / "cache" / "fresh_test"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "obs.npz",
        rgb_h=pool["rgb_h"],
        rgb_o=pool["rgb_o"],
        xyz_h=pool["xyz_h"],
        xyz_o=pool["xyz_o"],
        mask_h=pool["mask_h"],
        mask_o=pool["mask_o"],
        vis_h=pool["vis_h"],
        vis_o=pool["vis_o"],
        fov_h=pool["fov_h"],
        fov_o=pool["fov_o"],
        area_h=pool["area_h"],
        area_o=pool["area_o"],
    )
    np.savez_compressed(
        d / "gt.npz",
        p_gt=pool["p"],
        quat_gt=pool["quat"],
        n_gt=pool["n"],
        beta_deg=pool["beta_deg"],
        alpha_rad=pool["alpha_rad"],
        gamma_rad=pool["gamma_rad"],
    )


def _collect_fresh_test(cfg: RTWXO0E0R2Config, root: Path) -> dict[str, Any]:
    cache_obs = root / "cache" / "fresh_test" / "obs.npz"
    if cache_obs.is_file():
        obs = np.load(cache_obs)
        gt = np.load(root / "cache" / "fresh_test" / "gt.npz")
        return {
            "rgb_h": obs["rgb_h"],
            "rgb_o": obs["rgb_o"],
            "xyz_h": obs["xyz_h"],
            "xyz_o": obs["xyz_o"],
            "mask_h": obs["mask_h"],
            "mask_o": obs["mask_o"],
            "vis_h": obs["vis_h"],
            "vis_o": obs["vis_o"],
            "fov_h": obs["fov_h"],
            "fov_o": obs["fov_o"],
            "area_h": obs["area_h"],
            "area_o": obs["area_o"],
            "p": np.asarray(gt["p_gt"]),
            "quat": np.asarray(gt["quat_gt"]),
            "n": np.asarray(gt["n_gt"]),
            "beta_deg": np.asarray(gt["beta_deg"]),
            "alpha_rad": np.asarray(gt["alpha_rad"]),
            "gamma_rad": np.asarray(gt["gamma_rad"]),
        }
    p0cfg = RTWXO0E0P0Config(robotwin_repo=cfg.robotwin_repo, backend="numpy" if cfg.smoke else "robotwin")
    if cfg.smoke:
        pool = collect_split_numpy(p0cfg, cfg.n_test_fresh, cfg.fresh_test_seed)
    else:
        stop: list[str] = []
        pool = collect_split_robotwin(p0cfg, "fresh_test", cfg.n_test_fresh, cfg.fresh_test_seed, stop)
        if stop:
            raise RuntimeError(f"fresh test collect stop: {stop}")
    _write_fresh_cache(root, pool)
    return pool


def _prec_rec(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    tp = float(np.logical_and(p, g).sum())
    return tp / max(float(p.sum()), 1.0), tp / max(float(g.sum()), 1.0)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def _cloud_nn_mean(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return float("nan")
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(Q, dtype=np.float64))
    d, _ = tree.query(np.asarray(P, dtype=np.float64), k=1)
    return float(np.mean(d))


def _mask_cloud_diagnostics(
    obs: dict[str, Any],
    gt_masks: dict[str, np.ndarray],
    pred_masks: dict[str, np.ndarray],
    clouds: list[np.ndarray],
    gt_clouds: list[np.ndarray],
    beta_deg: np.ndarray,
) -> dict[str, Any]:
    n = beta_deg.shape[0]
    iou_h, iou_o, prec_h, rec_h, area_ratio_h, c_disp_h = [], [], [], [], [], []
    cloud_cent_disp, cloud_cnt_ratio, cloud_nn = [], [], []
    for i in range(n):
        mh_p, mo_p = pred_masks["mask_h"][i], pred_masks["mask_o"][i]
        mh_g, mo_g = gt_masks["mask_h"][i], gt_masks["mask_o"][i]
        iou_h.append(_iou(mh_p, mh_g))
        iou_o.append(_iou(mo_p, mo_g))
        ph, rh = _prec_rec(mh_p, mh_g)
        prec_h.append(ph)
        rec_h.append(rh)
        area_ratio_h.append(float(mh_p.sum()) / max(float(mh_g.sum()), 1.0))
        chx, chy = _mask_centroid(mh_p)
        cgx, cgy = _mask_centroid(mh_g)
        c_disp_h.append(float(np.hypot(chx - cgx, chy - cgy)) if np.isfinite(chx) else float("nan"))
        P, Q = clouds[i], gt_clouds[i]
        if P.shape[0] and Q.shape[0]:
            cloud_cent_disp.append(float(np.linalg.norm(P.mean(0) - Q.mean(0))))
            cloud_cnt_ratio.append(float(P.shape[0] / max(Q.shape[0], 1)))
            cloud_nn.append(_cloud_nn_mean(P, Q))
        else:
            cloud_cent_disp.append(float("nan"))
            cloud_cnt_ratio.append(float("nan"))
            cloud_nn.append(float("nan"))
    iou_h_arr = np.asarray(iou_h)
    iou_o_arr = np.asarray(iou_o)
    return {
        "diagnostic_only": True,
        "mean_iou_head": float(np.nanmean(iou_h_arr)),
        "mean_iou_obs": float(np.nanmean(iou_o_arr)),
        "mean_prec_head": float(np.nanmean(prec_h)),
        "mean_rec_head": float(np.nanmean(rec_h)),
        "mean_area_ratio_head": float(np.nanmean(area_ratio_h)),
        "mean_mask_centroid_disp_px": float(np.nanmean(c_disp_h)),
        "mean_cloud_centroid_disp_m": float(np.nanmean(cloud_cent_disp)),
        "mean_cloud_count_ratio": float(np.nanmean(cloud_cnt_ratio)),
        "mean_cloud_nn_m": float(np.nanmean(cloud_nn)),
        "tilt_iou_head": _tilt_bucket(iou_h_arr, beta_deg),
        "tilt_iou_obs": _tilt_bucket(iou_o_arr, beta_deg),
    }


def _instrument_branch(
    clouds: list[np.ndarray],
    p_surfs: list[np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_axis, e_pos = [], []
    for i in range(n_frames):
        P = clouds[i]
        n_hat, _ = estimate_axis_b2_at_center(P, gt_te["p_gt"][i], tree, sphere=sphere)
        e_axis.append(e_axis_deg(n_hat, gt_te["n_gt"][i]))
        p_ceil = p_surfs[i] - delta_y * gt_te["n_gt"][i]
        e_pos.append(float(np.linalg.norm(p_ceil - gt_te["p_gt"][i])))
    e_axis_arr = np.asarray(e_axis, dtype=np.float64)
    e_pos_arr = np.asarray(e_pos, dtype=np.float64)
    g_i1 = _axis_diag_pass(float(np.nanmedian(e_axis_arr)), float(np.nanpercentile(e_axis_arr, 90)))
    med_ep = float(np.nanmedian(e_pos_arr))
    ep_mean = float(np.nanmean(e_pos_arr))
    g_i2 = bool(med_ep <= MED_MAX and ep_mean <= EP_MAX)
    return {
        "G_I1_axis_oracle_p": {"ok": g_i1, **_axis_stats(e_axis_arr)},
        "G_I2_position_oracle_n": {"ok": g_i2, **_ep_stats(e_pos_arr), "E_p": ep_mean},
        "instrument_pass": bool(g_i1 and g_i2),
    }


def _formal_science(
    clouds: list[np.ndarray],
    obs_masks: dict[str, np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_ax_b0, e_ax_b2, p_hats = [], [], []
    for i in range(n_frames):
        obs_i = {"fused_points_B": clouds[i], "mask_h": obs_masks["mask_h"][i], "mask_o": obs_masks["mask_o"][i]}
        _, n_b0, _ = estimate_effective_pose(obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B0", n_const=n_const)
        p_hat, n_b2, _ = estimate_effective_pose(obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2", n_const=n_const)
        e_ax_b0.append(e_axis_deg(n_b0, gt_te["n_gt"][i]))
        e_ax_b2.append(e_axis_deg(n_b2, gt_te["n_gt"][i]))
        p_hats.append(p_hat)
    e_ax_b0 = np.asarray(e_ax_b0)
    e_ax_b2 = np.asarray(e_ax_b2)
    pos = _score_pose(np.stack(p_hats), gt_te["p_gt"])
    b0 = _axis_stats(e_ax_b0)
    b2_axis = _axis_stats(e_ax_b2)
    g1 = _axis_diag_pass(b2_axis["median_e_axis_deg"], b2_axis["p90_e_axis_deg"])
    g2 = bool(pos["E_p"] <= EP_MAX and pos["median_ep_m"] <= MED_MAX)
    b0_excluded = not _axis_diag_pass(b0["median_e_axis_deg"], b0["p90_e_axis_deg"])
    return {
        "B0": b0,
        "B2": {**b2_axis, "E_p": pos["E_p"], "median_ep_m": pos["median_ep_m"], "median_ep_cm": pos["median_ep_m"] * 100},
        "G1": g1,
        "G2": g2,
        "B0_excluded": b0_excluded,
        "formal_pass": bool(g1 and g2 and b0_excluded),
    }


def _pattern(*, instrument_pass: bool, formal_pass: bool, training_insufficient: bool) -> tuple[str, list[str]]:
    tags: list[str] = []
    if training_insufficient:
        tags.append("segmentation_training_insufficient")
    if not instrument_pass:
        return "segmentation_cloud_failure", tags
    if formal_pass:
        return "effective_pose_supported", tags
    return "reference_coupled_failure", tags


def run_rtwx_o0e0r2(output: str | Path, config: RTWXO0E0R2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0R2Config(output=str(output)))
    root = Path(output).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    natural_root = Path(cfg.natural_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r2] p0={p0_root} natural={natural_root} fresh_seed={cfg.fresh_test_seed}", flush=True)

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
        "p0_cache": str(p0_root),
        "natural_cache": str(natural_root),
        "fresh_test_seed": cfg.fresh_test_seed,
        "b2_frozen": True,
        "single_change": "segmentation_training_distribution",
    }
    _write_json(root / "header.json", header)

    print("[rtwx-o0e0r2] train S0 (natural upright)", flush=True)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )
    _cuda_gc()

    print("[rtwx-o0e0r2] train S1 (P0 controlled)", flush=True)
    p0_tr, _ = _load_p0_split(p0_root, "train")
    p0_va, _ = _load_p0_split(p0_root, "val")
    s1 = _train_unet(
        np.concatenate([p0_tr["rgb_h"], p0_tr["rgb_o"]]), np.concatenate([p0_tr["mask_h"], p0_tr["mask_o"]]),
        np.concatenate([p0_va["rgb_h"], p0_va["rgb_o"]]), np.concatenate([p0_va["mask_h"], p0_va["mask_o"]]),
        seed=S1_TRAIN_SEED, epochs=cfg.epochs_unet,
    )
    _cuda_gc()

    print("[rtwx-o0e0r2] fresh controlled test", flush=True)
    pool = _collect_fresh_test(cfg, root)
    obs_te = {k: pool[k] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "mask_h", "mask_o", "vis_h", "vis_o")}
    gt_te = {
        "p_gt": np.asarray(pool["p"]), "quat_gt": np.asarray(pool["quat"]), "n_gt": np.asarray(pool["n"]),
        "beta_deg": np.asarray(pool["beta_deg"]),
    }
    gt_masks = {"mask_h": obs_te["mask_h"], "mask_o": obs_te["mask_o"]}
    beta_deg = gt_te["beta_deg"]

    masks_s0 = {"mask_h": _predict_masks(s0, obs_te["rgb_h"]), "mask_o": _predict_masks(s0, obs_te["rgb_o"])}
    masks_s1 = {"mask_h": _predict_masks(s1, obs_te["rgb_h"]), "mask_o": _predict_masks(s1, obs_te["rgb_o"])}
    clouds_s0 = _fused_clouds(obs_te, masks_s0, seed=cfg.seed)
    clouds_s1 = _fused_clouds(obs_te, masks_s1, seed=cfg.seed + 1)
    clouds_gt = _fused_clouds(obs_te, gt_masks, seed=cfg.seed + 2)
    p_surfs_s0 = [p_surf_from_cloud(P) for P in clouds_s0]
    p_surfs_s1 = [p_surf_from_cloud(P) for P in clouds_s1]

    diag_s0 = _mask_cloud_diagnostics(obs_te, gt_masks, masks_s0, clouds_s0, clouds_gt, beta_deg)
    diag_s1 = _mask_cloud_diagnostics(obs_te, gt_masks, masks_s1, clouds_s1, clouds_gt, beta_deg)
    print(f"[rtwx-o0e0r2] S0 IoU_h={diag_s0['mean_iou_head']:.3f} S1={diag_s1['mean_iou_head']:.3f}", flush=True)

    inst_s0 = _instrument_branch(clouds_s0, p_surfs_s0, gt_te, delta_y=delta_y, tree=tree, sphere=sphere)
    inst_s1 = _instrument_branch(clouds_s1, p_surfs_s1, gt_te, delta_y=delta_y, tree=tree, sphere=sphere)
    print(
        f"[rtwx-o0e0r2] S0 G_I1={inst_s0['G_I1_axis_oracle_p']['median_e_axis_deg']:.1f}° "
        f"S1={inst_s1['G_I1_axis_oracle_p']['median_e_axis_deg']:.1f}°",
        flush=True,
    )

    s0_med = inst_s0["G_I1_axis_oracle_p"]["median_e_axis_deg"]
    s1_med = inst_s1["G_I1_axis_oracle_p"]["median_e_axis_deg"]
    training_insufficient = bool(not inst_s1["instrument_pass"] and (s0_med - s1_med) < AXIS_IMPROVE_DEG)

    formal: dict[str, Any] | None = None
    science_ran = False
    if inst_s1["instrument_pass"]:
        print("[rtwx-o0e0r2] instrument PASS → formal science", flush=True)
        gt_tr, _ = _load_p0_split(p0_root, "train")
        n_const = gt_tr["n_gt"].sum(0)
        n_const = n_const / max(np.linalg.norm(n_const), 1e-12)
        formal = _formal_science(clouds_s1, masks_s1, gt_te, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const)
        science_ran = True
    else:
        print("[rtwx-o0e0r2] instrument FAIL → STOP science", flush=True)

    pattern, tags = _pattern(
        instrument_pass=inst_s1["instrument_pass"],
        formal_pass=bool(formal and formal["formal_pass"]),
        training_insufficient=training_insufficient,
    )

    result = {
        "header": header,
        "pattern": pattern,
        "patterns": tags,
        "S0_natural": {"instrument": inst_s0, "mask_cloud_diag": diag_s0},
        "S1_controlled": {"instrument": inst_s1, "mask_cloud_diag": diag_s1},
        "instrument_pass": inst_s1["instrument_pass"],
        "science_ran": science_ran,
        "formal_science": formal,
        "r0_r1_patterns_unchanged": True,
        "unlocks_o0e1": pattern == "effective_pose_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0r2] pattern={pattern} tags={tags}", flush=True)
    return result

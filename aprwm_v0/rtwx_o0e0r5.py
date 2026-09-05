"""RTWX-O0E0R5: visibility-aware object reference (single mechanism swap)."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .geometry.visibility_reference import (
    AxialVisibilityReference,
    build_lut,
    camera_center_world,
    cloud_centroid,
)
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    N_OBS,
    REFINE_DEGS,
    SEED as O0E0_SEED,
    SPHERE_SEED,
    TOP_KR,
    _tangent_neighbors,
    cad_hr_profile,
    e_axis_deg,
    estimate_effective_pose,
    fibonacci_sphere,
    fuse_points,
    p_surf_from_cloud,
)
from .rtwx_o0e0p0 import (
    EXCITE_ALPHA_DEG,
    EXCITE_RHO,
    RTWXO0E0P0Config,
    TILT_DEG,
    YAW_OCTANTS,
    YAW_OCTANTS_MIN,
    _capture_one,
    _set_static_pose,
    _stratified_betas,
    collect_split_numpy,
    collect_split_robotwin,
    sample_pose,
)
from .rtwx_o0e0r1 import S_at_center, _axis_diag_pass, _axis_stats, _ep_stats, _fused_clouds, estimate_axis_b2_at_center
from .rtwx_o0e0r2 import _cuda_gc, _load_p0_split, _load_seg_split
from .rtwx_o0g1b import load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, _get_cam
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R5_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r5.visibility_reference.v1"
SEED = 43601
FRESH_TEST_SEED = 37605
N_TEST_FRESH = 200
S0_TRAIN_SEED = O0E0_SEED
MECH_TARGET_CM = 1.75
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n"})


@dataclass(frozen=True)
class RTWXO0E0R5Config:
    output: str = "runs/rtwx_o0e0r5"
    p0_cache: str = "runs/rtwx_o0e0p0"
    natural_cache: str = "runs/rtwx_o0e0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    fresh_test_seed: int = FRESH_TEST_SEED
    n_test_fresh: int = N_TEST_FRESH
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0E0R5Config) -> RTWXO0E0R5Config:
    if cfg.smoke:
        return replace(cfg, n_test_fresh=40, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R5 must not write there")


def assert_obs_no_gt(obs: dict[str, Any]) -> None:
    bad = OBS_FORBIDDEN.intersection(obs.keys())
    if bad:
        raise AssertionError(f"oracle leak into estimator obs keys: {sorted(bad)}")


def estimate_axis_b2_center_fn(
    P: np.ndarray,
    center_fn: Callable[[np.ndarray], np.ndarray],
    tree: Any,
    *,
    sphere: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if P.shape[0] < 8:
        from .rtwx_o0e0 import EY

        return EY.copy(), {"S_best": float("inf"), "S_second": float("inf"), "margin": float("nan")}
    scores = []
    for n in sphere:
        p_c = center_fn(n)
        scores.append(S_at_center(P, p_c, n, tree))
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    cands = [sphere[i] for i in order[:TOP_KR]]
    best_n, best_s = cands[0], float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else best_s
    for n0 in list(cands):
        cur_n, cur_s = n0, S_at_center(P, center_fn(n0), n0, tree)
        for deg in REFINE_DEGS:
            improved = True
            while improved:
                improved = False
                for nb in _tangent_neighbors(cur_n, deg):
                    s = S_at_center(P, center_fn(nb), nb, tree)
                    if s + 1e-12 < cur_s:
                        cur_n, cur_s = nb, s
                        improved = True
        if cur_s < best_s:
            second = best_s
            best_n, best_s = cur_n, cur_s
        elif cur_s < second and not np.allclose(cur_n, best_n):
            second = cur_s
    return best_n / np.linalg.norm(best_n), {"S_best": best_s, "S_second": second, "margin": second - best_s}


def estimate_effective_pose_vis(
    obs: dict[str, Any],
    lut: AxialVisibilityReference,
    *,
    tree: Any,
    sphere: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    assert_obs_no_gt(obs)
    P = np.asarray(obs["fused_points_B"], dtype=np.float64)
    c_h = np.asarray(obs["c_h"], dtype=np.float64)
    c_o = np.asarray(obs["c_o"], dtype=np.float64)
    cam_h = np.asarray(obs["cam_h"], dtype=np.float64)
    cam_o = np.asarray(obs["cam_o"], dtype=np.float64)

    def center_fn(n: np.ndarray) -> np.ndarray:
        return lut.estimate_center_dual(n, c_h, c_o, cam_h, cam_o)

    n_hat, meta = estimate_axis_b2_center_fn(P, center_fn, tree, sphere=sphere)
    p_hat = center_fn(n_hat)
    return p_hat, n_hat, meta


def _per_view_clouds(
    xyz: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    mm = np.asarray(mask, bool) & np.isfinite(xyz).all(-1)
    if not mm.any():
        return np.zeros((0, 3), dtype=np.float64)
    P = np.asarray(xyz, dtype=np.float64)[mm]
    if P.shape[0] > N_OBS // 2:
        P = P[rng.choice(P.shape[0], N_OBS // 2, replace=False)]
    return P


def _capture_with_cam(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
    cap = _capture_one(env, size, cup_ids, rep_h, rep_o)
    cam_h = _get_cam(env, CAM_HEAD)
    cam_o = _get_cam(env, CAM_OBS)
    cap["E_h"] = np.asarray(cam_h.get_extrinsic_matrix(), dtype=np.float64)
    cap["E_o"] = np.asarray(cam_o.get_extrinsic_matrix(), dtype=np.float64)
    cap["cam_h"] = camera_center_world(cap["E_h"])
    cap["cam_o"] = camera_center_world(cap["E_o"])
    return cap


def collect_test_numpy(cfg: RTWXO0E0R5Config, n: int, seed: int) -> dict[str, Any]:
    base = collect_split_numpy(RTWXO0E0P0Config(backend="numpy"), n, seed)
    rng = np.random.default_rng(seed + 99)
    nh = base["p"].shape[0]
    cam_h = np.tile(np.array([0.0, -0.3, 1.0], dtype=np.float64), (nh, 1))
    cam_o = np.tile(np.array([0.4, -0.2, 0.9], dtype=np.float64), (nh, 1))
    cam_h += rng.normal(0, 0.02, cam_h.shape)
    cam_o += rng.normal(0, 0.02, cam_o.shape)
    base["cam_h"] = cam_h
    base["cam_o"] = cam_o
    return base


def collect_test_robotwin(cfg: RTWXO0E0R5Config, n: int, seed: int, stop: list[str]) -> dict[str, Any]:
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
    rows: dict[str, list] = {k: [] for k in (
        "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o",
        "vis_h", "vis_o", "cam_h", "cam_o",
        "p", "quat", "n", "beta_deg", "gamma_rad",
    )}
    try:
        env = _setup_env(repo, TASK, seed, 32, args)
        env.check_success = lambda *a, **k: False
        cup_ids = _cup_ids(env)
        rep_h = rep_o = None
        for i, beta in enumerate(betas):
            if i % 25 == 0 or i + 1 == n:
                print(f"[rtwx-o0e0r5] collect {i}/{n}", flush=True)
            pose = sample_pose(rng, float(beta))
            _set_static_pose(env, pose["p"], pose["quat"])
            cap = _capture_with_cam(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap["rep_h"], cap["rep_o"]
            for k in ("rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o", "cam_h", "cam_o"):
                rows[k].append(cap[k])
            rows["p"].append(pose["p"])
            rows["quat"].append(pose["quat"])
            rows["n"].append(pose["n"])
            rows["beta_deg"].append(float(beta))
            rows["gamma_rad"].append(float(pose["gamma_rad"]))
        env.close()
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    out: dict[str, Any] = {
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "mask_h": np.stack(rows["mask_h"]),
        "mask_o": np.stack(rows["mask_o"]),
        "xyz_h": np.stack(rows["xyz_h"]),
        "xyz_o": np.stack(rows["xyz_o"]),
        "vis_h": np.asarray(rows["vis_h"], bool),
        "vis_o": np.asarray(rows["vis_o"], bool),
        "cam_h": np.stack(rows["cam_h"]),
        "cam_o": np.stack(rows["cam_o"]),
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "n": np.stack(rows["n"]),
        "beta_deg": np.asarray(rows["beta_deg"], dtype=np.float64),
        "gamma_rad": np.asarray(rows["gamma_rad"], dtype=np.float64),
    }
    return out


def _collect_test(cfg: RTWXO0E0R5Config, root: Path) -> dict[str, Any]:
    cache_obs = root / "cache" / "test" / "obs.npz"
    if cache_obs.is_file():
        obs = np.load(cache_obs)
        gt = np.load(root / "cache" / "test" / "gt.npz")
        return {
            "rgb_h": obs["rgb_h"], "rgb_o": obs["rgb_o"],
            "xyz_h": obs["xyz_h"], "xyz_o": obs["xyz_o"],
            "cam_h": obs["cam_h"], "cam_o": obs["cam_o"],
            "vis_h": obs["vis_h"], "vis_o": obs["vis_o"],
            "p": np.asarray(gt["p_gt"]), "quat": np.asarray(gt["quat_gt"]), "n": np.asarray(gt["n_gt"]),
            "beta_deg": np.asarray(gt["beta_deg"]), "gamma_rad": np.asarray(gt["gamma_rad"]),
        }
    if cfg.smoke:
        pool = collect_test_numpy(cfg, cfg.n_test_fresh, cfg.fresh_test_seed)
    else:
        stop: list[str] = []
        pool = collect_test_robotwin(cfg, cfg.n_test_fresh, cfg.fresh_test_seed, stop)
        if stop:
            raise RuntimeError(f"test collect stop: {stop}")
    d = root / "cache" / "test"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "obs.npz",
        rgb_h=pool["rgb_h"], rgb_o=pool["rgb_o"],
        xyz_h=pool["xyz_h"], xyz_o=pool["xyz_o"],
        cam_h=pool["cam_h"], cam_o=pool["cam_o"],
        vis_h=pool["vis_h"], vis_o=pool["vis_o"],
    )
    np.savez_compressed(
        d / "gt.npz",
        p_gt=pool["p"], quat_gt=pool["quat"], n_gt=pool["n"],
        beta_deg=pool["beta_deg"], gamma_rad=pool["gamma_rad"],
    )
    return _collect_test(cfg, root)


def _data_gates(gt_te: dict[str, Any], n_const: np.ndarray) -> dict[str, Any]:
    vis_any = np.asarray(gt_te["vis_h"], bool) | np.asarray(gt_te["vis_o"], bool)
    p_any = float(np.mean(vis_any))
    g0a = bool(p_any >= P_ANY_AGG)
    ang = np.array([e_axis_deg(gt_te["n_gt"][i], n_const) for i in range(gt_te["n_gt"].shape[0])])
    r_excite = float(np.mean(ang > EXCITE_ALPHA_DEG))
    g0b = bool(r_excite >= EXCITE_RHO)
    oct = (np.asarray(gt_te["gamma_rad"]) % (2.0 * np.pi) / (2.0 * np.pi / YAW_OCTANTS)).astype(int) % YAW_OCTANTS
    per_beta: dict[str, int] = {}
    g0c = True
    for b in TILT_DEG:
        m = np.isclose(gt_te["beta_deg"], b)
        occ = int(len(np.unique(oct[m]))) if m.any() else 0
        per_beta[str(float(b))] = occ
        if occ < YAW_OCTANTS_MIN:
            g0c = False
    return {
        "ok": bool(g0a and g0b and g0c),
        "G0a": {"ok": g0a, "P_any": p_any},
        "G0b": {"ok": g0b, "r_excite": r_excite},
        "G0c": {"ok": g0c, "octants_per_beta": per_beta},
    }


def _instrument_s0(
    clouds: list[np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_axis, e_pos = [], []
    for i in range(n_frames):
        n_hat, _ = estimate_axis_b2_at_center(clouds[i], gt_te["p_gt"][i], tree, sphere=sphere)
        e_axis.append(e_axis_deg(n_hat, gt_te["n_gt"][i]))
        p_surf = p_surf_from_cloud(clouds[i])
        p_ceil = p_surf - delta_y * gt_te["n_gt"][i]
        e_pos.append(float(np.linalg.norm(p_ceil - gt_te["p_gt"][i])))
    e_axis_arr = np.asarray(e_axis, dtype=np.float64)
    e_pos_arr = np.asarray(e_pos, dtype=np.float64)
    g_i0_axis = _axis_diag_pass(float(np.nanmedian(e_axis_arr)), float(np.nanpercentile(e_axis_arr, 90)))
    g_i0_pos = bool(float(np.nanmedian(e_pos_arr)) <= MED_MAX and float(np.nanmean(e_pos_arr)) <= EP_MAX)
    return {
        "G_I0_axis_oracle_p": {"ok": g_i0_axis, **_axis_stats(e_axis_arr)},
        "G_I0_position_oracle_n": {"ok": g_i0_pos, **_ep_stats(e_pos_arr), "E_p": float(np.nanmean(e_pos_arr))},
        "instrument_pass": bool(g_i0_axis and g_i0_pos),
    }


def _instrument_vis(
    frames: list[dict[str, Any]],
    gt_te: dict[str, Any],
    lut: AxialVisibilityReference,
    *,
    tree: Any,
    sphere: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_p, e_axis = [], []
    for i in range(n_frames):
        fr = frames[i]
        p_vis = lut.estimate_center_dual(gt_te["n_gt"][i], fr["c_h"], fr["c_o"], fr["cam_h"], fr["cam_o"])
        e_p.append(float(np.linalg.norm(p_vis - gt_te["p_gt"][i])))
        n_hat, _ = estimate_axis_b2_at_center(fr["cloud_fused"], p_vis, tree, sphere=sphere)
        e_axis.append(e_axis_deg(n_hat, gt_te["n_gt"][i]))
    e_p_arr = np.asarray(e_p, dtype=np.float64)
    e_axis_arr = np.asarray(e_axis, dtype=np.float64)
    g_pos = bool(float(np.nanmedian(e_p_arr)) <= MED_MAX and float(np.nanmean(e_p_arr)) <= EP_MAX)
    g_axis = _axis_diag_pass(float(np.nanmedian(e_axis_arr)), float(np.nanpercentile(e_axis_arr, 90)))
    return {
        "G_I1_position": {
            "ok": g_pos,
            **_ep_stats(e_p_arr),
            "E_p": float(np.nanmean(e_p_arr)),
            "P_ep_le_1p75cm": float(np.mean(e_p_arr <= MECH_TARGET_CM / 100.0)),
            "mech_target_cm": MECH_TARGET_CM,
        },
        "G_I1_axis_at_vis_center": {"ok": g_axis, **_axis_stats(e_axis_arr)},
        "instrument_pass": bool(g_pos and g_axis),
    }


def _formal_branch(
    frames: list[dict[str, Any]],
    gt_te: dict[str, Any],
    *,
    lut: AxialVisibilityReference | None,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_ax, p_hats = [], []
    for i in range(n_frames):
        fr = frames[i]
        if mode == "old":
            obs_i = {"fused_points_B": fr["cloud_fused"], "mask_h": fr["mask_h"], "mask_o": fr["mask_o"]}
            p_hat, n_hat, _ = estimate_effective_pose(
                obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2", n_const=n_const,
            )
        else:
            obs_i = {
                "fused_points_B": fr["cloud_fused"],
                "c_h": fr["c_h"], "c_o": fr["c_o"],
                "cam_h": fr["cam_h"], "cam_o": fr["cam_o"],
            }
            p_hat, n_hat, _ = estimate_effective_pose_vis(obs_i, lut, tree=tree, sphere=sphere)
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


def _diagnostics(
    frames: list[dict[str, Any]],
    gt_te: dict[str, Any],
    lut: AxialVisibilityReference,
    *,
    delta_y: float,
    formal_old: dict[str, Any],
    formal_vis: dict[str, Any],
) -> dict[str, Any]:
    n = gt_te["p_gt"].shape[0]
    e_p_vis, e_p_old, e_ax_vis, e_ax_old = [], [], [], []
    e_ph, e_po, d_ho = [], [], []
    by_beta: dict[str, list[float]] = {str(float(b)): [] for b in TILT_DEG}
    for i in range(n):
        fr = frames[i]
        p_vis = lut.estimate_center_dual(gt_te["n_gt"][i], fr["c_h"], fr["c_o"], fr["cam_h"], fr["cam_o"])
        p_surf = p_surf_from_cloud(fr["cloud_fused"])
        p_old = p_surf - delta_y * gt_te["n_gt"][i]
        ph = lut.estimate_center(gt_te["n_gt"][i], fr["c_h"], fr["cam_h"])
        po = lut.estimate_center(gt_te["n_gt"][i], fr["c_o"], fr["cam_o"])
        e_p_vis.append(float(np.linalg.norm(p_vis - gt_te["p_gt"][i])))
        e_p_old.append(float(np.linalg.norm(p_old - gt_te["p_gt"][i])))
        e_ph.append(float(np.linalg.norm(ph - gt_te["p_gt"][i])))
        e_po.append(float(np.linalg.norm(po - gt_te["p_gt"][i])))
        d_ho.append(float(np.linalg.norm(ph - po)))
        bkey = str(float(gt_te["beta_deg"][i]))
        if bkey in by_beta:
            by_beta[bkey].append(e_p_vis[-1])
    e_p_vis_a = np.asarray(e_p_vis, dtype=np.float64)
    e_p_old_a = np.asarray(e_p_old, dtype=np.float64)
    d_ho_a = np.asarray(d_ho, dtype=np.float64)
    return {
        "D1_center": {
            "median_ep_vis_cm": float(np.nanmedian(e_p_vis_a) * 100.0),
            "p90_ep_vis_cm": float(np.nanpercentile(e_p_vis_a, 90) * 100.0),
            "mech_target_cm": MECH_TARGET_CM,
            "P_ep_le_mech": float(np.mean(e_p_vis_a <= MECH_TARGET_CM / 100.0)),
        },
        "D2_by_tilt": {
            b: {"median_ep_vis_cm": float(np.nanmedian(np.asarray(v)) * 100.0) if v else float("nan")}
            for b, v in by_beta.items()
        },
        "D3_by_camera": {
            "median_ep_head_cm": float(np.nanmedian(np.asarray(e_ph)) * 100.0),
            "median_ep_obs_cm": float(np.nanmedian(np.asarray(e_po)) * 100.0),
        },
        "D4_head_observer_disagreement_median": float(np.nanmedian(d_ho_a)),
        "D5_legacy_vs_visibility": {
            "median_ep_old_cm": float(np.nanmedian(e_p_old_a) * 100.0),
            "median_ep_vis_cm": float(np.nanmedian(e_p_vis_a) * 100.0),
            "delta_axis_median_deg": float(
                formal_old["axis"]["median_e_axis_deg"] - formal_vis["axis"]["median_e_axis_deg"]
            ),
        },
    }


def _pattern(*, data_ok: bool, g_i0: bool, g_i1: bool, formal_ok: bool) -> str:
    if not data_ok:
        return "observation_support_failure"
    if not g_i0:
        return "segmentation_cloud_failure"
    if not g_i1:
        return "visibility_reference_instrument_failure"
    if formal_ok:
        return "effective_pose_supported"
    return "visibility_reference_joint_failure"


def run_rtwx_o0e0r5(output: str | Path, config: RTWXO0E0R5Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0R5Config(output=str(output)))
    root = Path(output).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    natural_root = Path(cfg.natural_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r5] fresh_seed={cfg.fresh_test_seed}", flush=True)

    repo = Path(cfg.robotwin_repo)
    prior = load_delta_O(str(repo) if repo.is_dir() else "/root/RoboTwin")
    delta_y = float(np.asarray(prior["delta_O"], dtype=np.float64)[1])

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

    lut_path = root / "visibility_lut.npz"
    if lut_path.is_file():
        lut = AxialVisibilityReference.load(lut_path)
    else:
        lut = build_lut(cad_pts, cad_path=cad_glb if cad_glb.is_file() else None)
        lut.save(lut_path)

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "fresh_test_seed": cfg.fresh_test_seed,
        "b2_frozen": True,
        "S0_natural": True,
        "mechanism_swap_only": "fixed_delta_y -> visibility_aware",
    }
    _write_json(root / "header.json", header)

    print("[rtwx-o0e0r5] train S0", flush=True)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )
    _cuda_gc()

    print("[rtwx-o0e0r5] collect test", flush=True)
    pool = _collect_test(cfg, root)
    obs_te = {k: pool[k] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "cam_h", "cam_o", "vis_h", "vis_o")}
    gt_te = {
        "p_gt": np.asarray(pool["p"]), "n_gt": np.asarray(pool["n"]),
        "beta_deg": np.asarray(pool["beta_deg"]), "gamma_rad": np.asarray(pool["gamma_rad"]),
        "vis_h": np.asarray(pool["vis_h"]), "vis_o": np.asarray(pool["vis_o"]),
    }

    _, gt_tr = _load_p0_split(p0_root, "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    data = _data_gates(gt_te, n_const)
    print(f"[rtwx-o0e0r5] L1 data ok={data['ok']}", flush=True)
    if not data["ok"]:
        result = {
            "header": header, "pattern": "observation_support_failure", "L1_data": data,
            "science_ran": False, "unlocks_o0e1_prereg": False, "unlocks_o1": False, "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        return result

    masks = {"mask_h": _predict_masks(s0, obs_te["rgb_h"]), "mask_o": _predict_masks(s0, obs_te["rgb_o"])}
    rng = np.random.default_rng(cfg.seed)
    frames: list[dict[str, Any]] = []
    for i in range(gt_te["p_gt"].shape[0]):
        Ph = _per_view_clouds(obs_te["xyz_h"][i], masks["mask_h"][i], rng=rng)
        Po = _per_view_clouds(obs_te["xyz_o"][i], masks["mask_o"][i], rng=rng)
        Pf = fuse_points(obs_te["xyz_h"][i], masks["mask_h"][i], obs_te["xyz_o"][i], masks["mask_o"][i], rng=rng)
        frames.append({
            "c_h": cloud_centroid(Ph),
            "c_o": cloud_centroid(Po),
            "cam_h": obs_te["cam_h"][i],
            "cam_o": obs_te["cam_o"][i],
            "cloud_fused": Pf,
            "mask_h": masks["mask_h"][i],
            "mask_o": masks["mask_o"][i],
        })
    clouds = [f["cloud_fused"] for f in frames]

    g_i0 = _instrument_s0(clouds, gt_te, delta_y=delta_y, tree=tree, sphere=sphere)
    print(
        f"[rtwx-o0e0r5] G_I0 axis={g_i0['G_I0_axis_oracle_p']['median_e_axis_deg']:.2f}° "
        f"pass={g_i0['instrument_pass']}",
        flush=True,
    )
    if not g_i0["instrument_pass"]:
        result = {
            "header": header, "pattern": "segmentation_cloud_failure",
            "L1_data": data, "L2_G_I0": g_i0, "science_ran": False,
            "unlocks_o0e1_prereg": False, "unlocks_o1": False, "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        return result

    g_i1 = _instrument_vis(frames, gt_te, lut, tree=tree, sphere=sphere)
    print(
        f"[rtwx-o0e0r5] G_I1 ep_med={g_i1['G_I1_position']['median_ep_cm']:.2f}cm "
        f"axis={g_i1['G_I1_axis_at_vis_center']['median_e_axis_deg']:.2f}° pass={g_i1['instrument_pass']}",
        flush=True,
    )
    if not g_i1["instrument_pass"]:
        result = {
            "header": header, "pattern": "visibility_reference_instrument_failure",
            "L1_data": data, "L2_G_I0": g_i0, "L2_G_I1": g_i1, "science_ran": False,
            "unlocks_o0e1_prereg": False, "unlocks_o1": False, "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        return result

    print("[rtwx-o0e0r5] L3 formal", flush=True)
    formal_old = _formal_branch(frames, gt_te, lut=None, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const, mode="old")
    formal_vis = _formal_branch(frames, gt_te, lut=lut, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const, mode="vis")
    print(
        f"[rtwx-o0e0r5] old med_axis={formal_old['axis']['median_e_axis_deg']:.2f}° "
        f"vis={formal_vis['axis']['median_e_axis_deg']:.2f}° pass={formal_vis['formal_pass']}",
        flush=True,
    )
    diag = _diagnostics(frames, gt_te, lut, delta_y=delta_y, formal_old=formal_old, formal_vis=formal_vis)
    pattern = _pattern(data_ok=True, g_i0=True, g_i1=True, formal_ok=formal_vis["formal_pass"])
    result = {
        "header": header,
        "pattern": pattern,
        "L1_data": data,
        "L2_G_I0": g_i0,
        "L2_G_I1": g_i1,
        "L3_formal_old": formal_old,
        "L3_formal_vis": formal_vis,
        "diagnostics": diag,
        "science_ran": True,
        "unlocks_o0e1_prereg": pattern == "effective_pose_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0r5] pattern={pattern}", flush=True)
    return result

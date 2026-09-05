"""RTWX-O0E0R3: S0 formal effective-pose confirmation on fresh controlled test."""

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
    p_surf_from_cloud,
)
from .rtwx_o0e0p0 import (
    EXCITE_ALPHA_DEG,
    EXCITE_RHO,
    RTWXO0E0P0Config,
    TILT_DEG,
    YAW_OCTANTS,
    YAW_OCTANTS_MIN,
    collect_split_numpy,
    collect_split_robotwin,
)
from .rtwx_o0e0r1 import (
    _axis_diag_pass,
    _axis_stats,
    _ep_stats,
    _fused_clouds,
    estimate_axis_b2_at_center,
)
from .rtwx_o0e0r2 import _cuda_gc, _load_seg_split, _load_p0_split, _write_fresh_cache
from .rtwx_o0g1b import load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0v import P_ANY_AGG
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R3_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r3.s0_formal_effective_pose.v1"
SEED = 41601
FRESH_TEST_SEED = 37603
N_TEST_FRESH = 200
S0_TRAIN_SEED = O0E0_SEED


@dataclass(frozen=True)
class RTWXO0E0R3Config:
    output: str = "runs/rtwx_o0e0r3"
    p0_cache: str = "runs/rtwx_o0e0p0"
    natural_cache: str = "runs/rtwx_o0e0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    fresh_test_seed: int = FRESH_TEST_SEED
    n_test_fresh: int = N_TEST_FRESH
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0E0R3Config) -> RTWXO0E0R3Config:
    if cfg.smoke:
        return replace(cfg, n_test_fresh=40, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R3 must not write there")


def _collect_fresh_test(cfg: RTWXO0E0R3Config, root: Path) -> dict[str, Any]:
    cache_obs = root / "cache" / "fresh_test" / "obs.npz"
    if cache_obs.is_file():
        obs = np.load(cache_obs)
        gt = np.load(root / "cache" / "fresh_test" / "gt.npz")
        return {
            "rgb_h": obs["rgb_h"], "rgb_o": obs["rgb_o"],
            "xyz_h": obs["xyz_h"], "xyz_o": obs["xyz_o"],
            "mask_h": obs["mask_h"], "mask_o": obs["mask_o"],
            "vis_h": obs["vis_h"], "vis_o": obs["vis_o"],
            "p": np.asarray(gt["p_gt"]), "quat": np.asarray(gt["quat_gt"]), "n": np.asarray(gt["n_gt"]),
            "beta_deg": np.asarray(gt["beta_deg"]), "gamma_rad": np.asarray(gt["gamma_rad"]),
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
        n_hat, _ = estimate_axis_b2_at_center(clouds[i], gt_te["p_gt"][i], tree, sphere=sphere)
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


def _formal_s0(
    clouds: list[np.ndarray],
    masks: dict[str, np.ndarray],
    gt_te: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
) -> dict[str, Any]:
    n_frames = gt_te["p_gt"].shape[0]
    e_ax_b2, p_hats = [], []
    for i in range(n_frames):
        obs_i = {"fused_points_B": clouds[i], "mask_h": masks["mask_h"][i], "mask_o": masks["mask_o"][i]}
        p_hat, n_hat, _ = estimate_effective_pose(obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode="B2", n_const=n_const)
        e_ax_b2.append(e_axis_deg(n_hat, gt_te["n_gt"][i]))
        p_hats.append(p_hat)
    e_ax_b2 = np.asarray(e_ax_b2, dtype=np.float64)
    pos = _score_pose(np.stack(p_hats), gt_te["p_gt"])
    b2 = _axis_stats(e_ax_b2)
    g1 = _axis_diag_pass(b2["median_e_axis_deg"], b2["p90_e_axis_deg"])
    g2 = bool(pos["E_p"] <= EP_MAX and pos["median_ep_m"] <= MED_MAX)
    return {
        "B2": {**b2, "E_p": pos["E_p"], "median_ep_m": pos["median_ep_m"], "median_ep_cm": pos["median_ep_m"] * 100},
        "G1_axis": g1,
        "G2_position": g2,
        "formal_pass": bool(g1 and g2),
    }


def _pattern(*, data_ok: bool, instrument_ok: bool, formal_ok: bool) -> str:
    if not data_ok:
        return "observation_support_failure"
    if not instrument_ok:
        return "segmentation_cloud_failure"
    if formal_ok:
        return "effective_pose_supported"
    return "reference_coupled_failure"


def run_rtwx_o0e0r3(output: str | Path, config: RTWXO0E0R3Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0R3Config(output=str(output)))
    root = Path(output).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    natural_root = Path(cfg.natural_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r3] S0 formal fresh_seed={cfg.fresh_test_seed} (≠37602)", flush=True)

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
        "segmentation": "S0_natural_frozen_unet",
        "fresh_test_seed": cfg.fresh_test_seed,
        "r2_seed_excluded": 37602,
        "b2_frozen": True,
        "claim_scope": "controlled_021_cup_rgbd_S0_only",
    }
    _write_json(root / "header.json", header)

    print("[rtwx-o0e0r3] train S0 (natural)", flush=True)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )
    _cuda_gc()

    print("[rtwx-o0e0r3] collect fresh test", flush=True)
    pool = _collect_fresh_test(cfg, root)
    obs_te = {k: pool[k] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "mask_h", "mask_o", "vis_h", "vis_o")}
    gt_te = {
        "p_gt": np.asarray(pool["p"]), "n_gt": np.asarray(pool["n"]),
        "beta_deg": np.asarray(pool["beta_deg"]), "gamma_rad": np.asarray(pool["gamma_rad"]),
        "vis_h": np.asarray(pool["vis_h"]), "vis_o": np.asarray(pool["vis_o"]),
    }

    _, gt_tr = _load_p0_split(p0_root, "train")
    n_sum = gt_tr["n_gt"].sum(0)
    n_const = n_sum / max(np.linalg.norm(n_sum), 1e-12)

    data = _data_gates(gt_te, n_const)
    print(f"[rtwx-o0e0r3] L1 data ok={data['ok']} P_any={data['G0a']['P_any']:.3f} r_excite={data['G0b']['r_excite']:.3f}", flush=True)
    if not data["ok"]:
        result = {
            "header": header, "pattern": "observation_support_failure", "L1_data": data,
            "science_ran": False, "unlocks_o0e1": False, "unlocks_o1": False, "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0r3] pattern=observation_support_failure", flush=True)
        return result

    masks_s0 = {"mask_h": _predict_masks(s0, obs_te["rgb_h"]), "mask_o": _predict_masks(s0, obs_te["rgb_o"])}
    clouds = _fused_clouds(obs_te, masks_s0, seed=cfg.seed)
    p_surfs = [p_surf_from_cloud(P) for P in clouds]

    inst = _instrument_s0(clouds, p_surfs, gt_te, delta_y=delta_y, tree=tree, sphere=sphere)
    print(
        f"[rtwx-o0e0r3] L2 G_I1 med={inst['G_I1_axis_oracle_p']['median_e_axis_deg']:.2f}° "
        f"G_I2 med_ep={inst['G_I2_position_oracle_n']['median_ep_cm']:.2f}cm pass={inst['instrument_pass']}",
        flush=True,
    )
    if not inst["instrument_pass"]:
        result = {
            "header": header, "pattern": "segmentation_cloud_failure",
            "L1_data": data, "L2_instrument": inst, "science_ran": False,
            "unlocks_o0e1": False, "unlocks_o1": False, "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0r3] pattern=segmentation_cloud_failure", flush=True)
        return result

    print("[rtwx-o0e0r3] L3 formal non-oracle B2", flush=True)
    formal = _formal_s0(clouds, masks_s0, gt_te, delta_y=delta_y, tree=tree, sphere=sphere, n_const=n_const)
    print(
        f"[rtwx-o0e0r3] formal B2 med={formal['B2']['median_e_axis_deg']:.2f}° "
        f"med_ep={formal['B2']['median_ep_cm']:.2f}cm pass={formal['formal_pass']}",
        flush=True,
    )

    pattern = _pattern(data_ok=True, instrument_ok=True, formal_ok=formal["formal_pass"])
    result = {
        "header": header,
        "pattern": pattern,
        "L1_data": data,
        "L2_instrument": inst,
        "L3_formal": formal,
        "science_ran": True,
        "unlocks_o0e1_prereg": pattern == "effective_pose_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0r3] pattern={pattern}", flush=True)
    return result

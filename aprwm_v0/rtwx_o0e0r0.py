"""RTWX-O0E0R0: effective-pose science on O0E0P0 controlled cache. Frozen B0/B1/B2."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    F90_THRESH,
    K_SPHERE,
    N_CAD,
    P_DET_MIN,
    REFINE_DEGS,
    SEED as O0E0_SEED,
    SPHERE_SEED,
    TOP_KR,
    cad_hr_profile,
    d_G_deg,
    e_axis_deg,
    estimate_effective_pose,
    fibonacci_sphere,
    fuse_points,
    R0_from_n,
)
from .rtwx_o0e0p0 import TILT_DEG, YAW_OCTANTS
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, STRONG_MED, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r0.controlled_pose_science.v1"
SEED = 38601
YAW_OCTANT_DIAG_BETA = 20.0


@dataclass(frozen=True)
class RTWXO0E0R0Config:
    output: str = "runs/rtwx_o0e0r0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0E0R0Config) -> RTWXO0E0R0Config:
    if cfg.smoke:
        return replace(cfg, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R0 must not write there")


def _cuda_gc() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _pattern(*, g0: bool, g1: bool, g2: bool, b0_excluded: bool) -> str:
    if not g0:
        return "controlled_pose_detection_failure"
    if not g1:
        return "effective_axis_failure"
    if not g2:
        return "effective_position_failure"
    if not b0_excluded:
        return "constant_baseline_not_excluded"
    return "effective_pose_supported"


def _branch_passes_g1(stats: dict[str, Any]) -> bool:
    return bool(stats["median_e_axis_deg"] <= MED_ER_MAX and stats["p90_e_axis_deg"] <= P90_ER_MAX)


def _branch_passes_g2(stats: dict[str, Any]) -> bool:
    return bool(stats["E_p"] <= EP_MAX and stats["median_ep_m"] <= MED_MAX)


def _write_science_cache(root: Path, split: str, obs: dict[str, Any], masks_learned: dict[str, np.ndarray]) -> None:
    d = root / "science_cache" / split
    d.mkdir(parents=True, exist_ok=True)
    mh, mo = masks_learned["mask_h"], masks_learned["mask_o"]
    fused = []
    rng = np.random.default_rng(O0E0_SEED + hash(split) % 997)
    n_frames = obs["rgb_h"].shape[0]
    for i in range(n_frames):
        fused.append(fuse_points(obs["xyz_h"][i], mh[i], obs["xyz_o"][i], mo[i], rng=rng))
    fused_arr = np.empty(n_frames, dtype=object)
    for i, P in enumerate(fused):
        fused_arr[i] = P
    np.savez_compressed(
        d / "obs.npz",
        mask_h=mh,
        mask_o=mo,
        fused_points_B=fused_arr,
        coverage_any=np.asarray(obs["vis_h"]) | np.asarray(obs["vis_o"]),
    )


def _load_p0_split(p0_root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obs_path = p0_root / "cache" / split / "obs.npz"
    gt_path = p0_root / "cache" / split / "gt.npz"
    if not obs_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"O0E0P0 cache missing for split={split}: {p0_root}")
    z_obs = np.load(obs_path)
    obs = {k: z_obs[k] for k in z_obs.files}
    z_gt = np.load(gt_path)
    gt = {k: np.asarray(z_gt[k]) for k in z_gt.files}
    return obs, gt


def _eval_branch(
    mode: str,
    *,
    obs_te: dict[str, Any],
    gt_te: dict[str, Any],
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    n_const: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, list[float]]:
    n_frames = gt_te["p_gt"].shape[0]
    per_frame: list[dict[str, Any]] = []
    e_ax: list[float] = []
    p_hats: list[np.ndarray] = []
    full_R_diag: list[float] = []
    for i in range(n_frames):
        obs_i = {
            "fused_points_B": np.asarray(obs_te["fused_points_B"][i], dtype=np.float64),
            "mask_h": obs_te["mask_h"][i],
            "mask_o": obs_te["mask_o"][i],
        }
        p_hat, n_hat, meta = estimate_effective_pose(
            obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode=mode, n_const=n_const
        )
        ea = e_axis_deg(n_hat, gt_te["n_gt"][i])
        ep = float(np.linalg.norm(p_hat - gt_te["p_gt"][i])) if np.isfinite(p_hat).all() else float("nan")
        R0 = R0_from_n(n_hat)
        Rgt = _quat_to_R(gt_te["quat_gt"][i])
        Rel = R0.T @ Rgt
        c = float(np.clip((np.trace(Rel) - 1.0) * 0.5, -1.0, 1.0))
        dR = float(np.degrees(np.arccos(c)))
        e_ax.append(ea)
        p_hats.append(p_hat)
        full_R_diag.append(dR)
        if mode == "B2":
            p_surf = meta.get("p_surf")
            per_frame.append(
                {
                    "frame": int(i),
                    "n_gt": gt_te["n_gt"][i].tolist(),
                    "n_hat": n_hat.tolist(),
                    "axis_err": ea,
                    "p_hat": p_hat.tolist() if np.isfinite(p_hat).all() else [None] * 3,
                    "p_gt": gt_te["p_gt"][i].tolist(),
                    "p_surf": p_surf.tolist() if p_surf is not None and np.asarray(p_surf).shape == (3,) else [None] * 3,
                    "position_error": ep,
                    "beta_deg": float(gt_te["beta_deg"][i]) if "beta_deg" in gt_te else float("nan"),
                    "gamma_rad": float(gt_te["gamma_rad"][i]) if "gamma_rad" in gt_te else float("nan"),
                    "S_best": meta.get("S_best"),
                    "S_second": meta.get("S_second"),
                    "objective_margin": meta.get("margin"),
                    "full_R_diag": dR,
                    "diagnostic_only": True,
                    "used_by_gate": False,
                }
            )
    e_ax_arr = np.asarray(e_ax, dtype=np.float64)
    p_hats_arr = np.stack(p_hats)
    pos = _score_pose(p_hats_arr, gt_te["p_gt"])
    stats = {
        "median_e_axis_deg": float(np.nanmedian(e_ax_arr)),
        "p90_e_axis_deg": float(np.nanpercentile(e_ax_arr, 90)),
        "f90": float(np.nanmean(e_ax_arr >= F90_THRESH)),
        "mean_e_axis_deg": float(np.nanmean(e_ax_arr)),
        "E_p": pos["E_p"],
        "median_ep_m": pos["median_ep_m"],
        "median_ep_cm": float(pos["median_ep_m"] * 100.0),
        "strong_2cm": bool(pos["median_ep_m"] <= STRONG_MED),
        "median_full_R_diag_deg": float(np.nanmedian(full_R_diag)),
        "p90_full_R_diag_deg": float(np.nanpercentile(full_R_diag, 90)),
        "n": int(np.isfinite(e_ax_arr).sum()),
    }
    return stats, per_frame, e_ax_arr, full_R_diag


def _tilt_diagnostics(e_axis: np.ndarray, beta_deg: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for b in TILT_DEG:
        m = np.isclose(beta_deg, b)
        if not m.any():
            continue
        ea = e_axis[m]
        out[str(float(b))] = {
            "median_e_axis_deg": float(np.nanmedian(ea)),
            "p90_e_axis_deg": float(np.nanpercentile(ea, 90)),
            "n": int(m.sum()),
        }
    return out


def _yaw_octant_diagnostics(
    e_axis: np.ndarray, gamma_rad: np.ndarray, beta_deg: np.ndarray, *, beta_ref: float
) -> dict[str, dict[str, float]]:
    oct = (np.asarray(gamma_rad) % (2.0 * np.pi) / (2.0 * np.pi / YAW_OCTANTS)).astype(int) % YAW_OCTANTS
    m_beta = np.isclose(beta_deg, beta_ref)
    out: dict[str, dict[str, float]] = {}
    for o in range(YAW_OCTANTS):
        mm = m_beta & (oct == o)
        if not mm.any():
            continue
        ea = e_axis[mm]
        out[str(o)] = {
            "median_e_axis_deg": float(np.nanmedian(ea)),
            "p90_e_axis_deg": float(np.nanpercentile(ea, 90)),
            "n": int(mm.sum()),
        }
    return out


def _oracle_axis_position_ceiling(
    per_frame: list[dict[str, Any]], gt_te: dict[str, Any], *, delta_y: float
) -> dict[str, Any]:
    e_ceil: list[float] = []
    e_b2: list[float] = []
    for i, fr in enumerate(per_frame):
        p_surf = np.asarray(fr.get("p_surf"), dtype=np.float64)
        if p_surf.shape != (3,) or not np.isfinite(p_surf).all():
            continue
        n_gt = gt_te["n_gt"][i]
        p_ceil = p_surf - float(delta_y) * n_gt
        e_ceil.append(float(np.linalg.norm(p_ceil - gt_te["p_gt"][i])))
        e_b2.append(float(fr["position_error"]))
    e_ceil_arr = np.asarray(e_ceil, dtype=np.float64)
    e_b2_arr = np.asarray(e_b2, dtype=np.float64)
    med_ceil = float(np.nanmedian(e_ceil_arr)) if len(e_ceil_arr) else float("nan")
    med_b2 = float(np.nanmedian(e_b2_arr)) if len(e_b2_arr) else float("nan")
    return {
        "diagnostic_only": True,
        "used_by_gate": False,
        "median_ep_ceil_m": med_ceil,
        "median_ep_ceil_cm": float(med_ceil * 100.0),
        "median_ep_b2_m": med_b2,
        "median_ep_b2_cm": float(med_b2 * 100.0),
        "ceil_passes_g2_median": bool(med_ceil <= MED_MAX) if len(e_ceil_arr) else False,
        "b2_passes_g2_median": bool(med_b2 <= MED_MAX) if len(e_b2_arr) else False,
        "n_frames": int(len(e_ceil_arr)),
    }


def run_rtwx_o0e0r0(output: str | Path, config: RTWXO0E0R0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0R0Config(output=str(output)))
    root = Path(output).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r0] p0_cache={p0_root} smoke={cfg.smoke}", flush=True)

    p0_summary = p0_root / "summary.json"
    if p0_summary.is_file():
        import json

        p0_res = json.loads(p0_summary.read_text())
        if p0_res.get("pattern") != "controlled_pose_observation_qualified":
            print(f"[rtwx-o0e0r0] WARN: P0 pattern={p0_res.get('pattern')} (expected qualified)", flush=True)
    else:
        print("[rtwx-o0e0r0] WARN: P0 summary.json missing", flush=True)

    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    delta_y = float(delta_O[1])

    repo = Path(cfg.robotwin_repo)
    cad_glb = repo / "assets/objects/021_cup/visual/base0.glb"
    if not cad_glb.is_file():
        hs = np.linspace(0, 0.088, 64)
        cad = []
        for h in hs:
            r = 0.02 + 0.22 * (h / 0.088)
            for th in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                cad.append([r * np.cos(th), h, r * np.sin(th)])
        cad_pts = np.asarray(cad, dtype=np.float64)
    else:
        cad_pts = _fps(load_scaled_cad(str(repo), 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
    from scipy.spatial import cKDTree

    tree = cKDTree(cad_hr_profile(cad_pts))
    sphere = fibonacci_sphere(K_SPHERE, SPHERE_SEED)

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "p0_cache": str(p0_root),
        "seed": cfg.seed,
        "unet_seed": O0E0_SEED,
        "delta_O": delta_O.tolist(),
        "delta_y": delta_y,
        "G_geom": "SO(2)_y",
        "no_yaw_in_primary": True,
        "obs_gt_isolated": True,
        "b2_frozen": True,
        "sphere_K": K_SPHERE,
        "top_Kr": TOP_KR,
        "refine_degs": list(REFINE_DEGS),
        "gates": {"det": P_DET_MIN, "axis_med": MED_ER_MAX, "axis_p90": P90_ER_MAX, "Ep": EP_MAX, "med_ep_m": MED_MAX},
    }
    _write_json(root / "header.json", header)

    pools: dict[str, dict[str, Any]] = {}
    gts: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        obs, gt = _load_p0_split(p0_root, split)
        pools[split] = obs
        gts[split] = gt
        print(f"[rtwx-o0e0r0] loaded P0 {split}: n={gt['p_gt'].shape[0]}", flush=True)

    print("[rtwx-o0e0r0] train frozen U-Net (O0G2 family) on P0 train", flush=True)
    tr, va = pools["train"], pools["val"]
    rgb_tr = np.concatenate([tr["rgb_h"], tr["rgb_o"]], 0)
    m_tr = np.concatenate([tr["mask_h"], tr["mask_o"]], 0)
    rgb_va = np.concatenate([va["rgb_h"], va["rgb_o"]], 0)
    m_va = np.concatenate([va["mask_h"], va["mask_o"]], 0)
    model = _train_unet(rgb_tr, m_tr, rgb_va, m_va, seed=O0E0_SEED, epochs=cfg.epochs_unet)
    _cuda_gc()

    learned: dict[str, dict[str, np.ndarray]] = {}
    for split, obs in pools.items():
        mh = _predict_masks(model, obs["rgb_h"])
        mo = _predict_masks(model, obs["rgb_o"])
        learned[split] = {"mask_h": mh, "mask_o": mo}
        _write_science_cache(root, split, obs, learned[split])
    _cuda_gc()

    obs_path = root / "science_cache" / "test" / "obs.npz"
    z_te = np.load(obs_path, allow_pickle=True)
    obs_te = {k: z_te[k] for k in z_te.files}
    gt_te = gts["test"]
    vis_any = np.asarray(obs_te["coverage_any"], bool)
    mh = learned["test"]["mask_h"]
    mo = learned["test"]["mask_o"]
    nonempty = (mh.reshape(mh.shape[0], -1).any(1)) | (mo.reshape(mo.shape[0], -1).any(1))
    p_det = float(np.mean(nonempty[vis_any])) if vis_any.any() else 0.0
    g0 = bool(p_det >= P_DET_MIN)
    print(f"[rtwx-o0e0r0] G0 det P(M≠∅|V)={p_det:.3f} ok={g0}", flush=True)

    if not g0:
        result = {
            "header": header,
            "pattern": "controlled_pose_detection_failure",
            "G0": {"ok": False, "P_det": p_det, "note": "STOP before B0/B1/B2"},
            "B2_untested": True,
            "unlocks_o0e1_prereg": False,
            "unlocks_o1": False,
            "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0r0] pattern=controlled_pose_detection_failure", flush=True)
        return result

    gt_tr = gts["train"]
    n_sum = gt_tr["n_gt"].sum(0)
    n_const = n_sum / max(np.linalg.norm(n_sum), 1e-12)

    branch_stats: dict[str, dict[str, Any]] = {}
    per_frame_b2: list[dict[str, Any]] = []
    e_axis_b2 = np.array([], dtype=np.float64)

    for mode in ("B0", "B1", "B2"):
        print(f"[rtwx-o0e0r0] eval {mode}", flush=True)
        stats, per_frame, e_ax, _ = _eval_branch(
            mode,
            obs_te=obs_te,
            gt_te=gt_te,
            delta_y=delta_y,
            tree=tree,
            sphere=sphere,
            n_const=n_const,
        )
        branch_stats[mode] = stats
        if mode == "B2":
            per_frame_b2 = per_frame
            e_axis_b2 = e_ax
        print(
            f"[rtwx-o0e0r0] {mode} med_axis={stats['median_e_axis_deg']:.2f} "
            f"p90={stats['p90_e_axis_deg']:.2f} Ep={stats['E_p']:.4f} "
            f"med_ep_cm={stats['median_ep_cm']:.2f}",
            flush=True,
        )

    b2, b0 = branch_stats["B2"], branch_stats["B0"]
    g1 = _branch_passes_g1(b2)
    g2 = _branch_passes_g2(b2)
    b0_excluded = not _branch_passes_g1(b0)
    pattern = _pattern(g0=True, g1=g1, g2=g2, b0_excluded=b0_excluded)

    beta_deg = np.asarray(gt_te["beta_deg"], dtype=np.float64)
    gamma_rad = np.asarray(gt_te["gamma_rad"], dtype=np.float64)
    diag_tilt = _tilt_diagnostics(e_axis_b2, beta_deg)
    diag_yaw = _yaw_octant_diagnostics(e_axis_b2, gamma_rad, beta_deg, beta_ref=YAW_OCTANT_DIAG_BETA)
    diag_ceil = _oracle_axis_position_ceiling(per_frame_b2, gt_te, delta_y=delta_y)

    dG_checks = []
    for i in range(min(5, len(per_frame_b2))):
        Rgt = _quat_to_R(gt_te["quat_gt"][i])
        n_hat = np.asarray(per_frame_b2[i]["n_hat"])
        R0 = R0_from_n(n_hat)
        dG_checks.append(abs(d_G_deg(R0, Rgt) - e_axis_deg(n_hat, gt_te["n_gt"][i])))

    result = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": True, "P_det": p_det},
        "B0_constant": b0,
        "B1_pca": branch_stats["B1"],
        "B2_quotient_cad": b2,
        "G1_axis": {"ok": g1, **{k: b2[k] for k in ("median_e_axis_deg", "p90_e_axis_deg", "f90")}},
        "G2_position": {"ok": g2, "E_p": b2["E_p"], "median_ep_m": b2["median_ep_m"], "strong_2cm": b2["strong_2cm"]},
        "B0_excluded": {"ok": b0_excluded, "B0_passes_G1": not b0_excluded},
        "full_R_diagnostic": {
            "diagnostic_only": True,
            "used_by_gate": False,
            "median_deg": b2["median_full_R_diag_deg"],
            "p90_deg": b2["p90_full_R_diag_deg"],
        },
        "mechanism_diagnostics": {
            "diagnostic_only": True,
            "tilt_bins": diag_tilt,
            "yaw_octants_beta_deg": YAW_OCTANT_DIAG_BETA,
            "yaw_octants": diag_yaw,
            "oracle_axis_position_ceiling": diag_ceil,
        },
        "dG_vs_eaxis_max_abs": float(max(dG_checks)) if dG_checks else float("nan"),
        "claim_scope": "controlled_021_cup_rgbd_only",
        "unlocks_o0e1_prereg": pattern == "effective_pose_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    _write_json(root / "per_frame_b2.json", {"frames": per_frame_b2[:500]})
    print(
        f"[rtwx-o0e0r0] pattern={pattern} G1={g1} G2={g2} B0_excluded={b0_excluded} "
        f"unlocks_e1={result['unlocks_o0e1_prereg']}",
        flush=True,
    )
    return result

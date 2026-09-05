"""RTWX-O0E0R1: oracle-substitution reference/centering audit on P0 cache. No train."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    REFINE_DEGS,
    SPHERE_SEED,
    TOP_KR,
    _tangent_neighbors,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
    fuse_points,
    obs_hr,
    p_surf_from_cloud,
)
from .rtwx_o0e0p0 import TILT_DEG, YAW_OCTANTS
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r1.reference_centering_audit.v1"
SEED = 39601
YAW_OCTANT_DIAG_BETA = 20.0
DELTA_TILT_SPREAD_M = 0.015
EP_IMPROVE_MASK_M = 0.03


@dataclass(frozen=True)
class RTWXO0E0R1Config:
    output: str = "runs/rtwx_o0e0r1"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r0_cache: str = "runs/rtwx_o0e0r0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R1 must not write there")


def _axis_diag_pass(median_deg: float, p90_deg: float) -> bool:
    return bool(median_deg <= MED_ER_MAX and p90_deg <= P90_ER_MAX)


def _axis_stats(e_axis: np.ndarray) -> dict[str, float]:
    e = np.asarray(e_axis, dtype=np.float64)
    return {
        "median_e_axis_deg": float(np.nanmedian(e)),
        "p90_e_axis_deg": float(np.nanpercentile(e, 90)),
        "mean_e_axis_deg": float(np.nanmean(e)),
        "n": int(np.isfinite(e).sum()),
    }


def _ep_stats(err_m: np.ndarray) -> dict[str, float]:
    e = np.asarray(err_m, dtype=np.float64)
    return {
        "median_ep_m": float(np.nanmedian(e)),
        "median_ep_cm": float(np.nanmedian(e) * 100.0),
        "p90_ep_m": float(np.nanpercentile(e, 90)),
        "n": int(np.isfinite(e).sum()),
    }


def S_at_center(P: np.ndarray, p_center: np.ndarray, n: np.ndarray, tree: Any) -> float:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    c = obs_hr(P, p_center, n)
    if c.shape[0] == 0:
        return float("inf")
    d, _ = tree.query(c, k=1)
    return float(np.mean(np.asarray(d, dtype=np.float64) ** 2))


def estimate_axis_b2_at_center(
    P: np.ndarray,
    p_center: np.ndarray,
    tree: Any,
    *,
    sphere: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Frozen B2 search with fixed world-frame center (oracle or reference)."""
    if P.shape[0] < 8:
        from .rtwx_o0e0 import EY

        return EY.copy(), {"S_best": float("inf"), "S_second": float("inf"), "margin": float("nan")}
    p_center = np.asarray(p_center, dtype=np.float64).reshape(3)
    scores = [S_at_center(P, p_center, n, tree) for n in sphere]
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    cands = [sphere[i] for i in order[:TOP_KR]]
    best_n, best_s = cands[0], float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else best_s
    for n0 in list(cands):
        cur_n, cur_s = n0, S_at_center(P, p_center, n0, tree)
        for deg in REFINE_DEGS:
            improved = True
            while improved:
                improved = False
                for nb in _tangent_neighbors(cur_n, deg):
                    s = S_at_center(P, p_center, nb, tree)
                    if s + 1e-12 < cur_s:
                        cur_n, cur_s = nb, s
                        improved = True
        if cur_s < best_s:
            second = best_s
            best_n, best_s = cur_n, cur_s
        elif cur_s < second and not np.allclose(cur_n, best_n):
            second = cur_s
    return best_n / np.linalg.norm(best_n), {"S_best": best_s, "S_second": second, "margin": second - best_s}


def _load_p0_split(p0_root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = np.load(p0_root / "cache" / split / "obs.npz")
    gt = np.load(p0_root / "cache" / split / "gt.npz")
    return {k: obs[k] for k in obs.files}, {k: np.asarray(gt[k]) for k in gt.files}


def _load_r0_learned_masks(r0_root: Path, split: str) -> dict[str, np.ndarray]:
    z = np.load(r0_root / "science_cache" / split / "obs.npz")
    return {"mask_h": np.asarray(z["mask_h"]), "mask_o": np.asarray(z["mask_o"])}


def _fused_clouds(
    obs: dict[str, Any],
    masks: dict[str, np.ndarray],
    *,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = obs["xyz_h"].shape[0]
    out = []
    for i in range(n):
        out.append(
            fuse_points(obs["xyz_h"][i], masks["mask_h"][i], obs["xyz_o"][i], masks["mask_o"][i], rng=rng)
        )
    return out


def _tilt_bucket(values: np.ndarray, beta_deg: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for b in TILT_DEG:
        m = np.isclose(beta_deg, b)
        if not m.any():
            continue
        v = values[m]
        out[str(float(b))] = {
            "median": float(np.nanmedian(v)),
            "p90": float(np.nanpercentile(v, 90)),
            "n": int(m.sum()),
        }
    return out


def _yaw_bucket(values: np.ndarray, gamma_rad: np.ndarray, beta_deg: np.ndarray, *, beta_ref: float) -> dict[str, dict[str, float]]:
    oct = (np.asarray(gamma_rad) % (2.0 * np.pi) / (2.0 * np.pi / YAW_OCTANTS)).astype(int) % YAW_OCTANTS
    m_beta = np.isclose(beta_deg, beta_ref)
    out: dict[str, dict[str, float]] = {}
    for o in range(YAW_OCTANTS):
        mm = m_beta & (oct == o)
        if not mm.any():
            continue
        v = values[mm]
        out[str(o)] = {"median": float(np.nanmedian(v)), "p90": float(np.nanpercentile(v, 90)), "n": int(mm.sum())}
    return out


def _eval_b2_branch(
    clouds: list[np.ndarray],
    gt_te: dict[str, Any],
    *,
    center_mode: str,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    p_surfs: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    n_frames = gt_te["p_gt"].shape[0]
    e_axis: list[float] = []
    per: list[dict[str, float]] = []
    for i in range(n_frames):
        P = clouds[i]
        p_gt = gt_te["p_gt"][i]
        if center_mode == "oracle_p":
            p_center = p_gt
        elif center_mode == "reference":
            p_surf = p_surfs[i] if p_surfs is not None else p_surf_from_cloud(P)
            # reference center uses candidate n inside search; pass p_surf+delta_y via custom loop
            n_hat, meta = _estimate_axis_b2_reference(P, p_surf, delta_y, tree, sphere=sphere)
            ea = e_axis_deg(n_hat, gt_te["n_gt"][i])
            e_axis.append(ea)
            per.append({**meta, "axis_err": ea})
            continue
        else:
            raise ValueError(center_mode)
        n_hat, meta = estimate_axis_b2_at_center(P, p_center, tree, sphere=sphere)
        ea = e_axis_deg(n_hat, gt_te["n_gt"][i])
        e_axis.append(ea)
        per.append({**meta, "axis_err": ea})
    return np.asarray(e_axis, dtype=np.float64), per


def _estimate_axis_b2_reference(
    P: np.ndarray,
    p_surf: np.ndarray,
    delta_y: float,
    tree: Any,
    *,
    sphere: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """B2 with p_pose(n)=p_surf-delta_y*n (formal reference coupling)."""
    if P.shape[0] < 8:
        from .rtwx_o0e0 import EY

        return EY.copy(), {"S_best": float("inf"), "S_second": float("inf"), "margin": float("nan")}

    def score(n: np.ndarray) -> float:
        p_pose = np.asarray(p_surf, dtype=np.float64).reshape(3) - float(delta_y) * np.asarray(n, dtype=np.float64).reshape(3)
        return S_at_center(P, p_pose, n, tree)

    scores = [score(n) for n in sphere]
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    cands = [sphere[i] for i in order[:TOP_KR]]
    best_n, best_s = cands[0], float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else best_s
    for n0 in list(cands):
        cur_n, cur_s = n0, score(n0)
        for deg in REFINE_DEGS:
            improved = True
            while improved:
                improved = False
                for nb in _tangent_neighbors(cur_n, deg):
                    s = score(nb)
                    if s + 1e-12 < cur_s:
                        cur_n, cur_s = nb, s
                        improved = True
        if cur_s < best_s:
            second = best_s
            best_n, best_s = cur_n, cur_s
        elif cur_s < second and not np.allclose(cur_n, best_n):
            second = cur_s
    return best_n / np.linalg.norm(best_n), {"S_best": best_s, "S_second": second, "margin": second - best_s}


def _patterns(
    *,
    ref_not_invariant: bool,
    ceil_pass: bool,
    formal_pass: bool,
    pgT_pass: bool,
    mask_improves_pos: bool,
) -> list[str]:
    pats = ["oracle_audit_complete"]
    if ref_not_invariant:
        pats.append("reference_offset_not_invariant")
    if ceil_pass:
        pats.append("quotient_viable_under_oracle_cloud")
    else:
        pats.append("quotient_insufficient_under_oracle")
    if not formal_pass and pgT_pass:
        pats.append("axis_failure_reference_coupled")
    if mask_improves_pos:
        pats.append("axis_failure_segmentation_coupled")
    return pats


def run_rtwx_o0e0r1(output: str | Path, config: RTWXO0E0R1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXO0E0R1Config(output=str(output))
    root = Path(output).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    r0_root = Path(cfg.r0_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0r1] p0={p0_root} r0={r0_root}", flush=True)

    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    delta_y = float(delta_O[1])

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
        "r0_cache": str(r0_root),
        "seed": cfg.seed,
        "delta_O": delta_O.tolist(),
        "delta_y": delta_y,
        "b2_frozen": True,
        "no_training": True,
        "r0_pattern_unchanged": True,
    }
    _write_json(root / "header.json", header)

    obs_te, gt_te = _load_p0_split(p0_root, "test")
    learned = _load_r0_learned_masks(r0_root, "test")
    gt_masks = {"mask_h": np.asarray(obs_te["mask_h"]), "mask_o": np.asarray(obs_te["mask_o"])}
    clouds_learned = _fused_clouds(obs_te, learned, seed=cfg.seed)
    clouds_gt = _fused_clouds(obs_te, gt_masks, seed=cfg.seed + 1)
    p_surfs_learned = [p_surf_from_cloud(P) for P in clouds_learned]
    p_surfs_gt = [p_surf_from_cloud(P) for P in clouds_gt]
    n_frames = gt_te["p_gt"].shape[0]
    beta_deg = np.asarray(gt_te["beta_deg"], dtype=np.float64)
    gamma_rad = np.asarray(gt_te["gamma_rad"], dtype=np.float64)

    # ---- A: delta_O invariance audit ----
    e_delta = np.zeros(n_frames, dtype=np.float64)
    delta_gt_obj = np.zeros((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        Rgt = _quat_to_R(gt_te["quat_gt"][i])
        delta_w = p_surfs_learned[i] - gt_te["p_gt"][i]
        delta_gt_obj[i] = Rgt.T @ delta_w
        e_delta[i] = float(np.linalg.norm(delta_gt_obj[i] - delta_O))
    branch_a = {
        "diagnostic_only": True,
        "delta_O_frozen": delta_O.tolist(),
        "median_e_delta_m": float(np.nanmedian(e_delta)),
        "median_e_delta_cm": float(np.nanmedian(e_delta) * 100.0),
        "p90_e_delta_m": float(np.nanpercentile(e_delta, 90)),
        "tilt_bins": _tilt_bucket(e_delta, beta_deg),
        "yaw_octants_beta_deg": YAW_OCTANT_DIAG_BETA,
        "yaw_octants": _yaw_bucket(e_delta, gamma_rad, beta_deg, beta_ref=YAW_OCTANT_DIAG_BETA),
        "delta_gt_obj_median": np.nanmedian(delta_gt_obj, axis=0).tolist(),
    }
    tilt_medians = [v["median"] for v in branch_a["tilt_bins"].values()]
    ref_not_invariant = bool(tilt_medians and (max(tilt_medians) - min(tilt_medians) > DELTA_TILT_SPREAD_M))
    print(f"[rtwx-o0e0r1] A median e_delta={branch_a['median_e_delta_cm']:.2f}cm invariant={not ref_not_invariant}", flush=True)

    # ---- B: mask substitution position (GT axis) ----
    ep_learned, ep_gtmask = [], []
    for i in range(n_frames):
        n_gt = gt_te["n_gt"][i]
        p_ceil_l = p_surfs_learned[i] - delta_y * n_gt
        p_ceil_g = p_surfs_gt[i] - delta_y * n_gt
        ep_learned.append(float(np.linalg.norm(p_ceil_l - gt_te["p_gt"][i])))
        ep_gtmask.append(float(np.linalg.norm(p_ceil_g - gt_te["p_gt"][i])))
    ep_learned = np.asarray(ep_learned)
    ep_gtmask = np.asarray(ep_gtmask)
    branch_b = {
        "diagnostic_only": True,
        "learned_mask": _ep_stats(ep_learned),
        "gt_mask": _ep_stats(ep_gtmask),
        "median_improvement_cm": float((np.nanmedian(ep_learned) - np.nanmedian(ep_gtmask)) * 100.0),
    }
    mask_improves_pos = bool(np.nanmedian(ep_learned) - np.nanmedian(ep_gtmask) > EP_IMPROVE_MASK_M)
    print(
        f"[rtwx-o0e0r1] B ep learned={branch_b['learned_mask']['median_ep_cm']:.2f}cm "
        f"gtmask={branch_b['gt_mask']['median_ep_cm']:.2f}cm",
        flush=True,
    )

    # ---- C: B2_pGT (learned cloud, oracle center) ----
    e_c, _ = _eval_b2_branch(
        clouds_learned, gt_te, center_mode="oracle_p", delta_y=delta_y, tree=tree, sphere=sphere
    )
    branch_c = {"diagnostic_only": True, "learned_mask_oracle_p": _axis_stats(e_c)}
    pgT_pass = _axis_diag_pass(branch_c["learned_mask_oracle_p"]["median_e_axis_deg"], branch_c["learned_mask_oracle_p"]["p90_e_axis_deg"])
    print(f"[rtwx-o0e0r1] C B2_pGT med={branch_c['learned_mask_oracle_p']['median_e_axis_deg']:.2f}°", flush=True)

    # ---- D: 2x2 grid ----
    e_formal, _ = _eval_b2_branch(
        clouds_learned,
        gt_te,
        center_mode="reference",
        delta_y=delta_y,
        tree=tree,
        sphere=sphere,
        p_surfs=p_surfs_learned,
    )
    e_seg, _ = _eval_b2_branch(
        clouds_gt,
        gt_te,
        center_mode="reference",
        delta_y=delta_y,
        tree=tree,
        sphere=sphere,
        p_surfs=p_surfs_gt,
    )
    e_ctr, _ = _eval_b2_branch(
        clouds_learned, gt_te, center_mode="oracle_p", delta_y=delta_y, tree=tree, sphere=sphere
    )
    e_ceil, _ = _eval_b2_branch(
        clouds_gt, gt_te, center_mode="oracle_p", delta_y=delta_y, tree=tree, sphere=sphere
    )
    grid = {
        "formal_learned_reference": _axis_stats(e_formal),
        "seg_gtmask_reference": _axis_stats(e_seg),
        "ctr_learned_oracle_p": _axis_stats(e_ctr),
        "ceil_gtmask_oracle_p": _axis_stats(e_ceil),
    }
    formal_pass = _axis_diag_pass(grid["formal_learned_reference"]["median_e_axis_deg"], grid["formal_learned_reference"]["p90_e_axis_deg"])
    ceil_pass = _axis_diag_pass(grid["ceil_gtmask_oracle_p"]["median_e_axis_deg"], grid["ceil_gtmask_oracle_p"]["p90_e_axis_deg"])
    print(
        f"[rtwx-o0e0r1] D formal={grid['formal_learned_reference']['median_e_axis_deg']:.1f}° "
        f"ceil={grid['ceil_gtmask_oracle_p']['median_e_axis_deg']:.1f}°",
        flush=True,
    )

    patterns = _patterns(
        ref_not_invariant=ref_not_invariant,
        ceil_pass=ceil_pass,
        formal_pass=formal_pass,
        pgT_pass=pgT_pass,
        mask_improves_pos=mask_improves_pos,
    )
    primary = "quotient_viable_under_oracle_cloud" if ceil_pass else "quotient_insufficient_under_oracle"

    result = {
        "header": header,
        "pattern": primary,
        "patterns": patterns,
        "A_reference_offset": branch_a,
        "reference_offset_not_invariant": ref_not_invariant,
        "B_mask_position_ceiling": branch_b,
        "C_B2_oracle_p_learned": branch_c,
        "D_oracle_grid_2x2": grid,
        "interpretation": {
            "ceil_axis_pass": ceil_pass,
            "formal_axis_pass": formal_pass,
            "B2_pGT_pass": pgT_pass,
            "mask_position_coupled": mask_improves_pos,
            "reference_offset_view_dependent": ref_not_invariant,
            "r0_effective_axis_failure_unchanged": True,
        },
        "unlocks_o0e1": False,
        "unlocks_o1": False,
        "unlocks_objective_hypothesis_cell": not ceil_pass,
        "unlocks_reference_repair_cell": ceil_pass,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0r1] pattern={primary} patterns={patterns}", flush=True)
    return result

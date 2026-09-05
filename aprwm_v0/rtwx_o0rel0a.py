"""RTWX-O0REL0A: ego-transform vs visibility-overlap audit (evaluator-only, zero train)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH
from .rtwx_o0e0 import CAD_FPS_SEED, N_CAD, fuse_points
from .rtwx_o0e0r1 import _axis_stats, _ep_stats
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0rel0 import (
    REGIMES,
    _eval_pair,
    _load_cache,
    _overlap_ratio,
    _resolve_apr,
)
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0REL0A_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0rel0a.overlap_audit.v1"
REL0_FORMAL_PATTERN = "ego_motion_compensation_failure"
OVERLAP_BINS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0))
CAD_DIST_GOOD_FRAC = 0.05
GT_MASK_GOOD_AXIS_DEG = 5.0


@dataclass(frozen=True)
class RTWXO0REL0AConfig:
    output: str = "runs/rtwx_o0rel0a"
    rel0_run: str = "runs/rtwx_o0rel0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = 45602
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0REL0A must not write there")


def _load_cad_pts3d(repo: Path) -> np.ndarray:
    cad_glb = repo / "assets/objects/021_cup/visual/base0.glb"
    if cad_glb.is_file():
        return _fps(load_scaled_cad(str(repo), 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
    hs = np.linspace(0, 0.088, 64)
    return np.array(
        [[0.02 + 0.22 * (h / 0.088) * np.cos(th), h, 0.02 + 0.22 * (h / 0.088) * np.sin(th)]
         for h in hs for th in np.linspace(0, 2 * np.pi, 32, endpoint=False)],
        dtype=np.float64,
    )


def _gt_fused_cloud(
    obs: dict[str, Any],
    gt: dict[str, Any],
    pair_i: int,
    frame_t: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    return fuse_points(
        obs[f"xyz{frame_t}_h"][pair_i],
        gt[f"mask{frame_t}_h"][pair_i],
        obs[f"xyz{frame_t}_o"][pair_i],
        gt[f"mask{frame_t}_o"][pair_i],
        rng=rng,
    )


def _cloud_cad_median_dist(cloud_b: np.ndarray, r: np.ndarray, p: np.ndarray, cad_obj: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    p_cloud = np.asarray(cloud_b, dtype=np.float64)
    if p_cloud.shape[0] == 0:
        return float("nan")
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    co = (r.T @ (p_cloud - p).T).T
    d, _ = cKDTree(cad_obj).query(co, k=1)
    return float(np.median(d))


def diagnostic_a_gtmask_icp_c0(
    obs: dict[str, Any],
    gt: dict[str, Any],
    *,
    d_o: float,
    rng: np.random.Generator,
    s0_per_pair: list[dict[str, Any]],
) -> dict[str, Any]:
    e_gt: list[float] = []
    e_s0 = [float(x["e_delta_n_deg"]) for x in s0_per_pair if not x.get("skip")]
    for i in range(gt["p0"].shape[0]):
        c0 = _gt_fused_cloud(obs, gt, i, 0, rng=rng)
        c1 = _gt_fused_cloud(obs, gt, i, 1, rng=rng)
        if c0.shape[0] < 8 or c1.shape[0] < 8:
            continue
        ev = _eval_pair(
            c0, c1,
            R0=gt["R0"][i], p0=gt["p0"][i], n0=gt["n0"][i],
            R1=gt["R1"][i], p1=gt["p1"][i], n1=gt["n1"][i],
            D_O=d_o,
        )
        e_gt.append(ev["e_delta_n_deg"])
    ax_gt = _axis_stats(np.asarray(e_gt, dtype=np.float64))
    ax_s0 = _axis_stats(np.asarray(e_s0, dtype=np.float64))
    seg_coupled = bool(ax_gt["median_e_axis_deg"] < GT_MASK_GOOD_AXIS_DEG)
    return {
        "regime": "c0",
        "n_pairs": int(len(e_gt)),
        "gt_mask_icp_axis": ax_gt,
        "s0_path_axis": ax_s0,
        "segmentation_excluded_as_primary": bool(not seg_coupled),
        "interpretation": (
            "segmentation_coupled_c0_failure" if seg_coupled else "segmentation_excluded_overlap_primary"
        ),
    }


def diagnostic_b_cad_consistency_c0(
    obs: dict[str, Any],
    gt: dict[str, Any],
    cad_obj: np.ndarray,
    *,
    d_o: float,
    rng: np.random.Generator,
    s0_per_pair: list[dict[str, Any]],
) -> dict[str, Any]:
    d0_list, d1_list, rho_list = [], [], []
    for i in range(gt["p0"].shape[0]):
        c0 = _gt_fused_cloud(obs, gt, i, 0, rng=rng)
        c1 = _gt_fused_cloud(obs, gt, i, 1, rng=rng)
        r0, p0 = gt["R0"][i], gt["p0"][i]
        d0_list.append(_cloud_cad_median_dist(c0, r0, p0, cad_obj))
        d1_list.append(_cloud_cad_median_dist(c1, r0, p0, cad_obj))
        r_gt, t_gt = np.eye(3), np.zeros(3)
        rho_list.append(float(s0_per_pair[i].get("rho_overlap", _overlap_ratio(c0, c1, r_gt, t_gt, d_o))))
    d0 = np.asarray(d0_list, dtype=np.float64)
    d1 = np.asarray(d1_list, dtype=np.float64)
    rho = np.asarray(rho_list, dtype=np.float64)
    geom_ok = bool(
        np.nanmedian(d0) <= CAD_DIST_GOOD_FRAC * d_o
        and np.nanmedian(d1) <= CAD_DIST_GOOD_FRAC * d_o
        and np.nanmedian(rho) < 0.1
    )
    return {
        "regime": "c0",
        "d0_m": _ep_stats(d0),
        "d1_m": _ep_stats(d1),
        "d0_over_DO_median": float(np.nanmedian(d0) / d_o),
        "d1_over_DO_median": float(np.nanmedian(d1) / d_o),
        "rho_overlap": {
            "median": float(np.nanmedian(rho)),
            "p90": float(np.nanpercentile(rho, 90)),
        },
        "frames_geometrically_consistent_low_overlap": geom_ok,
    }


def diagnostic_c_overlap_curve(
    pooled: list[dict[str, Any]],
) -> dict[str, Any]:
    bins_out: list[dict[str, Any]] = []
    for lo, hi in OVERLAP_BINS:
        sub = [x for x in pooled if lo <= float(x["rho_overlap"]) < hi]
        if not sub:
            bins_out.append({
                "bin": [lo, hi],
                "n": 0,
                "median_e_delta_n_deg": float("nan"),
                "p90_e_delta_n_deg": float("nan"),
            })
            continue
        e = np.asarray([x["e_delta_n_deg"] for x in sub], dtype=np.float64)
        bins_out.append({
            "bin": [lo, hi],
            "n": len(sub),
            **_axis_stats(e),
            "by_regime": {
                r: _axis_stats(np.asarray([x["e_delta_n_deg"] for x in sub if x["regime"] == r], dtype=np.float64))
                for r in REGIMES
            },
        })
    low = [b for b in bins_out if b["bin"][1] <= 0.1 and b["n"] > 0]
    high = [b for b in bins_out if b["bin"][0] >= 0.25 and b["n"] > 0]
    supported = False
    if low and high:
        supported = bool(
            low[0]["median_e_axis_deg"] > 15.0
            and high[0]["median_e_axis_deg"] < 10.0
        )
    return {
        "bins": bins_out,
        "overlap_support_hypothesis_supported": supported,
    }


def _descriptive_pattern(
    *,
    diag_a: dict[str, Any],
    diag_b: dict[str, Any],
    diag_c: dict[str, Any],
) -> tuple[str, list[str]]:
    tags: list[str] = []
    if diag_b.get("frames_geometrically_consistent_low_overlap"):
        tags.append("view_disjoint_surfaces_c0")
    if diag_a.get("segmentation_excluded_as_primary"):
        tags.append("segmentation_excluded_c0")
    elif diag_a.get("interpretation") == "segmentation_coupled_c0_failure":
        tags.append("segmentation_coupled_c0_failure")
    if diag_c.get("overlap_support_hypothesis_supported"):
        return "overlap_support_hypothesis_supported", tags
    return "overlap_support_hypothesis_weak", tags


def run_rtwx_o0rel0a(output: str | Path, config: RTWXO0REL0AConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWXO0REL0AConfig(output=str(output))
    root = _resolve_apr(cfg.output)
    rel0_root = _resolve_apr(cfg.rel0_run)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    rel0_summary_path = rel0_root / "summary.json"
    if not rel0_summary_path.is_file():
        raise FileNotFoundError(f"REL0 summary missing: {rel0_summary_path}")
    rel0_summary = json.loads(rel0_summary_path.read_text())
    if rel0_summary.get("pattern") != REL0_FORMAL_PATTERN:
        raise RuntimeError(
            f"REL0 formal pattern must remain {REL0_FORMAL_PATTERN!r}, "
            f"got {rel0_summary.get('pattern')!r}"
        )

    per_pair_path = rel0_root / "per_pair.json"
    per_pair_all = json.loads(per_pair_path.read_text())

    repo = Path(cfg.robotwin_repo)
    if not repo.is_absolute():
        repo = Path(cfg.robotwin_repo).resolve()
    d_o = load_D_O(str(repo) if repo.is_dir() else "/root/RoboTwin")
    cad_obj = _load_cad_pts3d(repo)

    rng = np.random.default_rng(cfg.seed)
    pooled: list[dict[str, Any]] = []
    for regime in REGIMES:
        for row in per_pair_all[regime]:
            pooled.append({**row, "regime": regime})

    obs_c0, gt_c0 = _load_cache(rel0_root, "c0")
    diag_a = diagnostic_a_gtmask_icp_c0(
        obs_c0, gt_c0, d_o=d_o, rng=rng, s0_per_pair=per_pair_all["c0"],
    )
    diag_b = diagnostic_b_cad_consistency_c0(
        obs_c0, gt_c0, cad_obj, d_o=d_o, rng=rng, s0_per_pair=per_pair_all["c0"],
    )
    diag_c = diagnostic_c_overlap_curve(pooled)

    pattern, tags = _descriptive_pattern(diag_a=diag_a, diag_b=diag_b, diag_c=diag_c)

    summary = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "rel0_run": str(rel0_root),
        "rel0_formal_pattern": REL0_FORMAL_PATTERN,
        "rel0_formal_pattern_unchanged": True,
        "descriptive_pattern": pattern,
        "tags": tags,
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "diagnostic_A_gtmask_icp_c0": diag_a,
        "diagnostic_B_cad_consistency_c0": diag_b,
        "diagnostic_C_overlap_curve": diag_c,
        "supports_overlap_gated_relative_tracking": bool(diag_c.get("overlap_support_hypothesis_supported")),
        "unlocks_o0e1": False,
        "unlocks_o1": False,
    }

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "rel0_run": str(rel0_root),
        "zero_train": True,
    }
    _write_json(root / "header.json", header)
    _write_json(root / "summary.json", summary)
    print(f"[rtwx-o0rel0a] descriptive_pattern={pattern}", flush=True)
    return summary

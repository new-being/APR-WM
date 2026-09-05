"""RTWX-O0G3A: correspondence availability audit (descriptive; no threshold rescue)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g3_cache import N_MIN, SEEDS, count_corr_frame, load_g3r_pools
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G3A_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g3a.correspondence_availability_audit.v1"
G3R_CACHE = "runs/rtwx_o0g3r"
N_BINS = 10


@dataclass(frozen=True)
class RTWXO0G3AConfig:
    output: str = "runs/rtwx_o0g3a"
    g3r_cache: str = G3R_CACHE
    seeds: tuple[int, ...] = SEEDS
    n_min: int = N_MIN
    smoke: bool = False


def _lock(cfg: RTWXO0G3AConfig) -> RTWXO0G3AConfig:
    if cfg.smoke:
        return cfg
    return replace(cfg, seeds=SEEDS, n_min=N_MIN)


def _bin_curve(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> list[dict[str, float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return []
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return [{"x_lo": float(x.min()), "x_hi": float(x.max()), "n": int(x.size), "P_enough": float(np.mean(y))}]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if not m.any():
            continue
        out.append({"x_lo": float(lo), "x_hi": float(hi), "n": int(m.sum()), "P_enough": float(np.mean(y[m]))})
    return out


def _audit_pool(pool: dict[str, Any], *, n_min: int) -> dict[str, Any]:
    n = pool["p"].shape[0]
    rows = []
    for fi in range(n):
        vis_any = bool(pool["vis_h"][fi] | pool["vis_o"][fi])
        if not vis_any:
            continue
        cnt = count_corr_frame(pool, fi, n_min=n_min)
        mh = np.asarray(pool["mask_h"][fi], bool)
        mo = np.asarray(pool["mask_o"][fi], bool)
        zh = pool["xyz_h"][fi, ..., 2]
        zo = pool["xyz_o"][fi, ..., 2]
        rows.append({
            "n_head": cnt["head"],
            "n_obs": cnt["observer"],
            "n_union": cnt["union"],
            "enough": bool(cnt["enough"]),
            "mask_area_h": float(mh.sum()),
            "mask_area_o": float(mo.sum()),
            "mask_area_union": float((mh | mo).sum()),
            "depth_frac_h": float(np.isfinite(zh).mean()),
            "depth_frac_o": float(np.isfinite(zo).mean()),
            "dual_valid": bool(mh.any() and mo.any()),
        })
    if not rows:
        raise RuntimeError("O0G3A: no visible frames")
    rec = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    vis = np.ones(rec["n_union"].shape[0], dtype=bool)
    p_any = float(np.mean(vis))
    p_enough = float(np.mean(rec["enough"]))
    return {
        "n_visible_frames": int(rec["n_union"].shape[0]),
        "P_visible_any": p_any,
        "P_enough": p_enough,
        "median_n_head": float(np.median(rec["n_head"])),
        "median_n_obs": float(np.median(rec["n_obs"])),
        "median_n_union": float(np.median(rec["n_union"])),
        "frac_dual_valid": float(np.mean(rec["dual_valid"])),
        "records": rec,
    }


def run_rtwx_o0g3a(output: str | Path, config: RTWXO0G3AConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G3AConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    header = {
        "stage": "RTWX-O0G3A",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "no_threshold_rescue": True,
        "depends_on": "O0G3R=coverage_failure",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    per_seed = []
    all_rec: list[dict[str, np.ndarray]] = []
    for seed in cfg.seeds:
        pool = load_g3r_pools(cfg.g3r_cache, (int(seed),), "test")
        aud = _audit_pool(pool, n_min=cfg.n_min)
        per_seed.append({"seed": int(seed), **{k: aud[k] for k in ("P_visible_any", "P_enough", "median_n_union", "n_visible_frames")}})
        all_rec.append(aud["records"])
        print(f"[rtwx-o0g3a] seed={seed} P_enough={aud['P_enough']:.3f} med_n_union={aud['median_n_union']:.0f}", flush=True)

    rec = {k: np.concatenate([r[k] for r in all_rec]) for k in all_rec[0]}
    agg_p_enough = float(np.mean(rec["enough"]))
    agg_p_any = 1.0  # by construction only visible frames

    curves = {
        "P_enough_given_mask_area_union": _bin_curve(rec["mask_area_union"], rec["enough"].astype(float), n_bins=N_BINS),
        "P_enough_given_depth_frac_h": _bin_curve(rec["depth_frac_h"], rec["enough"].astype(float), n_bins=N_BINS),
        "P_enough_given_n_head": _bin_curve(rec["n_head"].astype(float), rec["enough"].astype(float), n_bins=N_BINS),
    }

    # deficit frames: visible but not enough
    deficit = ~rec["enough"]
    deficit_stats = {
        "n_deficit": int(deficit.sum()),
        "frac_deficit": float(np.mean(deficit)),
        "median_n_union_deficit": float(np.median(rec["n_union"][deficit])) if deficit.any() else float("nan"),
        "median_mask_area_deficit": float(np.median(rec["mask_area_union"][deficit])) if deficit.any() else float("nan"),
    }

    pattern = "correspondence_availability_characterized"
    summary = {
        "header": header,
        "pattern": pattern,
        "N_min": cfg.n_min,
        "aggregate": {
            "P_visible_any": agg_p_any,
            "P_enough": agg_p_enough,
            "median_n_head": float(np.median(rec["n_head"])),
            "median_n_obs": float(np.median(rec["n_obs"])),
            "median_n_union": float(np.median(rec["n_union"])),
            "frac_dual_valid": float(np.mean(rec["dual_valid"])),
        },
        "per_seed": per_seed,
        "curves": curves,
        "deficit_stats": deficit_stats,
        "unlocks_o1": False,
        "unlocks_o0c2": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    print(f"[rtwx-o0g3a] pattern={pattern} P_enough={agg_p_enough:.3f}", flush=True)
    return summary

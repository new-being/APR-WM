"""RTWX-O0C1: orientation tail mechanism audit (diagnostic only). No O1 unlock. No O0C rewrite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import _e_R_deg, _quat_fix
from .rtwx_o0d3 import _coord_net, _e_R_after_yaw
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0c import (
    KEYS,
    ORI_OUT,
    SEED_PRIMARY,
    SEEDS,
    _R_to_quat,
    _batch_fuse_R,
    _concat_pools,
    _load,
    _predict_quat_CO,
    _quat_BO_from_CO,
    _score_ori,
    _train_ori,
)
from .rtwx_o0r import EPOCHS as ORI_EPOCHS
from .rtwx_o0v import CAM_HEAD, CAM_OBS
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0C1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0c1.orientation_tail_audit.v1"
RGB_SIZE = 64
GOOD_MAX = 30.0  # deg: "good" per-view threshold for D2
CATA_MIN = 30.0
BINS = (0.0, 15.0, 30.0, 60.0, 120.0, 180.0)
SYM_RESOLVE = 15.0
FRAC_RESOLVED_MIN = 0.70


@dataclass(frozen=True)
class RTWXO0C1Config:
    output: str = "runs/rtwx_o0c1"
    o0c_cache: str = "runs/rtwx_o0c"
    smoke: bool = False
    epochs_ori: int = ORI_EPOCHS
    rgb_size: int = RGB_SIZE
    seeds: tuple[int, ...] = SEEDS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0C1 must not write there")


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product; Sapien wxyz; applies q2 then q1 if column vectors."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _axis_angle_quat(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    th = np.radians(deg) / 2.0
    return _quat_fix(np.array([np.cos(th), *(np.sin(th) * a)]))[0]


def _G_sym_C4_Y() -> list[np.ndarray]:
    """Candidate discrete symmetries: C4 about object +Y (cup approximate upright)."""
    return [_axis_angle_quat(np.array([0.0, 1.0, 0.0]), 90.0 * k) for k in range(4)]


def _e_R_sym(qhat: np.ndarray, qgt: np.ndarray, G: list[np.ndarray]) -> np.ndarray:
    """min_g d(Rhat, Rgt @ g) with g in object frame."""
    n = qhat.shape[0]
    best = np.full(n, 180.0, dtype=np.float64)
    for g in G:
        # R_sym = R_gt @ g  →  q_sym = q_gt ⊗ g
        q_sym = np.stack([_quat_mul(qgt[i], g) for i in range(n)])
        best = np.minimum(best, _e_R_deg(qhat, q_sym))
    return best


def _hist_bins(err: np.ndarray, edges: tuple[float, ...] = BINS) -> dict[str, Any]:
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    counts = []
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (e >= lo) & (e <= hi)
            labels.append(f"[{lo:g},{hi:g}]")
        else:
            m = (e >= lo) & (e < hi)
            labels.append(f"[{lo:g},{hi:g})")
        counts.append(int(m.sum()))
    n = int(e.size)
    # concentration near 90 and 180
    near90 = float(np.mean((e >= 75.0) & (e <= 105.0))) if n else float("nan")
    near180 = float(np.mean(e >= 150.0)) if n else float("nan")
    return {
        "labels": labels,
        "counts": counts,
        "fracs": [c / max(n, 1) for c in counts],
        "n": n,
        "frac_near_90": near90,
        "frac_near_180": near180,
        "median": float(np.median(e)) if n else float("nan"),
        "p90": float(np.percentile(e, 90)) if n else float("nan"),
    }


def _per_view_quat_BO(pool: dict[str, Any], quat_CO_h: np.ndarray, quat_CO_o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = pool["p"].shape[0]
    qh = np.full((n, 4), np.nan, dtype=np.float64)
    qo = np.full((n, 4), np.nan, dtype=np.float64)
    for i in range(n):
        if pool["vis_h"][i]:
            qh[i] = _quat_BO_from_CO(quat_CO_h[i], pool["R_BC_h"][i])
        if pool["vis_o"][i]:
            qo[i] = _quat_BO_from_CO(quat_CO_o[i], pool["R_BC_o"][i])
    return qh, qo


def _d_view(qh: np.ndarray, qo: np.ndarray) -> np.ndarray:
    both = np.isfinite(qh).all(1) & np.isfinite(qo).all(1)
    out = np.full(qh.shape[0], np.nan, dtype=np.float64)
    if both.any():
        out[both] = _e_R_deg(qh[both], qo[both])
    return out


def _classify_cata(eh: np.ndarray, eo: np.ndarray, ef: np.ndarray, *, good: float = GOOD_MAX, cata: float = CATA_MIN) -> dict[str, Any]:
    m = np.isfinite(ef) & (ef > cata)
    n_c = int(m.sum())
    if n_c == 0:
        return {"n_catastrophe": 0, "fracs": {}, "counts": {}}
    eh_m, eo_m = eh[m], eo[m]
    # treat missing view as bad for classification
    h_good = np.isfinite(eh_m) & (eh_m <= good)
    o_good = np.isfinite(eo_m) & (eo_m <= good)
    h_bad = ~h_good
    o_bad = ~o_good
    counts = {
        "h_good_o_bad": int((h_good & o_bad).sum()),
        "h_bad_o_good": int((h_bad & o_good).sum()),
        "h_bad_o_bad": int((h_bad & o_bad).sum()),
        "h_good_o_good_fusion_bad": int((h_good & o_good).sum()),
    }
    fracs = {k: v / n_c for k, v in counts.items()}
    return {"n_catastrophe": n_c, "frac_catastrophe": float(m.mean()), "counts": counts, "fracs": fracs}


def _disagreement_curve(dview: np.ndarray, ef: np.ndarray) -> list[dict[str, float]]:
    both = np.isfinite(dview) & np.isfinite(ef)
    if not both.any():
        return []
    dv, e = dview[both], ef[both]
    edges = [0, 15, 30, 60, 90, 180]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (dv >= lo) & (dv < hi) if i < len(edges) - 2 else (dv >= lo) & (dv <= hi)
        if not sel.any():
            rows.append({"d_lo": lo, "d_hi": hi, "n": 0, "P_ef_gt_30": float("nan")})
            continue
        rows.append({"d_lo": lo, "d_hi": hi, "n": int(sel.sum()), "P_ef_gt_30": float(np.mean(e[sel] > 30.0))})
    return rows


def _d5_correlates(pool: dict[str, Any], ef: np.ndarray) -> dict[str, Any]:
    cata = np.isfinite(ef) & (ef > CATA_MIN)
    ok = np.isfinite(ef) & (ef <= CATA_MIN)
    area_h = pool["mask_h"].reshape(pool["mask_h"].shape[0], -1).sum(1).astype(np.float64)
    area_o = pool["mask_o"].reshape(pool["mask_o"].shape[0], -1).sum(1).astype(np.float64)
    single = (pool["vis_h"] & ~pool["vis_o"]) | (pool["vis_o"] & ~pool["vis_h"])
    both = pool["vis_h"] & pool["vis_o"]

    def mean_safe(x, m):
        return float(np.mean(x[m])) if m.any() else float("nan")

    return {
        "P_single_view_given_cata": mean_safe(single.astype(np.float64), cata),
        "P_single_view_given_ok": mean_safe(single.astype(np.float64), ok),
        "P_both_view_given_cata": mean_safe(both.astype(np.float64), cata),
        "mean_area_h_cata": mean_safe(area_h, cata),
        "mean_area_h_ok": mean_safe(area_h, ok),
        "mean_area_o_cata": mean_safe(area_o, cata),
        "mean_area_o_ok": mean_safe(area_o, ok),
        "P_vis_h_cata": mean_safe(pool["vis_h"].astype(np.float64), cata),
        "P_vis_o_cata": mean_safe(pool["vis_o"].astype(np.float64), cata),
    }


def _pick_pattern(d: dict[str, Any]) -> str:
    sym = d["D4_symmetry"]
    oracle = d["oracle_view_selector"]
    cata = d["D2_per_view"]["catastrophe_classes"]
    d5 = d["D5_conditions"]

    # A: symmetry resolves most catastrophes
    if (
        sym.get("frac_cata_resolved", 0.0) >= FRAC_RESOLVED_MIN
        and np.isfinite(sym.get("median_e_R_sym_on_cata", np.nan))
        and sym["median_e_R_sym_on_cata"] <= SYM_RESOLVE
    ):
        return "orientation_state_symmetry_mismatch"

    # C: fusion / selection
    if oracle.get("p90_e_best", 999.0) < 30.0 and oracle.get("p90_e_fusion", 0.0) >= 30.0:
        return "orientation_fusion_failure"
    fr = cata.get("fracs", {})
    if cata.get("n_catastrophe", 0) >= 10 and fr.get("h_good_o_good_fusion_bad", 0.0) >= 0.25:
        return "orientation_fusion_failure"

    # D: both wrong
    if oracle.get("p90_e_best", 0.0) >= 30.0 and fr.get("h_bad_o_bad", 0.0) >= 0.40:
        return "dual_view_both_wrong_mode"

    # B/E: view-dependent
    if (
        np.isfinite(d5.get("P_single_view_given_cata", np.nan))
        and d5["P_single_view_given_cata"] - d5.get("P_single_view_given_ok", 0.0) >= 0.15
    ):
        return "view_dependent_observability"

    return "orientation_tail_inconclusive"


def _numpy_tiny(cfg: RTWXO0C1Config) -> dict[str, dict[str, Any]]:
    """Synthetic dual-view for smoke when O0C caches absent."""
    from .rtwx_o0c import RTWXO0CConfig, _numpy_split
    from .rtwx_o0g1b import load_delta_O

    delta = np.asarray(load_delta_O("/root/RoboTwin")["delta_O"], dtype=np.float64)
    seeds = cfg.seeds if cfg.seeds else (41, 42)
    per = {}
    for s in seeds:
        rng = np.random.default_rng(int(s))
        c = RTWXO0CConfig(
            backend="numpy",
            smoke=True,
            n_train_ep=2,
            n_val_ep=1,
            n_test_ep=1,
            n_steps=16,
            seeds=(int(s),),
            rgb_size=cfg.rgb_size,
        )
        per[int(s)] = {
            "train": _numpy_split(c, "train", rng, delta, int(s)),
            "val": _numpy_split(c, "val", np.random.default_rng(int(s) + 1), delta, int(s)),
            "test": _numpy_split(c, "test", np.random.default_rng(int(s) + 2), delta, int(s)),
        }
    return per


def run_rtwx_o0c1(output: str | Path, config: RTWXO0C1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXO0C1Config()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    cache_root = Path(cfg.o0c_cache).resolve()

    header = {
        "stage": "RTWX-O0C1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "diagnostic_only": True,
        "unlocks_o1": False,
        "does_not_rewrite_o0c": True,
        "cameras": [CAM_HEAD, CAM_OBS],
        "seeds": list(cfg.seeds),
        "o0c_cache": str(cache_root),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    if cfg.smoke and not any((cache_root / f"cache_o0c_s{s}_test.npz").is_file() for s in cfg.seeds):
        per_seed_splits = _numpy_tiny(cfg)
    else:
        per_seed_splits = {}
        for seed in cfg.seeds:
            splits = {}
            for name in ("train", "val", "test"):
                hit = _load(cache_root / f"cache_o0c_s{seed}_{name}.npz", name)
                if hit is None:
                    raise RuntimeError(f"O0C1 missing cache: {cache_root}/cache_o0c_s{seed}_{name}.npz")
                splits[name] = hit
            per_seed_splits[int(seed)] = splits
            print(f"[rtwx-o0c1] seed={seed} cache ok", flush=True)

    train = _concat_pools([per_seed_splits[s]["train"] for s in cfg.seeds])
    val = _concat_pools([per_seed_splits[s]["val"] for s in cfg.seeds])
    test = _concat_pools([per_seed_splits[s]["test"] for s in cfg.seeds])

    rgb_tr = np.concatenate([train["rgb_h"], train["rgb_o"]], axis=0)
    q_tr = np.concatenate([train["quat_CO_h"], train["quat_CO_o"]], axis=0)
    rgb_va = np.concatenate([val["rgb_h"], val["rgb_o"]], axis=0)
    q_va = np.concatenate([val["quat_CO_h"], val["quat_CO_o"]], axis=0)
    epochs = 4 if cfg.smoke else cfg.epochs_ori
    print(f"[rtwx-o0c1] retrain Ori CoordConv epochs={epochs} (reproduce O0C family)", flush=True)
    ori = _coord_net(ORI_OUT, int(cfg.rgb_size))
    ori = _train_ori(ori, rgb_tr, q_tr, rgb_va, q_va, seed=SEED_PRIMARY + 7, epochs=epochs)

    pred_co_h = _predict_quat_CO(ori, test["rgb_h"])
    pred_co_o = _predict_quat_CO(ori, test["rgb_o"])
    q_f = _batch_fuse_R(test, pred_co_h, pred_co_o)
    q_h, q_o = _per_view_quat_BO(test, pred_co_h, pred_co_o)
    q_gt = test["quat"]

    # errors (nan-safe)
    def safe_er(qhat, q):
        out = np.full(q.shape[0], np.nan, dtype=np.float64)
        ok = np.isfinite(qhat).all(1)
        if ok.any():
            out[ok] = _e_R_deg(qhat[ok], q[ok])
        return out

    eh, eo, ef = safe_er(q_h, q_gt), safe_er(q_o, q_gt), safe_er(q_f, q_gt)
    e_best = np.nanmin(np.stack([eh, eo], axis=0), axis=0)

    d1 = {
        "fusion": _hist_bins(ef),
        "head": _hist_bins(eh[np.isfinite(eh)]),
        "observer": _hist_bins(eo[np.isfinite(eo)]),
        "best": _hist_bins(e_best[np.isfinite(e_best)]),
    }
    print(
        f"[rtwx-o0c1] D1 fusion med={d1['fusion']['median']:.2f} p90={d1['fusion']['p90']:.2f} "
        f"near90={d1['fusion']['frac_near_90']:.3f} near180={d1['fusion']['frac_near_180']:.3f}",
        flush=True,
    )

    d2_cls = _classify_cata(eh, eo, ef)
    d2 = {
        "score_head": _score_ori(q_h, q_gt),
        "score_observer": _score_ori(q_o, q_gt),
        "score_fusion": _score_ori(q_f, q_gt),
        "catastrophe_classes": d2_cls,
    }
    print(f"[rtwx-o0c1] D2 cata={d2_cls}", flush=True)

    dview = _d_view(q_h, q_o)
    d3 = {"curve": _disagreement_curve(dview, ef), "median_d_view": float(np.nanmedian(dview))}
    print(f"[rtwx-o0c1] D3 median_d_view={d3['median_d_view']:.2f}", flush=True)

    G = _G_sym_C4_Y()
    e_sym = _e_R_sym(q_f, q_gt, G)
    cata_m = np.isfinite(ef) & (ef > CATA_MIN)
    resolved = cata_m & (e_sym <= SYM_RESOLVE)
    e_yaw = _e_R_after_yaw(q_f, q_gt)
    d4 = {
        "G_sym": "C4_about_object_Y",
        "median_e_R_sym": float(np.median(e_sym)),
        "p90_e_R_sym": float(np.percentile(e_sym, 90)),
        "median_e_R_sym_on_cata": float(np.median(e_sym[cata_m])) if cata_m.any() else float("nan"),
        "p90_e_R_sym_on_cata": float(np.percentile(e_sym[cata_m], 90)) if cata_m.any() else float("nan"),
        "frac_cata_resolved": float(resolved.sum() / max(int(cata_m.sum()), 1)),
        "n_cata": int(cata_m.sum()),
        "continuous_yaw_audit": {
            "median": float(np.median(e_yaw)),
            "p90": float(np.percentile(e_yaw, 90)),
            "median_on_cata": float(np.median(e_yaw[cata_m])) if cata_m.any() else float("nan"),
            "p90_on_cata": float(np.percentile(e_yaw[cata_m], 90)) if cata_m.any() else float("nan"),
        },
    }
    print(
        f"[rtwx-o0c1] D4 sym_on_cata med={d4['median_e_R_sym_on_cata']:.2f} "
        f"resolved={d4['frac_cata_resolved']:.3f}",
        flush=True,
    )

    d5 = _d5_correlates(test, ef)
    print(f"[rtwx-o0c1] D5 {d5}", flush=True)

    oracle = {
        "median_e_best": float(np.nanmedian(e_best)),
        "p90_e_best": float(np.nanpercentile(e_best, 90)),
        "median_e_fusion": float(np.nanmedian(ef)),
        "p90_e_fusion": float(np.nanpercentile(ef, 90)),
        "implies_fusion_selection_issue": bool(
            np.isfinite(np.nanpercentile(e_best, 90))
            and np.nanpercentile(e_best, 90) < 30.0
            and np.nanpercentile(ef, 90) >= 30.0
        ),
    }
    print(f"[rtwx-o0c1] oracle e_best p90={oracle['p90_e_best']:.2f} fusion p90={oracle['p90_e_fusion']:.2f}", flush=True)

    # per-seed brief
    per_seed = []
    offset = 0
    for seed in cfg.seeds:
        n = per_seed_splits[int(seed)]["test"]["p"].shape[0]
        sl = slice(offset, offset + n)
        offset += n
        per_seed.append(
            {
                "seed": int(seed),
                "median_e_f": float(np.nanmedian(ef[sl])),
                "p90_e_f": float(np.nanpercentile(ef[sl], 90)),
                "p90_e_best": float(np.nanpercentile(e_best[sl], 90)),
                "frac_cata": float(np.mean(ef[sl] > CATA_MIN)),
            }
        )

    payload = {
        "D1_distribution": d1,
        "D2_per_view": d2,
        "D3_disagreement": d3,
        "D4_symmetry": d4,
        "D5_conditions": d5,
        "oracle_view_selector": oracle,
        "per_seed": per_seed,
    }
    pattern = _pick_pattern(payload)
    print(f"[rtwx-o0c1] pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "diagnostic_only": True,
        "unlocks_o1": False,
        "does_not_rewrite_o0c": True,
        **payload,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "D1_distribution", "D2_per_view", "D3_disagreement", "D4_symmetry", "D5_conditions", "oracle_view_selector", "per_seed")})
    return summary

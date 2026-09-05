"""RTWX-O0G6A: whole-object geometry orientation landscape. No ICP / no identity."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g3_cache import xo_grid
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5a import N_DESC, RGB_SIZE, RTWXO0G5AConfig, _collect_seed, _load_split
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0g5c import SEEDS_FRESH
from .rtwx_o0v import CAM_HEAD, CAM_OBS
from .rtwx_x0c import FORMAL_N_STEPS, _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G6A_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g6a.whole_object_orientation_landscape.v1"
SEEDS = SEEDS_FRESH
P_SUPPORT_AGG = 0.90
P_SUPPORT_SEED = 0.85
N_CAD = 2048
N_OBS = 256
K_HYP = 72
HYP_SEED = 34601
EPS_BASIN_DEG = 15.0
TOPK = 5
P_TOP1_AGG = 0.80
P_TOP1_SEED = 0.70
RMS_GT_MAX = 0.15
NEAR90 = (75.0, 105.0)
AXIS_MODES = tuple(f"R{ax}{deg}" for ax in "xyz" for deg in (90, 180, 270))


@dataclass(frozen=True)
class RTWXO0G6AConfig:
    output: str = "runs/rtwx_o0g6a"
    g5c_cache: str = "runs/rtwx_o0g5c"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_test_ep: int = 12
    n_steps: int = FORMAL_N_STEPS
    rgb_size: int = RGB_SIZE
    smoke: bool = False
    seed_attempts: int = 32
    max_resample: int = 16


def _lock(cfg: RTWXO0G6AConfig) -> RTWXO0G6AConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        seeds=SEEDS,
        n_test_ep=12,
        n_steps=FORMAL_N_STEPS,
        rgb_size=RGB_SIZE,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G6A must not write there")


def _axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    th = np.radians(float(deg))
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _build_generators(seed: int = HYP_SEED, k: int = K_HYP) -> tuple[np.ndarray, list[str]]:
    rs: list[np.ndarray] = [np.eye(3)]
    names: list[str] = ["I"]
    for ax_name, axis in (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))):
        for deg in (90, 180, 270):
            rs.append(_axis_angle(np.array(axis), deg))
            names.append(f"R{ax_name}{deg}")
    rng = np.random.default_rng(seed)
    for i in range(8):
        rs.append(_axis_angle(rng.normal(size=3), 10.0))
        names.append(f"loc10_{i}")
    tries = 0
    while len(rs) < k and tries < 4000:
        tries += 1
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        R = _quat_to_R(q)
        if any(_geodesic_deg(R, e) < 5.0 for e in rs):
            continue
        rs.append(R)
        names.append(f"hopf_{len(rs)}")
    if len(rs) < k:
        raise RuntimeError(f"O0G6A: only {len(rs)} generators")
    return np.stack(rs[:k], 0), names[:k]


def _n_usable_frame(pool: dict[str, Any], fi: int) -> int:
    p, q = pool["p"][fi], pool["quat"][fi]
    n = 0
    for view in ("h", "o"):
        xyz = pool[f"xyz_{view}"][fi]
        mask = np.asarray(pool[f"mask_{view}"][fi], bool)
        xog = xo_grid(xyz, mask, p, q)
        n += int((mask & np.isfinite(xyz).all(-1) & np.isfinite(xog).all(-1)).sum())
    return n


def _obs_cloud(pool: dict[str, Any], fi: int, rng: np.random.Generator, n_max: int) -> np.ndarray:
    chunks = []
    for view in ("h", "o"):
        xyz = pool[f"xyz_{view}"][fi]
        mask = np.asarray(pool[f"mask_{view}"][fi], bool)
        m = mask & np.isfinite(xyz).all(-1)
        if m.any():
            chunks.append(np.asarray(xyz, dtype=np.float64)[m])
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    P = np.concatenate(chunks, 0)
    if P.shape[0] > n_max:
        P = P[rng.choice(P.shape[0], n_max, replace=False)]
    return P


def _support(pools: list[dict[str, Any]], seeds: tuple[int, ...]) -> tuple[float, list[dict[str, Any]]]:
    per, flags = [], []
    for pool, seed in zip(pools, seeds):
        ok = []
        for fi in range(pool["p"].shape[0]):
            if not (pool["vis_h"][fi] or pool["vis_o"][fi]):
                continue
            ok.append(_n_usable_frame(pool, fi) >= N_DESC)
        p = float(np.mean(ok)) if ok else 0.0
        per.append({"seed": int(seed), "P_support": p, "ok": p >= P_SUPPORT_SEED, "n_vis": int(len(ok))})
        flags.extend(ok)
    return (float(np.mean(flags)) if flags else 0.0), per


def _pattern(*, g0: bool, g1: bool, g2: bool) -> str:
    if not g0:
        return "surface_support_failure"
    if g1 and g2:
        return "global_shape_orientation_supported"
    return "global_geometry_ambiguous"


def _score_frame(y_true: np.ndarray, tree: Any, gens: np.ndarray, basin: np.ndarray) -> dict[str, Any]:
    xq = np.einsum("ni,kij->knj", y_true, gens).reshape(-1, 3)
    dist = np.asarray(tree.query(xq)[0], dtype=np.float64).reshape(gens.shape[0], y_true.shape[0])
    S = np.mean(dist * dist, axis=1)
    k_star = int(np.argmin(S))
    s_b = float(np.min(S[basin]))
    s_w = float(np.min(S[~basin]))
    order = np.argsort(S)
    return {
        "S": S,
        "k_star": k_star,
        "d_star_deg": _geodesic_deg(gens[k_star], np.eye(3)),
        "top1": bool(basin[k_star]),
        "topk": bool(np.any(basin[order[:TOPK]])),
        "margin": s_w - s_b,
        "S_I": float(S[0]),
    }


def run_rtwx_o0g6a(output: str | Path, config: RTWXO0G6AConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G6AConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    D_O = load_D_O(repo)
    gens, names = _build_generators(HYP_SEED, K_HYP)
    basin = np.array([_geodesic_deg(R, np.eye(3)) <= EPS_BASIN_DEG for R in gens], dtype=bool)
    mode_idx = {n: names.index(n) for n in AXIS_MODES if n in names}
    print(f"[rtwx-o0g6a] K={gens.shape[0]} basin={int(basin.sum())} modes={list(mode_idx)}", flush=True)

    header = {
        "stage": "RTWX-O0G6A",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_icp": True,
        "no_ransac": True,
        "no_learned_identity": True,
        "p_is_gt_nuisance": True,
        "metric": "one_way_obs_to_cad_mse",
        "cameras": [CAM_HEAD, CAM_OBS],
        "seeds": list(cfg.seeds),
        "K_hyp": int(gens.shape[0]),
        "N_cad": N_CAD,
        "eps_basin_deg": EPS_BASIN_DEG,
        "D_O_m": D_O,
        "generators": names,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    if cfg.smoke:
        rng = np.random.default_rng(0)
        # elongated box: yaw about y is distinguishable
        cad = np.stack(np.meshgrid(
            np.linspace(-0.04, 0.04, 12),
            np.linspace(-0.02, 0.02, 6),
            np.linspace(-0.01, 0.01, 4),
            indexing="ij",
        ), -1).reshape(-1, 3).astype(np.float64)
        n_f = 48
        pools = []
        P_list, p_list, q_list = [], [], []
        for i in range(n_f):
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            R = _quat_to_R(q)
            p = rng.normal(size=3) * 0.05
            vis = cad[cad[:, 0] > 0]
            if vis.shape[0] < 80:
                vis = cad
            Pb = vis @ R.T + p
            P_list.append(Pb)
            p_list.append(p)
            q_list.append(q)
        pools = None
        p_sup, per_seed = 1.0, [{"seed": 41, "P_support": 1.0, "ok": True, "n_vis": n_f}]
        g0_pts = True
        frames = list(zip(P_list, p_list, q_list, [41] * n_f))
    else:
        cad = _fps(load_scaled_cad(repo, 8192), N_CAD, np.random.default_rng(HYP_SEED))
        stop: list[str] = []
        pools = []
        for seed in cfg.seeds:
            path = Path(cfg.g5c_cache) / f"cache_o0g5c_s{seed}_test.npz"
            hit = _load_split(path, "test")
            if hit is None:
                g5a = RTWXO0G5AConfig(
                    output=str(root),
                    robotwin_repo=cfg.robotwin_repo,
                    backend=cfg.backend,
                    seeds=(int(seed),),
                    n_test_ep=cfg.n_test_ep,
                    n_steps=cfg.n_steps,
                    rgb_size=cfg.rgb_size,
                    seed_attempts=cfg.seed_attempts,
                    max_resample=cfg.max_resample,
                    smoke=False,
                )
                hit = _collect_seed(g5a, root, int(seed), stop, cache_tag="o0g6a")
            pools.append(hit)
            print(f"[rtwx-o0g6a] seed={seed} n={hit['p'].shape[0]}", flush=True)
        if stop:
            raise RuntimeError(f"O0G6A collect stop: {stop}")
        p_sup, per_seed = _support(pools, cfg.seeds)
        g0_pts = bool(p_sup >= P_SUPPORT_AGG and all(r["ok"] for r in per_seed))
        print(f"[rtwx-o0g6a] G0 P_support={p_sup:.3f} ok={g0_pts}", flush=True)
        frames = []
        rng = np.random.default_rng(HYP_SEED + 9)
        for pool, seed in zip(pools, cfg.seeds):
            for fi in range(pool["p"].shape[0]):
                if not (pool["vis_h"][fi] or pool["vis_o"][fi]):
                    continue
                if _n_usable_frame(pool, fi) < N_DESC:
                    continue
                P = _obs_cloud(pool, fi, rng, N_OBS)
                if P.shape[0] < 16:
                    continue
                frames.append((P, pool["p"][fi], pool["quat"][fi], int(seed)))
        gc.collect()

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(cad, dtype=np.float64))
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for P, p, quat, seed in frames:
        R_gt = _quat_to_R(quat)
        y_true = (np.asarray(P, dtype=np.float64) - np.asarray(p, dtype=np.float64).reshape(1, 3)) @ R_gt
        sc = _score_frame(y_true, tree, gens, basin)
        row = {
            "seed": int(seed),
            "top1": sc["top1"],
            "topk": sc["topk"],
            "d_star_deg": sc["d_star_deg"],
            "margin": sc["margin"],
            "S_I": sc["S_I"],
            "rms_I_norm": float(np.sqrt(sc["S_I"]) / D_O),
            **{n: float(sc["S"][i]) for n, i in mode_idx.items()},
        }
        by_seed.setdefault(int(seed), []).append(row)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n": 0, "P_top1": float("nan"), "P_topk": float("nan"), "median_margin": float("nan"), "median_rms_I_norm": float("nan")}
        top1 = np.array([r["top1"] for r in rows], dtype=bool)
        dstar = np.array([r["d_star_deg"] for r in rows], dtype=np.float64)
        out: dict[str, Any] = {
            "n": int(len(rows)),
            "P_top1": float(np.mean(top1)),
            "P_topk": float(np.mean([r["topk"] for r in rows])),
            "median_d_star_deg": float(np.median(dstar)),
            "p90_d_star_deg": float(np.percentile(dstar, 90)),
            "P_near90": float(np.mean((dstar >= NEAR90[0]) & (dstar <= NEAR90[1]))),
            "median_margin": float(np.median([r["margin"] for r in rows])),
            "median_rms_I_norm": float(np.median([r["rms_I_norm"] for r in rows])),
            "median_S_I": float(np.median([r["S_I"] for r in rows])),
        }
        for n in AXIS_MODES:
            if n not in rows[0]:
                continue
            s_m = np.array([r[n] for r in rows], dtype=np.float64)
            s_i = np.array([r["S_I"] for r in rows], dtype=np.float64)
            out[n] = {
                "median_dS": float(np.median(s_m - s_i)),
                "median_ratio": float(np.median(s_m / np.maximum(s_i, 1e-18))),
                "P_gt_cheaper": float(np.mean(s_i < s_m)),
            }
        return out

    all_rows = [r for rs in by_seed.values() for r in rs]
    agg = _agg(all_rows)
    per = []
    for seed in (list(cfg.seeds) if not cfg.smoke else sorted(by_seed)):
        a = _agg(by_seed.get(int(seed), []))
        a["seed"] = int(seed)
        a["ok_top1_seed"] = bool(np.isfinite(a["P_top1"]) and a["P_top1"] >= P_TOP1_SEED)
        per.append(a)

    g0_fit = bool(np.isfinite(agg.get("median_rms_I_norm", np.nan)) and agg["median_rms_I_norm"] <= RMS_GT_MAX)
    g0 = bool((True if cfg.smoke else g0_pts) and g0_fit)
    g1 = bool(
        np.isfinite(agg.get("P_top1", np.nan))
        and agg["P_top1"] >= P_TOP1_AGG
        and all(r.get("ok_top1_seed", False) for r in per)
    )
    g2 = bool(np.isfinite(agg.get("median_margin", np.nan)) and agg["median_margin"] > 0.0)
    pattern = _pattern(g0=g0, g1=g1, g2=g2)
    unlock_r = pattern == "global_shape_orientation_supported"
    print(
        f"[rtwx-o0g6a] G0={g0} rmsI={agg.get('median_rms_I_norm', float('nan')):.4f} "
        f"P_top1={agg.get('P_top1', float('nan')):.3f} top5={agg.get('P_topk', float('nan')):.3f} "
        f"m={agg.get('median_margin', float('nan')):.6g} pattern={pattern}",
        flush=True,
    )
    for n in ("Ry90", "Rx90", "Rz90"):
        if n in agg:
            print(f"[rtwx-o0g6a] {n} ratio={agg[n]['median_ratio']:.3f} P_gt_cheaper={agg[n]['P_gt_cheaper']:.3f}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_support": {
            "ok": g0,
            "P_support": p_sup,
            "per_seed": per_seed,
            "N_desc": N_DESC,
            "cad_gt_fit_ok": g0_fit,
            "median_rms_I_norm": agg.get("median_rms_I_norm"),
            "gate_rms": RMS_GT_MAX,
        },
        "G1_top1": {"ok": g1, "P_top1": agg.get("P_top1"), "gate_agg": P_TOP1_AGG, "gate_seed": P_TOP1_SEED, "per_seed": per},
        "G2_margin": {"ok": g2, "median_margin": agg.get("median_margin"), "gate": 0.0},
        "landscape": agg,
        "unlocks_o0g6r_prereg": unlock_r,
        "unlocks_o0g5r": False,
        "unlocks_o0c2": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

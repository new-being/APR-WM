"""RTWX-O0G5C: fresh CAD-anchor identity generalization. No RANSAC/Kabsch primary."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g3_cache import xo_grid
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5a import N_DESC, RGB_SIZE, RTWXO0G5AConfig, _collect_seed, _load_split
from .rtwx_o0g5a import SEEDS as TRAIN_SEEDS
from .rtwx_o0g5b import (
    E_MED_MAX,
    EPOCH_DESC,
    FPS_SEED,
    K_ANCHOR,
    N_GLOBAL,
    N_SAMPLES,
    PTS_PER_FRAME,
    TOPK_RECALL_MIN,
    _fps,
    _infer_desc,
    _labels,
    _pack_instrument,
    _score,
    _train_desc,
    load_scaled_cad,
)
from .rtwx_o0v import CAM_HEAD, CAM_OBS
from .rtwx_x0c import FORMAL_N_STEPS, _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G5C_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g5c.fresh_canonical_identity.v1"
SEEDS_FRESH = (33601, 33602, 33603)
P_SUPPORT_AGG = 0.90
P_SUPPORT_SEED = 0.85
AREA_SMALL = 256.0
AREA_LARGE = 1024.0
UV_BINS = 8
N_TEST_SAMPLES = 16384
NAN_SCORE = {
    "median_e_norm": float("nan"),
    "p90_e_norm": float("nan"),
    "top1_label_acc": float("nan"),
    "top1_recall_r005": float("nan"),
    "top5_recall": float("nan"),
    "recall_r010": float("nan"),
    "n": 0,
}


@dataclass(frozen=True)
class RTWXO0G5CConfig:
    output: str = "runs/rtwx_o0g5c"
    g5a_cache: str = "runs/rtwx_o0g5a"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    fresh_seeds: tuple[int, ...] = SEEDS_FRESH
    n_test_ep: int = 12
    n_steps: int = FORMAL_N_STEPS
    rgb_size: int = RGB_SIZE
    epoch_desc: int = EPOCH_DESC
    smoke: bool = False
    seed_attempts: int = 32
    max_resample: int = 16


def _lock(cfg: RTWXO0G5CConfig) -> RTWXO0G5CConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        train_seeds=TRAIN_SEEDS,
        fresh_seeds=SEEDS_FRESH,
        n_test_ep=12,
        n_steps=FORMAL_N_STEPS,
        rgb_size=RGB_SIZE,
        epoch_desc=EPOCH_DESC,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G5C must not write there")


def _cuda_gc() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gate(sc: dict[str, float]) -> bool:
    return bool(np.isfinite(sc["median_e_norm"]) and sc["median_e_norm"] <= E_MED_MAX and sc["top5_recall"] >= TOPK_RECALL_MIN)


def _cond_entropy(u: np.ndarray, v: np.ndarray, y: np.ndarray, bins: int = UV_BINS, min_n: int = 8) -> dict[str, float]:
    ui = np.clip((np.asarray(u) * bins).astype(int), 0, bins - 1)
    vi = np.clip((np.asarray(v) * bins).astype(int), 0, bins - 1)
    cell = ui * bins + vi
    hs = []
    for c in np.unique(cell):
        m = cell == c
        if int(m.sum()) < min_n:
            continue
        _, cnt = np.unique(y[m], return_counts=True)
        p = cnt / cnt.sum()
        hs.append(float(-(p * np.log(np.maximum(p, 1e-12))).sum()))
    return {"H_Y_given_UV": float(np.mean(hs)) if hs else float("nan"), "n_cells": int(len(hs))}


def _n_usable_frame(pool: dict[str, Any], fi: int) -> int:
    p, q = pool["p"][fi], pool["quat"][fi]
    n = 0
    for view in ("h", "o"):
        xyz = pool[f"xyz_{view}"][fi]
        mask = np.asarray(pool[f"mask_{view}"][fi], bool)
        xog = xo_grid(xyz, mask, p, q)
        n += int((mask & np.isfinite(xyz).all(-1) & np.isfinite(xog).all(-1)).sum())
    return n


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


def _pattern(*, g0: bool, b1: bool, b2: bool) -> str:
    if not g0:
        return "coverage_failure"
    if b1 and b2:
        return "both_supported"
    if b1 and not b2:
        return "local_identity_supported"
    if (not b1) and b2:
        return "global_context_supported"
    return "canonical_identity_generalization_failure"


def _eval_branch(model, samples, anchors, D_O, *, local_only: bool, mu, sd, local_override=None) -> dict[str, float]:
    loc = samples["local"] if local_override is None else local_override
    z, cad = _infer_desc(model, loc, samples["gidx"], samples["global"], local_only=local_only, loc_mu=mu, loc_sd=sd)
    logits = z @ cad.T
    pred = np.argmax(logits, axis=1)
    return _score(anchors[pred], samples["xo"], anchors, logits, D_O)


def _subset(samples: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    sub = {
        "local": samples["local"][mask],
        "xo": samples["xo"][mask],
        "gidx": samples["gidx"][mask],
        "global": samples["global"],
    }
    used = np.unique(sub["gidx"])
    remap = {int(o): n for n, o in enumerate(used)}
    sub["global"] = samples["global"][used]
    sub["gidx"] = np.array([remap[int(v)] for v in sub["gidx"]], dtype=np.int32)
    return sub


def _strata(area: np.ndarray, samples, anchors, D_O, model, mu, sd, *, local_only: bool) -> dict[str, Any]:
    bands = {
        "small": area < AREA_SMALL,
        "medium": (area >= AREA_SMALL) & (area <= AREA_LARGE),
        "large": area > AREA_LARGE,
    }
    out: dict[str, Any] = {}
    for name, m in bands.items():
        if not m.any():
            out[name] = {"n": 0, "ok": False, **NAN_SCORE}
            continue
        sc = _eval_branch(model, _subset(samples, m), anchors, D_O, local_only=local_only, mu=mu, sd=sd)
        out[name] = {"n": int(m.sum()), "ok": _gate(sc), **sc}
    return out


def run_rtwx_o0g5c(output: str | Path, config: RTWXO0G5CConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G5CConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    D_O = load_D_O(repo)
    rng = np.random.default_rng(FPS_SEED)

    header = {
        "stage": "RTWX-O0G5C",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_ransac": True,
        "no_kabsch_primary": True,
        "no_pose_rescue": True,
        "cameras": [CAM_HEAD, CAM_OBS],
        "train_seeds": list(cfg.train_seeds),
        "fresh_seeds": list(cfg.fresh_seeds),
        "K_anchor": K_ANCHOR,
        "rgb_size": cfg.rgb_size,
        "D_O_m": D_O,
        "gate": {"median_e_norm": E_MED_MAX, "top5_recall": TOPK_RECALL_MIN},
        "config": {**asdict(cfg), "train_seeds": list(cfg.train_seeds), "fresh_seeds": list(cfg.fresh_seeds)},
    }
    _write_json(root / "header.json", header)

    if cfg.smoke:
        from .rtwx_o0g5a import _numpy_cad_pool

        cad = _numpy_cad_pool(512, np.random.default_rng(0))
        n, n_g = 1024, 128

        def _syn(seed: int) -> dict[str, np.ndarray]:
            r = np.random.default_rng(seed)
            loc = r.normal(size=(n, 6)).astype(np.float32)
            loc[:, 0] = r.random(n)
            loc[:, 1] = r.random(n)
            return {
                "local": loc,
                "xo": cad[r.integers(0, cad.shape[0], n)] + r.normal(size=(n, 3)) * 0.002,
                "global": r.normal(size=(n_g, 32, 3)).astype(np.float32),
                "gidx": np.clip(np.arange(n) // 8, 0, n_g - 1).astype(np.int32),
                "area": r.uniform(50, 2000, n).astype(np.float32),
            }

        train_s, test_s = _syn(1), _syn(99)
        p_sup, per_seed = 1.0, [{"seed": 41, "P_support": 1.0, "ok": True, "n_vis": n}]
        g0 = True
    else:
        cad = load_scaled_cad(repo, 8192)
        train_pools = []
        for seed in cfg.train_seeds:
            hit = _load_split(Path(cfg.g5a_cache) / f"cache_o0g5a_s{seed}_test.npz", "test")
            if hit is None:
                raise FileNotFoundError(f"missing train cache seed={seed}")
            train_pools.append(hit)
            print(f"[rtwx-o0g5c] train seed={seed} cache", flush=True)
        train_s = _pack_instrument(train_pools, D_O=D_O, rng=rng, max_samples=N_SAMPLES, k_global=N_GLOBAL, per_frame=PTS_PER_FRAME)

        stop: list[str] = []
        test_pools = []
        for seed in cfg.fresh_seeds:
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
            pool = _collect_seed(g5a, root, int(seed), stop, cache_tag="o0g5c")
            test_pools.append(pool)
            _cuda_gc()
        if stop:
            raise RuntimeError(f"O0G5C collect stop: {stop}")
        p_sup, per_seed = _support(test_pools, cfg.fresh_seeds)
        g0 = bool(p_sup >= P_SUPPORT_AGG and all(r["ok"] for r in per_seed))
        print(f"[rtwx-o0g5c] G0 P_support={p_sup:.3f} ok={g0}", flush=True)
        test_s = _pack_instrument(
            test_pools,
            D_O=D_O,
            rng=np.random.default_rng(FPS_SEED + 7),
            max_samples=N_TEST_SAMPLES,
            k_global=N_GLOBAL,
            per_frame=PTS_PER_FRAME,
        )
        _cuda_gc()

    anchors = _fps(cad, K_ANCHOR, np.random.default_rng(FPS_SEED))
    y_tr = _labels(train_s["xo"], anchors)
    y_te = _labels(test_s["xo"], anchors)
    q_tr = np.linalg.norm(anchors[y_tr] - train_s["xo"], axis=1) / D_O
    q_te = np.linalg.norm(anchors[y_te] - test_s["xo"], axis=1) / D_O
    ep = 60 if cfg.smoke else cfg.epoch_desc
    print(
        f"[rtwx-o0g5c] n_train={train_s['xo'].shape[0]} n_fresh={test_s['xo'].shape[0]} "
        f"quant_train={float(np.median(q_tr)):.4f} quant_fresh={float(np.median(q_te)):.4f}",
        flush=True,
    )

    print("[rtwx-o0g5c] train B1", flush=True)
    _, _, m1, mu1, sd1 = _train_desc(
        train_s["local"], train_s["gidx"], train_s["global"], y_tr, anchors.shape[0],
        local_only=True, epochs=ep, seed=FPS_SEED + 1,
    )
    print("[rtwx-o0g5c] train B2", flush=True)
    _, _, m2, mu2, sd2 = _train_desc(
        train_s["local"], train_s["gidx"], train_s["global"], y_tr, anchors.shape[0],
        local_only=False, epochs=ep, seed=FPS_SEED + 2,
    )

    b1 = _eval_branch(m1, test_s, anchors, D_O, local_only=True, mu=mu1, sd=sd1)
    b2 = _eval_branch(m2, test_s, anchors, D_O, local_only=False, mu=mu2, sd=sd2)
    b1_ok, b2_ok = _gate(b1), _gate(b2)
    print(
        f"[rtwx-o0g5c] B1 med={b1['median_e_norm']:.4f} top1={b1['top1_label_acc']:.3f} "
        f"top5={b1['top5_recall']:.3f} p90={b1['p90_e_norm']:.3f} ok={b1_ok}",
        flush=True,
    )
    print(
        f"[rtwx-o0g5c] B2 med={b2['median_e_norm']:.4f} top1={b2['top1_label_acc']:.3f} "
        f"top5={b2['top5_recall']:.3f} p90={b2['p90_e_norm']:.3f} ok={b2_ok}",
        flush=True,
    )

    loc_sh = test_s["local"].copy()
    loc_sh[:, 0:2] = loc_sh[rng.permutation(loc_sh.shape[0]), 0:2]
    b1_shuf = _eval_branch(m1, test_s, anchors, D_O, local_only=True, mu=mu1, sd=sd1, local_override=loc_sh)
    print(
        f"[rtwx-o0g5c] B1-uv-shuffle med={b1_shuf['median_e_norm']:.4f} top5={b1_shuf['top5_recall']:.3f}",
        flush=True,
    )

    h_tr = _cond_entropy(train_s["local"][:, 0], train_s["local"][:, 1], y_tr)
    h_te = _cond_entropy(test_s["local"][:, 0], test_s["local"][:, 1], y_te)
    print(
        f"[rtwx-o0g5c] H(Y|UV) train={h_tr['H_Y_given_UV']:.3f} fresh={h_te['H_Y_given_UV']:.3f}",
        flush=True,
    )

    strata = {
        "B1": _strata(test_s["area"], test_s, anchors, D_O, m1, mu1, sd1, local_only=True),
        "B2": _strata(test_s["area"], test_s, anchors, D_O, m2, mu2, sd2, local_only=False),
    }
    for br, rows in strata.items():
        for band, row in rows.items():
            print(
                f"[rtwx-o0g5c] {br}/{band} n={row['n']} med={row['median_e_norm']:.4f} top5={row['top5_recall']:.3f}",
                flush=True,
            )

    pattern = _pattern(g0=g0, b1=b1_ok, b2=b2_ok)
    unlock_r = pattern in {"local_identity_supported", "both_supported", "global_context_supported"}
    print(f"[rtwx-o0g5c] pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_support": {"ok": g0, "P_support": p_sup, "per_seed": per_seed, "N_desc": N_DESC},
        "B1_local_fresh": {"ok": b1_ok, **b1},
        "B2_global_fresh": {"ok": b2_ok, **b2},
        "B1_uv_shuffle": {
            **b1_shuf,
            "delta_top5": float(b1_shuf["top5_recall"] - b1["top5_recall"]),
            "delta_med": float(b1_shuf["median_e_norm"] - b1["median_e_norm"]),
        },
        "H_Y_given_UV": {"train": h_tr, "fresh": h_te},
        "mask_area_strata": strata,
        "quantization": {
            "train_median": float(np.median(q_tr)),
            "fresh_median": float(np.median(q_te)),
            "fresh_p90": float(np.percentile(q_te, 90)),
        },
        "n_train": int(train_s["xo"].shape[0]),
        "n_fresh": int(test_s["xo"].shape[0]),
        "unlocks_o0g5r_prereg": unlock_r,
        "unlocks_o0c2": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

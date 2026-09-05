"""RTWX-X0EH: receding-horizon sufficiency audit on frozen X0E-M2. Diagnostic only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .rtwx_x0 import _nrmse
from .rtwx_x0c import FORMAL_N_STEPS, _write_json
from .rtwx_x0e import _phi_m2, _x
from .rtwx_x0e1 import _delta, _load_cache, _lstsq
from .rtwx_x0es import CATA_EH, H_MAX, _rho, _step
from .rtwx_x0es1 import N_FRESH, RTWX0ES1Config, collect_fresh

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0EH_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0eh.receding_horizon.v1"
SEED_FRESH = 11601
OLD_CACHE = "runs/rtwx_x0e1/cache_splits.npz"
K_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 50)
K_PRIMARY = 4
H_PREDICT = 16
N_CAT_MIN = 5
G2_RATIO = 0.75
SPEARMAN_MIN = 0.7
SURVIVAL_HS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 50)


@dataclass(frozen=True)
class RTWX0EHConfig:
    output: str = "runs/rtwx_x0eh"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    old_cache: str = OLD_CACHE
    seed: int = SEED_FRESH
    smoke: bool = False
    n_ep: int = N_FRESH
    n_steps: int = FORMAL_N_STEPS
    n_train_ep: int = N_FRESH
    n_val_ep: int = 0
    n_test_ep: int = 0
    seed_attempts: int = 32
    max_resample: int = 16
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0EH must not write there")


def _lock(cfg: RTWX0EHConfig) -> RTWX0EHConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_ep=N_FRESH,
        n_train_ep=N_FRESH,
        n_steps=FORMAL_N_STEPS,
        seed=SEED_FRESH,
    )


def _e_h(hat: np.ndarray, ref: np.ndarray, scale: float) -> float:
    if not np.isfinite(hat).all():
        return float("inf")
    return float(np.linalg.norm(hat - ref) / (scale + 1.0e-8))


def _first_cross(errs: list[float], thr: float) -> float:
    for i, e in enumerate(errs, start=1):
        if not np.isfinite(e) or e > thr:
            return float(i)
    return float("inf")


def _open_loop_audit(pool: dict[str, Any], w: np.ndarray, scale: float) -> dict[str, Any]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    n_start = n_steps - H_MAX
    t_divs: list[float] = []
    t1s: list[float] = []
    t10s: list[float] = []
    t100s: list[float] = []
    n_cata = 0
    n_nf = 0
    n_win = 0
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        for t0 in range(max(0, n_start)):
            n_win += 1
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            errs: list[float] = []
            finite = True
            for h in range(H_MAX):
                qh, qdh = _step(qh, qdh, qt[t0 + h], w)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    finite = False
                    errs.append(float("inf"))
                    break
                errs.append(_e_h(_x(qh, qdh), _x(qnn[t0 + h], qdnn[t0 + h]), scale))
            while len(errs) < H_MAX:
                errs.append(float("inf"))
            td = _first_cross(errs, CATA_EH)
            t_divs.append(td)
            t1s.append(_first_cross(errs, 1.0))
            t10s.append(_first_cross(errs, 10.0))
            t100s.append(_first_cross(errs, 100.0))
            if (not finite) or (np.isfinite(td) and td <= H_MAX) or max(errs) > CATA_EH:
                n_cata += 1
            if not finite:
                n_nf += 1
    td = np.asarray(t_divs, dtype=np.float64)
    surv = {f"P_Tdiv_le_{h}": float(np.mean(td <= h)) for h in SURVIVAL_HS}
    # K99: largest K in 1..50 with P(T_div > K) >= 0.99
    k99 = 0
    for k in range(1, H_MAX + 1):
        if float(np.mean(td > k)) >= 0.99:
            k99 = k
    k999 = 0
    for k in range(1, H_MAX + 1):
        if float(np.mean(td > k)) >= 0.999:
            k999 = k
    return {
        "n_windows": n_win,
        "n_cata": n_cata,
        "n_nonfinite": n_nf,
        "r_cat": float(n_cata / max(n_win, 1)),
        "survival": surv,
        "K99": int(k99),
        "K999_exploratory": int(k999),
        "Tdiv_median_finite": float(np.median(td[np.isfinite(td)])) if np.any(np.isfinite(td)) else float("inf"),
        "T1_median": float(np.median(t1s)),
        "T10_median": float(np.median(t10s)),
        "T100_median": float(np.median(t100s)),
        "P_Tdiv_finite": float(np.mean(np.isfinite(td))),
    }


def _segment_roll(
    qq: np.ndarray,
    qddt: np.ndarray,
    qnn: np.ndarray,
    qdnn: np.ndarray,
    qt: np.ndarray,
    t0: int,
    k: int,
    w: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    """Roll K steps from true state at t0; score executed interval only."""
    qh, qdh = qq[t0].copy(), qddt[t0].copy()
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    rho_hit = False
    mx = 0.0
    finite = True
    for h in range(k):
        rho_hit = rho_hit or (_rho(qh, qdh, qt[t0 + h], w) > 1.0)
        qh, qdh = _step(qh, qdh, qt[t0 + h], w)
        if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
            finite = False
            mx = float("inf")
            break
        pred = _x(qh, qdh)
        ref = _x(qnn[t0 + h], qdnn[t0 + h])
        preds.append(pred)
        refs.append(ref)
        mx = max(mx, _e_h(pred, ref, scale))
    cata = (not finite) or mx > CATA_EH
    if preds:
        e_exec = _nrmse(np.stack(preds), np.stack(refs))
        e_end = _nrmse(preds[-1], refs[-1]) if finite else float("inf")
    else:
        e_exec = float("inf")
        e_end = float("inf")
    # identity: hold x_t0
    x0 = _x(qq[t0], qddt[t0])
    id_preds = np.stack([x0 for _ in range(k)])
    id_refs = np.stack([_x(qnn[t0 + h], qdnn[t0 + h]) for h in range(k)])
    e_id = _nrmse(id_preds, id_refs)
    return {
        "cata": cata,
        "nonfinite": not finite,
        "e_exec": e_exec,
        "e_end": e_end,
        "e_id": e_id,
        "rho_hit": rho_hit,
        "e_max": mx,
    }


def _receding(pool: dict[str, Any], w: np.ndarray, scale: float, k: int) -> dict[str, Any]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    segs: list[dict[str, Any]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        t0 = 0
        while t0 + k <= n_steps:
            # need k future transitions: indices t0 .. t0+k-1 for actions and qn
            segs.append(_segment_roll(qq, qddt, qnn, qdnn, qt, t0, k, w, scale))
            t0 += k
    n = len(segs)
    n_cata = sum(1 for s in segs if s["cata"])
    n_nf = sum(1 for s in segs if s["nonfinite"])
    e_execs = np.array([s["e_exec"] for s in segs], dtype=np.float64)
    e_ends = np.array([s["e_end"] for s in segs], dtype=np.float64)
    e_ids = np.array([s["e_id"] for s in segs], dtype=np.float64)
    # aggregate like rollout: any nonfinite -> inf mean via nrmse stacking is per-seg; report mean of segs
    def _mean_or_inf(a: np.ndarray) -> float:
        if a.size == 0:
            return float("inf")
        if not np.isfinite(a).all():
            return float("inf")
        return float(np.mean(a))

    return {
        "K": k,
        "n_segments": n,
        "n_cata": n_cata,
        "n_nonfinite": n_nf,
        "r_cat": float(n_cata / max(n, 1)),
        "E_exec": _mean_or_inf(e_execs),
        "E_end": _mean_or_inf(e_ends),
        "E_identity": _mean_or_inf(e_ids),
        "R_rho": float(np.mean([s["rho_hit"] for s in segs])) if segs else float("nan"),
        "H_predict_concept": H_PREDICT if k <= H_PREDICT else k,
    }


def _adaptive_shadow(pool: dict[str, Any], w: np.ndarray, scale: float) -> dict[str, Any]:
    """Shadow: rho>1 -> K=1 else K=8. Does not affect formal pattern."""
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    segs: list[dict[str, Any]] = []
    n_replan = 0
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        t0 = 0
        while t0 < n_steps:
            rho0 = _rho(qq[t0], qddt[t0], qt[t0], w)
            k = 1 if rho0 > 1.0 else 8
            if t0 + k > n_steps:
                break
            segs.append(_segment_roll(qq, qddt, qnn, qdnn, qt, t0, k, w, scale))
            n_replan += 1
            t0 += k
    n = len(segs)
    n_cata = sum(1 for s in segs if s["cata"])
    e_execs = np.array([s["e_exec"] for s in segs], dtype=np.float64)
    return {
        "policy": "adaptive_rho_gt1_K1_else_K8",
        "shadow_only": True,
        "n_segments": n,
        "n_cata": n_cata,
        "r_cat": float(n_cata / max(n, 1)),
        "n_nonfinite": sum(1 for s in segs if s["nonfinite"]),
        "E_exec": float(np.mean(e_execs)) if e_execs.size and np.isfinite(e_execs).all() else float("inf"),
        "replans_per_episode": float(n_replan / max(n_ep, 1)),
    }


def run_rtwx_x0eh(output: str | Path, config: RTWX0EHConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0EHConfig())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = {
        "stage": "RTWX-X0EH",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_m2": True,
        "no_retrain": True,
        "no_contraction": True,
        "no_capacity": True,
        "does_not_retune_prior": True,
        "K_grid": list(K_GRID),
        "K_primary": K_PRIMARY,
        "H_predict": H_PREDICT,
        "fresh_seed": SEED_FRESH,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0EH Receding-Horizon Sufficiency Audit\n"
        "frozen_M2=true no_retrain=true no_R_P=true\n"
        f"n_ep={cfg.n_ep} n_steps={cfg.n_steps} seed={cfg.seed}\n"
        f"K={list(K_GRID)} primary={K_PRIMARY} H_predict={H_PREDICT}\n",
        encoding="utf-8",
    )

    old = _load_cache(Path(cfg.old_cache))
    if old is None:
        raise RuntimeError(f"need {cfg.old_cache} for frozen M2 + scale")
    train = old["train"]
    w = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), _delta(train))
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    np.save(root / "W_m2.npy", w)

    cache_f = root / "fresh_splits.npz"
    if cache_f.is_file():
        z = np.load(cache_f)
        fresh = {k: np.asarray(z[k]) for k in ("q", "qd", "qn", "qdn", "q_tar")}
        fresh["n_ep"] = int(z["n_ep"][0])
        fresh["n_steps"] = int(z["n_steps"][0])
        fresh["eq"] = fresh["q_tar"] - fresh["q"]
        print("[rtwx-x0eh] using cached fresh", flush=True)
    else:
        print("[rtwx-x0eh] collecting fresh episodes", flush=True)
        es1 = RTWX0ES1Config(
            output=str(root / "fresh"),
            robotwin_repo=cfg.robotwin_repo,
            backend=cfg.backend,
            n_ep=int(cfg.n_ep),
            n_train_ep=int(cfg.n_ep),
            n_steps=int(cfg.n_steps),
            seed=int(cfg.seed),
            smoke=cfg.smoke,
            old_cache=cfg.old_cache,
            dt_numpy=cfg.dt_numpy,
            n_dof_numpy=cfg.n_dof_numpy,
            seed_attempts=int(cfg.seed_attempts),
            max_resample=int(cfg.max_resample),
        )
        fresh = collect_fresh(es1)
        fresh["eq"] = fresh["q_tar"] - fresh["q"]
        np.savez_compressed(
            cache_f,
            q=fresh["q"],
            qd=fresh["qd"],
            qn=fresh["qn"],
            qdn=fresh["qdn"],
            q_tar=fresh["q_tar"],
            n_ep=np.array([fresh["n_ep"]]),
            n_steps=np.array([fresh["n_steps"]]),
            n_dof=np.array([fresh["q"].shape[1]]),
        )

    print("[rtwx-x0eh] Part A open-loop T_div", flush=True)
    part_a = _open_loop_audit(fresh, w, scale)
    g0 = bool(part_a["n_cata"] >= N_CAT_MIN)
    if not g0:
        summary = {
            "header": header,
            "pattern": "rare_event_insufficient",
            "G0": False,
            "part_a": part_a,
            "no_capacity_claim": True,
            "unlocks_k4_capacity": False,
            "unlocks_contraction": False,
        }
        _write_json(root / "run.json", summary)
        _write_json(root / "summary.json", summary)
        _write_json(root / "metrics.json", {"pattern": "rare_event_insufficient", "part_a": part_a})
        return summary

    print("[rtwx-x0eh] Part B receding horizon", flush=True)
    by_k = {k: _receding(fresh, w, scale, k) for k in K_GRID}
    for k, blk in by_k.items():
        print(
            f"  K={k} r_cat={blk['r_cat']:.4g} E_exec={blk['E_exec']} E_end={blk['E_end']} n_nf={blk['n_nonfinite']}",
            flush=True,
        )

    k4, k50, k16 = by_k[4], by_k[50], by_k[16]
    g1 = bool(k4["n_nonfinite"] == 0 and k4["r_cat"] <= 0.1 * max(k50["r_cat"], 1.0e-12))
    g2 = bool(
        np.isfinite(k4["E_exec"])
        and np.isfinite(k4["E_identity"])
        and k4["E_identity"] > 0
        and k4["E_exec"] <= G2_RATIO * k4["E_identity"]
    )
    ks = np.array(list(K_GRID), dtype=np.float64)
    e_ends = np.array([by_k[k]["E_end"] for k in K_GRID], dtype=np.float64)
    e_rank = np.where(np.isfinite(e_ends), e_ends, 1.0e12)
    sp = spearmanr(ks, e_rank)
    spe_val = float(sp.correlation) if sp.correlation is not None else float("nan")
    g3 = bool(
        np.isfinite(spe_val)
        and spe_val >= SPEARMAN_MIN
        and (
            (np.isfinite(k16["E_end"]) and np.isfinite(k4["E_end"]) and k16["E_end"] > k4["E_end"])
            or (not np.isfinite(k16["E_end"]) and np.isfinite(k4["E_end"]))
        )
    )

    if g0 and g1 and g2 and g3:
        pattern = "short_horizon_sufficient"
    elif g1 and not g2:
        pattern = "short_horizon_stable_but_inaccurate"
    else:
        pattern = "replanning_insufficient"

    print("[rtwx-x0eh] shadow adaptive", flush=True)
    shadow = {
        "adaptive": _adaptive_shadow(fresh, w, scale),
        "fixed_K4": {kk: by_k[4][kk] for kk in ("r_cat", "E_exec", "n_segments", "n_cata")},
        "fixed_K8": {kk: by_k[8][kk] for kk in ("r_cat", "E_exec", "n_segments", "n_cata")},
    }

    summary = {
        "header": header,
        "pattern": pattern,
        "G0": g0,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "spearman_K_Eend": spe_val,
        "part_a": part_a,
        "by_K": {str(k): by_k[k] for k in K_GRID},
        "shadow": shadow,
        "no_capacity_claim": True,
        "unlocks_k4_capacity": pattern == "short_horizon_sufficient",
        "unlocks_contraction": pattern == "replanning_insufficient",
        "does_not_retune_prior": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "G0": g0,
            "G1": g1,
            "G2": g2,
            "G3": g3,
            "K99": part_a["K99"],
            "survival": part_a["survival"],
            "by_K": {str(k): {kk: by_k[k][kk] for kk in ("r_cat", "E_exec", "E_end", "E_identity", "R_rho", "n_cata", "n_nonfinite")} for k in K_GRID},
            "shadow_adaptive_r_cat": shadow["adaptive"]["r_cat"],
        },
    )
    return summary

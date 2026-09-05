"""RTWX-X0ES: frozen-M2 macro-transition rollout stability audit. No retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0c import _write_json
from .rtwx_x0e import _apply_dx, _phi_m2, _x
from .rtwx_x0e1 import _delta, _load_cache, _lstsq

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0ES_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0es.macro_rollout_audit.v2"
H_MAX = 50
TAU_DIV = 10.0
CATA_EH = 1.0e6
P_OOD_FIRST = 0.80
RHO_MED_GAP = 0.10
P_RHO_CAT = 0.50
RISK_SMD = 1.0
EPS_REL = 1.0e-3
A_HS = (1, 5, 10)


@dataclass(frozen=True)
class RTWX0ESConfig:
    output: str = "runs/rtwx_x0es"
    cache: str = "runs/rtwx_x0e1/cache_splits.npz"
    x0e1_summary: str = "runs/rtwx_x0e1/run.json"


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0ES must not write there")


def _z(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd, qtar - q], axis=-1)


def _robust_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(z, axis=0)
    q75 = np.percentile(z, 75, axis=0)
    q25 = np.percentile(z, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 1.0e-8, 1.0, iqr)
    return med, iqr


def _standardize(z: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return (z - med) / iqr


def _nn_min(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """query (n,d), bank (m,d) -> (n,) min L2."""
    q2 = np.sum(query * query, axis=1, keepdims=True)
    b2 = np.sum(bank * bank, axis=1, keepdims=True).T
    out = np.empty(query.shape[0], dtype=np.float64)
    bs = 256
    for i0 in range(0, query.shape[0], bs):
        sl = query[i0 : i0 + bs]
        d2 = q2[i0 : i0 + bs] + b2 - 2.0 * sl @ bank.T
        np.maximum(d2, 0.0, out=d2)
        out[i0 : i0 + bs] = np.sqrt(np.min(d2, axis=1))
    return out


def _loo_nn(z: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    q2 = np.sum(z * z, axis=1, keepdims=True)
    d2 = q2 + q2.T - 2.0 * z @ z.T
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.min(d2, axis=1))


def _dphi_dx(q: np.ndarray, qd: np.ndarray, eq: np.ndarray) -> np.ndarray:
    d = int(q.size)
    j = np.zeros((6 * d + 1, 2 * d), dtype=np.float64)
    eye = np.eye(d)
    j[0:d, 0:d] = eye
    j[d : 2 * d, d : 2 * d] = eye
    j[2 * d : 3 * d, 0:d] = -eye
    j[3 * d : 4 * d, 0:d] = np.diag(np.cos(q))
    j[4 * d : 5 * d, 0:d] = np.diag(-np.sin(q))
    j[5 * d : 6 * d, 0:d] = np.diag(-2.0 * np.abs(eq))
    return j


def _rho(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray, w: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    qtar = np.asarray(qtar, dtype=np.float64).reshape(-1)
    jac = np.eye(2 * q.size) + _dphi_dx(q, qd, qtar - q).T @ w
    return float(np.max(np.abs(np.linalg.eigvals(jac))))


def _step(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eq = qtar - q
    dx = (_phi_m2(q.reshape(1, -1), qd.reshape(1, -1), eq.reshape(1, -1)) @ w).reshape(-1)
    return _apply_dx(q, qd, dx)


def _fd_rho_check(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray, w: np.ndarray, eps: float = 1.0e-6) -> float:
    x0 = _x(q, qd)
    d = q.size
    jac = np.zeros((2 * d, 2 * d), dtype=np.float64)
    for j in range(2 * d):
        dx = np.zeros(2 * d)
        dx[j] = eps
        qp, qdp = x0[:d] + dx[:d], x0[d:] + dx[d:]
        qn, qdn = _step(qp, qdp, qtar, w)
        qn0, qdn0 = _step(q, qd, qtar, w)
        jac[:, j] = (_x(qn, qdn) - _x(qn0, qdn0)) / eps
    return float(np.max(np.abs(np.linalg.eigvals(jac))))


def _percentiles(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def _risks(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray, qtar_next: np.ndarray | None, q_lo: np.ndarray, q_hi: np.ndarray) -> dict[str, float]:
    eq = qtar - q
    dlim = np.minimum(q - q_lo, q_hi - q)
    dlim = np.maximum(dlim, 1.0e-8)
    dqtar = np.zeros_like(qtar) if qtar_next is None else (qtar_next - qtar)
    return {
        "r1": float(np.linalg.norm(eq)),
        "r2": float(np.linalg.norm(qd)),
        "r3": float(np.linalg.norm(dqtar)),
        "r4": float(1.0 / float(np.min(dlim))),
    }


def _smd(cat: np.ndarray, sta: np.ndarray) -> float:
    cat = np.asarray(cat, dtype=np.float64)
    sta = np.asarray(sta, dtype=np.float64)
    if cat.size == 0 or sta.size == 0:
        return float("nan")
    pooled = np.concatenate([cat, sta])
    iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
    scale = iqr / 1.349 if iqr > 1.0e-12 else float(np.std(pooled) + 1.0e-8)
    return float((np.median(cat) - np.median(sta)) / scale)


def _audit_split(
    pool: dict[str, Any],
    w: np.ndarray,
    *,
    z_bank: np.ndarray,
    med: np.ndarray,
    iqr: np.ndarray,
    d_ood: float,
    scale_x: float,
    x_scale_dim: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    do_amp: bool,
) -> dict[str, Any]:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    d = q.shape[1]
    wins: list[dict[str, Any]] = []
    n_start = n_steps - H_MAX
    pred_z_chunks: list[np.ndarray] = []
    pred_meta: list[tuple[int, int]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        for t0 in range(max(0, n_start)):
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            e_hist = []
            z_hist = [_z(qh, qdh, qt[t0])]
            rho0 = _rho(qh, qdh, qt[t0], w)
            rhos = [rho0]
            tf_err = []
            finite = True
            for h in range(H_MAX):
                q_tf, qd_tf = _step(qq[t0 + h], qddt[t0 + h], qt[t0 + h], w)
                true_n = _x(qnn[t0 + h], qdnn[t0 + h])
                tf_err.append(float(np.linalg.norm(_x(q_tf, qd_tf) - true_n) / (scale_x + 1.0e-8)))
                qh, qdh = _step(qh, qdh, qt[t0 + h], w)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    finite = False
                    e_hist.append(float("inf"))
                    break
                e_hist.append(float(np.linalg.norm(_x(qh, qdh) - true_n) / (scale_x + 1.0e-8)))
                ut = qt[min(t0 + h + 1, n_steps - 1)]
                z_hist.append(_z(qh, qdh, ut))
                rhos.append(_rho(qh, qdh, ut, w))
            e_arr = np.array(e_hist, dtype=np.float64)
            t_div = None
            for h, eh in enumerate(e_arr, start=1):
                if not np.isfinite(eh) or eh > TAU_DIV:
                    t_div = h
                    break
            mx = float(np.nanmax(e_arr)) if e_arr.size and np.isfinite(e_arr).any() else float("inf")
            cata = (not finite) or (not np.isfinite(mx)) or mx > CATA_EH
            qnxt = qt[t0 + 1] if t0 + 1 < n_steps else None
            rk = _risks(qq[t0], qddt[t0], qt[t0], qnxt, q_lo, q_hi)
            amp = {}
            want_amp = do_amp and (cata or (t0 % 20 == 0 and ep % 4 == 0))
            if want_amp:
                for hh in A_HS:
                    acc = []
                    x0 = _x(qq[t0], qddt[t0])
                    for j in range(2 * d):
                        for sgn in (1.0, -1.0):
                            dx = np.zeros(2 * d)
                            dx[j] = sgn * EPS_REL * max(float(x_scale_dim[j]), 1.0e-6)
                            qe, qde = x0[:d] + dx[:d], x0[d:] + dx[d:]
                            qn0, qdn0 = qq[t0].copy(), qddt[t0].copy()
                            qne, qde2 = qe.copy(), qde.copy()
                            ok = True
                            for s in range(hh):
                                qn0, qdn0 = _step(qn0, qdn0, qt[t0 + s], w)
                                qne, qde2 = _step(qne, qde2, qt[t0 + s], w)
                                if not np.isfinite(qne).all():
                                    ok = False
                                    break
                            if ok:
                                num = float(np.linalg.norm(_x(qne, qde2) - _x(qn0, qdn0)))
                                den = float(np.linalg.norm(dx) + 1.0e-12)
                                acc.append(num / den)
                    amp[f"A_{hh}"] = float(np.median(acc)) if acc else float("inf")
            wins.append(
                {
                    "ep": ep,
                    "t0": t0,
                    "cata": cata,
                    "t_div": t_div,
                    "e_max": mx,
                    "e1": float(e_arr[0]) if e_arr.size else float("inf"),
                    "tf_med": float(np.median(tf_err)) if tf_err else float("inf"),
                    "rho0": rho0,
                    "rho_max": float(np.nanmax(rhos)) if rhos else float("nan"),
                    "rho_prediv": float(rhos[0] if t_div is None else np.nanmax(rhos[:t_div])),
                    **rk,
                    **amp,
                    "n_z": len(z_hist),
                }
            )
            pred_z_chunks.append(np.stack(z_hist, axis=0))
            pred_meta.append((len(wins) - 1, len(z_hist)))

    z_all = np.concatenate(pred_z_chunks, axis=0) if pred_z_chunks else np.zeros((0, z_bank.shape[1]))
    z_std = _standardize(z_all, med, iqr) if z_all.size else z_all
    dnn = _nn_min(z_std, z_bank) if z_all.size else np.zeros(0)
    offset = 0
    for idx, nz in pred_meta:
        dd = dnn[offset : offset + nz]
        offset += nz
        t_ood = None
        for k, dv in enumerate(dd):
            if float(dv) > d_ood:
                t_ood = int(k)
                break
        t_div = wins[idx]["t_div"]
        td = float("inf") if t_div is None else float(t_div)
        to = float("inf") if t_ood is None else float(t_ood)
        wins[idx]["t_ood"] = t_ood
        wins[idx]["delta_t"] = None if (t_div is None and t_ood is None) else (td - to)
        wins[idx]["ood_before_div"] = bool(t_ood is not None and t_div is not None and t_ood < t_div)

    cat = [w for w in wins if w["cata"]]
    sta = [w for w in wins if not w["cata"]]
    e_roll = [w["e_max"] for w in wins]
    return {
        "n_windows": len(wins),
        "n_cata": len(cat),
        "cata_frac": float(len(cat) / max(len(wins), 1)),
        "t_div_finite_frac": float(np.mean([w["t_div"] is not None for w in wins])),
        "p_ood_before_div_cata": float(np.mean([w["ood_before_div"] for w in cat])) if cat else float("nan"),
        "median_tf_cata": float(np.median([w["tf_med"] for w in cat])) if cat else float("nan"),
        "median_tf_sta": float(np.median([w["tf_med"] for w in sta])) if sta else float("nan"),
        "median_e1_cata": float(np.median([w["e1"] for w in cat])) if cat else float("nan"),
        "median_e1_sta": float(np.median([w["e1"] for w in sta])) if sta else float("nan"),
        "rho0": _percentiles(np.array([w["rho0"] for w in wins])),
        "rho_cata": _percentiles(np.array([w["rho_prediv"] for w in cat])) if cat else _percentiles(np.array([])),
        "rho_sta": _percentiles(np.array([w["rho_prediv"] for w in sta])) if sta else _percentiles(np.array([])),
        "p_rho_gt1_cata": float(np.mean([w["rho_prediv"] > 1.0 for w in cat])) if cat else float("nan"),
        "p_rho_gt1_sta": float(np.mean([w["rho_prediv"] > 1.0 for w in sta])) if sta else float("nan"),
        "A10_cata": float(np.median([w.get("A_10", np.nan) for w in cat])) if cat and do_amp else float("nan"),
        "A10_sta": float(np.median([w.get("A_10", np.nan) for w in sta])) if sta and do_amp else float("nan"),
        "e_max_p99": float(np.nanpercentile(np.array(e_roll, dtype=np.float64), 99)) if e_roll else float("nan"),
        "by_ep_cata": {str(ep): float(np.mean([w["cata"] for w in wins if w["ep"] == ep])) for ep in range(n_ep)},
        "cat_examples": cat[:8],
        "risk_smd": {k: _smd(np.array([w[k] for w in cat]), np.array([w[k] for w in sta])) for k in ("r1", "r2", "r3", "r4")}
        if cat and sta
        else {k: float("nan") for k in ("r1", "r2", "r3", "r4")},
        "risk_q99": {k: {"all": float(np.percentile([w[k] for w in wins], 99)) if wins else float("nan")} for k in ("r1", "r2", "r3", "r4")},
    }


def _pattern(tr: dict[str, Any], va: dict[str, Any], te: dict[str, Any], feat_q99: dict[str, Any]) -> dict[str, Any]:
    g0 = bool(tr["n_cata"] > 0 and va["n_cata"] == 0 and te["n_cata"] == 0)
    if not g0:
        return {"G0": False, "pattern": "pathology_not_reproduced"}
    p_ood = tr["p_ood_before_div_cata"]
    g1 = bool(np.isfinite(p_ood) and p_ood >= P_OOD_FIRST)
    if g1:
        return {"G0": True, "G1": True, "G2": False, "G3": False, "pattern": "support_excursion", "p_ood_before_div": p_ood}
    med_c = tr["rho_cata"]["p50"]
    med_s = tr["rho_sta"]["p50"]
    g2 = bool(
        np.isfinite(med_c)
        and np.isfinite(med_s)
        and (med_c - med_s) >= RHO_MED_GAP
        and tr["p_rho_gt1_cata"] >= P_RHO_CAT
    )
    if g2:
        return {
            "G0": True,
            "G1": False,
            "G2": True,
            "G3": False,
            "pattern": "local_instability",
            "rho_med_gap": float(med_c - med_s),
            "p_rho_gt1_cata": tr["p_rho_gt1_cata"],
        }
    smd = tr["risk_smd"]
    hit = [k for k, v in smd.items() if np.isfinite(v) and abs(v) >= RISK_SMD]
    train_heavier = False
    for k in hit:
        if feat_q99["train"][k] > feat_q99["val"][k] and feat_q99["train"][k] > feat_q99["test"][k]:
            train_heavier = True
    g3 = bool(len(hit) > 0 and train_heavier)
    if g3:
        return {"G0": True, "G1": False, "G2": False, "G3": True, "pattern": "split_support_pathology", "risk_hit": hit, "smd": smd}
    return {"G0": True, "G1": False, "G2": False, "G3": False, "pattern": "instability_unresolved", "smd": smd, "p_ood_before_div": p_ood}


def run_rtwx_x0es(output: str | Path, config: RTWX0ESConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0ESConfig()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = {
        "stage": "RTWX-X0ES",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "does_not_retune_x0e_or_x0e1": True,
        "no_retrain": True,
        "no_capacity": True,
        "H_max": H_MAX,
        "tau_div": TAU_DIV,
        "cata_EH": CATA_EH,
        "d_ood": "Q99 LOO train NN in robust-standardized z=[q,qd,eq]",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0ES Macro-Transition Rollout Stability Audit\n"
        "no_retrain=true no_capacity=true does_not_retune_x0e_or_x0e1=true\n"
        f"H_max={H_MAX} tau_div={TAU_DIV} cata_EH={CATA_EH}\n",
        encoding="utf-8",
    )
    splits = _load_cache(Path(cfg.cache))
    if splits is None:
        raise RuntimeError(f"X0ES requires cache at {cfg.cache}")
    train, val, test = splits["train"], splits["val"], splits["test"]
    w = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), _delta(train))
    z_tr = _z(train["q"], train["qd"], train["q_tar"])
    med, iqr = _robust_fit(z_tr)
    z_bank = _standardize(z_tr, med, iqr)
    loo = _loo_nn(z_bank)
    d_ood = float(np.percentile(loo, 99))
    x_tr = _x(train["q"], train["qd"])
    scale_x = float(np.sqrt(np.mean(np.square(x_tr))))
    x_scale_dim = np.std(x_tr, axis=0)
    x_scale_dim = np.where(x_scale_dim < 1.0e-8, 1.0, x_scale_dim)
    q_lo = np.min(train["q"], axis=0)
    q_hi = np.max(train["q"], axis=0)

    q0 = train["q"][0]
    fd = _fd_rho_check(q0, train["qd"][0], train["q_tar"][0], w)
    an = _rho(q0, train["qd"][0], train["q_tar"][0], w)

    print("[rtwx-x0es] audit train", flush=True)
    tr = _audit_split(train, w, z_bank=z_bank, med=med, iqr=iqr, d_ood=d_ood, scale_x=scale_x, x_scale_dim=x_scale_dim, q_lo=q_lo, q_hi=q_hi, do_amp=True)
    print("[rtwx-x0es] audit val", flush=True)
    va = _audit_split(val, w, z_bank=z_bank, med=med, iqr=iqr, d_ood=d_ood, scale_x=scale_x, x_scale_dim=x_scale_dim, q_lo=q_lo, q_hi=q_hi, do_amp=False)
    print("[rtwx-x0es] audit test", flush=True)
    te = _audit_split(test, w, z_bank=z_bank, med=med, iqr=iqr, d_ood=d_ood, scale_x=scale_x, x_scale_dim=x_scale_dim, q_lo=q_lo, q_hi=q_hi, do_amp=False)

    def feat_q99(pool: dict[str, Any]) -> dict[str, float]:
        q, qd, qtar = pool["q"], pool["qd"], pool["q_tar"]
        eq = qtar - q
        dqt = np.zeros_like(qtar)
        dqt[1:] = qtar[1:] - qtar[:-1]
        dlim = np.minimum(q - q_lo, q_hi - q)
        r4 = 1.0 / np.maximum(np.min(dlim, axis=1), 1.0e-8)
        return {
            "r1": float(np.percentile(np.linalg.norm(eq, axis=1), 99)),
            "r2": float(np.percentile(np.linalg.norm(qd, axis=1), 99)),
            "r3": float(np.percentile(np.linalg.norm(dqt, axis=1), 99)),
            "r4": float(np.percentile(r4, 99)),
        }

    fq = {"train": feat_q99(train), "val": feat_q99(val), "test": feat_q99(test)}
    scored = _pattern(tr, va, te, fq)
    summary = {
        "header": header,
        "no_capacity_claim": True,
        "does_not_retune_x0e_or_x0e1": True,
        "unlocks_r10": False,
        "d_ood": d_ood,
        "scale_x": scale_x,
        "W_shape": list(w.shape),
        "fd_vs_analytic_rho": {"fd": fd, "analytic": an, "abs_diff": abs(fd - an)},
        "feature_q99": fq,
        "train": tr,
        "val": va,
        "test": te,
        **scored,
        "teacher_forced_vs_free": {
            "note": "if TF one-step stays ordinary while free explodes, recursive stability not local fit",
            "train_tf_cata": tr["median_tf_cata"],
            "train_e1_cata": tr["median_e1_cata"],
            "train_tf_sta": tr["median_tf_sta"],
            "train_e1_sta": tr["median_e1_sta"],
        },
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": scored["pattern"],
            "G0": scored.get("G0"),
            "n_cata": {"train": tr["n_cata"], "val": va["n_cata"], "test": te["n_cata"]},
            "p_ood_before_div_cata": tr["p_ood_before_div_cata"],
            "rho_cata_p50": tr["rho_cata"]["p50"],
            "rho_sta_p50": tr["rho_sta"]["p50"],
            "risk_smd": tr["risk_smd"],
            "d_ood": d_ood,
        },
    )
    return summary

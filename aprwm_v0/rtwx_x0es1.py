"""RTWX-X0ES1: confirm rare high-rho_J predicts catastrophe on fresh data. Frozen M2."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact

from .rtwx_x0c import FORMAL_N_STEPS, collect_robotwin, _write_json
from .rtwx_x0e import _phi_m2, _x
from .rtwx_x0e1 import _delta, _load_cache, _lstsq
from .rtwx_x0es import CATA_EH, H_MAX, _rho, _step
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0ES1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0es1.rare_rho_confirm.v1"
N_FRESH = 96
SEED_FRESH = 9601
N_CAT_MIN = 5
RECALL_MIN = 0.80
RR_MIN = 10.0
FISHER_P = 0.01
K_NN = 5
G_H_MIN = 0.20
N_BOOT = 2000
OLD_CACHE = "runs/rtwx_x0e1/cache_splits.npz"


@dataclass(frozen=True)
class RTWX0ES1Config:
    output: str = "runs/rtwx_x0es1"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_ep: int = N_FRESH
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED_FRESH
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    old_cache: str = OLD_CACHE
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    n_train_ep: int = N_FRESH
    n_val_ep: int = 0
    n_test_ep: int = 0


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0ES1 must not write there")


def _lock(cfg: RTWX0ES1Config) -> RTWX0ES1Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, n_ep=N_FRESH, n_steps=FORMAL_N_STEPS, seed=SEED_FRESH, n_train_ep=N_FRESH)


def _e_h(hat: np.ndarray, ref: np.ndarray, scale: float) -> float:
    if not np.isfinite(hat).all():
        return float("inf")
    return float(np.linalg.norm(hat - ref) / (scale + 1.0e-8))


def _windows(pool: dict[str, Any], w: np.ndarray, scale: float) -> list[dict[str, Any]]:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    out: list[dict[str, Any]] = []
    n_start = n_steps - H_MAX
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        for t0 in range(max(0, n_start)):
            rho0 = _rho(qq[t0], qddt[t0], qt[t0], w)
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            mx = 0.0
            finite = True
            for h in range(H_MAX):
                qh, qdh = _step(qh, qdh, qt[t0 + h], w)
                eh = _e_h(_x(qh, qdh), _x(qnn[t0 + h], qdnn[t0 + h]), scale)
                if not np.isfinite(eh):
                    finite = False
                    mx = float("inf")
                    break
                mx = max(mx, eh)
            cata = (not finite) or mx > CATA_EH
            eq = qt[t0] - qq[t0]
            rec: dict[str, Any] = {
                "ep": ep,
                "t0": t0,
                "rho": rho0,
                "high": bool(rho0 > 1.0),
                "cata": cata,
                "e_max": mx,
            }
            if t0 >= 1:
                rec["z0"] = np.concatenate([qq[t0], qddt[t0], eq])
                rec["z1"] = np.concatenate(
                    [
                        qq[t0],
                        qddt[t0],
                        eq,
                        qq[t0] - qq[t0 - 1],
                        qddt[t0] - qddt[t0 - 1],
                        qt[t0 - 1] - qq[t0 - 1],
                    ]
                )
                rec["dx"] = np.concatenate([qnn[t0] - qq[t0], qdnn[t0] - qddt[t0]])
            out.append(rec)
    return out


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hr_c = sum(1 for r in rows if r["high"] and r["cata"])
    hr_s = sum(1 for r in rows if r["high"] and not r["cata"])
    lr_c = sum(1 for r in rows if (not r["high"]) and r["cata"])
    lr_s = sum(1 for r in rows if (not r["high"]) and not r["cata"])
    n_cat = hr_c + lr_c
    n_hr = hr_c + hr_s
    n_lr = lr_c + lr_s
    recall = float(hr_c / n_cat) if n_cat else float("nan")
    p_c_hr = float(hr_c / n_hr) if n_hr else float("nan")
    p_c_lr = float(lr_c / n_lr) if n_lr else 0.0
    if n_hr == 0:
        rr = float("nan")
    elif p_c_lr == 0.0 and p_c_hr > 0:
        rr = float("inf")
    else:
        rr = float(p_c_hr / (p_c_lr + 1.0e-15))
    table = np.array([[hr_c, hr_s], [lr_c, lr_s]], dtype=np.int64)
    if table.min() < 0 or table.sum() == 0:
        p_fish = 1.0
    else:
        _, p_fish = fisher_exact(table, alternative="greater")
    prec = float(hr_c / n_hr) if n_hr else float("nan")
    return {
        "hr_cat": hr_c,
        "hr_sta": hr_s,
        "lr_cat": lr_c,
        "lr_sta": lr_s,
        "n_cat": n_cat,
        "n_windows": len(rows),
        "recall": recall,
        "precision": prec,
        "RR": rr,
        "fisher_p": float(p_fish),
        "frac_high": float(n_hr / max(len(rows), 1)),
    }


def _auprc(rows: list[dict[str, Any]]) -> float:
    scores = np.array([r["rho"] for r in rows], dtype=np.float64)
    y = np.array([1 if r["cata"] else 0 for r in rows], dtype=np.int32)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / float(y.sum())
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.trapz(prec, rec))


def _nn_predict(z: np.ndarray, bank_z: np.ndarray, bank_y: np.ndarray, k: int) -> np.ndarray:
    d2 = np.sum((bank_z - z.reshape(1, -1)) ** 2, axis=1)
    idx = np.argpartition(d2, min(k, bank_z.shape[0] - 1))[:k]
    return bank_y[idx].mean(axis=0)


def _history_gain(rows: list[dict[str, Any]], bank0: tuple[np.ndarray, np.ndarray], bank1: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    hrs = [r for r in rows if r["high"] and "z1" in r and "dx" in r]
    if len(hrs) < 3:
        return {"n": len(hrs), "G_H": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "E0": float("nan"), "E1": float("nan")}
    e0, e1 = [], []
    for r in hrs:
        p0 = _nn_predict(r["z0"], bank0[0], bank0[1], K_NN)
        p1 = _nn_predict(r["z1"], bank1[0], bank1[1], K_NN)
        dx = r["dx"]
        nrm = float(np.linalg.norm(dx) + 1.0e-8)
        e0.append(float(np.linalg.norm(p0 - dx) / nrm))
        e1.append(float(np.linalg.norm(p1 - dx) / nrm))
    e0 = np.array(e0)
    e1 = np.array(e1)
    E0, E1 = float(e0.mean()), float(e1.mean())
    gh = float(1.0 - E1 / (E0 + 1.0e-15))
    rng = np.random.default_rng(0)
    boots = []
    n = e0.size
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, size=n)
        boots.append(1.0 - float(e1[ix].mean()) / (float(e0[ix].mean()) + 1.0e-15))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"n": n, "G_H": gh, "ci_lo": lo, "ci_hi": hi, "E0": E0, "E1": E1}


def _bank_from_old(old: dict[str, Any]) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    q, qd, qn, qdn, qtar = old["q"], old["qd"], old["qn"], old["qdn"], old["q_tar"]
    n_steps = int(old["n_steps"])
    n_ep = int(old["n_ep"])
    z0, z1, y = [], [], []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        for t in range(1, n_steps):
            eq = qt[t] - qq[t]
            z0.append(np.concatenate([qq[t], qddt[t], eq]))
            z1.append(
                np.concatenate(
                    [qq[t], qddt[t], eq, qq[t] - qq[t - 1], qddt[t] - qddt[t - 1], qt[t - 1] - qq[t - 1]]
                )
            )
            y.append(np.concatenate([qnn[t] - qq[t], qdnn[t] - qddt[t]]))
    z0 = np.asarray(z0)
    z1 = np.asarray(z1)
    y = np.asarray(y)
    return (z0, y), (z1, y)


def collect_fresh(cfg: RTWX0ES1Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.backend == "numpy":
        from .rtwx_x0e1 import RTWX0E1Config, collect_numpy_x0e1

        c1 = RTWX0E1Config(
            backend="numpy",
            n_train_ep=int(cfg.n_ep),
            n_val_ep=1,
            n_test_ep=1,
            n_steps=int(cfg.n_steps),
            seed=int(cfg.seed),
            dt_numpy=cfg.dt_numpy,
            n_dof_numpy=cfg.n_dof_numpy,
            smoke=True,
        )
        return collect_numpy_x0e1(c1, split="train", rng=rng)
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    stop: list[str] = []
    return collect_robotwin(cfg, split="train", rng=rng, stop_mode=stop)  # type: ignore[arg-type]


def run_rtwx_x0es1(output: str | Path, config: RTWX0ES1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0ES1Config(output=str(output)))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = {
        "stage": "RTWX-X0ES1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_m2": True,
        "no_retrain": True,
        "no_capacity": True,
        "does_not_retune_x0e_x0e1_x0es": True,
        "n_fresh": cfg.n_ep,
        "seed": cfg.seed,
        "high_risk": "rho_J > 1",
        "cata": "max E_H > 1e6 or nonfinite, H=1..50",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0ES1 Rare Local Instability Confirmation\n"
        "frozen_M2=true no_retrain=true no_capacity=true\n"
        f"n_ep={cfg.n_ep} n_steps={cfg.n_steps} seed={cfg.seed}\n"
        "rho_threshold=1 cata_EH=1e6 H_max=50\n",
        encoding="utf-8",
    )
    old = _load_cache(Path(cfg.old_cache))
    if old is None:
        raise RuntimeError(f"need old X0E train cache {cfg.old_cache}")
    old_tr = old["train"]
    w = _lstsq(_phi_m2(old_tr["q"], old_tr["qd"], old_tr["eq"]), _delta(old_tr))
    scale = float(np.sqrt(np.mean(np.square(_x(old_tr["q"], old_tr["qd"])))))
    cache_f = root / "fresh_splits.npz"
    if cache_f.is_file():
        z = np.load(cache_f)
        fresh = {k: np.asarray(z[k]) for k in ("q", "qd", "qn", "qdn", "q_tar")}
        fresh["n_ep"] = int(z["n_ep"][0])
        fresh["n_steps"] = int(z["n_steps"][0])
        fresh["n_dof"] = int(z["n_dof"][0])
        fresh["eq"] = fresh["q_tar"] - fresh["q"]
    else:
        print("[rtwx-x0es1] collecting fresh episodes", flush=True)
        fresh = collect_fresh(cfg)
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
            n_dof=np.array([fresh.get("n_dof", fresh["q"].shape[1])]),
        )
    print("[rtwx-x0es1] scoring windows", flush=True)
    rows = _windows(fresh, w, scale)
    tab = _table(rows)
    auprc = _auprc(rows)
    g0 = bool(tab["n_cat"] >= N_CAT_MIN)
    rr = tab["RR"]
    rr_ok = bool(np.isfinite(rr) and rr >= RR_MIN) or (not np.isfinite(rr) and tab["hr_cat"] > 0 and tab["lr_cat"] == 0)
    g1 = bool(g0 and tab["recall"] >= RECALL_MIN and rr_ok and tab["fisher_p"] < FISHER_P)
    hist: dict[str, Any] = {}
    if g1:
        b0, b1 = _bank_from_old(old_tr)
        hist = _history_gain(rows, b0, b1)
        g2 = bool(np.isfinite(hist["G_H"]) and hist["G_H"] >= G_H_MIN and hist["ci_lo"] > 0)
        pattern = "rare_instability_history_aliasing" if g2 else "rare_instability_intrinsic_map"
    elif not g0:
        pattern = "rare_event_insufficient"
    else:
        pattern = "rare_instability_not_confirmed"
    summary = {
        "header": header,
        "pattern": pattern,
        "no_capacity_claim": True,
        "does_not_retune_x0e_x0e1_x0es": True,
        "G0": g0,
        "G1": g1,
        "table": tab,
        "AUPRC": auprc,
        "history": hist,
        "scale_x": scale,
        "W_shape": list(w.shape),
        "n_ep": int(fresh["n_ep"]),
        "n_steps": int(fresh["n_steps"]),
        "source": fresh.get("source", "robotwin"),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(root / "metrics.json", {"pattern": pattern, "G0": g0, "G1": g1, "table": tab, "AUPRC": auprc, "history": hist})
    return summary

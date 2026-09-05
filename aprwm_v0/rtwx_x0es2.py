"""RTWX-X0ES2: contractive penalty on same X0E-M2 family. No new basis/residual/capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0c import FORMAL_N_STEPS, _write_json
from .rtwx_x0e import _phi_m2, _x
from .rtwx_x0e1 import _delta, _eval_m2, _load_cache, _lstsq
from .rtwx_x0es import CATA_EH, H_MAX, _dphi_dx, _rho, _step
from .rtwx_x0es1 import N_FRESH, RTWX0ES1Config, collect_fresh

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0ES2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0es2.contractive_m2.v1"
LAMBDAS: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
GAMMA = 0.98
E1_MAX = 0.802
ER10_MAX = 0.763
ER50_MAX = 0.835
SEED_FRESH = 10601
EPOCHS = 80
BATCH = 256
LR = 1.0e-3
OLD_CACHE = "runs/rtwx_x0e1/cache_splits.npz"


@dataclass(frozen=True)
class RTWX0ES2Config:
    output: str = "runs/rtwx_x0es2"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    old_cache: str = OLD_CACHE
    seed: int = SEED_FRESH
    smoke: bool = False
    n_ep: int = N_FRESH
    n_steps: int = 120
    n_train_ep: int = N_FRESH
    n_val_ep: int = 0
    n_test_ep: int = 0
    seed_attempts: int = 32
    max_resample: int = 16
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    epochs: int = EPOCHS
    select_only: bool = False


def _lock(cfg: RTWX0ES2Config) -> RTWX0ES2Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_ep=N_FRESH,
        n_train_ep=N_FRESH,
        n_steps=FORMAL_N_STEPS,
        seed=SEED_FRESH,
        epochs=EPOCHS,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0ES2 must not write there")


def _sigma_max_np(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray, w: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    qtar = np.asarray(qtar, dtype=np.float64).reshape(-1)
    d = q.size
    j = np.eye(2 * d) + _dphi_dx(q, qd, qtar - q).T @ w
    s = np.linalg.svd(j, compute_uv=False)
    return float(s[0])


def _count_cata(pool: dict[str, Any], w: np.ndarray, scale: float) -> dict[str, Any]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    n_start = n_steps - H_MAX
    n_nf = 0
    n_cata = 0
    n_win = 0
    rhos = []
    sigs = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        for t0 in range(max(0, n_start)):
            n_win += 1
            rho0 = _rho(qq[t0], qddt[t0], qt[t0], w)
            sigs.append(_sigma_max_np(qq[t0], qddt[t0], qt[t0], w))
            rhos.append(rho0)
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            mx = 0.0
            finite = True
            for h in range(H_MAX):
                qh, qdh = _step(qh, qdh, qt[t0 + h], w)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    finite = False
                    n_nf += 1
                    break
                err = float(np.linalg.norm(_x(qh, qdh) - _x(qnn[t0 + h], qdnn[t0 + h])) / (scale + 1.0e-8))
                mx = max(mx, err)
            if (not finite) or mx > CATA_EH:
                n_cata += 1
    rhos = np.array(rhos, dtype=np.float64)
    sigs = np.array(sigs, dtype=np.float64)
    return {
        "n_windows": n_win,
        "n_cata": n_cata,
        "n_nonfinite": n_nf,
        "r_cat": float(n_cata / max(n_win, 1)),
        "p_rho_gt1": float(np.mean(rhos > 1.0)) if rhos.size else float("nan"),
        "rho_q50": float(np.percentile(rhos, 50)) if rhos.size else float("nan"),
        "rho_q90": float(np.percentile(rhos, 90)) if rhos.size else float("nan"),
        "rho_q99": float(np.percentile(rhos, 99)) if rhos.size else float("nan"),
        "smax_q99": float(np.percentile(sigs, 99)) if sigs.size else float("nan"),
    }


def _dphi_torch(q, qd, eq):
    import torch

    b, d = q.shape
    eye = torch.eye(d, device=q.device, dtype=q.dtype).expand(b, d, d)
    z = torch.zeros(b, d, d, device=q.device, dtype=q.dtype)
    # p = 6d+1, xdim=2d
    blocks = [
        torch.cat([eye, z], dim=2),  # dphi_q / dx : I on q
        torch.cat([z, eye], dim=2),  # qd
        torch.cat([-eye, z], dim=2),  # eq
        torch.cat([torch.diag_embed(torch.cos(q)), z], dim=2),
        torch.cat([torch.diag_embed(-torch.sin(q)), z], dim=2),
        torch.cat([torch.diag_embed(-2.0 * eq.abs()), z], dim=2),
        torch.zeros(b, 1, 2 * d, device=q.device, dtype=q.dtype),
    ]
    return torch.cat(blocks, dim=1)


def _train_w(q, qd, eq, y, w0: np.ndarray, lam: float, epochs: int) -> np.ndarray:
    import torch
    from torch import nn

    device = torch.device("cpu")
    qt = torch.tensor(q, dtype=torch.float64, device=device)
    qdt = torch.tensor(qd, dtype=torch.float64, device=device)
    eqt = torch.tensor(eq, dtype=torch.float64, device=device)
    yt = torch.tensor(y, dtype=torch.float64, device=device)
    w = nn.Parameter(torch.tensor(w0, dtype=torch.float64, device=device))
    opt = torch.optim.Adam([w], lr=LR)
    torch.manual_seed(10601)
    n = qt.shape[0]
    d = q.shape[1]
    eye = torch.eye(2 * d, dtype=torch.float64)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i0 in range(0, n, BATCH):
            sl = perm[i0 : i0 + BATCH]
            qb, qdb, eqb, yb = qt[sl], qdt[sl], eqt[sl], yt[sl]
            ones = torch.ones(qb.shape[0], 1, dtype=torch.float64)
            phi = torch.cat([qb, qdb, eqb, torch.sin(qb), torch.cos(qb), eqb * eqb.abs(), ones], dim=1)
            pred = phi @ w
            l_fit = torch.mean((pred - yb) ** 2)
            dphi = _dphi_torch(qb, qdb, eqb)
            jac = eye.expand(qb.shape[0], 2 * d, 2 * d) + torch.einsum("bpd,po->bdo", dphi, w)
            smax = torch.linalg.svdvals(jac)[:, 0]
            l_c = torch.mean(torch.relu(smax - GAMMA) ** 2)
            loss = l_fit + float(lam) * l_c
            opt.zero_grad()
            loss.backward()
            opt.step()
    return w.detach().cpu().numpy()


def _val_ok(metrics: dict[str, Any], cata: dict[str, Any]) -> bool:
    if cata["n_nonfinite"] != 0 or cata["n_cata"] != 0:
        return False
    if not np.isfinite(metrics["E_1"]) or not np.isfinite(metrics["E_roll10"]) or not np.isfinite(metrics["E_roll50"]):
        return False
    return bool(metrics["E_1"] <= E1_MAX and metrics["E_roll10"] <= ER10_MAX and metrics["E_roll50"] <= ER50_MAX)


def _g1(s1c: dict[str, Any], b0c: dict[str, Any]) -> bool:
    if s1c["n_nonfinite"] != 0:
        return False
    if b0c["r_cat"] <= 0.0:
        return bool(s1c["n_cata"] == 0)
    return bool(s1c["r_cat"] <= 0.1 * b0c["r_cat"])


def _g3(s1c: dict[str, Any], b0c: dict[str, Any]) -> bool:
    return bool(s1c["p_rho_gt1"] <= 0.5 * b0c["p_rho_gt1"])


def run_rtwx_x0es2(output: str | Path, config: RTWX0ES2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0ES2Config())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = {
        "stage": "RTWX-X0ES2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "same_basis": True,
        "P_infer": 1752,
        "gamma": GAMMA,
        "lambdas": list(LAMBDAS),
        "fresh_seed": SEED_FRESH,
        "no_capacity": True,
        "does_not_retune_prior": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0ES2 Stability-Constrained Increment\n"
        f"lambda={list(LAMBDAS)} gamma={GAMMA} P=1752\n"
        f"V3 E1<={E1_MAX} Er10<={ER10_MAX} Er50<={ER50_MAX}\n"
        "select smallest lambda; no R_P\n",
        encoding="utf-8",
    )
    old = _load_cache(Path(cfg.old_cache))
    if old is None:
        raise RuntimeError(f"need {cfg.old_cache}")
    train, val = old["train"], old["val"]
    y = _delta(train)
    w0 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), y)
    if cfg.backend == "robotwin" and not cfg.smoke and int(w0.size) != 1752:
        raise RuntimeError(f"W size {w0.size} != 1752")
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    sel_path = root / "selection.json"
    resume = (root / "W_s1.npy").is_file() and sel_path.is_file() and not cfg.select_only
    if resume:
        import json

        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        w_s1 = np.load(root / "W_s1.npy")
        chosen = {
            "lambda": float(sel["lambda_selected"]),
            "W": w_s1,
            "val": sel.get("val_chosen", {}).get("val"),
            "cata": sel.get("val_chosen", {}).get("cata"),
            "ok": True,
        }
        grid = sel.get("grid", [])
        print(f"[rtwx-x0es2] resume frozen lambda={chosen['lambda']}", flush=True)
    else:
        grid = []
        for lam in LAMBDAS:
            print(f"[rtwx-x0es2] train lambda={lam}", flush=True)
            w = _train_w(train["q"], train["qd"], train["eq"], y, w0, lam, cfg.epochs)
            if int(w.size) != int(w0.size):
                raise RuntimeError("inference param count changed")
            np.save(root / f"W_lam_{lam:g}.npy", w)
            met = _eval_m2(val, w)
            cat = _count_cata(val, w, scale)
            ok = _val_ok(met, cat)
            grid.append({"lambda": lam, "val": met, "cata": cat, "ok": ok, "W": w})
            print(
                f"  E1={met['E_1']:.4f} r10={met['E_roll10']:.4f} r50={met['E_roll50']:.4f} n_cata={cat['n_cata']} ok={ok}",
                flush=True,
            )

        cands = [g for g in grid if g["ok"]]
        if not cands:
            summary = {
                "header": header,
                "pattern": "constraint_selection_failure",
                "no_capacity_claim": True,
                "grid": [{k: v for k, v in g.items() if k != "W"} for g in grid],
            }
            _write_json(root / "run.json", summary)
            _write_json(root / "summary.json", summary)
            _write_json(
                root / "metrics.json",
                {
                    "pattern": "constraint_selection_failure",
                    "grid": [{k: g[k] for k in ("lambda", "ok", "val", "cata")} for g in grid],
                },
            )
            return summary

        chosen = min(cands, key=lambda g: g["lambda"])
        np.save(root / "W_s1.npy", chosen["W"])
        np.save(root / "W_b0.npy", w0)
        selection = {
            "header": header,
            "lambda_selected": chosen["lambda"],
            "val_chosen": {k: chosen[k] for k in ("lambda", "val", "cata", "ok")},
            "grid": [{k: g[k] for k in ("lambda", "ok", "val", "cata")} for g in grid],
            "P_infer": int(w0.size),
            "select_only": cfg.select_only,
        }
        _write_json(root / "selection.json", selection)
        if cfg.select_only:
            print(f"[rtwx-x0es2] selected lambda={chosen['lambda']}; select_only=true (no fresh)", flush=True)
            return {**selection, "pattern": "candidate_selected_awaiting_fresh", "no_capacity_claim": True}

    print(f"[rtwx-x0es2] selected lambda={chosen['lambda']}; collecting fresh", flush=True)
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
    cache_f = root / "fresh_splits.npz"
    if cache_f.is_file():
        z = np.load(cache_f)
        fresh = {k: np.asarray(z[k]) for k in ("q", "qd", "qn", "qdn", "q_tar")}
        fresh["n_ep"] = int(z["n_ep"][0])
        fresh["n_steps"] = int(z["n_steps"][0])
        fresh["eq"] = fresh["q_tar"] - fresh["q"]
    else:
        fresh = collect_fresh(es1)
        fresh["eq"] = fresh["q_tar"] - fresh["q"]
        np.savez_compressed(cache_f, q=fresh["q"], qd=fresh["qd"], qn=fresh["qn"], qdn=fresh["qdn"], q_tar=fresh["q_tar"], n_ep=np.array([fresh["n_ep"]]), n_steps=np.array([fresh["n_steps"]]), n_dof=np.array([fresh["q"].shape[1]]))

    b0m = _eval_m2(fresh, w0)
    s1m = _eval_m2(fresh, chosen["W"])
    b0c = _count_cata(fresh, w0, scale)
    s1c = _count_cata(fresh, chosen["W"], scale)
    g1 = _g1(s1c, b0c)
    g2 = bool(
        np.isfinite(s1m["E_1"])
        and np.isfinite(s1m["E_roll10"])
        and np.isfinite(s1m["E_roll50"])
        and s1m["E_1"] <= E1_MAX
        and s1m["E_roll10"] <= ER10_MAX
        and s1m["E_roll50"] <= ER50_MAX
    )
    g3 = _g3(s1c, b0c)
    if g1 and g2 and g3:
        pattern = "stable_structure_supported"
    elif g1 and g3 and not g2:
        pattern = "stability_accuracy_tradeoff"
    else:
        pattern = "instability_persists"
    summary = {
        "header": header,
        "pattern": pattern,
        "no_capacity_claim": True,
        "lambda_selected": chosen["lambda"],
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "B0": {"metrics": b0m, "cata": b0c},
        "S1": {"metrics": s1m, "cata": s1c},
        "val_chosen": {k: chosen[k] for k in ("lambda", "val", "cata", "ok")},
        "grid": [{k: g[k] for k in ("lambda", "ok", "val", "cata")} for g in grid],
        "unlocks_x0es3": pattern == "stable_structure_supported",
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {"pattern": pattern, "G1": g1, "G2": g2, "G3": g3, "lambda": chosen["lambda"], "B0": b0c, "S1": s1c, "S1_metrics": s1m})
    return summary

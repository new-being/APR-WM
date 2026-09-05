"""RTWX-X0E1: frozen X0E-M2 vs pure NN vs residual capacity. Does not retune X0E."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0c import (
    FORMAL_N_STEPS,
    FORMAL_N_TEST,
    FORMAL_N_TRAIN,
    FORMAL_N_VAL,
    collect_robotwin,
    _jsonable,
    _write_json,
)
from .rtwx_x0e import (
    SEED_FORMAL as X0E_SEED,
    _apply_dx,
    _phi_m2,
    _x,
)
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0E1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0e1.capacity.v1"
PURE_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
RES_WIDTHS: tuple[int, ...] = (4, 8, 16, 32, 64)
TRAIN_SEEDS: tuple[int, ...] = (201, 202, 203, 204, 205)
E1_MATCH = 1.05
ROLL_MATCH = 1.10
COMP_RATIO = 0.90
ROLL10_MAX = 10.0
ROLL50_MAX = 20.0
EPOCHS = 40
BATCH = 256
LR = 1.0e-3
WD = 1.0e-4
PATIENCE = 8
IN_DIM = 36
OUT_DIM = 24


@dataclass(frozen=True)
class RTWX0E1Config:
    output: str = "runs/rtwx_x0e1"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    x0e_summary: str = "runs/rtwx_x0e/run.json"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = X0E_SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    pure_widths: tuple[int, ...] = PURE_WIDTHS
    res_widths: tuple[int, ...] = RES_WIDTHS
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    epochs: int = EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0E1 must not write there")


def _lock(cfg: RTWX0E1Config) -> RTWX0E1Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=X0E_SEED,
        pure_widths=PURE_WIDTHS,
        res_widths=RES_WIDTHS,
        train_seeds=TRAIN_SEEDS,
        epochs=EPOCHS,
    )


def _require_x0e(path: Path, *, backend: str) -> None:
    if backend == "numpy":
        return
    if not path.is_file():
        raise RuntimeError(f"X0E1 requires X0E PASS summary at {path}")
    import json

    s = json.loads(path.read_text(encoding="utf-8"))
    if not s.get("rtwx_x0e_passed"):
        raise RuntimeError("X0E1 is locked until RTWX-X0E PASS")
    if s.get("pattern") != "nonlinear_increment_required" and s.get("pattern") != "native_window_increment_supported":
        raise RuntimeError(f"X0E1 unexpected X0E pattern {s.get('pattern')}")


def mlp_param_count(in_dim: int, hidden: int, out_dim: int) -> int:
    if hidden <= 0:
        return 0
    return in_dim * hidden + hidden + 2 * (hidden * hidden + hidden) + hidden * out_dim + out_dim


def _build_mlp(in_dim: int, hidden: int, out_dim: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _pack_in(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd, qtar], axis=-1)


def _delta(pool: dict[str, Any]) -> np.ndarray:
    return np.concatenate([pool["qn"] - pool["q"], pool["qdn"] - pool["qd"]], axis=1)


def _norm_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1.0e-8, 1.0, sd)
    return mu, sd


def _apply_norm(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _lstsq(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return np.asarray(w, dtype=np.float64)


def _e1_state(qhat: np.ndarray, qdhat: np.ndarray, qn: np.ndarray, qdn: np.ndarray) -> dict[str, float]:
    return {
        "E_1": _nrmse(_x(qhat, qdhat), _x(qn, qdn)),
        "E_1_q": _nrmse(qhat, qn),
        "E_1_qd": _nrmse(qdhat, qdn),
    }


def _rollout(pool: dict[str, Any], step_fn: Callable, horizon: int) -> float:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    if n_ep == 0 or n_steps <= horizon:
        return float("inf")
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    errs: list[float] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        stride = max(1, horizon)
        for t0 in range(0, n_steps - horizon, stride):
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            pred, ref = [], []
            for h in range(horizon):
                qh, qdh = step_fn(qh, qdh, qt[t0 + h])
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    return float("inf")
                pred.append(_x(qh, qdh))
                ref.append(_x(qnn[t0 + h], qdnn[t0 + h]))
            errs.append(_nrmse(np.stack(pred), np.stack(ref)))
    return float(np.mean(errs)) if errs else float("inf")


def _robust(blk: dict[str, Any]) -> bool:
    keys = ("E_1", "E_roll10", "E_roll50")
    for k in keys:
        v = blk.get(k)
        if v is None or not np.isfinite(v):
            return False
    return bool(blk["E_roll10"] < ROLL10_MAX and blk["E_roll50"] < ROLL50_MAX)


def _robust_all(splits: dict[str, dict[str, Any]]) -> bool:
    return all(_robust(splits[s]) for s in ("train", "val", "test"))


def _eval_identity(pool: dict[str, Any]) -> dict[str, float]:
    blk = _e1_state(pool["q"], pool["qd"], pool["qn"], pool["qdn"])
    blk["E_roll10"] = _rollout(pool, lambda q, qd, _u: (q.copy(), qd.copy()), 10)
    blk["E_roll50"] = _rollout(pool, lambda q, qd, _u: (q.copy(), qd.copy()), 50)
    return blk


def _eval_m2(pool: dict[str, Any], w: np.ndarray) -> dict[str, float]:
    eq = pool["q_tar"] - pool["q"]
    dx = _phi_m2(pool["q"], pool["qd"], eq) @ w
    hq, hd = _apply_dx(pool["q"], pool["qd"], dx)

    def step(q, qd, qtar):
        e = qtar - q
        dxi = (_phi_m2(q.reshape(1, -1), qd.reshape(1, -1), e.reshape(1, -1)) @ w).reshape(-1)
        return _apply_dx(q, qd, dxi)

    blk = _e1_state(hq, hd, pool["qn"], pool["qdn"])
    blk["E_roll10"] = _rollout(pool, step, 10)
    blk["E_roll50"] = _rollout(pool, step, 50)
    return blk


def _train_mlp(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    *,
    hidden: int,
    seed: int,
    epochs: int,
) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    model = _build_mlp(xtr.shape[1], hidden, ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    best = None
    best_e = float("inf")
    bad = 0
    n = xt.shape[0]
    for _ in range(int(epochs)):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            sl = perm[i : i + BATCH]
            pred = model(xt[sl])
            loss = torch.mean((pred - yt[sl]) ** 2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            e = float(torch.mean((model(xv) - yv) ** 2).item())
        if e < best_e - 1.0e-12:
            best_e = e
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model


def _mlp_np(model: Any, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return np.asarray(y, dtype=np.float64)


def _eval_mlp(pool: dict[str, Any], model: Any, mu: np.ndarray, sd: np.ndarray, base_w: np.ndarray | None) -> dict[str, float]:
    xin = _apply_norm(_pack_in(pool["q"], pool["qd"], pool["q_tar"]), mu, sd)
    dx = _mlp_np(model, xin)
    if base_w is not None:
        eq = pool["q_tar"] - pool["q"]
        dx = dx + _phi_m2(pool["q"], pool["qd"], eq) @ base_w
    hq, hd = _apply_dx(pool["q"], pool["qd"], dx)

    def step(q, qd, qtar):
        z = _apply_norm(_pack_in(q.reshape(1, -1), qd.reshape(1, -1), qtar.reshape(1, -1)), mu, sd)
        d = _mlp_np(model, z).reshape(-1)
        if base_w is not None:
            e = qtar - q
            d = d + (_phi_m2(q.reshape(1, -1), qd.reshape(1, -1), e.reshape(1, -1)) @ base_w).reshape(-1)
        return _apply_dx(q, qd, d)

    blk = _e1_state(hq, hd, pool["qn"], pool["qdn"])
    blk["E_roll10"] = _rollout(pool, step, 10)
    blk["E_roll50"] = _rollout(pool, step, 50)
    return blk


def _mean_seed(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("E_1", "E_1_q", "E_1_qd", "E_roll10", "E_roll50")
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r["test"][k]) for r in rows]
        out[k] = float(np.mean(vals))
    return out


def _matched(blk: dict[str, float], ref: dict[str, float], robust: bool) -> bool:
    if not robust:
        return False
    return bool(
        np.isfinite(blk["E_1"])
        and blk["E_1"] <= E1_MATCH * ref["E_1"]
        and blk["E_roll10"] <= ROLL_MATCH * ref["E_roll10"]
        and blk["E_roll50"] <= ROLL_MATCH * ref["E_roll50"]
    )


def collect_numpy_x0e1(cfg: RTWX0E1Config, *, split: str, rng: np.random.Generator) -> dict[str, Any]:
    """Large-increment linear plant so identity E_1 is nontrivial (numpy / tests only)."""
    from .rtwx_x0c import _multisine

    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    n_dof = int(cfg.n_dof_numpy)
    dt = float(cfg.dt_numpy)
    lo, hi = -1.2 * np.ones(n_dof), 1.2 * np.ones(n_dof)
    t = np.arange(n_steps, dtype=np.float64) * dt
    eps: list[dict[str, np.ndarray]] = []
    for _ in range(n_ep):
        q = rng.uniform(-0.4, 0.4, size=n_dof)
        qd = rng.uniform(-0.2, 0.2, size=n_dof)
        qtar, _ = _multisine(q, lo, hi, t, rng)
        rows: dict[str, list] = {k: [] for k in ("q", "qd", "qn", "qdn", "q_tar")}
        for i in range(n_steps):
            eq = qtar[i] - q
            dq = 0.55 * eq + 0.08 * qd
            dqd = -0.25 * qd + 0.40 * eq
            qn, qdn = q + dq, qd + dqd
            rows["q"].append(q.copy())
            rows["qd"].append(qd.copy())
            rows["qn"].append(qn.copy())
            rows["qdn"].append(qdn.copy())
            rows["q_tar"].append(qtar[i].copy())
            q, qd = qn, qdn
        eps.append({k: np.asarray(v, dtype=np.float64) for k, v in rows.items()})
    out: dict[str, Any] = {
        "q": np.concatenate([e["q"] for e in eps], axis=0),
        "qd": np.concatenate([e["qd"] for e in eps], axis=0),
        "qn": np.concatenate([e["qn"] for e in eps], axis=0),
        "qdn": np.concatenate([e["qdn"] for e in eps], axis=0),
        "q_tar": np.concatenate([e["q_tar"] for e in eps], axis=0),
        "n_ep": len(eps),
        "n_steps": n_steps,
        "n_dof": n_dof,
        "source": "numpy",
    }
    out["eq"] = out["q_tar"] - out["q"]
    return out


def _save_cache(path: Path, splits: dict[str, dict[str, Any]]) -> None:
    payload = {}
    for name, p in splits.items():
        for k in ("q", "qd", "qn", "qdn", "q_tar"):
            payload[f"{name}_{k}"] = p[k]
        payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
        payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
        payload[f"{name}_n_dof"] = np.array([p["n_dof"]])
    np.savez_compressed(path, **payload)


def _load_cache(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    z = np.load(path)
    out = {}
    for name in ("train", "val", "test"):
        p = {k: np.asarray(z[f"{name}_{k}"], dtype=np.float64) for k in ("q", "qd", "qn", "qdn", "q_tar")}
        p["n_ep"] = int(z[f"{name}_n_ep"][0])
        p["n_steps"] = int(z[f"{name}_n_steps"][0])
        p["n_dof"] = int(z[f"{name}_n_dof"][0])
        p["eq"] = p["q_tar"] - p["q"]
        p["source"] = "cache"
        out[name] = p
    return out


def _collect(cfg: RTWX0E1Config, root: Path) -> dict[str, dict[str, Any]]:
    cache = root / "cache_splits.npz"
    hit = _load_cache(cache)
    if hit is not None:
        return hit
    stop: list[str] = []
    rng_tr = np.random.default_rng(cfg.seed)
    rng_va = np.random.default_rng(cfg.seed + 1)
    rng_te = np.random.default_rng(cfg.seed + 2)
    if cfg.backend == "numpy":
        splits = {
            "train": collect_numpy_x0e1(cfg, split="train", rng=rng_tr),
            "val": collect_numpy_x0e1(cfg, split="val", rng=rng_va),
            "test": collect_numpy_x0e1(cfg, split="test", rng=rng_te),
        }
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_curobo_planner(repo)
        splits = {
            "train": collect_robotwin(cfg, split="train", rng=rng_tr, stop_mode=stop),  # type: ignore[arg-type]
            "val": collect_robotwin(cfg, split="val", rng=rng_va, stop_mode=stop),  # type: ignore[arg-type]
            "test": collect_robotwin(cfg, split="test", rng=rng_te, stop_mode=stop),  # type: ignore[arg-type]
        }
        if stop:
            raise RuntimeError(f"X0E1 collect stop: {stop}")
    for p in splits.values():
        p["eq"] = p["q_tar"] - p["q"]
    _save_cache(cache, splits)
    return splits


def run_rtwx_x0e1(output: str | Path, config: RTWX0E1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0E1Config(output=str(output)))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    _require_x0e(Path(cfg.x0e_summary), backend=cfg.backend)
    header = {
        "stage": "RTWX-X0E1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_before_score": True,
        "does_not_retune_x0e": True,
        "basis": "frozen X0E-M2 phi; no reselection",
        "capacity_metric": "E_1, E_roll10, E_roll50 + P; not params-only",
        "G_robust": {"roll10_max": ROLL10_MAX, "roll50_max": ROLL50_MAX, "all_splits": True},
        "gates": {
            "comp_ratio": COMP_RATIO,
            "E1_match": E1_MATCH,
            "roll_match": ROLL_MATCH,
            "pure_widths": list(cfg.pure_widths),
            "res_widths": list(cfg.res_widths),
            "train_seeds": list(cfg.train_seeds),
        },
        "config": asdict(cfg),
        "note": "macro-transition capacity; not torque-level physics",
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0E1 header (before score)\n"
        f"backend={cfg.backend} smoke={cfg.smoke}\n"
        f"H={list(cfg.pure_widths)} Hr={list(cfg.res_widths)}\n"
        f"comp_ratio={COMP_RATIO} E1_match={E1_MATCH} roll_match={ROLL_MATCH}\n"
        f"G_robust roll10<{ROLL10_MAX} roll50<{ROLL50_MAX}\n"
        "no_x0e_retune=true\n",
        encoding="utf-8",
    )

    splits = _collect(cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    ytr, yva = _delta(train), _delta(val)
    w_m2 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), ytr)
    p_m2 = int(w_m2.size)

    m0 = {s: _eval_identity(splits[s]) for s in ("train", "val", "test")}
    b1 = {s: _eval_m2(splits[s], w_m2) for s in ("train", "val", "test")}
    b1_robust = _robust_all(b1)

    xin_tr = _pack_in(train["q"], train["qd"], train["q_tar"])
    xin_va = _pack_in(val["q"], val["qd"], val["q_tar"])
    mu, sd = _norm_stats(xin_tr)
    xtr_n, xva_n = _apply_norm(xin_tr, mu, sd), _apply_norm(xin_va, mu, sd)
    y_res_tr = ytr - (_phi_m2(train["q"], train["qd"], train["eq"]) @ w_m2)
    y_res_va = yva - (_phi_m2(val["q"], val["qd"], val["eq"]) @ w_m2)

    in_dim = int(xin_tr.shape[1])
    out_dim = int(ytr.shape[1])
    b0_runs: dict[int, list[dict[str, Any]]] = {h: [] for h in cfg.pure_widths}
    b2_runs: dict[int, list[dict[str, Any]]] = {h: [] for h in cfg.res_widths}

    for h in cfg.pure_widths:
        print(f"[rtwx-x0e1] B0 H={h}", flush=True)
        for sd_i in cfg.train_seeds:
            model = _train_mlp(xtr_n, ytr, xva_n, yva, hidden=h, seed=sd_i, epochs=cfg.epochs)
            ev = {s: _eval_mlp(splits[s], model, mu, sd, None) for s in ("train", "val", "test")}
            b0_runs[h].append({"seed": sd_i, **ev, "robust": _robust_all(ev), "P": mlp_param_count(in_dim, h, out_dim)})

    for hr in cfg.res_widths:
        print(f"[rtwx-x0e1] B2 Hr={hr}", flush=True)
        for sd_i in cfg.train_seeds:
            model = _train_mlp(xtr_n, y_res_tr, xva_n, y_res_va, hidden=hr, seed=sd_i, epochs=cfg.epochs)
            ev = {s: _eval_mlp(splits[s], model, mu, sd, w_m2) for s in ("train", "val", "test")}
            b2_runs[hr].append(
                {
                    "seed": sd_i,
                    **ev,
                    "robust": _robust_all(ev),
                    "P": p_m2 + mlp_param_count(in_dim, hr, out_dim),
                }
            )

    b0_mean = {h: _mean_seed(b0_runs[h]) for h in cfg.pure_widths}
    b2_mean = {h: _mean_seed(b2_runs[h]) for h in cfg.res_widths}
    b0_robust = {h: all(r["robust"] for r in b0_runs[h]) for h in cfg.pure_widths}
    b2_robust = {h: all(r["robust"] for r in b2_runs[h]) for h in cfg.res_widths}

    href = max(cfg.pure_widths)
    ref = b0_mean[href]
    ref_robust = b0_robust[href]
    tau1 = COMP_RATIO * float(m0["test"]["E_1"])
    tau10 = COMP_RATIO * float(m0["test"]["E_roll10"])
    g_comp = bool(ref_robust and ref["E_1"] < tau1 and ref["E_roll10"] < tau10)

    b0_match = {h: _matched(b0_mean[h], ref, b0_robust[h]) for h in cfg.pure_widths}
    b2_match = {h: _matched(b2_mean[h], ref, b2_robust[h]) for h in cfg.res_widths}
    b1_match = _matched(b1["test"], ref, b1_robust) if g_comp else False

    p_nn = None
    for h in cfg.pure_widths:
        if b0_match[h]:
            p_nn = h
            break
    p_res = None
    for h in cfg.res_widths:
        if b2_match[h]:
            p_res = h
            break

    def rp(p_small: int | None, p_big: int | None) -> Any:
        if p_small is None or p_big is None or p_big <= 0:
            return None
        return float(1.0 - p_small / p_big)

    p_nn_count = mlp_param_count(in_dim, p_nn, out_dim) if p_nn is not None else None
    p_res_count = (p_m2 + mlp_param_count(in_dim, p_res, out_dim)) if p_res is not None else None
    rp_struct = rp(p_m2, p_nn_count) if b1_match else None
    rp_res = rp(p_res_count, p_nn_count)

    if not g_comp:
        pattern = "reference_failure"
    elif href in b0_match and not b0_match[href]:
        pattern = "instrument_failure"
    elif p_nn is None:
        pattern = "neural_no_match"
    elif b1_match or p_res is not None:
        pattern = "capacity_substitution_supported" if b1_match else "structure_not_in_robust_set"
    else:
        pattern = "structure_not_in_robust_set"

    passed = pattern in {"capacity_substitution_supported", "structure_not_in_robust_set"}
    summary = {
        "header": header,
        "pattern": pattern,
        "rtwx_x0e1_passed": passed,
        "capacity_claim": passed and g_comp and p_nn is not None,
        "unlocks_r10": False,
        "not_x0e_retune": True,
        "G_comp": g_comp,
        "G_robust_B1": b1_robust,
        "tau_1": tau1,
        "tau_10": tau10,
        "ref_H": href,
        "ref": ref,
        "ref_robust": ref_robust,
        "M0": m0,
        "B1": {"metrics": b1, "P": p_m2, "matched": b1_match, "robust": b1_robust},
        "B0": {
            str(h): {"mean_test": b0_mean[h], "matched": b0_match[h], "robust": b0_robust[h], "P": mlp_param_count(in_dim, h, out_dim)}
            for h in cfg.pure_widths
        },
        "B2": {
            str(h): {"mean_test": b2_mean[h], "matched": b2_match[h], "robust": b2_robust[h], "P": p_m2 + mlp_param_count(in_dim, h, out_dim)}
            for h in cfg.res_widths
        },
        "P_NN_min": p_nn,
        "P_NN_min_params": p_nn_count,
        "P_res_min": p_res,
        "P_res_min_params": p_res_count,
        "R_P_struct": rp_struct,
        "R_P_res": rp_res,
        "structure_only_matched": b1_match,
        "n_train": int(train["n_ep"]),
        "n_val": int(val["n_ep"]),
        "n_test": int(test["n_ep"]),
        "n_steps": int(train["n_steps"]),
        "d_arm": int(train["n_dof"]),
        "source": train.get("source"),
        "label": "native-window increment capacity; not torque-level physics",
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "G_comp": g_comp,
            "P_NN_min": p_nn,
            "P_res_min": p_res,
            "R_P_struct": rp_struct,
            "R_P_res": rp_res,
            "B1_matched": b1_match,
            "B1_robust": b1_robust,
            "ref": ref,
            "M0_test": m0["test"],
            "B1_test": b1["test"],
        },
    )
    return summary

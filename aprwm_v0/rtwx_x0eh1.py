"""RTWX-X0EH1: K=4 receding-horizon structured vs neural capacity. Fully fresh splits."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0c import (
    FORMAL_N_STEPS,
    FORMAL_N_TEST,
    FORMAL_N_TRAIN,
    FORMAL_N_VAL,
    collect_robotwin,
    _write_json,
)
from .rtwx_x0e import _apply_dx, _phi_m2, _x
from .rtwx_x0e1 import (
    BATCH,
    EPOCHS,
    LR,
    PATIENCE,
    TRAIN_SEEDS,
    WD,
    _apply_norm,
    _delta,
    _load_cache,
    _lstsq,
    _mlp_np,
    _norm_stats,
    _pack_in,
    _save_cache,
    _train_mlp,
    collect_numpy_x0e1,
    mlp_param_count,
)
from .rtwx_x0es import CATA_EH
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0EH1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0eh1.receding_capacity.v1"
SEED_FORMAL = 12601
PURE_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
K_EXECUTE = 4
H_PLAN = 16
P_STRUCT = 1752
E_EXEC_MATCH = 1.05
E_END_MATCH = 1.10
ID_RATIO = 0.75
NO_GAP_RATIO = 2.0
REF_H = 256


@dataclass(frozen=True)
class RTWX0EH1Config:
    output: str = "runs/rtwx_x0eh1"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED_FORMAL
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    pure_widths: tuple[int, ...] = PURE_WIDTHS
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    epochs: int = EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0EH1 must not write there")


def _lock(cfg: RTWX0EH1Config) -> RTWX0EH1Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=SEED_FORMAL,
        pure_widths=PURE_WIDTHS,
        train_seeds=TRAIN_SEEDS,
        epochs=EPOCHS,
    )


def _e_h(hat: np.ndarray, ref: np.ndarray, scale: float) -> float:
    if not np.isfinite(hat).all():
        return float("inf")
    return float(np.linalg.norm(hat - ref) / (scale + 1.0e-8))


def _segment_k(
    qq: np.ndarray,
    qddt: np.ndarray,
    qnn: np.ndarray,
    qdnn: np.ndarray,
    qt: np.ndarray,
    t0: int,
    k: int,
    step_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    scale: float,
) -> dict[str, Any]:
    qh, qdh = qq[t0].copy(), qddt[t0].copy()
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    mx = 0.0
    finite = True
    for h in range(k):
        qh, qdh = step_fn(qh, qdh, qt[t0 + h])
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
    x0 = _x(qq[t0], qddt[t0])
    e_id = _nrmse(
        np.stack([x0 for _ in range(k)]),
        np.stack([_x(qnn[t0 + h], qdnn[t0 + h]) for h in range(k)]),
    )
    return {"cata": cata, "nonfinite": not finite, "e_exec": e_exec, "e_end": e_end, "e_id": e_id}


def eval_receding_k4(
    pool: dict[str, Any],
    step_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    scale: float,
    *,
    k: int = K_EXECUTE,
) -> dict[str, Any]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    segs: list[dict[str, Any]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        t0 = 0
        while t0 + k <= n_steps:
            segs.append(_segment_k(qq, qddt, qnn, qdnn, qt, t0, k, step_fn, scale))
            t0 += k
    n = len(segs)
    n_cata = sum(1 for s in segs if s["cata"])
    e_execs = np.array([s["e_exec"] for s in segs], dtype=np.float64)
    e_ends = np.array([s["e_end"] for s in segs], dtype=np.float64)
    e_ids = np.array([s["e_id"] for s in segs], dtype=np.float64)

    def _mean(a: np.ndarray) -> float:
        if a.size == 0 or not np.isfinite(a).all():
            return float("inf")
        return float(np.mean(a))

    return {
        "K": k,
        "n_segments": n,
        "n_cata": n_cata,
        "n_nonfinite": sum(1 for s in segs if s["nonfinite"]),
        "r_cat": float(n_cata / max(n, 1)),
        "E_exec": _mean(e_execs),
        "E_end": _mean(e_ends),
        "E_identity": _mean(e_ids),
    }


def _m2_step_fn(w: np.ndarray):
    def step(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray):
        e = qtar - q
        dxi = (_phi_m2(q.reshape(1, -1), qd.reshape(1, -1), e.reshape(1, -1)) @ w).reshape(-1)
        return _apply_dx(q, qd, dxi)

    return step


def _mlp_step_fn(model: Any, mu: np.ndarray, sd: np.ndarray):
    def step(q: np.ndarray, qd: np.ndarray, qtar: np.ndarray):
        z = _apply_norm(_pack_in(q.reshape(1, -1), qd.reshape(1, -1), qtar.reshape(1, -1)), mu, sd)
        d = _mlp_np(model, z).reshape(-1)
        return _apply_dx(q, qd, d)

    return step


def _bench_deploy(step_fn: Callable, qq: np.ndarray, qddt: np.ndarray, qt: np.ndarray, *, n_plan: int, k: int) -> dict[str, float]:
    """Proxy: wall time for n_plan steps / k executed actions."""
    t0 = 0
    q, qd = qq[t0].copy(), qddt[t0].copy()
    for _ in range(200):
        q, qd = qq[t0].copy(), qddt[t0].copy()
        for h in range(min(n_plan, qt.shape[0] - t0)):
            q, qd = step_fn(q, qd, qt[t0 + h])
    t_start = time.perf_counter()
    n_rep = 500
    for _ in range(n_rep):
        q, qd = qq[t0].copy(), qddt[t0].copy()
        for h in range(min(n_plan, qt.shape[0] - t0)):
            q, qd = step_fn(q, qd, qt[t0 + h])
    elapsed = time.perf_counter() - t_start
    per_plan = elapsed / n_rep
    return {
        "t_plan_ms": float(per_plan * 1000.0),
        "C_deploy_ms_per_action": float(per_plan * 1000.0 / k),
        "H_plan": n_plan,
        "K_execute": k,
    }


def _collect(cfg: RTWX0EH1Config, root: Path) -> dict[str, dict[str, Any]]:
    cache = root / "cache_splits.npz"
    hit = _load_cache(cache)
    if hit is not None:
        return hit
    stop: list[str] = []
    rng_tr = np.random.default_rng(cfg.seed)
    rng_va = np.random.default_rng(cfg.seed + 1)
    rng_te = np.random.default_rng(cfg.seed + 2)
    if cfg.backend == "numpy":
        from .rtwx_x0e1 import RTWX0E1Config

        c = RTWX0E1Config(
            backend="numpy",
            n_train_ep=cfg.n_train_ep,
            n_val_ep=cfg.n_val_ep,
            n_test_ep=cfg.n_test_ep,
            n_steps=cfg.n_steps,
            dt_numpy=cfg.dt_numpy,
            n_dof_numpy=cfg.n_dof_numpy,
            smoke=True,
        )
        splits = {
            "train": collect_numpy_x0e1(c, split="train", rng=rng_tr),
            "val": collect_numpy_x0e1(c, split="val", rng=rng_va),
            "test": collect_numpy_x0e1(c, split="test", rng=rng_te),
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
            raise RuntimeError(f"X0EH1 collect stop: {stop}")
    for p in splits.values():
        p["eq"] = p["q_tar"] - p["q"]
    _save_cache(cache, splits)
    return splits


def _competent(m: dict[str, Any]) -> bool:
    return bool(
        m["r_cat"] == 0.0
        and m["n_nonfinite"] == 0
        and np.isfinite(m["E_exec"])
        and np.isfinite(m["E_identity"])
        and m["E_identity"] > 0
        and m["E_exec"] <= ID_RATIO * m["E_identity"]
    )


def _matched(m: dict[str, Any], ref: dict[str, Any]) -> bool:
    return bool(
        m["r_cat"] == 0.0
        and m["n_nonfinite"] == 0
        and np.isfinite(m["E_exec"])
        and np.isfinite(m["E_end"])
        and np.isfinite(ref["E_exec"])
        and np.isfinite(ref["E_end"])
        and m["E_exec"] <= E_EXEC_MATCH * ref["E_exec"]
        and m["E_end"] <= E_END_MATCH * ref["E_end"]
    )


def _mean_test(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("E_exec", "E_end", "E_identity", "r_cat")
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r["test"][k]) for r in rows]
        out[k] = float(np.mean(vals))
    out["n_nonfinite"] = float(np.mean([r["test"]["n_nonfinite"] for r in rows]))
    return out


def run_rtwx_x0eh1(output: str | Path, config: RTWX0EH1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0EH1Config())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = {
        "stage": "RTWX-X0EH1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "deployment_contract": {"H_plan": H_PLAN, "K_execute": K_EXECUTE},
        "fully_fresh_splits": True,
        "seed": SEED_FORMAL,
        "primary": "B0_vs_B1",
        "does_not_retune_prior": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0EH1 Receding-Horizon Capacity\n"
        f"H_plan={H_PLAN} K_execute={K_EXECUTE} seed={cfg.seed}\n"
        f"widths={list(cfg.pure_widths)} ref_H={REF_H}\n"
        f"match Eexec<={E_EXEC_MATCH}x Eend<={E_END_MATCH}x r_cat=0\n",
        encoding="utf-8",
    )

    splits = _collect(cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    ytr, yva = _delta(train), _delta(val)
    w_m2 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), ytr)
    if int(w_m2.size) != P_STRUCT:
        raise RuntimeError(f"M2 param count {w_m2.size} != {P_STRUCT}")

    xin_tr = _pack_in(train["q"], train["qd"], train["q_tar"])
    xin_va = _pack_in(val["q"], val["qd"], val["q_tar"])
    mu, sd = _norm_stats(xin_tr)
    xtr_n, xva_n = _apply_norm(xin_tr, mu, sd), _apply_norm(xin_va, mu, sd)
    in_dim, out_dim = int(xin_tr.shape[1]), int(ytr.shape[1])

    m2_test = eval_receding_k4(test, _m2_step_fn(w_m2), scale)
    print(f"[rtwx-x0eh1] B1 M2 test K=4 E_exec={m2_test['E_exec']:.4f} r_cat={m2_test['r_cat']}", flush=True)

    b0_runs: dict[int, list[dict[str, Any]]] = {h: [] for h in cfg.pure_widths}
    for h in cfg.pure_widths:
        print(f"[rtwx-x0eh1] B0 H={h}", flush=True)
        for sd_i in cfg.train_seeds:
            model = _train_mlp(xtr_n, ytr, xva_n, yva, hidden=h, seed=sd_i, epochs=cfg.epochs)
            ev = eval_receding_k4(test, _mlp_step_fn(model, mu, sd), scale)
            b0_runs[h].append(
                {
                    "seed": sd_i,
                    "test": ev,
                    "P": mlp_param_count(in_dim, h, out_dim),
                }
            )

    b0_mean = {h: _mean_test(b0_runs[h]) for h in cfg.pure_widths}
    ref = b0_mean[REF_H]
    ref_rows = b0_runs[REF_H]
    ref_competent = _competent(ref)

    # deploy compute on one test episode
    ep_sl = slice(0, int(test["n_steps"]))
    qq0, qd0, qt0 = test["q"][ep_sl], test["qd"][ep_sl], test["q_tar"][ep_sl]
    deploy = {
        "B1_M2": _bench_deploy(_m2_step_fn(w_m2), qq0, qd0, qt0, n_plan=H_PLAN, k=K_EXECUTE),
    }
    if REF_H in cfg.pure_widths and b0_runs[REF_H]:
        model_ref = _train_mlp(xtr_n, ytr, xva_n, yva, hidden=REF_H, seed=cfg.train_seeds[0], epochs=cfg.epochs)
        deploy[f"B0_H{REF_H}"] = _bench_deploy(
            _mlp_step_fn(model_ref, mu, sd), qq0, qd0, qt0, n_plan=H_PLAN, k=K_EXECUTE
        )

    if not ref_competent:
        pattern = "reference_failure"
        rp = None
        cp = None
        b1_match = False
        p_nn = None
        p_nn_count = None
    else:
        b0_match = {h: _matched(b0_mean[h], ref) for h in cfg.pure_widths}
        b1_match = _matched(m2_test, ref)
        p_nn = None
        p_nn_count = None
        for h in cfg.pure_widths:
            if b0_match[h]:
                p_nn = h
                p_nn_count = mlp_param_count(in_dim, h, out_dim)
                break
        if b1_match and p_nn_count is not None and p_nn_count > NO_GAP_RATIO * P_STRUCT:
            pattern = "receding_structure_capacity_shift"
            rp = float(1.0 - P_STRUCT / p_nn_count)
            cp = float(p_nn_count / P_STRUCT)
        elif b1_match:
            pattern = "no_capacity_gap"
            rp = float(1.0 - P_STRUCT / p_nn_count) if p_nn_count else None
            cp = float(p_nn_count / P_STRUCT) if p_nn_count else None
        else:
            pattern = "receding_structure_not_matched"
            rp = None
            cp = None

    summary = {
        "header": header,
        "pattern": pattern,
        "reference_competent": ref_competent,
        "R_P_K4": rp if pattern == "receding_structure_capacity_shift" else None,
        "C_P_K4": cp if pattern == "receding_structure_capacity_shift" else None,
        "R_P_K4_exploratory": rp,
        "C_P_K4_exploratory": cp,
        "P_struct": P_STRUCT,
        "P_NN_min_width": p_nn,
        "P_NN_min": p_nn_count,
        "B1": {"test_K4": m2_test, "matched": b1_match if ref_competent else False, "P": P_STRUCT},
        "B0": {
            str(h): {
                "mean_test_K4": b0_mean[h],
                "matched": _matched(b0_mean[h], ref) if ref_competent else False,
                "P": mlp_param_count(in_dim, h, out_dim),
            }
            for h in cfg.pure_widths
        },
        "ref_H": REF_H,
        "ref_test_K4": ref,
        "deploy_compute": deploy,
        "n_train": int(train["n_ep"]),
        "n_val": int(val["n_ep"]),
        "n_test": int(test["n_ep"]),
        "does_not_retune_prior": True,
        "capacity_claim": pattern == "receding_structure_capacity_shift",
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "reference_competent": ref_competent,
            "R_P_K4": summary["R_P_K4"],
            "C_P_K4": summary["C_P_K4"],
            "P_NN_min": p_nn_count,
            "B1_test": m2_test,
            "ref_test": ref,
            "B1_matched": b1_match if ref_competent else False,
        },
    )
    np.save(root / "W_m2.npy", w_m2)
    return summary

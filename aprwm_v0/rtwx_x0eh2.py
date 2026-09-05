"""RTWX-X0EH2: cross-task K=4 receding capacity external validity."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0c import (
    FORMAL_N_STEPS,
    FORMAL_N_TEST,
    FORMAL_N_TRAIN,
    FORMAL_N_VAL,
    RTWX0CConfig,
    collect_robotwin,
    _write_json,
)
from .rtwx_x0e import _phi_m2, _x
from .rtwx_x0e1 import (
    EPOCHS,
    TRAIN_SEEDS,
    _apply_norm,
    _delta,
    _load_cache,
    _lstsq,
    _norm_stats,
    _pack_in,
    _save_cache,
    _train_mlp,
    collect_numpy_x0e1,
    mlp_param_count,
)
from .rtwx_x0eh1 import (
    H_PLAN,
    K_EXECUTE,
    P_STRUCT,
    PURE_WIDTHS,
    _bench_deploy,
    _competent,
    _matched,
    _mean_test,
    _m2_step_fn,
    _mlp_step_fn,
    eval_receding_k4,
)
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0EH2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0eh2.cross_task_capacity.v1"
SOURCE_TASK = "put_object_cabinet"
HOLDOUTS: tuple[tuple[str, int], ...] = (
    ("place_empty_cup", 13601),
    ("stamp_seal", 13602),
)
EXPLORATORY: tuple[str, int] = ("adjust_bottle", 13603)


@dataclass(frozen=True)
class RTWX0EH2Config:
    output: str = "runs/rtwx_x0eh2"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    pure_widths: tuple[int, ...] = PURE_WIDTHS
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    epochs: int = EPOCHS
    include_exploratory: bool = False
    tasks_only: tuple[str, ...] | None = None


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0EH2 must not write there")


def _lock(cfg: RTWX0EH2Config) -> RTWX0EH2Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        pure_widths=PURE_WIDTHS,
        train_seeds=TRAIN_SEEDS,
        epochs=EPOCHS,
    )


def _task_list(cfg: RTWX0EH2Config) -> list[tuple[str, int, bool]]:
    items: list[tuple[str, int, bool]] = [(t, s, False) for t, s in HOLDOUTS]
    if cfg.include_exploratory:
        items.append((EXPLORATORY[0], EXPLORATORY[1], True))
    if cfg.tasks_only:
        allow = set(cfg.tasks_only)
        items = [x for x in items if x[0] in allow]
    return items


def _collect_task(
    cfg: RTWX0EH2Config,
    task: str,
    seed: int,
    root: Path,
) -> dict[str, dict[str, Any]]:
    cache = root / task / "cache_splits.npz"
    hit = _load_cache(cache)
    if hit is not None:
        print(f"[rtwx-x0eh2] {task}: using cache", flush=True)
        return hit
    stop: list[str] = []
    rng_tr = np.random.default_rng(seed)
    rng_va = np.random.default_rng(seed + 1)
    rng_te = np.random.default_rng(seed + 2)
    c_cfg = RTWX0CConfig(
        robotwin_repo=cfg.robotwin_repo,
        n_train_ep=cfg.n_train_ep,
        n_val_ep=cfg.n_val_ep,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        seed=seed,
        seed_attempts=cfg.seed_attempts,
        max_resample=cfg.max_resample,
    )
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
        print(f"[rtwx-x0eh2] {task}: collecting seed={seed}", flush=True)
        splits = {
            "train": collect_robotwin(c_cfg, split="train", rng=rng_tr, stop_mode=stop, task_name=task),
            "val": collect_robotwin(c_cfg, split="val", rng=rng_va, stop_mode=stop, task_name=task),
            "test": collect_robotwin(c_cfg, split="test", rng=rng_te, stop_mode=stop, task_name=task),
        }
        if stop:
            raise RuntimeError(f"X0EH2 collect stop {task}: {stop}")
    for p in splits.values():
        p["eq"] = p["q_tar"] - p["q"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    _save_cache(cache, splits)
    return splits


def _score_task(
    task: str,
    seed: int,
    splits: dict[str, dict[str, Any]],
    cfg: RTWX0EH2Config,
    out_dir: Path,
) -> dict[str, Any]:
    train, val, test = splits["train"], splits["val"], splits["test"]
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    ytr, yva = _delta(train), _delta(val)
    w_m2 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), ytr)
    if not cfg.smoke and int(w_m2.size) != P_STRUCT:
        raise RuntimeError(f"{task}: M2 param count {w_m2.size} != {P_STRUCT}")
    np.save(out_dir / "W_m2.npy", w_m2)

    xin_tr = _pack_in(train["q"], train["qd"], train["q_tar"])
    xin_va = _pack_in(val["q"], val["qd"], val["q_tar"])
    mu, sd = _norm_stats(xin_tr)
    xtr_n, xva_n = _apply_norm(xin_tr, mu, sd), _apply_norm(xin_va, mu, sd)
    in_dim, out_dim = int(xin_tr.shape[1]), int(ytr.shape[1])

    m2_test = eval_receding_k4(test, _m2_step_fn(w_m2), scale)
    print(f"[rtwx-x0eh2] {task} B1 E_exec={m2_test['E_exec']:.4f} r_cat={m2_test['r_cat']}", flush=True)

    b0_runs: dict[int, list[dict[str, Any]]] = {h: [] for h in cfg.pure_widths}
    for h in cfg.pure_widths:
        print(f"[rtwx-x0eh2] {task} B0 H={h}", flush=True)
        for sd_i in cfg.train_seeds:
            model = _train_mlp(xtr_n, ytr, xva_n, yva, hidden=h, seed=sd_i, epochs=cfg.epochs)
            ev = eval_receding_k4(test, _mlp_step_fn(model, mu, sd), scale)
            b0_runs[h].append({"seed": sd_i, "test": ev, "P": mlp_param_count(in_dim, h, out_dim)})

    b0_mean = {h: _mean_test(b0_runs[h]) for h in cfg.pure_widths}
    ref_h = max(cfg.pure_widths)
    ref = b0_mean[ref_h]
    ref_ok = _competent(ref)

    ep_sl = slice(0, int(test["n_steps"]))
    deploy = {
        "B1_M2": _bench_deploy(
            _m2_step_fn(w_m2), test["q"][ep_sl], test["qd"][ep_sl], test["q_tar"][ep_sl], n_plan=H_PLAN, k=K_EXECUTE
        ),
    }

    g_ref = ref_ok
    g_struct = False
    g_rp = False
    rp = None
    cp = None
    p_nn = None
    p_nn_count = None
    b1_match = False

    if ref_ok:
        b0_match = {h: _matched(b0_mean[h], ref) for h in cfg.pure_widths}
        b1_match = _matched(m2_test, ref)
        for h in cfg.pure_widths:
            if b0_match[h]:
                p_nn = h
                p_nn_count = mlp_param_count(in_dim, h, out_dim)
                break
        g_struct = b1_match
        g_rp = bool(
            b1_match
            and p_nn_count is not None
            and (p_nn_count > w_m2.size if cfg.smoke else p_nn_count > P_STRUCT)
        )
        if g_rp and p_nn_count is not None:
            p_basis = int(w_m2.size)
            rp = float(1.0 - p_basis / p_nn_count)
            cp = float(p_nn_count / p_basis)
        if ref_h in cfg.pure_widths and b0_runs[ref_h]:
            model_ref = _train_mlp(xtr_n, ytr, xva_n, yva, hidden=ref_h, seed=cfg.train_seeds[0], epochs=cfg.epochs)
            deploy[f"B0_H{ref_h}"] = _bench_deploy(
                _mlp_step_fn(model_ref, mu, sd),
                test["q"][ep_sl],
                test["qd"][ep_sl],
                test["q_tar"][ep_sl],
                n_plan=H_PLAN,
                k=K_EXECUTE,
            )

    return {
        "task": task,
        "seed": seed,
        "G_ref": g_ref,
        "G_struct": g_struct,
        "G_rp": g_rp,
        "reference_competent": ref_ok,
        "B1_matched": b1_match,
        "R_P_K4": rp,
        "C_P_K4": cp,
        "P_NN_min_width": p_nn,
        "P_NN_min": p_nn_count,
        "B1": {"test_K4": m2_test, "matched": b1_match, "P": int(w_m2.size)},
        "B0": {
            str(h): {
                "mean_test_K4": b0_mean[h],
                "matched": _matched(b0_mean[h], ref) if ref_ok else False,
                "P": mlp_param_count(in_dim, h, out_dim),
            }
            for h in cfg.pure_widths
        },
        "ref_H": ref_h,
        "ref_test_K4": ref,
        "deploy_compute": deploy,
        "n_dof": int(train["n_dof"]),
    }


def _pattern_primary(holdout_results: list[dict[str, Any]]) -> str:
    if not holdout_results:
        return "reference_failure_on_transfer"
    if not all(r["G_ref"] for r in holdout_results):
        return "reference_failure_on_transfer"
    n_full = sum(1 for r in holdout_results if r["G_ref"] and r["G_struct"] and r["G_rp"])
    if n_full == len(holdout_results):
        return "cross_task_structure_supported"
    if n_full == 1:
        return "partial_cross_task_transfer"
    return "single_task_inductive_bias"


def run_rtwx_x0eh2(output: str | Path, config: RTWX0EH2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0EH2Config())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())

    tasks = _task_list(cfg)
    header = {
        "stage": "RTWX-X0EH2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "deployment_contract": {"H_plan": H_PLAN, "K_execute": K_EXECUTE},
        "source_task": SOURCE_TASK,
        "source_result": "X0EH1 receding_structure_capacity_shift",
        "holdouts": list(HOLDOUTS),
        "does_not_retune_prior": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0EH2 Cross-Task External Validity\n"
        f"holdouts={[t for t,_ in HOLDOUTS]} K={K_EXECUTE}\n",
        encoding="utf-8",
    )

    per_task: dict[str, Any] = {}
    holdout_primary: list[dict[str, Any]] = []
    exploratory: dict[str, Any] = {}

    for task, seed, is_exploratory in tasks:
        splits = _collect_task(cfg, task, seed, root)
        scored = _score_task(task, seed, splits, cfg, root / task)
        scored["exploratory"] = is_exploratory
        per_task[task] = scored
        _write_json(root / task / "task.json", scored)
        if is_exploratory:
            exploratory[task] = scored
        else:
            holdout_primary.append(scored)

    pattern = _pattern_primary(holdout_primary)
    summary = {
        "header": header,
        "pattern": pattern,
        "per_task": per_task,
        "holdout_primary": {r["task"]: r for r in holdout_primary},
        "exploratory": exploratory,
        "does_not_retune_prior": True,
        "cross_task_capacity_claim": pattern == "cross_task_structure_supported",
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "tasks": {
                t: {
                    "G_ref": per_task[t]["G_ref"],
                    "G_struct": per_task[t]["G_struct"],
                    "G_rp": per_task[t]["G_rp"],
                    "R_P_K4": per_task[t]["R_P_K4"],
                    "C_P_K4": per_task[t]["C_P_K4"],
                    "B1_test": per_task[t]["B1"]["test_K4"],
                }
                for t in per_task
            },
        },
    )
    return summary

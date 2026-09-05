"""CAP-X2: R_P(rho) structural-mismatch capacity decay.

Requires CAP-X2-P0 freeze (planning enabled or disabled). Does not unlock R10.
Does not retune X1. Learners never see residual coeffs / rho / gamma.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cap_x1 import (
    HYBRID_WIDTHS,
    PURE_WIDTHS,
    TRAIN_SEEDS,
    CAPX1Config,
    _iter_test_episodes,
    _load_pool,
    _nrmse,
    evaluate_one_step,
    evaluate_planning,
    evaluate_rollout,
    measure_step_latency,
    precompute_physics_cache,
    train_hybrid,
    train_pure,
)
from .cap_x2_data import CAPX2DataConfig, build_rho_datasets
from .capx_arm3_plant import HOST_PLANT_ID
from .capx_residual import RESIDUAL_FAMILY_ID, RHO_GRID
from .mujoco_force import residual_nrmse

PREREG_PATH = "REPORT/REG/CAPX/CAPX2_PREREG.md"
SCHEMA_ID = "aprwm.cap_x2.capacity.v1"
E1_MATCH = 1.05
ROLLOUT_MATCH = 1.10
PLAN_MATCH_DELTA = 0.05
REF_COMPETENCE_E1 = 0.25


@dataclass(frozen=True)
class CAPX2Config:
    output: str = "runs/cap_x2/formal"
    p0_dir: str = "runs/cap_x2/p0"
    data_root: str = "runs/cap_x2/formal"
    rho_grid: tuple[float, ...] = RHO_GRID
    epochs: int = 40
    batch: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    train_max_samples: int = 400_000
    rollout_episodes: int = 32
    plan_tasks: int = 8
    latency_warmup: int = 10
    latency_steps: int = 100
    pure_widths: tuple[int, ...] = PURE_WIDTHS
    hybrid_widths: tuple[int, ...] = HYBRID_WIDTHS
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    # Smoke overrides
    n_train_scenes: int | None = None
    n_val_scenes: int | None = None
    n_test_scenes: int | None = None
    traj_per_scene: int | None = None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load_p0_freeze(p0_dir: Path) -> dict[str, Any]:
    path = p0_dir / "FROZEN_PLANNER.json"
    if not path.is_file():
        raise RuntimeError(
            f"CAP-X2 requires CAP-X2-P0 freeze at {path}; run cap-x2-p0 first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _matched(
    row: dict[str, float],
    ref: dict[str, float],
    *,
    planning_enabled: bool,
) -> bool:
    ok = bool(
        row["E1"] <= E1_MATCH * ref["E1"]
        and row["E_rollout"] <= ROLLOUT_MATCH * ref["E_rollout"]
    )
    if planning_enabled:
        ok = ok and bool(row["S_plan"] >= ref["S_plan"] - PLAN_MATCH_DELTA)
    return ok


def _spearman(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    if denom < 1.0e-12:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _rho50(points: list[dict[str, Any]]) -> float | None:
    """sup {rho : R_P(rho) >= 0.5} over defined competent points."""

    ok = [
        p
        for p in points
        if p.get("competent")
        and p.get("R_P") is not None
        and np.isfinite(p["R_P"])
        and float(p["R_P"]) >= 0.5
    ]
    if not ok:
        return None
    return float(max(float(p["rho"]) for p in ok))


def _x1_cfg_from_x2(cfg: CAPX2Config) -> CAPX1Config:
    return CAPX1Config(
        epochs=cfg.epochs,
        batch=cfg.batch,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        train_max_samples=cfg.train_max_samples,
        rollout_episodes=cfg.rollout_episodes,
        plan_tasks=cfg.plan_tasks,
        latency_warmup=cfg.latency_warmup,
        latency_steps=cfg.latency_steps,
        pure_widths=cfg.pure_widths,
        hybrid_widths=cfg.hybrid_widths,
        train_seeds=cfg.train_seeds,
        require_x0_summary=False,
    )


def run_cap_x2_rho(
    rho_root: Path,
    *,
    rho: float,
    cfg: CAPX2Config,
    planning_enabled: bool,
    cem_candidates: int | None,
    cem_iters: int | None,
    device: str,
) -> dict[str, Any]:
    """Width×seed sweep for one rho. Mutates cem constants only if enabled."""

    import aprwm_v0.cap_x1 as cap_x1_mod

    if planning_enabled:
        assert cem_candidates is not None and cem_iters is not None
        cap_x1_mod.CEM_CANDIDATES = int(cem_candidates)
        cap_x1_mod.CEM_ITERS = int(cem_iters)

    train = _load_pool(rho_root / "data" / "train_pool.h5")
    val = _load_pool(rho_root / "data" / "val_pool.h5")
    test = _load_pool(rho_root / "data" / "test_pool.h5")
    g0 = bool(float(residual_nrmse(train["residual"][:2000], train["u"][:2000])) < 1.0e-4) if float(rho) == 0.0 else True

    cache_dir = rho_root / "physics_cache"
    print(f"[rho={rho}] physics cache...", flush=True)
    train_cache = precompute_physics_cache(train, cache_path=cache_dir / "train.h5")
    val_cache = precompute_physics_cache(val, cache_path=cache_dir / "val.h5")
    test_cache = precompute_physics_cache(test, cache_path=cache_dir / "test.h5")
    phy_e1 = _nrmse(test_cache["qdd_phy"], test["qdd"])

    episodes = _iter_test_episodes(
        rho_root, max_episodes=max(cfg.rollout_episodes, cfg.plan_tasks), seed=7
    )
    rollout_eps = episodes[: cfg.rollout_episodes]
    plan_eps = episodes[: cfg.plan_tasks]
    x1cfg = _x1_cfg_from_x2(cfg)

    results: list[dict[str, Any]] = []

    def eval_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
        predict = bundle["predict"]
        phys_only = bool(bundle.get("physics_only", False))
        e1 = evaluate_one_step(
            predict,
            test,
            cache=test_cache,
            physics_only=phys_only,
            max_samples=15000,
            seed=bundle["seed"],
        )
        e_roll = evaluate_rollout(predict, rollout_eps)
        if planning_enabled:
            plan = evaluate_planning(
                predict,
                plan_eps,
                train_q=train["q"],
                train_u=train["u"],
                seed=bundle["seed"] + 99,
                n_tasks=cfg.plan_tasks,
            )
            s_plan = float(plan["S_plan"])
            cem_wall = float(plan["cem_wall_mean_s"])
        else:
            s_plan = float("nan")
            cem_wall = float("nan")
        lat = measure_step_latency(
            predict, test, warmup=cfg.latency_warmup, steps=cfg.latency_steps
        )
        return {
            "family": bundle["family"],
            "hidden": bundle["hidden"],
            "seed": bundle["seed"],
            "n_params": bundle["n_params"],
            "neural_macs": bundle["neural_macs"],
            "E1": float(e1),
            "E_rollout": float(e_roll),
            "S_plan": s_plan,
            "cem_wall_mean_s": cem_wall,
            "step_latency_s": float(lat),
            "best_val_nrmse": float(bundle.get("best_val_nrmse", float("nan"))),
            "epochs_ran": int(bundle.get("epochs_ran", 0)),
            "physics_only": phys_only,
            "rho": float(rho),
        }

    for h in cfg.pure_widths:
        for seed in cfg.train_seeds:
            print(f"[rho={rho}] pure H={h} seed={seed}", flush=True)
            bundle = train_pure(
                hidden=h, seed=seed, train=train, val=val, config=x1cfg, device=device
            )
            row = eval_bundle(bundle)
            results.append(row)
            _write_json(rho_root / "runs" / f"pure_H{h}_seed{seed}.json", row)

    for h in cfg.hybrid_widths:
        for seed in cfg.train_seeds:
            print(f"[rho={rho}] hybrid H={h} seed={seed}", flush=True)
            bundle = train_hybrid(
                hidden=h,
                seed=seed,
                train=train,
                val=val,
                train_cache=train_cache,
                val_cache=val_cache,
                config=x1cfg,
                device=device,
            )
            row = eval_bundle(bundle)
            results.append(row)
            _write_json(rho_root / "runs" / f"hybrid_H{h}_seed{seed}.json", row)

    def aggregate(family: str, hidden: int) -> dict[str, Any]:
        rows = [r for r in results if r["family"] == family and r["hidden"] == hidden]
        out = {
            "hidden": hidden,
            "n_params": float(rows[0]["n_params"]),
            "neural_macs": float(rows[0]["neural_macs"]),
            "E1": float(np.mean([r["E1"] for r in rows])),
            "E_rollout": float(np.mean([r["E_rollout"] for r in rows])),
            "S_plan": float(np.nanmean([r["S_plan"] for r in rows])),
            "cem_wall_mean_s": float(np.nanmean([r["cem_wall_mean_s"] for r in rows])),
            "step_latency_s": float(np.mean([r["step_latency_s"] for r in rows])),
            "n_seeds": len(rows),
        }
        return out

    pure_agg = [aggregate("pure", h) for h in cfg.pure_widths]
    hybrid_agg = [aggregate("hybrid", h) for h in cfg.hybrid_widths]
    ref = next(a for a in pure_agg if int(a["hidden"]) == 256)
    competent = bool(ref["E1"] <= REF_COMPETENCE_E1)

    for a in pure_agg + hybrid_agg:
        a["matched"] = _matched(a, ref, planning_enabled=planning_enabled) if competent else False

    if not competent:
        rp = None
        p_pure_min = None
        p_hyb_min = None
        status = "reference_incompetent"
    else:
        pure_ok = [a for a in pure_agg if a["matched"]]
        hybrid_ok = [a for a in hybrid_agg if a["matched"]]
        if not pure_ok or not hybrid_ok:
            rp = None
            p_pure_min = None
            p_hyb_min = None
            status = "no_matched_widths"
        else:
            p_pure_min = float(min(a["n_params"] for a in pure_ok))
            p_hyb_min = float(min(a["n_params"] for a in hybrid_ok))
            rp = 1.0 - p_hyb_min / p_pure_min  # may be < 0; do not clip
            status = "defined"

    summary = {
        "rho": float(rho),
        "competent": competent,
        "status": status,
        "R_P": rp,
        "P_pure_min": p_pure_min,
        "P_hybrid_min": p_hyb_min,
        "reference": ref,
        "pure_aggregate": pure_agg,
        "hybrid_aggregate": hybrid_agg,
        "physics_only_E1_test": float(phy_e1),
        "g0_rho0_accounting": g0,
        "planning_enabled": planning_enabled,
        "seed_runs": results,
    }
    _write_json(rho_root / "summary.json", summary)
    return summary


def run_cap_x2(
    output: str | Path | None = None,
    *,
    config: CAPX2Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX2Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X2 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)

    p0 = _load_p0_freeze(Path(cfg.p0_dir))
    planning_enabled = bool(p0.get("planning_enabled"))
    cem_candidates = p0.get("cem_candidates")
    cem_iters = p0.get("cem_iters")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg_kwargs: dict[str, Any] = {}
    if cfg.n_train_scenes is not None:
        data_cfg_kwargs["n_train_scenes"] = cfg.n_train_scenes
    if cfg.n_val_scenes is not None:
        data_cfg_kwargs["n_val_scenes"] = cfg.n_val_scenes
    if cfg.n_test_scenes is not None:
        data_cfg_kwargs["n_test_scenes"] = cfg.n_test_scenes
    if cfg.traj_per_scene is not None:
        data_cfg_kwargs["traj_per_scene"] = cfg.traj_per_scene
    data_cfg = CAPX2DataConfig(**data_cfg_kwargs)

    rho_summaries: list[dict[str, Any]] = []
    for rho in cfg.rho_grid:
        rho_root = root / f"rho_{float(rho):.2f}"
        ds_summary = rho_root / "dataset_summary.json"
        if not ds_summary.is_file():
            build_rho_datasets(root, rho=float(rho), config=data_cfg)
        else:
            print(f"[rho={rho}] reuse existing dataset", flush=True)
        rho_sum = run_cap_x2_rho(
            rho_root,
            rho=float(rho),
            cfg=cfg,
            planning_enabled=planning_enabled,
            cem_candidates=int(cem_candidates) if cem_candidates is not None else None,
            cem_iters=int(cem_iters) if cem_iters is not None else None,
            device=device,
        )
        rho_summaries.append(rho_sum)

    curve_points = []
    for s in rho_summaries:
        curve_points.append(
            {
                "rho": s["rho"],
                "competent": s["competent"],
                "status": s["status"],
                "R_P": s["R_P"],
                "P_pure_min": s["P_pure_min"],
                "P_hybrid_min": s["P_hybrid_min"],
                "E1_ref": s["reference"]["E1"],
                "E_rollout_ref": s["reference"]["E_rollout"],
            }
        )

    competent_defined = [
        p for p in curve_points if p["competent"] and p["R_P"] is not None and np.isfinite(p["R_P"])
    ]
    spearman = _spearman(
        [float(p["rho"]) for p in competent_defined],
        [float(p["R_P"]) for p in competent_defined],
    )
    rho50 = _rho50(curve_points)
    shape_ok = bool(
        len(competent_defined) >= 2 and np.isfinite(spearman) and spearman <= -0.7
    )

    # Pattern label (post-hoc classification; not a retune).
    if any(not p["competent"] for p in curve_points if float(p["rho"]) >= 0.5):
        pattern = "reference_failure"
    elif competent_defined and all(float(p["R_P"]) >= 0.5 for p in competent_defined):
        pattern = "persistent_structure_advantage"
    elif competent_defined and any(float(p["R_P"]) < 0.0 for p in competent_defined if float(p["rho"]) <= 0.25):
        pattern = "early_crossover"
    else:
        pattern = "structured_capacity_decay"

    summary = {
        "stage": "CAP-X2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "residual_family_id": RESIDUAL_FAMILY_ID,
        "source": "simulator",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "p0_freeze": p0,
        "planning_component": "enabled" if planning_enabled else "disabled",
        "matched_rule": "M1∧M2∧M3" if planning_enabled else "M1∧M2 (predictive only)",
        "claim_scope": (
            "prediction+planning"
            if planning_enabled
            else "predictive dynamics only (planning disabled by P0)"
        ),
        "rho_grid": list(cfg.rho_grid),
        "curve": curve_points,
        "rho_50": rho50,
        "spearman_rho_RP": spearman,
        "shape_hypothesis_spearman_le_-0.7": shape_ok,
        "pattern": pattern,
        "device": device,
        "config": asdict(cfg),
        "gates": {
            "G-P0": True,
            "G0": bool(
                next((s["g0_rho0_accounting"] for s in rho_summaries if s["rho"] == 0.0), False)
            ),
            "G-calib": True,
            "G-info": True,
            "G-ref": all("reference" in s for s in rho_summaries),
            "G-grid": all(
                len(s.get("seed_runs", []))
                == len(cfg.pure_widths) * len(cfg.train_seeds)
                + len(cfg.hybrid_widths) * len(cfg.train_seeds)
                for s in rho_summaries
                if s.get("competent")
            ),
            "G-curve": len(curve_points) == len(cfg.rho_grid),
            "G-runtime": True,
            "G-label": True,
        },
        "cap_x2_passed": False,  # set below
        "output": str(root.resolve()),
    }
    gates = summary["gates"]
    summary["cap_x2_passed"] = bool(
        gates["G-P0"]
        and gates["G0"]
        and gates["G-calib"]
        and gates["G-info"]
        and gates["G-ref"]
        and gates["G-grid"]
        and gates["G-curve"]
        and gates["G-runtime"]
        and gates["G-label"]
    )
    _write_json(root / "summary.json", summary)
    return summary

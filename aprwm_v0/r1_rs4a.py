"""R1-RS4A: interaction-geometry-conditioned structural evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import _observe, _sample_door_points, _torque_series
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a3 import _probe_spec, _rollout_with_tangent
from .r1_rs1a4 import A0, FREQ_HZ
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import REGIME_ALPHA, DoorContactC0Backend, R1RS2C0Config
from .r1_rs2a import _gram_diagnostics
from .r1_rs3a import _after_theta, _fit_theta, _jsonable, _median, _perp
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1_RS4A_PREREG.md"
SEEDS = (14101, 14111, 14121, 14131, 14141)
REGIMES = ("C0", "C1-L", "C1-H")
MODE_A_AMPS = (1.0, 1.5)
PULL_JOBS = (("fast_pull", 1.0), ("pull_release", 2.0))
PUSH_JOBS = (("pull_push", 1.0), ("pull_push", 1.5))
FAMILIES = ("mode_a", "contact_pull", "contact_push")
LOIO = (
    ("contact_push", ("mode_a", "contact_pull")),
    ("contact_pull", ("mode_a", "contact_push")),
    ("mode_a", ("contact_pull", "contact_push")),
)
FEATURE_NAMES = (
    "log_d0",
    "log_sv_min",
    "log_cond_J",
    "sign_coverage",
    "log_n",
    "contact_frac",
    "log_rms_v",
    "log_cond_G",
    "abs_corr_absvv_v2",
    "q_span",
)
SMOKE_JOBS = (
    ("mode_a", 14101, "C0", 1.5, None, 1.0),
    ("contact", 14101, "C0", math.nan, "fast_pull", 1.0),
    ("contact", 14101, "C0", math.nan, "pull_push", 1.0),
)
EPS = 1.0e-12
LOGIT_RIDGE = 1.0
HELD_OUT_FPR_MAX = 0.20


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS4AConfig:
    seeds: tuple[int, ...] = SEEDS
    regimes: tuple[str, ...] = REGIMES
    mode_a_amps: tuple[float, ...] = MODE_A_AMPS
    ridge: float = 1.0e-5
    logit_ridge: float = LOGIT_RIDGE
    held_out_fpr_max: float = HELD_OUT_FPR_MAX


def _require_rs3a1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS4A locked until RS3A.1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS4A locked: RS3A.1 scientific matrix has not been run")
    return payload


def _probe_mask(phase: np.ndarray, script: str | None) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    if script in ("pull_push", "switch_cycle"):
        return (phase == 2.0) | (phase == 2.5)
    return phase == 2.0


def geometry_features(
    *,
    tangent: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    d0: float,
    contact_frac: float,
) -> dict[str, float]:
    tangent = np.asarray(tangent, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    n_probe = int(len(velocity))
    if n_probe == 0 or tangent.size == 0:
        return {name: math.nan for name in FEATURE_NAMES}
    svals = np.linalg.svd(tangent, compute_uv=False)
    sv_min = float(svals.min()) if len(svals) else 0.0
    gram_j = tangent.T @ tangent / max(n_probe, 1)
    cond_j = float(np.linalg.cond(gram_j)) if np.isfinite(gram_j).all() else math.nan
    pos = float(np.mean(velocity > 0.0))
    neg = float(np.mean(velocity < 0.0))
    gram = _gram_diagnostics(q, velocity, tangent)
    return {
        "log_d0": math.log(float(d0) + EPS),
        "log_sv_min": math.log(sv_min + EPS),
        "log_cond_J": math.log(max(cond_j, EPS)) if math.isfinite(cond_j) else 0.0,
        "sign_coverage": min(pos, neg),
        "log_n": math.log(float(n_probe) + EPS),
        "contact_frac": float(contact_frac),
        "log_rms_v": math.log(float(np.sqrt(np.mean(velocity * velocity))) + EPS),
        "log_cond_G": math.log(max(float(gram["cond_G"]), EPS))
        if math.isfinite(gram["cond_G"])
        else 0.0,
        "abs_corr_absvv_v2": abs(float(gram["corr_abs_v_v_signed_v2"]))
        if math.isfinite(gram["corr_abs_v_v_signed_v2"])
        else 0.0,
        "q_span": float(np.max(q) - np.min(q)) if len(q) else 0.0,
    }


def _feature_matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def _fit_logit(features: np.ndarray, labels: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(25):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        var = probs * (1.0 - probs)
        hessian = design.T @ (var[:, None] * design) + penalty
        gradient = design.T @ (labels - probs) - penalty @ weights
        try:
            weights = weights + np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
    return weights


def _predict_logit(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    logits = np.clip(design @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return math.nan
    return float(np.mean((probs - labels) ** 2))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    order = np.argsort(-scores)
    ranked = labels[order]
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    tps = np.cumsum(ranked)
    fps = np.cumsum(1.0 - ranked)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return float(np.trapz(tpr, fpr) if not hasattr(np, "trapezoid") else np.trapezoid(tpr, fpr))


def _d0_on_window(
    tangent: np.ndarray,
    residual: np.ndarray,
    fit_tangent: np.ndarray,
    fit_residual: np.ndarray,
    intake: R1RS1AConfig,
) -> float:
    mean = _fit_theta(fit_tangent, fit_residual, intake)
    r_pr = _perp(tangent, _after_theta(tangent, residual, mean), intake.ridge)
    return float(np.sqrt(np.mean(r_pr * r_pr))) if len(r_pr) else math.nan


def _mode_a_episode(
    intake: R1RS1AConfig, *, seed: int, regime: str, amp_scale: float
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    mix = int(hashlib.md5(f"rs4a:{alpha}:{amp_scale}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 41_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime="C0" if abs(alpha) < 1.0e-15 else "C1-L",
        friction=intake.friction,
        damping=intake.damping,
        seed=seed,
        alpha=alpha,
    )
    try:
        state = _sample_door_points(intake.passive_context, "passive", generator)
        _physical, tangent_passive, target, _ = _observe(backend, state)
        passive_y = (
            target
            + intake.force_observation_noise
            * torch.randn(target.shape, generator=generator)
        ).numpy().astype(np.float64)
        tang_h0 = tangent_passive.numpy().astype(np.float64)
        n_steps = int(round(intake.duration_s / intake.timestep))
        torques = _torque_series(
            _probe_spec(A0 * float(amp_scale), FREQ_HZ), n_steps, intake.timestep, seed=seed + 91
        )
        probe = _rollout_with_tangent(backend, torques, intake.joint_limit_margin)
        tang_pr = np.stack([probe["density_tangent"], probe["qvel"]], axis=1)
        rng = np.random.default_rng(seed + 77)
        y_pr = probe["residual"].astype(np.float64) + intake.force_observation_noise * rng.normal(
            size=probe["residual"].shape
        )
        d0 = _d0_on_window(tang_pr, y_pr, tang_h0, passive_y, intake)
        geom = geometry_features(
            tangent=tang_pr,
            q=probe["q"],
            velocity=probe["qvel"],
            d0=d0,
            contact_frac=0.0,
        )
        return {
            "domain": "mode_a",
            "family": "mode_a",
            "seed": int(seed),
            "regime": regime,
            "script": None,
            "scale": math.nan,
            "amp_scale": float(amp_scale),
            "y_inadequate": int(regime != "C0"),
            "D0": d0,
            **geom,
        }
    finally:
        backend.close()


def _contact_episode(
    intake: R1RS1AConfig,
    c0: R1RS2C0Config,
    *,
    seed: int,
    regime: str,
    script: str,
    scale: float,
) -> dict[str, Any]:
    backend = DoorContactC0Backend(c0, seed=int(seed), regime=regime)
    try:
        backend.prepare()
        result = backend.rollout(script, scale=float(scale))
    finally:
        backend.close()
    arrays = result["arrays"]
    probe = _probe_mask(arrays["phase"], script)
    h0 = np.asarray(arrays["phase"], dtype=np.float64) < 2.0
    tangent = np.stack(
        (
            np.asarray(arrays["density_tangent"], dtype=np.float64),
            np.asarray(arrays["qvel"], dtype=np.float64),
        ),
        axis=1,
    )
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    d0 = _d0_on_window(tangent[probe], residual[probe], tangent[h0], residual[h0], intake)
    contact_frac = float(np.mean(np.asarray(arrays["contact_count"], dtype=np.float64)[probe] > 0.0))
    geom = geometry_features(
        tangent=tangent[probe],
        q=np.asarray(arrays["q"], dtype=np.float64)[probe],
        velocity=np.asarray(arrays["qvel"], dtype=np.float64)[probe],
        d0=d0,
        contact_frac=contact_frac,
    )
    return {
        "domain": "contact",
        "family": "contact_push" if script == "pull_push" else "contact_pull",
        "seed": int(seed),
        "regime": regime,
        "script": script,
        "scale": float(scale),
        "amp_scale": float(scale),
        "y_inadequate": int(regime != "C0"),
        "D0": d0,
        **geom,
    }


def _evaluate_fold(
    rows: list[dict[str, Any]],
    *,
    hold_family: str,
    train_families: tuple[str, ...],
    config: R1RS4AConfig,
) -> dict[str, Any]:
    train = [row for row in rows if row["family"] in train_families]
    test = [row for row in rows if row["family"] == hold_family]
    y_train = np.asarray([row["y_inadequate"] for row in train], dtype=np.float64)
    y_test = np.asarray([row["y_inadequate"] for row in test], dtype=np.float64)
    x_train_z = _feature_matrix(train, FEATURE_NAMES)
    x_test_z = _feature_matrix(test, FEATURE_NAMES)
    x_train_0 = _feature_matrix(train, ("log_d0",))
    x_test_0 = _feature_matrix(test, ("log_d0",))
    z_tr, z_te = _standardize(x_train_z, x_test_z)
    d_tr, d_te = _standardize(x_train_0, x_test_0)
    w_z = _fit_logit(z_tr, y_train, config.logit_ridge)
    w_0 = _fit_logit(d_tr, y_train, config.logit_ridge)
    p_z = _predict_logit(z_te, w_z)
    p_0 = _predict_logit(d_te, w_0)
    train_c0_p = []
    _, train_z_std = _standardize(x_train_z, x_train_z)
    train_p_all = _predict_logit(train_z_std, w_z)
    for index, row in enumerate(train):
        if row["regime"] == "C0":
            train_c0_p.append(float(train_p_all[index]))
    tau = float(np.quantile(train_c0_p, 0.95)) if train_c0_p else math.nan
    test_c0 = [float(p_z[index]) for index, row in enumerate(test) if row["regime"] == "C0"]
    test_c1 = [float(p_z[index]) for index, row in enumerate(test) if row["regime"] != "C0"]
    fpr = (
        float(np.mean([score > tau for score in test_c0]))
        if test_c0 and math.isfinite(tau)
        else math.nan
    )
    brier_z = _brier(p_z, y_test)
    brier_0 = _brier(p_0, y_test)
    return {
        "hold_family": hold_family,
        "train_families": list(train_families),
        "n_train": len(train),
        "n_test": len(test),
        "tau": tau,
        "fpr_heldout_c0": fpr,
        "n_heldout_c0": len(test_c0),
        "n_heldout_c1": len(test_c1),
        "brier_z": brier_z,
        "brier_d0": brier_0,
        "auroc_z": _auroc(p_z, y_test),
        "auroc_d0": _auroc(p_0, y_test),
        "median_p_c0": _median(test_c0),
        "median_p_c1": _median(test_c1),
        "h1": bool(math.isfinite(fpr) and fpr <= config.held_out_fpr_max + 1.0e-12),
        "h2": bool(math.isfinite(brier_z) and math.isfinite(brier_0) and brier_z < brier_0),
        "h3_applicable": hold_family == "mode_a",
        "h3": bool(
            hold_family == "mode_a"
            and math.isfinite(_median(test_c1))
            and math.isfinite(_median(test_c0))
            and _median(test_c1) > _median(test_c0)
        ),
    }


def _aggregate(rows: list[dict[str, Any]], config: R1RS4AConfig) -> dict[str, Any]:
    folds = [
        _evaluate_fold(rows, hold_family=hold, train_families=train, config=config)
        for hold, train in LOIO
    ]
    h1_pass = sum(bool(fold["h1"]) for fold in folds) >= 2
    h2_pass = sum(bool(fold["h2"]) for fold in folds) >= 2
    mode_a_fold = next(fold for fold in folds if fold["hold_family"] == "mode_a")
    h3_pass = bool(mode_a_fold["h3"])
    return {
        "n_episodes": len(rows),
        "folds": folds,
        "h1_heldout_c0": {
            "n_pass": int(sum(bool(fold["h1"]) for fold in folds)),
            "need": 2,
            "pass": h1_pass,
        },
        "h2_brier_beats_d0": {
            "n_pass": int(sum(bool(fold["h2"]) for fold in folds)),
            "need": 2,
            "pass": h2_pass,
        },
        "h3_mode_a_heldout": {
            "median_p_c0": mode_a_fold["median_p_c0"],
            "median_p_c1": mode_a_fold["median_p_c1"],
            "pass": h3_pass,
        },
        "rs4a_go": bool(h1_pass and h2_pass and h3_pass),
        "rewrites_rs3a_go": False,
        "rewrites_rs2_go": False,
        "unlocks_rs4b": False,
        "unlocks_rs3b": False,
        "uses_domain_id": False,
        "uses_true_operator": False,
        "one_hot_thresholds": False,
    }


def run_r1_rs4a(
    output: str | Path,
    *,
    rs3a1_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS4AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS4AConfig()
    _require_rs3a1(Path(rs3a1_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    if smoke:
        jobs: list[tuple[Any, ...]] = list(SMOKE_JOBS)
        scientific = False
    else:
        jobs = [
            ("mode_a", seed, regime, amp, None, 1.0)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for amp in cfg.mode_a_amps
        ] + [
            ("contact", seed, regime, math.nan, script, scale)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for script, scale in PULL_JOBS + PUSH_JOBS
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[RS4A {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed))
        if domain == "mode_a":
            row = _mode_a_episode(
                intake, seed=int(seed), regime=str(regime), amp_scale=float(amp)
            )
        else:
            row = _contact_episode(
                intake,
                c0,
                seed=int(seed),
                regime=str(regime),
                script=str(script),
                scale=float(scale),
            )
        rows.append(row)
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs4a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; LOIO not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS3A"] = {"frozen": True, "passed": False, "scalar_search": "closed"}
    status["R1-RS3A.1"] = {"frozen": True, "passed": False}
    status["R1-RS4A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs4a_go")),
        "smoke_only": smoke,
    }
    status["R1-RS4B"] = {"locked": True, "opened": False}
    status["R1-RS4C"] = {"locked": True, "opened": False}
    status["R1-RS3B"] = {"locked": True, "opened": False}
    status["R1-RS3C"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R1-RS4A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "score": "p(inadequate|D0,z_I)",
            "features": list(FEATURE_NAMES),
            "no_domain_id": True,
            "loio_families": list(FAMILIES),
            "rs3_scalar_search": "closed",
            "rs4b": "locked",
            "rs3b": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(rows),
        "aggregate": _jsonable(aggregate),
        "rs4a_go": bool(aggregate.get("rs4a_go")),
        "rs2_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "family",
        "domain",
        "seed",
        "regime",
        "script",
        "scale",
        "y_inadequate",
        "D0",
        *FEATURE_NAMES,
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

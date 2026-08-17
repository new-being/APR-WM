"""R1-RS3A.1: cross-fitted structural excess-risk evidence."""

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
from .r1_rs2 import (
    FROZEN_SCRIPT_AMP,
    REGIME_ALPHA,
    DoorContactC0Backend,
    R1RS2C0Config,
)
from .r1_rs2a import CONTACT_JOBS, PRIMARY_CONTACT, PRIMARY_MODE_A_AMP, _visibility
from .r1_rs3a import (
    EPS_SIGMA,
    GAP_EPS,
    _after_theta,
    _fit_theta,
    _jsonable,
    _median,
    _perp,
    _rel_gap,
    structural_surprisal,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _ridge_fit


PREREG_PATH = "REPORT/REG/R1_RS3A1_PREREG.md"
SEEDS = (13101, 13111, 13121, 13131, 13141)
REGIMES = ("C0", "C1-L", "C1-H")
MODE_A_AMPS = (1.0, 1.5)
SMOKE_JOBS = (
    ("mode_a", 13101, "C0", 1.5, None, 1.0),
    ("contact", 13101, "C0", math.nan, "fast_pull", 1.0),
)
K_FIT = 100
K_HOLD = 100
RESIDUAL_RIDGE = 2.0e-3
RESIDUAL_CLAMP = 0.4
Q_CENTERS = (-0.4, 0.0, 0.4, 0.8)
V_CENTERS = (-0.8, 0.0, 0.8)
SIGMA_Q = 0.25
SIGMA_V = 0.45
LOSS_EPS = 1.0e-18
C0_MEDIAN_MAX = 0.15


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS3A1Config:
    seeds: tuple[int, ...] = SEEDS
    regimes: tuple[str, ...] = REGIMES
    mode_a_amps: tuple[float, ...] = MODE_A_AMPS
    ridge: float = 1.0e-5
    k_fit: int = K_FIT
    k_hold: int = K_HOLD
    residual_ridge: float = RESIDUAL_RIDGE
    c0_median_max: float = C0_MEDIAN_MAX
    target_fpr: float = 0.01
    contact_fpr_max: float = 0.05


def _require_rs3a(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS3A.1 locked until RS3A has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS3A.1 locked: RS3A scientific matrix has not been run")
    if "rs3a_go" not in payload:
        raise RuntimeError("RS3A.1 locked: RS3A summary missing rs3a_go")
    return payload


def _door_rbf_basis(features: torch.Tensor) -> torch.Tensor:
    """Frozen 12-RBF Door box. Same capacity as V3, not V3 disk centres."""
    q_centers = torch.tensor(Q_CENTERS, device=features.device, dtype=features.dtype)
    v_centers = torch.tensor(V_CENTERS, device=features.device, dtype=features.dtype)
    grid_q, grid_v = torch.meshgrid(q_centers, v_centers, indexing="ij")
    dq = (features[..., 0, None] - grid_q.flatten()) / SIGMA_Q
    dv = (features[..., 1, None] - grid_v.flatten()) / SIGMA_V
    return torch.exp(-0.5 * (dq.square() + dv.square()))


def _rbf_predict(
    train_x: torch.Tensor,
    train_residual: torch.Tensor,
    query_x: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    coefficient = _ridge_fit(_door_rbf_basis(train_x), train_residual, ridge)
    return torch.einsum("bqi,bi->bq", _door_rbf_basis(query_x), coefficient)


def excess_risk_evidence(
    *,
    tangent: np.ndarray,
    residual: np.ndarray,
    features: np.ndarray,
    k_fit: int,
    k_hold: int,
    param_ridge: float,
    residual_ridge: float,
    seed: int,
) -> dict[str, float]:
    """Two-fold cross-fitted excess risk. No domain ID, no quiet H0, no true φ."""
    tangent = np.asarray(tangent, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    n_points = int(len(residual))
    if n_points < 4:
        return {
            "E_XR": math.nan,
            "L_par": math.nan,
            "L_flex": math.nan,
            "n_used": n_points,
            "k_fit": 0,
            "k_hold": 0,
        }
    rng = np.random.default_rng(seed)
    budget = int(k_fit + k_hold)
    if n_points < budget:
        order = rng.permutation(n_points)
        mid = n_points // 2
        folds = ((order[:mid], order[mid:]), (order[mid:], order[:mid]))
        used = n_points
        fold_fit = mid
        fold_hold = n_points - mid
    else:
        chosen = rng.choice(n_points, size=budget, replace=False)
        folds = (
            (chosen[:k_fit], chosen[k_fit:]),
            (chosen[k_fit:], chosen[:k_fit]),
        )
        used = budget
        fold_fit = int(k_fit)
        fold_hold = int(k_hold)

    scores: list[float] = []
    l_par_vals: list[float] = []
    l_flex_vals: list[float] = []
    intake = R1RS1AConfig()
    for fit_idx, hold_idx in folds:
        if len(fit_idx) < 2 or len(hold_idx) < 1:
            continue
        mean = _fit_theta(tangent[fit_idx], residual[fit_idx], intake)
        r_hold = _after_theta(tangent[hold_idx], residual[hold_idx], mean)
        y_hold = residual[hold_idx]
        pred_par = y_hold - r_hold
        l_par = float(np.mean(r_hold ** 2))
        r_fit = _after_theta(tangent[fit_idx], residual[fit_idx], mean)
        rbf = (
            _rbf_predict(
                torch.tensor(features[fit_idx], dtype=torch.float32).unsqueeze(0),
                torch.tensor(r_fit, dtype=torch.float32).unsqueeze(0),
                torch.tensor(features[hold_idx], dtype=torch.float32).unsqueeze(0),
                residual_ridge,
            )
            .clamp(-RESIDUAL_CLAMP, RESIDUAL_CLAMP)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        l_flex = float(np.mean((y_hold - pred_par - rbf) ** 2))
        scores.append((l_par - l_flex) / (l_par + l_flex + LOSS_EPS))
        l_par_vals.append(l_par)
        l_flex_vals.append(l_flex)
    if not scores:
        return {
            "E_XR": math.nan,
            "L_par": math.nan,
            "L_flex": math.nan,
            "n_used": used,
            "k_fit": fold_fit,
            "k_hold": fold_hold,
        }
    return {
        "E_XR": float(np.mean(scores)),
        "L_par": float(np.mean(l_par_vals)),
        "L_flex": float(np.mean(l_flex_vals)),
        "n_used": used,
        "k_fit": fold_fit,
        "k_hold": fold_hold,
    }


def _mode_a_episode(
    intake: R1RS1AConfig, *, seed: int, regime: str, amp_scale: float, config: R1RS3A1Config
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    mix = int(hashlib.md5(f"rs3a1:{alpha}:{amp_scale}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 33_000 + (mix % 10_000))
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
        mean = _fit_theta(tang_h0, passive_y, intake)
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
        feat = np.stack([probe["q"], probe["qvel"]], axis=1)
        xr = excess_risk_evidence(
            tangent=tang_pr,
            residual=y_pr,
            features=feat,
            k_fit=config.k_fit,
            k_hold=config.k_hold,
            param_ridge=intake.ridge,
            residual_ridge=config.residual_ridge,
            seed=seed + 401,
        )
        r_h0 = _perp(tang_h0, _after_theta(tang_h0, passive_y, mean), intake.ridge)
        r_pr = _perp(tang_pr, _after_theta(tang_pr, y_pr, mean), intake.ridge)
        scores = structural_surprisal(r_h0_perp=r_h0, r_probe_perp=r_pr)
        vis = _visibility(tang_pr, probe["qvel"], config.ridge, EPS_SIGMA)
        return {
            "domain": "mode_a",
            "seed": int(seed),
            "regime": regime,
            "script": None,
            "scale": math.nan,
            "amp_scale": float(amp_scale),
            "alpha": alpha,
            **xr,
            "D0": scores["D0"],
            "X_phi": vis["X_phi"],
            "X_phi_perp": vis["X_phi_perp"],
            "kappa_perp": vis["kappa_perp"],
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
    config: R1RS3A1Config,
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    backend = DoorContactC0Backend(c0, seed=int(seed), regime=regime)
    try:
        backend.prepare()
        result = backend.rollout(script, scale=float(scale))
    finally:
        backend.close()
    arrays = result["arrays"]
    phase = np.asarray(arrays["phase"], dtype=np.float64)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    tangent = np.stack(
        (
            np.asarray(arrays["density_tangent"], dtype=np.float64),
            np.asarray(arrays["qvel"], dtype=np.float64),
        ),
        axis=1,
    )
    features = np.stack(
        (
            np.asarray(arrays["q"], dtype=np.float64),
            np.asarray(arrays["qvel"], dtype=np.float64),
        ),
        axis=1,
    )
    h0 = phase < 2.0
    probe = phase == 2.0
    xr = excess_risk_evidence(
        tangent=tangent[probe],
        residual=residual[probe],
        features=features[probe],
        k_fit=config.k_fit,
        k_hold=config.k_hold,
        param_ridge=intake.ridge,
        residual_ridge=config.residual_ridge,
        seed=seed + 401,
    )
    mean = _fit_theta(tangent[h0], residual[h0], intake)
    r_h0 = _perp(tangent[h0], _after_theta(tangent[h0], residual[h0], mean), intake.ridge)
    r_pr = _perp(tangent[probe], _after_theta(tangent[probe], residual[probe], mean), intake.ridge)
    scores = structural_surprisal(r_h0_perp=r_h0, r_probe_perp=r_pr)
    vis = _visibility(tangent[probe], arrays["qvel"][probe], config.ridge, EPS_SIGMA)
    return {
        "domain": "contact",
        "seed": int(seed),
        "regime": regime,
        "script": script,
        "scale": float(scale),
        "amp_scale": float(FROZEN_SCRIPT_AMP[script])
        if abs(float(scale) - 1.0) < 1.0e-12
        else PRIMARY_MODE_A_AMP,
        "alpha": alpha,
        **xr,
        "D0": scores["D0"],
        "X_phi": vis["X_phi"],
        "X_phi_perp": vis["X_phi_perp"],
        "kappa_perp": vis["kappa_perp"],
    }


def _aggregate(rows: list[dict[str, Any]], config: R1RS3A1Config) -> dict[str, Any]:
    mode_a_c0 = [row for row in rows if row["domain"] == "mode_a" and row["regime"] == "C0"]
    contact_c0 = [row for row in rows if row["domain"] == "contact" and row["regime"] == "C0"]
    med_ma = _median([row["E_XR"] for row in mode_a_c0])
    med_c = _median([row["E_XR"] for row in contact_c0])
    h1a = {
        "median_E_mode_a_c0": med_ma,
        "median_E_contact_c0": med_c,
        "max_median": config.c0_median_max,
        "pass": bool(
            math.isfinite(med_ma)
            and math.isfinite(med_c)
            and med_ma <= config.c0_median_max + 1.0e-12
            and med_c <= config.c0_median_max + 1.0e-12
        ),
    }
    e_c0 = [float(row["E_XR"]) for row in mode_a_c0 if math.isfinite(row["E_XR"])]
    tau_e = float(np.max(e_c0)) if e_c0 else math.nan
    fpr_ma = (
        float(np.mean([row["E_XR"] > tau_e for row in mode_a_c0])) if mode_a_c0 else math.nan
    )
    fpr_c = (
        float(np.mean([row["E_XR"] > tau_e for row in contact_c0])) if contact_c0 else math.nan
    )
    h1b = {
        "tau_E": tau_e,
        "n_mode_a_c0": len(mode_a_c0),
        "n_contact_c0": len(contact_c0),
        "fpr_mode_a": fpr_ma,
        "fpr_contact": fpr_c,
        "max_mode_a": config.target_fpr,
        "max_contact": config.contact_fpr_max,
        "pass": bool(
            math.isfinite(tau_e)
            and math.isfinite(fpr_ma)
            and math.isfinite(fpr_c)
            and fpr_ma <= config.target_fpr + 1.0e-12
            and fpr_c <= config.contact_fpr_max + 1.0e-12
        ),
        "note": "H1b is not evidence-scale transport; H1a blocks the RS3A over-read",
    }
    c1_ma = [
        row
        for row in rows
        if row["domain"] == "mode_a"
        and row["regime"] in {"C1-L", "C1-H"}
        and abs(float(row.get("amp_scale", math.nan)) - PRIMARY_MODE_A_AMP) < 1.0e-12
    ]
    c1_c = [
        row
        for row in rows
        if row["domain"] == "contact"
        and row["regime"] in {"C1-L", "C1-H"}
        and row.get("script") == PRIMARY_CONTACT[0]
        and abs(float(row.get("scale", math.nan)) - PRIMARY_CONTACT[1]) < 1.0e-12
    ]
    med_d0_ma = _median([row["D0"] for row in c1_ma])
    med_d0_c = _median([row["D0"] for row in c1_c])
    med_e_ma = _median([row["E_XR"] for row in c1_ma])
    med_e_c = _median([row["E_XR"] for row in c1_c])
    g_d0 = _rel_gap(med_d0_ma, med_d0_c)
    g_e = _rel_gap(med_e_ma, med_e_c)
    h2 = {
        "pair": "pull_release@s* vs Mode-A 1.5A0, C1",
        "median_D0_mode_a": med_d0_ma,
        "median_D0_contact": med_d0_c,
        "median_E_mode_a": med_e_ma,
        "median_E_contact": med_e_c,
        "rel_gap_D0": g_d0,
        "rel_gap_E": g_e,
        "pass": bool(math.isfinite(g_d0) and math.isfinite(g_e) and g_e < g_d0),
    }
    go = bool(h1a["pass"] and h1b["pass"] and h2["pass"])
    return {
        "n_episodes": len(rows),
        "h1a_c0_near_zero": h1a,
        "h1b_shared_threshold": h1b,
        "h2_evidence_gap": h2,
        "rs3a1_go": go,
        "rewrites_rs3a_go": False,
        "rewrites_rs2_go": False,
        "unlocks_rs3b": False,
        "unlocks_domain_threshold": False,
        "uses_true_operator": False,
        "uses_domain_id": False,
        "uses_quiet_h0": False,
        "last_scalar_attempt": True,
    }


def run_r1_rs3a1(
    output: str | Path,
    *,
    rs3a_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS3A1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS3A1Config()
    _require_rs3a(Path(rs3a_summary))
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
            for script, scale in CONTACT_JOBS
        ]
        scientific = True
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[RS3A.1 {index}/{len(jobs)}] {domain} seed={seed} {regime} "
            f"amp={amp} script={script} scale={scale}",
            flush=True,
        )
        seed_everything(int(seed))
        if domain == "mode_a":
            row = _mode_a_episode(
                intake,
                seed=int(seed),
                regime=str(regime),
                amp_scale=float(amp),
                config=cfg,
            )
        else:
            row = _contact_episode(
                intake,
                c0,
                seed=int(seed),
                regime=str(regime),
                script=str(script),
                scale=float(scale),
                config=cfg,
            )
        rows.append(row)
    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs3a1_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; H1–H2 not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS2"] = {"frozen": True, "passed": False, "scientific_close": True}
    status["R1-RS3A"] = {"frozen": True, "passed": False}
    status["R1-RS3A.1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs3a1_go")),
        "smoke_only": smoke,
    }
    status["R1-RS3B"] = {"locked": True, "opened": False}
    status["R1-RS3C"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R1-RS3A.1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "statistic": "E_XR",
            "k_fit": cfg.k_fit,
            "k_hold": cfg.k_hold,
            "rbf": "door_box_12",
            "no_quiet_h0": True,
            "no_domain_id": True,
            "rs3b": "locked",
            "rs3c": "locked",
            "last_scalar_attempt": True,
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(rows),
        "aggregate": _jsonable(aggregate),
        "rs3a1_go": bool(aggregate.get("rs3a1_go")),
        "rs3a_go_unchanged": True,
        "rs2_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "domain",
        "seed",
        "regime",
        "script",
        "scale",
        "amp_scale",
        "E_XR",
        "L_par",
        "L_flex",
        "D0",
        "n_used",
        "X_phi",
        "X_phi_perp",
        "kappa_perp",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

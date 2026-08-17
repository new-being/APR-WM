"""R1-RS3A: transport-calibrated learner-visible structural evidence."""

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
from .r1_rs2a import (
    CONTACT_JOBS,
    PRIMARY_CONTACT,
    PRIMARY_MODE_A_AMP,
    _visibility,
)
from .train import seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/REG/R1/R1_RS3A_PREREG.md"
SEEDS = (12101, 12111, 12121, 12131, 12141)
REGIMES = ("C0", "C1-L", "C1-H")
MODE_A_AMPS = (1.0, 1.5)
SMOKE_JOBS = (
    ("mode_a", 12101, "C0", 1.5, None, 1.0),
    ("contact", 12101, "C0", math.nan, "fast_pull", 1.0),
)
P_FLOOR = 1.0e-300
EPS_SIGMA = 1.0e-18
GAP_EPS = 1.0e-18


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS3AConfig:
    seeds: tuple[int, ...] = SEEDS
    regimes: tuple[str, ...] = REGIMES
    mode_a_amps: tuple[float, ...] = MODE_A_AMPS
    ridge: float = 1.0e-5
    target_fpr: float = 0.01
    contact_fpr_max: float = 0.05


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _require_rs2b(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS3A locked until RS2B has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS3A locked: RS2B scientific matrix has not been run")
    return payload


def _perp(tangent: np.ndarray, residual: np.ndarray, ridge: float) -> np.ndarray:
    tang_t = torch.tensor(np.asarray(tangent, dtype=np.float64), dtype=torch.float32).unsqueeze(
        0
    )
    y_t = torch.tensor(np.asarray(residual, dtype=np.float64), dtype=torch.float32).unsqueeze(0)
    _, orthogonal, _ = tangent_decomposition(tang_t, y_t, ridge)
    return orthogonal[0].detach().cpu().numpy().astype(np.float64)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a,x). Portable (no scipy / 3.13 math)."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return math.nan
    log_pre = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(200):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < 1.0e-14 * abs(total):
                break
        lower = math.exp(min(log_pre, 700.0)) * total
        return float(max(0.0, min(1.0, 1.0 - lower)))
    b = x + 1.0 - a
    c = 1.0 / 1.0e-30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = b + an / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-12:
            break
    upper = math.exp(min(log_pre, 700.0)) * h
    return float(max(0.0, min(1.0, upper)))


def _chi2_sf(t: float, df: float) -> float:
    if not math.isfinite(t) or t <= 0.0:
        return 1.0
    if df <= 0.0:
        return math.nan
    return _regularized_gamma_q(0.5 * df, 0.5 * t)


def structural_surprisal(
    *,
    r_h0_perp: np.ndarray,
    r_probe_perp: np.ndarray,
) -> dict[str, float]:
    """Learner-visible H0-whitened surprisal. No domain ID, no true operator."""
    r_h0 = np.asarray(r_h0_perp, dtype=np.float64)
    r_pr = np.asarray(r_probe_perp, dtype=np.float64)
    sigma2 = float(np.mean(r_h0 * r_h0)) if len(r_h0) else math.nan
    d0 = float(np.sqrt(np.mean(r_pr * r_pr))) if len(r_pr) else math.nan
    t_stat = float(np.sum(r_pr * r_pr) / (sigma2 + EPS_SIGMA)) if len(r_pr) else math.nan
    df = float(max(len(r_pr) - 2, 1))
    p_val = _chi2_sf(t_stat, df) if math.isfinite(t_stat) else math.nan
    p_clip = max(float(p_val), P_FLOOR) if math.isfinite(p_val) else P_FLOOR
    return {
        "D0": d0,
        "sigma2_perp": sigma2,
        "T_perp": t_stat,
        "df": df,
        "p_perp": float(p_val) if math.isfinite(p_val) else math.nan,
        "S_perp": -math.log(p_clip),
        "n_h0": int(len(r_h0)),
        "n_probe": int(len(r_pr)),
    }


def _fit_theta(
    tangent: np.ndarray, residual: np.ndarray, intake: R1RS1AConfig
) -> torch.Tensor:
    v5 = V5Config()
    v5.ridge = intake.ridge
    v5.posterior_noise_floor = intake.posterior_noise_floor
    mean, _cov = _fit_base_posterior(
        torch.tensor(tangent, dtype=torch.float32).unsqueeze(0),
        torch.tensor(residual, dtype=torch.float32).unsqueeze(0),
        intake.force_observation_noise,
        v5,
    )
    return mean


def _after_theta(tangent: np.ndarray, residual: np.ndarray, mean: torch.Tensor) -> np.ndarray:
    tang_t = torch.tensor(np.asarray(tangent, dtype=np.float64), dtype=torch.float32).unsqueeze(
        0
    )
    y_t = torch.tensor(np.asarray(residual, dtype=np.float64), dtype=torch.float32).unsqueeze(0)
    pred = torch.einsum("bni,bi->bn", tang_t, mean)
    return (y_t - pred)[0].detach().cpu().numpy().astype(np.float64)


def _mode_a_episode(
    intake: R1RS1AConfig, *, seed: int, regime: str, amp_scale: float, config: R1RS3AConfig
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    mix = int(hashlib.md5(f"rs3a:{alpha}:{amp_scale}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 23_000 + (mix % 10_000))
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
            **scores,
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
    config: R1RS3AConfig,
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
    h0 = phase < 2.0
    probe = phase == 2.0
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
        **scores,
        "X_phi": vis["X_phi"],
        "X_phi_perp": vis["X_phi_perp"],
        "kappa_perp": vis["kappa_perp"],
    }


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(np.median(finite))


def _rel_gap(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return abs(a - b) / (0.5 * (abs(a) + abs(b)) + GAP_EPS)


def _aggregate(rows: list[dict[str, Any]], config: R1RS3AConfig) -> dict[str, Any]:
    mode_a_c0 = [
        row
        for row in rows
        if row["domain"] == "mode_a" and row["regime"] == "C0"
    ]
    contact_c0 = [
        row
        for row in rows
        if row["domain"] == "contact" and row["regime"] == "C0"
    ]
    s_c0 = [float(row["S_perp"]) for row in mode_a_c0 if math.isfinite(row["S_perp"])]
    tau_s = float(np.max(s_c0)) if s_c0 else math.nan
    fpr_ma = (
        float(np.mean([row["S_perp"] > tau_s for row in mode_a_c0])) if mode_a_c0 else math.nan
    )
    fpr_c = (
        float(np.mean([row["S_perp"] > tau_s for row in contact_c0])) if contact_c0 else math.nan
    )
    h1 = {
        "tau_S": tau_s,
        "n_mode_a_c0": len(mode_a_c0),
        "n_contact_c0": len(contact_c0),
        "fpr_mode_a": fpr_ma,
        "fpr_contact": fpr_c,
        "max_mode_a": config.target_fpr,
        "max_contact": config.contact_fpr_max,
        "pass": bool(
            math.isfinite(tau_s)
            and math.isfinite(fpr_ma)
            and math.isfinite(fpr_c)
            and fpr_ma <= config.target_fpr + 1.0e-12
            and fpr_c <= config.contact_fpr_max + 1.0e-12
        ),
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
    med_s_ma = _median([row["S_perp"] for row in c1_ma])
    med_s_c = _median([row["S_perp"] for row in c1_c])
    g_d0 = _rel_gap(med_d0_ma, med_d0_c)
    g_s = _rel_gap(med_s_ma, med_s_c)
    h2 = {
        "pair": "pull_release@s* vs Mode-A 1.5A0, C1",
        "median_D0_mode_a": med_d0_ma,
        "median_D0_contact": med_d0_c,
        "median_S_mode_a": med_s_ma,
        "median_S_contact": med_s_c,
        "rel_gap_D0": g_d0,
        "rel_gap_S": g_s,
        "pass": bool(math.isfinite(g_d0) and math.isfinite(g_s) and g_s < g_d0),
    }
    go = bool(h1["pass"] and h2["pass"])
    return {
        "n_episodes": len(rows),
        "h1_null_calibration": h1,
        "h2_evidence_gap": h2,
        "rs3a_go": go,
        "rewrites_rs2_go": False,
        "unlocks_rs3b": False,
        "unlocks_domain_threshold": False,
        "uses_true_operator": False,
        "uses_domain_id": False,
    }


def run_r1_rs3a(
    output: str | Path,
    *,
    rs2b_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS3AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS3AConfig()
    _require_rs2b(Path(rs2b_summary))
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
            f"[RS3A {index}/{len(jobs)}] {domain} seed={seed} {regime} "
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
            "rs3a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; H1–H2 not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS2"] = {"frozen": True, "passed": False, "scientific_close": True}
    status["R1-RS3A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs3a_go")),
        "smoke_only": smoke,
    }
    status["R1-RS3B"] = {"locked": True, "opened": False}
    status["R1-RS3C"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R1-RS3A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "statistic": "S_perp",
            "oracle_only": ["X_phi", "X_phi_perp", "kappa_perp"],
            "no_domain_id": True,
            "rs3b": "locked",
            "rs3c": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(rows),
        "aggregate": _jsonable(aggregate),
        "rs3a_go": bool(aggregate.get("rs3a_go")),
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
        "D0",
        "S_perp",
        "T_perp",
        "p_perp",
        "sigma2_perp",
        "X_phi",
        "X_phi_perp",
        "kappa_perp",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

"""R1-RS2A: contact-mediated structural identifiability (diagnosis only)."""

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
from .r1_rs1a3 import _exposure, _probe_spec, _rollout_with_tangent, _spearman
from .r1_rs1a4 import A0, FREQ_HZ
from .r1_rs1b import _require_rs1a5_unlock
from .r1_rs2 import (
    FROZEN_ADAPTER_SCALE,
    FROZEN_SCRIPT_AMP,
    REGIME_ALPHA,
    DoorContactC0Backend,
    R1RS2C0Config,
)
from .r1_rs2_formal import FORMAL_SEEDS, _d0_from_contact
from .train import seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v4 import OPERATOR_NAMES, operator_library
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/REG/R1/R1_RS2A_PREREG.md"
REGIMES = ("C0", "C1-L", "C1-H")
MODE_A_AMPS = (1.0, 1.5)
CONTACT_JOBS = (
    ("slow_pull", 1.0),
    ("fast_pull", 1.0),
    ("pull_release", 1.0),
    ("pull_release", FROZEN_ADAPTER_SCALE),
)
SMOKE_JOBS = (
    ("mode_a", 11101, "C1-H", 1.5, None, 1.0),
    ("contact", 11101, "C1-H", math.nan, "fast_pull", 1.0),
)
PRIMARY_CONTACT = ("pull_release", FROZEN_ADAPTER_SCALE)
PRIMARY_MODE_A_AMP = 1.5


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS2AConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = REGIMES
    mode_a_amps: tuple[float, ...] = MODE_A_AMPS
    ridge: float = 1.0e-5
    epsilon: float = 1.0e-18


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


def _require_rs2_formal(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS2A locked until RS2 Formal has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS2A locked: RS2 Formal scientific matrix has not been run")
    if "rs2_go" not in payload:
        raise RuntimeError("RS2A locked: RS2 Formal summary missing rs2_go")
    return payload


def _project_orthogonal(tangent: np.ndarray, signal: np.ndarray, ridge: float) -> np.ndarray:
    tang_t = torch.tensor(np.asarray(tangent, dtype=np.float64), dtype=torch.float32).unsqueeze(
        0
    )
    sig_t = torch.tensor(np.asarray(signal, dtype=np.float64), dtype=torch.float32).unsqueeze(0)
    _, orthogonal, _ = tangent_decomposition(tang_t, sig_t, ridge)
    return orthogonal[0].detach().cpu().numpy().astype(np.float64)


def _visibility(
    tangent: np.ndarray, velocity: np.ndarray, ridge: float, epsilon: float
) -> dict[str, float]:
    velocity = np.asarray(velocity, dtype=np.float64)
    phi = np.abs(velocity) * velocity
    x_phi = float(np.mean(phi * phi)) if len(phi) else 0.0
    if len(phi) == 0 or tangent.shape[0] == 0:
        return {"X_phi": x_phi, "X_phi_perp": math.nan, "kappa_perp": math.nan}
    phi_perp = _project_orthogonal(tangent, phi, ridge)
    x_perp = float(np.mean(phi_perp * phi_perp))
    return {
        "X_phi": x_phi,
        "X_phi_perp": x_perp,
        "kappa_perp": x_perp / (x_phi + epsilon),
    }


def _gram_diagnostics(q: np.ndarray, velocity: np.ndarray, tangent: np.ndarray) -> dict[str, float]:
    if len(q) < 2:
        return {"cond_G": math.nan, "corr_abs_v_v_signed_v2": math.nan}
    features = torch.tensor(
        np.stack(
            [np.asarray(q, dtype=np.float64), np.asarray(velocity, dtype=np.float64)],
            axis=1,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    ops = operator_library(features)[0].detach().cpu().numpy().astype(np.float64)
    gram = (ops.T @ ops) / max(len(ops), 1)
    cond = float(np.linalg.cond(gram)) if np.isfinite(gram).all() else math.nan

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1.0e-15 or b.std() < 1.0e-15:
            return math.nan
        return float(np.corrcoef(a, b)[0, 1])

    abs_v_v = ops[:, 4]
    signed_v2 = ops[:, 2]
    out = {
        "cond_G": cond,
        "corr_abs_v_v_signed_v2": corr(abs_v_v, signed_v2),
        "corr_abs_v_v_density_tangent": corr(
            abs_v_v, np.asarray(tangent[:, 0], dtype=np.float64)
        ),
        "corr_abs_v_v_qvel": corr(abs_v_v, np.asarray(tangent[:, 1], dtype=np.float64)),
    }
    for index, name in enumerate(OPERATOR_NAMES):
        out[f"gram_{name}_abs_v_v"] = float(gram[index, 4])
    return out


def _mode_a_episode(
    intake: R1RS1AConfig,
    *,
    seed: int,
    regime: str,
    amp_scale: float,
    config: R1RS2AConfig,
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    amp = A0 * float(amp_scale)
    mix = int(hashlib.md5(f"rs2a:{alpha}:{amp_scale}".encode()).hexdigest()[:8], 16)
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
        passive_y = target + intake.force_observation_noise * torch.randn(
            target.shape, generator=generator
        )
        v5 = V5Config()
        v5.ridge = intake.ridge
        v5.posterior_noise_floor = intake.posterior_noise_floor
        mean, _cov = _fit_base_posterior(
            tangent_passive.unsqueeze(0),
            passive_y.unsqueeze(0),
            intake.force_observation_noise,
            v5,
        )
        n_steps = int(round(intake.duration_s / intake.timestep))
        torques = _torque_series(
            _probe_spec(amp, FREQ_HZ), n_steps, intake.timestep, seed=seed + 91
        )
        probe = _rollout_with_tangent(backend, torques, intake.joint_limit_margin)
        tangent = np.stack([probe["density_tangent"], probe["qvel"]], axis=1)
        y = probe["residual"].astype(np.float64)
        rng = np.random.default_rng(seed + 77)
        y_obs = y + intake.force_observation_noise * rng.normal(size=y.shape)
        tang_t = torch.tensor(tangent, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(0)
        pred = torch.einsum("bni,bi->bn", tang_t, mean)
        resid = y_t - pred
        _, r_perp, _ = tangent_decomposition(tang_t, resid, intake.ridge)
        r = r_perp[0].detach().cpu().numpy()
        d0 = float(np.sqrt(np.mean(r * r)))
        vis = _visibility(tangent, probe["qvel"], config.ridge, config.epsilon)
        gram = _gram_diagnostics(probe["q"], probe["qvel"], tangent)
        exp = _exposure(probe["qvel"])
        return {
            "domain": "mode_a",
            "seed": int(seed),
            "regime": regime,
            "script": None,
            "scale": math.nan,
            "amp_scale": float(amp_scale),
            "alpha": alpha,
            "D0": d0,
            **vis,
            **gram,
            "max_abs_v": exp["max_abs_v"],
            "rms_v": exp["rms_v"],
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
    config: R1RS2AConfig,
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    backend = DoorContactC0Backend(c0, seed=int(seed), regime=regime)
    try:
        backend.prepare()
        result = backend.rollout(script, scale=float(scale))
    finally:
        backend.close()
    arrays = result["arrays"]
    d0, _r = _d0_from_contact(arrays, intake)
    probe = np.asarray(arrays["phase"], dtype=np.float64) == 2.0
    tangent = np.stack(
        (
            np.asarray(arrays["density_tangent"], dtype=np.float64)[probe],
            np.asarray(arrays["qvel"], dtype=np.float64)[probe],
        ),
        axis=1,
    )
    vis = _visibility(
        tangent,
        np.asarray(arrays["qvel"], dtype=np.float64)[probe],
        config.ridge,
        config.epsilon,
    )
    gram = _gram_diagnostics(
        np.asarray(arrays["q"], dtype=np.float64)[probe],
        np.asarray(arrays["qvel"], dtype=np.float64)[probe],
        tangent,
    )
    exp = _exposure(np.asarray(arrays["qvel"], dtype=np.float64)[probe])
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
        "D0": d0,
        **vis,
        **gram,
        "max_abs_v": exp["max_abs_v"],
        "rms_v": exp["rms_v"],
        "X_phi_rollout": float(result["X_phi"]),
    }


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(np.median(finite))


def _rel_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return abs(a - b) / max(abs(b), 1.0e-15)


def _nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not bool(np.any(mask)):
        return math.nan
    y = actual[mask]
    p = predicted[mask]
    denom = float(np.sqrt(np.mean(y * y))) + 1.0e-15
    return float(np.sqrt(np.mean((y - p) ** 2)) / denom)


def _c1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("regime") in {"C1-L", "C1-H"}]


def _aggregate(rows: list[dict[str, Any]], config: R1RS2AConfig) -> dict[str, Any]:
    del config
    c1 = _c1(rows)
    primary_contact = [
        row
        for row in c1
        if row["domain"] == "contact"
        and row.get("script") == PRIMARY_CONTACT[0]
        and abs(float(row.get("scale", math.nan)) - PRIMARY_CONTACT[1]) < 1.0e-12
    ]
    primary_mode_a = [
        row
        for row in c1
        if row["domain"] == "mode_a"
        and abs(float(row.get("amp_scale", math.nan)) - PRIMARY_MODE_A_AMP) < 1.0e-12
    ]
    x_c = _median([row["X_phi"] for row in primary_contact])
    x_m = _median([row["X_phi"] for row in primary_mode_a])
    k_c = _median([row["kappa_perp"] for row in primary_contact])
    k_m = _median([row["kappa_perp"] for row in primary_mode_a])
    rel = _rel_error(x_c, x_m)
    h1 = {
        "pair": "pull_release@s* vs Mode-A 1.5A0",
        "median_X_phi_contact": x_c,
        "median_X_phi_mode_a": x_m,
        "relative_error": rel,
        "max": 0.20,
        "pass": bool(math.isfinite(rel) and rel < 0.20),
    }
    h2 = {
        "median_kappa_contact": k_c,
        "median_kappa_mode_a": k_m,
        "ratio": (k_c / k_m)
        if (math.isfinite(k_c) and math.isfinite(k_m) and k_m > 0.0)
        else math.nan,
        "pass": bool(math.isfinite(k_c) and math.isfinite(k_m) and k_c < k_m),
    }

    alphas = np.asarray([abs(float(row["alpha"])) for row in c1], dtype=np.float64)
    d0 = np.asarray([float(row["D0"]) for row in c1], dtype=np.float64)
    raw = alphas * np.sqrt(np.asarray([float(row["X_phi"]) for row in c1], dtype=np.float64))
    vis = alphas * np.sqrt(
        np.asarray([float(row["X_phi_perp"]) for row in c1], dtype=np.float64)
    )
    rho_raw = _spearman(d0, raw)
    rho_vis = _spearman(d0, vis)
    nrmse_raw = _nrmse(d0, raw)
    nrmse_vis = _nrmse(d0, vis)
    h3a = {
        "spearman_raw": rho_raw,
        "spearman_perp": rho_vis,
        "pass": bool(math.isfinite(rho_raw) and math.isfinite(rho_vis) and rho_vis > rho_raw),
    }
    h3b = {
        "nrmse_raw": nrmse_raw,
        "nrmse_perp": nrmse_vis,
        "pass": bool(
            math.isfinite(nrmse_raw) and math.isfinite(nrmse_vis) and nrmse_vis < nrmse_raw
        ),
    }
    bin_pairs = []
    for script, amp in (("slow_pull", 1.0), ("fast_pull", 1.5)):
        contact = [
            row
            for row in c1
            if row["domain"] == "contact"
            and row.get("script") == script
            and abs(float(row.get("scale", 1.0)) - 1.0) < 1.0e-12
        ]
        mode_a = [
            row
            for row in c1
            if row["domain"] == "mode_a"
            and abs(float(row.get("amp_scale", math.nan)) - amp) < 1.0e-12
        ]
        xc = _median([row["X_phi"] for row in contact])
        xm = _median([row["X_phi"] for row in mode_a])
        bin_pairs.append(
            {
                "script": script,
                "mode_a_amp": amp,
                "median_X_phi_contact": xc,
                "median_X_phi_mode_a": xm,
                "relative_error": _rel_error(xc, xm),
                "median_kappa_contact": _median([row["kappa_perp"] for row in contact]),
                "median_kappa_mode_a": _median([row["kappa_perp"] for row in mode_a]),
            }
        )
    go = bool(h2["pass"] and h3a["pass"] and h3b["pass"])
    return {
        "n_episodes": len(rows),
        "n_c1": len(c1),
        "h1_raw_exposure": h1,
        "h2_visibility": h2,
        "h3a_spearman": h3a,
        "h3b_nrmse": h3b,
        "bin_assigned_pairs": bin_pairs,
        "secondary_gram": {
            "median_cond_G_mode_a": _median(
                [row.get("cond_G", math.nan) for row in c1 if row["domain"] == "mode_a"]
            ),
            "median_cond_G_contact": _median(
                [row.get("cond_G", math.nan) for row in c1 if row["domain"] == "contact"]
            ),
            "median_corr_abs_v_v_signed_v2_mode_a": _median(
                [
                    row.get("corr_abs_v_v_signed_v2", math.nan)
                    for row in c1
                    if row["domain"] == "mode_a"
                ]
            ),
            "median_corr_abs_v_v_signed_v2_contact": _median(
                [
                    row.get("corr_abs_v_v_signed_v2", math.nan)
                    for row in c1
                    if row["domain"] == "contact"
                ]
            ),
        },
        "rs2a_go": go,
        "rewrites_rs2_go": False,
        "rewrites_rs1c_go": False,
        "unlocks_threshold_retune": False,
        "unlocks_rs2b": False,
    }


def run_r1_rs2a(
    output: str | Path,
    *,
    rs2_formal_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS2AConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS2AConfig()
    _require_rs2_formal(Path(rs2_formal_summary))
    _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0 = R1RS2C0Config()
    rows: list[dict[str, Any]] = []

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

    for index, job in enumerate(jobs, start=1):
        domain, seed, regime, amp, script, scale = job
        print(
            f"[RS2A {index}/{len(jobs)}] {domain} seed={seed} {regime} "
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
        alpha = abs(float(row["alpha"]))
        row["pred_raw"] = alpha * math.sqrt(max(float(row["X_phi"]), 0.0))
        row["pred_perp"] = alpha * math.sqrt(max(float(row.get("X_phi_perp") or 0.0), 0.0))
        rows.append(row)

    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs2a_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; H1–H3 not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS1C"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS2-C0"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS2-Formal"] = {"unlocked": True, "passed": False, "frozen": True}
    status["R1-RS2A"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs2a_go")),
        "smoke_only": smoke,
        "diagnosis_only": True,
    }
    status["R1-RS2B"] = {"locked": True, "opened": False}
    summary = {
        "stage": "R1-RS2A",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "s_star": FROZEN_ADAPTER_SCALE,
            "script_amp_bins": dict(FROZEN_SCRIPT_AMP),
            "decision_policy": "untouched",
            "rs2b": "locked",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(rows),
        "aggregate": _jsonable(aggregate),
        "rs2a_go": bool(aggregate.get("rs2a_go")),
        "rs2_go_unchanged": True,
        "rs1c_go_unchanged": True,
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
        "alpha",
        "D0",
        "X_phi",
        "X_phi_perp",
        "kappa_perp",
        "pred_raw",
        "pred_perp",
        "cond_G",
        "corr_abs_v_v_signed_v2",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

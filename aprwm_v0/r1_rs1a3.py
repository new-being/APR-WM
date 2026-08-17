"""R1-RS1A.3: excitation-limited inadequacy identifiability."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import (
    ProbeSpec,
    _observe,
    _require_rs0_unlock,
    _sample_door_points,
    _torque_series,
)
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import (
    EPS,
    R1RS1AConfig,
    _auroc,
    _auprc,
    _operating_point,
    _recall_at_threshold,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v5 import V5Config, _fit_base_posterior


PREREG_PATH = "REPORT/REG/R1/R1_RS1A3_PREREG.md"
FORMAL_SEEDS = (9601, 9611, 9621, 9631, 9641)
SMOKE_SEED = 8981
A0 = 0.12
AMP_SCALES = (0.5, 1.0, 1.5, 2.0)
FREQS_HZ = (0.20, 0.40, 0.80)
REGIMES = ("C0", "C1-L")
TARGET_C0_FPR = 0.01


@dataclass
class R1RS1A3Config:
    base: R1RS1AConfig | None = None
    seeds: tuple[int, ...] = FORMAL_SEEDS
    amp_scales: tuple[float, ...] = AMP_SCALES
    freqs_hz: tuple[float, ...] = FREQS_HZ
    regimes: tuple[str, ...] = REGIMES
    a0: float = A0
    target_c0_fpr: float = TARGET_C0_FPR
    duration_s: float | None = None

    def resolved_base(self) -> R1RS1AConfig:
        cfg = self.base or R1RS1AConfig()
        if self.duration_s is not None:
            cfg = replace(cfg, duration_s=self.duration_s)
        return cfg


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_spec(amp: float, freq_hz: float) -> ProbeSpec:
    return ProbeSpec(
        name=f"sine_A{amp:.3f}_f{freq_hz:.2f}",
        kind="sine",
        amplitude=float(amp),
        freq_hz=float(freq_hz),
        phase=0.0,
    )


def _rollout_with_tangent(
    backend: DoorModeABackend, torques: np.ndarray, margin: float
) -> dict[str, np.ndarray]:
    if backend.hi > backend.lo:
        backend.data.qpos[backend.hinge_qpos_adr] = 0.5 * (backend.lo + backend.hi)
    backend.data.qvel[:] = 0.0
    backend._freeze_robot()
    backend.mujoco.mj_forward(backend.model, backend.data)
    records: dict[str, list[float]] = {
        "q": [],
        "qvel": [],
        "qacc": [],
        "tau_command": [],
        "tau_hidden": [],
        "residual": [],
        "density_tangent": [],
        "validity": [],
    }
    code = {"modeled": 1.0, "boundary": 0.5, "unsupported": 0.0}
    for tau in torques:
        backend._freeze_robot()
        v = float(backend.data.qvel[backend.hinge_dof])
        hidden = backend._hidden(v)
        backend.data.qfrc_applied[:] = 0.0
        backend.data.qfrc_applied[backend.hinge_dof] = float(tau) + hidden
        backend.mujoco.mj_step(backend.model, backend.data)
        residual = backend._learner_residual(tau)
        from .mujoco_force import get_mass_matrix

        mass = get_mass_matrix(backend.model, backend.data)
        qacc = float(backend.data.qacc[backend.hinge_dof])
        density_tangent = float(mass[backend.hinge_dof, backend.hinge_dof] * qacc) / (
            backend.mass_proxy
        )
        q = float(backend.data.qpos[backend.hinge_qpos_adr])
        support = backend._validity(q, margin)
        records["q"].append(q)
        records["qvel"].append(float(backend.data.qvel[backend.hinge_dof]))
        records["qacc"].append(qacc)
        records["tau_command"].append(float(tau))
        records["tau_hidden"].append(hidden)
        records["residual"].append(residual)
        records["density_tangent"].append(density_tangent)
        records["validity"].append(code[support])
    return {k: np.asarray(v, dtype=np.float64) for k, v in records.items()}


def _exposure(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    phi = np.abs(v) * v
    return {
        "max_abs_v": float(np.max(np.abs(v))) if v.size else 0.0,
        "rms_v": float(np.sqrt(np.mean(v * v))) if v.size else 0.0,
        "rms_abs_v_v": float(np.sqrt(np.mean(phi * phi))) if v.size else 0.0,
        "X_phi": float(np.mean(v**4)) if v.size else 0.0,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom < EPS:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _compute_episode(
    config: R1RS1AConfig,
    *,
    seed: int,
    regime: str,
    amp_scale: float,
    freq_hz: float,
    a0: float,
) -> dict[str, Any]:
    amp = float(a0) * float(amp_scale)
    mix = int(
        hashlib.md5(f"rs1a3:{regime}:{amp_scale}:{freq_hz}".encode()).hexdigest()[:8],
        16,
    )
    generator = torch.Generator().manual_seed(seed + 21_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime=regime,
        friction=config.friction,
        damping=config.damping,
        seed=seed,
    )
    try:
        # Passive base fit (independent of excitation).
        state = _sample_door_points(config.passive_context, "passive", generator)
        physical, tangent, target, _ = _observe(backend, state)
        passive_y = target + config.force_observation_noise * torch.randn(
            target.shape, generator=generator
        )
        v5 = V5Config()
        v5.ridge = config.ridge
        v5.posterior_noise_floor = config.posterior_noise_floor
        mean, _cov = _fit_base_posterior(
            tangent.unsqueeze(0),
            passive_y.unsqueeze(0),
            config.force_observation_noise,
            v5,
        )

        n_steps = int(round(config.duration_s / config.timestep))
        spec = _probe_spec(amp, freq_hz)
        torques = _torque_series(spec, n_steps, config.timestep, seed=seed + 91)
        probe = _rollout_with_tangent(backend, torques, config.joint_limit_margin)

        # Probe-tangent features and residual projection.
        phys = np.stack([probe["q"], probe["qvel"]], axis=1)
        tang = np.stack([probe["density_tangent"], probe["qvel"]], axis=1)
        y = probe["residual"].astype(np.float64)
        # Add tiny observation noise for parity with prior stages.
        rng = np.random.default_rng(seed + 77)
        y_obs = y + config.force_observation_noise * rng.normal(size=y.shape)

        tang_t = torch.tensor(tang, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(0)
        pred = torch.einsum("bni,bi->bn", tang_t, mean)
        resid = y_t - pred
        _, r_perp, _ = tangent_decomposition(tang_t, resid, config.ridge)
        r = r_perp[0].detach().cpu().numpy()
        d0 = float(np.sqrt(np.mean(r * r)))
        exp = _exposure(probe["qvel"])

        return {
            "seed": seed,
            "regime": regime,
            "amp_scale": float(amp_scale),
            "freq_hz": float(freq_hz),
            "amplitude": amp,
            "D0": d0,
            **exp,
            "probe_arrays": {
                **probe,
                "r_perp": r,
                "physical": phys,
            },
        }
    finally:
        backend.close()


def _aggregate(rows: list[dict[str, Any]], config: R1RS1A3Config) -> dict[str, Any]:
    cells = []
    for amp in config.amp_scales:
        for freq in config.freqs_hz:
            cell_rows = [
                r
                for r in rows
                if abs(r["amp_scale"] - amp) < 1e-12 and abs(r["freq_hz"] - freq) < 1e-12
            ]
            c0 = [r for r in cell_rows if r["regime"] == "C0"]
            c1 = [r for r in cell_rows if r["regime"] == "C1-L"]
            scores = np.asarray(
                [r["D0"] for r in c0] + [r["D0"] for r in c1], dtype=np.float64
            )
            labels = np.asarray([0] * len(c0) + [1] * len(c1), dtype=np.int32)
            auroc = _auroc(scores, labels)
            auprc = _auprc(scores, labels)
            op = _operating_point(scores, labels, target_fpr=config.target_c0_fpr)
            thr = op["threshold"]
            recall = (
                _recall_at_threshold(
                    np.asarray([r["D0"] for r in c1]),
                    np.ones(len(c1), dtype=np.int32),
                    thr,
                )
                if c1
                else float("nan")
            )
            x_phi = float(np.mean([r["X_phi"] for r in c1])) if c1 else float("nan")
            cells.append(
                {
                    "amp_scale": float(amp),
                    "freq_hz": float(freq),
                    "n_c0": len(c0),
                    "n_c1": len(c1),
                    "auroc": auroc,
                    "auprc": auprc,
                    "threshold": thr,
                    "c0_fpr": op["fpr"],
                    "recall_c1_l": recall,
                    "mean_X_phi_c1": x_phi,
                    "mean_rms_v_c1": float(np.mean([r["rms_v"] for r in c1]))
                    if c1
                    else float("nan"),
                    "mean_D0_c0": float(np.mean([r["D0"] for r in c0]))
                    if c0
                    else float("nan"),
                    "mean_D0_c1": float(np.mean([r["D0"] for r in c1]))
                    if c1
                    else float("nan"),
                }
            )

    # Seed-mean recall by amplitude (avg over freq)
    amp_summary = []
    for amp in config.amp_scales:
        seed_recalls = []
        for seed in sorted({r["seed"] for r in rows}):
            recalls = []
            for freq in config.freqs_hz:
                cell = next(
                    c
                    for c in cells
                    if abs(c["amp_scale"] - amp) < 1e-12
                    and abs(c["freq_hz"] - freq) < 1e-12
                )
                # Recompute seed-specific recall with cell threshold
                c0 = [
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r["regime"] == "C0"
                    and abs(r["amp_scale"] - amp) < 1e-12
                    and abs(r["freq_hz"] - freq) < 1e-12
                ]
                c1 = [
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r["regime"] == "C1-L"
                    and abs(r["amp_scale"] - amp) < 1e-12
                    and abs(r["freq_hz"] - freq) < 1e-12
                ]
                # Use cell-level threshold from pooled seeds for stability
                thr = cell["threshold"]
                if c1:
                    recalls.append(float(np.mean([r["D0"] >= thr for r in c1])))
            if recalls:
                seed_recalls.append(float(np.mean(recalls)))
        amp_summary.append(
            {
                "amp_scale": float(amp),
                "seed_mean_recall": float(np.mean(seed_recalls))
                if seed_recalls
                else float("nan"),
                "seed_recalls": seed_recalls,
            }
        )

    r_low = next(a["seed_mean_recall"] for a in amp_summary if abs(a["amp_scale"] - 0.5) < 1e-12)
    r_high = next(a["seed_mean_recall"] for a in amp_summary if abs(a["amp_scale"] - 2.0) < 1e-12)

    # Episode-level Spearman on C1-L: X_phi vs detect under cell threshold
    c1_all = [r for r in rows if r["regime"] == "C1-L"]
    detects = []
    xphis = []
    for r in c1_all:
        cell = next(
            c
            for c in cells
            if abs(c["amp_scale"] - r["amp_scale"]) < 1e-12
            and abs(c["freq_hz"] - r["freq_hz"]) < 1e-12
        )
        detects.append(1.0 if r["D0"] >= cell["threshold"] else 0.0)
        xphis.append(r["X_phi"])
    spearman = _spearman(np.asarray(xphis), np.asarray(detects))

    # Seed-level Spearman mean
    seed_sp = []
    for seed in sorted({r["seed"] for r in c1_all}):
        xs, ys = [], []
        for r in c1_all:
            if r["seed"] != seed:
                continue
            cell = next(
                c
                for c in cells
                if abs(c["amp_scale"] - r["amp_scale"]) < 1e-12
                and abs(c["freq_hz"] - r["freq_hz"]) < 1e-12
            )
            xs.append(r["X_phi"])
            ys.append(1.0 if r["D0"] >= cell["threshold"] else 0.0)
        seed_sp.append(_spearman(np.asarray(xs), np.asarray(ys)))

    # Simple logistic MLEs for reporting (2-param)
    x = np.asarray(xphis, dtype=np.float64)
    y = np.asarray(detects, dtype=np.float64)
    logistic = {"a": float("nan"), "b": float("nan"), "note": "insufficient variation"}
    if x.size >= 8 and np.unique(y).size > 1 and np.std(x) > EPS:
        # Newton on logit model
        a, b = 0.0, 0.0
        for _ in range(50):
            eta = a * x + b
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            w = p * (1.0 - p) + 1e-8
            # gradient / hessian
            ga = float(np.sum((y - p) * x))
            gb = float(np.sum(y - p))
            haa = float(-np.sum(w * x * x))
            hab = float(-np.sum(w * x))
            hbb = float(-np.sum(w))
            det = haa * hbb - hab * hab
            if abs(det) < 1e-18:
                break
            da = (hbb * ga - hab * gb) / det
            db = (-hab * ga + haa * gb) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-8:
                break
        logistic = {"a": a, "b": b, "note": "MLE logit(P)=a X_phi + b"}

    gates = {
        "recall_2A_gt_0.5A": bool(r_high > r_low),
        "seed_mean_recall_0.5A": r_low,
        "seed_mean_recall_2A": r_high,
        "spearman_Xphi_detect": spearman,
        "spearman_positive": bool(spearman > 0.0),
        "seed_mean_spearman": float(np.nanmean(seed_sp)) if seed_sp else float("nan"),
    }
    gates["rs1a3_go"] = bool(
        gates["recall_2A_gt_0.5A"] and gates["spearman_positive"]
    )

    return {
        "cells": cells,
        "amp_summary": amp_summary,
        "logistic": logistic,
        "gates": gates,
        "rs1a3_go": gates["rs1a3_go"],
        "n_episodes": len(rows),
    }


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = row["probe_arrays"]
    meta = {k: v for k, v in row.items() if k != "probe_arrays"}
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["json"] = json.dumps(
            meta, sort_keys=True, default=str
        )
        raw = handle.create_group("raw_truth")
        learner = handle.create_group("learner_visible")
        for key in (
            "q",
            "qvel",
            "qacc",
            "tau_command",
            "tau_hidden",
            "residual",
            "validity",
            "density_tangent",
            "r_perp",
        ):
            if key in arrays:
                raw.create_dataset(key, data=arrays[key], compression="gzip")
        for key in ("q", "qvel", "qacc", "tau_command", "residual", "validity", "r_perp"):
            if key in arrays:
                learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_tau_hidden"] = True
        scores = handle.create_group("detector_scores")
        scores.attrs["D0"] = float(row["D0"])
        scores.attrs["X_phi"] = float(row["X_phi"])
        scores.attrs["revision_pipeline_executed"] = False
        scores.attrs["typed_support_frozen"] = True


def run_r1_rs1a3(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    config: R1RS1A3Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1A3Config()
    base = cfg.resolved_base()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _ = resolve_device("cpu")

    if smoke:
        base = replace(base, duration_s=2.0, discovery_count=8, passive_context=8)
        jobs = [
            (SMOKE_SEED, regime, 1.0, 0.40) for regime in REGIMES
        ]
        scientific = False
    else:
        jobs = [
            (seed, regime, amp, freq)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for amp in cfg.amp_scales
            for freq in cfg.freqs_hz
        ]
        scientific = True

    rows: list[dict[str, Any]] = []
    for index, (seed, regime, amp, freq) in enumerate(jobs, start=1):
        print(
            f"[RS1A.3 {index}/{len(jobs)}] seed={seed} regime={regime} "
            f"A={amp}*A0 f={freq}",
            flush=True,
        )
        seed_everything(seed)
        row = _compute_episode(
            base,
            seed=seed,
            regime=regime,
            amp_scale=float(amp),
            freq_hz=float(freq),
            a0=cfg.a0,
        )
        rel = (
            f"seed_{seed}/{regime}/A{amp:.1f}_f{freq:.2f}.hdf5"
        )
        _write_episode_h5(root / rel, row)
        slim = {k: v for k, v in row.items() if k != "probe_arrays"}
        slim["path"] = rel
        rows.append(slim)

    aggregate = (
        _aggregate(rows, cfg)
        if scientific
        else {
            "rs1a3_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only",
        }
    )

    csv_rows = [
        {
            "seed": r["seed"],
            "regime": r["regime"],
            "amp_scale": r["amp_scale"],
            "freq_hz": r["freq_hz"],
            "amplitude": r["amplitude"],
            "D0": r["D0"],
            "max_abs_v": r["max_abs_v"],
            "rms_v": r["rms_v"],
            "rms_abs_v_v": r["rms_abs_v_v"],
            "X_phi": r["X_phi"],
        }
        for r in rows
    ]
    _write_csv(root / "excitation_scores.csv", csv_rows)
    if scientific:
        _write_csv(root / "cells.csv", aggregate["cells"])

    status = stage_status()
    status["R1-RS1A"] = {"unlocked": True, "informative": True}
    status["R1-RS1A.1"] = {"unlocked": True, "informative": True}
    status["R1-RS1A.2"] = {"unlocked": True, "informative": True}
    status["R1-RS1A.3"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1a3_go")),
        "smoke_only": smoke,
    }
    status["R1-RS1A.4"] = {"locked_until_RS1A3": True}
    status["R1-RS1B"] = {"locked": True}
    status["R1-RS2"] = {"locked": True}

    summary = {
        "stage": "R1-RS1A.3",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen": {
            "detector": "D0_probe_trajectory",
            "typed_support_channel": "frozen_offline",
            "revision_pipeline": False,
            "a0": cfg.a0,
            "amp_scales": list(cfg.amp_scales),
            "freqs_hz": list(cfg.freqs_hz),
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1a3_go": bool(aggregate.get("rs1a3_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

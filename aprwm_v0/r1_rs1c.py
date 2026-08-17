"""R1-RS1C: layered epistemic–physical policy (new hypothesis, not RS1B repair)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .r1_mj import stage_status
from .r1_rs1 import ProbeSpec, R1RS1Config, _evaluate_episode, _mean_ci, _require_rs0_unlock
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a4 import A0, FREQ_HZ, _compute_episode
from .r1_rs1a5 import LAMBDA_COST
from .r1_rs1b import (
    DAMPING_TOL,
    H32,
    MACRO_STEPS,
    POWER_TOL,
    R1RS1BConfig,
    _local_diagnostics,
    _require_rs1a5_unlock,
)
from .r1_rs1b1 import R1RS1B1Config, _blind_h32_calibrated
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1_RS1C_PREREG.md"
FORMAL_SEEDS = (10101, 10111, 10121, 10131, 10141)
SMOKE_SEED = 9041
ALPHAS = (0.0, -0.09, -0.12, -0.18, -0.24)
AMP_SCALES = (0.5, 1.0, 1.5, 2.0)
A_STAR = 1.5
FALSE_INSTALL_T_MAX = 0.01
SMOKE_JOBS = (
    (SMOKE_SEED, 0.0, 0.5),
    (SMOKE_SEED, 0.0, 1.5),
    (SMOKE_SEED, -0.24, 0.5),
    (SMOKE_SEED, -0.24, 1.5),
)


@dataclass
class R1RS1CConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    alphas: tuple[float, ...] = ALPHAS
    amp_scales: tuple[float, ...] = AMP_SCALES
    a_star: float = A_STAR
    lambda_cost: float = LAMBDA_COST
    h32: int = H32
    macro_steps: int = MACRO_STEPS
    query_count: int = 8
    diagnostic_points: int = 48
    jacobian_epsilon: float = 1.0e-4
    damping_tolerance: float = DAMPING_TOL
    power_tolerance: float = POWER_TOL
    joint_limit_margin: float = 0.05
    friction: float = 0.10
    damping: float = 0.10
    false_install_t_max: float = FALSE_INSTALL_T_MAX


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_rs1b_config(config: R1RS1CConfig) -> R1RS1BConfig:
    return R1RS1BConfig(
        seeds=config.seeds,
        alphas=config.alphas,
        amp_scales=config.amp_scales,
        h32=config.h32,
        macro_steps=config.macro_steps,
        query_count=config.query_count,
        diagnostic_points=config.diagnostic_points,
        jacobian_epsilon=config.jacobian_epsilon,
        damping_tolerance=config.damping_tolerance,
        power_tolerance=config.power_tolerance,
        joint_limit_margin=config.joint_limit_margin,
        friction=config.friction,
        damping=config.damping,
    )


def _as_rs1b1_config(config: R1RS1CConfig) -> R1RS1B1Config:
    return R1RS1B1Config(
        seeds=config.seeds,
        alphas=config.alphas,
        amp_scales=config.amp_scales,
        h32=config.h32,
        macro_steps=config.macro_steps,
        query_count=config.query_count,
        diagnostic_points=config.diagnostic_points,
        jacobian_epsilon=config.jacobian_epsilon,
        damping_tolerance=config.damping_tolerance,
        power_tolerance=config.power_tolerance,
        joint_limit_margin=config.joint_limit_margin,
        friction=config.friction,
        damping=config.damping,
    )


def _regime(alpha: float) -> str:
    if abs(alpha) < 1.0e-15:
        return "C0"
    if abs(alpha) <= 0.12 + 1.0e-12:
        return "C1-L"
    return "C1-H"


def _bucket(*, consequential: bool, detected: bool) -> str:
    if not consequential:
        return "tolerate"
    if detected:
        return "revise_worthy"
    return "probe"


def _monitor_intensity(*, installed: bool, i_exit: float) -> str:
    if not installed:
        return "none"
    if math.isfinite(i_exit) and i_exit > 0.0:
        return "elevated"
    return "normal"


def _policy_gain(row: dict[str, Any], policy: str) -> float:
    gain = row.get("gain_h32")
    if gain is None or not math.isfinite(float(gain)):
        return 0.0
    value = float(gain)
    if policy == "none":
        return 0.0
    if policy == "rs1c":
        return value if row.get("accepted") else 0.0
    if policy == "veto":
        i_exit = float(row.get("I_exit", 0.0) or 0.0)
        return value if row.get("accepted") and i_exit <= 0.0 else 0.0
    if policy == "always":
        return value if row.get("accepted_always") else 0.0
    raise ValueError(policy)


def _seed_mean_gains(rows: list[dict[str, Any]], seeds: tuple[int, ...], policy: str) -> dict[str, float]:
    seed_means: list[float] = []
    for seed in seeds:
        values = [_policy_gain(row, policy) for row in rows if row["seed"] == seed]
        if values:
            seed_means.append(float(np.mean(values)))
    return _mean_ci(seed_means)


def _fail_layer(gates: dict[str, Any]) -> str | None:
    if not gates["nontrivial_installs"]["pass"]:
        return "allocation"
    if not gates["passivity_among_installs"]["pass"]:
        return "passivity"
    if not gates["false_install_on_T"]["pass"]:
        return "allocation"
    if not gates["gain_vs_none"]["pass"]:
        return "utility"
    if not gates["gain_vs_veto"]["pass"]:
        return "utility"
    if not gates["monitor_consistency"]["pass"]:
        return "monitoring_leakage_into_accept"
    return None


def _aggregate(rows: list[dict[str, Any]], queries: list[dict[str, Any]], config: R1RS1CConfig) -> dict[str, Any]:
    installed = [row for row in rows if row.get("accepted")]
    tolerate = [row for row in rows if row.get("bucket_initial") == "tolerate"]
    false_t = [row for row in tolerate if row.get("accepted")]
    false_t_rate = float(len(false_t) / len(tolerate)) if tolerate else 0.0
    passivity_violations = int(sum(bool(row.get("passivity_violation")) for row in installed))

    monitor_ok = True
    for row in installed:
        i_exit = float(row.get("I_exit", math.nan))
        intensity = row.get("monitor_intensity")
        if row.get("accepted_flipped_by_support"):
            monitor_ok = False
        if math.isfinite(i_exit) and i_exit > 0.0:
            if intensity != "elevated":
                monitor_ok = False
        else:
            if intensity != "normal":
                monitor_ok = False

    gain_rs1c = _seed_mean_gains(rows, config.seeds, "rs1c")
    gain_none = _seed_mean_gains(rows, config.seeds, "none")
    gain_veto = _seed_mean_gains(rows, config.seeds, "veto")
    gain_always = _seed_mean_gains(rows, config.seeds, "always")
    vs_none = float(gain_rs1c["mean"] - gain_none["mean"]) if gain_rs1c["n"] else math.nan
    vs_veto = float(gain_rs1c["mean"] - gain_veto["mean"]) if gain_rs1c["n"] else math.nan

    installed_exit_q = [
        query
        for query in queries
        if query.get("accepted") and float(query.get("episode_I_exit", 0.0) or 0.0) > 0.0
    ]
    n_harmful_exit = int(sum(bool(query.get("harmful")) for query in installed_exit_q))
    p_harmful_exit = (
        float(n_harmful_exit / len(installed_exit_q)) if installed_exit_q else math.nan
    )

    gates = {
        "nontrivial_installs": {
            "value": len(installed),
            "pass": len(installed) > 0,
        },
        "passivity_among_installs": {
            "value": passivity_violations,
            "max": 0,
            "pass": passivity_violations == 0,
        },
        "false_install_on_T": {
            "value": false_t_rate,
            "n_tolerate": len(tolerate),
            "n_false": len(false_t),
            "max": config.false_install_t_max,
            "pass": false_t_rate <= config.false_install_t_max,
        },
        "gain_vs_none": {
            **gain_rs1c,
            "vs_none": vs_none,
            "pass": bool(gain_rs1c["n"] > 0 and vs_none > 0.0),
        },
        "gain_vs_veto": {
            "rs1c": gain_rs1c["mean"],
            "veto": gain_veto["mean"],
            "delta": vs_veto,
            "pass": bool(gain_rs1c["n"] > 0 and math.isfinite(vs_veto) and vs_veto >= 0.0),
        },
        "monitor_consistency": {
            "pass": monitor_ok,
        },
    }
    go = bool(all(item["pass"] for item in gates.values()))
    return {
        "n_episodes": len(rows),
        "n_tolerate": sum(row.get("bucket_final") == "tolerate" for row in rows),
        "n_probe": sum(row.get("bucket_final") == "probe" for row in rows),
        "n_revise_worthy": sum(row.get("revise_worthy") for row in rows),
        "n_promoted": sum(bool(row.get("promoted")) for row in rows),
        "n_installed": len(installed),
        "n_accepted_always": sum(bool(row.get("accepted_always")) for row in rows),
        "policy_gains": {
            "rs1c": gain_rs1c,
            "none": gain_none,
            "veto": gain_veto,
            "always": gain_always,
        },
        "p_harmful_given_install_and_exit": p_harmful_exit,
        "n_queries_install_and_exit": len(installed_exit_q),
        "gates": gates,
        "fail_layer": None if go else _fail_layer(gates),
        "rs1c_go": go,
        "rewrites_rs1b_go": False,
        "installs_support_exit_hard_filter": False,
    }


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = row.pop("_probe_arrays")
    h32 = row.pop("_h32_arrays", None)
    try:
        with h5py.File(path, "w") as handle:
            handle.create_group("metadata").attrs["json"] = json.dumps(
                row, sort_keys=True, default=str
            )
            raw = handle.create_group("raw_truth")
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
                if key in probe:
                    raw.create_dataset(key, data=probe[key], compression="gzip")
            learner = handle.create_group("learner_visible")
            for key in ("q", "qvel", "qacc", "tau_command", "residual", "validity"):
                if key in probe:
                    learner.create_dataset(key, data=probe[key], compression="gzip")
            learner.attrs["excludes_tau_hidden"] = True
            learner.attrs["excludes_truth_qfrc_passive"] = True
            decision = handle.create_group("decision")
            decision.attrs["bucket_initial"] = str(row["bucket_initial"])
            decision.attrs["bucket_final"] = str(row["bucket_final"])
            decision.attrs["revise_worthy"] = bool(row["revise_worthy"])
            decision.attrs["accepted"] = bool(row.get("accepted", False))
            decision.attrs["accepted_veto"] = bool(row.get("accepted_veto", False))
            decision.attrs["accepted_always"] = bool(row.get("accepted_always", False))
            decision.attrs["monitor_intensity"] = str(row.get("monitor_intensity", "none"))
            decision.attrs["support_veto"] = False
            decision.attrs["h32_blind"] = True
            diagnostics = handle.create_group("diagnostics")
            for key in (
                "effective_damping_min",
                "power_max",
                "I_exit",
                "D_exit",
                "gain_h32",
            ):
                if key in row and row[key] is not None:
                    diagnostics.attrs[key] = row[key]
            if h32 is not None:
                evaluation = handle.create_group("evaluation")
                for key, value in h32.items():
                    evaluation.create_dataset(key, data=value, compression="gzip")
    finally:
        row["_probe_arrays"] = probe
        if h32 is not None:
            row["_h32_arrays"] = h32


def run_r1_rs1c(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS1CConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1CConfig()
    _require_rs0_unlock(Path(rs0_summary))
    policy = _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")

    policy_aggregate = policy["aggregate"]
    c_tol = float(policy_aggregate["gates"]["C_tol"])
    thresholds = {
        float(amplitude): float(cell["threshold"])
        for amplitude, cell in policy_aggregate["cells"].items()
    }
    a_star = float(cfg.a_star)
    lambda_cost = float(cfg.lambda_cost)
    if smoke:
        jobs = list(SMOKE_JOBS)
        scientific = False
    else:
        jobs = [
            (seed, alpha, amplitude)
            for seed in cfg.seeds
            for alpha in cfg.alphas
            for amplitude in cfg.amp_scales
        ]
        scientific = True

    intake_config = R1RS1AConfig()
    revision_config = R1RS1Config()
    b_cfg = _as_rs1b_config(cfg)
    b1_cfg = _as_rs1b1_config(cfg)
    rows: list[dict[str, Any]] = []
    query_catalog: list[dict[str, Any]] = []

    for index, (seed, alpha, amplitude) in enumerate(jobs, start=1):
        print(
            f"[RS1C {index}/{len(jobs)}] seed={seed} alpha={alpha} A={amplitude}*A0",
            flush=True,
        )
        seed_everything(seed)
        intake = _compute_episode(
            intake_config,
            seed=seed,
            alpha=float(alpha),
            amp_scale=float(amplitude),
            freq_hz=FREQ_HZ,
            a0=A0,
        )
        threshold = thresholds[float(amplitude)]
        detected = bool(intake["D0"] >= threshold)
        consequential = bool(intake["C"] >= c_tol)
        bucket_initial = _bucket(consequential=consequential, detected=detected)
        bucket = bucket_initial
        promoted = False
        voi_acquired = False
        voi_flip = False
        voi_v = math.nan
        revision_amp = float(amplitude)
        revision_intake = intake

        if bucket == "probe" and float(amplitude) < a_star - 1.0e-12:
            extra = _compute_episode(
                intake_config,
                seed=seed,
                alpha=float(alpha),
                amp_scale=a_star,
                freq_hz=FREQ_HZ,
                a0=A0,
            )
            voi_acquired = True
            star_threshold = thresholds[a_star]
            voi_flip = bool(extra["D0"] >= star_threshold)
            voi_v = (1.0 if voi_flip else 0.0) * float(intake["C"]) - lambda_cost * (a_star**2)
            if voi_flip:
                promoted = True
                bucket = "revise_worthy"
                detected = True
                revision_amp = a_star
                revision_intake = extra

        revise_worthy = bucket == "revise_worthy"
        row: dict[str, Any] = {
            "seed": seed,
            "alpha": float(alpha),
            "amp_scale": float(amplitude),
            "revision_amp_scale": revision_amp,
            "D0": float(intake["D0"]),
            "C": float(intake["C"]),
            "X_phi": float(intake["X_phi"]),
            "detect_threshold": threshold,
            "C_tol": c_tol,
            "detected_initial": bool(intake["D0"] >= threshold),
            "detected": detected,
            "consequential": consequential,
            "bucket_initial": bucket_initial,
            "bucket_final": bucket,
            "revise_worthy": revise_worthy,
            "promoted": promoted,
            "voi_acquired": voi_acquired,
            "voi_flip": voi_flip,
            "voi_V": voi_v,
            "pipeline_executed": False,
            "pipeline_accepted": False,
            "accepted": False,
            "accepted_always": False,
            "accepted_veto": False,
            "accepted_flipped_by_support": False,
            "monitor_intensity": "none",
            "_probe_arrays": revision_intake["probe_arrays"],
        }

        candidate = None
        if revise_worthy:
            probe = ProbeSpec(
                name=f"RS1C-A{revision_amp:.1f}",
                kind="sine",
                amplitude=A0 * float(revision_amp),
                freq_hz=FREQ_HZ,
            )
            proposal = _evaluate_episode(
                revision_config,
                seed=seed,
                regime=_regime(float(alpha)),
                probe_name=probe.name,
                device=device,
                scientific=scientific,
                alpha=float(alpha),
                probe_spec=probe,
                return_candidate=True,
            )
            candidate = proposal.pop("_candidate")
            row.update(
                {
                    "pipeline_executed": True,
                    "pipeline_accepted": bool(proposal["accepted"]),
                    "triggered": bool(proposal["triggered"]),
                    "selected_operator": proposal["selected_operator"],
                    "alpha_hat": float(proposal["alpha_hat"]),
                    "short_utility": float(proposal["utility"]),
                    "proposal_force_accepted_diagnostic": bool(candidate["force_accepted"]),
                    "proposal_short_safe": bool(candidate["short_safe"]),
                }
            )
            local = _local_diagnostics(
                b_cfg,
                seed=seed,
                alpha=float(alpha),
                candidate=candidate,
                probe_arrays=revision_intake["probe_arrays"],
            )
            row.update(local)
            row["accepted"] = bool(
                row["pipeline_accepted"]
                and row["effective_damping_pass"]
                and not row["passivity_violation"]
            )
            row["accepted_always"] = True
            frozen_accept = bool(row["accepted"])
            blind, arrays, query_rows = _blind_h32_calibrated(
                b1_cfg,
                seed=seed,
                alpha=float(alpha),
                candidate=candidate,
                fit_support=local["fit_support"],
            )
            row.update(blind)
            row["_h32_arrays"] = arrays
            i_exit = float(row.get("I_exit", 0.0) or 0.0)
            row["monitor_intensity"] = _monitor_intensity(
                installed=frozen_accept, i_exit=i_exit
            )
            row["accepted"] = frozen_accept
            row["accepted_flipped_by_support"] = bool(row["accepted"] != frozen_accept)
            row["accepted_veto"] = bool(frozen_accept and i_exit <= 0.0)
            for query in query_rows:
                query_catalog.append(
                    {
                        **query,
                        "seed": seed,
                        "alpha": float(alpha),
                        "amp_scale": float(amplitude),
                        "accepted": frozen_accept,
                        "episode_I_exit": i_exit,
                    }
                )

        rel = f"seed_{seed}/alpha_{alpha:+.2f}/A{amplitude:.1f}.hdf5"
        _write_episode_h5(root / rel, row)
        row["path"] = rel
        rows.append(row)

    aggregate = (
        _aggregate(rows, query_catalog, cfg)
        if scientific
        else {
            "rs1c_go": False,
            "n_episodes": len(rows),
            "n_installed": sum(row["accepted"] for row in rows),
            "note": "plumbing smoke only; confirmatory gates not evaluated",
            "rewrites_rs1b_go": False,
            "installs_support_exit_hard_filter": False,
        }
    )

    csv_keys = (
        "seed",
        "alpha",
        "amp_scale",
        "revision_amp_scale",
        "D0",
        "C",
        "bucket_initial",
        "bucket_final",
        "revise_worthy",
        "promoted",
        "voi_acquired",
        "pipeline_accepted",
        "accepted",
        "accepted_veto",
        "accepted_always",
        "monitor_intensity",
        "I_exit",
        "gain_h32",
        "passivity_violation",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )

    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {"unlocked": True, "passed": True}
    status["R1-RS1A.5"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS1B"] = {"unlocked": True, "passed": False, "frozen": True}
    status["R1-RS1C"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1c_go")),
        "smoke_only": smoke,
        "rewrites_rs1b_go": False,
    }
    status["R1-RS2"] = {
        "locked": not bool(aggregate.get("rs1c_go")),
        "unlocked": bool(aggregate.get("rs1c_go")),
    }

    slim_rows = []
    for row in rows:
        slim_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"_probe_arrays", "_h32_arrays"}
            }
        )
    summary = {
        "stage": "R1-RS1C",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "frozen_intake": {
            "source": str(Path(rs1a5_summary)),
            "C_tol": c_tol,
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "A_star": a_star,
            "lambda": lambda_cost,
            "support_role": "confidence_monitoring_not_veto",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "acceptance_order": (
            "tolerate/probe/revise_worthy -> optional VoI A* evidence -> "
            "candidate -> passivity -> short utility -> freeze install -> "
            "blind H32 monitor flag (no veto)"
        ),
        "episodes": slim_rows,
        "aggregate": aggregate,
        "rs1c_go": bool(aggregate.get("rs1c_go")),
        "rs1b_go_unchanged": True,
        "unlocks_rs2_draft": bool(aggregate.get("rs1c_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    return summary

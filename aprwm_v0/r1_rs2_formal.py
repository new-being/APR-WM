"""R1-RS2 Formal: frozen RS1C policy under contact-mediated excitation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import ProbeSpec, R1RS1Config, _evaluate_episode, _mean_ci, _sample_door_points
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a4 import A0, FREQ_HZ, HORIZON_MACRO, MACRO_STEPS, QUERY_COUNT, _compute_episode
from .r1_rs1a5 import LAMBDA_COST
from .r1_rs1b import (
    DAMPING_TOL,
    H32,
    POWER_TOL,
    _local_diagnostics,
    _require_rs1a5_unlock,
    _revision_force,
)
from .r1_rs1b1 import (
    _terminal_rmse,
    fit_box_distance,
    modeled_distance,
)
from .r1_rs1c import (
    _as_rs1b_config,
    _bucket,
    _monitor_intensity,
    _seed_mean_gains,
)
from .r1_rs1_backend import DoorModeABackend
from .r1_rs2 import (
    DEV_SEEDS,
    FROZEN_ADAPTER_SCALE,
    FROZEN_SCRIPT_AMP,
    PREREG_PATH,
    REGIME_ALPHA,
    SCRIPTS,
    DoorContactC0Backend,
    R1RS2C0Config,
    _prereg_sha256,
    _script_manifest,
)
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import tangent_decomposition
from .v5 import V5Config, _fit_base_posterior


FORMAL_SEEDS = (11101, 11111, 11121, 11131, 11141)
REGIMES = ("C0", "C1-L", "C1-H", "C2")
REPEATS = (0, 1, 2)
A_STAR = 1.5
SMOKE_JOBS = (
    (11101, "C0", "slow_pull", 0),
    (11101, "C1-H", "slow_pull", 0),
)


@dataclass(frozen=True)
class R1RS2FormalConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = REGIMES
    scripts: tuple[str, ...] = SCRIPTS
    repeats: tuple[int, ...] = REPEATS
    a_star: float = A_STAR
    adapter_scale: float = FROZEN_ADAPTER_SCALE
    lambda_cost: float = LAMBDA_COST
    h32: int = H32
    query_count: int = 8
    diagnostic_points: int = 48
    jacobian_epsilon: float = 1.0e-4
    damping_tolerance: float = DAMPING_TOL
    power_tolerance: float = POWER_TOL
    joint_limit_margin: float = 0.05
    friction: float = 0.10
    damping: float = 0.10
    always_lambda_r: tuple[float, ...] = (0.0, 2.5e-4, 5.0e-4, 1.0e-3)


def _require_rs2_c0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS2 Formal locked until RS2-C0 passes; missing summary")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("rs2_c0_pass"):
        raise RuntimeError("RS2 Formal locked: rs2_c0_pass is false")
    adapter = payload.get("adapter", {})
    if abs(float(adapter.get("selected_scale", math.nan)) - FROZEN_ADAPTER_SCALE) > 1.0e-12:
        raise RuntimeError("RS2 Formal locked: C0 adapter scale is not the frozen s*=2.0")
    return payload


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


def _as_rs1c_like(config: R1RS2FormalConfig) -> Any:
    from .r1_rs1c import R1RS1CConfig

    return R1RS1CConfig(
        seeds=config.seeds,
        h32=config.h32,
        query_count=config.query_count,
        diagnostic_points=config.diagnostic_points,
        jacobian_epsilon=config.jacobian_epsilon,
        damping_tolerance=config.damping_tolerance,
        power_tolerance=config.power_tolerance,
        joint_limit_margin=config.joint_limit_margin,
        friction=config.friction,
        damping=config.damping,
    )


def _d0_from_contact(arrays: dict[str, np.ndarray], intake_config: R1RS1AConfig) -> tuple[float, np.ndarray]:
    phase = np.asarray(arrays["phase"], dtype=np.float64)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    tangent = np.stack(
        (
            np.asarray(arrays["density_tangent"], dtype=np.float64),
            np.asarray(arrays["qvel"], dtype=np.float64),
        ),
        axis=1,
    )
    passive = phase < 2.0
    probe = phase == 2.0
    if not bool(np.any(passive)) or not bool(np.any(probe)):
        d0 = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else math.nan
        return d0, residual
    v5 = V5Config()
    v5.ridge = intake_config.ridge
    v5.posterior_noise_floor = intake_config.posterior_noise_floor
    mean, _cov = _fit_base_posterior(
        torch.tensor(tangent[passive], dtype=torch.float32).unsqueeze(0),
        torch.tensor(residual[passive], dtype=torch.float32).unsqueeze(0),
        intake_config.force_observation_noise,
        v5,
    )
    tang_t = torch.tensor(tangent[probe], dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(residual[probe], dtype=torch.float32).unsqueeze(0)
    pred = torch.einsum("bni,bi->bn", tang_t, mean)
    resid = y_t - pred
    _, r_perp, _ = tangent_decomposition(tang_t, resid, intake_config.ridge)
    r = r_perp[0].detach().cpu().numpy()
    return float(np.sqrt(np.mean(r * r))), r


def _consequence(
    *,
    seed: int,
    regime: str,
    intake_config: R1RS1AConfig,
) -> float:
    alpha = float(REGIME_ALPHA[regime])
    if regime != "C2":
        episode = _compute_episode(
            intake_config,
            seed=seed,
            alpha=alpha,
            amp_scale=1.0,
            freq_hz=FREQ_HZ,
            a0=A0,
        )
        return float(episode["C"])
    backend = DoorModeABackend(
        regime="C2-latch",
        friction=intake_config.friction,
        damping=intake_config.damping,
        seed=seed,
        alpha=0.0,
    )
    try:
        generator = torch.Generator().manual_seed(
            seed + 41_000 + int(hashlib.md5(b"rs2-c2").hexdigest()[:8], 16) % 10_000
        )
        query = _sample_door_points(QUERY_COUNT, "intervention", generator)
        horizon = HORIZON_MACRO * MACRO_STEPS
        scores = []
        for row in query:
            out = backend.forecast_pair(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                horizon=horizon,
                margin=intake_config.joint_limit_margin,
            )
            scores.append(float(out["rmse_nominal"]) - float(out["rmse_oracle"]))
        return float(np.mean(scores)) if scores else math.nan
    finally:
        backend.close()


def _contact_rollout(
    c0: R1RS2C0Config,
    *,
    seed: int,
    repeat: int,
    regime: str,
    script: str,
    scale: float,
) -> dict[str, Any]:
    backend = DoorContactC0Backend(
        c0, seed=int(seed) + 100 * int(repeat), regime=regime
    )
    try:
        backend.prepare()
        result = backend.rollout(script, scale=scale)
        result["prepared_qpos"] = backend._prepared_qpos
        result["lo"] = backend.lo
        result["hi"] = backend.hi
        return result
    finally:
        backend.close()


def _contact_h32(
    c0: R1RS2C0Config,
    *,
    seed: int,
    repeat: int,
    regime: str,
    actions: np.ndarray,
    candidate: dict[str, Any],
    fit_support: dict[str, float],
    config: R1RS2FormalConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    force = _revision_force(
        float(np.asarray(candidate["coefficient"], dtype=np.float64)[2]),
        int(candidate["operator_index"]),
    )
    generator = torch.Generator().manual_seed(
        seed + 59_000 + int(abs(REGIME_ALPHA[regime]) * 10_000)
    )
    queries = _sample_door_points(config.query_count, "intervention", generator)
    scale = np.asarray([1.0, 2.0], dtype=np.float64)
    backend = DoorContactC0Backend(
        c0, seed=int(seed) + 100 * int(repeat), regime=regime
    )
    query_rows: list[dict[str, Any]] = []
    truth_traces: list[dict[str, Any]] = []
    no_traces: list[dict[str, Any]] = []
    rev_traces: list[dict[str, Any]] = []
    try:
        backend.prepare()
        lo, hi = float(backend.lo), float(backend.hi)
        n_steps = int(config.h32)
        action_tail = np.asarray(actions, dtype=np.float64)
        if len(action_tail) > n_steps:
            action_tail = action_tail[-n_steps:]
        for index, state in enumerate(queries):
            q0, v0, _torque = [float(x) for x in state]
            truth = backend.replay_actions(
                action_tail,
                q0=q0,
                v0=v0,
                include_hidden=True,
                revision_fn=None,
                n_steps=n_steps,
            )
            no_revision = backend.replay_actions(
                action_tail,
                q0=q0,
                v0=v0,
                include_hidden=False,
                revision_fn=None,
                n_steps=n_steps,
            )
            revised = backend.replay_actions(
                action_tail,
                q0=q0,
                v0=v0,
                include_hidden=False,
                revision_fn=force,
                n_steps=n_steps,
            )
            truth_traces.append(truth)
            no_traces.append(no_revision)
            rev_traces.append(revised)
            d_modeled = modeled_distance(
                revised["q"], lo, hi, config.joint_limit_margin
            )
            d_fit = fit_box_distance(
                np.asarray(revised["q"], dtype=np.float64),
                np.asarray(revised["qvel"], dtype=np.float64),
                fit_support,
            )
            rmse_no = _terminal_rmse(truth, no_revision, scale)
            rmse_rev = _terminal_rmse(truth, revised, scale)
            gain = (
                float(rmse_no - rmse_rev)
                if math.isfinite(rmse_no) and math.isfinite(rmse_rev)
                else math.nan
            )
            i_exit = float(np.mean(d_modeled > 0.0)) if len(d_modeled) else math.nan
            query_rows.append(
                {
                    "query_index": index,
                    "finite": bool(
                        truth["finite"] and no_revision["finite"] and revised["finite"]
                    ),
                    "D_exit": float(np.max(d_modeled)) if len(d_modeled) else math.nan,
                    "I_exit": i_exit,
                    "D_exit_fit": float(np.max(d_fit)) if len(d_fit) else math.nan,
                    "rmse_h32_no_revision": rmse_no,
                    "rmse_h32_revised": rmse_rev,
                    "gain_h32": gain,
                    "harmful": bool(
                        (
                            not (
                                truth["finite"]
                                and no_revision["finite"]
                                and revised["finite"]
                            )
                        )
                        or (math.isfinite(gain) and gain < 0.0)
                    ),
                }
            )
    finally:
        backend.close()

    gains = [row["gain_h32"] for row in query_rows if math.isfinite(row["gain_h32"])]
    episode = {
        "D_exit": float(np.mean([row["D_exit"] for row in query_rows])),
        "I_exit": float(np.mean([row["I_exit"] for row in query_rows])),
        "rmse_h32_no_revision": float(
            np.nanmean([row["rmse_h32_no_revision"] for row in query_rows])
        ),
        "rmse_h32_revised": float(
            np.nanmean([row["rmse_h32_revised"] for row in query_rows])
        ),
        "gain_h32": float(np.mean(gains)) if gains else math.nan,
        "h32_blind": True,
        "n_harmful_queries": int(sum(row["harmful"] for row in query_rows)),
    }
    arrays = {
        "query_state": queries.detach().cpu().numpy(),
        "truth_q": np.stack(
            [
                np.pad(t["q"], (0, max(0, n_steps - len(t["q"]))), constant_values=np.nan)
                for t in truth_traces
            ]
        )
        if truth_traces
        else np.zeros((0, n_steps)),
        "gain_query": np.asarray([row["gain_h32"] for row in query_rows]),
    }
    return episode, arrays, query_rows


def _aggregate_formal(
    rows: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    config: R1RS2FormalConfig,
) -> dict[str, Any]:
    installed = [row for row in rows if row.get("accepted")]
    c0_rows = [row for row in rows if row.get("regime") == "C0"]
    c0_installs = [row for row in c0_rows if row.get("accepted")]
    tolerate = [row for row in rows if row.get("bucket_initial") == "tolerate"]
    false_t = [row for row in tolerate if row.get("accepted")]
    passivity_violations = int(
        sum(bool(row.get("passivity_violation")) for row in installed)
    )
    monitor_ok = True
    support_flips = 0
    for row in installed:
        i_exit = float(row.get("I_exit", math.nan))
        intensity = row.get("monitor_intensity")
        if row.get("accepted_flipped_by_support"):
            monitor_ok = False
            support_flips += 1
        if math.isfinite(i_exit) and i_exit > 0.0:
            if intensity != "elevated":
                monitor_ok = False
        elif intensity != "normal":
            monitor_ok = False

    gain_rs1c = _seed_mean_gains(rows, config.seeds, "rs1c")
    gain_none = _seed_mean_gains(rows, config.seeds, "none")
    gain_veto = _seed_mean_gains(rows, config.seeds, "veto")
    gain_always = _seed_mean_gains(rows, config.seeds, "always")
    vs_none = float(gain_rs1c["mean"] - gain_none["mean"]) if gain_rs1c["n"] else math.nan
    vs_veto = float(gain_rs1c["mean"] - gain_veto["mean"]) if gain_rs1c["n"] else math.nan

    probes = [row for row in rows if row.get("bucket_initial") == "probe"]
    voi_pos = [row for row in probes if row.get("voi_acquired") and float(row.get("voi_V", 0.0) or 0.0) > 0.0]
    promotions = [row for row in rows if row.get("promoted")]
    promo_seeds = {int(row["seed"]) for row in promotions}

    extra_probe_frac = float(
        np.mean([1.0 if row.get("voi_acquired") else 0.0 for row in rows])
    )
    install_frac = float(np.mean([1.0 if row.get("accepted") else 0.0 for row in rows]))
    always_install_frac = float(
        np.mean([1.0 if row.get("accepted_always") else 0.0 for row in rows])
    )
    e_s = 2.25 * extra_probe_frac
    always_j = {
        str(lam): float(gain_always["mean"] - 0.0015 * e_s - float(lam) * always_install_frac)
        if gain_always["n"]
        else math.nan
        for lam in config.always_lambda_r
    }

    p_c0_install = float(len(c0_installs) / len(c0_rows)) if c0_rows else math.nan
    gates = {
        "c0_specificity": {
            "n_c0": len(c0_rows),
            "n_install": len(c0_installs),
            "rate": p_c0_install,
            "max": 0.02,
            "pass": bool(c0_rows and len(c0_installs) == 0),
        },
        "tolerate_safety": {
            "n_tolerate": len(tolerate),
            "n_false": len(false_t),
            "pass": len(false_t) == 0,
        },
        "passivity_among_installs": {
            "value": passivity_violations,
            "pass": passivity_violations == 0,
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
        "voi_path": {
            "n_initial_probe": len(probes),
            "n_promoted": len(promotions),
            "n_seeds_promoted": len(promo_seeds),
            "n_voi_positive": len(voi_pos),
            "pass": bool(
                len(probes) >= 10
                and len(promotions) >= 1
                and len(promo_seeds) >= 3
            ),
        },
        "monitor_consistency": {
            "support_accept_flips": support_flips,
            "pass": monitor_ok and support_flips == 0,
        },
    }
    go = bool(all(item["pass"] for item in gates.values()))
    return {
        "n_episodes": len(rows),
        "n_tolerate": sum(row.get("bucket_final") == "tolerate" for row in rows),
        "n_probe": sum(row.get("bucket_final") == "probe" for row in rows),
        "n_revise_worthy": sum(bool(row.get("revise_worthy")) for row in rows),
        "n_promoted": len(promotions),
        "n_installed": len(installed),
        "policy_gains": {
            "rs1c": gain_rs1c,
            "none": gain_none,
            "veto": gain_veto,
            "always": gain_always,
        },
        "always_sensitivity": {
            "E": e_s,
            "R": always_install_frac,
            "J": always_j,
        },
        "gates": gates,
        "rs2_go": go,
        "rewrites_rs1b_go": False,
        "rewrites_rs1c_go": False,
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
                "tau_hidden",
                "tau_contact_efc",
                "qfrc_constraint_truth",
                "residual",
                "phase",
            ):
                if key in probe:
                    raw.create_dataset(key, data=probe[key], compression="gzip")
            learner = handle.create_group("learner_visible")
            for key in (
                "q",
                "qvel",
                "qacc",
                "tau_contact_jtf",
                "tau_actuator",
                "tau_nominal_damping",
                "tau_nominal_friction",
                "residual",
                "phase",
            ):
                if key in probe:
                    learner.create_dataset(key, data=probe[key], compression="gzip")
            learner.attrs["contact_force_is_oracle"] = True
            learner.attrs["excludes_tau_hidden"] = True
            learner.attrs["excludes_full_qfrc_constraint"] = True
            decision = handle.create_group("decision")
            decision.attrs["accepted"] = bool(row.get("accepted", False))
            decision.attrs["monitor_intensity"] = str(row.get("monitor_intensity", "none"))
            decision.attrs["support_veto"] = False
            decision.attrs["h32_blind"] = True
            if h32 is not None:
                evaluation = handle.create_group("evaluation")
                for key, value in h32.items():
                    evaluation.create_dataset(key, data=value, compression="gzip")
    finally:
        row["_probe_arrays"] = probe
        if h32 is not None:
            row["_h32_arrays"] = h32


def run_r1_rs2_formal(
    output: str | Path,
    *,
    rs2_c0_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS2FormalConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS2FormalConfig()
    c0_payload = _require_rs2_c0(Path(rs2_c0_summary))
    policy = _require_rs1a5_unlock(Path(rs1a5_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")
    c0_cfg = R1RS2C0Config()
    policy_aggregate = policy["aggregate"]
    c_tol = float(policy_aggregate["gates"]["C_tol"])
    thresholds = {
        float(amplitude): float(cell["threshold"])
        for amplitude, cell in policy_aggregate["cells"].items()
    }
    if smoke:
        jobs = list(SMOKE_JOBS)
        scientific = False
    else:
        jobs = [
            (seed, regime, script, repeat)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for script in cfg.scripts
            for repeat in cfg.repeats
        ]
        scientific = True

    intake_config = R1RS1AConfig()
    revision_config = R1RS1Config()
    rs1c_like = _as_rs1c_like(cfg)
    b_cfg = _as_rs1b_config(rs1c_like)
    rows: list[dict[str, Any]] = []
    query_catalog: list[dict[str, Any]] = []

    for index, (seed, regime, script, repeat) in enumerate(jobs, start=1):
        print(
            f"[RS2 {index}/{len(jobs)}] seed={seed} {regime} {script} r{repeat}",
            flush=True,
        )
        seed_everything(int(seed) + 17 * int(repeat))
        amp_bin = float(FROZEN_SCRIPT_AMP[script])
        contact = _contact_rollout(
            c0_cfg,
            seed=int(seed),
            repeat=int(repeat),
            regime=str(regime),
            script=str(script),
            scale=1.0,
        )
        d0, r_perp = _d0_from_contact(contact["arrays"], intake_config)
        consequence = _consequence(
            seed=int(seed), regime=str(regime), intake_config=intake_config
        )
        threshold = thresholds[amp_bin]
        detected = bool(d0 >= threshold)
        consequential = bool(consequence >= c_tol)
        bucket_initial = _bucket(consequential=consequential, detected=detected)
        bucket = bucket_initial
        promoted = False
        voi_acquired = False
        voi_flip = False
        voi_v = math.nan
        revision_script = str(script)
        revision_scale = 1.0
        revision_amp = amp_bin
        revision_arrays = contact["arrays"]
        revision_actions = contact["actions"]
        revision_d0 = d0

        if bucket == "probe" and amp_bin < cfg.a_star - 1.0e-12:
            extra = _contact_rollout(
                c0_cfg,
                seed=int(seed),
                repeat=int(repeat),
                regime=str(regime),
                script="pull_release",
                scale=float(cfg.adapter_scale),
            )
            extra_d0, extra_r = _d0_from_contact(extra["arrays"], intake_config)
            voi_acquired = True
            star_threshold = thresholds[cfg.a_star]
            voi_flip = bool(extra_d0 >= star_threshold)
            voi_v = (1.0 if voi_flip else 0.0) * float(consequence) - cfg.lambda_cost * (
                cfg.a_star**2
            )
            if voi_flip:
                promoted = True
                bucket = "revise_worthy"
                detected = True
                revision_script = "pull_release"
                revision_scale = float(cfg.adapter_scale)
                revision_amp = cfg.a_star
                revision_arrays = extra["arrays"]
                revision_actions = extra["actions"]
                revision_d0 = extra_d0
                r_perp = extra_r

        revise_worthy = bucket == "revise_worthy"
        row: dict[str, Any] = {
            "seed": int(seed),
            "regime": str(regime),
            "script": str(script),
            "repeat": int(repeat),
            "amp_scale": amp_bin,
            "revision_amp_scale": revision_amp,
            "D0": float(d0),
            "D0_revision": float(revision_d0),
            "C": float(consequence),
            "X_phi": float(contact["X_phi"]),
            "detect_threshold": threshold,
            "C_tol": c_tol,
            "detected_initial": bool(d0 >= threshold),
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
            "accepted": False,
            "accepted_always": False,
            "accepted_veto": False,
            "accepted_flipped_by_support": False,
            "monitor_intensity": "none",
            "_probe_arrays": {**revision_arrays, "r_perp": r_perp},
        }

        candidate = None
        # Always-baseline requires a candidate on every episode.
        mode_a_regime = "C2-latch" if regime == "C2" else regime
        if regime == "C0":
            mode_a_regime = "C0"
        probe = ProbeSpec(
            name=f"RS2-A{revision_amp:.1f}",
            kind="sine",
            amplitude=A0 * float(revision_amp),
            freq_hz=FREQ_HZ,
        )
        proposal = _evaluate_episode(
            revision_config,
            seed=int(seed),
            regime=mode_a_regime,
            probe_name=probe.name,
            device=device,
            scientific=scientific,
            alpha=float(REGIME_ALPHA[regime]),
            probe_spec=probe,
            return_candidate=True,
        )
        candidate = proposal.pop("_candidate")
        local = _local_diagnostics(
            b_cfg,
            seed=int(seed),
            alpha=float(REGIME_ALPHA[regime]),
            candidate=candidate,
            probe_arrays={
                "q": revision_arrays["q"],
                "qvel": revision_arrays["qvel"],
            },
        )
        pipeline_accepted = bool(
            proposal["accepted"]
            and local["effective_damping_pass"]
            and not local["passivity_violation"]
        )
        row.update(local)
        row.update(
            {
                "pipeline_executed": True,
                "pipeline_accepted": bool(proposal["accepted"]),
                "selected_operator": proposal["selected_operator"],
                "alpha_hat": float(proposal["alpha_hat"]),
                "short_utility": float(proposal["utility"]),
            }
        )
        row["accepted_always"] = bool(pipeline_accepted)
        frozen_accept = bool(pipeline_accepted and revise_worthy)
        row["accepted"] = frozen_accept
        if pipeline_accepted:
            blind, arrays, query_rows = _contact_h32(
                c0_cfg,
                seed=int(seed),
                repeat=int(repeat),
                regime=str(regime),
                actions=revision_actions,
                candidate=candidate,
                fit_support=local["fit_support"],
                config=cfg,
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
                        "seed": int(seed),
                        "regime": str(regime),
                        "script": str(script),
                        "accepted": frozen_accept,
                        "episode_I_exit": i_exit,
                    }
                )
        else:
            row["monitor_intensity"] = "none"

        rel = f"seed_{seed}/{regime}/{script}/repeat_{repeat}.hdf5"
        _write_episode_h5(root / rel, row)
        row["path"] = rel
        rows.append(row)

    aggregate = (
        _aggregate_formal(rows, query_catalog, cfg)
        if scientific
        else {
            "rs2_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; confirmatory gates not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS1C"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS2-C0"] = {
        "unlocked": True,
        "passed": True,
        "source": str(Path(rs2_c0_summary)),
    }
    status["R1-RS2-Formal"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs2_go")),
        "smoke_only": smoke,
    }
    slim = [
        _jsonable(
            {k: v for k, v in row.items() if k not in {"_probe_arrays", "_h32_arrays"}}
        )
        for row in rows
    ]
    summary = {
        "stage": "R1-RS2-Formal",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "c0_source": str(Path(rs2_c0_summary)),
        "c0_adapter_scale": FROZEN_ADAPTER_SCALE,
        "script_amp_bins": dict(FROZEN_SCRIPT_AMP),
        "script_manifest": _script_manifest(c0_cfg),
        "frozen_intake": {
            "source": str(Path(rs1a5_summary)),
            "C_tol": c_tol,
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "A_star": cfg.a_star,
            "lambda": cfg.lambda_cost,
            "support_role": "confidence_monitoring_not_veto",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": slim,
        "aggregate": aggregate,
        "rs2_go": bool(aggregate.get("rs2_go")),
        "rs2_c0_pass": True,
        "rs1b_go_unchanged": True,
        "rs1c_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "seed",
        "regime",
        "script",
        "repeat",
        "amp_scale",
        "D0",
        "C",
        "bucket_initial",
        "bucket_final",
        "revise_worthy",
        "promoted",
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
    return summary

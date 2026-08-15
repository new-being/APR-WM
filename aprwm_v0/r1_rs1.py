"""R1-RS1 formal experiment: frozen R0.6 on robosuite Door Mode-A."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch

from .r02 import (
    DRAG_OPERATOR,
    _candidate_design,
    _fit_candidate_posteriors,
    _normalize_fallback_features,
    _predictive_moments,
)
from .r06 import DISSIPATIVE_OPERATORS, constrained_dissipative_map_fit
from .r1_mj import stage_status
from .r1_rs import RS0_NOMINAL_DAMPING, RS0_NOMINAL_FRICTION
from .r1_rs1_backend import DoorModeABackend
from .train import resolve_device, seed_everything
from .v06 import _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import (
    OPERATOR_COMPLEXITY,
    OPERATOR_NAMES,
    operator_library,
    orthogonal_operator_gain,
)
from .v5 import (
    _action_score,
    _fit_base_posterior,
    _update_base_belief,
    _update_candidate_belief,
)
from .v6 import V6Config, _sequential_decision


REGIMES = ("C0", "C1-L", "C1-H", "CNEG", "C2-latch")
FORMAL_SEEDS = (9101, 9111, 9121, 9131, 9141)
SMOKE_SEED = 8901
PROBES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
PREREG_PATH = "REPORT/R1_RS1_PREREG.md"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    kind: str
    amplitude: float
    freq_hz: float = 0.25
    freq2_hz: float = 0.25
    phase: float = 0.0
    hold_s: float = 0.5


PROBE_BANK: dict[str, ProbeSpec] = {
    "P1": ProbeSpec("P1", "sine", 0.12, freq_hz=0.20, phase=0.0),
    "P2": ProbeSpec("P2", "sine", 0.12, freq_hz=0.28, phase=1.047),
    "P3": ProbeSpec("P3", "sine", 0.12, freq_hz=0.55, phase=0.0),
    "P4": ProbeSpec("P4", "sine", 0.12, freq_hz=0.80, phase=0.7),
    "P5": ProbeSpec("P5", "chirp_up", 0.12, freq_hz=0.10, freq2_hz=0.90),
    "P6": ProbeSpec("P6", "chirp_down", 0.12, freq_hz=0.90, freq2_hz=0.10),
    "P7": ProbeSpec("P7", "piecewise", 0.10, hold_s=0.80),
    "P8": ProbeSpec("P8", "piecewise", 0.12, hold_s=0.25),
}


@dataclass
class R1RS1Config:
    passive_context: int = 8
    discovery_count: int = 16
    action_candidates: int = 32
    selection_probes: int = 3
    validation_max: int = 32
    query_count: int = 8
    short_queries: int = 8
    proposal_topk: int = 3
    force_observation_noise: float = 0.002
    posterior_noise_floor: float = 0.01
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    proposal_complexity_price: float = 2.0e-4
    proposal_temperature: float = 0.10
    trigger_noise_multiplier: float = 1.35
    parameter_prior_weight: float = 0.05
    utility_weights: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    rollout_margin: float = 1.0e-4
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = REGIMES
    probes: tuple[str, ...] = PROBES
    duration_s: float = 10.0
    timestep: float = 0.002
    macro_steps: int = 4
    short_horizons: tuple[int, ...] = (2, 4, 8)
    horizons: tuple[int, ...] = (2, 4, 8, 32)
    friction: float = RS0_NOMINAL_FRICTION
    damping: float = RS0_NOMINAL_DAMPING
    joint_limit_margin: float = 0.05
    c0_false_revision_max: float = 0.01
    c1_exact_recovery_min: float = 0.85
    c1_coefficient_rel_err_max: float = 0.10
    c1_coefficient_coverage_min: float = 0.85
    cneg_accepted_max: int = 0
    c2_conditional_unknown_min: float = 0.90
    c2_conditional_wrong_max: float = 0.05
    accepted_stability_min: float = 0.99


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_rs0_unlock(rs0_summary: Path) -> dict[str, Any]:
    if not rs0_summary.is_file():
        raise RuntimeError("R1-RS1 locked until R1-RS0 passes; missing RS0 summary")
    payload = json.loads(rs0_summary.read_text(encoding="utf-8"))
    if not payload.get("rs0_passed"):
        raise RuntimeError("R1-RS1 locked: rs0_passed is false")
    return payload


def _torque_series(spec: ProbeSpec, n_steps: int, dt: float, seed: int) -> np.ndarray:
    t = np.arange(n_steps, dtype=np.float64) * dt
    if spec.kind == "sine":
        return spec.amplitude * np.sin(2.0 * np.pi * spec.freq_hz * t + spec.phase)
    if spec.kind in {"chirp_up", "chirp_down"}:
        inst = spec.freq_hz + (spec.freq2_hz - spec.freq_hz) * (t / max(t[-1], 1e-12))
        phase = 2.0 * np.pi * np.cumsum(inst) * dt
        return spec.amplitude * np.sin(phase)
    if spec.kind == "piecewise":
        rng = np.random.default_rng(seed)
        hold = max(1, int(round(spec.hold_s / dt)))
        values: list[float] = []
        while len(values) < n_steps:
            values.extend(
                [float(rng.uniform(-spec.amplitude, spec.amplitude))] * hold
            )
        return np.asarray(values[:n_steps], dtype=np.float64)
    raise ValueError(spec.kind)


def _sample_door_points(
    count: int, regime: str, generator: torch.Generator
) -> torch.Tensor:
    if regime == "passive":
        q = 0.12 + 0.16 * torch.rand(count, generator=generator)
        velocity = 0.25 * torch.rand(count, generator=generator) - 0.125
        torque = 0.08 * torch.rand(count, generator=generator) - 0.04
    elif regime == "discovery":
        q = 0.08 + 0.24 * torch.rand(count, generator=generator)
        velocity = 0.90 * torch.rand(count, generator=generator) - 0.45
        torque = 0.12 * torch.rand(count, generator=generator) - 0.06
    else:
        q = 0.08 + 0.24 * torch.rand(count, generator=generator)
        velocity = 1.10 * torch.rand(count, generator=generator) - 0.55
        torque = 0.14 * torch.rand(count, generator=generator) - 0.07
    return torch.stack((q, velocity, torque), dim=-1)


def _observe(
    backend: DoorModeABackend, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    physical, tangent, target, supports = [], [], [], []
    for row in state:
        residual, density_tangent, _, _, support = backend.force_sample(
            float(row[0]), float(row[1]), float(row[2])
        )
        physical.append((float(row[0]), float(row[1])))
        tangent.append((density_tangent, float(row[1])))
        target.append(residual)
        supports.append(support)
    return (
        torch.tensor(physical, dtype=torch.float32),
        torch.tensor(tangent, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
        supports,
    )


def _truth_rollouts(
    backend: DoorModeABackend,
    state: torch.Tensor,
    horizons: tuple[int, ...],
    macro_steps: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    truth, validity = [], []
    for horizon in horizons:
        rows_t, rows_v = [], []
        for row in state:
            q, v, valid, _ = backend.transition_with_validity(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                horizon=horizon * macro_steps,
                residual_force=0.0,
                margin=margin,
                include_hidden=True,
            )
            rows_t.append((q, v))
            rows_v.append(valid)
        truth.append(torch.tensor(rows_t, dtype=torch.float32))
        validity.append(torch.tensor(rows_v, dtype=torch.bool))
    return torch.stack(truth, dim=1), torch.stack(validity, dim=1)


def _selected_force(
    tangent: torch.Tensor,
    physical: torch.Tensor,
    coefficient: torch.Tensor,
    operator_index: torch.Tensor,
) -> torch.Tensor:
    operator = operator_library(physical).gather(
        -1,
        operator_index[:, None, None].expand(-1, physical.shape[1], 1),
    ).squeeze(-1)
    return (
        torch.einsum("bni,bi->bn", tangent, coefficient[:, :2])
        + coefficient[:, 2, None] * operator
    )


def _rmse_masked(
    prediction: torch.Tensor, truth: torch.Tensor, validity: torch.Tensor
) -> float:
    scale = np.asarray([1.0, 2.0], dtype=np.float64)
    pred = prediction.detach().cpu().numpy()
    tru = truth.detach().cpu().numpy()
    valid = validity.detach().cpu().numpy().astype(bool)
    if pred.ndim == 2:
        err = (pred - tru) / scale
        if not valid.any():
            return float("nan")
        return float(np.sqrt(np.mean(err[valid] ** 2)))
    # (Q, H, 2)
    total = []
    for h in range(pred.shape[1]):
        mask = valid[:, h]
        if not mask.any():
            continue
        err = (pred[mask, h] - tru[mask, h]) / scale
        total.append(np.mean(err**2))
    if not total:
        return float("nan")
    return float(np.sqrt(np.mean(total)))


def _evaluate_episode(
    config: R1RS1Config,
    *,
    seed: int,
    regime: str,
    probe_name: str,
    device: torch.device,
    scientific: bool,
    alpha: float | None = None,
    probe_spec: ProbeSpec | None = None,
    return_candidate: bool = False,
) -> dict[str, Any]:
    mix = int(hashlib.md5(f"{regime}:{probe_name}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + 17_000 + (mix % 10_000))
    backend = DoorModeABackend(
        regime=regime,
        friction=config.friction,
        damping=config.damping,
        seed=seed,
        alpha=alpha,
    )
    try:
        n_steps = int(round(config.duration_s / config.timestep))
        torques = _torque_series(
            probe_spec or PROBE_BANK[probe_name], n_steps, config.timestep, seed=seed + 91
        )
        probe = backend.rollout_probe(torques, margin=config.joint_limit_margin)

        pools = {}
        support_counts = {"modeled": 0, "boundary": 0, "unsupported": 0}
        for prefix, count, sampling in (
            ("passive", config.passive_context, "passive"),
            ("discovery", config.discovery_count, "discovery"),
            ("action", config.action_candidates, "diagnostic"),
            ("validation", config.validation_max, "validation"),
        ):
            state = _sample_door_points(count, sampling, generator)
            physical, tangent, target, supports = _observe(backend, state)
            for item in supports:
                support_counts[item] += 1
            pools[f"{prefix}_physical"] = physical
            pools[f"{prefix}_tangent"] = tangent
            pools[f"{prefix}_y"] = target + config.force_observation_noise * torch.randn(
                target.shape, generator=generator
            )

        short_state = _sample_door_points(config.short_queries, "intervention", generator)
        query_state = _sample_door_points(config.query_count, "intervention", generator)
        short_truth, short_validity = _truth_rollouts(
            backend, short_state, config.short_horizons, config.macro_steps,
            config.joint_limit_margin,
        )
        rollout_truth, rollout_validity = _truth_rollouts(
            backend, query_state, config.horizons, config.macro_steps,
            config.joint_limit_margin,
        )

        # ---- frozen R0.6 decision (single-episode batch dim = 1) ----
        def unsqueeze(name: str) -> torch.Tensor:
            return pools[name].unsqueeze(0).to(device)

        passive_tangent = unsqueeze("passive_tangent")
        passive_y = unsqueeze("passive_y")
        discovery_tangent = unsqueeze("discovery_tangent")
        discovery_physical = unsqueeze("discovery_physical")
        discovery_y = unsqueeze("discovery_y")
        action_tangent = unsqueeze("action_tangent")
        action_physical = unsqueeze("action_physical")
        action_y = unsqueeze("action_y")
        validation_tangent = unsqueeze("validation_tangent")
        validation_physical = unsqueeze("validation_physical")
        validation_y = unsqueeze("validation_y")
        passive_physical = unsqueeze("passive_physical")

        initial_theta = _ridge_fit(passive_tangent, passive_y, config.ridge)
        discovery_residual = discovery_y - torch.einsum(
            "bni,bi->bn", discovery_tangent, initial_theta
        )
        _, residual_orthogonal, _ = tangent_decomposition(
            discovery_tangent, discovery_residual, config.ridge
        )
        triggered = residual_orthogonal.square().mean(dim=-1).sqrt() > (
            config.trigger_noise_multiplier * config.force_observation_noise
        )
        gain = orthogonal_operator_gain(
            discovery_tangent,
            residual_orthogonal,
            operator_library(discovery_physical),
            ridge=config.ridge,
        )
        complexity = torch.tensor(OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device)
        proposal_score = gain - config.proposal_complexity_price * complexity
        proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
        train_tangent = torch.cat((passive_tangent, discovery_tangent), dim=1)
        train_physical = torch.cat((passive_physical, discovery_physical), dim=1)
        train_y = torch.cat((passive_y, discovery_y), dim=1)
        v6 = V6Config(
            episodes=1,
            passive_context=config.passive_context,
            discovery_count=config.discovery_count,
            action_candidates=config.action_candidates,
            selection_probes=config.selection_probes,
            validation_max=config.validation_max,
            proposal_topk=config.proposal_topk,
            posterior_noise_floor=config.posterior_noise_floor,
            ridge=config.ridge,
            residual_ridge=config.residual_ridge,
        )
        means, covariances = _fit_candidate_posteriors(
            train_tangent, train_physical, train_y, proposed, config
        )
        base_mean, base_covariance = _fit_base_posterior(
            train_tangent, train_y, config.force_observation_noise, v6
        )
        parameter_prior_mean, parameter_prior_covariance = _fit_base_posterior(
            passive_tangent, passive_y, config.force_observation_noise, v6
        )
        log_probabilities = torch.log_softmax(
            proposal_score.gather(1, proposed) / config.proposal_temperature, dim=-1
        )
        observation_variance = max(
            config.force_observation_noise, config.posterior_noise_floor
        ) ** 2
        used = torch.zeros_like(action_y, dtype=torch.bool)
        selected_action_tangent = []
        selected_action_physical = []
        selected_action_y = []
        correct_position = torch.full((1,), -1, dtype=torch.long, device=device)
        for _ in range(config.selection_probes):
            pool_means, pool_variances = _predictive_moments(
                action_tangent, action_physical, proposed, means, covariances
            )
            score = _action_score(
                "weighted",
                pool_means,
                pool_variances,
                log_probabilities.exp(),
                observation_variance,
                correct_position,
                False,
            ).masked_fill(used, float("-inf"))
            action_index = score.argmax(dim=-1)
            at = action_tangent.gather(
                1, action_index[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            ap = action_physical.gather(
                1, action_index[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            ay = action_y.gather(1, action_index.unsqueeze(-1)).squeeze(-1)
            selected_action_tangent.append(at)
            selected_action_physical.append(ap)
            selected_action_y.append(ay)
            design = _candidate_design(at.unsqueeze(1), ap.unsqueeze(1), proposed).squeeze(1)
            means, covariances, log_probabilities = _update_candidate_belief(
                means, covariances, log_probabilities, design, ay,
                observation_variance, triggered,
            )
            base_mean, base_covariance = _update_base_belief(
                base_mean, base_covariance, at, ay, observation_variance, triggered
            )
            used[0, int(action_index[0])] = True

        order = log_probabilities.topk(2, dim=-1).indices
        selected_position = order[:, 0]
        competitor_position = order[:, 1]
        selected_operator = proposed.gather(1, selected_position[:, None]).squeeze(-1)
        evidence_tangent = torch.cat(
            (train_tangent, torch.stack(selected_action_tangent, dim=1)), dim=1
        )
        evidence_physical = torch.cat(
            (train_physical, torch.stack(selected_action_physical, dim=1)), dim=1
        )
        evidence_y = torch.cat(
            (train_y, torch.stack(selected_action_y, dim=1)), dim=1
        )
        constrained_coefficient, _ = constrained_dissipative_map_fit(
            evidence_tangent,
            evidence_physical,
            evidence_y,
            selected_operator,
            parameter_prior_mean,
            parameter_prior_covariance,
            config,
        )
        dissipative_tensor = torch.tensor(
            DISSIPATIVE_OPERATORS, dtype=selected_operator.dtype, device=device
        )
        feasible = (selected_operator[:, None] == dissipative_tensor).any(dim=-1)
        feasible &= constrained_coefficient[:, 2] <= 0.0

        validation_design = _candidate_design(
            validation_tangent, validation_physical, proposed
        )
        validation_all = torch.einsum("bank,bnk->ban", validation_design, means)
        competitor_validation = validation_all.gather(
            -1, competitor_position[:, None, None].expand(-1, config.validation_max, 1)
        ).squeeze(-1)
        constrained_validation = _selected_force(
            validation_tangent, validation_physical, constrained_coefficient, selected_operator
        )
        base_validation = torch.einsum("bni,bi->bn", validation_tangent, base_mean)
        incumbent_sq = (validation_y - base_validation).square()
        competitor_sq = (validation_y - competitor_validation).square()
        constrained_sq = (validation_y - constrained_validation).square()
        constrained_force_accepted, _, _, _ = _sequential_decision(
            incumbent_sq,
            competitor_sq,
            constrained_sq,
            complexity[selected_operator],
            config.force_observation_noise ** 2,
            v6,
            method="bf",
        )
        constrained_force_accepted &= triggered & feasible

        evidence_base = torch.einsum("bni,bi->bn", evidence_tangent, base_mean)
        evidence_residual = evidence_y - evidence_base
        normalized_evidence = _normalize_fallback_features(evidence_physical)

        def physics_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
            return torch.zeros(tangent.shape[:2], dtype=tangent.dtype, device=device)

        def fallback_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
            base = torch.einsum("bqi,bi->bq", tangent, base_mean)
            return base + _residual_predict(
                normalized_evidence,
                evidence_residual,
                _normalize_fallback_features(physical),
                config.residual_ridge,
            ).clamp(-0.4, 0.4)

        def constrained_force(tangent: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
            return _selected_force(
                tangent, physical, constrained_coefficient, selected_operator
            )

        # Short-horizon utility (acceptance-visible).
        def integrate(state, horizons, force_model, include_hidden_truth: bool):
            # Build per-query predictions by Door transitions.
            preds = []
            stables = []
            for horizon in horizons:
                rows = []
                ok_rows = []
                physical = state[:, :2].unsqueeze(0).to(device)
                # approximate tangent for force_model call
                tangent = torch.stack(
                    (torch.ones(state.shape[0]), state[:, 1]), dim=-1
                ).unsqueeze(0).to(device)
                residual = force_model(tangent, physical).squeeze(0)
                for index in range(state.shape[0]):
                    q, v, ok, _ = backend.transition_with_validity(
                        float(state[index, 0]),
                        float(state[index, 1]),
                        float(state[index, 2]),
                        horizon=horizon * config.macro_steps,
                        residual_force=float(residual[index]),
                        margin=config.joint_limit_margin,
                        include_hidden=include_hidden_truth,
                    )
                    rows.append((q, v))
                    ok_rows.append(ok)
                preds.append(torch.tensor(rows, dtype=torch.float32))
                stables.append(torch.tensor(ok_rows, dtype=torch.bool))
            return torch.stack(preds, dim=1), torch.stack(stables, dim=-1)

        # Acceptance uses models WITHOUT re-injecting hidden truth on learner side.
        physics_short, _ = integrate(short_state, config.short_horizons, physics_force, False)
        fallback_short, _ = integrate(short_state, config.short_horizons, fallback_force, False)
        constrained_short, constrained_short_stable = integrate(
            short_state, config.short_horizons, constrained_force, False
        )
        # Compare against truth that DID include hidden dynamics.
        physics_short_rmse = torch.tensor([
            _rmse_masked(physics_short[:, h], short_truth[:, h], short_validity[:, h])
            for h in range(len(config.short_horizons))
        ])
        fallback_short_rmse = torch.tensor([
            _rmse_masked(fallback_short[:, h], short_truth[:, h], short_validity[:, h])
            for h in range(len(config.short_horizons))
        ])
        constrained_short_rmse = torch.tensor([
            _rmse_masked(constrained_short[:, h], short_truth[:, h], short_validity[:, h])
            for h in range(len(config.short_horizons))
        ])
        weights = torch.tensor(config.utility_weights)
        utility = ((fallback_short_rmse - constrained_short_rmse) * weights).sum()
        short_safe = bool(constrained_short_stable.all())
        passivity_only = bool(triggered[0] and feasible[0])
        accepted = bool(
            passivity_only and short_safe and float(utility) > 0.0
        )
        # Freeze decision before H32.
        decision_accepted = accepted

        # H32 blind evaluation (and full horizon set) AFTER freeze.
        physics_final, physics_stable = integrate(
            query_state, config.horizons, physics_force, False
        )
        fallback_final, fallback_stable = integrate(
            query_state, config.horizons, fallback_force, False
        )
        constrained_final, constrained_stable = integrate(
            query_state, config.horizons, constrained_force, False
        )
        h_index = {h: i for i, h in enumerate(config.horizons)}
        h32 = h_index[32]
        no_rev = fallback_final if True else physics_final
        chosen = constrained_final if decision_accepted else no_rev
        chosen_stable = constrained_stable if decision_accepted else fallback_stable
        rmse_no = _rmse_masked(
            no_rev[:, h32], rollout_truth[:, h32], rollout_validity[:, h32]
        )
        rmse_rev = _rmse_masked(
            chosen[:, h32], rollout_truth[:, h32], rollout_validity[:, h32]
        )
        gain_h32 = float(rmse_no - rmse_rev) if np.isfinite(rmse_no) and np.isfinite(rmse_rev) else 0.0
        stable_h32 = bool(chosen_stable[:, h32].all()) if decision_accepted else True

        alpha_hat = float(constrained_coefficient[0, 2])
        op_name = OPERATOR_NAMES[int(selected_operator[0])]
        truth_alpha = float(backend.alpha)
        exact = (
            regime in {"C1-L", "C1-H"}
            and decision_accepted
            and int(selected_operator[0]) == DRAG_OPERATOR
            and alpha_hat < 0.0
        )
        rel_err = (
            abs(alpha_hat - truth_alpha) / max(abs(truth_alpha), 1e-12)
            if regime in {"C1-L", "C1-H"}
            else float("nan")
        )
        top3 = [OPERATOR_NAMES[int(i)] for i in proposed[0].tolist()]
        truth_in_top3 = ("abs_v_v" in top3) if regime in {"C1-L", "C1-H", "CNEG"} else False

        # C2 diagnostics
        evaluable = support_counts["modeled"] > 0
        unknown_decision = (not decision_accepted) or (op_name != "abs_v_v")
        wrong_explicit = decision_accepted and regime == "C2-latch"
        cneg_accepted_active = (
            regime == "CNEG"
            and decision_accepted
            and int(selected_operator[0]) == DRAG_OPERATOR
            and alpha_hat > 0.0
        )

        funnel = {
            "total": 1,
            "evaluable": int(evaluable),
            "triggered": int(bool(triggered[0])),
            "truth_in_top3": int(truth_in_top3),
            "correctly_selected": int(
                regime in {"C1-L", "C1-H"} and int(selected_operator[0]) == DRAG_OPERATOR
            ),
            "statistically_force_accepted": int(bool(constrained_force_accepted[0])),
            "physically_admissible": int(passivity_only),
            "utility_accepted": int(decision_accepted),
            "h32_useful": int(decision_accepted and gain_h32 > 0.0 and stable_h32),
        }

        result = {
            "seed": seed,
            "regime": regime,
            "probe": probe_name,
            "scientific": scientific,
            "triggered": bool(triggered[0]),
            "selected_operator": op_name,
            "alpha_hat": alpha_hat,
            "truth_alpha": truth_alpha,
            "feasible": passivity_only,
            "accepted": decision_accepted,
            "false_revision": bool(regime == "C0" and decision_accepted),
            "exact_recovery": bool(exact),
            "coefficient_rel_err": rel_err,
            "cneg_accepted_active": bool(cneg_accepted_active),
            "c2_unknown": bool(regime == "C2-latch" and unknown_decision),
            "c2_wrong_explicit": bool(wrong_explicit),
            "support_counts": support_counts,
            "p_modeled": support_counts["modeled"] / max(sum(support_counts.values()), 1),
            "p_boundary": support_counts["boundary"] / max(sum(support_counts.values()), 1),
            "p_unsupported": support_counts["unsupported"] / max(sum(support_counts.values()), 1),
            "rmse_h32_no_revision": rmse_no,
            "rmse_h32_r06": rmse_rev,
            "gain_h32": gain_h32,
            "stable_h32": stable_h32,
            "utility": float(utility),
            "top3": top3,
            "funnel": funnel,
            "probe_arrays": probe,
        }
        if return_candidate:
            result["_candidate"] = {
                "coefficient": constrained_coefficient[0].detach().cpu().numpy(),
                "operator_index": int(selected_operator[0]),
                "fit_physical": evidence_physical[0].detach().cpu().numpy(),
                "force_accepted": bool(constrained_force_accepted[0]),
                "short_safe": bool(short_safe),
                "passivity_utility_accepted": bool(decision_accepted),
            }
        return result
    finally:
        backend.close()


def _write_episode_h5(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = row["probe_arrays"]
    meta = {k: v for k, v in row.items() if k != "probe_arrays"}
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["json"] = json.dumps(meta, sort_keys=True, default=str)
        raw = handle.create_group("raw_truth")
        learner = handle.create_group("learner_visible")
        for key in ("q", "qvel", "qacc", "tau_command", "tau_hidden", "residual", "validity"):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        for key in ("q", "qvel", "qacc", "tau_command", "residual", "validity"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_tau_hidden"] = True
        learner.attrs["excludes_truth_qfrc_passive"] = True
        decision = handle.create_group("decision")
        decision.attrs["accepted"] = row["accepted"]
        decision.attrs["triggered"] = row["triggered"]
        decision.attrs["operator"] = row["selected_operator"]
        decision.attrs["h32_blind"] = True
        evaluation = handle.create_group("evaluation")
        evaluation.attrs["gain_h32"] = row["gain_h32"]
        evaluation.attrs["stable_h32"] = row["stable_h32"]


def _mean_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": n}
    se = float(arr.std(ddof=1) / math.sqrt(n))
    # t_{0.975, n-1}; for n=5, t4≈2.776
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    return {
        "mean": mean,
        "ci95_low": mean - tcrit * se,
        "ci95_high": mean + tcrit * se,
        "n": n,
    }


def _aggregate(rows: list[dict[str, Any]], config: R1RS1Config) -> dict[str, Any]:
    def subset(*regimes: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["regime"] in regimes]

    c0 = subset("C0")
    c1 = subset("C1-L", "C1-H")
    c1l = subset("C1-L")
    c1h = subset("C1-H")
    cneg = subset("CNEG")
    c2 = subset("C2-latch")

    false_revision = float(np.mean([r["false_revision"] for r in c0])) if c0 else 0.0
    exact = float(np.mean([r["exact_recovery"] for r in c1])) if c1 else 0.0
    exact_l = float(np.mean([r["exact_recovery"] for r in c1l])) if c1l else 0.0
    exact_h = float(np.mean([r["exact_recovery"] for r in c1h])) if c1h else 0.0
    correct_accepted = [r for r in c1 if r["exact_recovery"]]
    coef_ok = (
        float(np.mean([
            r["coefficient_rel_err"] <= config.c1_coefficient_rel_err_max
            for r in correct_accepted
        ]))
        if correct_accepted else 0.0
    )
    cneg_accepted = int(sum(r["cneg_accepted_active"] for r in cneg))
    cneg_trigger = float(np.mean([r["triggered"] for r in cneg])) if cneg else 0.0

    c2_trig_eval = [r for r in c2 if r["triggered"] and r["p_modeled"] > 0]
    c2_cond_unknown = (
        float(np.mean([r["c2_unknown"] for r in c2_trig_eval])) if c2_trig_eval else float("nan")
    )
    c2_cond_wrong = (
        float(np.mean([r["c2_wrong_explicit"] for r in c2_trig_eval])) if c2_trig_eval else float("nan")
    )
    c2_trigger_coverage = float(np.mean([r["triggered"] for r in c2])) if c2 else 0.0

    accepted = [r for r in rows if r["accepted"]]
    stability = (
        float(np.mean([r["stable_h32"] for r in accepted])) if accepted else 1.0
    )

    # Seed-level paired gain on C1 only.
    seed_gains = []
    for seed in sorted({r["seed"] for r in c1}):
        vals = [r["gain_h32"] for r in c1 if r["seed"] == seed]
        seed_gains.append(float(np.mean(vals)))
    gain_stats = _mean_ci(seed_gains)

    gates = {
        "c0_false_revision": {
            "value": false_revision,
            "max": config.c0_false_revision_max,
            "pass": false_revision <= config.c0_false_revision_max,
        },
        "c1_exact_recovery": {
            "value": exact,
            "c1_l": exact_l,
            "c1_h": exact_h,
            "min": config.c1_exact_recovery_min,
            "pass": exact >= config.c1_exact_recovery_min,
        },
        "c1_coefficient_coverage": {
            "value": coef_ok,
            "min": config.c1_coefficient_coverage_min,
            "pass": coef_ok >= config.c1_coefficient_coverage_min,
        },
        "cneg_accepted_active": {
            "value": cneg_accepted,
            "trigger_rate": cneg_trigger,
            "max": config.cneg_accepted_max,
            "pass": cneg_accepted <= config.cneg_accepted_max,
        },
        "c2_conditional_unknown": {
            "value": c2_cond_unknown,
            "trigger_coverage": c2_trigger_coverage,
            "n_triggered_evaluable": len(c2_trig_eval),
            "min": config.c2_conditional_unknown_min,
            "pass": (
                (not math.isnan(c2_cond_unknown))
                and c2_cond_unknown >= config.c2_conditional_unknown_min
            ),
        },
        "c2_conditional_wrong": {
            "value": c2_cond_wrong,
            "max": config.c2_conditional_wrong_max,
            "pass": (
                (not math.isnan(c2_cond_wrong))
                and c2_cond_wrong <= config.c2_conditional_wrong_max
            ),
        },
        "accepted_h32_stability": {
            "value": stability,
            "min": config.accepted_stability_min,
            "pass": stability >= config.accepted_stability_min,
        },
        "c1_seed_h32_gain": {
            **gain_stats,
            "pass": gain_stats["mean"] > 0.0,
        },
    }
    rs1_go = all(item["pass"] for item in gates.values())
    return {
        "gates": gates,
        "rs1_go": rs1_go,
        "n_episodes": len(rows),
        "acceptance_rate": float(np.mean([r["accepted"] for r in rows])) if rows else 0.0,
    }


def run_r1_rs1(
    output: str | Path,
    *,
    rs0_summary: str | Path,
    config: R1RS1Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS1Config()
    _require_rs0_unlock(Path(rs0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")

    if smoke:
        jobs = [(SMOKE_SEED, regime, "P1") for regime in cfg.regimes]
        scientific = False
        cfg = R1RS1Config(
            **{
                **asdict(cfg),
                "duration_s": 2.0,
                "discovery_count": 8,
                "action_candidates": 12,
                "validation_max": 12,
                "query_count": 4,
                "short_queries": 4,
            }
        )
    else:
        jobs = [
            (seed, regime, probe)
            for seed in cfg.seeds
            for regime in cfg.regimes
            for probe in cfg.probes
        ]
        scientific = True

    rows: list[dict[str, Any]] = []
    for index, (seed, regime, probe) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] seed={seed} regime={regime} probe={probe}", flush=True)
        seed_everything(seed)
        row = _evaluate_episode(
            cfg, seed=seed, regime=regime, probe_name=probe,
            device=device, scientific=scientific,
        )
        rel = f"seed_{seed}/{regime}/{probe}.hdf5"
        _write_episode_h5(root / rel, row)
        slim = {k: v for k, v in row.items() if k != "probe_arrays"}
        slim["path"] = rel
        rows.append(slim)

    aggregate = _aggregate(rows, cfg) if scientific else {
        "gates": None,
        "rs1_go": False,
        "n_episodes": len(rows),
        "note": "plumbing smoke only; scientific gates not evaluated",
    }
    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {"unlocked": True, "passed": True}
    status["R1-RS1"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs1_go")),
        "smoke_only": smoke,
    }
    if aggregate.get("rs1_go"):
        status["R1-RS2"] = {"locked": False, "unlocked": True}

    summary = {
        "stage": "R1-RS1",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "config": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
        },
        "episodes": rows,
        "aggregate": aggregate,
        "rs1_go": bool(aggregate.get("rs1_go")),
        "unlocks_rs2": bool(aggregate.get("rs1_go")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    _write_csv(
        root / "episodes.csv",
        [
            {
                "seed": r["seed"],
                "regime": r["regime"],
                "probe": r["probe"],
                "triggered": int(r["triggered"]),
                "accepted": int(r["accepted"]),
                "operator": r["selected_operator"],
                "alpha_hat": r["alpha_hat"],
                "exact_recovery": int(r["exact_recovery"]),
                "gain_h32": r["gain_h32"],
                "stable_h32": int(r["stable_h32"]),
                "false_revision": int(r["false_revision"]),
                "cneg_accepted_active": int(r["cneg_accepted_active"]),
            }
            for r in rows
        ],
    )
    return summary

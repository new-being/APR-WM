from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from .train import resolve_device, seed_everything
from .v05 import _bootstrap_ci
from .v06 import _binary_auroc, _write_csv, _write_json
from .v3 import _residual_predict, _ridge_fit, tangent_decomposition
from .v4 import OPERATOR_COMPLEXITY, operator_library, orthogonal_operator_gain
from .v5 import (
    _action_score,
    _candidate_design,
    _fit_base_posterior,
    _fit_posteriors,
    _predictive_moments,
    _update_base_belief,
    _update_candidate_belief,
)
from .v6 import V6Config, _sequential_decision


REGIME_NAMES = ("c0_parameter", "c1_drag", "c2_hysteresis")
DRAG_OPERATOR = 4  # |v|v in the frozen V4 operator library.


@dataclass
class V6R0Config:
    seeds: tuple[int, ...] = (13, 23, 33)
    episodes_per_regime: int = 24
    passive_context: int = 8
    discovery_count: int = 16
    action_candidates: int = 32
    selection_probes: int = 3
    validation_max: int = 32
    query_count: int = 8
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    proposal_topk: int = 3
    observation_noise: float = 0.01
    target_scale: float = 20.0
    physics_dt: float = 1.0 / 250.0
    macro_steps: int = 4
    damping_min: float = 0.025
    damping_max: float = 0.075
    density_min: float = 110.0
    density_max: float = 230.0
    drag_strength: float = 0.12
    hysteresis_strength: float = 0.10
    ridge: float = 1.0e-5
    residual_ridge: float = 2.0e-3
    proposal_complexity_price: float = 2.0e-4
    proposal_temperature: float = 0.10
    posterior_noise_floor: float = 0.01


class ArticulationState(Protocol):
    def get_qpos(self) -> Any: ...

    def get_qvel(self) -> Any: ...


@dataclass(frozen=True)
class LaptopLocalState:
    q: float
    qvel: float
    applied_torque: float

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor((self.q, self.qvel, self.applied_torque))


class LaptopStateAdapter:
    """Oracle-state adapter shared by the proxy and official RoboTwin actor."""

    @staticmethod
    def extract(actor: ArticulationState, applied_torque: float) -> LaptopLocalState:
        qpos = actor.get_qpos()
        qvel = actor.get_qvel()
        if len(qpos) != 1 or len(qvel) != 1:
            raise ValueError("Laptop adapter expects exactly one active hinge DOF")
        return LaptopLocalState(float(qpos[0]), float(qvel[0]), float(applied_torque))

    @staticmethod
    def model_features(state: LaptopLocalState) -> torch.Tensor:
        # The robotics bridge must preserve physics closure.  Any projection
        # into the frozen two-dimensional operator library happens only after
        # explicit dynamics have converted this state to a force residual.
        return state.as_tensor()


def residual_assimilation_ratio(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("Residual usage before revision must be positive")
    return 1.0 - after / before


def counterfactual_branch_manifest(
    *,
    task: str,
    seeds: tuple[int, ...],
    windows_per_seed: int,
    branches_per_window: int,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for window in range(windows_per_seed):
            for branch in range(branches_per_window):
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "window": window,
                        "branch": branch,
                        "replay_prefix": True,
                        "horizons": list(horizons),
                    }
                )
    return rows


def inspect_robotwin(
    repo: str | Path = "/home/dong/Projects/RoboTwin",
    python: str | Path = "/home/dong/miniconda3/envs/RoboTwin/bin/python",
) -> dict[str, Any]:
    root = Path(repo)
    python_path = Path(python)
    task_paths = {
        "open_laptop": root / "envs/open_laptop.py",
        "beat_block_hammer": root / "envs/beat_block_hammer.py",
        "handover_block": root / "envs/handover_block.py",
    }
    hammer_text = (
        task_paths["beat_block_hammer"].read_text(encoding="utf-8")
        if task_paths["beat_block_hammer"].exists()
        else ""
    )
    import_check: dict[str, bool] = {}
    if python_path.exists():
        script = (
            "import importlib.util,json;"
            "print(json.dumps({n:importlib.util.find_spec(n) is not None "
            "for n in ['sapien','gymnasium','h5py','mplib']}))"
        )
        completed = subprocess.run(
            (str(python_path), "-c", script),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            import_check = json.loads(completed.stdout.strip())
    xpolicy = root / "XPolicyLab"
    assets = root / "assets"
    report = {
        "repo": str(root),
        "repo_present": root.exists(),
        "python": str(python_path),
        "python_present": python_path.exists(),
        "imports": import_check,
        "tasks": {name: path.exists() for name, path in task_paths.items()},
        "xpolicylab_initialized": (xpolicy / "pyproject.toml").exists()
        or (xpolicy / "setup.py").exists(),
        "object_assets_present": (assets / "objects").exists(),
        "embodiment_assets_present": (assets / "embodiments").exists(),
        "hammer_block_is_static": "is_static=True" in hammer_text,
    }
    report["official_task_smoke_ready"] = bool(
        report["repo_present"]
        and import_check.get("sapien", False)
        and import_check.get("mplib", False)
        and report["xpolicylab_initialized"]
        and report["object_assets_present"]
        and report["embodiment_assets_present"]
    )
    report["physics_only_proxy_ready"] = bool(import_check.get("sapien", False))
    return report


class SapienHingeBackend:
    """Headless SAPIEN articulated hinge used only as the R0-C bridge proxy."""

    def __init__(
        self,
        *,
        density: float,
        damping: float,
        regime: int,
        config: V6R0Config,
        preserve_engine_dissipation: bool = False,
    ):
        try:
            import sapien
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SAPIEN is unavailable; run V6R0 with the RoboTwin conda Python"
            ) from exc
        self.sapien = sapien
        self.regime = regime
        self.config = config
        self.density = density
        self.damping = damping
        self.preserve_engine_dissipation = preserve_engine_dissipation
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(config.physics_dt)
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_name("laptop_base")
        root.add_box_collision(half_size=(0.15, 0.10, 0.01), density=500)
        lid = builder.create_link_builder(root)
        lid.set_name("laptop_lid")
        lid.set_joint_name("hinge")
        lid.add_box_collision(
            pose=sapien.Pose((0, 0.15, 0)),
            half_size=(0.15, 0.15, 0.01),
            density=density,
        )
        lid.set_joint_properties(
            "revolute",
            ((-0.10, 1.50),),
            sapien.Pose((0, 0.10, 0.02)),
            sapien.Pose((0, -0.15, 0)),
            damping=0.0,
            friction=0.0,
        )
        self.actor = builder.build(fix_root_link=True)
        self.actor.set_sleep_threshold(0.0)
        # SAPIEN 3.0.0b1 retains a 0.05 joint-friction default after the
        # builder call above.  R0.1 requires a closed C0 control, so all
        # engine-side dissipation must either be explicit in h(q, qdot) or be
        # disabled here.  Do not leave it for a learned residual to absorb.
        if not preserve_engine_dissipation:
            for joint in self.actor.get_active_joints():
                joint.set_friction(0.0)
            for link in self.actor.get_links():
                link.set_angular_damping(0.0)
                link.set_linear_damping(0.0)
        self.pinocchio = self.actor.create_pinocchio_model()

    def _extra_torque(self, velocity: float, memory: float) -> float:
        if self.regime == 1:
            return -self.config.drag_strength * abs(velocity) * velocity
        if self.regime == 2:
            return -self.config.hysteresis_strength * memory
        return 0.0

    def transition(
        self,
        q: float,
        velocity: float,
        torque: float,
        memory: float,
        *,
        horizon: int = 1,
    ) -> tuple[float, float]:
        q_final, velocity_final, _ = self.transition_with_validity(
            q, velocity, torque, memory, horizon=horizon
        )
        return q_final, velocity_final

    def transition_with_validity(
        self,
        q: float,
        velocity: float,
        torque: float,
        memory: float,
        *,
        horizon: int = 1,
        limit_margin: float = 0.02,
    ) -> tuple[float, float, bool]:
        """Roll out truth and mark windows that activate joint-limit support."""
        self.actor.set_qpos((q,))
        self.actor.set_qvel((velocity,))
        self.actor.set_qacc((0.0,))
        self.actor.set_qf((0.0,))
        limits = self.actor.get_qlimits()[0]
        valid = bool(limits[0] + limit_margin < q < limits[1] - limit_margin)
        for _ in range(horizon * self.config.macro_steps):
            current_velocity = float(self.actor.get_qvel()[0])
            passive = self.actor.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            applied = (
                torque
                - self.damping * current_velocity
                + self._extra_torque(current_velocity, memory)
            )
            self.actor.set_qf(passive + applied)
            self.scene.step()
            current_q = float(self.actor.get_qpos()[0])
            valid &= bool(
                limits[0] + limit_margin < current_q < limits[1] - limit_margin
            )
        return (
            float(self.actor.get_qpos()[0]),
            float(self.actor.get_qvel()[0]),
            valid,
        )

    def generalized_force_sample(
        self,
        q: float,
        velocity: float,
        torque: float,
        memory: float,
    ) -> tuple[float, float, float]:
        """Return (force residual, density tangent, acceleration).

        The known generalized force includes gravity/Coriolis compensation,
        commanded torque, and the declared linear damping.  C1/C2 additions
        are deliberately excluded, so they appear directly in force space.
        """
        import numpy as np

        self.actor.set_qpos((q,))
        self.actor.set_qvel((velocity,))
        self.actor.set_qacc((0.0,))
        self.actor.set_qf((0.0,))
        for link in self.actor.get_links():
            link.wake_up()
        passive = self.actor.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True
        )
        known_force = float(passive[0]) + torque - self.damping * velocity
        applied_force = known_force + self._extra_torque(velocity, memory)
        self.actor.set_qf((applied_force,))
        self.scene.step()
        acceleration = float(self.actor.get_qacc()[0])
        inverse_force = float(
            self.pinocchio.compute_inverse_dynamics(
                np.asarray((q,)),
                np.asarray((velocity,)),
                np.asarray((acceleration,)),
            )[0]
        )
        residual = inverse_force - known_force
        # With fixed geometry, inertial/gravity terms scale linearly with
        # density.  This is the local parameter tangent in force space.
        density_tangent = (inverse_force - float(passive[0])) / self.density
        return residual, density_tangent, acceleration


def _sample_local_points(
    count: int,
    regime: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if regime == "passive":
        q = 0.30 + 0.25 * torch.rand(count, generator=generator)
        velocity = 0.40 * torch.rand(count, generator=generator) - 0.20
        torque = 0.50 * torch.rand(count, generator=generator) - 0.25
    elif regime == "discovery":
        q = 0.20 + 0.65 * torch.rand(count, generator=generator)
        velocity = 2.0 * torch.rand(count, generator=generator) - 1.0
        torque = 1.2 * torch.rand(count, generator=generator) - 0.6
    elif regime in ("diagnostic", "validation", "intervention"):
        q = 0.18 + 0.75 * torch.rand(count, generator=generator)
        velocity = 2.8 * torch.rand(count, generator=generator) - 1.4
        torque = 1.6 * torch.rand(count, generator=generator) - 0.8
    else:
        raise ValueError(regime)
    memory = torch.where(
        torch.rand(count, generator=generator) > 0.5,
        torch.ones(count),
        -torch.ones(count),
    )
    state = torch.stack((q, velocity, torque), dim=-1)
    return state, memory


def _simulate_points(
    backend: SapienHingeBackend,
    state: torch.Tensor,
    memory: torch.Tensor,
    config: V6R0Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack((state[:, 2], state[:, 1]), dim=-1)
    targets = []
    macro_dt = config.physics_dt * config.macro_steps
    for row, hidden in zip(state, memory):
        _, final_velocity = backend.transition(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(hidden),
        )
        targets.append((final_velocity - float(row[1])) / macro_dt / config.target_scale)
    return features, torch.tensor(targets, dtype=torch.float32)


def _collect_seed(
    config: V6R0Config, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 130_000)
    fields: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "passive_x",
            "passive_y",
            "discovery_x",
            "discovery_y",
            "action_x",
            "action_y",
            "validation_x",
            "validation_y",
            "validation_clean",
            "query_state",
            "query_memory",
            "rollout_truth",
        )
    }
    regimes = []
    macro_dt = config.physics_dt * config.macro_steps
    for regime in range(3):
        for _ in range(config.episodes_per_regime):
            density = config.density_min + (
                config.density_max - config.density_min
            ) * float(torch.rand((), generator=generator))
            damping = config.damping_min + (
                config.damping_max - config.damping_min
            ) * float(torch.rand((), generator=generator))
            backend = SapienHingeBackend(
                density=density, damping=damping, regime=regime, config=config
            )
            for prefix, count, sampling_regime in (
                ("passive", config.passive_context, "passive"),
                ("discovery", config.discovery_count, "discovery"),
                ("action", config.action_candidates, "diagnostic"),
                ("validation", config.validation_max, "validation"),
            ):
                state, memory = _sample_local_points(
                    count, sampling_regime, generator
                )
                features, target = _simulate_points(backend, state, memory, config)
                fields[f"{prefix}_x"].append(features)
                if prefix == "validation":
                    fields["validation_clean"].append(target)
                fields[f"{prefix}_y"].append(target)
            query_state, query_memory = _sample_local_points(
                config.query_count, "intervention", generator
            )
            truth = []
            for horizon in config.horizons:
                horizon_rows = []
                for row, hidden in zip(query_state, query_memory):
                    q, velocity = backend.transition(
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(hidden),
                        horizon=horizon,
                    )
                    horizon_rows.append((q, velocity))
                truth.append(torch.tensor(horizon_rows))
            fields["query_state"].append(query_state)
            fields["query_memory"].append(query_memory)
            fields["rollout_truth"].append(torch.stack(truth, dim=1))
            regimes.append(regime)
    output = {name: torch.stack(values).to(device) for name, values in fields.items()}
    output["regime"] = torch.tensor(regimes, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 131_000)
    for name in ("passive_y", "discovery_y", "action_y", "validation_y"):
        output[name] = output[name] + config.observation_noise * torch.randn(
            output[name].shape, generator=noise_generator, device=device
        )
    output["macro_dt"] = torch.tensor(macro_dt, device=device)
    return output


def _ensure_drag_candidate(
    proposed: torch.Tensor, regime: torch.Tensor
) -> torch.Tensor:
    output = proposed.clone()
    missing = (regime == 1) & ~(
        output == DRAG_OPERATOR
    ).any(dim=-1)
    output[missing, -1] = DRAG_OPERATOR
    return output


def _rollout_model(
    initial_state: torch.Tensor,
    horizons: tuple[int, ...],
    macro_dt: float,
    target_scale: float,
    acceleration,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = initial_state[..., 0].clone()
    velocity = initial_state[..., 1].clone()
    torque = initial_state[..., 2]
    snapshots = []
    stable_snapshots = []
    stable = torch.ones_like(q, dtype=torch.bool)
    horizon_set = set(horizons)
    for step in range(1, max(horizons) + 1):
        features = torch.stack((torque, velocity), dim=-1)
        acc = acceleration(features) * target_scale
        q = q + macro_dt * velocity
        velocity = velocity + macro_dt * acc
        step_stable = (
            torch.isfinite(q)
            & torch.isfinite(velocity)
            & (q.abs() < 10.0)
            & (velocity.abs() < 50.0)
        )
        stable &= step_stable
        q = torch.nan_to_num(q, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        velocity = torch.nan_to_num(
            velocity, nan=0.0, posinf=50.0, neginf=-50.0
        ).clamp(-50.0, 50.0)
        if step in horizon_set:
            snapshots.append(torch.stack((q, velocity), dim=-1))
            stable_snapshots.append(stable.clone())
    return torch.stack(snapshots, dim=2), torch.stack(stable_snapshots, dim=-1)


def _evaluate_seed(
    config: V6R0Config, seed: int, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _collect_seed(config, seed, device)
    regime = data["regime"]
    structural = regime > 0
    c0 = regime == 0
    c1 = regime == 1
    c2 = regime == 2
    initial_theta = _ridge_fit(data["passive_x"], data["passive_y"], config.ridge)
    discovery_residual = data["discovery_y"] - torch.einsum(
        "bni,bi->bn", data["discovery_x"], initial_theta
    )
    _, residual_orthogonal, tangent_ratio = tangent_decomposition(
        data["discovery_x"], discovery_residual, config.ridge
    )
    magnitude_score = discovery_residual.square().mean(dim=-1).sqrt()
    tangent_score = residual_orthogonal.square().mean(dim=-1).sqrt()
    gain = orthogonal_operator_gain(
        data["discovery_x"],
        residual_orthogonal,
        operator_library(data["discovery_x"]),
        ridge=config.ridge,
    )
    complexity_table = torch.tensor(
        OPERATOR_COMPLEXITY, dtype=gain.dtype, device=device
    )
    proposal_score = gain - config.proposal_complexity_price * complexity_table
    natural_proposed = proposal_score.topk(config.proposal_topk, dim=-1).indices
    natural_recall = (natural_proposed[c1] == DRAG_OPERATOR).any(dim=-1)
    proposed = _ensure_drag_candidate(natural_proposed, regime)
    correct_position = torch.where(
        c1,
        (proposed == DRAG_OPERATOR).float().argmax(dim=-1),
        torch.full_like(regime, -1),
    )
    train_x = torch.cat((data["passive_x"], data["discovery_x"]), dim=1)
    train_y = torch.cat((data["passive_y"], data["discovery_y"]), dim=1)
    selection_config = V6Config(
        episodes=train_x.shape[0],
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
    means, covariances = _fit_posteriors(
        train_x, train_y, proposed, config.observation_noise, selection_config
    )
    base_mean, base_covariance = _fit_base_posterior(
        train_x, train_y, config.observation_noise, selection_config
    )
    log_probabilities = torch.log_softmax(
        proposal_score.gather(1, proposed) / config.proposal_temperature, dim=-1
    )
    observation_variance = max(
        config.observation_noise, config.posterior_noise_floor
    ) ** 2
    used = torch.zeros_like(data["action_y"], dtype=torch.bool)
    active = torch.ones_like(regime, dtype=torch.bool)
    for _ in range(config.selection_probes):
        pool_means, pool_variances = _predictive_moments(
            data["action_x"], proposed, means, covariances
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
        selected_x = data["action_x"].gather(
            1, action_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        selected_y = data["action_y"].gather(
            1, action_index.unsqueeze(-1)
        ).squeeze(-1)
        design = _candidate_design(selected_x.unsqueeze(1), proposed).squeeze(1)
        means, covariances, log_probabilities = _update_candidate_belief(
            means,
            covariances,
            log_probabilities,
            design,
            selected_y,
            observation_variance,
            active,
        )
        base_mean, base_covariance = _update_base_belief(
            base_mean,
            base_covariance,
            selected_x,
            selected_y,
            observation_variance,
            active,
        )
        used[torch.arange(regime.shape[0], device=device), action_index] = True
    order = log_probabilities.topk(2, dim=-1).indices
    selected_position, competitor_position = order[:, 0], order[:, 1]
    selected_operator = proposed.gather(1, selected_position[:, None]).squeeze(-1)
    selection_correct = selected_operator == DRAG_OPERATOR
    validation_design = _candidate_design(data["validation_x"], proposed)
    validation_all = torch.einsum("bank,bnk->ban", validation_design, means)
    selected_prediction = validation_all.gather(
        -1,
        selected_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    competitor_prediction = validation_all.gather(
        -1,
        competitor_position[:, None, None].expand(-1, config.validation_max, 1),
    ).squeeze(-1)
    base_prediction = torch.einsum(
        "bni,bi->bn", data["validation_x"], base_mean
    )
    selected_complexity = complexity_table[selected_operator]
    accepted, rejected, deferred, samples = _sequential_decision(
        (data["validation_y"] - base_prediction).square(),
        (data["validation_y"] - competitor_prediction).square(),
        (data["validation_y"] - selected_prediction).square(),
        selected_complexity,
        config.observation_noise**2,
        selection_config,
        method="bf",
    )
    exact = c1 & selection_correct & accepted
    fitted_residual = train_y - torch.einsum("bni,bi->bn", train_x, base_mean)

    selected_mean = means.gather(
        1, selected_position[:, None, None].expand(-1, 1, means.shape[-1])
    ).squeeze(1)

    def base_acc(features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqi,bi->bq", features, base_mean)

    def candidate_acc(features: torch.Tensor) -> torch.Tensor:
        design = _candidate_design(features, proposed)
        all_predictions = torch.einsum("bank,bnk->ban", design, means)
        return all_predictions.gather(
            -1,
            selected_position[:, None, None].expand(-1, features.shape[1], 1),
        ).squeeze(-1)

    def fallback_acc(features: torch.Tensor) -> torch.Tensor:
        return base_acc(features) + _residual_predict(
            train_x, fitted_residual, features, config.residual_ridge
        ).clamp(-1.5, 1.5)

    def revised_acc(features: torch.Tensor) -> torch.Tensor:
        return torch.where(accepted[:, None], candidate_acc(features), fallback_acc(features))

    macro_dt = float(data["macro_dt"])
    rollout_outputs = {
        "physics": _rollout_model(
            data["query_state"], config.horizons, macro_dt, config.target_scale, base_acc
        ),
        "no_revision": _rollout_model(
            data["query_state"], config.horizons, macro_dt, config.target_scale, fallback_acc
        ),
        "frozen_v6": _rollout_model(
            data["query_state"], config.horizons, macro_dt, config.target_scale, revised_acc
        ),
    }
    horizon_rows: list[dict[str, Any]] = []
    truth = data["rollout_truth"]
    scale = torch.tensor((1.0, 2.0), device=device)
    for model, (prediction, stable) in rollout_outputs.items():
        for index, horizon in enumerate(config.horizons):
            error = (prediction[:, :, index] - truth[:, :, index]) / scale
            for regime_index, regime_name in enumerate(REGIME_NAMES):
                mask = regime == regime_index
                horizon_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "regime": regime_name,
                        "horizon": horizon,
                        "state_rmse": float(torch.sqrt(error[mask].square().mean())),
                        "stable_fraction": float(stable[mask, :, index].float().mean()),
                    }
                )
    before = 1.0
    after = float((~accepted)[c1].float().mean())
    summary = {
        "seed": seed,
        "tangent_detection_auroc": _binary_auroc(tangent_score, structural),
        "magnitude_detection_auroc": _binary_auroc(magnitude_score, structural),
        "mean_tangent_ratio": float(tangent_ratio.mean()),
        "natural_top3_recall_c1": float(natural_recall.float().mean()),
        "controlled_candidate_coverage_c1": 1.0,
        "selection_accuracy_c1": float(selection_correct[c1].float().mean()),
        "acceptance_power_c1_correct_selection": float(
            accepted[c1 & selection_correct].float().mean()
        ),
        "exact_operator_recovery_c1": float(exact[c1].float().mean()),
        "false_revision_c0": float(accepted[c0].float().mean()),
        "unknown_rejection_c2": float((~accepted)[c2].float().mean()),
        "validation_samples": float(samples.mean()),
        "defer_rate": float(deferred.float().mean()),
        "residual_usage_before_c1": before,
        "residual_usage_after_c1": after,
        "residual_assimilation_ratio_c1": residual_assimilation_ratio(before, after),
        "selected_parameter_norm": float(selected_mean.norm(dim=-1).mean()),
    }
    return summary, horizon_rows


def _aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = sorted(set(rows[0]) - {"seed"})
    output = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        low, high = _bootstrap_ci(values)
        output.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return output


def run_v6r0_smoke(
    output_dir: str | Path,
    *,
    device_name: str = "cpu",
    config: V6R0Config | None = None,
    robotwin_repo: str | Path = "/home/dong/Projects/RoboTwin",
) -> dict[str, Any]:
    cfg = config or V6R0Config()
    device = resolve_device(device_name)
    seed_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        summary, horizons = _evaluate_seed(cfg, seed, device)
        seed_rows.append(summary)
        horizon_rows.extend(horizons)
    root = Path(output_dir)
    capability = inspect_robotwin(robotwin_repo)
    _write_json(root / "capability.json", capability)
    _write_csv(root / "seed_metrics.csv", seed_rows)
    _write_csv(root / "metrics_aggregate.csv", _aggregate_seed_rows(seed_rows))
    _write_csv(root / "rollout_horizons.csv", horizon_rows)
    manifest = counterfactual_branch_manifest(
        task="open_laptop",
        seeds=cfg.seeds,
        windows_per_seed=cfg.episodes_per_regime,
        branches_per_window=cfg.query_count,
        horizons=cfg.horizons,
    )
    _write_json(root / "counterfactual_manifest.json", manifest)
    summary = {
        "scope": "R0-C headless SAPIEN hinge proxy",
        "official_robotwin_task_executed": False,
        "mechanism_frozen_from_v6": True,
        "operator_library_frozen": True,
        "oracle_state": True,
        "visual_perception": False,
        "learned_policy": False,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "horizons": list(cfg.horizons)},
        "capability": capability,
        "rows": len(seed_rows),
    }
    _write_json(root / "summary.json", summary)
    return summary

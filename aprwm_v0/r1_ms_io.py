from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .r1_ms import SupportSignals, classify_support
from .v06 import _write_json


@dataclass(frozen=True)
class R1MSIOConfig:
    seeds: tuple[int, ...] = (8201, 8211, 8221)
    task_id: str = "PickCube-v1"
    horizons: tuple[int, ...] = (2, 4, 8, 32)
    decision_horizons: tuple[int, ...] = (2, 4, 8)
    warmup_steps: int = 2
    perturbation: float = 0.5
    replay_tolerance: float = 1.0e-6


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _nominal_actions(length: int, action_dim: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed + 91_000)
    actions = generator.normal(0.0, 0.05, size=(length, action_dim)).astype(np.float32)
    phase = np.arange(length, dtype=np.float32)
    actions[:, 0] += 0.10 * np.sin(phase * 0.3)
    return np.clip(actions, -0.25, 0.25)


def _branch_actions(nominal: np.ndarray, delta: float) -> dict[str, np.ndarray]:
    branches = {"nominal": nominal.copy()}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        action = nominal.copy()
        action[0, 0] = np.clip(action[0, 0] + sign * delta, -1.0, 1.0)
        branches[name] = action
    return branches


def _restore(env: Any, state_dict: dict[str, Any], elapsed_steps: torch.Tensor) -> None:
    env.set_state_dict(_clone_tree(state_dict))
    # BaseEnv state_dict is simulator/controller state, not episode bookkeeping.
    # Restoring this counter explicitly prevents branch order from affecting
    # termination/truncation behavior.
    env._elapsed_steps.copy_(elapsed_steps)


def _rollout(env: Any, actions: np.ndarray) -> dict[str, np.ndarray]:
    states = [env.get_state().detach().cpu().numpy()[0].copy()]
    qpos = [env.agent.robot.get_qpos().detach().cpu().numpy()[0].copy()]
    qvel = [env.agent.robot.get_qvel().detach().cpu().numpy()[0].copy()]
    terminated = []
    truncated = []
    for action in actions:
        _, _, term, trunc, _ = env.step(action)
        states.append(env.get_state().detach().cpu().numpy()[0].copy())
        qpos.append(env.agent.robot.get_qpos().detach().cpu().numpy()[0].copy())
        qvel.append(env.agent.robot.get_qvel().detach().cpu().numpy()[0].copy())
        terminated.append(bool(torch.as_tensor(term).any()))
        truncated.append(bool(torch.as_tensor(trunc).any()))
    return {
        "states": np.asarray(states, dtype=np.float32),
        "qpos": np.asarray(qpos, dtype=np.float32),
        "qvel": np.asarray(qvel, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
    }


def _finite_difference_rmse(rollout: dict[str, np.ndarray], dt: float) -> float:
    qpos = rollout["qpos"]
    qvel = rollout["qvel"]
    finite_difference = np.diff(qpos, axis=0) / dt
    midpoint_velocity = 0.5 * (qvel[:-1] + qvel[1:])
    return float(np.sqrt(np.mean(np.square(finite_difference - midpoint_velocity))))


def _write_trajectory(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    config: R1MSIOConfig,
) -> None:
    with h5py.File(path, "w") as stream:
        stream.attrs["scope"] = "R1-MS0-I/O smoke; no scientific gate authority"
        stream.attrs["task_id"] = config.task_id
        stream.attrs["decision_horizons"] = config.decision_horizons
        stream.attrs["evaluation_horizons"] = config.horizons
        for row in rows:
            group = stream.create_group(f"seed_{row['seed']}")
            group.create_dataset("snapshot", data=row["snapshot"])
            group.create_dataset("elapsed_steps", data=row["elapsed_steps"])
            for name, actions in row["actions"].items():
                branch = group.create_group(name)
                branch.create_dataset("actions", data=actions)
                for key, value in row["rollouts"][name].items():
                    branch.create_dataset(key, data=value)
            group.create_dataset("nominal_replay_states", data=row["nominal_replay"])


def _serialization_max_error(path: Path, rows: list[dict[str, Any]]) -> float:
    maximum = 0.0
    with h5py.File(path, "r") as stream:
        for row in rows:
            group = stream[f"seed_{row['seed']}"]
            maximum = max(
                maximum,
                float(np.max(np.abs(group["snapshot"][:] - row["snapshot"]))),
            )
            for name in row["actions"]:
                maximum = max(
                    maximum,
                    float(
                        np.max(
                            np.abs(
                                group[name]["states"][:]
                                - row["rollouts"][name]["states"]
                            )
                        )
                    ),
                )
    return maximum


def run_r1_ms_io_smoke(
    output: str | Path,
    *,
    config: R1MSIOConfig = R1MSIOConfig(),
) -> dict[str, Any]:
    if 32 in config.decision_horizons:
        raise ValueError("H32 is evaluation-only and cannot enter the decision inputs")
    if not set(config.decision_horizons).issubset(config.horizons):
        raise ValueError("Decision horizons must be a subset of evaluation horizons")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - task registration

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    maximum_horizon = max(config.horizons)
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    env = gym.make(
        config.task_id,
        obs_mode="state_dict",
        render_mode=None,
        sim_backend="cpu",
        num_envs=1,
    )
    unwrapped = env.unwrapped
    sim_timestep = float(unwrapped.sim_timestep)
    control_timestep = float(unwrapped.control_timestep)
    try:
        for seed in config.seeds:
            env.reset(seed=seed)
            zero = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(config.warmup_steps):
                unwrapped.step(zero)

            snapshot_dict = _clone_tree(unwrapped.get_state_dict())
            snapshot = unwrapped.get_state().detach().cpu().numpy()[0].copy()
            elapsed_steps = unwrapped._elapsed_steps.detach().clone()
            _restore(unwrapped, snapshot_dict, elapsed_steps)
            restore_error = float(
                np.max(
                    np.abs(
                        unwrapped.get_state().detach().cpu().numpy()[0] - snapshot
                    )
                )
            )

            nominal = _nominal_actions(maximum_horizon, env.action_space.shape[0], seed)
            actions = _branch_actions(nominal, config.perturbation)
            rollouts = {}
            for name, branch_actions in actions.items():
                _restore(unwrapped, snapshot_dict, elapsed_steps)
                rollouts[name] = _rollout(unwrapped, branch_actions)

            _restore(unwrapped, snapshot_dict, elapsed_steps)
            nominal_replay = _rollout(unwrapped, actions["nominal"])["states"]
            replay_error = float(
                np.max(np.abs(nominal_replay - rollouts["nominal"]["states"]))
            )
            terminal_distances = [
                float(
                    np.linalg.norm(
                        rollouts[name]["states"][-1]
                        - rollouts["nominal"]["states"][-1]
                    )
                )
                for name in ("positive", "negative")
            ]
            horizon_distances = {
                str(horizon): {
                    name: float(
                        np.linalg.norm(
                            rollouts[name]["states"][horizon]
                            - rollouts["nominal"]["states"][horizon]
                        )
                    )
                    for name in ("positive", "negative")
                }
                for horizon in config.horizons
            }
            fd_rmse = _finite_difference_rmse(
                rollouts["nominal"], control_timestep
            )
            rows.append(
                {
                    "seed": seed,
                    "snapshot": snapshot,
                    "elapsed_steps": elapsed_steps.detach().cpu().numpy(),
                    "actions": actions,
                    "rollouts": rollouts,
                    "nominal_replay": nominal_replay,
                }
            )
            metric_rows.append(
                {
                    "seed": seed,
                    "restore_max_abs": restore_error,
                    "deterministic_replay_max_abs": replay_error,
                    "counterfactual_terminal_l2_min": min(terminal_distances),
                    "qvel_finite_difference_rmse": fd_rmse,
                    "horizon_distances": horizon_distances,
                    "terminated_any": bool(rollouts["nominal"]["terminated"].any()),
                    "truncated_any": bool(rollouts["nominal"]["truncated"].any()),
                }
            )
    finally:
        env.close()

    trajectory_path = root / "trajectory.h5"
    _write_trajectory(trajectory_path, rows, config=config)
    serialization_error = _serialization_max_error(trajectory_path, rows)
    state_machine_probe = {
        "modeled": classify_support(SupportSignals(0.2)).value,
        "boundary": classify_support(
            SupportSignals(0.01, near_joint_limit=True)
        ).value,
        "unsupported": classify_support(
            SupportSignals(0.2, unsupported_collision=True)
        ).value,
    }
    max_restore = max(row["restore_max_abs"] for row in metric_rows)
    max_replay = max(row["deterministic_replay_max_abs"] for row in metric_rows)
    min_branch = min(row["counterfactual_terminal_l2_min"] for row in metric_rows)
    passed = bool(
        max_restore <= config.replay_tolerance
        and max_replay <= config.replay_tolerance
        and serialization_error == 0.0
        and min_branch > config.replay_tolerance
        and state_machine_probe
        == {
            "modeled": "modeled",
            "boundary": "boundary",
            "unsupported": "unsupported",
        }
    )
    summary = {
        "scope": "R1-MS0-I/O smoke",
        "scientific_result": False,
        "can_unlock_drawer_gate": False,
        "task": config.task_id,
        "observation_mode": "state_dict",
        "sim_backend": "cpu",
        "seeds": list(config.seeds),
        "sim_timestep": sim_timestep,
        "control_timestep": control_timestep,
        "decision_horizons": list(config.decision_horizons),
        "evaluation_horizons": list(config.horizons),
        "h32_blind_to_decision": 32 not in config.decision_horizons,
        "max_restore_abs": max_restore,
        "max_replay_abs": max_replay,
        "min_counterfactual_terminal_l2": min_branch,
        "serialization_max_abs": serialization_error,
        "state_machine_probe": state_machine_probe,
        "passed": passed,
        "drawer_assets_required": False,
        "drawer_c0_evaluated": False,
        "metrics": metric_rows,
        "trajectory": str(trajectory_path),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "config.json", config.__dict__)
    return summary

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .v06 import _write_json


TASK_ID = "OpenCabinetDrawer-v1"
DEVELOPMENT_SEEDS = (8101, 8111, 8121)
# Fixed before any Drawer rollout. These are the first, middle, and last IDs
# in the sorted 25-asset Drawer registry and also span 1/2/3 target links.
DRAWER_DEVELOPMENT_ASSETS = ("1000", "1040", "1082")
ROLLOUT_HORIZONS = (1, 4, 8, 16, 32)
ACCEPTANCE_HORIZONS = (2, 4, 8)
DRAWER_DYNAMICS_METADATA = (
    "joint_friction",
    "joint_damping",
    "drive_stiffness",
    "drive_damping",
    "joint_type",
    "joint_limits",
    "active_dof_index",
    "link_mass",
    "link_inertia",
    "gravity_contribution",
)


class SupportState(str, Enum):
    """Whether a transition belongs to the frozen analytic model class."""

    MODELED = "modeled"
    BOUNDARY = "boundary"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class R1MSGates:
    c0_false_revision_max: float = 0.01
    c1_exact_recovery_min: float = 0.90
    accepted_stability_min: float = 0.99
    native_h16_gain_min: float = 0.0
    acceptance_rate_min: float = 0.20
    c2_forced_wrong_revision_max: float = 0.10


@dataclass(frozen=True)
class DrawerLocalState:
    """Privileged state at the selected prismatic drawer joint."""

    q: torch.Tensor
    qvel: torch.Tensor
    qacc: torch.Tensor
    qf: torch.Tensor

    def stacked(self) -> torch.Tensor:
        return torch.stack((self.q, self.qvel, self.qacc, self.qf), dim=-1)


@dataclass(frozen=True)
class SupportSignals:
    joint_limit_margin: float
    near_joint_limit: bool = False
    contact_mode_changed: bool = False
    grasp_constraint_active: bool = False
    unsupported_collision: bool = False
    task_terminated: bool = False


@dataclass(frozen=True)
class GateMeasurements:
    c0_false_revision_rate: float | None = None
    c0_structured_residual: bool | None = None
    c1_exact_recovery_rate: float | None = None
    accepted_stability: float | None = None
    native_h16_gain: float | None = None
    native_h32_catastrophic_reversal: bool | None = None
    acceptance_rate: float | None = None
    c2_forced_wrong_revision_rate: float | None = None
    history_gain_over_memoryless: float | None = None


def classify_support(signals: SupportSignals) -> SupportState:
    if (
        signals.unsupported_collision
        or signals.grasp_constraint_active
        or signals.task_terminated
    ):
        return SupportState.UNSUPPORTED
    if signals.near_joint_limit or signals.contact_mode_changed:
        return SupportState.BOUNDARY
    return SupportState.MODELED


def all_c0_assets_individually_close(
    asset_results: Mapping[str, bool],
    required_assets: Sequence[str] = DRAWER_DEVELOPMENT_ASSETS,
) -> bool:
    """Prevent aggregate statistics from bypassing a failed/missing asset."""
    required = {str(asset_id) for asset_id in required_assets}
    observed = {str(asset_id) for asset_id in asset_results}
    return observed == required and all(
        bool(asset_results[asset_id]) for asset_id in required
    )


def generalized_residual_power(
    residual_force: torch.Tensor, generalized_velocity: torch.Tensor
) -> torch.Tensor:
    """Compute r_tau^T qdot (or r_w^T nu) on the last dimension."""
    if residual_force.shape != generalized_velocity.shape:
        raise ValueError("Residual force and velocity must have identical shapes")
    return (residual_force * generalized_velocity).sum(dim=-1)


def drawer_state_from_articulation(
    articulation: Any,
    joint_index: int | torch.Tensor = 0,
    *,
    previous_qvel: torch.Tensor | None = None,
    dt: float | None = None,
) -> DrawerLocalState:
    """Extract the oracle Drawer Mode-A state from a ManiSkill articulation."""
    def select_joint(value: torch.Tensor) -> torch.Tensor:
        if isinstance(joint_index, int):
            if value.shape[-1] <= joint_index:
                raise IndexError(
                    f"Joint index {joint_index} is outside articulation shape"
                )
            return value[..., joint_index]
        index = torch.as_tensor(joint_index, device=value.device, dtype=torch.long)
        if index.shape != value.shape[:-1]:
            raise ValueError(
                "Batched joint indices must match articulation batch dimensions"
            )
        if bool((index < 0).any()) or bool((index >= value.shape[-1]).any()):
            raise IndexError("At least one target joint index is outside articulation shape")
        return value.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    fields = {}
    for name in ("qpos", "qvel", "qf"):
        getter = getattr(articulation, f"get_{name}", None)
        if getter is None:
            raise TypeError(f"Articulation does not expose get_{name}()")
        value = torch.as_tensor(getter())
        fields[name] = select_joint(value)
    qacc_getter = getattr(articulation, "get_qacc", None)
    if qacc_getter is not None:
        qacc = select_joint(torch.as_tensor(qacc_getter()))
    elif previous_qvel is not None and dt is not None and dt > 0:
        qacc = (fields["qvel"] - torch.as_tensor(previous_qvel)) / dt
    else:
        raise TypeError(
            "Acceleration is not exposed; pass previous_qvel and positive simulator dt"
        )
    return DrawerLocalState(
        q=fields["qpos"],
        qvel=fields["qvel"],
        qacc=qacc,
        qf=fields["qf"],
    )


def drawer_target_state_from_env(
    env: Any,
    *,
    previous_qvel: torch.Tensor | None = None,
    dt: float | None = None,
) -> DrawerLocalState:
    """Use ManiSkill's per-environment target joint mapping for mixed cabinets."""
    cabinet = getattr(env, "cabinet", None)
    handle_link = getattr(env, "handle_link", None)
    if cabinet is None or handle_link is None or getattr(handle_link, "joint", None) is None:
        raise TypeError("Expected an OpenCabinetDrawer environment after reset")
    return drawer_state_from_articulation(
        cabinet,
        joint_index=handle_link.joint.active_index,
        previous_qvel=previous_qvel,
        dt=dt,
    )


def frozen_protocol() -> dict[str, Any]:
    """The preregistered R0.6 contract; R1-MS may not tune these fields."""
    return {
        "revision_algorithm": "V6R0.6 corrected two-layer mechanism",
        "frozen": True,
        "observation_mode": "state_dict",
        "perception_error": False,
        "tangent_evidence": "unchanged from R0.6",
        "operator_proposal": "unchanged from R0.6",
        "controlled_c1_operator": "-alpha*abs(qvel)*qvel, alpha>0",
        "physical_feasibility": "dissipative-family hard constraint",
        "drawer_passivity": "r_tau^T qdot <= 0",
        "utility_gate": {
            "horizons": list(ACCEPTANCE_HORIZONS),
            "weights": [1.0 / 3.0] * 3,
            "threshold": 0.0,
        },
        "history_aware_fallback": "unchanged from R0.4-B/R0.6",
        "validity_mask": [state.value for state in SupportState],
        "evaluation_horizons": list(ROLLOUT_HORIZONS),
        "h32_visible_to_acceptance": False,
    }


def stage_manifest() -> list[dict[str, Any]]:
    return [
        {
            "stage": "MS0-A-C0",
            "task": TASK_ID,
            "mode": "instrumented_dynamics",
            "asset_ids": list(DRAWER_DEVELOPMENT_ASSETS),
            "seeds": list(DEVELOPMENT_SEEDS),
            "matrix": {
                "assets": len(DRAWER_DEVELOPMENT_ASSETS),
                "seeds_per_asset": len(DEVELOPMENT_SEEDS),
                "total_asset_seed_cells": len(DRAWER_DEVELOPMENT_ASSETS)
                * len(DEVELOPMENT_SEEDS),
            },
            "regime": "adequate_physics",
            "purpose": "geometry-conditioned object-physics closure",
            "closure_granularity": "per_asset_before_aggregation",
            "go_rule": "all sampled assets individually close",
            "required_dynamics_metadata": list(DRAWER_DYNAMICS_METADATA),
            "unlocks": "MS0-A-C1",
        },
        {
            "stage": "MS0-A-C1",
            "task": TASK_ID,
            "mode": "instrumented_dynamics",
            "seeds": list(DEVELOPMENT_SEEDS),
            "regime": "controlled_in_library_drag",
            "locked_until": "MS0-A-C0 passes",
            "unlocks": "Mode B implementation",
        },
        {
            "stage": "MS1-B",
            "task": TASK_ID,
            "mode": "embodied_contact",
            "locked_until": "MS0-A-C0 and MS0-A-C1 pass",
            "regimes": ["C0", "C1", "C2", "N0"],
        },
        {
            "stage": "contact_transfer",
            "task": "PushT-v1",
            "locked_until": "formal Drawer Mode A/B passes all gates",
        },
        {
            "stage": "out_of_design_stress",
            "task": "PegInsertionSide-v1",
            "locked_until": "PushT transfer passes",
        },
    ]


def counterfactual_manifest(
    *, seeds: Sequence[int] = DEVELOPMENT_SEEDS, windows_per_asset: int = 4,
    branches_per_window: int = 4
) -> list[dict[str, Any]]:
    rows = []
    phases = ("pre_contact", "early_contact", "sustained_contact", "near_transition")
    for seed in seeds:
        for window in range(windows_per_asset):
            for branch in range(branches_per_window):
                rows.append(
                    {
                        "seed": int(seed),
                        "window": window,
                        "phase": phases[window % len(phases)],
                        "branch": branch,
                        "restore_with_state_dict": True,
                        "simulator_executes_perturbation": True,
                        "horizons": list(ROLLOUT_HORIZONS),
                    }
                )
    return rows


def evaluate_ordered_gates(
    measurements: GateMeasurements, gates: R1MSGates = R1MSGates()
) -> list[dict[str, Any]]:
    """Evaluate gates in order and stop expansion after the first failure."""
    checks = (
        (
            "closure",
            measurements.c0_false_revision_rate is not None
            and measurements.c0_structured_residual is not None,
            lambda: measurements.c0_false_revision_rate < gates.c0_false_revision_max
            and not measurements.c0_structured_residual,
        ),
        (
            "c1_recoverability",
            measurements.c1_exact_recovery_rate is not None,
            lambda: measurements.c1_exact_recovery_rate > gates.c1_exact_recovery_min,
        ),
        (
            "safety",
            measurements.accepted_stability is not None,
            lambda: measurements.accepted_stability >= gates.accepted_stability_min,
        ),
        (
            "native_utility",
            measurements.native_h16_gain is not None
            and measurements.native_h32_catastrophic_reversal is not None,
            lambda: measurements.native_h16_gain > gates.native_h16_gain_min
            and not measurements.native_h32_catastrophic_reversal,
        ),
        (
            "nontriviality",
            measurements.acceptance_rate is not None,
            lambda: measurements.acceptance_rate > gates.acceptance_rate_min,
        ),
        (
            "unknown_safety",
            measurements.c2_forced_wrong_revision_rate is not None
            and measurements.history_gain_over_memoryless is not None,
            lambda: measurements.c2_forced_wrong_revision_rate
            < gates.c2_forced_wrong_revision_max
            and measurements.history_gain_over_memoryless > 0.0,
        ),
    )
    rows: list[dict[str, Any]] = []
    blocked = False
    for name, available, check in checks:
        if blocked:
            status = "blocked_by_earlier_gate"
        elif not available:
            status = "not_evaluated"
            blocked = True
        elif check():
            status = "pass"
        else:
            status = "fail"
            blocked = True
        rows.append({"gate": name, "status": status})
    return rows


def _python_probe(python_path: Path) -> dict[str, Any]:
    if not python_path.exists():
        return {}
    script = r'''
import importlib.metadata, importlib.util, json
mods = {n: importlib.util.find_spec(n) is not None for n in ["mani_skill", "sapien", "gymnasium", "torch"]}
versions = {}
for dist in ["mani-skill", "sapien", "gymnasium", "torch"]:
    try: versions[dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError: pass
tasks = {}
if mods["mani_skill"] and mods["gymnasium"]:
    from pathlib import Path
    import mani_skill.envs
    from mani_skill import PACKAGE_ASSET_DIR
    import gymnasium as gym
    for env_id in ["OpenCabinetDrawer-v1", "PushT-v1", "PegInsertionSide-v1"]:
        try:
            spec = gym.spec(env_id)
            tasks[env_id] = {"registered": True, "max_episode_steps": spec.max_episode_steps}
        except Exception as exc:
            tasks[env_id] = {"registered": False, "error": type(exc).__name__}
    drawer_meta = PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    drawer_asset_ids = list(json.loads(drawer_meta.read_text()).keys())
else:
    drawer_asset_ids = []
print(json.dumps({"modules": mods, "versions": versions, "tasks": tasks, "drawer_asset_ids": drawer_asset_ids}))
'''
    completed = subprocess.run(
        (str(python_path), "-c", script), capture_output=True, text=True, check=False
    )
    if completed.returncode:
        return {"probe_error": completed.stderr.strip(), "returncode": completed.returncode}
    return json.loads(completed.stdout.strip())


def asset_directory_complete(path: Path) -> bool:
    """Reject empty directories left behind by interrupted downloads."""
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def inspect_maniskill(
    python: str | Path,
    asset_root: str | Path,
) -> dict[str, Any]:
    python_path = Path(python)
    asset_path = Path(asset_root)
    probe = _python_probe(python_path)
    drawer_asset_ids = tuple(probe.get("drawer_asset_ids", ()))
    drawer_asset_root = asset_path / "data" / "partnet_mobility" / "dataset"
    present_ids = tuple(
        asset_id for asset_id in drawer_asset_ids
        if asset_directory_complete(drawer_asset_root / asset_id)
    )
    cabinet_assets = bool(drawer_asset_ids) and len(present_ids) == len(drawer_asset_ids)
    drawer_registered = bool(
        probe.get("tasks", {}).get(TASK_ID, {}).get("registered", False)
    )
    return {
        "python": str(python_path),
        "python_present": python_path.exists(),
        "asset_root": str(asset_path),
        "probe": probe,
        "drawer_assets_present": cabinet_assets,
        "drawer_asset_count": len(present_ids),
        "drawer_asset_required": len(drawer_asset_ids),
        "missing_drawer_asset_ids": sorted(set(drawer_asset_ids) - set(present_ids)),
        "drawer_task_registered": drawer_registered,
        "mode_a_smoke_ready": bool(
            python_path.exists()
            and probe.get("modules", {}).get("mani_skill", False)
            and drawer_registered
            and cabinet_assets
        ),
        "scope": "Drawer-only, oracle state_dict, Mode A before Mode B",
    }


def run_preflight(
    output: str | Path,
    *,
    python: str | Path,
    asset_root: str | Path,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    report = inspect_maniskill(python, asset_root)
    protocol = frozen_protocol()
    stages = stage_manifest()
    branches = counterfactual_manifest()
    gates = asdict(R1MSGates())
    _write_json(root / "preflight.json", report)
    _write_json(root / "frozen_protocol.json", protocol)
    _write_json(root / "stage_manifest.json", stages)
    _write_json(root / "counterfactual_manifest.json", branches)
    _write_json(root / "gates.json", gates)
    payload = json.dumps(
        {"protocol": protocol, "stages": stages, "gates": gates}, sort_keys=True
    ).encode()
    summary = {
        **report,
        "protocol_sha256": hashlib.sha256(payload).hexdigest(),
        "formal_matrix_unlocked": False,
        "next_action": "run MS0-A-C0 smoke" if report["mode_a_smoke_ready"] else (
            "download the 25 drawer-only PartNetMobility assets"
            if report.get("probe", {}).get("modules", {}).get("mani_skill", False)
            else "install ManiSkill 3.0.1, then download drawer-only assets"
        ),
        "output": str(root),
    }
    _write_json(root / "summary.json", summary)
    return summary

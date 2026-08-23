"""CAP-X3 scene/trajectory generation (fresh seeds; ρ=0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cap_x0 import EXCITATIONS, _downsample_idx, _simulate_trajectory, torque_sequence
from .capx_arm3_plant import N_DOF, SceneTheta, load_capx_arm3, sample_scene_theta, scene_theta_to_dict


def generate_scene_bundle(
    *,
    scene_seed: int,
    n_cal: int,
    n_query: int,
    duration_s: float = 1.0,
    timestep: float = 0.002,
    model_hz: float = 100.0,
    torque_scale: float = 0.35,
    traj_seed0: int = 150_000,
) -> dict[str, Any]:
    model, data = load_capx_arm3(timestep)
    rng = np.random.default_rng(scene_seed)
    theta = sample_scene_theta(rng)
    n_phys = int(round(duration_s / timestep))
    ds = _downsample_idx(n_phys, timestep, model_hz)
    q0 = rng.uniform(-0.4, 0.4, size=N_DOF)
    qd0 = rng.uniform(-0.2, 0.2, size=N_DOF)

    def _one(t_i: int, tag: str) -> dict[str, np.ndarray]:
        kind = EXCITATIONS[t_i % len(EXCITATIONS)]
        seed = traj_seed0 + 1000 * scene_seed + t_i + (0 if tag == "cal" else 10_000)
        u = torque_sequence(kind, n_steps=n_phys, dt=timestep, seed=seed, scale=torque_scale, duration_s=duration_s)
        raw = _simulate_trajectory(model=model, data=data, theta=theta, torques=u, q0=q0, qd0=qd0)
        return {k: v[ds] for k, v in raw.items() if k in ("q", "qd", "qdd", "u")}

    cal = [_one(i, "cal") for i in range(n_cal)]
    query = [_one(i, "query") for i in range(n_query)]
    return {"theta": theta, "scene_seed": scene_seed, "cal": cal, "query": query, "meta": scene_theta_to_dict(theta)}


def stack_split(trajs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([t[k] for t in trajs], axis=0) for k in ("q", "qd", "qdd", "u")}


def load_scene_h5(path: Path) -> dict[str, Any]:
    from .capx_arm3_plant import SceneTheta

    with h5py.File(path, "r") as handle:
        theta = SceneTheta.from_vector(np.asarray(handle["theta"], dtype=np.float64))
        bundle: dict[str, Any] = {
            "theta": theta,
            "scene_seed": int(handle.attrs["scene_seed"]),
            "cal": [],
            "query": [],
        }
        for split in ("cal", "query"):
            grp = handle[split]
            keys = sorted(grp.keys())
            for key in keys:
                tgrp = grp[key]
                bundle[split].append({k: np.asarray(tgrp[k], dtype=np.float64) for k in ("q", "qd", "qdd", "u")})
        return bundle


def write_scene_h5(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta: SceneTheta = bundle["theta"]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("theta", data=theta.as_vector())
        handle.attrs["scene_seed"] = int(bundle["scene_seed"])
        for split in ("cal", "query"):
            grp = handle.create_group(split)
            for i, traj in enumerate(bundle[split]):
                tgrp = grp.create_group(f"traj_{i:03d}")
                for k, v in traj.items():
                    tgrp.create_dataset(k, data=v)

"""CAP-X2 dataset generation with tau_perp and held-out rho calibration traj.

Calibration trajectories never enter train/val/test pools. Learners never
see alpha/beta/eta/rho/gamma. Does not unlock R10.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from .cap_x0 import (
    EXCITATIONS,
    ExcitationKind,
    _downsample_idx,
    torque_sequence,
)
from .capx_arm3_plant import (
    HOST_PLANT_ID,
    N_DOF,
    SceneTheta,
    apply_scene_theta,
    assert_capx_arm3,
    load_capx_arm3,
    oracle_force_residual,
    sample_scene_theta,
    scene_theta_to_dict,
)
from .capx_residual import (
    RESIDUAL_FAMILY_ID,
    RHO_GRID,
    SceneResidualCoeffs,
    calibrate_gamma,
    g_unscaled,
    residual_coeffs_to_dict,
    rms,
    sample_residual_coeffs,
    tau_perp,
    tau_phy_library,
)
from .mujoco_physics import import_mujoco

SCHEMA_ID = "aprwm.cap_x2.dataset.v1"
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class CAPX2DataConfig:
    timestep: float = 0.002
    duration_s: float = 2.0
    model_hz: float = 100.0
    n_train_scenes: int = 128
    n_val_scenes: int = 32
    n_test_scenes: int = 64
    traj_per_scene: int = 8
    train_scene_seed0: int = 70_000
    val_scene_seed0: int = 80_000
    test_scene_seed0: int = 90_000
    traj_seed0: int = 50_000
    calib_traj_seed0: int = 60_000
    residual_seed_offset: int = 17_777
    torque_scale: float = 0.35
    rho_grid: tuple[float, ...] = RHO_GRID


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _scene_ids(config: CAPX2DataConfig, split: SplitName) -> list[int]:
    if split == "train":
        return [config.train_scene_seed0 + i for i in range(config.n_train_scenes)]
    if split == "val":
        return [config.val_scene_seed0 + i for i in range(config.n_val_scenes)]
    if split == "test":
        return [config.test_scene_seed0 + i for i in range(config.n_test_scenes)]
    raise ValueError(split)


def _simulate_trajectory(
    *,
    model: Any,
    data: Any,
    theta: SceneTheta,
    coeffs: SceneResidualCoeffs,
    gamma: float,
    torques: np.ndarray,
    q0: np.ndarray,
    qd0: np.ndarray,
) -> dict[str, np.ndarray]:
    mujoco = import_mujoco()
    apply_scene_theta(model, data, theta)
    n = int(torques.shape[0])
    q = np.zeros((n, N_DOF), dtype=np.float64)
    qd = np.zeros((n, N_DOF), dtype=np.float64)
    qdd = np.zeros((n, N_DOF), dtype=np.float64)
    u = np.asarray(torques, dtype=np.float64)
    residual = np.zeros((n, N_DOF), dtype=np.float64)
    tau_p = np.zeros((n, N_DOF), dtype=np.float64)

    data.qpos[:] = np.asarray(q0, dtype=np.float64)
    data.qvel[:] = np.asarray(qd0, dtype=np.float64)
    if model.nu:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    for i in range(n):
        q[i] = np.asarray(data.qpos, dtype=np.float64)
        qd[i] = np.asarray(data.qvel, dtype=np.float64)
        qvel_pre = qd[i].copy()
        tp = tau_perp(q[i], qd[i], coeffs, gamma=gamma)
        tau_p[i] = tp
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[:N_DOF] = u[i] + tp
        mujoco.mj_step(model, data)
        qdd[i] = np.asarray(data.qacc, dtype=np.float64)
        # Accounting residual vs library (theta passive only) — includes tau_perp.
        residual[i] = oracle_force_residual(model, data, theta, qvel_pre=qvel_pre)

    return {
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "u": u,
        "residual": residual,
        "tau_perp": tau_p,
    }


def _calibrate_scene_gamma(
    *,
    model: Any,
    data: Any,
    theta: SceneTheta,
    coeffs: SceneResidualCoeffs,
    rho: float,
    config: CAPX2DataConfig,
    scene_seed: int,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Roll held-out calib traj at gamma=0; set gamma so RMS ratio ≈ rho."""

    n_phys = int(round(config.duration_s / config.timestep))
    ds_idx = _downsample_idx(n_phys, config.timestep, config.model_hz)
    rng = np.random.default_rng(scene_seed + 333)
    kind: ExcitationKind = "bandlimited"
    torques = torque_sequence(
        kind,
        n_steps=n_phys,
        dt=config.timestep,
        seed=config.calib_traj_seed0 + scene_seed,
        scale=config.torque_scale,
        duration_s=config.duration_s,
    )
    q0 = rng.uniform(-0.6, 0.6, size=N_DOF)
    qd0 = rng.uniform(-0.3, 0.3, size=N_DOF)
    arrays = _simulate_trajectory(
        model=model,
        data=data,
        theta=theta,
        coeffs=coeffs,
        gamma=0.0,
        torques=torques,
        q0=q0,
        qd0=qd0,
    )
    q = arrays["q"][ds_idx]
    qd = arrays["qd"][ds_idx]
    u = arrays["u"][ds_idx]
    g = g_unscaled(q, qd, coeffs)
    tau_phy = np.zeros_like(u)
    apply_scene_theta(model, data, theta)
    for i in range(q.shape[0]):
        tau_phy[i] = tau_phy_library(
            model, data, q=q[i], qd=qd[i], u=u[i], theta=theta
        )
    r_perp = rms(g)
    r_phy = rms(tau_phy)
    gamma = calibrate_gamma(r_perp=r_perp, r_phy=r_phy, rho=rho)
    meta = {
        "r_perp": r_perp,
        "r_phy": r_phy,
        "rho_target": float(rho),
        "gamma": float(gamma),
        "excitation": kind,
        "held_out": True,
        "in_train": False,
        "in_val": False,
        "in_test": False,
    }
    calib_arrays = {
        "q": q,
        "qd": qd,
        "qdd": arrays["qdd"][ds_idx],
        "u": u,
        "g_unscaled": g,
        "tau_phy": tau_phy,
    }
    return gamma, meta, calib_arrays


def build_rho_split(
    root: Path,
    *,
    split: SplitName,
    rho: float,
    config: CAPX2DataConfig,
) -> dict[str, Any]:
    model, data = load_capx_arm3(config.timestep)
    inventory = assert_capx_arm3(model)
    scene_ids = _scene_ids(config, split)
    split_dir = root / f"rho_{rho:.2f}" / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_phys = int(round(config.duration_s / config.timestep))
    ds_idx = _downsample_idx(n_phys, config.timestep, config.model_hz)

    all_q: list[np.ndarray] = []
    all_qd: list[np.ndarray] = []
    all_qdd: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_r: list[np.ndarray] = []
    all_theta: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []

    for scene_seed in scene_ids:
        rng = np.random.default_rng(scene_seed)
        theta = sample_scene_theta(rng)
        coeffs = sample_residual_coeffs(
            np.random.default_rng(scene_seed + config.residual_seed_offset)
        )
        gamma, calib_meta, calib_arr = _calibrate_scene_gamma(
            model=model,
            data=data,
            theta=theta,
            coeffs=coeffs,
            rho=rho,
            config=config,
            scene_seed=scene_seed,
        )
        scene_path = split_dir / f"scene_{scene_seed:06d}.h5"
        with h5py.File(scene_path, "w") as handle:
            handle.attrs["schema"] = SCHEMA_ID
            handle.attrs["host_plant_id"] = HOST_PLANT_ID
            handle.attrs["residual_family_id"] = RESIDUAL_FAMILY_ID
            handle.attrs["split"] = split
            handle.attrs["scene_seed"] = int(scene_seed)
            handle.attrs["rho"] = float(rho)
            handle.attrs["gamma"] = float(gamma)
            handle.attrs["theta_fingerprint"] = theta.fingerprint()
            handle.attrs["residual_fingerprint"] = coeffs.fingerprint()
            handle.create_dataset("theta", data=theta.as_vector())
            # Oracle-only residual coeffs — not packed into learner pools.
            handle.create_dataset("residual_coeffs", data=coeffs.as_vector())
            meta = handle.create_group("metadata")
            meta.attrs["theta_json"] = json.dumps(scene_theta_to_dict(theta), sort_keys=True)
            meta.attrs["residual_json"] = json.dumps(
                residual_coeffs_to_dict(coeffs), sort_keys=True
            )
            meta.attrs["calib_json"] = json.dumps(calib_meta, sort_keys=True)

            calib = handle.create_group("rho_calibration_trajectory")
            calib.attrs["held_out"] = True
            for key, arr in calib_arr.items():
                calib.create_dataset(key, data=arr, compression="gzip")
            calib_rows.append({"scene_seed": scene_seed, **calib_meta})

            for t_i in range(config.traj_per_scene):
                kind = EXCITATIONS[t_i % len(EXCITATIONS)]
                traj_seed = config.traj_seed0 + 1000 * scene_seed + t_i
                torques = torque_sequence(
                    kind,
                    n_steps=n_phys,
                    dt=config.timestep,
                    seed=traj_seed,
                    scale=config.torque_scale,
                    duration_s=config.duration_s,
                )
                q0 = rng.uniform(-0.6, 0.6, size=N_DOF)
                qd0 = rng.uniform(-0.3, 0.3, size=N_DOF)
                arrays = _simulate_trajectory(
                    model=model,
                    data=data,
                    theta=theta,
                    coeffs=coeffs,
                    gamma=gamma,
                    torques=torques,
                    q0=q0,
                    qd0=qd0,
                )
                q = arrays["q"][ds_idx]
                qd = arrays["qd"][ds_idx]
                qdd = arrays["qdd"][ds_idx]
                u = arrays["u"][ds_idx]
                r = arrays["residual"][ds_idx]
                grp = handle.create_group(f"traj_{t_i:02d}")
                grp.attrs["excitation"] = kind
                grp.attrs["traj_seed"] = int(traj_seed)
                for key, arr in (
                    ("q", q),
                    ("qd", qd),
                    ("qdd", qdd),
                    ("u", u),
                    ("residual", r),
                ):
                    grp.create_dataset(key, data=arr, compression="gzip")
                all_q.append(q)
                all_qd.append(qd)
                all_qdd.append(qdd)
                all_u.append(u)
                all_r.append(r)
                all_theta.append(np.repeat(theta.as_vector()[None, :], q.shape[0], axis=0))
                rows.append(
                    {
                        "scene_seed": scene_seed,
                        "traj_index": t_i,
                        "excitation": kind,
                        "n_steps_model": int(q.shape[0]),
                    }
                )

    q_cat = np.concatenate(all_q, axis=0)
    pool_path = root / f"rho_{rho:.2f}" / "data" / f"{split}_pool.h5"
    with h5py.File(pool_path, "w") as handle:
        handle.create_dataset("q", data=q_cat, compression="gzip")
        handle.create_dataset("qd", data=np.concatenate(all_qd), compression="gzip")
        handle.create_dataset("qdd", data=np.concatenate(all_qdd), compression="gzip")
        handle.create_dataset("u", data=np.concatenate(all_u), compression="gzip")
        handle.create_dataset("residual", data=np.concatenate(all_r), compression="gzip")
        handle.create_dataset("theta", data=np.concatenate(all_theta), compression="gzip")
        # Explicit: no residual coeffs / gamma / rho columns for learners.
        handle.attrs["learner_visible_keys"] = "q,qd,qdd,u,residual,theta"
        handle.attrs["rho"] = float(rho)
        handle.attrs["excludes_calibration"] = True

    summary = {
        "split": split,
        "rho": float(rho),
        "n_scenes": len(scene_ids),
        "n_trajectories": len(rows),
        "n_samples_model_rate": int(q_cat.shape[0]),
        "pool": str(pool_path),
        "inventory": inventory,
        "calibration": calib_rows,
        "trajectories": rows,
    }
    _write_json(root / f"rho_{rho:.2f}" / "data" / f"{split}_summary.json", summary)
    return summary


def build_rho_datasets(
    root: str | Path,
    *,
    rho: float,
    config: CAPX2DataConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX2DataConfig()
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    if "r10_c0" in str(out).replace("\\", "/"):
        raise RuntimeError("CAP-X2 data must not write under runs/r10_c0/")
    print(f"building CAP-X2 datasets rho={rho}...", flush=True)
    train = build_rho_split(out, split="train", rho=rho, config=cfg)
    val = build_rho_split(out, split="val", rho=rho, config=cfg)
    test = build_rho_split(out, split="test", rho=rho, config=cfg)
    payload = {
        "rho": float(rho),
        "residual_family_id": RESIDUAL_FAMILY_ID,
        "host_plant_id": HOST_PLANT_ID,
        "schema": SCHEMA_ID,
        "config": asdict(cfg),
        "train": {"n_scenes": train["n_scenes"], "n_samples": train["n_samples_model_rate"]},
        "val": {"n_scenes": val["n_scenes"], "n_samples": val["n_samples_model_rate"]},
        "test": {"n_scenes": test["n_scenes"], "n_samples": test["n_samples_model_rate"]},
        "calibration_held_out": True,
    }
    _write_json(out / f"rho_{rho:.2f}" / "dataset_summary.json", payload)
    return payload

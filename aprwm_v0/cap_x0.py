"""CAP-X0: benchmark / accounting / PureNN sanity on ``capx_arm3.v1``.

Does not claim capacity replacement. Does not unlock R10-C0.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from .capx_arm3_plant import (
    HOST_PLANT_ID,
    N_DOF,
    THETA_DIM,
    SceneTheta,
    apply_scene_theta,
    assert_capx_arm3,
    capx_arm3_xml,
    capx_arm3_xml_sha256,
    load_capx_arm3,
    oracle_force_residual,
    sample_scene_theta,
    scene_theta_to_dict,
)
from .mujoco_force import residual_nrmse
from .mujoco_physics import import_mujoco

PREREG_PATH = "REPORT/REG/CAPX/CAPX0_PREREG.md"
SCHEMA_ID = "aprwm.cap_x0.benchmark.v1"
NRMSE_ORACLE_MAX = 1.0e-4
VAR_Q_MIN = 1.0e-4
VAR_QD_MIN = 1.0e-3
VAR_QDD_MIN = 1.0e-2
PURENN_VAL_NRMSE_MAX = 0.25

ExcitationKind = Literal["multi_sine", "chirp", "bandlimited", "piecewise_smooth"]
EXCITATIONS: tuple[ExcitationKind, ...] = (
    "multi_sine",
    "chirp",
    "bandlimited",
    "piecewise_smooth",
)


@dataclass(frozen=True)
class CAPX0Config:
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
    nrmse_oracle_max: float = NRMSE_ORACLE_MAX
    var_q_min: float = VAR_Q_MIN
    var_qd_min: float = VAR_QD_MIN
    var_qdd_min: float = VAR_QDD_MIN
    purenn_hidden: int = 256
    purenn_epochs: int = 40
    purenn_batch: int = 512
    purenn_lr: float = 1.0e-3
    purenn_train_max_samples: int = 400_000
    purenn_val_nrmse_max: float = PURENN_VAL_NRMSE_MAX
    torque_scale: float = 0.35
    # Smoke / unit-test overrides keep defaults large for formal.


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _downsample_idx(n_phys: int, dt: float, model_hz: float) -> np.ndarray:
    stride = max(1, int(round(1.0 / (float(model_hz) * float(dt)))))
    return np.arange(0, n_phys, stride, dtype=np.int64)


def torque_sequence(
    kind: ExcitationKind,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    scale: float,
    duration_s: float,
) -> np.ndarray:
    """Smooth 3-DoF generalized torques (no discontinuous impulses)."""

    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64) * dt
    u = np.zeros((n_steps, N_DOF), dtype=np.float64)
    if kind == "multi_sine":
        for j in range(N_DOF):
            for k in range(3):
                amp = float(rng.uniform(0.3, 1.0)) * scale
                freq = float(rng.uniform(0.15, 1.2))
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                u[:, j] += amp * np.sin(2.0 * np.pi * freq * t + phase)
        return u
    if kind == "chirp":
        for j in range(N_DOF):
            amp = float(rng.uniform(0.6, 1.2)) * scale
            f0 = float(rng.uniform(0.1, 0.3))
            f1 = float(rng.uniform(0.8, 1.5))
            k = (f1 - f0) / max(duration_s, dt)
            phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
            u[:, j] = amp * np.sin(phase + float(rng.uniform(0, np.pi)))
        return u
    if kind == "bandlimited":
        # Smooth colored noise via cumulative sum of Gaussian then high-pass-ish detrend.
        raw = rng.normal(0.0, 1.0, size=(n_steps, N_DOF))
        kernel = np.exp(-0.5 * (np.arange(-25, 26) / 8.0) ** 2)
        kernel /= kernel.sum()
        for j in range(N_DOF):
            sm = np.convolve(raw[:, j], kernel, mode="same")
            sm = sm - np.mean(sm)
            u[:, j] = scale * sm / (np.std(sm) + 1.0e-8)
        return u
    if kind == "piecewise_smooth":
        hold = max(1, int(round(0.25 / dt)))
        for j in range(N_DOF):
            levels: list[float] = []
            while len(levels) < n_steps:
                a = float(rng.uniform(-scale, scale))
                b = float(rng.uniform(-scale, scale))
                ramp = np.linspace(a, b, hold)
                levels.extend(float(x) for x in ramp)
            u[:, j] = np.asarray(levels[:n_steps], dtype=np.float64)
        return u
    raise ValueError(f"unknown excitation {kind}")


def _simulate_trajectory(
    *,
    model: Any,
    data: Any,
    theta: SceneTheta,
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

    data.qpos[:] = np.asarray(q0, dtype=np.float64)
    data.qvel[:] = np.asarray(qd0, dtype=np.float64)
    if model.nu:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    for i in range(n):
        q[i] = np.asarray(data.qpos, dtype=np.float64)
        qd[i] = np.asarray(data.qvel, dtype=np.float64)
        qvel_pre = qd[i].copy()
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[:N_DOF] = u[i]
        mujoco.mj_step(model, data)
        qdd[i] = np.asarray(data.qacc, dtype=np.float64)
        residual[i] = oracle_force_residual(model, data, theta, qvel_pre=qvel_pre)

    return {"q": q, "qd": qd, "qdd": qdd, "u": u, "residual": residual}


def _scene_ids(config: CAPX0Config, split: str) -> list[int]:
    if split == "train":
        return [config.train_scene_seed0 + i for i in range(config.n_train_scenes)]
    if split == "val":
        return [config.val_scene_seed0 + i for i in range(config.n_val_scenes)]
    if split == "test":
        return [config.test_scene_seed0 + i for i in range(config.n_test_scenes)]
    raise ValueError(split)


def build_split_dataset(
    root: Path,
    *,
    split: str,
    config: CAPX0Config,
) -> dict[str, Any]:
    mujoco = import_mujoco()
    model, data = load_capx_arm3(config.timestep)
    inventory = assert_capx_arm3(model)
    scene_ids = _scene_ids(config, split)
    split_dir = root / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_phys = int(round(config.duration_s / config.timestep))
    ds_idx = _downsample_idx(n_phys, config.timestep, config.model_hz)
    fingerprints: list[str] = []
    theta_vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    all_q: list[np.ndarray] = []
    all_qd: list[np.ndarray] = []
    all_qdd: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_r: list[np.ndarray] = []
    all_theta: list[np.ndarray] = []

    for s_i, scene_seed in enumerate(scene_ids):
        rng = np.random.default_rng(scene_seed)
        theta = sample_scene_theta(rng)
        fp = theta.fingerprint()
        fingerprints.append(fp)
        theta_vectors.append(theta.as_vector())
        scene_path = split_dir / f"scene_{scene_seed:06d}.h5"
        with h5py.File(scene_path, "w") as handle:
            handle.attrs["schema"] = SCHEMA_ID
            handle.attrs["host_plant_id"] = HOST_PLANT_ID
            handle.attrs["split"] = split
            handle.attrs["scene_seed"] = int(scene_seed)
            handle.attrs["theta_fingerprint"] = fp
            handle.create_dataset("theta", data=theta.as_vector())
            meta = handle.create_group("metadata")
            meta.attrs["json"] = json.dumps(scene_theta_to_dict(theta), sort_keys=True)
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
                    torques=torques,
                    q0=q0,
                    qd0=qd0,
                )
                # Downsample for learner tables.
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
                        "nrmse_oracle": float(residual_nrmse(r, u)),
                        "path": str(scene_path),
                    }
                )

    q_cat = np.concatenate(all_q, axis=0)
    qd_cat = np.concatenate(all_qd, axis=0)
    qdd_cat = np.concatenate(all_qdd, axis=0)
    u_cat = np.concatenate(all_u, axis=0)
    r_cat = np.concatenate(all_r, axis=0)
    theta_cat = np.concatenate(all_theta, axis=0)

    pool_path = root / "data" / f"{split}_pool.h5"
    with h5py.File(pool_path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["split"] = split
        for key, arr in (
            ("q", q_cat),
            ("qd", qd_cat),
            ("qdd", qdd_cat),
            ("u", u_cat),
            ("residual", r_cat),
            ("theta", theta_cat),
        ):
            handle.create_dataset(key, data=arr, compression="gzip")

    summary = {
        "split": split,
        "n_scenes": len(scene_ids),
        "scene_seeds": scene_ids,
        "fingerprints": fingerprints,
        "n_trajectories": len(rows),
        "n_samples_model_rate": int(q_cat.shape[0]),
        "mean_nrmse_oracle": float(residual_nrmse(r_cat, u_cat)),
        "var_q": np.var(q_cat, axis=0).tolist(),
        "var_qd": np.var(qd_cat, axis=0).tolist(),
        "var_qdd": np.var(qdd_cat, axis=0).tolist(),
        "pool_path": str(pool_path),
        "plant_inventory": inventory,
        "trajectories": rows,
    }
    _write_json(root / "data" / f"{split}_summary.json", summary)
    return summary


def _load_pool(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {k: np.asarray(handle[k], dtype=np.float64) for k in handle.keys()}


def train_purenn_h256(
    train_pool: dict[str, np.ndarray],
    val_pool: dict[str, np.ndarray],
    *,
    config: CAPX0Config,
    device: str | None = None,
) -> dict[str, Any]:
    import torch
    from torch import nn

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    def pack(pool: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([pool["q"], pool["qd"], pool["u"], pool["theta"]], axis=1)
        y = pool["qdd"]
        return x.astype(np.float32), y.astype(np.float32)

    x_tr, y_tr = pack(train_pool)
    x_va, y_va = pack(val_pool)
    if x_tr.shape[0] > config.purenn_train_max_samples:
        rng = np.random.default_rng(12345)
        idx = rng.choice(x_tr.shape[0], size=config.purenn_train_max_samples, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]

    # Standardize on train.
    x_mean = x_tr.mean(axis=0)
    x_std = x_tr.std(axis=0) + 1.0e-6
    y_mean = y_tr.mean(axis=0)
    y_std = y_tr.std(axis=0) + 1.0e-6
    x_tr_n = (x_tr - x_mean) / x_std
    y_tr_n = (y_tr - y_mean) / y_std
    x_va_n = (x_va - x_mean) / x_std

    h = int(config.purenn_hidden)
    in_dim = int(x_tr.shape[1])
    model = nn.Sequential(
        nn.Linear(in_dim, h),
        nn.SiLU(),
        nn.Linear(h, h),
        nn.SiLU(),
        nn.Linear(h, h),
        nn.SiLU(),
        nn.Linear(h, N_DOF),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.purenn_lr, weight_decay=1.0e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(config.purenn_epochs)))
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_tr_n).to(dev)
    y_t = torch.from_numpy(y_tr_n).to(dev)
    n = x_t.shape[0]
    batch = int(config.purenn_batch)
    history: list[float] = []
    model.train()
    for _epoch in range(int(config.purenn_epochs)):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        count = 0
        for i0 in range(0, n, batch):
            idx = perm[i0 : i0 + batch]
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * int(idx.numel())
            count += int(idx.numel())
        sched.step()
        history.append(total / max(count, 1))

    model.eval()
    with torch.no_grad():
        pred_tr = model(x_t).cpu().numpy() * y_std + y_mean
        pred_va = (
            model(torch.from_numpy(x_va_n).to(dev)).cpu().numpy() * y_std + y_mean
        )
    def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
        rmse = float(np.sqrt(np.mean(np.square(pred - ref))))
        rms_ref = float(np.sqrt(np.mean(np.square(ref))))
        return rmse / (rms_ref + 1.0e-8)

    nrmse_tr = _nrmse(pred_tr, y_tr)
    nrmse_va = _nrmse(pred_va, y_va)

    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "hidden": h,
        "n_params": n_params,
        "device": device,
        "train_samples": int(x_tr.shape[0]),
        "val_samples": int(x_va.shape[0]),
        "train_loss_last": float(history[-1]) if history else float("nan"),
        "train_nrmse_qdd": nrmse_tr,
        "val_nrmse_qdd": nrmse_va,
        "val_rmse_qdd": float(np.sqrt(np.mean(np.square(pred_va - y_va)))),
        "epochs": int(config.purenn_epochs),
        "input_dim": in_dim,
        "theta_in_input": True,
        "note": "X0 sanity only; not a capacity claim",
    }


def _split_integrity(
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    fp_tr = set(train["fingerprints"])
    fp_va = set(val["fingerprints"])
    fp_te = set(test["fingerprints"])
    dup_tv = sorted(fp_tr & fp_va)
    dup_tt = sorted(fp_tr & fp_te)
    dup_vt = sorted(fp_va & fp_te)
    seed_tr = set(train["scene_seeds"])
    seed_va = set(val["scene_seeds"])
    seed_te = set(test["scene_seeds"])
    return {
        "duplicate_theta_train_val": len(dup_tv),
        "duplicate_theta_train_test": len(dup_tt),
        "duplicate_theta_val_test": len(dup_vt),
        "duplicate_scene_seeds": bool(
            (seed_tr & seed_va) or (seed_tr & seed_te) or (seed_va & seed_te)
        ),
        "ok": (
            len(dup_tv) == 0
            and len(dup_tt) == 0
            and len(dup_vt) == 0
            and not ((seed_tr & seed_va) or (seed_tr & seed_te) or (seed_va & seed_te))
        ),
    }


def run_cap_x0(
    output: str | Path,
    *,
    config: CAPX0Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX0Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)
    (root / "capx_arm3.xml").write_text(capx_arm3_xml(cfg.timestep), encoding="utf-8")
    _write_json(
        root / "plant_inventory.json",
        {
            "host_plant_id": HOST_PLANT_ID,
            "xml_sha256": capx_arm3_xml_sha256(cfg.timestep),
            "theta_dim": THETA_DIM,
            "n_dof": N_DOF,
            "config": asdict(cfg),
        },
    )

    train = build_split_dataset(root, split="train", config=cfg)
    val = build_split_dataset(root, split="val", config=cfg)
    test = build_split_dataset(root, split="test", config=cfg)

    # G0: oracle residual on train+val
    tr_pool = _load_pool(Path(train["pool_path"]))
    va_pool = _load_pool(Path(val["pool_path"]))
    te_pool = _load_pool(Path(test["pool_path"]))
    r_tv = np.concatenate([tr_pool["residual"], va_pool["residual"]], axis=0)
    u_tv = np.concatenate([tr_pool["u"], va_pool["u"]], axis=0)
    nrmse_oracle = float(residual_nrmse(r_tv, u_tv))
    g0 = bool(nrmse_oracle < cfg.nrmse_oracle_max)

    # G1: excitation coverage on train
    var_q = np.asarray(train["var_q"], dtype=np.float64)
    var_qd = np.asarray(train["var_qd"], dtype=np.float64)
    var_qdd = np.asarray(train["var_qdd"], dtype=np.float64)
    g1 = bool(
        np.all(var_q > cfg.var_q_min)
        and np.all(var_qd > cfg.var_qd_min)
        and np.all(var_qdd > cfg.var_qdd_min)
    )

    # G2: split integrity
    integrity = _split_integrity(train, val, test)
    g2 = bool(integrity["ok"])

    # G3: PureNN H=256 sanity
    purenn = train_purenn_h256(tr_pool, va_pool, config=cfg)
    g3 = bool(purenn["val_nrmse_qdd"] <= cfg.purenn_val_nrmse_max)

    passed = bool(g0 and g1 and g2 and g3)
    summary = {
        "stage": "CAP-X0",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": (
            "CAP-X benchmark closed: oracle accounting, excitation, "
            "split integrity, and PureNN H=256 learnability"
            if passed
            else "CAP-X0 gates not all satisfied; no capacity claims"
        ),
        "real_physics": False,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "unlocks_cap_x1": bool(passed),
        "config": asdict(cfg),
        "xml_sha256": capx_arm3_xml_sha256(cfg.timestep),
        "splits": {
            "train": {
                "n_scenes": train["n_scenes"],
                "n_samples": train["n_samples_model_rate"],
                "mean_nrmse_oracle": train["mean_nrmse_oracle"],
            },
            "val": {
                "n_scenes": val["n_scenes"],
                "n_samples": val["n_samples_model_rate"],
                "mean_nrmse_oracle": val["mean_nrmse_oracle"],
            },
            "test": {
                "n_scenes": test["n_scenes"],
                "n_samples": test["n_samples_model_rate"],
                "mean_nrmse_oracle": test["mean_nrmse_oracle"],
            },
        },
        "metrics": {
            "nrmse_oracle_train_val": nrmse_oracle,
            "var_q_train": var_q.tolist(),
            "var_qd_train": var_qd.tolist(),
            "var_qdd_train": var_qdd.tolist(),
            "purenn_val_nrmse_qdd": purenn["val_nrmse_qdd"],
            "purenn_n_params": purenn["n_params"],
        },
        "purenn": purenn,
        "split_integrity": integrity,
        "gates": {
            "G0_physics_accounting": g0,
            "G1_excitation_coverage": g1,
            "G2_split_integrity": g2,
            "G3_neural_upper_bound": g3,
        },
        "cap_x0_passed": passed,
        "pass_iff": (
            "G0 NRMSE(r_phy)<1e-4; G1 per-DoF variance floors; "
            "G2 no theta/seed leakage; G3 PureNN H=256 val NRMSE(qdd)<=0.25"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

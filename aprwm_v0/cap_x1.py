"""CAP-X1: matched-family capacity replacement at rho=0 (mu=0).

Convention (frozen): Hybrid residual outputs generalized force tau_res;
qdd_hat = qdd_phy + M^{-1} tau_res. H_hybrid=0 is Physics-only.
Does not unlock R10. Does not open rho>0.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .capx_arm3_plant import (
    HOST_PLANT_ID,
    N_DOF,
    THETA_DIM,
    SceneTheta,
    apply_scene_theta,
    load_capx_arm3,
    per_dof_passive_force,
)
from .mujoco_force import get_bias_force, get_mass_matrix, residual_nrmse
from .mujoco_physics import import_mujoco

PREREG_PATH = "REPORT/REG/CAPX/CAPX1_PREREG.md"
SCHEMA_ID = "aprwm.cap_x1.capacity.v1"

PURE_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
HYBRID_WIDTHS: tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256)
TRAIN_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105)
ROLLOUT_HORIZONS: tuple[int, ...] = (10, 50, 100)

# Frozen planner / matched thresholds from CAPX1_PREREG.
PLAN_HORIZON_S = 1.0
MODEL_DT = 0.01
ACTION_KNOTS = 20
CEM_CANDIDATES = 256
CEM_ITERS = 5
CEM_ELITE_FRAC = 0.10
LAMBDA_V = 0.05
LAMBDA_U = 0.01
EPS_Q = 0.15
E1_MATCH = 1.05
ROLLOUT_MATCH = 1.10
PLAN_MATCH_DELTA = 0.05


@dataclass(frozen=True)
class CAPX1Config:
    x0_data: str = "runs/cap_x0/formal"
    epochs: int = 40
    batch: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    train_max_samples: int = 400_000
    # Eval budgets (same for all models).
    rollout_episodes: int = 32
    plan_tasks: int = 8
    latency_warmup: int = 10
    latency_steps: int = 100
    require_x0_summary: bool = True
    # Smoke overrides.
    pure_widths: tuple[int, ...] = PURE_WIDTHS
    hybrid_widths: tuple[int, ...] = HYBRID_WIDTHS
    train_seeds: tuple[int, ...] = TRAIN_SEEDS


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_pool(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {k: np.asarray(handle[k], dtype=np.float64) for k in handle.keys()}


def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.square(pred - ref))))
    rms = float(np.sqrt(np.mean(np.square(ref))))
    return rmse / (rms + 1.0e-8)


def physics_qdd_and_minv(
    model: Any,
    data: Any,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    u: np.ndarray,
    theta: SceneTheta,
    theta_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qdd_phy, M_inv) with tau_res=0. Convention: M qdd = u + passive - bias."""

    mujoco = import_mujoco()
    fp = theta.fingerprint()
    if theta_cache is None or theta_cache.get("fp") != fp:
        apply_scene_theta(model, data, theta)
        if theta_cache is not None:
            theta_cache["fp"] = fp
            theta_cache["theta"] = theta
    else:
        theta = theta_cache["theta"]
    data.qpos[:] = np.asarray(q, dtype=np.float64)
    data.qvel[:] = np.asarray(qd, dtype=np.float64)
    if model.nu:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    mass = get_mass_matrix(model, data)
    bias = get_bias_force(data)
    passive = per_dof_passive_force(qd, theta)
    rhs = np.asarray(u, dtype=np.float64) + passive - bias
    qdd = np.linalg.solve(mass, rhs)
    minv = np.linalg.inv(mass)
    return qdd, minv


def precompute_physics_cache(
    pool: dict[str, np.ndarray],
    *,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    if cache_path.is_file():
        with h5py.File(cache_path, "r") as handle:
            return {k: np.asarray(handle[k], dtype=np.float64) for k in handle.keys()}

    model, data = load_capx_arm3(0.002)
    n = int(pool["q"].shape[0])
    qdd_phy = np.zeros((n, N_DOF), dtype=np.float64)
    minv = np.zeros((n, N_DOF, N_DOF), dtype=np.float64)
    # Group by theta fingerprint to reduce mj_setConst calls.
    theta = pool["theta"]
    # Round for grouping stability.
    keys = [tuple(np.round(theta[i], decimals=10)) for i in range(n)]
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)

    for key, idxs in groups.items():
        th = SceneTheta.from_vector(np.asarray(key, dtype=np.float64))
        apply_scene_theta(model, data, th)
        mujoco = import_mujoco()
        for i in idxs:
            data.qpos[:] = pool["q"][i]
            data.qvel[:] = pool["qd"][i]
            if model.nu:
                data.ctrl[:] = 0.0
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            mass = get_mass_matrix(model, data)
            bias = get_bias_force(data)
            passive = per_dof_passive_force(pool["qd"][i], th)
            rhs = pool["u"][i] + passive - bias
            qdd_phy[i] = np.linalg.solve(mass, rhs)
            minv[i] = np.linalg.inv(mass)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("qdd_phy", data=qdd_phy, compression="gzip")
        handle.create_dataset("minv", data=minv, compression="gzip")
    return {"qdd_phy": qdd_phy, "minv": minv}


def mlp_param_count(in_dim: int, hidden: int, out_dim: int = N_DOF) -> int:
    if hidden <= 0:
        return 0
    # 3 hidden SiLU layers + linear head (matches CAP-X0/X1 prereg).
    return (
        in_dim * hidden
        + hidden
        + hidden * hidden
        + hidden
        + hidden * hidden
        + hidden
        + hidden * out_dim
        + out_dim
    )


def approx_mlp_macs(in_dim: int, hidden: int, out_dim: int = N_DOF) -> int:
    if hidden <= 0:
        return 0
    return (
        in_dim * hidden
        + hidden * hidden
        + hidden * hidden
        + hidden * out_dim
    )


def build_mlp(in_dim: int, hidden: int, out_dim: int = N_DOF):
    import torch
    from torch import nn

    if hidden <= 0:
        return None
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _pack_x(pool: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [pool["q"], pool["qd"], pool["u"], pool["theta"]], axis=1
    ).astype(np.float32)


def train_pure(
    *,
    hidden: int,
    seed: int,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    config: CAPX1Config,
    device: str,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = _pack_x(train)
    y_tr = train["qdd"].astype(np.float32)
    x_va = _pack_x(val)
    y_va = val["qdd"].astype(np.float32)
    if x_tr.shape[0] > config.train_max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_tr.shape[0], size=config.train_max_samples, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]

    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1.0e-6
    y_mean, y_std = y_tr.mean(0), y_tr.std(0) + 1.0e-6
    x_trn = (x_tr - x_mean) / x_std
    y_trn = (y_tr - y_mean) / y_std
    x_van = (x_va - x_mean) / x_std

    dev = torch.device(device)
    model = build_mlp(x_tr.shape[1], hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, config.epochs))
    x_t = torch.from_numpy(x_trn).to(dev)
    y_t = torch.from_numpy(y_trn).to(dev)
    best_state = None
    best_val = float("inf")
    bad = 0
    history: list[float] = []
    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(x_t.shape[0], device=dev)
        total = 0.0
        count = 0
        for i0 in range(0, x_t.shape[0], config.batch):
            idx = perm[i0 : i0 + config.batch]
            loss = nn.functional.mse_loss(model(x_t[idx]), y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * int(idx.numel())
            count += int(idx.numel())
        sched.step()
        history.append(total / max(count, 1))
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_van).to(dev)).cpu().numpy() * y_std + y_mean
        val_e = _nrmse(pred, y_va)
        if val_e + 1.0e-12 < best_val:
            best_val = val_e
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(q, qd, u, theta):
        model.eval()
        x = np.concatenate([q, qd, u, theta], axis=-1).astype(np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        xn = (x - x_mean) / x_std
        with torch.no_grad():
            out = model(torch.from_numpy(xn).to(dev)).cpu().numpy() * y_std + y_mean
        return out[0] if single else out

    return {
        "family": "pure",
        "hidden": hidden,
        "seed": seed,
        "n_params": mlp_param_count(x_tr.shape[1], hidden),
        "neural_macs": approx_mlp_macs(x_tr.shape[1], hidden),
        "best_val_nrmse": float(best_val),
        "epochs_ran": len(history),
        "predict": predict,
        "norm": {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
        "state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
        "in_dim": int(x_tr.shape[1]),
    }


def train_hybrid(
    *,
    hidden: int,
    seed: int,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    train_cache: dict[str, np.ndarray],
    val_cache: dict[str, np.ndarray],
    config: CAPX1Config,
    device: str,
) -> dict[str, Any]:
    """Train residual force MLP; H=0 returns Physics-only predictor."""

    import torch
    from torch import nn

    in_dim = 3 * N_DOF + THETA_DIM
    plant_model, plant_data = load_capx_arm3(0.002)
    theta_cache: dict[str, Any] = {}

    def physics_predict(q, qd, u, theta_vec):
        th = SceneTheta.from_vector(theta_vec)
        qdd, _ = physics_qdd_and_minv(
            plant_model,
            plant_data,
            q=q,
            qd=qd,
            u=u,
            theta=th,
            theta_cache=theta_cache,
        )
        return qdd

    if hidden <= 0:
        return {
            "family": "hybrid",
            "hidden": 0,
            "seed": seed,
            "n_params": 0,
            "neural_macs": 0,
            "best_val_nrmse": float(_nrmse(val_cache["qdd_phy"], val["qdd"])),
            "epochs_ran": 0,
            "predict": physics_predict,
            "physics_only": True,
        }

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = _pack_x(train)
    y_tr = train["qdd"].astype(np.float32)
    minv_tr = train_cache["minv"].astype(np.float32)
    qdd_phy_tr = train_cache["qdd_phy"].astype(np.float32)
    x_va = _pack_x(val)
    y_va = val["qdd"].astype(np.float32)
    minv_va = val_cache["minv"].astype(np.float32)
    qdd_phy_va = val_cache["qdd_phy"].astype(np.float32)

    if x_tr.shape[0] > config.train_max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_tr.shape[0], size=config.train_max_samples, replace=False)
        x_tr = x_tr[idx]
        y_tr = y_tr[idx]
        minv_tr = minv_tr[idx]
        qdd_phy_tr = qdd_phy_tr[idx]

    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1.0e-6
    # tau_res target: M (qdd - qdd_phy); minv = M^{-1}.
    m_tr = np.linalg.inv(minv_tr)
    tau_tr = np.einsum("nij,nj->ni", m_tr, y_tr - qdd_phy_tr).astype(np.float32)
    tau_mean, tau_std = tau_tr.mean(0), tau_tr.std(0) + 1.0e-6

    x_trn = (x_tr - x_mean) / x_std
    tau_trn = (tau_tr - tau_mean) / tau_std
    x_van = (x_va - x_mean) / x_std

    dev = torch.device(device)
    net = build_mlp(in_dim, hidden, out_dim=N_DOF).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, config.epochs))
    x_t = torch.from_numpy(x_trn).to(dev)
    tau_t = torch.from_numpy(tau_trn).to(dev)
    best_state = None
    best_val = float("inf")
    bad = 0
    history: list[float] = []
    for _ in range(config.epochs):
        net.train()
        perm = torch.randperm(x_t.shape[0], device=dev)
        total = 0.0
        count = 0
        for i0 in range(0, x_t.shape[0], config.batch):
            idx = perm[i0 : i0 + config.batch]
            loss = nn.functional.mse_loss(net(x_t[idx]), tau_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().item()) * int(idx.numel())
            count += int(idx.numel())
        sched.step()
        history.append(total / max(count, 1))
        net.eval()
        with torch.no_grad():
            tau_hat = net(torch.from_numpy(x_van).to(dev)).cpu().numpy() * tau_std + tau_mean
        qdd_hat = qdd_phy_va + np.einsum("nij,nj->ni", minv_va, tau_hat)
        val_e = _nrmse(qdd_hat, y_va)
        if val_e + 1.0e-12 < best_val:
            best_val = val_e
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    def predict(q, qd, u, theta_vec):
        th = SceneTheta.from_vector(theta_vec)
        qdd_phy, minv = physics_qdd_and_minv(
            plant_model,
            plant_data,
            q=q,
            qd=qd,
            u=u,
            theta=th,
            theta_cache=theta_cache,
        )
        x = np.concatenate([q, qd, u, theta_vec], axis=-1).astype(np.float32)
        xn = (x - x_mean) / x_std
        net.eval()
        with torch.no_grad():
            tau = (
                net(torch.from_numpy(xn[None, :]).to(dev)).cpu().numpy()[0] * tau_std
                + tau_mean
            )
        return qdd_phy + minv @ tau

    return {
        "family": "hybrid",
        "hidden": hidden,
        "seed": seed,
        "n_params": mlp_param_count(in_dim, hidden),
        "neural_macs": approx_mlp_macs(in_dim, hidden),
        "best_val_nrmse": float(best_val),
        "epochs_ran": len(history),
        "predict": predict,
        "physics_only": False,
        "norm": {
            "x_mean": x_mean,
            "x_std": x_std,
            "tau_mean": tau_mean,
            "tau_std": tau_std,
        },
        "state_dict": {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()},
        "in_dim": in_dim,
    }


def _iter_test_episodes(x0_root: Path, *, max_episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    test_dir = x0_root / "data" / "test"
    files = sorted(test_dir.glob("scene_*.h5"))
    episodes: list[dict[str, Any]] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            theta = np.asarray(handle["theta"], dtype=np.float64)
            for key in handle.keys():
                if not key.startswith("traj_"):
                    continue
                grp = handle[key]
                episodes.append(
                    {
                        "theta": theta,
                        "q": np.asarray(grp["q"], dtype=np.float64),
                        "qd": np.asarray(grp["qd"], dtype=np.float64),
                        "qdd": np.asarray(grp["qdd"], dtype=np.float64),
                        "u": np.asarray(grp["u"], dtype=np.float64),
                    }
                )
    rng.shuffle(episodes)
    return episodes[:max_episodes]


def evaluate_one_step(
    predict: Callable,
    pool: dict[str, np.ndarray],
    *,
    cache: dict[str, np.ndarray] | None = None,
    physics_only: bool = False,
    max_samples: int = 20000,
    seed: int = 0,
) -> float:
    n = pool["q"].shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
    if physics_only and cache is not None:
        return _nrmse(cache["qdd_phy"][idx], pool["qdd"][idx])
    preds = []
    for i in idx:
        preds.append(
            predict(pool["q"][i], pool["qd"][i], pool["u"][i], pool["theta"][i])
        )
    return _nrmse(np.asarray(preds), pool["qdd"][idx])


def evaluate_rollout(
    predict: Callable,
    episodes: list[dict[str, Any]],
    *,
    dt: float = MODEL_DT,
) -> float:
    # Pool is already at ~100Hz (dt=0.01). Horizons in model steps.
    errors: list[float] = []
    for ep in episodes:
        q = ep["q"]
        qd = ep["qd"]
        u = ep["u"]
        theta = ep["theta"]
        n = q.shape[0]
        for h in ROLLOUT_HORIZONS:
            if n <= h + 1:
                continue
            # Multiple starts per episode.
            starts = [0, n // 4, n // 2]
            for t0 in starts:
                if t0 + h >= n:
                    continue
                q_hat = q[t0].copy()
                qd_hat = qd[t0].copy()
                for k in range(h):
                    uu = u[t0 + k]
                    qdd = np.asarray(predict(q_hat, qd_hat, uu, theta), dtype=np.float64)
                    q_hat = q_hat + qd_hat * dt
                    qd_hat = qd_hat + qdd * dt
                err = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.concatenate([q_hat, qd_hat])
                                - np.concatenate([q[t0 + h], qd[t0 + h]])
                            )
                        )
                    )
                )
                errors.append(err)
    return float(np.mean(errors)) if errors else float("nan")


def _predict_supports_batch(predict: Callable) -> bool:
    """True if predict(q,qd,u,theta) accepts batched (B,n) arrays (PureNN)."""
    try:
        q = np.zeros((2, N_DOF), dtype=np.float64)
        qd = np.zeros((2, N_DOF), dtype=np.float64)
        u = np.zeros((2, N_DOF), dtype=np.float64)
        theta = np.zeros((2, THETA_DIM), dtype=np.float64)
        out = np.asarray(predict(q, qd, u, theta))
        return out.shape == (2, N_DOF)
    except Exception:
        return False


def _cem_plan(
    predict: Callable,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    theta: np.ndarray,
    q_star: np.ndarray,
    q_abs_max: np.ndarray,
    u_scale: float,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    horizon_steps = int(round(PLAN_HORIZON_S / MODEL_DT))
    knots = ACTION_KNOTS
    # Piecewise-constant controls over knot segments.
    seg = max(1, horizon_steps // knots)
    mean = np.zeros((knots, N_DOF), dtype=np.float64)
    std = np.full((knots, N_DOF), u_scale, dtype=np.float64)
    t0 = time.perf_counter()
    best_cost = float("inf")
    best_ok = False
    n_elite = max(1, int(round(CEM_ELITE_FRAC * CEM_CANDIDATES)))
    batchable = _predict_supports_batch(predict)
    theta_b = np.broadcast_to(np.asarray(theta, dtype=np.float64), (CEM_CANDIDATES, THETA_DIM)).copy()
    for _ in range(CEM_ITERS):
        samples = rng.normal(mean[None, :, :], std[None, :, :], size=(CEM_CANDIDATES, knots, N_DOF))
        costs = np.zeros(CEM_CANDIDATES, dtype=np.float64)
        oks = np.zeros(CEM_CANDIDATES, dtype=bool)
        if batchable:
            q = np.broadcast_to(q0, (CEM_CANDIDATES, N_DOF)).copy()
            qd = np.broadcast_to(qd0, (CEM_CANDIDATES, N_DOF)).copy()
            violated = np.zeros(CEM_CANDIDATES, dtype=bool)
            for t in range(horizon_steps):
                u = samples[:, min(t // seg, knots - 1), :]
                qdd = np.asarray(predict(q, qd, u, theta_b), dtype=np.float64)
                q = q + qd * MODEL_DT
                qd = qd + qdd * MODEL_DT
                violated |= np.any(np.abs(q) > q_abs_max, axis=1)
                costs += (
                    np.sum((q - q_star) ** 2, axis=1)
                    + LAMBDA_V * np.sum(qd**2, axis=1)
                    + LAMBDA_U * np.sum(u**2, axis=1)
                )
            costs = costs + 1.0e3 * violated.astype(np.float64)
            oks = (~violated) & (np.max(np.abs(q - q_star), axis=1) < EPS_Q)
        else:
            for c in range(CEM_CANDIDATES):
                q = q0.copy()
                qd = qd0.copy()
                cost = 0.0
                violated = False
                for t in range(horizon_steps):
                    u = samples[c, min(t // seg, knots - 1)]
                    qdd = np.asarray(predict(q, qd, u, theta), dtype=np.float64)
                    q = q + qd * MODEL_DT
                    qd = qd + qdd * MODEL_DT
                    if np.any(np.abs(q) > q_abs_max):
                        violated = True
                    cost += float(
                        np.sum((q - q_star) ** 2)
                        + LAMBDA_V * np.sum(qd**2)
                        + LAMBDA_U * np.sum(u**2)
                    )
                costs[c] = cost + (1.0e3 if violated else 0.0)
                oks[c] = (not violated) and float(np.max(np.abs(q - q_star))) < EPS_Q
        elite_idx = np.argpartition(costs, n_elite - 1)[:n_elite]
        mean = samples[elite_idx].mean(axis=0)
        std = samples[elite_idx].std(axis=0) + 1.0e-3
        j = int(np.argmin(costs))
        if costs[j] < best_cost:
            best_cost = float(costs[j])
            best_ok = bool(oks[j])
    wall = time.perf_counter() - t0
    return best_ok, wall


def evaluate_planning(
    predict: Callable,
    episodes: list[dict[str, Any]],
    *,
    train_q: np.ndarray,
    train_u: np.ndarray,
    seed: int,
    n_tasks: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    q_lo = np.percentile(train_q, 5, axis=0)
    q_hi = np.percentile(train_q, 95, axis=0)
    q_abs_max = np.maximum(np.abs(q_lo), np.abs(q_hi)) * 1.25 + 0.5
    u_scale = float(np.std(train_u) + 0.1)
    successes = 0
    walls: list[float] = []
    n_tasks = min(n_tasks, len(episodes))
    for i in range(n_tasks):
        ep = episodes[i]
        q0 = ep["q"][0]
        qd0 = ep["qd"][0]
        theta = ep["theta"]
        q_star = rng.uniform(q_lo, q_hi)
        ok, wall = _cem_plan(
            predict,
            q0=q0,
            qd0=qd0,
            theta=theta,
            q_star=q_star,
            q_abs_max=q_abs_max,
            u_scale=u_scale,
            rng=rng,
        )
        successes += int(ok)
        walls.append(wall)
    return {
        "S_plan": float(successes / max(n_tasks, 1)),
        "cem_wall_mean_s": float(np.mean(walls)) if walls else float("nan"),
    }


def measure_step_latency(predict: Callable, pool: dict[str, np.ndarray], *, warmup: int, steps: int) -> float:
    n = pool["q"].shape[0]
    idx = np.arange(min(n, warmup + steps))
    for i in idx[:warmup]:
        predict(pool["q"][i], pool["qd"][i], pool["u"][i], pool["theta"][i])
    t0 = time.perf_counter()
    for i in idx[warmup : warmup + steps]:
        predict(pool["q"][i], pool["qd"][i], pool["u"][i], pool["theta"][i])
    return (time.perf_counter() - t0) / max(steps, 1)


def _matched(row: dict[str, float], ref: dict[str, float]) -> bool:
    return bool(
        row["E1"] <= E1_MATCH * ref["E1"]
        and row["E_rollout"] <= ROLLOUT_MATCH * ref["E_rollout"]
        and row["S_plan"] >= ref["S_plan"] - PLAN_MATCH_DELTA
    )


def run_cap_x1(
    output: str | Path,
    *,
    config: CAPX1Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX1Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X1 must not write under runs/r10_c0/")
    x0_root = Path(cfg.x0_data)
    if cfg.require_x0_summary:
        summary_path = x0_root / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"CAP-X1 requires CAP-X0 summary at {summary_path}")
        x0 = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(x0.get("cap_x0_passed")):
            raise RuntimeError("CAP-X1 locked until cap_x0_passed=true")

    root.mkdir(parents=True, exist_ok=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = _load_pool(x0_root / "data" / "train_pool.h5")
    val = _load_pool(x0_root / "data" / "val_pool.h5")
    test = _load_pool(x0_root / "data" / "test_pool.h5")

    # G0 smoke accounting on a tiny subset via residual already in pool.
    g0 = bool(float(residual_nrmse(train["residual"][:2000], train["u"][:2000])) < 1.0e-4)

    cache_dir = root / "physics_cache"
    print("precomputing physics cache...", flush=True)
    train_cache = precompute_physics_cache(train, cache_path=cache_dir / "train.h5")
    val_cache = precompute_physics_cache(val, cache_path=cache_dir / "val.h5")
    test_cache = precompute_physics_cache(test, cache_path=cache_dir / "test.h5")

    # Physics-only baseline E1 for diagnostics.
    phy_e1_test = _nrmse(test_cache["qdd_phy"], test["qdd"])

    episodes = _iter_test_episodes(
        x0_root, max_episodes=max(cfg.rollout_episodes, cfg.plan_tasks), seed=7
    )
    rollout_eps = episodes[: cfg.rollout_episodes]
    plan_eps = episodes[: cfg.plan_tasks]

    results: list[dict[str, Any]] = []

    def eval_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
        predict = bundle["predict"]
        phys_only = bool(bundle.get("physics_only", False))
        e1 = evaluate_one_step(
            predict,
            test,
            cache=test_cache,
            physics_only=phys_only,
            max_samples=15000,
            seed=bundle["seed"],
        )
        e_roll = evaluate_rollout(predict, rollout_eps)
        plan = evaluate_planning(
            predict,
            plan_eps,
            train_q=train["q"],
            train_u=train["u"],
            seed=bundle["seed"] + 99,
            n_tasks=cfg.plan_tasks,
        )
        lat = measure_step_latency(
            predict, test, warmup=cfg.latency_warmup, steps=cfg.latency_steps
        )
        row = {
            "family": bundle["family"],
            "hidden": bundle["hidden"],
            "seed": bundle["seed"],
            "n_params": bundle["n_params"],
            "neural_macs": bundle["neural_macs"],
            "E1": float(e1),
            "E_rollout": float(e_roll),
            "S_plan": float(plan["S_plan"]),
            "cem_wall_mean_s": float(plan["cem_wall_mean_s"]),
            "step_latency_s": float(lat),
            "best_val_nrmse": float(bundle.get("best_val_nrmse", float("nan"))),
            "epochs_ran": int(bundle.get("epochs_ran", 0)),
            "physics_only": phys_only,
        }
        return row

    # --- PureNN grid ---
    for h in cfg.pure_widths:
        for seed in cfg.train_seeds:
            print(f"train pure H={h} seed={seed}", flush=True)
            bundle = train_pure(
                hidden=h,
                seed=seed,
                train=train,
                val=val,
                config=cfg,
                device=device,
            )
            # Drop heavy predict closure from JSON later; evaluate now.
            row = eval_bundle(bundle)
            results.append(row)
            _write_json(root / "runs" / f"pure_H{h}_seed{seed}.json", row)

    # --- Hybrid grid ---
    for h in cfg.hybrid_widths:
        for seed in cfg.train_seeds:
            print(f"train hybrid H={h} seed={seed}", flush=True)
            bundle = train_hybrid(
                hidden=h,
                seed=seed,
                train=train,
                val=val,
                train_cache=train_cache,
                val_cache=val_cache,
                config=cfg,
                device=device,
            )
            row = eval_bundle(bundle)
            results.append(row)
            _write_json(root / "runs" / f"hybrid_H{h}_seed{seed}.json", row)

    def aggregate(family: str, hidden: int) -> dict[str, float]:
        rows = [r for r in results if r["family"] == family and r["hidden"] == hidden]
        return {
            "hidden": hidden,
            "n_params": float(rows[0]["n_params"]),
            "neural_macs": float(rows[0]["neural_macs"]),
            "E1": float(np.mean([r["E1"] for r in rows])),
            "E_rollout": float(np.mean([r["E_rollout"] for r in rows])),
            "S_plan": float(np.mean([r["S_plan"] for r in rows])),
            "cem_wall_mean_s": float(np.mean([r["cem_wall_mean_s"] for r in rows])),
            "step_latency_s": float(np.mean([r["step_latency_s"] for r in rows])),
            "n_seeds": len(rows),
        }

    pure_agg = [aggregate("pure", h) for h in cfg.pure_widths]
    hybrid_agg = [aggregate("hybrid", h) for h in cfg.hybrid_widths]
    ref = next(a for a in pure_agg if int(a["hidden"]) == 256)

    for a in pure_agg + hybrid_agg:
        a["matched"] = _matched(a, ref)

    pure_ok = [a for a in pure_agg if a["matched"]]
    hybrid_ok = [a for a in hybrid_agg if a["matched"]]
    g_ref = bool(np.isfinite(ref["E1"]))
    g_grid = len(results) == len(cfg.pure_widths) * len(cfg.train_seeds) + len(
        cfg.hybrid_widths
    ) * len(cfg.train_seeds)

    if not pure_ok:
        rp = float("nan")
        p_pure_min = float("nan")
        p_hyb_min = float("nan")
        upper_bound = False
        match_ok = False
        claim = "no PureNN width matched H=256 reference; STOP protocol"
    else:
        p_pure_min = min(a["n_params"] for a in pure_ok)
        if not hybrid_ok:
            rp = float("nan")
            p_hyb_min = float("nan")
            upper_bound = False
            match_ok = False
            claim = "no Hybrid/Physics-only width matched; no R_P(0)"
        else:
            p_hyb_min = min(a["n_params"] for a in hybrid_ok)
            if p_hyb_min == 0.0:
                rp = 1.0
                upper_bound = True
                claim = "matched-family replacement upper bound (Physics-only matched)"
            else:
                rp = 1.0 - float(p_hyb_min) / float(p_pure_min)
                upper_bound = False
                claim = f"R_P(0)={rp:.4f} at matched one-step/rollout/planning"
            match_ok = True

    summary = {
        "stage": "CAP-X1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "rho": 0.0,
        "mu": 0.0,
        "source": "simulator",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "capacity_claim": bool(match_ok),
        "capacity_claim_wording": claim,
        "matched_family_replacement_upper_bound": bool(upper_bound),
        "x0_data": str(x0_root),
        "device": device,
        "config": asdict(cfg),
        "reference": ref,
        "pure_aggregate": pure_agg,
        "hybrid_aggregate": hybrid_agg,
        "seed_runs": results,
        "physics_only_E1_test": float(phy_e1_test),
        "metrics": {
            "P_pure_min": p_pure_min,
            "P_hybrid_min": p_hyb_min,
            "R_P_0": rp,
        },
        "gates": {
            "G0_physics_accounting": g0,
            "G_ref": g_ref,
            "G_grid": g_grid,
            "G_match": match_ok,
            "G_label": True,
        },
        "cap_x1_passed": bool(g0 and g_ref and g_grid and match_ok),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

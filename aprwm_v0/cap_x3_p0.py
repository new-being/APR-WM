"""CAP-X3-P0: identifiability + latent-optimizable preflight (no K90 claim)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .cap_x1 import _nrmse, physics_qdd_and_minv
from .cap_x3_data import generate_scene_bundle, stack_split
from .cap_x3_theta import (
    ACTIVE_NAMES,
    D_ACTIVE,
    active_bounds,
    active_to_theta,
    nominal_active,
    relative_e_theta,
    theta_to_active,
)
from .capx_arm3_plant import HOST_PLANT_ID, N_DOF, load_capx_arm3

PREREG_PATH = "REPORT/REG/CAPX/CAPX3_P0_PREREG.md"
SCHEMA_ID = "aprwm.cap_x3_p0.ident.v1"
HIDDEN = 64
E1_ID_MAX = 0.05
E_THETA_MAX = 0.35
LATENT_RATIO_MAX = 0.80


@dataclass(frozen=True)
class CAPX3P0Config:
    output: str = "runs/cap_x3/p0"
    n_meta_scenes: int = 32
    n_probe_scenes: int = 8
    meta_seed0: int = 100_000
    probe_seed0: int = 140_000
    n_cal: int = 16
    n_query: int = 4
    n_meta_traj: int = 8
    duration_s: float = 1.0
    latent_epochs: int = 25
    latent_adapt_steps: int = 200
    latent_lr: float = 1.0e-3
    physics_maxiter: int = 80


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def e1_physics(pool: dict[str, np.ndarray], theta_vec_active: np.ndarray, model: Any, data: Any) -> float:
    th = active_to_theta(theta_vec_active)
    cache: dict[str, Any] = {}
    preds = []
    for i in range(pool["q"].shape[0]):
        qdd, _ = physics_qdd_and_minv(
            model, data, q=pool["q"][i], qd=pool["qd"][i], u=pool["u"][i], theta=th, theta_cache=cache
        )
        preds.append(qdd)
    return _nrmse(np.stack(preds), pool["qdd"])


def fit_physics_active(cal: dict[str, np.ndarray], *, maxiter: int) -> np.ndarray:
    model, data = load_capx_arm3(0.002)
    lo, hi = active_bounds()
    bounds = list(zip(lo.tolist(), hi.tolist()))
    x0 = nominal_active()
    cache: dict[str, Any] = {}
    n = int(cal["q"].shape[0])
    # Subsample for speed while covering the trajectory.
    stride = max(1, n // 400)
    idx = np.arange(0, n, stride)

    def loss(x: np.ndarray) -> float:
        th = active_to_theta(x)
        acc = 0.0
        for i in idx:
            qdd, _ = physics_qdd_and_minv(
                model, data, q=cal["q"][i], qd=cal["qd"][i], u=cal["u"][i], theta=th, theta_cache=cache
            )
            acc += float(np.mean(np.square(qdd - cal["qdd"][i])))
        return acc / max(1, idx.size)

    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1.0e-9})
    return np.asarray(res.x, dtype=np.float64)


def _mlp(in_dim: int, hidden: int = HIDDEN):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, N_DOF),
    )


def train_latent_backbone(
    meta_bundles: list[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    device: str,
    hidden: int = HIDDEN,
):
    import torch
    from torch import nn

    xs = []
    ys = []
    scene_ids = []
    for s_i, b in enumerate(meta_bundles):
        pool = stack_split(b["cal"])
        x = np.concatenate([pool["q"], pool["qd"], pool["u"]], axis=1).astype(np.float32)
        xs.append(x)
        ys.append(pool["qdd"].astype(np.float32))
        scene_ids.append(np.full(x.shape[0], s_i, dtype=np.int64))
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    sid = torch.from_numpy(np.concatenate(scene_ids)).to(device)
    x_mean = x.mean(0)
    x_std = x.std(0).clamp_min(1.0e-6)
    xn = (x - x_mean) / x_std

    n_scenes = len(meta_bundles)
    z_table = nn.Embedding(n_scenes, D_ACTIVE).to(device)
    net = _mlp(9 + D_ACTIVE, hidden).to(device)
    opt = torch.optim.AdamW(list(net.parameters()) + list(z_table.parameters()), lr=lr)
    batch = 512
    for _ in range(epochs):
        perm = torch.randperm(xn.shape[0], device=device)
        net.train()
        for i0 in range(0, xn.shape[0], batch):
            idx = perm[i0 : i0 + batch]
            z = z_table(sid[idx])
            pred = net(torch.cat([xn[idx], z], dim=-1))
            loss = nn.functional.mse_loss(pred, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net, z_table, x_mean.detach().cpu().numpy(), x_std.detach().cpu().numpy()


def adapt_latent_z(
    net,
    pool: dict[str, np.ndarray],
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    steps: int,
    lr: float,
    device: str,
) -> np.ndarray:
    import torch
    from torch import nn

    x = np.concatenate([pool["q"], pool["qd"], pool["u"]], axis=1).astype(np.float32)
    y = pool["qdd"].astype(np.float32)
    xt = torch.from_numpy((x - x_mean) / x_std).to(device)
    yt = torch.from_numpy(y).to(device)
    z = torch.zeros(D_ACTIVE, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        pred = net(torch.cat([xt, z.expand(xt.shape[0], -1)], dim=-1))
        loss = nn.functional.mse_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return z.detach().cpu().numpy().astype(np.float64)


def e1_latent(net, pool, z, x_mean, x_std, device: str) -> float:
    import torch

    x = np.concatenate([pool["q"], pool["qd"], pool["u"]], axis=1).astype(np.float32)
    xt = torch.from_numpy((x - x_mean) / x_std).to(device)
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)
    net.eval()
    with torch.no_grad():
        pred = net(torch.cat([xt, zt.expand(xt.shape[0], -1)], dim=-1)).cpu().numpy()
    return _nrmse(pred, pool["qdd"])


def run_cap_x3_p0(
    output: str | Path | None = None,
    *,
    config: CAPX3P0Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX3P0Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X3-P0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)

    print("CAP-X3-P0: generate meta scenes...", flush=True)
    meta = [
        generate_scene_bundle(scene_seed=cfg.meta_seed0 + i, n_cal=cfg.n_meta_traj, n_query=0, duration_s=cfg.duration_s)
        for i in range(cfg.n_meta_scenes)
    ]
    print("CAP-X3-P0: generate probe scenes...", flush=True)
    probes = [
        generate_scene_bundle(
            scene_seed=cfg.probe_seed0 + i,
            n_cal=cfg.n_cal,
            n_query=cfg.n_query,
            duration_s=cfg.duration_s,
        )
        for i in range(cfg.n_probe_scenes)
    ]

    model, data = load_capx_arm3(0.002)
    phy_rows = []
    for b in probes:
        cal = stack_split(b["cal"])
        query = stack_split(b["query"])
        true_a = theta_to_active(b["theta"])
        print(f"  physics fit scene={b['scene_seed']}", flush=True)
        hat = fit_physics_active(cal, maxiter=cfg.physics_maxiter)
        e1_q = e1_physics(query, hat, model, data)
        e1_or = e1_physics(query, true_a, model, data)
        e1_nom = e1_physics(query, nominal_active(), model, data)
        phy_rows.append(
            {
                "scene_seed": int(b["scene_seed"]),
                "E_theta": relative_e_theta(hat, true_a),
                "E1_query": e1_q,
                "E1_oracle": e1_or,
                "E1_nominal": e1_nom,
                "hat": hat.tolist(),
                "true": true_a.tolist(),
                "coord_abs_err": np.abs(hat - true_a).tolist(),
            }
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CAP-X3-P0: train latent backbone on {device}...", flush=True)
    net, _, x_mean, x_std = train_latent_backbone(
        meta, epochs=cfg.latent_epochs, lr=cfg.latent_lr, device=device
    )
    lat_rows = []
    for b in probes:
        cal = stack_split(b["cal"])
        query = stack_split(b["query"])
        z_hat = adapt_latent_z(
            net, cal, x_mean=x_mean, x_std=x_std, steps=cfg.latent_adapt_steps, lr=3.0e-2, device=device
        )
        e_fit = e1_latent(net, query, z_hat, x_mean, x_std, device)
        e0 = e1_latent(net, query, np.zeros(D_ACTIVE), x_mean, x_std, device)
        lat_rows.append(
            {
                "scene_seed": int(b["scene_seed"]),
                "E1_zfit": e_fit,
                "E1_z0": e0,
                "ratio": float(e_fit / (e0 + 1.0e-12)),
            }
        )

    e1s = np.array([r["E1_query"] for r in phy_rows])
    eths = np.array([r["E_theta"] for r in phy_rows])
    ratios = np.array([r["ratio"] for r in lat_rows])
    g_id_pred = bool(np.median(e1s) <= E1_ID_MAX)
    g_id_param = bool(np.median(eths) <= E_THETA_MAX)
    g_latent = bool(np.median(ratios) <= LATENT_RATIO_MAX)
    slack = bool(g_id_pred and not g_id_param)
    passed = bool(g_id_pred and g_latent)
    summary = {
        "stage": "CAP-X3-P0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "unlocks_r10_c0": False,
        "rho": 0.0,
        "d_active": D_ACTIVE,
        "active_names": list(ACTIVE_NAMES),
        "k90_claim": False,
        "identification_slack": slack,
        "n_probe": cfg.n_probe_scenes,
        "config": asdict(cfg),
        "physics": {
            "median_E1_query": float(np.median(e1s)),
            "median_E_theta": float(np.median(eths)),
            "median_E1_oracle": float(np.median([r["E1_oracle"] for r in phy_rows])),
            "rows": phy_rows,
        },
        "latent": {
            "median_ratio": float(np.median(ratios)),
            "median_E1_zfit": float(np.median([r["E1_zfit"] for r in lat_rows])),
            "median_E1_z0": float(np.median([r["E1_z0"] for r in lat_rows])),
            "rows": lat_rows,
        },
        "gates": {
            "G_active": True,
            "G_id_pred": g_id_pred,
            "G_id_param": g_id_param,
            "G_latent": g_latent,
            "G_rho": True,
            "G_label": True,
        },
        "cap_x3_p0_passed": passed,
        "unlocks_cap_x3_formal": passed,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

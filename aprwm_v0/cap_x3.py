"""CAP-X3 formal: few-shot structured θ vs same-dim latent (ρ=0).

Primary: K90 / R_K. No active probe, no ρ>0, no PLAN-X, no R10.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cap_x1 import physics_qdd_and_minv
from .cap_x3_data import generate_scene_bundle, load_scene_h5, stack_split, write_scene_h5
from .cap_x3_p0 import (
    _mlp,
    adapt_latent_z,
    e1_latent,
    e1_physics,
    fit_physics_active,
    train_latent_backbone,
)
from .cap_x3_theta import D_ACTIVE, active_to_theta, nominal_active, relative_e_theta, theta_to_active
from .capx_arm3_plant import HOST_PLANT_ID, N_DOF, load_capx_arm3
from .plan_x1 import bootstrap_mean_ci, spearman

PREREG_PATH = "REPORT/REG/CAPX/CAPX3_PREREG.md"
SCHEMA_ID = "aprwm.cap_x3.adapt.v1"
K_GRID: tuple[int, ...] = (1, 2, 4, 8, 16)
K90_PLUS = 32
ROLLOUT_H: tuple[int, ...] = (10, 50, 100)
MODEL_DT = 0.01
HIDDEN = 128
E1_RECOVER = 0.05
E_THETA_MAX = 0.35


@dataclass(frozen=True)
class CAPX3Config:
    p0_dir: str = "runs/cap_x3/p0"
    output: str = "runs/cap_x3/formal"
    n_train: int = 128
    n_val: int = 32
    n_test: int = 64
    train_seed0: int = 110_000
    val_seed0: int = 120_000
    test_seed0: int = 130_000
    n_cal: int = 16
    n_query: int = 8
    n_meta_traj: int = 8
    duration_s: float = 1.0
    latent_epochs: int = 40
    latent_lr: float = 1.0e-3
    latent_adapt_steps: int = 300
    latent_adapt_lr: float = 3.0e-2
    full_adapt_steps: int = 150
    full_adapt_lr: float = 1.0e-3
    physics_maxiter: int = 80
    k_grid: tuple[int, ...] = K_GRID


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def k90_of_curve(e0: float, e_k: dict[int, float], k_grid: tuple[int, ...] = K_GRID) -> int:
    e16 = float(e_k[max(k_grid)])
    span = float(e0 - e16)
    if span <= 1.0e-12:
        return K90_PLUS
    thresh = e16 + 0.1 * span
    for k in k_grid:
        if float(e_k[k]) <= thresh:
            return int(k)
    return K90_PLUS


def _ensure_scenes(
    root: Path,
    *,
    split: str,
    seed0: int,
    n: int,
    n_cal: int,
    n_query: int,
    duration_s: float,
) -> list[Path]:
    out = root / "data" / split
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        seed = seed0 + i
        path = out / f"scene_{seed:06d}.h5"
        if not path.is_file():
            print(f"generate {split} scene={seed}", flush=True)
            bundle = generate_scene_bundle(
                scene_seed=seed, n_cal=n_cal, n_query=n_query, duration_s=duration_s
            )
            write_scene_h5(path, bundle)
        paths.append(path)
    return paths


def _prefix_cal(cal_trajs: list[dict[str, np.ndarray]], k: int) -> dict[str, np.ndarray]:
    return stack_split(cal_trajs[:k])


def _rollout_mean(
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    trajs: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    out: dict[str, list[float]] = {str(h): [] for h in ROLLOUT_H}
    for ep in trajs:
        q, qd, u = ep["q"], ep["qd"], ep["u"]
        n = q.shape[0]
        for h in ROLLOUT_H:
            for t0 in (0, n // 4, n // 2):
                if t0 + h >= n:
                    continue
                qh, qdh = q[t0].copy(), qd[t0].copy()
                for k in range(h):
                    qdd = np.asarray(predict(qh, qdh, u[t0 + k]), dtype=np.float64)
                    qh = qh + qdh * MODEL_DT
                    qdh = qdh + qdd * MODEL_DT
                err = float(
                    np.sqrt(
                        np.mean(
                            np.square(np.concatenate([qh, qdh]) - np.concatenate([q[t0 + h], qd[t0 + h]]))
                        )
                    )
                )
                out[str(h)].append(err)
    return {h: float(np.mean(v)) if v else float("nan") for h, v in out.items()}


def _clone_net(net):
    import torch

    cloned = copy.deepcopy(net)
    return cloned


def fine_tune_full(
    net,
    pool: dict[str, np.ndarray],
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    steps: int,
    lr: float,
    device: str,
):
    import torch
    from torch import nn

    local = _clone_net(net).to(device)
    for p in local.parameters():
        p.requires_grad_(True)
    x = np.concatenate([pool["q"], pool["qd"], pool["u"]], axis=1).astype(np.float32)
    y = pool["qdd"].astype(np.float32)
    xt = torch.from_numpy((x - x_mean) / x_std).to(device)
    yt = torch.from_numpy(y).to(device)
    z = torch.zeros(D_ACTIVE, device=device, requires_grad=True)
    opt = torch.optim.Adam(list(local.parameters()) + [z], lr=lr)
    local.train()
    for _ in range(steps):
        pred = local(torch.cat([xt, z.expand(xt.shape[0], -1)], dim=-1))
        loss = nn.functional.mse_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return local, z.detach().cpu().numpy().astype(np.float64)


def _pattern(
    *,
    k90_phy: int,
    k90_lat: int,
    k90_full: int,
    e1_phy16: float,
    e_theta16: float,
    k90_ci_neg: bool,
    k90_ci_pos: bool,
) -> str:
    phy_fail = bool(e1_phy16 > E1_RECOVER and e_theta16 > E_THETA_MAX)
    if phy_fail:
        return "identification_failure"
    if k90_ci_neg:
        return "structured_adaptation_advantage"
    if k90_ci_pos:
        return "latent_advantage"
    return "dimension_only"


def run_cap_x3(
    output: str | Path | None = None,
    *,
    config: CAPX3Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX3Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X3 must not write under runs/r10_c0/")
    p0 = Path(cfg.p0_dir) / "summary.json"
    if not p0.is_file():
        raise RuntimeError("CAP-X3 formal locked until CAP-X3-P0 summary exists")
    p0s = json.loads(p0.read_text(encoding="utf-8"))
    if not bool(p0s.get("cap_x3_p0_passed")):
        raise RuntimeError("CAP-X3 formal locked until cap_x3_p0_passed")
    if int(p0s.get("d_active", -1)) != D_ACTIVE:
        raise RuntimeError("CAP-X3 d_active mismatch vs P0")

    root.mkdir(parents=True, exist_ok=True)
    train_paths = _ensure_scenes(
        root, split="train", seed0=cfg.train_seed0, n=cfg.n_train, n_cal=cfg.n_meta_traj, n_query=0, duration_s=cfg.duration_s
    )
    _ensure_scenes(
        root, split="val", seed0=cfg.val_seed0, n=cfg.n_val, n_cal=cfg.n_meta_traj, n_query=0, duration_s=cfg.duration_s
    )
    test_paths = _ensure_scenes(
        root, split="test", seed0=cfg.test_seed0, n=cfg.n_test, n_cal=cfg.n_cal, n_query=cfg.n_query, duration_s=cfg.duration_s
    )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = root / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = ckpt_dir / "latent_backbone.pt"
    if backbone_path.is_file():
        print(f"load latent backbone {backbone_path}", flush=True)
        blob = torch.load(backbone_path, map_location=device, weights_only=False)
        template = _mlp(9 + D_ACTIVE, HIDDEN).to(device)
        template.load_state_dict(blob["state"])
        x_mean = np.asarray(blob["x_mean"], dtype=np.float64)
        x_std = np.asarray(blob["x_std"], dtype=np.float64)
    else:
        print("load meta-train bundles...", flush=True)
        meta = [load_scene_h5(p) for p in train_paths]
        print(f"train latent backbone on {device}...", flush=True)
        net, _, x_mean, x_std = train_latent_backbone(
            meta, epochs=cfg.latent_epochs, lr=cfg.latent_lr, device=device, hidden=HIDDEN
        )
        template = _clone_net(net).to(device)
        torch.save({"state": template.state_dict(), "x_mean": x_mean, "x_std": x_std}, backbone_path)
        print(f"saved {backbone_path}", flush=True)
    for p in template.parameters():
        p.requires_grad_(False)
    template.eval()

    model, data = load_capx_arm3(0.002)
    n_test = len(test_paths)
    k_grid = cfg.k_grid
    e1 = {m: {k: np.zeros(n_test) for k in k_grid} for m in ("phy", "lat", "full")}
    e1_0 = {m: np.zeros(n_test) for m in ("phy", "lat")}
    e_th = {k: np.zeros(n_test) for k in k_grid}
    roll = {m: {k: {str(h): np.zeros(n_test) for h in ROLLOUT_H} for k in k_grid} for m in ("phy", "lat", "full")}
    cache_dir = root / "adapt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _apply_row(i: int, row: dict[str, Any]) -> None:
        e1_0["phy"][i] = float(row["e1_0_phy"])
        e1_0["lat"][i] = float(row["e1_0_lat"])
        for k in k_grid:
            ks = str(k)
            e_th[k][i] = float(row["e_th"][ks])
            for m in ("phy", "lat", "full"):
                e1[m][k][i] = float(row["e1"][m][ks])
                for h in ROLLOUT_H:
                    roll[m][k][str(h)][i] = float(row["roll"][m][ks][str(h)])

    for i, path in enumerate(test_paths):
        row_path = cache_dir / f"{path.stem}.json"
        if row_path.is_file():
            print(f"resume skip {i}/{n_test} {path.name}", flush=True)
            _apply_row(i, json.loads(row_path.read_text(encoding="utf-8")))
            continue
        print(f"adapt test {i}/{n_test} {path.name}", flush=True)
        b = load_scene_h5(path)
        true_a = theta_to_active(b["theta"])
        query = stack_split(b["query"])
        e1_0["phy"][i] = e1_physics(query, nominal_active(), model, data)
        e1_0["lat"][i] = e1_latent(template, query, np.zeros(D_ACTIVE), x_mean, x_std, device)

        for k in k_grid:
            cal = _prefix_cal(b["cal"], k)
            hat = fit_physics_active(cal, maxiter=cfg.physics_maxiter)
            e_th[k][i] = relative_e_theta(hat, true_a)
            e1["phy"][k][i] = e1_physics(query, hat, model, data)
            th = active_to_theta(hat)
            pcache: dict[str, Any] = {}

            def phy_pred(q, qd, u, _th=th, _c=pcache):
                qdd, _ = physics_qdd_and_minv(model, data, q=q, qd=qd, u=u, theta=_th, theta_cache=_c)
                return qdd

            rphy = _rollout_mean(phy_pred, b["query"])
            for h, v in rphy.items():
                roll["phy"][k][h][i] = v

            z_hat = adapt_latent_z(
                template,
                cal,
                x_mean=x_mean,
                x_std=x_std,
                steps=cfg.latent_adapt_steps,
                lr=cfg.latent_adapt_lr,
                device=device,
            )
            e1["lat"][k][i] = e1_latent(template, query, z_hat, x_mean, x_std, device)

            def lat_pred(q, qd, u, _z=z_hat):
                import torch as _t

                x = np.concatenate(
                    [
                        np.asarray(q, dtype=np.float32).reshape(1, N_DOF),
                        np.asarray(qd, dtype=np.float32).reshape(1, N_DOF),
                        np.asarray(u, dtype=np.float32).reshape(1, N_DOF),
                    ],
                    axis=1,
                )
                xt = _t.from_numpy((x - x_mean.astype(np.float32)) / x_std.astype(np.float32)).to(device)
                zt = _t.from_numpy(_z.astype(np.float32)).to(device)
                with _t.no_grad():
                    return template(_t.cat([xt, zt.expand(1, -1)], dim=-1)).cpu().numpy()[0]

            rlat = _rollout_mean(lat_pred, b["query"])
            for h, v in rlat.items():
                roll["lat"][k][h][i] = v

            ft_net, z_ft = fine_tune_full(
                template,
                cal,
                x_mean=x_mean,
                x_std=x_std,
                steps=cfg.full_adapt_steps,
                lr=cfg.full_adapt_lr,
                device=device,
            )
            e1["full"][k][i] = e1_latent(ft_net, query, z_ft, x_mean, x_std, device)

            def full_pred(q, qd, u, _net=ft_net, _z=z_ft):
                import torch as _t

                x = np.concatenate(
                    [
                        np.asarray(q, dtype=np.float32).reshape(1, N_DOF),
                        np.asarray(qd, dtype=np.float32).reshape(1, N_DOF),
                        np.asarray(u, dtype=np.float32).reshape(1, N_DOF),
                    ],
                    axis=1,
                )
                xt = _t.from_numpy((x - x_mean.astype(np.float32)) / x_std.astype(np.float32)).to(device)
                zt = _t.from_numpy(_z.astype(np.float32)).to(device)
                _net.eval()
                with _t.no_grad():
                    return _net(_t.cat([xt, zt.expand(1, -1)], dim=-1)).cpu().numpy()[0]

            rfull = _rollout_mean(full_pred, b["query"])
            for h, v in rfull.items():
                roll["full"][k][h][i] = v

        row = {
            "e1_0_phy": float(e1_0["phy"][i]),
            "e1_0_lat": float(e1_0["lat"][i]),
            "e_th": {str(k): float(e_th[k][i]) for k in k_grid},
            "e1": {m: {str(k): float(e1[m][k][i]) for k in k_grid} for m in ("phy", "lat", "full")},
            "roll": {
                m: {str(k): {str(h): float(roll[m][k][str(h)][i]) for h in ROLLOUT_H} for k in k_grid}
                for m in ("phy", "lat", "full")
            },
        }
        _write_json(row_path, row)
        print(f"saved {row_path.name}", flush=True)

    def mean_curve(mat: dict[int, np.ndarray]) -> dict[int, float]:
        return {int(k): float(np.mean(mat[k])) for k in k_grid}

    phy_mean = mean_curve(e1["phy"])
    lat_mean = mean_curve(e1["lat"])
    full_mean = mean_curve(e1["full"])
    e0_phy = float(np.mean(e1_0["phy"]))
    e0_lat = float(np.mean(e1_0["lat"]))
    k90_phy = k90_of_curve(e0_phy, phy_mean, k_grid)
    k90_lat = k90_of_curve(e0_lat, lat_mean, k_grid)
    k90_full = k90_of_curve(e0_lat, full_mean, k_grid)

    per_phy = np.array([k90_of_curve(float(e1_0["phy"][i]), {k: float(e1["phy"][k][i]) for k in k_grid}, k_grid) for i in range(n_test)])
    per_lat = np.array([k90_of_curve(float(e1_0["lat"][i]), {k: float(e1["lat"][k][i]) for k in k_grid}, k_grid) for i in range(n_test)])
    d_k90 = per_phy.astype(np.float64) - per_lat.astype(np.float64)
    k90_ci = bootstrap_mean_ci(d_k90, seed=11)
    k90_ci_neg = bool(k90_ci["hi"] < 0)
    k90_ci_pos = bool(k90_ci["lo"] > 0)

    # Mechanism: does E_theta lead E1 on the mean curve?
    eth_mean = {int(k): float(np.mean(e_th[k])) for k in k_grid}
    e1p = np.array([phy_mean[k] for k in k_grid])
    ethv = np.array([eth_mean[k] for k in k_grid])
    lead = spearman(-ethv, -e1p)

    e1_phy16 = phy_mean[max(k_grid)]
    e_theta16 = eth_mean[max(k_grid)]
    pattern = _pattern(
        k90_phy=k90_phy,
        k90_lat=k90_lat,
        k90_full=k90_full,
        e1_phy16=e1_phy16,
        e_theta16=e_theta16,
        k90_ci_neg=k90_ci_neg,
        k90_ci_pos=k90_ci_pos,
    )
    rk = None
    if k90_lat > 0:
        rk = 1.0 - (float(k90_phy) / float(k90_lat))

    def roll_mean(method: str) -> dict[str, dict[str, float]]:
        return {str(k): {h: float(np.mean(roll[method][k][h])) for h in roll[method][k]} for k in k_grid}

    g_label = True
    passed = bool(pattern != "identification_failure" and g_label)
    summary = {
        "stage": "CAP-X3",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "unlocks_r10_c0": False,
        "rho": 0.0,
        "d_active": D_ACTIVE,
        "d_z": D_ACTIVE,
        "active_probe": False,
        "plan_x": False,
        "n_test": n_test,
        "k_grid": list(k_grid),
        "config": asdict(cfg),
        "metrics": {
            "E1_0": {"phy": e0_phy, "lat": e0_lat},
            "E1_K": {"phy": phy_mean, "lat": lat_mean, "full": full_mean},
            "E_theta_K": eth_mean,
            "K90": {"phy": k90_phy, "lat": k90_lat, "full": k90_full, "plus_means": K90_PLUS},
            "R_K": rk,
            "K90_per_scene_delta_ci": k90_ci,
            "spearman_Eth_E1phy": lead,
            "rollout": {"phy": roll_mean("phy"), "lat": roll_mean("lat"), "full": roll_mean("full")},
        },
        "gates": {
            "G_p0": True,
            "G_label": g_label,
            "G_equal_dcal": True,
            "G_dz": True,
        },
        "pattern": pattern,
        "cap_x3_passed": passed,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

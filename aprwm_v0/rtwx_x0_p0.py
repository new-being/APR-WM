"""RoboTwin-X0-P0: 1-DoF drawer instrument (no RGB, no official task claim)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .r0 import inspect_robotwin
from .rtwx_drawer import HOST_PLANT_ID, DrawerBackend, DrawerPhi

PREREG_PATH = "REPORT/REG/RTWX/RTWX0_P0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0_p0.drawer.v1"


@dataclass(frozen=True)
class RTWX0P0Config:
    output: str = "runs/rtwx_x0/p0"
    n_train_ep: int = 40
    n_test_ep: int = 16
    n_steps: int = 80
    seed: int = 501
    latent_epochs: int = 40
    hidden: int = 64
    robotwin_repo: str = "/home/dong/Projects/RoboTwin"
    robotwin_python: str = "/home/dong/miniconda3/envs/RoboTwin/bin/python"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(ref))))
    return float(np.sqrt(np.mean(np.square(pred - ref))) / (rms + 1.0e-8))


def sample_train_phi(rng: np.random.Generator) -> DrawerPhi:
    return DrawerPhi(
        mass=float(rng.uniform(0.8, 1.2)),
        damping=float(rng.uniform(0.04, 0.12)),
        friction=float(rng.uniform(0.2, 0.4)),
    )


def test_phi() -> DrawerPhi:
    return DrawerPhi(mass=1.5, damping=0.08, friction=0.5)


def random_forces(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)
    for _ in range(3):
        amp = float(rng.uniform(0.4, 2.0))
        freq = float(rng.uniform(0.15, 1.4))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        u += amp * np.sin(2.0 * np.pi * freq * t / n + phase)
    return u


def collect_split(
    *,
    n_ep: int,
    n_steps: int,
    rng: np.random.Generator,
    train: bool,
) -> list[dict[str, np.ndarray]]:
    rows = []
    for _ in range(n_ep):
        phi = sample_train_phi(rng) if train else test_phi()
        backend = DrawerBackend(phi)
        q0 = float(rng.uniform(0.02, 0.12))
        qd0 = float(rng.uniform(-0.05, 0.05))
        u = random_forces(rng, n_steps)
        traj = backend.rollout(q0, qd0, u)
        s = np.stack([traj["q"], traj["qd"]], axis=1)
        rows.append({"s": s, "u": traj["u"], "phi": phi.as_vector(), "s_next": np.roll(s, -1, axis=0)})
        rows[-1]["s_next"][-1] = s[-1]
    return rows


def stack_transitions(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    s = np.concatenate([r["s"][:-1] for r in rows], axis=0)
    sp = np.concatenate([r["s_next"][:-1] for r in rows], axis=0)
    u = np.concatenate([r["u"][:-1] for r in rows], axis=0)
    return {"s": s, "sp": sp, "u": u.reshape(-1, 1)}


def physics_predict(s: np.ndarray, u: np.ndarray, phi: DrawerPhi) -> np.ndarray:
    backend = DrawerBackend(phi)
    out = np.zeros_like(s)
    for i in range(s.shape[0]):
        backend.reset(float(s[i, 0]), float(s[i, 1]))
        nq, nqd = backend.step_force(float(u[i, 0]))
        out[i] = (nq, nqd)
    return out


def fit_physics(pool: dict[str, np.ndarray], *, maxiter: int = 25) -> DrawerPhi:
    idx = np.arange(0, pool["s"].shape[0], max(1, pool["s"].shape[0] // 200))

    def loss(x: np.ndarray) -> float:
        phi = DrawerPhi(mass=float(np.clip(x[0], 0.3, 3.0)), damping=float(np.clip(x[1], 0.0, 0.5)), friction=float(np.clip(x[2], 0.0, 1.0)))
        pred = physics_predict(pool["s"][idx], pool["u"][idx], phi)
        return float(np.mean(np.square(pred - pool["sp"][idx])))

    x0 = np.array([1.0, 0.08, 0.3], dtype=np.float64)
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    x = np.clip(res.x, [0.3, 0.0, 0.0], [3.0, 0.5, 1.0])
    return DrawerPhi(mass=float(x[0]), damping=float(x[1]), friction=float(x[2]))


def train_latent(pool: dict[str, np.ndarray], *, epochs: int, hidden: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = np.concatenate([pool["s"], pool["u"]], axis=1).astype(np.float32)
    y = pool["sp"].astype(np.float32)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    net = nn.Sequential(
        nn.Linear(3, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 2),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        pred = net(xt)
        loss = nn.functional.mse_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def predict(s, u):
        inp = np.concatenate([s, u], axis=1).astype(np.float32)
        with torch.no_grad():
            return net(torch.from_numpy((inp - mean) / std)).numpy()

    return predict


def run_rtwx_x0_p0(
    output: str | Path | None = None,
    *,
    config: RTWX0P0Config | None = None,
) -> dict[str, Any]:
    cfg = config or RTWX0P0Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("RoboTwin-X0-P0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)

    cap = inspect_robotwin(cfg.robotwin_repo, cfg.robotwin_python)
    try:
        import sapien  # noqa: F401

        g_sapien = True
    except ModuleNotFoundError:
        g_sapien = False

    if not g_sapien:
        summary = {
            "stage": "RoboTwin-X0-P0",
            "rtwx_x0_p0_passed": False,
            "gates": {"G_sapien": False},
            "capability": cap,
            "output": str(root.resolve()),
        }
        _write_json(root / "summary.json", summary)
        return summary

    rng = np.random.default_rng(cfg.seed)
    print("collect train...", flush=True)
    train_rows = collect_split(n_ep=cfg.n_train_ep, n_steps=cfg.n_steps, rng=rng, train=True)
    print("collect test...", flush=True)
    test_rows = collect_split(n_ep=cfg.n_test_ep, n_steps=cfg.n_steps, rng=np.random.default_rng(cfg.seed + 1), train=False)
    tr = stack_transitions(train_rows)
    te = stack_transitions(test_rows)

    # Identifiability: same u, train-nominal vs test phi.
    u_id = random_forces(np.random.default_rng(7), cfg.n_steps)
    nom = DrawerPhi(mass=1.0, damping=0.08, friction=0.3)
    tphi = test_phi()
    a = DrawerBackend(nom).rollout(0.05, 0.0, u_id)["q"]
    b = DrawerBackend(tphi).rollout(0.05, 0.0, u_id)["q"]
    ident = float(np.median(np.abs(a - b)))

    print("physics accounting (true phi on test)...", flush=True)
    phy_true = []
    for row in test_rows:
        phi = DrawerPhi(mass=float(row["phi"][0]), damping=float(row["phi"][1]), friction=float(row["phi"][2]))
        s, u, sp = row["s"][:-1], row["u"][:-1].reshape(-1, 1), row["s_next"][:-1]
        phy_true.append(_nrmse(physics_predict(s, u, phi), sp))
    e1_phy = float(np.median(phy_true))
    e1_hold = _nrmse(te["s"], te["sp"])

    print("train latent...", flush=True)
    lat_predict = train_latent(tr, epochs=cfg.latent_epochs, hidden=cfg.hidden, seed=cfg.seed)
    e1_lat = _nrmse(lat_predict(te["s"], te["u"]), te["sp"])

    print("global-phi OOD diagnostic (not a gate)...", flush=True)
    phi_hat = fit_physics(tr, maxiter=12)
    e1_phy_ood = _nrmse(physics_predict(te["s"], te["u"], phi_hat), te["sp"])

    g_ident = bool(ident > 5.0 * 1e-3)
    g_phy = bool(e1_phy <= 0.10)
    g_latent = bool(e1_lat < 0.80)
    g_split = True
    g_norgb = True
    g_official = False
    passed = bool(g_sapien and g_norgb and g_ident and g_phy and g_latent and g_split)
    summary = {
        "stage": "RoboTwin-X0-P0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "official_robotwin_task_executed": False,
        "rgb": False,
        "diffusion": False,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "capability": cap,
        "config": asdict(cfg),
        "metrics": {
            "ident_median_abs_dq": ident,
            "E1_physics_true_phi": e1_phy,
            "E1_physics_global_ood": e1_phy_ood,
            "E1_hold": e1_hold,
            "E1_latent": e1_lat,
            "phi_hat": phi_hat.as_vector().tolist(),
            "test_phi": tphi.as_vector().tolist(),
        },
        "gates": {
            "G_sapien": g_sapien,
            "G_norgb": g_norgb,
            "G_ident": g_ident,
            "G_phy": g_phy,
            "G_latent": g_latent,
            "G_split": g_split,
            "G_official": g_official,
            "G_label": True,
        },
        "rtwx_x0_p0_passed": passed,
        "unlocks_rtwx_x0_formal": False,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary

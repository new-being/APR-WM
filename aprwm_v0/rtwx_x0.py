"""RoboTwin-X0 formal: oracle-state physics vs latent capacity (no RGB, not R10)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rtwx_x0_smoke import TASK_NAME as CABINET_TASK
from .rtwx_x0_smoke import _load_task_args, _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0.formal.v1"
TASKS: tuple[tuple[str, str, str], ...] = (
    ("A", "place_empty_cup", "cup"),
    ("B", "put_object_cabinet", "object"),
    ("C", "stamp_seal", "seal"),
)


@dataclass(frozen=True)
class RTWX0Config:
    output: str = "runs/rtwx_x0/formal"
    smoke_summary: str = "runs/rtwx_x0/smoke/summary.json"
    robotwin_repo: str = "/root/RoboTwin"
    n_train_ep: int = 1000
    n_val_ep: int = 200
    n_test_ep: int = 300
    n_steps: int = 120
    seed: int = 7001
    latent_widths: tuple[int, ...] = (16, 32, 64, 128)
    residual_widths: tuple[int, ...] = (0, 8, 16, 32)
    latent_epochs: int = 40
    seed_attempts: int = 32


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.size == 0 or ref.size == 0:
        return float("inf")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        return float("inf")
    rms = float(np.sqrt(np.mean(np.square(ref))))
    return float(np.sqrt(np.mean(np.square(pred - ref))) / (rms + 1.0e-8))


def _require_smoke(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"RoboTwin-X0 formal requires smoke summary at {path}")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if not smoke.get("rtwx_x0_smoke_passed"):
        raise RuntimeError("RoboTwin-X0 formal is locked until rtwx-x0-smoke PASS")
    return smoke


def _manip_actor(env: Any, attr: str) -> Any:
    return getattr(env, attr)


def extract_state(env: Any, actor_attr: str) -> np.ndarray:
    actor = _manip_actor(env, actor_attr)
    ql = np.asarray(env.robot.get_left_arm_real_jointState(), dtype=np.float64)
    qr = np.asarray(env.robot.get_right_arm_real_jointState(), dtype=np.float64)
    pose = actor.get_pose()
    x = np.asarray(pose.p, dtype=np.float64)
    q = np.asarray(pose.q, dtype=np.float64)
    parts = [ql, qr, x, q]
    if hasattr(env, "cabinet"):
        parts.append(np.asarray(env.cabinet.get_qpos(), dtype=np.float64).reshape(-1))
        parts.append(np.asarray(env.cabinet.get_qvel(), dtype=np.float64).reshape(-1))
    else:
        parts.append(np.zeros(1, dtype=np.float64))
        parts.append(np.zeros(1, dtype=np.float64))
    return np.concatenate(parts)


def _set_mass(actor: Any, mass: float) -> None:
    if hasattr(actor, "set_mass"):
        actor.set_mass(float(mass))
        return
    ent = getattr(actor, "actor", actor)
    for component in ent.get_components():
        if type(component).__name__ == "PhysxRigidDynamicComponent":
            component.mass = float(mass)


def _set_friction(actor: Any, mu: float) -> None:
    if hasattr(actor, "set_properties"):
        try:
            actor.set_properties(damping=0.05, stiffness=0.0, friction=float(mu))
            return
        except Exception:
            pass
    art = getattr(actor, "actor", None)
    if art is not None and hasattr(art, "get_joints"):
        for joint in art.get_joints():
            if hasattr(joint, "set_friction"):
                try:
                    joint.set_friction(float(mu))
                except Exception:
                    continue


def _apply_wrench(actor: Any, force: np.ndarray) -> None:
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    ent = getattr(actor, "actor", actor)
    if hasattr(ent, "set_qf") and hasattr(ent, "get_qpos"):
        n = int(np.asarray(ent.get_qpos()).reshape(-1).size)
        qf = np.zeros(n, dtype=np.float64)
        qf[0] = float(force[0]) if force.size else 0.0
        try:
            passive = ent.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            ent.set_qf(np.asarray(passive, dtype=np.float64).reshape(-1)[:n] + qf)
            return
        except Exception:
            ent.set_qf(qf)
            return
    f3 = np.zeros(3, dtype=np.float64)
    f3[: min(3, force.size)] = force[: min(3, force.size)]
    comps = ent.get_components() if hasattr(ent, "get_components") else []
    for component in comps:
        if type(component).__name__ != "PhysxRigidDynamicComponent":
            continue
        if hasattr(component, "set_force_and_torque"):
            component.set_force_and_torque(f3, np.zeros(3))
            return
        if hasattr(component, "add_force_at_point"):
            component.add_force_at_point(f3, np.asarray(ent.get_pose().p, dtype=np.float64))
            return
        if hasattr(component, "add_force_torque"):
            component.add_force_torque(f3, np.zeros(3))
            return


def _phi_train(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(0.8, 1.2), rng.uniform(0.04, 0.12), rng.uniform(0.2, 0.4)], dtype=np.float64)


def _phi_test() -> np.ndarray:
    return np.array([1.5, 0.08, 0.5], dtype=np.float64)


def _random_force(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    u = np.zeros((n, dim), dtype=np.float64)
    for d in range(dim):
        for _ in range(3):
            amp = float(rng.uniform(0.15, 1.2))
            freq = float(rng.uniform(0.12, 1.6))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            u[:, d] += amp * np.sin(2.0 * np.pi * freq * t / n + phase)
    return u


def _setup_env(repo: Path, task: str, seed: int, attempts: int, args: dict[str, Any]) -> Any:
    import importlib

    mod = importlib.import_module(f"envs.{task}")
    cls = getattr(mod, task)
    last: Exception | None = None
    for s in range(seed, seed + max(1, attempts)):
        env = cls()
        try:
            env.setup_demo(now_ep_num=0, seed=s, **args)
            return env
        except ValueError as exc:
            if "model_data" not in str(exc):
                raise
            last = exc
            try:
                env.close()
            except Exception:
                pass
    assert last is not None
    raise last


def collect_task(
    *,
    repo: Path,
    task: str,
    actor_attr: str,
    n_ep: int,
    n_steps: int,
    rng: np.random.Generator,
    train: bool,
    args: dict[str, Any],
    seed0: int,
    attempts: int,
) -> dict[str, np.ndarray]:
    s_rows: list[np.ndarray] = []
    sp_rows: list[np.ndarray] = []
    u_rows: list[np.ndarray] = []
    dim_u = 4 if task == CABINET_TASK else 3
    for i in range(n_ep):
        if i % 10 == 0 or i + 1 == n_ep:
            print(f"[rtwx-x0] {task} {'train' if train else 'test'} ep {i+1}/{n_ep}", flush=True)
        env = _setup_env(repo, task, seed0 + i * 17, attempts, args)
        actor = _manip_actor(env, actor_attr)
        phi = _phi_train(rng) if train else _phi_test()
        _set_mass(actor, float(phi[0]))
        _set_friction(actor, float(phi[2]))
        if hasattr(env, "cabinet"):
            _set_friction(env.cabinet, float(phi[2]))
        u_seq = _random_force(rng, n_steps, dim_u)
        s_traj = []
        for t in range(n_steps):
            s_traj.append(extract_state(env, actor_attr))
            _apply_wrench(actor, u_seq[t, :3])
            if hasattr(env, "cabinet"):
                _apply_wrench(env.cabinet, u_seq[t, 3:])
            env.scene.step()
        s_traj.append(extract_state(env, actor_attr))
        s = np.stack(s_traj[:-1], axis=0)
        sp = np.stack(s_traj[1:], axis=0)
        s_rows.append(s)
        sp_rows.append(sp)
        u_rows.append(u_seq)
        try:
            env.close()
        except Exception:
            pass
    return {
        "s": np.concatenate(s_rows, axis=0),
        "sp": np.concatenate(sp_rows, axis=0),
        "u": np.concatenate(u_rows, axis=0),
        "n_steps": int(n_steps),
        "n_ep": int(n_ep),
    }


def physics_predict(s: np.ndarray, u: np.ndarray, phi: np.ndarray, dt: float = 1.0 / 250.0) -> np.ndarray:
    """Explicit object translation + optional 1-DoF cabinet; robot pose copied."""
    m, b, _mu = [float(x) for x in phi]
    m = max(m, 1.0e-3)
    out = np.array(s, copy=True)
    x = s[:, 14:17]
    # crude velocity from not having object v in s: use force Euler on position only
    a = u[:, :3] / m
    out[:, 14:17] = x + dt * (a - b * 0.0)
    if s.shape[1] > 21:
        q = s[:, 21]
        u_art = u[:, 3] if u.shape[1] > 3 else 0.0
        out[:, 21] = q + dt * (u_art / m)
    return out


def train_mlp(
    pool: dict[str, np.ndarray],
    *,
    hidden: int,
    out_dim: int,
    epochs: int,
    seed: int,
    residual_of: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = np.concatenate([pool["s"], pool["u"]], axis=1).astype(np.float32)
    y = pool["sp"].astype(np.float32)
    if residual_of is not None:
        y = (y - residual_of(pool["s"], pool["u"])).astype(np.float32)
    mean, std = x.mean(0).astype(np.float64), (x.std(0) + 1e-6).astype(np.float64)
    xt = torch.from_numpy(((x - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y)
    layers: list[nn.Module] = [nn.Linear(x.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        pred = net(xt)
        loss = nn.functional.mse_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def predict(s, u):
        inp = np.concatenate([np.asarray(s, dtype=np.float64), np.asarray(u, dtype=np.float64)], axis=1)
        z = np.clip((inp - mean) / std, -30.0, 30.0).astype(np.float32)
        with torch.no_grad():
            raw = net(torch.from_numpy(z)).numpy().astype(np.float64)
        if residual_of is not None:
            raw = np.asarray(residual_of(s, u), dtype=np.float64) + raw
        return np.nan_to_num(raw, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

    n_param = int(sum(p.numel() for p in net.parameters()))
    return predict, n_param


def _rollout_nrmse(predict, pool: dict[str, np.ndarray], h: int) -> float:
    """Open-loop NRMSE, episode-aligned. Non-finite windows score as +inf, then mean of finites.

    Concatenated transitions must not be rolled across episode boundaries.
    Any non-finite window makes the score +inf (not NaN).
    """
    s = np.asarray(pool["s"], dtype=np.float64)
    u = np.asarray(pool["u"], dtype=np.float64)
    sp = np.asarray(pool["sp"], dtype=np.float64)
    n_steps = int(pool.get("n_steps", 0))
    if n_steps <= 0:
        n_steps = int(s.shape[0])
    if h <= 0 or s.shape[0] < h or n_steps < h:
        return float("inf")
    n_ep = int(pool.get("n_ep", s.shape[0] // n_steps))
    starts: list[int] = []
    for ep in range(n_ep):
        base = ep * n_steps
        last = base + n_steps - h
        if last < base:
            continue
        starts.append(base)
        if n_steps - h >= 4:
            starts.append(base + (n_steps - h) // 2)
    if not starts:
        return float("inf")
    if len(starts) > 64:
        pick = np.linspace(0, len(starts) - 1, num=64, dtype=int)
        starts = [starts[i] for i in pick]
    errs: list[float] = []
    for i in starts:
        pred = s[i : i + 1].copy()
        finite = True
        for k in range(h):
            pred = np.asarray(predict(pred, u[i + k : i + k + 1]), dtype=np.float64)
            if not np.isfinite(pred).all():
                finite = False
                break
        errs.append(_nrmse(pred, sp[i + h - 1 : i + h]) if finite else float("inf"))
    if not errs or any(not np.isfinite(e) for e in errs):
        return float("inf")
    return float(np.mean(errs))


def run_rtwx_x0(
    output: str | Path,
    *,
    config: RTWX0Config | None = None,
) -> dict[str, Any]:
    cfg = config or RTWX0Config()
    root = Path(output).resolve()
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("r10_c0 is locked; RoboTwin-X0 formal must not write there")
    smoke = _require_smoke(Path(cfg.smoke_summary))
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _patch_curobo_planner(repo)

    summary: dict[str, Any] = {
        "stage": "RoboTwin-X0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "smoke": str(Path(cfg.smoke_summary).resolve()),
        "rgb_in_s": False,
        "capacity_claim": True,
        "unlocks_r10_c0": False,
        "official_robotwin_task_executed": True,
        "config": asdict(cfg),
        "tasks": {},
        "rtwx_x0_passed": False,
    }
    rng = np.random.default_rng(cfg.seed)
    patterns: list[str] = []
    for tid, task, attr in TASKS:
        args = _load_task_args(repo, "demo_clean")
        args["task_name"] = task
        train = collect_task(
            repo=repo,
            task=task,
            actor_attr=attr,
            n_ep=cfg.n_train_ep,
            n_steps=cfg.n_steps,
            rng=rng,
            train=True,
            args=args,
            seed0=cfg.seed + 100 * ord(tid[0]),
            attempts=cfg.seed_attempts,
        )
        test = collect_task(
            repo=repo,
            task=task,
            actor_attr=attr,
            n_ep=cfg.n_test_ep,
            n_steps=cfg.n_steps,
            rng=rng,
            train=False,
            args=args,
            seed0=cfg.seed + 10_000 + 100 * ord(tid[0]),
            attempts=cfg.seed_attempts,
        )
        out_dim = int(train["s"].shape[1])
        phi_hat = np.array([1.0, 0.08, 0.3], dtype=np.float64)
        phy = lambda s, u, p=phi_hat: physics_predict(s, u, p)
        e1_phy = _nrmse(phy(test["s"], test["u"]), test["sp"])
        latent_curve = []
        best_lat = (1e9, 0)
        for h in cfg.latent_widths:
            pred, n_param = train_mlp(train, hidden=h, out_dim=out_dim, epochs=cfg.latent_epochs, seed=cfg.seed + h)
            e1 = _nrmse(pred(test["s"], test["u"]), test["sp"])
            latent_curve.append({"hidden": h, "P": n_param, "E1": e1, "E_roll10": _rollout_nrmse(pred, test, 10)})
            if e1 < best_lat[0]:
                best_lat = (e1, n_param)
        hybrid = []
        for rh in cfg.residual_widths:
            if rh == 0:
                e1 = e1_phy
                n_param = 0
                roll = _rollout_nrmse(phy, test, 10)
            else:
                pred, n_param = train_mlp(
                    train,
                    hidden=rh,
                    out_dim=out_dim,
                    epochs=cfg.latent_epochs,
                    seed=cfg.seed + 500 + rh,
                    residual_of=phy,
                )
                e1 = _nrmse(pred(test["s"], test["u"]), test["sp"])
                roll = _rollout_nrmse(pred, test, 10)
            hybrid.append({"hidden": rh, "P": n_param, "E1": e1, "E_roll10": roll})
        lat_rolls = [float(c["E_roll10"]) for c in latent_curve]
        phy_roll = float(hybrid[0]["E_roll10"])
        phy_left = bool(
            np.isfinite(phy_roll)
            and all(np.isfinite(v) for v in lat_rolls)
            and phy_roll <= min(lat_rolls) + 1e-6
        )
        task_rec = {
            "task": task,
            "d_s": out_dim,
            "E1_physics": e1_phy,
            "latent": latent_curve,
            "hybrid": hybrid,
            "physics_left_of_latent": bool(phy_left),
        }
        summary["tasks"][tid] = task_rec
        _write_json(root / f"task_{tid}.json", task_rec)
        patterns.append("physics_capacity_shift" if phy_left else "latent_sufficient")

    if any(summary["tasks"][k].get("d_s", 0) < 10 for k in summary["tasks"]):
        pattern = "instrument_failure"
    elif all(p == "physics_capacity_shift" for p in patterns):
        pattern = "physics_capacity_shift"
    elif patterns[0] != "physics_capacity_shift" or patterns[1] != "physics_capacity_shift":
        pattern = "latent_sufficient"
    elif patterns[2] != "physics_capacity_shift":
        pattern = "contact_breaks_physics"
    else:
        pattern = "latent_sufficient"
    summary["pattern"] = pattern
    summary["rtwx_x0_passed"] = pattern != "instrument_failure"
    summary["smoke_passed"] = True
    summary["smoke_metrics"] = smoke.get("metrics", {})
    _write_json(root / "summary.json", summary)
    return summary

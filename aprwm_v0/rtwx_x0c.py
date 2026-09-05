"""RTWX-X0C: native qpos → effective PD/error input (no nets, no qf API)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0_smoke import TASK_NAME, _load_task_args, _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0C_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0c.native_qpos.v1"
TASK = TASK_NAME
FREQS_HZ = (0.15, 0.31, 0.57)
FORMAL_N_TRAIN = 48
FORMAL_N_VAL = 24
FORMAL_N_TEST = 24
FORMAL_N_STEPS = 120
E_QDD_MAX = 0.75
E_J_MAX = 0.9
FRAC_DOF = 0.80
G3_RATIO = 1.10
CONTACT_IMPULSE = 1.0e-5
EXCITE_STD_MIN = 1.0e-4
SEED_FORMAL = 8401


@dataclass(frozen=True)
class RTWX0CConfig:
    output: str = "runs/rtwx_x0c"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED_FORMAL
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or type(obj) is np.bool_:
        return bool(obj)
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


def _as_float(x: Any) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(a[0]) if a.size else float("nan")


def _write_header(root: Path, cfg: RTWX0CConfig) -> dict[str, Any]:
    header = {
        "stage": "RTWX-X0C",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "frozen_before_collect": True,
        "native_qpos_only": True,
        "forbids": ["set_qf", "get_qf", "qf_post", "command_torque", "qacc_buffer", "lag_sweep", "neural"],
        "primary_qacc": "qdd_fd=(qd_post-qd_pre)/dt_control",
        "tau_ID": "required generalized force from motion + rigid-body model (not a sensor)",
        "gripper_in_primary_metrics": False,
        "free_space_only": True,
        "capacity_claim": False,
        "unlocks_x0c1_only_if_pass": True,
        "gates_frozen": {
            "E_qdd_max": E_QDD_MAX,
            "E_j_max": E_J_MAX,
            "frac_dof": FRAC_DOF,
            "G3_ratio": G3_RATIO,
            "freqs_hz": list(FREQS_HZ),
        },
        "scale": {
            "n_train_ep": cfg.n_train_ep,
            "n_val_ep": cfg.n_val_ep,
            "n_test_ep": cfg.n_test_ep,
            "n_steps": cfg.n_steps,
            "backend": cfg.backend,
            "smoke": cfg.smoke,
            "seed": cfg.seed,
        },
        "formal_robotwin_frozen": {
            "n_train_ep": FORMAL_N_TRAIN,
            "n_val_ep": FORMAL_N_VAL,
            "n_test_ep": FORMAL_N_TEST,
            "n_steps": FORMAL_N_STEPS,
        },
        "note": "oracle simulator state / native-qpos closed-loop diagnostic; not official RoboTwin observation benchmark",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0C header (before collect)\n"
        f"backend={cfg.backend} smoke={cfg.smoke}\n"
        f"n_train_ep={cfg.n_train_ep} n_val_ep={cfg.n_val_ep} n_test_ep={cfg.n_test_ep} n_steps={cfg.n_steps}\n"
        f"E_qdd_max={E_QDD_MAX} E_j_max={E_J_MAX} frac_dof={FRAC_DOF} G3_ratio={G3_RATIO}\n"
        "native_qpos_only=true capacity_claim=false gripper_excluded=true\n"
        "tau_ID=required_generalized_force_from_motion_plus_rigid_body_model\n",
        encoding="utf-8",
    )
    return header


def _lock_robotwin_scale(cfg: RTWX0CConfig) -> RTWX0CConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
    )


def _empty_pool(n_dof: int = 0) -> dict[str, Any]:
    z = np.zeros((0, n_dof), dtype=np.float64)
    return {
        "q": z.copy(),
        "qd": z.copy(),
        "qn": z.copy(),
        "qdn": z.copy(),
        "qdd": z.copy(),
        "eq": z.copy(),
        "ev": z.copy(),
        "q_tar": z.copy(),
        "qd_tar": z.copy(),
        "kp": z.copy(),
        "kd": z.copy(),
        "tau_id": z.copy(),
        "h": z.copy(),
        "M": np.zeros((0, n_dof, n_dof), dtype=np.float64),
        "dt": np.zeros((0,), dtype=np.float64),
        "ep_index": np.zeros((0,), dtype=np.int32),
        "n_ep": 0,
        "n_steps": 0,
        "n_dof": n_dof,
        "drive_mode": "unresolved",
        "contact_rate": 1.0,
        "native_qpos": False,
        "invalid_reasons": [],
        "source": "empty",
        "pinocchio": False,
        "kp_record": None,
        "kd_record": None,
    }


def _stack_eps(eps: list[dict[str, np.ndarray]], meta: dict[str, Any]) -> dict[str, Any]:
    if not eps:
        out = _empty_pool(int(meta.get("n_dof", 0)))
        out.update(meta)
        return out
    keys = ("q", "qd", "qn", "qdn", "qdd", "eq", "ev", "q_tar", "qd_tar", "kp", "kd", "tau_id", "h")
    out: dict[str, Any] = {k: np.concatenate([e[k] for e in eps], axis=0) for k in keys}
    out["M"] = np.concatenate([e["M"] for e in eps], axis=0)
    out["dt"] = np.concatenate([e["dt"] for e in eps], axis=0)
    out["ep_index"] = np.concatenate(
        [np.full(e["q"].shape[0], i, dtype=np.int32) for i, e in enumerate(eps)], axis=0
    )
    out["n_ep"] = len(eps)
    out["n_steps"] = int(eps[0]["q"].shape[0])
    out["n_dof"] = int(eps[0]["q"].shape[1])
    out.update(meta)
    return out


def _multisine(
    q0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_t, n_dof = int(t.size), int(q0.size)
    qtar = np.broadcast_to(q0.reshape(1, -1), (n_t, n_dof)).astype(np.float64).copy()
    qdtar = np.zeros_like(qtar)
    span = np.maximum(hi - lo, 1e-6)
    inner_lo = lo + 0.10 * span
    inner_hi = hi - 0.10 * span
    amp_budget = 0.10 * span
    for j in range(n_dof):
        raw = rng.uniform(0.2, 1.0, size=3)
        raw = raw / max(float(np.sum(raw)), 1e-12) * float(amp_budget[j])
        phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
        for k, fk in enumerate(FREQS_HZ):
            qtar[:, j] += raw[k] * np.sin(2.0 * np.pi * fk * t + phases[k])
            qdtar[:, j] += raw[k] * 2.0 * np.pi * fk * np.cos(2.0 * np.pi * fk * t + phases[k])
    qtar = np.clip(qtar, inner_lo.reshape(1, -1), inner_hi.reshape(1, -1))
    return qtar, qdtar


def _numpy_truth() -> dict[str, np.ndarray]:
    n = 4
    m = np.array([0.55, 0.80, 1.10, 1.35], dtype=np.float64)
    b = np.array([0.04, 0.05, 0.04, 0.06], dtype=np.float64)
    kp = np.array([90.0, 80.0, 70.0, 85.0], dtype=np.float64)
    kd = np.array([8.0, 7.5, 6.5, 7.0], dtype=np.float64)
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    return {"m": m, "b": b, "kp": kp, "kd": kd, "lo": lo, "hi": hi}


def collect_numpy(cfg: RTWX0CConfig, *, split: str, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    dt = float(cfg.dt_numpy)
    th = _numpy_truth()
    n_dof = int(th["m"].size)
    t = np.arange(n_steps, dtype=np.float64) * dt
    eps: list[dict[str, np.ndarray]] = []
    for _ in range(n_ep):
        q = rng.uniform(-0.15, 0.15, size=n_dof)
        qd = np.zeros(n_dof)
        qtar, qdtar = _multisine(q, th["lo"], th["hi"], t, rng)
        rows = {k: [] for k in ("q", "qd", "qn", "qdn", "qdd", "eq", "ev", "q_tar", "qd_tar", "kp", "kd", "tau_id", "h", "M", "dt")}
        Mconst = np.diag(th["m"])
        for i in range(n_steps):
            eq = qtar[i] - q
            ev = qdtar[i] - qd
            tau = th["kp"] * eq + th["kd"] * ev
            h = th["b"] * qd
            qdd = (tau - h) / th["m"]
            tau_id = th["m"] * qdd + h
            rows["q"].append(q.copy())
            rows["qd"].append(qd.copy())
            rows["qdd"].append(qdd.copy())
            rows["eq"].append(eq.copy())
            rows["ev"].append(ev.copy())
            rows["q_tar"].append(qtar[i].copy())
            rows["qd_tar"].append(qdtar[i].copy())
            rows["kp"].append(th["kp"].copy())
            rows["kd"].append(th["kd"].copy())
            rows["tau_id"].append(tau_id.copy())
            rows["h"].append(h.copy())
            rows["M"].append(Mconst.copy())
            rows["dt"].append(dt)
            qd_n = qd + dt * qdd
            q_n = q + dt * qd_n
            rows["qn"].append(q_n.copy())
            rows["qdn"].append(qd_n.copy())
            qd = qd_n
            q = q_n
        ep = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items() if k != "dt"}
        ep["dt"] = np.asarray(rows["dt"], dtype=np.float64)
        eps.append(ep)
    return _stack_eps(
        eps,
        {
            "drive_mode": "force",
            "contact_rate": 0.0,
            "native_qpos": True,
            "invalid_reasons": [],
            "source": "numpy",
            "pinocchio": False,
            "kp_record": th["kp"],
            "kd_record": th["kd"],
            "m_diag": th["m"],
            "b_diag": th["b"],
        },
    )


def _joint_limits(joints: list[Any], q0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(joints), -np.pi)
    hi = np.full(len(joints), np.pi)
    for i, j in enumerate(joints):
        lim = None
        for name in ("get_limits", "get_limit"):
            fn = getattr(j, name, None)
            if fn is None:
                continue
            try:
                lim = np.asarray(fn(), dtype=np.float64).reshape(-1)
                break
            except Exception:
                continue
        if lim is not None and lim.size >= 2:
            lo[i], hi[i] = float(lim[0]), float(lim[1])
        else:
            lo[i], hi[i] = float(q0[i] - 0.4), float(q0[i] + 0.4)
    return lo, hi


def _arm_q(env: Any, side: str) -> np.ndarray:
    fn = env.robot.get_left_arm_real_jointState if side == "left" else env.robot.get_right_arm_real_jointState
    x = np.asarray(fn(), dtype=np.float64).reshape(-1)
    return x[:-1] if x.size else x


def _arm_qd(env: Any, side: str) -> np.ndarray:
    entity = env.robot.left_entity if side == "left" else env.robot.right_entity
    joints = env.robot.left_arm_joints if side == "left" else env.robot.right_arm_joints
    qvel = np.asarray(entity.get_qvel(), dtype=np.float64).reshape(-1)
    active = list(entity.get_active_joints())
    out = []
    for j in joints:
        out.append(float(qvel[active.index(j)]))
    return np.asarray(out, dtype=np.float64)


def _drive_pack(joints: list[Any]) -> dict[str, Any]:
    kp, kd, qtar, qdtar, modes, flim = [], [], [], [], [], []
    for j in joints:
        kp.append(_as_float(j.get_stiffness()))
        kd.append(_as_float(j.get_damping()))
        qtar.append(_as_float(j.get_drive_target()))
        vel = getattr(j, "get_drive_velocity_target", None)
        qdtar.append(_as_float(vel()) if vel is not None else float("nan"))
        modes.append(str(j.get_drive_mode()).lower())
        fl = getattr(j, "get_force_limit", None)
        flim.append(_as_float(fl()) if fl is not None else float("nan"))
    parsed: list[str] = []
    for m in modes:
        if "accel" in m:
            parsed.append("acceleration")
        elif "force" in m:
            parsed.append("force")
        else:
            parsed.append("unresolved")
    uniq = set(parsed)
    mode = uniq.pop() if len(uniq) == 1 else "unresolved"
    if "unresolved" in uniq or mode == "unresolved" or len(set(parsed)) != 1:
        mode = "unresolved"
    return {
        "kp": np.asarray(kp, dtype=np.float64),
        "kd": np.asarray(kd, dtype=np.float64),
        "q_tar": np.asarray(qtar, dtype=np.float64),
        "qd_tar": np.asarray(qdtar, dtype=np.float64),
        "mode": mode,
        "modes": parsed,
        "force_limit": np.asarray(flim, dtype=np.float64),
    }


def _robot_entity_ids(env: Any) -> set[int]:
    ids: set[int] = set()
    for ent in (env.robot.left_entity, env.robot.right_entity):
        try:
            ids.add(id(ent))
        except Exception:
            pass
        for link in ent.get_links():
            ids.add(id(link))
            e = getattr(link, "entity", None)
            if e is not None:
                ids.add(id(e))
            c = getattr(link, "entity", link)
            ids.add(id(c))
    return ids


def _body_root(body: Any) -> Any:
    e = getattr(body, "entity", None)
    return e if e is not None else body


def _link_name(body: Any) -> str:
    e = getattr(body, "entity", None)
    if e is not None:
        return str(getattr(e, "name", "") or "")
    return str(getattr(body, "name", "") or "")


def _names_from_joints(joints: list[Any]) -> set[str]:
    names: set[str] = set()
    for j in joints:
        for attr in ("get_child_link", "get_parent_link"):
            fn = getattr(j, attr, None)
            if fn is None:
                continue
            try:
                lk = fn()
            except Exception:
                continue
            if lk is None:
                continue
            names.add(str(getattr(lk, "name", "") or ""))
            e = getattr(lk, "entity", None)
            if e is not None:
                names.add(str(getattr(e, "name", "") or ""))
    names.discard("")
    return names


def _actor_link_names(actor: Any) -> set[str]:
    names: set[str] = set()
    ent = getattr(actor, "actor", actor)
    if hasattr(ent, "get_links"):
        try:
            links = ent.get_links()
        except Exception:
            links = []
        for lk in links:
            names.add(str(getattr(lk, "name", "") or ""))
            e = getattr(lk, "entity", None)
            if e is not None:
                names.add(str(getattr(e, "name", "") or ""))
    nm = str(getattr(ent, "name", "") or "") or str(getattr(actor, "name", "") or "")
    if nm:
        names.add(nm)
    names.discard("")
    return names


def _contact_invalid(env: Any) -> bool:
    """Arm/gripper vs object/cabinet, or left-right arm collision. Ignore wheels/ground/table/rest."""
    left_j = list(env.robot.left_arm_joints)
    right_j = list(env.robot.right_arm_joints)
    lg = [x for x in (getattr(env.robot, "left_gripper", None) or []) if x is not None]
    rg = [x for x in (getattr(env.robot, "right_gripper", None) or []) if x is not None]
    if lg and isinstance(lg[0], (list, tuple)):
        lg = [j for pair in lg for j in (pair if isinstance(pair, (list, tuple)) else [pair])]
    if rg and isinstance(rg[0], (list, tuple)):
        rg = [j for pair in rg for j in (pair if isinstance(pair, (list, tuple)) else [pair])]
    left_names = _names_from_joints(left_j + [j for j in lg if hasattr(j, "get_child_link")])
    right_names = _names_from_joints(right_j + [j for j in rg if hasattr(j, "get_child_link")])
    arm_names = left_names | right_names
    obj_names: set[str] = set()
    for attr in ("object", "cabinet", "goal_object"):
        if hasattr(env, attr):
            obj_names |= _actor_link_names(getattr(env, attr))
    ambient = {"ground", "table", "wall"}
    try:
        contacts = env.scene.get_contacts()
    except Exception:
        return False
    for c in contacts:
        pts = getattr(c, "points", None)
        impulse = 0.0
        if pts is not None:
            for p in pts:
                impulse = max(impulse, float(np.linalg.norm(np.asarray(p.impulse, dtype=np.float64))))
        bodies = list(c.bodies)
        if len(bodies) < 2:
            continue
        n0, n1 = _link_name(bodies[0]), _link_name(bodies[1])
        if not n0 or not n1:
            continue
        l0, l1 = n0 in left_names, n1 in left_names
        r0, r1 = n0 in right_names, n1 in right_names
        a0, a1 = n0 in arm_names, n1 in arm_names
        if (l0 and r1) or (r0 and l1):
            if impulse >= CONTACT_IMPULSE:
                return True
        if a0 != a1:
            other = n1 if a0 else n0
            if other in ambient:
                continue
            if other in obj_names or other not in arm_names:
                if other in ambient:
                    continue
                # wheels/base of same embodiment: skip if clearly chassis
                if any(k in other for k in ("wheel", "castor", "footprint", "ground")):
                    continue
                if other in obj_names and impulse >= CONTACT_IMPULSE:
                    return True
    return False


def _pinocchio(entity: Any) -> Any | None:
    fn = getattr(entity, "create_pinocchio_model", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _arm_indices(entity: Any, arm_joints: list[Any]) -> np.ndarray:
    active = list(entity.get_active_joints())
    return np.asarray([active.index(j) for j in arm_joints], dtype=np.int64)


def _tau_id_arm(pin: Any, entity: Any, idx: np.ndarray, qdd_arm: np.ndarray, qd_arm: np.ndarray) -> np.ndarray:
    q = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    qd = np.asarray(entity.get_qvel(), dtype=np.float64).reshape(-1)
    qdd = np.zeros_like(q)
    qdd[idx] = qdd_arm
    qd_use = qd.copy()
    qd_use[idx] = qd_arm
    tau = np.asarray(pin.compute_inverse_dynamics(q, qd_use, qdd), dtype=np.float64).reshape(-1)
    return tau[idx]


def _mh_arm(pin: Any, entity: Any, idx: np.ndarray, qd_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    qd = np.asarray(entity.get_qvel(), dtype=np.float64).reshape(-1)
    qd_use = qd.copy()
    qd_use[idx] = qd_arm
    m = np.asarray(pin.compute_generalized_mass_matrix(q), dtype=np.float64)
    h = np.asarray(pin.compute_inverse_dynamics(q, qd_use, np.zeros_like(q)), dtype=np.float64).reshape(-1)
    return m[np.ix_(idx, idx)], h[idx]


def _build_action(env: Any, q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    lg = float(env.robot.get_left_gripper_val())
    rg = float(env.robot.get_right_gripper_val())
    return np.concatenate([q_left.reshape(-1), np.array([lg]), q_right.reshape(-1), np.array([rg])])


def _count_steps(env: Any) -> tuple[Callable[[], Any], list[int]]:
    orig = env.scene.step
    box = [0]

    def counted() -> Any:
        box[0] += 1
        return orig()

    env.scene.step = counted
    return orig, box


def collect_robotwin(
    cfg: RTWX0CConfig,
    *,
    split: str,
    rng: np.random.Generator,
    stop_mode: list[str],
    task_name: str | None = None,
) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    task = str(task_name or TASK)
    args = _load_task_args(repo, "demo_clean")
    args["task_name"] = task
    args["render_freq"] = 0
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = cfg.seed + {"train": 0, "val": 10_000, "test": 20_000}[split]
    eps: list[dict[str, np.ndarray]] = []
    reasons: list[str] = []
    mode_global = "unresolved"
    kp_rec = None
    kd_rec = None
    pin_ok = False
    native = True
    attempts_used = 0
    slot = 0
    while len(eps) < n_ep and attempts_used < n_ep * max(2, cfg.max_resample):
        attempts_used += 1
        seed = int(seed0 + slot * 17 + attempts_used)
        slot += 1
        if (len(eps) % 2 == 0) or len(eps) + 1 == n_ep:
            print(f"[rtwx-x0c] {split} valid {len(eps)}/{n_ep} attempt {attempts_used}", flush=True)
        env = None
        try:
            env = setup(repo, task, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            if getattr(env, "step_lim", 1000) is None or int(env.step_lim) <= int(cfg.n_steps):
                env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            pack0 = _drive_pack(jl + jr)
            if pack0["mode"] == "unresolved":
                stop_mode.append("unresolved")
                reasons.append("drive_mode_unresolved")
                try:
                    env.close()
                except Exception:
                    pass
                return _stack_eps(
                    [],
                    {
                        "drive_mode": "unresolved",
                        "contact_rate": 1.0,
                        "native_qpos": True,
                        "invalid_reasons": reasons,
                        "source": "robotwin",
                        "pinocchio": False,
                    },
                )
            mode_global = pack0["mode"]
            kp_rec = pack0["kp"].copy()
            kd_rec = pack0["kd"].copy()
            if _contact_invalid(env):
                reasons.append("contact_at_start")
                env.close()
                continue
            pin_l = _pinocchio(env.robot.left_entity)
            pin_r = _pinocchio(env.robot.right_entity)
            pin_ok = pin_l is not None and pin_r is not None
            idx_l = _arm_indices(env.robot.left_entity, jl)
            idx_r = _arm_indices(env.robot.right_entity, jr)
            q0_l = _arm_q(env, "left")
            q0_r = _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            for getter in ("get_timestep",):
                fn = getattr(env.scene, getter, None)
                if callable(fn):
                    try:
                        scene_dt = float(fn())
                        break
                    except Exception:
                        pass
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            dt_guess = scene_dt * 50.0
            t = np.arange(cfg.n_steps, dtype=np.float64) * dt_guess
            qtar_l, qdtar_l_syn = _multisine(q0_l, lo_l, hi_l, t, rng)
            qtar_r, qdtar_r_syn = _multisine(q0_r, lo_r, hi_r, t, rng)
            rows: dict[str, list] = {
                k: []
                for k in ("q", "qd", "qn", "qdn", "qdd", "eq", "ev", "q_tar", "qd_tar", "kp", "kd", "tau_id", "h", "M", "dt")
            }
            invalid = None
            for i in range(int(cfg.n_steps)):
                q_l = _arm_q(env, "left")
                q_r = _arm_q(env, "right")
                qd_l = _arm_qd(env, "left")
                qd_r = _arm_qd(env, "right")
                drv_pre = _drive_pack(jl + jr)
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig_step, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig_step
                n_int = int(box[0])
                cnt1 = int(getattr(env, "take_action_cnt", 0))
                if cnt1 == cnt0:
                    invalid = "take_action_noop"
                    break
                dt_c = max(n_int * scene_dt, scene_dt)
                drv_post = _drive_pack(jl + jr)
                q_l2 = _arm_q(env, "left")
                q_r2 = _arm_q(env, "right")
                qd_l2 = _arm_qd(env, "left")
                qd_r2 = _arm_qd(env, "right")
                n_l = q_l.size
                cmd_l = np.asarray(action[:n_l], dtype=np.float64)
                cmd_r = np.asarray(action[n_l + 1 : n_l + 1 + q_r.size], dtype=np.float64)
                moved_l = not np.allclose(drv_post["q_tar"][:n_l], drv_pre["q_tar"][:n_l], atol=1e-5)
                moved_r = not np.allclose(drv_post["q_tar"][n_l:], drv_pre["q_tar"][n_l:], atol=1e-5)
                match_l = np.allclose(drv_post["q_tar"][:n_l], cmd_l, atol=5e-3)
                match_r = np.allclose(drv_post["q_tar"][n_l:], cmd_r, atol=5e-3)
                if (not moved_l and not match_l) or (not moved_r and not match_r):
                    invalid = "topp_skip_drive_not_updated"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
                q = np.concatenate([q_l, q_r])
                qd = np.concatenate([qd_l, qd_r])
                qdd = np.concatenate([qd_l2 - qd_l, qd_r2 - qd_r]) / dt_c
                q_tar = np.concatenate([cmd_l, cmd_r])
                qd_tar_rec = drv_post["qd_tar"]
                if not np.all(np.isfinite(qd_tar_rec)):
                    invalid = "qd_target_unreadable"
                    break
                eq = q_tar - q
                ev = qd_tar_rec - qd
                tau_id = np.full(q.size, np.nan)
                h_arm = np.zeros(q.size)
                M_arm = np.eye(q.size)
                if pin_l is not None and pin_r is not None:
                    tau_id = np.concatenate(
                        [
                            _tau_id_arm(pin_l, env.robot.left_entity, idx_l, qdd[:n_l], qd_l),
                            _tau_id_arm(pin_r, env.robot.right_entity, idx_r, qdd[n_l:], qd_r),
                        ]
                    )
                    Ml, hl = _mh_arm(pin_l, env.robot.left_entity, idx_l, qd_l)
                    Mr, hr = _mh_arm(pin_r, env.robot.right_entity, idx_r, qd_r)
                    h_arm = np.concatenate([hl, hr])
                    M_arm = np.zeros((q.size, q.size))
                    M_arm[:n_l, :n_l] = Ml
                    M_arm[n_l:, n_l:] = Mr
                rows["q"].append(q)
                rows["qd"].append(qd)
                rows["qn"].append(np.concatenate([q_l2, q_r2]))
                rows["qdn"].append(np.concatenate([qd_l2, qd_r2]))
                rows["qdd"].append(qdd)
                rows["eq"].append(eq)
                rows["ev"].append(ev)
                rows["q_tar"].append(q_tar)
                rows["qd_tar"].append(qd_tar_rec)
                rows["kp"].append(drv_post["kp"])
                rows["kd"].append(drv_post["kd"])
                rows["tau_id"].append(tau_id)
                rows["h"].append(h_arm)
                rows["M"].append(M_arm)
                rows["dt"].append(dt_c)
            try:
                env.close()
            except Exception:
                pass
            if invalid is not None:
                reasons.append(invalid)
                continue
            ep = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items() if k != "dt"}
            ep["dt"] = np.asarray(rows["dt"], dtype=np.float64)
            if ep["q"].shape[0] != int(cfg.n_steps):
                reasons.append("short_episode")
                continue
            eps.append(ep)
        except Exception as exc:
            reasons.append(f"setup:{type(exc).__name__}")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            continue
    contact_rate = 0.0 if len(eps) == n_ep else 1.0
    n_c = sum(1 for r in reasons if r in {"contact", "contact_at_start"})
    meta = {
        "drive_mode": mode_global,
        "contact_rate": contact_rate,
        "native_qpos": native,
        "invalid_reasons": reasons[:80],
        "source": "robotwin",
        "pinocchio": pin_ok,
        "kp_record": kp_rec,
        "kd_record": kd_rec,
        "attempts": attempts_used,
        "target_ep": n_ep,
        "resample_contact_n": n_c,
    }
    return _stack_eps(eps, meta)


def _per_dof_nrmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = ref.shape[1]
    return np.asarray([_nrmse(pred[:, j], ref[:, j]) for j in range(d)], dtype=np.float64)


def _lstsq_ab(eq: np.ndarray, ev: np.ndarray | None, y: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    n_dof = y.shape[1]
    a = np.zeros(n_dof)
    b = np.zeros(n_dof) if ev is not None else None
    for j in range(n_dof):
        if ev is None:
            x = eq[:, j : j + 1]
        else:
            x = np.column_stack([eq[:, j], ev[:, j]])
        coef = np.linalg.lstsq(x, y[:, j], rcond=None)[0]
        a[j] = float(coef[0])
        if b is not None:
            b[j] = float(coef[1])
    return a, b


def _nnls_gain(eq: np.ndarray, ev: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_dof = tau.shape[1]
    kp = np.zeros(n_dof)
    kd = np.zeros(n_dof)
    for j in range(n_dof):
        x = np.column_stack([eq[:, j], ev[:, j]])
        y = tau[:, j]
        m = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if int(m.sum()) < 8:
            continue
        coef = np.linalg.lstsq(x[m], y[m], rcond=None)[0]
        kp[j] = max(0.0, float(coef[0]))
        kd[j] = max(0.0, float(coef[1]))
    return kp, kd


def _pred_m1(eq: np.ndarray, a: np.ndarray) -> np.ndarray:
    return eq * a.reshape(1, -1)


def _pred_m2(eq: np.ndarray, ev: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return eq * a.reshape(1, -1) + ev * b.reshape(1, -1)


def _solve_mh(mass: np.ndarray, h: np.ndarray, tau: np.ndarray) -> np.ndarray:
    rhs = np.asarray(tau, dtype=np.float64) - np.asarray(h, dtype=np.float64)
    try:
        return np.linalg.solve(mass, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(mass, rhs, rcond=None)[0]


def _pred_structured(pool: dict[str, Any], kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
    tau_hat = pool["eq"] * kp.reshape(1, -1) + pool["ev"] * kd.reshape(1, -1)
    if pool.get("drive_mode") == "acceleration":
        return tau_hat
    out = np.zeros_like(tau_hat)
    mass, h = pool["M"], pool["h"]
    for t in range(out.shape[0]):
        out[t] = _solve_mh(mass[t], h[t], tau_hat[t])
    return out


def _rollout_e(
    pool: dict[str, Any],
    qdd_fn: Callable[..., np.ndarray],
    horizon: int,
) -> float:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    if n_ep == 0 or n_steps <= horizon:
        return float("inf")
    errs: list[float] = []
    q, qd = pool["q"], pool["qd"]
    qtar, qdtar = pool["q_tar"], pool["qd_tar"]
    dt = pool["dt"]
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qt, qdt, dts = q[sl], qd[sl], qtar[sl], qdtar[sl], dt[sl]
        for t0 in range(0, n_steps - horizon, max(1, horizon)):
            qh = qq[t0].copy()
            qdh = qddt[t0].copy()
            pred = []
            ref = []
            for h in range(horizon):
                eq = qt[t0 + h] - qh
                ev = qdt[t0 + h] - qdh
                qddh = np.asarray(qdd_fn(eq, ev, qh, qdh, ep * n_steps + t0 + h), dtype=np.float64).reshape(-1)
                qdh = qdh + float(dts[t0 + h]) * qddh
                qh = qh + float(dts[t0 + h]) * qdh
                pred.append(np.concatenate([qh, qdh]))
                ref.append(np.concatenate([qq[t0 + h + 1], qddt[t0 + h + 1]]))
            errs.append(_nrmse(np.stack(pred), np.stack(ref)))
    return float(np.mean(errs)) if errs else float("inf")


def _metrics_block(pred: np.ndarray, pool: dict[str, Any], tau_hat: np.ndarray | None) -> dict[str, Any]:
    e = _nrmse(pred, pool["qdd"])
    ej = _per_dof_nrmse(pred, pool["qdd"])
    out: dict[str, Any] = {
        "E_qdd": e,
        "E_j": ej.tolist(),
        "frac_Ej_lt_0.9": float(np.mean(ej < E_J_MAX)) if ej.size else 0.0,
    }
    if tau_hat is not None and np.isfinite(pool["tau_id"]).all():
        out["E_tau"] = _nrmse(tau_hat, pool["tau_id"])
        a = tau_hat.reshape(-1)
        b = pool["tau_id"].reshape(-1)
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) > 16 and float(np.std(a[m])) > 1e-12 and float(np.std(b[m])) > 1e-12:
            out["rho_tau"] = float(np.corrcoef(a[m], b[m])[0, 1])
        else:
            out["rho_tau"] = 0.0
    else:
        out["E_tau"] = None
        out["rho_tau"] = None
    return out


def score_pools(
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    mode = str(train.get("drive_mode") or "unresolved")
    if mode == "unresolved" or train["n_ep"] == 0:
        return {
            "G0": False,
            "G1": False,
            "G2": False,
            "G3": False,
            "pattern": "controller_mode_unresolved" if mode == "unresolved" else "instrument_failure",
            "drive_mode": mode,
        }
    g0_bits = {
        "native_qpos": bool(train.get("native_qpos")),
        "target_readable": bool(np.isfinite(train["q_tar"]).all() and np.isfinite(train["qd_tar"]).all()),
        "gains_recorded": bool(np.isfinite(train["kp"]).all() and np.isfinite(train["kd"]).all()),
        "drive_mode_known": mode in {"force", "acceleration"},
        "contact_rate_zero": float(train.get("contact_rate", 1.0)) == 0.0 and float(val.get("contact_rate", 1.0)) == 0.0 and float(test.get("contact_rate", 1.0)) == 0.0,
        "enough_ep": int(train["n_ep"]) > 0 and int(val["n_ep"]) > 0 and int(test["n_ep"]) > 0,
        "excitation": bool(float(np.std(train["eq"])) >= EXCITE_STD_MIN and float(np.std(train["qdd"])) >= EXCITE_STD_MIN),
    }
    g0 = all(g0_bits.values())
    a1, _ = _lstsq_ab(train["eq"], None, train["qdd"])
    a2, b2 = _lstsq_ab(train["eq"], train["ev"], train["qdd"])
    assert b2 is not None
    kp_rec = np.median(train["kp"], axis=0)
    kd_rec = np.median(train["kd"], axis=0)
    y_id = train["tau_id"] if mode == "force" and np.isfinite(train["tau_id"]).any() else train["qdd"]
    kp4, kd4 = _nnls_gain(train["eq"], train["ev"], y_id)

    def eval_split(pool: dict[str, Any]) -> dict[str, Any]:
        p0 = np.zeros_like(pool["qdd"])
        p1 = _pred_m1(pool["eq"], a1)
        p2 = _pred_m2(pool["eq"], pool["ev"], a2, b2)
        p3 = _pred_structured(pool, kp_rec, kd_rec)
        p4 = _pred_structured(pool, kp4, kd4)
        t3 = pool["eq"] * kp_rec.reshape(1, -1) + pool["ev"] * kd_rec.reshape(1, -1)
        t4 = pool["eq"] * kp4.reshape(1, -1) + pool["ev"] * kd4.reshape(1, -1)
        if mode == "acceleration":
            t3, t4 = None, None
        raw0 = _nrmse(p0, pool["qdd"])
        m0 = _metrics_block(p0, pool, None)
        m0["E_qdd_raw"] = raw0
        m0["E_qdd"] = 1.0 if np.isfinite(raw0) else float("inf")
        out = {
            "M0": m0,
            "M1": _metrics_block(p1, pool, None),
            "M2": _metrics_block(p2, pool, None),
            "M3": _metrics_block(p3, pool, t3),
            "M4": _metrics_block(p4, pool, t4),
        }

        def fn_zero(eq, ev, q, qd, t):
            return np.zeros_like(eq)

        def fn_m2(eq, ev, q, qd, t):
            return a2 * eq + b2 * ev

        def fn_pd(kp, kd):
            def _fn(eq, ev, q, qd, t):
                tau = kp * eq + kd * ev
                if mode == "acceleration":
                    return tau
                if pool.get("source") == "numpy":
                    return (tau - np.asarray(pool["b_diag"]) * qd) / np.maximum(np.asarray(pool["m_diag"]), 1e-8)
                return _solve_mh(pool["M"][int(t)], pool["h"][int(t)], tau)

            return _fn

        out["M0"]["E_roll10"] = _rollout_e(pool, fn_zero, 10)
        out["M0"]["E_roll50"] = _rollout_e(pool, fn_zero, 50)
        out["M2"]["E_roll10"] = _rollout_e(pool, fn_m2, 10)
        out["M2"]["E_roll50"] = _rollout_e(pool, fn_m2, 50)
        out["M3"]["E_roll10"] = _rollout_e(pool, fn_pd(kp_rec, kd_rec), 10)
        out["M3"]["E_roll50"] = _rollout_e(pool, fn_pd(kp_rec, kd_rec), 50)
        out["M4"]["E_roll10"] = _rollout_e(pool, fn_pd(kp4, kd4), 10)
        out["M4"]["E_roll50"] = _rollout_e(pool, fn_pd(kp4, kd4), 50)
        out["identity_E_qdd"] = out["M0"]["E_qdd"]
        return out

    te = eval_split(test)
    va = eval_split(val)
    tr = eval_split(train)
    m2 = te["M2"]
    g1 = bool(m2["E_qdd"] <= E_QDD_MAX and float(m2["frac_Ej_lt_0.9"]) >= FRAC_DOF)
    id_e = te["M0"]["E_qdd"]
    id_r = te["M0"]["E_roll10"]

    def g2_ok(blk: dict[str, Any]) -> bool:
        return bool(blk["E_qdd"] < E_QDD_MAX and blk["E_qdd"] < id_e and blk["E_roll10"] < id_r)

    g2_m3 = g2_ok(te["M3"])
    g2_m4 = g2_ok(te["M4"])
    g2 = bool(g2_m3 or g2_m4)
    g2_via = "M3" if g2_m3 else ("M4" if g2_m4 else None)
    g3 = True
    if g2_via == "M4":
        g3 = bool(te["M4"]["E_qdd"] <= G3_RATIO * va["M4"]["E_qdd"])
    if not g0:
        pattern = "instrument_failure"
    elif not g1:
        pattern = "servo_signal_insufficient"
    elif not g2:
        pattern = "servo_signal_only"
    elif g2_via == "M4" and not g3:
        pattern = "servo_signal_only"
    else:
        pattern = "closed_loop_structure_closed"
    return {
        "G0": g0,
        "G0_bits": g0_bits,
        "G1": g1,
        "G2": g2 and (g3 if g2_via == "M4" else True),
        "G3": g3,
        "G2_via": g2_via,
        "pattern": pattern,
        "drive_mode": mode,
        "M1_coef_a": a1.tolist(),
        "M2_coef_a": a2.tolist(),
        "M2_coef_b": b2.tolist(),
        "M3_kp_recorded": kp_rec.tolist(),
        "M3_kd_recorded": kd_rec.tolist(),
        "M4_kp_fit": kp4.tolist(),
        "M4_kd_fit": kd4.tolist(),
        "train": tr,
        "val": va,
        "test": te,
        "note_tau_ID": "required generalized force from observed motion + rigid-body model, not measured sensor torque",
    }


def run_rtwx_x0c(output: str | Path, *, config: RTWX0CConfig | None = None) -> dict[str, Any]:
    cfg = _lock_robotwin_scale(config or RTWX0CConfig())
    root = Path(output).resolve()
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("r10_c0 is locked; RTWX-X0C must not write there")
    root.mkdir(parents=True, exist_ok=True)
    header = _write_header(root, cfg)
    rng = np.random.default_rng(cfg.seed)
    stop_mode: list[str] = []
    if cfg.backend == "numpy":
        train = collect_numpy(cfg, split="train", rng=rng)
        val = collect_numpy(cfg, split="val", rng=rng)
        test = collect_numpy(cfg, split="test", rng=rng)
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        _patch_curobo_planner(repo)
        train = collect_robotwin(cfg, split="train", rng=rng, stop_mode=stop_mode)
        if stop_mode:
            val = _empty_pool(int(train.get("n_dof", 0)))
            test = _empty_pool(int(train.get("n_dof", 0)))
            train["drive_mode"] = "unresolved"
        else:
            val = collect_robotwin(cfg, split="val", rng=rng, stop_mode=stop_mode)
            test = collect_robotwin(cfg, split="test", rng=rng, stop_mode=stop_mode)
            if stop_mode:
                train["drive_mode"] = "unresolved"
    scored = score_pools(train, val, test)
    passed = scored["pattern"] == "closed_loop_structure_closed"
    summary = {
        "stage": "RTWX-X0C",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "header": header,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "unlocks_x0c1": passed,
        "neural": False,
        "label": "oracle simulator state / native-qpos closed-loop diagnostic",
        "not_official_robotwin_observation_benchmark": True,
        "not_force_level_id": True,
        "rtwx_x0c_passed": passed,
        "n_train": int(train.get("n_ep", 0)),
        "n_val": int(val.get("n_ep", 0)),
        "n_test": int(test.get("n_ep", 0)),
        "n_steps": int(train.get("n_steps", 0)),
        "d_arm": int(train.get("n_dof", 0)),
        "source": train.get("source"),
        "pinocchio": bool(train.get("pinocchio")),
        "invalid_reasons_train": train.get("invalid_reasons", []),
        **scored,
    }
    metrics = {
        "G0": scored["G0"],
        "G1": scored["G1"],
        "G2": scored["G2"],
        "G3": scored["G3"],
        "pattern": scored["pattern"],
        "drive_mode": scored.get("drive_mode"),
        "test": scored.get("test"),
        "val": scored.get("val"),
        "train": scored.get("train"),
    }
    np.savez_compressed(
        root / "logs.npz",
        q=train.get("q", np.zeros((0, 1))),
        qd=train.get("qd", np.zeros((0, 1))),
        qdd=train.get("qdd", np.zeros((0, 1))),
        eq=train.get("eq", np.zeros((0, 1))),
        ev=train.get("ev", np.zeros((0, 1))),
        q_tar=train.get("q_tar", np.zeros((0, 1))),
        qd_tar=train.get("qd_tar", np.zeros((0, 1))),
        tau_id=train.get("tau_id", np.zeros((0, 1))),
        dt=train.get("dt", np.zeros((0,))),
    )
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(root / "metrics.json", metrics)
    return summary

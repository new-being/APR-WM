"""RTWX-O0Q0R0: native counterfactual instrument. No yaw science. No RGB. No teleport into contact."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose
from .rtwx_o0q0 import (
    D_H_pair,
    EY,
    G0_MAX,
    H,
    I_RATIO,
    REGIMES,
    TAU,
    _R_y,
    _apply_yaw,
    _contact_J,
    _hold_action,
    _physx,
    _read_yq,
    _restore,
    _robot_q,
    _set_inertia_ratio,
    _twist,
)
from .rtwx_x0c import _arm_q, _build_action, _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0r0.native_counterfactual_instrument.v1"
SEEDS = (0, 1, 2)  # official place_empty_cup seed.txt ↔ episode_0000000–2
N_EP = 1
N_MIN_REGIME = 3
G1_FRAC = 0.80
PROBE_YAW = 90.0
P0_LIFT = 0.12
P0_W = np.array([1.5, 0.2, 0.8])
P0_V = np.array([0.05, 0.0, 0.0])
P1_V0 = np.array([0.10, 0.0, 0.0])
MU_A = 8.0
COM_DX = 0.015
DIST_P2 = (0.02, 0.12)
GRIP_OPEN = 0.35
GRIP_CLOSED = 0.25
N_P3 = 3
EE_FAR = 0.15
PROBES = {"P0": "inertia", "P1": "friction", "P2": "com", "P3": "com"}
FORWARD = frozenset({"hdf5", "play_once", "scripted"})


@dataclass(frozen=True)
class RTWXO0Q0R0Config:
    output: str = "runs/rtwx_o0q0r0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_ep: int = N_EP
    horizon: int = H
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0R0Config) -> RTWXO0Q0R0Config:
    if cfg.smoke:
        return replace(cfg, seeds=(62,), n_ep=1, horizon=3)
    return replace(cfg, seeds=SEEDS, n_ep=N_EP, horizon=H)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0R0 must not write there")


def _pattern(*, g0: bool, g_native: bool, g1: bool) -> str:
    if not g0:
        return "counterfactual_restore_failure"
    if not g_native:
        return "native_contact_snapshot_failure"
    if not g1:
        return "regime_excitation_failure"
    return "counterfactual_instrument_qualified"


def _pack_snap(env: Any) -> dict[str, Any]:
    p, quat = _cup_pose(env)
    v, w = _twist(env.cup)
    q, qd = _robot_q(env)
    el = np.asarray(env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:3]
    er = np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64).reshape(-1)[:3]
    dl = float(np.linalg.norm(el - p))
    dr = float(np.linalg.norm(er - p))
    if dl <= dr:
        dist, arm, grip = dl, "left", float(env.robot.get_left_gripper_val())
    else:
        dist, arm, grip = dr, "right", float(env.robot.get_right_gripper_val())
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return {
        "p": p.copy(),
        "quat": quat.copy(),
        "v": v.copy(),
        "w": w.copy(),
        "q": q.copy(),
        "qd": qd.copy(),
        "dist": dist,
        "arm": arm,
        "grip": grip,
        "J": float(_contact_J(env, env.cup)),
        "z": float(p[2]),
        "w_norm": float(np.linalg.norm(w)),
        "w_cross_ey": float(np.linalg.norm(np.cross(w, EY))),
    }


def _clone_dense(seq: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in seq.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            inner = {}
            for kk, vv in v.items():
                inner[kk] = np.asarray(vv).copy() if isinstance(vv, np.ndarray) else deepcopy(vv)
            out[k] = inner
        else:
            out[k] = deepcopy(v)
    return out


def _play_one(env: Any, a: dict[str, Any]) -> None:
    k = a["kind"]
    if k == "dense":
        env.take_dense_action(a["seq"])
        return
    if k == "joints":
        q = np.asarray(a["q"], dtype=np.float64)
        n_l = q.size // 2
        env.robot.set_arm_joints(q[: n_l - 1], np.zeros(n_l - 1), "left")
        env.robot.set_arm_joints(q[n_l : q.size - 1], np.zeros(n_l - 1), "right")
        env.robot.set_gripper(float(q[n_l - 1]), "left")
        env.robot.set_gripper(float(q[-1]), "right")
        for _ in range(int(a.get("n_step", 8))):
            env.scene.step()
        return
    env.take_action(np.asarray(a["u"], dtype=np.float64), action_type=str(a.get("action_type", "qpos")))


def _install_recorders(env: Any) -> tuple[list, list, Any, Any]:
    snaps: list[dict[str, Any]] = []
    acts: list[dict[str, Any]] = []
    orig_ta = env.take_action
    orig_td = env.take_dense_action

    def ta(action, action_type="qpos"):
        snaps.append(_pack_snap(env))
        acts.append(
            {
                "kind": "ee" if action_type == "ee" else "qpos",
                "u": np.asarray(action, dtype=np.float64).copy(),
                "action_type": action_type,
            }
        )
        return orig_ta(action, action_type=action_type)

    def td(control_seq, save_freq=-1):
        snaps.append(_pack_snap(env))
        acts.append({"kind": "dense", "seq": _clone_dense(control_seq)})
        return orig_td(control_seq, save_freq=save_freq)

    env.take_action = ta  # type: ignore[method-assign]
    env.take_dense_action = td  # type: ignore[method-assign]
    return snaps, acts, orig_ta, orig_td


def _uninstall(env: Any, orig_ta: Any, orig_td: Any) -> None:
    env.take_action = orig_ta  # type: ignore[method-assign]
    env.take_dense_action = orig_td  # type: ignore[method-assign]


def _hold_pack(env: Any, horizon: int) -> dict[str, Any]:
    hold = _hold_action(env)
    return {
        "snap": _pack_snap(env),
        "actions": [{"kind": "qpos", "u": hold.copy(), "action_type": "qpos"} for _ in range(horizon)],
        "source": "table_reset",
    }


def _p0_ic(env: Any, z0: float, quat0: np.ndarray) -> bool:
    import sapien

    p, _ = _cup_pose(env)
    p_up = p.copy()
    p_up[2] = z0 + P0_LIFT
    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p_up, quat0))
    c = _physx(env.cup)
    if c is not None:
        c.linear_velocity = P0_V
        c.angular_velocity = P0_W
    s = _pack_snap(env)
    return bool(s["z"] >= z0 + 0.08 and s["w_cross_ey"] > 0.1 and s["dist"] > EE_FAR)


def _select_p2(snaps: list[dict[str, Any]], n_act: int, horizon: int) -> int | None:
    lo, hi = DIST_P2
    for i, s in enumerate(snaps):
        if i + horizon > n_act:
            break
        if lo <= float(s["dist"]) <= hi and float(s["grip"]) >= GRIP_OPEN:
            return i
    return None


def _select_p3(snaps: list[dict[str, Any]], n_act: int, horizon: int) -> int | None:
    streak = 0
    start: int | None = None
    for i, s in enumerate(snaps):
        ok = float(s["grip"]) <= GRIP_CLOSED and (float(s["J"]) > 1e-6 or float(s["dist"]) < 0.04)
        if ok:
            if streak == 0:
                start = i
            streak += 1
            if streak >= N_P3 and start is not None and start + horizon <= n_act:
                return start
        else:
            streak = 0
            start = None
    return None


def _slice_pack(snaps: list[dict[str, Any]], acts: list[dict[str, Any]], i: int, horizon: int, source: str) -> dict[str, Any]:
    return {"snap": snaps[i], "actions": acts[i : i + horizon], "source": source, "index": i}


def _cmass(c: Any):
    return getattr(c, "cmass_local_pose", None)


def _push_probe(env: Any, probe: str | None) -> dict[str, Any]:
    c = _physx(env.cup)
    saved: dict[str, Any] = {"probe": probe, "I": None, "cmass": None}
    if c is None or probe is None:
        return saved
    if hasattr(c, "inertia"):
        saved["I"] = np.asarray(c.inertia, dtype=np.float64).copy()
    saved["cmass"] = _cmass(c)
    if probe == "inertia":
        _set_inertia_ratio(env.cup, I_RATIO)
    elif probe == "com" and saved["cmass"] is not None:
        import sapien

        pose = saved["cmass"]
        p = np.asarray(pose.p, dtype=np.float64).reshape(3)
        p2 = p.copy()
        p2[0] += COM_DX
        c.cmass_local_pose = sapien.Pose(p2, pose.q)
    return saved


def _pop_probe(env: Any, saved: dict[str, Any]) -> None:
    c = _physx(env.cup)
    if c is None:
        return
    if saved.get("I") is not None:
        c.inertia = saved["I"]
    if saved.get("cmass") is not None:
        c.cmass_local_pose = saved["cmass"]


def _set_force(cup: Any, f: np.ndarray) -> None:
    c = _physx(cup)
    if c is None:
        return
    f3 = np.asarray(f, dtype=np.float64).reshape(3)
    if hasattr(c, "set_force_and_torque"):
        c.set_force_and_torque(f3, np.zeros(3))
        return
    if hasattr(c, "add_force_torque"):
        c.add_force_torque(f3, np.zeros(3))


def _aniso_drag(env: Any) -> None:
    p, quat = _cup_pose(env)
    from .rtwx_o0g1b import _quat_to_R

    R = _quat_to_R(quat)
    v, _ = _twist(env.cup)
    t = R @ np.array([1.0, 0.0, 0.0])
    f = -MU_A * float(np.dot(v, t)) * t
    _set_force(env.cup, f)


def _roll(env: Any, pack: dict[str, Any], *, yaw_deg: float | None, probe: str | None) -> list[dict[str, np.ndarray]]:
    _restore(env, pack["snap"])
    saved = _push_probe(env, probe)
    try:
        if yaw_deg is not None and abs(float(yaw_deg)) > 1e-9:
            _apply_yaw(env, float(yaw_deg))
        if probe == "friction":
            c = _physx(env.cup)
            if c is not None:
                c.linear_velocity = P1_V0
        out = []
        for a in pack["actions"]:
            if probe == "friction":
                _aniso_drag(env)
            _play_one(env, a)
            out.append(_read_yq(env))
        return out
    finally:
        _pop_probe(env, saved)


def _cup_rigid_audit(cup: Any) -> dict[str, Any]:
    c = _physx(cup)
    out: dict[str, Any] = {
        "mass": None,
        "I_B": None,
        "c_COM": None,
        "I_x": None,
        "I_z": None,
        "I_x_eq_I_z": None,
        "rel_Ixz": None,
        "commutes_Ry90": None,
    }
    if c is None:
        return out
    if hasattr(c, "mass"):
        out["mass"] = float(c.mass)
    if hasattr(c, "inertia"):
        I = np.asarray(c.inertia, dtype=np.float64).reshape(-1)
        if I.size >= 3:
            ix, iy, iz = float(I[0]), float(I[1]), float(I[2])
            out["I_B"] = [ix, iy, iz]
            out["I_x"], out["I_z"] = ix, iz
            den = max(abs(ix), abs(iz), 1e-12)
            out["rel_Ixz"] = abs(ix - iz) / den
            out["I_x_eq_I_z"] = bool(abs(ix - iz) / den < 0.02)
            Ig = np.diag([ix, iy, iz])
            g = _R_y(90.0)
            out["commutes_Ry90"] = bool(np.allclose(Ig, g @ Ig @ g.T, rtol=0.02, atol=1e-8))
    pose = _cmass(c)
    if pose is not None:
        out["c_COM"] = np.asarray(pose.p, dtype=np.float64).reshape(3).tolist()
    return out


def _ee_action(env: Any, arm: str, pose7: np.ndarray) -> np.ndarray:
    el = np.asarray(env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:7]
    er = np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64).reshape(-1)[:7]
    lg = float(env.robot.get_left_gripper_val())
    rg = float(env.robot.get_right_gripper_val())
    if arm == "left":
        el = np.asarray(pose7, dtype=np.float64).reshape(-1)[:7]
    else:
        er = np.asarray(pose7, dtype=np.float64).reshape(-1)[:7]
    return np.concatenate([el, [lg], er, [rg]])


def _follow_pose_qpos(env: Any, arm: str, pose7: list[float] | np.ndarray) -> bool:
    pose = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose.size < 7:
        return False
    fn = env.robot.right_plan_path if arm == "right" else env.robot.left_plan_path
    try:
        res = fn(pose.tolist())
    except Exception:
        return False
    if not res or str(res.get("status", "")) != "Success":
        return False
    pos = np.asarray(res.get("position"), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 1:
        return False
    stride = max(1, pos.shape[0] // 12)
    for q_arm in pos[::stride]:
        ql, qr = _arm_q(env, "left"), _arm_q(env, "right")
        qa = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        if arm == "right":
            qr = qa[: qr.size] if qa.size >= qr.size else qr
        else:
            ql = qa[: ql.size] if qa.size >= ql.size else ql
        env.take_action(_build_action(env, ql, qr), action_type="qpos")
    return True


def _scripted_forward(env: Any) -> str:
    from envs.utils.action import ArmTag

    p, _ = _cup_pose(env)
    arm = "right" if float(p[0]) > 0 else "left"
    arm_tag = ArmTag(arm)
    env.plan_success = True
    try:
        env.move(env.close_gripper(arm_tag, pos=0.6))
    except Exception as exc:
        print(f"[rtwx-o0q0r0] scripted close_gripper: {type(exc).__name__}: {exc}", flush=True)
    try:
        cid = 0 if arm == "right" else 2
        env.move(env.grasp_actor(env.cup, arm_tag, pre_grasp_dis=0.1, contact_point_id=cid))
        env.move(env.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        return "scripted"
    except Exception as exc:
        print(f"[rtwx-o0q0r0] scripted grasp_actor: {type(exc).__name__}: {exc}", flush=True)
        env.plan_success = True
    ee = np.asarray(env.robot.get_right_ee_pose() if arm == "right" else env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)
    cup_p, _ = _cup_pose(env)
    quat = ee[3:7] if ee.size >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
    waypoints = [cup_p + np.array([0.0, 0.0, 0.12]), cup_p + np.array([0.0, 0.0, 0.04])]
    used = False
    for tgt in waypoints:
        pose = np.concatenate([tgt, quat])
        if _follow_pose_qpos(env, arm, pose):
            used = True
            continue
        n = 10
        e0 = np.asarray(env.robot.get_right_ee_pose() if arm == "right" else env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:3]
        for k in range(1, n + 1):
            a = k / n
            pk = (1.0 - a) * e0 + a * tgt
            try:
                env.take_action(_ee_action(env, arm, np.concatenate([pk, quat])), action_type="ee")
                used = True
            except Exception:
                env.plan_success = True
    q, _ = _robot_q(env)
    n_l = q.size // 2
    for _ in range(6):
        lg = float(q[n_l - 1])
        rg = float(q[-1])
        if arm == "right":
            rg = max(0.0, rg - 0.18)
        else:
            lg = max(0.0, lg - 0.18)
        env.take_action(
            np.concatenate([_arm_q(env, "left"), np.array([lg]), _arm_q(env, "right"), np.array([rg])]),
            action_type="qpos",
        )
        q, _ = _robot_q(env)
    ee2 = np.asarray(env.robot.get_right_ee_pose() if arm == "right" else env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)
    lift = ee2[:3].copy()
    lift[2] += 0.08
    _follow_pose_qpos(env, arm, np.concatenate([lift, ee2[3:7] if ee2.size >= 7 else quat]))
    return "scripted" if used else "scripted"


def _try_play_once(env: Any) -> str | None:
    try:
        env.play_once()
        return "play_once"
    except Exception as exc:
        print(f"[rtwx-o0q0r0] play_once: {type(exc).__name__}: {exc}", flush=True)
        env.plan_success = True
        return None


def _hdf5_for_seed(repo: Path, seed: int) -> Path | None:
    from .task_x0 import _discover_hdf5

    files = _discover_hdf5(repo, TASK)
    sp = repo / "data" / "demo_clean" / TASK / "aloha_agilex" / "seed.txt"
    if not files or not sp.is_file():
        return None
    seeds = [int(x) for x in sp.read_text(encoding="utf-8").split() if x.strip()]
    for s, f in zip(seeds, files):
        if int(s) == int(seed):
            return f
    return None


def _try_hdf5(env: Any, path: Path, snaps: list, acts: list) -> str | None:
    from .task_x0 import _load_hdf5_qpos

    pack = _load_hdf5_qpos(path)
    if pack is None:
        return None
    qpos, qact = pack["q"], pack["qa"]
    n = int(qpos.shape[0])
    for t in range(n):
        snaps.append(_pack_snap(env))
        acts.append({"kind": "joints", "q": qpos[t].copy(), "qa": qact[t].copy(), "n_step": 8})
        q = qpos[t]
        n_l = q.size // 2
        env.robot.set_arm_joints(q[: n_l - 1], np.zeros(n_l - 1), "left")
        env.robot.set_arm_joints(q[n_l : q.size - 1], np.zeros(n_l - 1), "right")
        env.robot.set_gripper(float(q[n_l - 1]), "left")
        env.robot.set_gripper(float(q[-1]), "right")
        for _ in range(8):
            env.scene.step()
    print(f"[rtwx-o0q0r0] hdf5 replay {path.name} frames={n}", flush=True)
    return "hdf5"


def _collect_contact(cfg: RTWXO0Q0R0Config, seed: int, args: dict[str, Any], planner: str) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 4000
    snaps, acts, orig_ta, orig_td = _install_recorders(env)
    source = None
    try:
        hdf = _hdf5_for_seed(repo, seed)
        if hdf is not None:
            source = _try_hdf5(env, hdf, snaps, acts)
        i2 = _select_p2(snaps, len(acts), cfg.horizon)
        i3 = _select_p3(snaps, len(acts), cfg.horizon)
        if i2 is None or i3 is None:
            psrc = _try_play_once(env)
            if source is None:
                source = psrc
            i2 = _select_p2(snaps, len(acts), cfg.horizon)
            i3 = _select_p3(snaps, len(acts), cfg.horizon)
            if i2 is None or i3 is None:
                _scripted_forward(env)
                source = source or "scripted"
                i2 = _select_p2(snaps, len(acts), cfg.horizon) if i2 is None else i2
                i3 = _select_p3(snaps, len(acts), cfg.horizon) if i3 is None else i3
        out: dict[str, Any] = {}
        fwd = source if source in FORWARD else (source or "scripted")
        if fwd not in FORWARD:
            fwd = "scripted"
        if i2 is not None:
            out["P2"] = _slice_pack(snaps, acts, i2, cfg.horizon, fwd)
        if i3 is not None:
            out["P3"] = _slice_pack(snaps, acts, i3, cfg.horizon, fwd)
        return {"packs": out, "n_rec": len(acts), "source": source, "planner": planner}
    finally:
        _uninstall(env, orig_ta, orig_td)
        try:
            env.close()
        except Exception:
            pass


def _collect_p0_p1(cfg: RTWXO0Q0R0Config, seed: int, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import sapien

    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 4000
    try:
        audit = _cup_rigid_audit(env.cup)
        p0, quat0 = _cup_pose(env)
        z0 = float(p0[2])
        packs: dict[str, Any] = {}
        table = _hold_pack(env, cfg.horizon)
        s1 = table["snap"]
        if s1["z"] <= z0 + 0.015 and s1["dist"] > EE_FAR and s1["w_norm"] < 0.5:
            packs["P1"] = table
        table_pose = (p0.copy(), quat0.copy())
        if _p0_ic(env, z0, quat0):
            p0pack = _hold_pack(env, cfg.horizon)
            p0pack["source"] = "free_flight_ic"
            packs["P0"] = p0pack
        getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(table_pose[0], table_pose[1]))
        c = _physx(env.cup)
        if c is not None:
            c.linear_velocity = np.zeros(3)
            c.angular_velocity = np.zeros(3)
        return packs, audit
    finally:
        try:
            env.close()
        except Exception:
            pass


def _numpy_dh(s0: dict[str, np.ndarray], a, deg: float, body_name: str, horizon: int) -> float:
    from .symx_plant import apply_g, make_bodies, rollout, transform_action

    body = make_bodies()[body_name]
    g = _R_y(deg)
    s_ref = {k: s0[k].copy() for k in ("p", "R", "v", "w")}
    s_g = apply_g(s_ref, g) if abs(deg) > 1e-9 else s_ref
    ag = transform_action(a, g) if abs(deg) > 1e-9 else a
    from .rtwx_o0q0 import _yq_from_parts

    tr0 = rollout(body, s_ref, a, horizon + 1, 0.005)
    trg = rollout(body, s_g, ag, horizon + 1, 0.005)
    z = np.zeros(14)

    def pack(tr):
        return [_yq_from_parts(s["p"], s["R"], s["v"], s["R"] @ s["w"], z, z, 0.0) for s in tr[1:]]

    return D_H_pair(pack(tr0), pack(trg))


def _run_numpy(cfg: RTWXO0Q0R0Config) -> dict[str, Any]:
    from .symx_x0 import sample_action, sample_state

    rng = np.random.default_rng(int(cfg.seeds[0]))
    ident: list[float] = []
    excite: dict[str, list[float]] = {r: [] for r in REGIMES}
    sources: dict[str, list[str]] = {r: [] for r in REGIMES}
    specs = [
        ("P0", True, "C4", np.array([0.5, 0.1, 0.4])),
        ("P1", False, "C0fr", np.array([0.08, 0.0, 0.0])),
        ("P2", False, "C1", np.array([0.12, 0.2, 0.05])),
        ("P3", False, "C1", np.array([0.25, 0.05, 0.35])),
    ]
    for name, air, body, w0 in specs:
        s0 = sample_state(rng, airborne=air)
        s0["w"] = w0
        if not air:
            s0["p"][2] = 0.02
        a = sample_action(rng)
        ident.append(_numpy_dh(s0, a, 0.0, "C0", cfg.horizon))
        excite[name].append(_numpy_dh(s0, a, PROBE_YAW, body, cfg.horizon))
        sources[name].append("free_flight_ic" if name == "P0" else ("table_reset" if name == "P1" else "scripted"))
    audit = {
        "mass": 0.2,
        "I_B": None,
        "I_x_eq_I_z": False,
        "commutes_Ry90": False,
        "note": "numpy smoke diagnostic only",
    }
    return {"ident": ident, "excite": excite, "sources": sources, "audit": audit, "planner": "numpy"}


def _measure_seed(cfg: RTWXO0Q0R0Config, seed: int, args: dict[str, Any], planner: str) -> dict[str, Any]:
    p01, audit = _collect_p0_p1(cfg, seed, args)
    contact = _collect_contact(cfg, seed, args, planner)
    packs = {**p01, **contact["packs"]}
    print(
        f"[rtwx-o0q0r0] seed={seed} packs={sorted(packs)} n_rec={contact.get('n_rec')} contact_src={contact.get('source')}",
        flush=True,
    )
    ident: list[float] = []
    excite: dict[str, list[float]] = {r: [] for r in REGIMES}
    sources: dict[str, list[str]] = {r: [] for r in REGIMES}
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 4000
    try:
        if audit is None:
            audit = _cup_rigid_audit(env.cup)
        for name in REGIMES:
            pack = packs.get(name)
            if pack is None:
                continue
            src = str(pack.get("source"))
            if name in ("P2", "P3") and src not in FORWARD:
                print(f"[rtwx-o0q0r0] refuse non-forward {name} source={src}", flush=True)
                continue
            sources[name].append(src)
            ref = _roll(env, pack, yaw_deg=None, probe=None)
            idn = _roll(env, pack, yaw_deg=None, probe=None)
            dh_i = D_H_pair(ref, idn)
            ident.append(dh_i)
            alt = _roll(env, pack, yaw_deg=PROBE_YAW, probe=PROBES[name])
            ref_p = _roll(env, pack, yaw_deg=None, probe=PROBES[name])
            dh_e = D_H_pair(ref_p, alt)
            excite[name].append(dh_e)
            print(
                f"[rtwx-o0q0r0] seed={seed} {name} src={src} D_H(I)={dh_i:.4f} D_H(Ry90,{PROBES[name]})={dh_e:.4f}",
                flush=True,
            )
        return {
            "ident": ident,
            "excite": excite,
            "sources": sources,
            "audit": audit,
            "n_rec": contact.get("n_rec"),
            "contact_source": contact.get("source"),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    ident = [float(x) for x in raw["ident"] if np.isfinite(x)]
    g0 = (not ident) or max(ident) < G0_MAX
    need = 1 if smoke else N_MIN_REGIME
    n_snap = {r: int(len(raw["excite"][r])) for r in REGIMES}
    missing = [r for r in REGIMES if n_snap[r] < need]
    src_ok = True
    for r in ("P2", "P3"):
        for s in raw["sources"].get(r, []):
            if s not in FORWARD:
                src_ok = False
    g_native = (not missing) and src_ok
    g1_reg = {}
    for r in REGIMES:
        xs = [float(x) > TAU for x in raw["excite"][r]]
        g1_reg[r] = bool(xs) and (sum(xs) / len(xs) >= G1_FRAC)
    g1 = bool(g_native) and all(g1_reg.values())
    return {
        "G0": {"ok": g0, "D_H_I_max": max(ident) if ident else None, "n": len(ident), "gate": G0_MAX},
        "G_native": {"ok": g_native, "n_snap": n_snap, "missing": missing, "need": need},
        "G1": {
            "ok": g1,
            "per_regime": g1_reg,
            "Ry90": {r: list(raw["excite"][r]) for r in REGIMES},
        },
        "gates": {"g0": g0, "g_native": g_native, "g1": g1},
    }


def run_rtwx_o0q0r0(output: str | Path, config: RTWXO0Q0R0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0R0Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0R0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "no_teleport_into_contact": True,
        "tau": TAU,
        "G0_max": G0_MAX,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
        "no_yaw_science": True,
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0r0] backend={cfg.backend} smoke={cfg.smoke} H={cfg.horizon}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _run_numpy(cfg)
    else:
        import os
        import sys

        from .task_x0 import _install_mplib_fallback, _load_task_args

        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_raster_shader()
        planner = _install_mplib_fallback(repo)
        args = _load_task_args(repo, TASK, "demo_clean")
        args["task_name"] = TASK
        args["render_freq"] = 0
        chunks = [_measure_seed(cfg, int(s), args, planner) for s in cfg.seeds]
        ident = [x for c in chunks for x in c["ident"]]
        excite = {r: [x for c in chunks for x in c["excite"][r]] for r in REGIMES}
        sources = {r: [x for c in chunks for x in c["sources"][r]] for r in REGIMES}
        raw = {
            "ident": ident,
            "excite": excite,
            "sources": sources,
            "audit": chunks[0]["audit"] if chunks else {},
            "planner": planner,
        }

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(g0=g["g0"], g_native=g["g_native"], g1=g["g1"])
    qualified = pattern == "counterfactual_instrument_qualified"
    print(
        f"[rtwx-o0q0r0] G0={g['g0']} G_native={g['g_native']} G1={g['g1']} pattern={pattern}",
        flush=True,
    )
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G_native", "G1")},
        "sources": raw.get("sources"),
        "rigid_audit": raw.get("audit"),
        "planner": raw.get("planner"),
        "unlocks_o0q0r1_prereg": bool(qualified),
        "unlocks_o0q1_prereg": False,
        "unlocks_o1": False,
        "unlocks_o0g6r": False,
        "unlocks_symx3": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

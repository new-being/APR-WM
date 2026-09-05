"""RTWX-O0Q0: task-object causal yaw audit. No RGB in D_H. Does not change O0 target."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _quat_fix
from .rtwx_o0g1b import _quat_to_R
from .rtwx_x0c import _arm_q, _arm_qd, _build_action, _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0.task_causal_quotient.v1"
SEEDS = (45101, 45102, 45103)
YAW_DEG = tuple(15.0 * k for k in range(1, 24))  # 15°..345°; I excluded
H = 8
N_EP = 3
I_RATIO = 1.5
TAU = 0.05
G0_MAX = 0.02
G2_MIN = 0.90
G3_MIN = 0.80
G1_B2_FRAC = 0.80
N_MIN_REGIME = 3
LP, LN, LV, LND, LQ, LQD, LJ = 0.05, 0.20, 0.30, 2.0, 0.30, 1.5, 0.02
EY = np.array([0.0, 1.0, 0.0])
REGIMES = ("P0", "P1", "P2", "P3")


@dataclass(frozen=True)
class RTWXO0Q0Config:
    output: str = "runs/rtwx_o0q0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_ep: int = N_EP
    horizon: int = H
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0Config) -> RTWXO0Q0Config:
    if cfg.smoke:
        return replace(cfg, seeds=(61,), n_ep=1, horizon=3)
    return replace(cfg, seeds=SEEDS, n_ep=N_EP, horizon=H)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0 must not write there")


def _R_y(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        nxt = (1, 2, 0)
        j, k = nxt[i], nxt[nxt[i]]
        s = 2.0 * np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12))
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return _quat_fix(q)[0]


def d_Q(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    return float(
        np.linalg.norm(a["p"] - b["p"]) / LP
        + np.linalg.norm(a["n"] - b["n"]) / LN
        + np.linalg.norm(a["v"] - b["v"]) / LV
        + np.linalg.norm(a["nd"] - b["nd"]) / LND
        + np.linalg.norm(a["q"] - b["q"]) / LQ
        + np.linalg.norm(a["qd"] - b["qd"]) / LQD
        + abs(float(np.asarray(a["J"]).reshape(-1)[0]) - float(np.asarray(b["J"]).reshape(-1)[0])) / LJ
    )


def _yq_from_parts(p, R, v, w, q, qd, J) -> dict[str, np.ndarray]:
    n = np.asarray(R, dtype=np.float64) @ EY
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return {
        "p": np.asarray(p, dtype=np.float64).reshape(3),
        "n": n,
        "v": np.asarray(v, dtype=np.float64).reshape(3),
        "nd": np.cross(w, n),
        "q": np.asarray(q, dtype=np.float64).reshape(-1),
        "qd": np.asarray(qd, dtype=np.float64).reshape(-1),
        "J": np.array([float(J)]),
    }


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool) -> str:
    if not g0:
        return "counterfactual_instrument_failure"
    if not g1:
        return "causal_symmetry_not_excited"
    if not (g2 and g3):
        return "task_yaw_causally_relevant"
    return "task_yaw_causal_quotient_supported"


def _physx(cup: Any):
    ent = getattr(cup, "actor", cup)
    if hasattr(ent, "get_components"):
        for c in ent.get_components():
            if type(c).__name__ in ("PhysxRigidDynamicComponent", "PhysxRigidBodyComponent"):
                return c
    return None


def _twist(cup: Any) -> tuple[np.ndarray, np.ndarray]:
    c = _physx(cup)
    if c is None:
        return np.zeros(3), np.zeros(3)
    return np.asarray(c.linear_velocity, dtype=np.float64).reshape(3), np.asarray(c.angular_velocity, dtype=np.float64).reshape(3)


def _contact_J(env: Any, cup: Any) -> float:
    try:
        contacts = env.scene.get_contacts()
    except Exception:
        return 0.0
    s = 0.0
    cup_ent = getattr(cup, "actor", cup)
    for c in contacts:
        bodies = list(getattr(c, "bodies", []) or [])
        hit = False
        for b in bodies:
            e = getattr(b, "entity", b)
            if e is cup_ent or b is cup_ent:
                hit = True
                break
            nm = str(getattr(e, "name", "") or getattr(b, "name", "") or "").lower()
            if "cup" in nm:
                hit = True
                break
        if not hit:
            continue
        for p in getattr(c, "points", None) or []:
            s += float(np.linalg.norm(np.asarray(getattr(p, "impulse", np.zeros(3)), dtype=np.float64)))
    return s


def _robot_q(env: Any) -> tuple[np.ndarray, np.ndarray]:
    ql, qr = _arm_q(env, "left"), _arm_q(env, "right")
    qdl, qdr = _arm_qd(env, "left"), _arm_qd(env, "right")
    lg = float(env.robot.get_left_gripper_val())
    rg = float(env.robot.get_right_gripper_val())
    q = np.concatenate([ql, [lg], qr, [rg]])
    qd = np.concatenate([qdl, [0.0], qdr, [0.0]])
    return q, qd


def _read_yq(env: Any) -> dict[str, np.ndarray]:
    p, quat = _cup_pose(env)
    R = _quat_to_R(quat)
    v, w = _twist(env.cup)
    q, qd = _robot_q(env)
    return _yq_from_parts(p, R, v, w, q, qd, _contact_J(env, env.cup))


def _set_inertia_ratio(cup: Any, ratio: float) -> None:
    c = _physx(cup)
    if c is None or not hasattr(c, "inertia"):
        return
    I = np.asarray(c.inertia, dtype=np.float64).reshape(-1)
    if I.size < 3:
        return
    I2 = I.copy()
    I2[0] = float(ratio) * float(I[2])
    c.inertia = I2


def _apply_yaw(env: Any, deg: float) -> None:
    import sapien

    p, quat = _cup_pose(env)
    R = _quat_to_R(quat)
    v, w = _twist(env.cup)
    qn = _R_to_quat(R @ _R_y(deg))
    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p, qn))
    c = _physx(env.cup)
    if c is not None:
        c.linear_velocity = v
        c.angular_velocity = w


def _restore(env: Any, snap: dict[str, Any]) -> None:
    import sapien

    q = np.asarray(snap["q"], dtype=np.float64)
    n_l = q.size // 2
    left, right = q[:n_l], q[n_l:]
    env.robot.set_arm_joints(left[:-1], np.zeros(left.size - 1), "left")
    env.robot.set_arm_joints(right[:-1], np.zeros(right.size - 1), "right")
    env.robot.set_gripper(float(left[-1]), "left")
    env.robot.set_gripper(float(right[-1]), "right")
    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(snap["p"], snap["quat"]))
    c = _physx(env.cup)
    if c is not None:
        c.linear_velocity = snap["v"]
        c.angular_velocity = snap["w"]


def _hold_action(env: Any) -> np.ndarray:
    q, _ = _robot_q(env)
    n_l = q.size // 2
    return _build_action(env, q[: n_l - 1], q[n_l : q.size - 1])


def _roll_actions(env: Any, actions: list[np.ndarray], yaw_deg: float | None, aniso: bool) -> list[dict[str, np.ndarray]]:
    if aniso:
        _set_inertia_ratio(env.cup, I_RATIO)
    if yaw_deg is not None and abs(yaw_deg) > 1e-9:
        _apply_yaw(env, yaw_deg)
    out = []
    for a in actions:
        env.take_action(np.asarray(a, dtype=np.float64), action_type="qpos")
        out.append(_read_yq(env))
    return out


def D_H_pair(ref: list[dict[str, np.ndarray]], alt: list[dict[str, np.ndarray]]) -> float:
    n = min(len(ref), len(alt))
    if n == 0:
        return float("inf")
    return float(np.mean([d_Q(ref[i], alt[i]) for i in range(n)]))


def _numpy_dh(s0: dict[str, np.ndarray], a, deg: float, aniso: bool, horizon: int) -> float:
    from .symx_plant import apply_g, make_bodies, rollout, transform_action

    body = make_bodies()["C4" if aniso else "C0"]
    g = _R_y(deg)
    s_ref = {k: s0[k].copy() for k in ("p", "R", "v", "w")}
    s_g = apply_g(s_ref, g) if abs(deg) > 1e-9 else s_ref
    ag = transform_action(a, g) if abs(deg) > 1e-9 else a
    tr0 = rollout(body, s_ref, a, horizon + 1, 0.005)
    trg = rollout(body, s_g, ag, horizon + 1, 0.005)
    z = np.zeros(14)

    def pack(tr):
        return [_yq_from_parts(s["p"], s["R"], s["v"], s["R"] @ s["w"], z, z, 0.0) for s in tr[1:]]

    return D_H_pair(pack(tr0), pack(trg))


def _run_numpy(cfg: RTWXO0Q0Config) -> dict[str, Any]:
    from .symx_x0 import sample_action, sample_state

    rng = np.random.default_rng(int(cfg.seeds[0]))
    yaws = (90.0, 180.0, 45.0) if cfg.smoke else YAW_DEG
    per: dict[str, Any] = {r: {"B1": [], "B2": []} for r in REGIMES}
    ident: list[float] = []
    specs = [
        ("P0", True, np.array([0.5, 0.1, 0.4])),
        ("P1", False, np.array([0.05, 0.0, 0.02])),
        ("P2", False, np.array([0.12, 0.2, 0.05])),
        ("P3", False, np.array([0.25, 0.05, 0.35])),
    ]
    for name, air, w0 in specs:
        s0 = sample_state(rng, airborne=air)
        s0["w"] = w0
        if not air:
            s0["p"][2] = 0.02
        a = sample_action(rng)
        ident.append(_numpy_dh(s0, a, 0.0, False, cfg.horizon))
        ident.append(_numpy_dh(s0, a, 0.0, True, cfg.horizon))
        for deg in yaws:
            per[name]["B1"].append({"deg": deg, "D_H": _numpy_dh(s0, a, deg, False, cfg.horizon)})
            per[name]["B2"].append({"deg": deg, "D_H": _numpy_dh(s0, a, deg, True, cfg.horizon)})
    return {"ident": ident, "per": per, "yaws": list(yaws), "planner": "numpy", "plan_ok": True}


def _pack_snap(env: Any) -> dict[str, Any]:
    p, quat = _cup_pose(env)
    v, w = _twist(env.cup)
    q, qd = _robot_q(env)
    return {"p": p.copy(), "quat": quat.copy(), "v": v.copy(), "w": w.copy(), "q": q.copy(), "qd": qd.copy()}


def _collect_robotwin_seed(cfg: RTWXO0Q0Config, seed: int) -> dict[str, Any]:
    import os
    import sys

    from .rtwx_x0 import _setup_env as setup
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
    snaps: dict[str, dict[str, Any]] = {}
    ident: list[float] = []
    per = {r: {"B1": [], "B2": []} for r in REGIMES}
    env = None
    try:
        env = setup(repo, TASK, seed, cfg.seed_attempts, args)
        env.check_success = lambda *a, **k: False
        env.step_lim = 4000
        p0, quat0 = _cup_pose(env)
        z0 = float(p0[2])
        rec_act: list[np.ndarray] = []
        rec_yq: list[dict[str, np.ndarray]] = []
        rec_snap: list[dict[str, Any]] = []
        orig = env.take_action

        def wrapped(action, action_type="qpos"):
            rec_snap.append(_pack_snap(env))
            rec_yq.append(_read_yq(env))
            rec_act.append(np.asarray(action, dtype=np.float64).copy())
            return orig(action, action_type=action_type)

        env.take_action = wrapped  # type: ignore[method-assign]
        plan_ok = True
        try:
            env.play_once()
        except Exception as exc:
            plan_ok = False
            print(f"[rtwx-o0q0] play_once fail seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        env.take_action = orig  # type: ignore[method-assign]

        for i, y in enumerate(rec_yq):
            z = float(rec_snap[i]["p"][2])
            J = float(y["J"][0])
            if z >= z0 + 0.05 and J > 1e-4:
                rg = "P3"
            elif z >= z0 + 0.03 and J > 1e-5:
                rg = "P2"
            else:
                rg = "P1"
            if rg not in snaps and i + cfg.horizon <= len(rec_act):
                snaps[rg] = {"snap": rec_snap[i], "actions": rec_act[i : i + cfg.horizon]}

        import sapien

        hold = _hold_action(env)
        p_up = p0.copy()
        p_up[2] = z0 + 0.12
        getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p_up, quat0))
        c = _physx(env.cup)
        if c is not None:
            c.linear_velocity = np.array([0.05, 0.0, 0.0])
            c.angular_velocity = np.array([1.5, 0.2, 0.8])
        snaps["P0"] = {
            "snap": {
                "p": p_up,
                "quat": quat0.copy(),
                "v": np.array([0.05, 0.0, 0.0]),
                "w": np.array([1.5, 0.2, 0.8]),
                "q": _robot_q(env)[0],
                "qd": _robot_q(env)[1],
            },
            "actions": [hold.copy() for _ in range(cfg.horizon)],
        }
        if "P1" not in snaps:
            snaps["P1"] = {
                "snap": {
                    "p": p0.copy(),
                    "quat": quat0.copy(),
                    "v": np.zeros(3),
                    "w": np.zeros(3),
                    "q": _robot_q(env)[0],
                    "qd": _robot_q(env)[1],
                },
                "actions": [hold.copy() for _ in range(cfg.horizon)],
            }
        ee = np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64).reshape(-1)
        ep = ee[:3]
        if "P2" not in snaps:
            p2 = ep + np.array([0.0, 0.0, -0.06])
            snaps["P2"] = {
                "snap": {
                    "p": p2,
                    "quat": quat0.copy(),
                    "v": np.zeros(3),
                    "w": np.zeros(3),
                    "q": _robot_q(env)[0],
                    "qd": _robot_q(env)[1],
                },
                "actions": [hold.copy() for _ in range(cfg.horizon)],
            }
        if "P3" not in snaps:
            qg = _robot_q(env)[0].copy()
            n_l = qg.size // 2
            qg[n_l - 1] = 0.0
            qg[-1] = 0.0
            close = _build_action(env, qg[: n_l - 1], qg[n_l : qg.size - 1])
            snaps["P3"] = {
                "snap": {
                    "p": ep.copy(),
                    "quat": quat0.copy(),
                    "v": np.zeros(3),
                    "w": np.zeros(3),
                    "q": qg,
                    "qd": _robot_q(env)[1],
                },
                "actions": [close.copy() for _ in range(cfg.horizon)],
            }

        yaws = list(YAW_DEG)
        I0 = None
        c0 = _physx(env.cup)
        if c0 is not None and hasattr(c0, "inertia"):
            I0 = np.asarray(c0.inertia, dtype=np.float64).copy()
        for name, pack in snaps.items():
            for aniso, tag in ((False, "B1"), (True, "B2")):
                if I0 is not None:
                    c0.inertia = I0
                _restore(env, pack["snap"])
                ref = _roll_actions(env, pack["actions"], None, aniso)
                _restore(env, pack["snap"])
                idn = _roll_actions(env, pack["actions"], None, aniso)
                ident.append(D_H_pair(ref, idn))
                for deg in yaws:
                    _restore(env, pack["snap"])
                    alt = _roll_actions(env, pack["actions"], deg, aniso)
                    dh = D_H_pair(ref, alt)
                    per[name][tag].append({"deg": deg, "D_H": dh})
                    print(f"[rtwx-o0q0] seed={seed} {name} {tag} Ry{int(deg)} D_H={dh:.4f}", flush=True)
        return {
            "ident": ident,
            "per": per,
            "yaws": yaws,
            "regimes": sorted(snaps),
            "planner": planner,
            "plan_ok": plan_ok,
            "z0": z0,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    ident = [float(x) for x in raw["ident"] if np.isfinite(x)]
    g0 = bool(ident) and max(ident) < G0_MAX
    yaws = [float(y) for y in raw["yaws"]]
    n_yaw = max(len(yaws), 1)
    n_snap = {r: int(len(raw["per"][r]["B1"]) / n_yaw) for r in REGIMES}
    b2_90, b2_hi = [], []
    for r in REGIMES:
        for row in raw["per"][r]["B2"]:
            if abs(row["deg"] - 90.0) < 1e-6:
                b2_90.append(row["D_H"])
            if min(abs(row["deg"] - 180.0), abs(row["deg"] - 0.0)) > 1e-6:
                b2_hi.append(row["D_H"] > TAU)
    need = 1 if smoke else N_MIN_REGIME
    g1_cov = all(n_snap[r] >= need for r in REGIMES)
    g1_b2 = bool(b2_90) and min(b2_90) > TAU and (sum(b2_hi) / max(len(b2_hi), 1) >= G1_B2_FRAC)
    g1 = g1_cov and g1_b2
    b1_ok = [row["D_H"] <= TAU for r in REGIMES for row in raw["per"][r]["B1"]]
    g2 = (sum(b1_ok) / max(len(b1_ok), 1) >= G2_MIN) if b1_ok else False
    g3_reg = {}
    for r in REGIMES:
        rows = raw["per"][r]["B1"]
        g3_reg[r] = bool(rows) and (sum(x["D_H"] <= TAU for x in rows) / len(rows)) >= G3_MIN
    g3 = all(g3_reg.values())
    return {
        "G0": {"ok": g0, "D_H_I_max": max(ident) if ident else None, "n": len(ident), "gate": G0_MAX},
        "G1": {
            "ok": g1,
            "n_snap": n_snap,
            "B2_Ry90_min": min(b2_90) if b2_90 else None,
            "B2_frac_reject": (sum(b2_hi) / max(len(b2_hi), 1)) if b2_hi else 0.0,
            "coverage": g1_cov,
        },
        "G2": {"ok": g2, "P_accept": (sum(b1_ok) / max(len(b1_ok), 1)) if b1_ok else 0.0, "n": len(b1_ok)},
        "G3": {"ok": g3, "per_regime": g3_reg},
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3},
    }


def run_rtwx_o0q0(output: str | Path, config: RTWXO0Q0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "tau": TAU,
        "I_x_over_I_z": I_RATIO,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0] backend={cfg.backend} smoke={cfg.smoke} H={cfg.horizon}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _run_numpy(cfg)
    else:
        chunks = [_collect_robotwin_seed(cfg, int(s)) for s in cfg.seeds]
        raw = {
            "ident": [x for c in chunks for x in c["ident"]],
            "per": {
                r: {
                    "B1": [x for c in chunks for x in c["per"][r]["B1"]],
                    "B2": [x for c in chunks for x in c["per"][r]["B2"]],
                }
                for r in REGIMES
            },
            "yaws": chunks[0]["yaws"] if chunks else list(YAW_DEG),
            "planner": chunks[0]["planner"] if chunks else None,
            "plan_ok": all(bool(c.get("plan_ok")) for c in chunks) if chunks else False,
        }

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(g0=g["g0"], g1=g["g1"], g2=g["g2"], g3=g["g3"])
    pass_q = pattern == "task_yaw_causal_quotient_supported"
    print(f"[rtwx-o0q0] G0={g['g0']} G1={g['g1']} G2={g['g2']} G3={g['g3']} pattern={pattern}", flush=True)
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G1", "G2", "G3")},
        "per_regime": raw["per"],
        "planner": raw.get("planner"),
        "unlocks_o0q1_prereg": bool(pass_q),
        "unlocks_o1": False,
        "unlocks_o0g6r": False,
        "unlocks_o0c2": False,
        "unlocks_symx3": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

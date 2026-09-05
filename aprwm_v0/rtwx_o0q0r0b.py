"""RTWX-O0Q0R0B: pre-contact branch instrument. No restore of contact. No yaw science."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose
from .rtwx_o0q0 import (
    D_H_pair,
    G0_MAX,
    H,
    REGIMES,
    TAU,
    _apply_yaw,
    _hold_action,
    _physx,
    _read_yq,
    _restore,
    _robot_q,
)
from .rtwx_o0q0r0 import (
    COM_DX,
    EE_FAR,
    GRIP_CLOSED,
    I_RATIO,
    MU_A,
    P0_LIFT,
    P0_V,
    P0_W,
    P1_V0,
    PROBE_YAW,
    PROBES,
    _aniso_drag,
    _cup_rigid_audit,
    _follow_pose_qpos,
    _pack_snap,
    _play_one,
    _pop_probe,
    _p0_ic,
    _push_probe,
)
from .rtwx_x0c import _arm_q, _build_action, _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0B_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0r0b.precontact_branch.v1"
SEEDS = (0, 1, 2)
N_MIN = 3
G3_FRAC = 0.80
G = 9.81
DT = 1.0 / 250.0
K_PHYS = 5
DT_A = K_PHYS * DT
DT_MARGIN = 0.03
D_NEAR = 0.08
D_GRASP = 0.05
N_P3 = 3
DELTA_T = 1
ROOT_DIST = 0.12
H_P1 = H


@dataclass(frozen=True)
class RTWXO0Q0R0BConfig:
    output: str = "runs/rtwx_o0q0r0b"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0R0BConfig) -> RTWXO0Q0R0BConfig:
    if cfg.smoke:
        return replace(cfg, seeds=(63,))
    return replace(cfg, seeds=SEEDS)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0R0B must not write there")


def t_hit(h: float, vz: float, g: float = G) -> float:
    h = float(h)
    if h <= 1e-9:
        return 0.0
    disc = max(float(vz) ** 2 + 2.0 * g * h, 0.0)
    return (float(vz) + float(np.sqrt(disc))) / g


def h_p0(h: float, vz: float, dt_a: float = DT_A, margin: float = DT_MARGIN) -> int:
    t = t_hit(h, vz)
    return int(np.floor((t - margin) / dt_a))


def _pattern(*, p0_ok: bool, branch_ok: bool, g3: bool) -> str:
    if not p0_ok:
        return "freeflight_horizon_failure"
    if not branch_ok:
        return "precontact_branch_failure"
    if not g3:
        return "regime_excitation_failure"
    return "counterfactual_instrument_qualified"


def _t_p2(evs: list[dict[str, Any]]) -> int | None:
    for i, s in enumerate(evs):
        if float(s["dist"]) < D_NEAR:
            return i
    return None


def _t_p3(evs: list[dict[str, Any]]) -> int | None:
    streak = 0
    start: int | None = None
    for i, s in enumerate(evs):
        if float(s["grip"]) <= GRIP_CLOSED and float(s["dist"]) < D_GRASP:
            if streak == 0:
                start = i
            streak += 1
            if streak >= N_P3:
                return start
        else:
            streak = 0
            start = None
    return None


def _qpos_u(env: Any, ql: np.ndarray, qr: np.ndarray, lg: float, rg: float) -> dict[str, Any]:
    u = np.concatenate([np.asarray(ql, dtype=np.float64).reshape(-1), [float(lg)], np.asarray(qr, dtype=np.float64).reshape(-1), [float(rg)]])
    return {"kind": "qpos", "u": u, "action_type": "qpos"}


def _hold_phys(env: Any, q: np.ndarray, k: int) -> None:
    n_l = q.size // 2
    z = np.zeros(n_l - 1)
    env.robot.set_arm_joints(q[: n_l - 1], z, "left")
    env.robot.set_arm_joints(q[n_l : q.size - 1], z, "right")
    env.robot.set_gripper(float(q[n_l - 1]), "left")
    env.robot.set_gripper(float(q[-1]), "right")
    for _ in range(int(k)):
        env.scene.step()


def _roll_p0(env: Any, snap: dict[str, Any], n: int, *, yaw: float | None, probe: str | None) -> tuple[list, list]:
    _restore(env, snap)
    saved = _push_probe(env, probe)
    try:
        if yaw is not None and abs(float(yaw)) > 1e-9:
            _apply_yaw(env, float(yaw))
        q = np.asarray(snap["q"], dtype=np.float64)
        ys, ev = [], []
        for _ in range(n):
            _hold_phys(env, q, K_PHYS)
            ys.append(_read_yq(env))
            ev.append(_pack_snap(env))
        return ys, ev
    finally:
        _pop_probe(env, saved)


def _roll_A(env: Any, root: dict[str, Any], actions: list[dict[str, Any]], *, yaw: float | None, probe: str | None) -> tuple[list, list]:
    _restore(env, root)
    saved = _push_probe(env, probe)
    try:
        if yaw is not None and abs(float(yaw)) > 1e-9:
            _apply_yaw(env, float(yaw))
        if probe == "friction":
            c = _physx(env.cup)
            if c is not None:
                c.linear_velocity = P1_V0
        ys, ev = [], []
        for a in actions:
            if probe == "friction":
                _aniso_drag(env)
            _play_one(env, a)
            ys.append(_read_yq(env))
            ev.append(_pack_snap(env))
        return ys, ev
    finally:
        _pop_probe(env, saved)


def _generate_A_star(env: Any) -> list[dict[str, Any]]:
    p, _ = _cup_pose(env)
    arm = "right" if float(p[0]) > 0 else "left"
    ee = np.asarray(env.robot.get_right_ee_pose() if arm == "right" else env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)
    quat = ee[3:7] if ee.size >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
    pre = p + np.array([0.0, 0.0, 0.10])
    near = p + np.array([0.0, 0.0, 0.035])
    A: list[dict[str, Any]] = []
    orig = env.take_action

    def rec(action, action_type="qpos"):
        A.append({"kind": "qpos" if action_type != "ee" else "ee", "u": np.asarray(action, dtype=np.float64).copy(), "action_type": action_type})
        return orig(action, action_type=action_type)

    env.take_action = rec  # type: ignore[method-assign]
    try:
        e0 = ee[:3].copy()
        for tgt, n in ((pre, 12), (near, 8)):
            for k in range(1, n + 1):
                pk = (1.0 - k / n) * e0 + (k / n) * tgt
                _follow_pose_qpos(env, arm, np.concatenate([pk, quat]))
            e0 = np.asarray(env.robot.get_right_ee_pose() if arm == "right" else env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:3]
        q, _ = _robot_q(env)
        n_l = q.size // 2
        lg, rg = float(q[n_l - 1]), float(q[-1])
        for _ in range(8):
            if arm == "right":
                rg = max(0.0, rg - 0.15)
            else:
                lg = max(0.0, lg - 0.15)
            u = np.concatenate([_arm_q(env, "left"), [lg], _arm_q(env, "right"), [rg]])
            env.take_action(u, action_type="qpos")
            q, _ = _robot_q(env)
            lg, rg = float(q[n_l - 1]), float(q[-1])
        hold = _hold_action(env)
        for _ in range(6):
            env.take_action(hold, action_type="qpos")
        return list(A)
    finally:
        env.take_action = orig  # type: ignore[method-assign]


def _collect(cfg: RTWXO0Q0R0BConfig, seed: int, args: dict[str, Any]) -> dict[str, Any]:
    import sapien

    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 4000
    try:
        audit = _cup_rigid_audit(env.cup)
        p_tab, quat0 = _cup_pose(env)
        z_tab = float(p_tab[2])
        root = _pack_snap(env)
        root_ok = float(root["dist"]) > ROOT_DIST and float(root["z"]) <= z_tab + 0.015
        A: list[dict[str, Any]] = []
        if root_ok:
            A = _generate_A_star(env)
        getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p_tab, quat0))
        c = _physx(env.cup)
        if c is not None:
            c.linear_velocity = np.zeros(3)
            c.angular_velocity = np.zeros(3)
        p0 = None
        hp0 = 0
        if _p0_ic(env, z_tab, quat0):
            p0 = _pack_snap(env)
            hp0 = h_p0(float(p0["z"]) - z_tab, float(p0["v"][2]))
        return {"root": root if root_ok else None, "A": A, "p0": p0, "H_P0": hp0, "z_tab": z_tab, "audit": audit}
    finally:
        try:
            env.close()
        except Exception:
            pass


def _numpy_run(cfg: RTWXO0Q0R0BConfig) -> dict[str, Any]:
    hp = max(1, h_p0(P0_LIFT, 0.0))
    return {
        "H_P0": hp,
        "ident_p0": [0.0],
        "p0_contact": [False],
        "ident_root": [0.0],
        "ident_A": [0.0],
        "n_evt": {r: 1 for r in REGIMES},
        "dt_evt": {"P2": [0], "P3": [0]},
        "excite": {r: [0.2] for r in REGIMES},
        "audit": {"note": "numpy smoke", "I_x_eq_I_z": True},
        "planner": "numpy",
        "min_dist": [0.03],
    }


def _measure(cfg: RTWXO0Q0R0BConfig, seed: int, args: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 4000
    out: dict[str, Any] = {
        "ident_p0": [],
        "p0_contact": [],
        "ident_root": [],
        "ident_A": [],
        "excite": {r: [] for r in REGIMES},
        "n_evt": {r: 0 for r in REGIMES},
        "dt_evt": {"P2": [], "P3": []},
        "min_dist": None,
        "H_P0": blob.get("H_P0"),
    }
    try:
        hp = int(blob.get("H_P0") or 0)
        if blob.get("p0") is not None and hp >= 1:
            y1, e1 = _roll_p0(env, blob["p0"], hp, yaw=None, probe=None)
            y2, e2 = _roll_p0(env, blob["p0"], hp, yaw=None, probe=None)
            out["ident_p0"].append(D_H_pair(y1, y2))
            hit = any(float(s["J"]) > 1e-5 or float(s["z"]) <= float(blob["z_tab"]) + 0.01 for s in e1 + e2)
            out["p0_contact"].append(bool(hit))
            yp, _ = _roll_p0(env, blob["p0"], hp, yaw=None, probe="inertia")
            ya, _ = _roll_p0(env, blob["p0"], hp, yaw=PROBE_YAW, probe="inertia")
            out["excite"]["P0"].append(D_H_pair(yp, ya))
            if not hit:
                out["n_evt"]["P0"] = 1
            print(f"[rtwx-o0q0r0b] seed={seed} P0 H={hp} D_H(I)={out['ident_p0'][-1]:.4f} contact={hit} D_H(Ry90)={out['excite']['P0'][-1]:.4f}", flush=True)
        root, A = blob.get("root"), blob.get("A") or []
        if root is not None:
            hold = [{"kind": "qpos", "u": _hold_action(env), "action_type": "qpos"}]
            # rebuild hold from restored root
            _restore(env, root)
            hold = [{"kind": "qpos", "u": _hold_action(env).copy(), "action_type": "qpos"} for _ in range(H_P1)]
            y1, _ = _roll_A(env, root, hold, yaw=None, probe=None)
            y2, _ = _roll_A(env, root, hold, yaw=None, probe=None)
            dh = D_H_pair(y1, y2)
            out["ident_root"].append(dh)
            yp, _ = _roll_A(env, root, hold, yaw=None, probe="friction")
            ya, _ = _roll_A(env, root, hold, yaw=PROBE_YAW, probe="friction")
            out["excite"]["P1"].append(D_H_pair(yp, ya))
            out["n_evt"]["P1"] = 1
            print(f"[rtwx-o0q0r0b] seed={seed} P1 D_H(I)={dh:.4f} D_H(Ry90,fric)={out['excite']['P1'][-1]:.4f}", flush=True)
        if root is not None and A:
            y1, e1 = _roll_A(env, root, A, yaw=None, probe=None)
            y2, e2 = _roll_A(env, root, A, yaw=None, probe=None)
            out["ident_A"].append(D_H_pair(y1, y2))
            out["min_dist"] = min(float(s["dist"]) for s in e1) if e1 else None
            t2a, t2b = _t_p2(e1), _t_p2(e2)
            t3a, t3b = _t_p3(e1), _t_p3(e2)
            if t2a is not None and t2b is not None:
                out["dt_evt"]["P2"].append(abs(t2a - t2b))
                out["n_evt"]["P2"] = 1
            if t3a is not None and t3b is not None:
                out["dt_evt"]["P3"].append(abs(t3a - t3b))
                out["n_evt"]["P3"] = 1
            yp, _ = _roll_A(env, root, A, yaw=None, probe="com")
            ya, _ = _roll_A(env, root, A, yaw=PROBE_YAW, probe="com")
            dhc = D_H_pair(yp, ya)
            if out["n_evt"]["P2"]:
                out["excite"]["P2"].append(dhc)
            if out["n_evt"]["P3"]:
                out["excite"]["P3"].append(dhc)
            print(
                f"[rtwx-o0q0r0b] seed={seed} A* n={len(A)} D_H(I)={out['ident_A'][-1]:.4f} min_d={out['min_dist']} tP2={t2a}/{t2b} tP3={t3a}/{t3b} D_H(Ry90,com)={dhc:.4f}",
                flush=True,
            )
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    need = 1 if smoke else N_MIN
    hp = raw.get("H_P0")
    ident_p0 = [float(x) for x in raw["ident_p0"] if np.isfinite(x)]
    p0_hit = any(raw.get("p0_contact") or [])
    p0_n = int(raw["n_evt"].get("P0", 0) if isinstance(raw["n_evt"]["P0"], int) else raw["n_evt"]["P0"])
    # n_evt summed in merge
    g0_p0 = bool(ident_p0) and max(ident_p0) < G0_MAX and not p0_hit and (hp is None or int(hp) >= 1)
    if smoke and ident_p0 and max(ident_p0) < G0_MAX:
        g0_p0 = True
    ident_root = [float(x) for x in raw["ident_root"] if np.isfinite(x)]
    ident_A = [float(x) for x in raw["ident_A"] if np.isfinite(x)]
    g0_root = (not ident_root) or max(ident_root) < G0_MAX
    g0_A = (not ident_A) or max(ident_A) < G0_MAX
    n_evt = {r: int(raw["n_evt"][r]) for r in REGIMES}
    g1 = all(n_evt[r] >= need for r in REGIMES)
    g2 = True
    for k in ("P2", "P3"):
        dts = [int(x) for x in raw["dt_evt"][k]]
        if n_evt[k] < need:
            g2 = False
        elif any(d > DELTA_T for d in dts):
            g2 = False
    g3_reg = {}
    for r in REGIMES:
        xs = [float(x) > TAU for x in raw["excite"][r]]
        g3_reg[r] = bool(xs) and (sum(xs) / len(xs) >= G3_FRAC)
    g3 = all(g3_reg.values())
    branch = bool(g0_root and g0_A and g1 and g2)
    p0_ok = bool(g0_p0) and n_evt["P0"] >= need
    return {
        "G0": {
            "ok": g0_p0 and g0_root and g0_A,
            "P0": {"ok": g0_p0, "D_H_I": ident_p0, "H_P0": hp, "contact": raw.get("p0_contact")},
            "root": {"ok": g0_root, "D_H_I": ident_root},
            "A_star": {"ok": g0_A, "D_H_I": ident_A},
            "gate": G0_MAX,
        },
        "G1": {"ok": g1, "n_evt": n_evt, "need": need, "min_dist": raw.get("min_dist")},
        "G2": {"ok": g2, "dt": raw["dt_evt"], "delta_t": DELTA_T},
        "G3": {"ok": g3, "per_regime": g3_reg, "Ry90": {r: list(raw["excite"][r]) for r in REGIMES}},
        "gates": {"p0_ok": p0_ok, "branch_ok": branch, "g3": g3},
    }


def run_rtwx_o0q0r0b(output: str | Path, config: RTWXO0Q0R0BConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0R0BConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0R0B",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "no_restore_contact": True,
        "no_teleport_into_contact": True,
        "no_yaw_science": True,
        "tau": TAU,
        "G0_max": G0_MAX,
        "delta_t": DELTA_T,
        "dt_a_p0": DT_A,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0r0b] backend={cfg.backend} smoke={cfg.smoke}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _numpy_run(cfg)
        raw["n_evt"] = {r: int(raw["n_evt"][r]) for r in REGIMES}
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
        chunks = []
        for s in cfg.seeds:
            blob = _collect(cfg, int(s), args)
            print(f"[rtwx-o0q0r0b] seed={s} H_P0={blob.get('H_P0')} |A*|={len(blob.get('A') or [])} root={blob.get('root') is not None}", flush=True)
            chunks.append(_measure(cfg, int(s), args, blob))
            chunks[-1]["audit"] = blob.get("audit")
            chunks[-1]["H_P0"] = blob.get("H_P0")
        raw = {
            "H_P0": chunks[0]["H_P0"] if chunks else None,
            "ident_p0": [x for c in chunks for x in c["ident_p0"]],
            "p0_contact": [x for c in chunks for x in c["p0_contact"]],
            "ident_root": [x for c in chunks for x in c["ident_root"]],
            "ident_A": [x for c in chunks for x in c["ident_A"]],
            "n_evt": {r: sum(int(c["n_evt"][r]) for c in chunks) for r in REGIMES},
            "dt_evt": {k: [x for c in chunks for x in c["dt_evt"][k]] for k in ("P2", "P3")},
            "excite": {r: [x for c in chunks for x in c["excite"][r]] for r in REGIMES},
            "audit": chunks[0].get("audit") if chunks else {},
            "planner": planner,
            "min_dist": [c.get("min_dist") for c in chunks],
        }

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(p0_ok=g["p0_ok"], branch_ok=g["branch_ok"], g3=g["g3"])
    qual = pattern == "counterfactual_instrument_qualified"
    print(f"[rtwx-o0q0r0b] p0={g['p0_ok']} branch={g['branch_ok']} G3={g['g3']} pattern={pattern}", flush=True)
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G1", "G2", "G3")},
        "rigid_audit": raw.get("audit"),
        "planner": raw.get("planner"),
        "unlocks_o0q0r1_prereg": bool(qual),
        "unlocks_o0q1_prereg": False,
        "unlocks_o1": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

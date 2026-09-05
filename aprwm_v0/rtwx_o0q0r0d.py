"""RTWX-O0Q0R0D: native control-channel audit. No IK. No hdf5-as-action. No yaw science."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import traceback

import numpy as np

from .rtwx_o0 import TASK, _cup_pose
from .rtwx_o0q0 import D_H_pair, G0_MAX, TAU, _read_yq, _restore, _robot_q
from .rtwx_o0q0r0 import _pack_snap
from .rtwx_o0q0r0c import (
    H_P0,
    I_RATIO_P0,
    P0_W_C,
    _t_p2,
    _t_p3,
)
from .rtwx_o0q0r0b import ROOT_DIST
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0D_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0r0d.native_control_channel.v1"
SEEDS = (0, 1, 2)
EPS_Q = 0.05
EPS_EE = 0.03
N_P2_MIN = 3
N_P3_MIN = 3
SAVE_FREQ_DOC = 15  # official demo_clean.yml; not used as A*
PATTERNS = (
    "control_channel_unresolved",
    "capture_incomplete",
    "robot_replay_failure",
    "task_event_replay_failure",
    "precontact_identity_failure",
    "native_control_trace_qualified",
)


@dataclass(frozen=True)
class RTWXO0Q0R0DConfig:
    output: str = "runs/rtwx_o0q0r0d"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0R0DConfig) -> RTWXO0Q0R0DConfig:
    if cfg.smoke:
        return replace(cfg, seeds=(65,))
    return replace(cfg, seeds=SEEDS)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0R0D must not write there")


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "control_channel_unresolved"
    if not g1:
        return "capture_incomplete"
    if not g2:
        return "robot_replay_failure"
    if not g3:
        return "task_event_replay_failure"
    if not g4:
        return "precontact_identity_failure"
    return "native_control_trace_qualified"


def _as1(x: Any) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(a[0]) if a.size else float("nan")


def _drive_q(robot: Any) -> np.ndarray:
    left = [_as1(j.get_drive_target()) for j in robot.left_arm_joints]
    right = [_as1(j.get_drive_target()) for j in robot.right_arm_joints]
    lg = _as1(robot.left_gripper[0][0].get_drive_target()) if robot.left_gripper else 0.0
    rg = _as1(robot.right_gripper[0][0].get_drive_target()) if robot.right_gripper else 0.0
    return np.asarray(left + [lg] + right + [rg], dtype=np.float64)


def _drive_qd(robot: Any) -> np.ndarray:
    def vel(j: Any) -> float:
        fn = getattr(j, "get_drive_velocity_target", None)
        return _as1(fn()) if fn is not None else 0.0

    left = [vel(j) for j in robot.left_arm_joints]
    right = [vel(j) for j in robot.right_arm_joints]
    lg = vel(robot.left_gripper[0][0]) if robot.left_gripper else 0.0
    rg = vel(robot.right_gripper[0][0]) if robot.right_gripper else 0.0
    return np.asarray(left + [lg] + right + [rg], dtype=np.float64)


def _meas_q(env: Any) -> np.ndarray:
    q, _ = _robot_q(env)
    return q


def _ee_p(env: Any) -> np.ndarray:
    el = np.asarray(env.robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:3]
    er = np.asarray(env.robot.get_right_ee_pose(), dtype=np.float64).reshape(-1)[:3]
    return np.concatenate([el, er])


def _joint_state_vec(robot: Any) -> np.ndarray:
    return np.asarray(robot.get_left_arm_jointState() + robot.get_right_arm_jointState(), dtype=np.float64)


def _set_grip_drive(robot: Any, arm: str, target: float, vel: float) -> None:
    joints = robot.left_gripper if arm == "left" else robot.right_gripper
    if not joints:
        return
    for joint in joints:
        real = joint[0]
        real.set_drive_target(float(target))
        if hasattr(real, "set_drive_velocity_target"):
            real.set_drive_velocity_target(float(vel))


def _apply_drive(env: Any, a: dict[str, np.ndarray]) -> None:
    q = np.asarray(a["q"], dtype=np.float64).reshape(-1)
    qd = np.asarray(a["qd"], dtype=np.float64).reshape(-1)
    n = q.size // 2
    env.robot.set_arm_joints(q[: n - 1], qd[: n - 1], "left")
    env.robot.set_arm_joints(q[n : q.size - 1], qd[n : q.size - 1], "right")
    _set_grip_drive(env.robot, "left", float(q[n - 1]), float(qd[n - 1]))
    _set_grip_drive(env.robot, "right", float(q[-1]), float(qd[-1]))


def _shape_of(x: Any) -> list[int] | str:
    if x is None:
        return "none"
    if isinstance(x, dict):
        if "position" in x:
            return list(np.asarray(x["position"]).shape)
        return {k: _shape_of(x[k]) for k in list(x)[:8]}  # type: ignore[return-value]
    try:
        return list(np.asarray(x).shape)
    except Exception:
        return type(x).__name__


class _Tap:
    def __init__(self) -> None:
        self.counts = {
            "take_action": 0,
            "take_dense_action": 0,
            "plan_path": 0,
            "set_arm_joints": 0,
            "set_gripper": 0,
            "scene_step": 0,
        }
        self.action_types: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.A: list[dict[str, np.ndarray]] = []
        self.q_meas: list[np.ndarray] = []
        self.q_js: list[np.ndarray] = []
        self.ee: list[np.ndarray] = []
        self.snaps: list[dict[str, Any]] = []
        self._orig: dict[str, Any] = {}

    def install(self, env: Any) -> None:
        robot = env.robot
        self._orig["take_action"] = env.take_action
        self._orig["take_dense_action"] = env.take_dense_action
        self._orig["set_arm"] = robot.set_arm_joints
        self._orig["set_grip"] = robot.set_gripper
        self._orig["step"] = env.scene.step
        self._orig["left_plan"] = robot.left_plan_path
        self._orig["right_plan"] = robot.right_plan_path
        self._orig["left_multi"] = getattr(robot, "left_plan_multi_path", None)
        self._orig["right_multi"] = getattr(robot, "right_plan_multi_path", None)
        tap = self

        def take_action(action, action_type="qpos"):
            tap.counts["take_action"] += 1
            tap.action_types.append(str(action_type))
            tap.events.append({"layer": "take_action_input", "action_type": str(action_type), "shape": _shape_of(action)})
            return tap._orig["take_action"](action, action_type=action_type)

        def take_dense(control_seq, save_freq=-1):
            tap.counts["take_dense_action"] += 1
            arms = {}
            if isinstance(control_seq, dict):
                for k in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
                    arms[k] = _shape_of(control_seq.get(k))
            tap.events.append({"layer": "dense_interpolated_q", "seq_shapes": arms})
            return tap._orig["take_dense_action"](control_seq, save_freq=save_freq)

        def set_arm(target_position, target_velocity, arm_tag):
            tap.counts["set_arm_joints"] += 1
            tap.events.append({"layer": "joint_drive_target", "arm": str(arm_tag), "shape": _shape_of(target_position)})
            return tap._orig["set_arm"](target_position, target_velocity, arm_tag)

        def set_grip(gripper_val, arm_tag, gripper_eps=0.1):
            tap.counts["set_gripper"] += 1
            tap.events.append({"layer": "gripper_command", "arm": str(arm_tag), "val": float(gripper_val)})
            return tap._orig["set_grip"](gripper_val, arm_tag, gripper_eps)

        def plan(fn, tag):
            def wrapped(*a, **k):
                tap.counts["plan_path"] += 1
                pose = a[0] if a else k.get("target_pose")
                tap.events.append({"layer": "planner_input_ee", "arm": tag, "shape": _shape_of(pose)})
                out = fn(*a, **k)
                st = out.get("status") if isinstance(out, dict) else type(out).__name__
                pos = out.get("position") if isinstance(out, dict) else None
                tap.events.append({"layer": "planner_output_q", "arm": tag, "status": st, "shape": _shape_of(pos)})
                return out

            return wrapped

        def stepped():
            tap.counts["scene_step"] += 1
            tap.A.append({"q": _drive_q(robot), "qd": _drive_qd(robot)})
            tap.q_meas.append(_meas_q(env))
            tap.q_js.append(_joint_state_vec(robot))
            tap.ee.append(_ee_p(env))
            tap.snaps.append(_pack_snap(env))
            return tap._orig["step"]()

        env.take_action = take_action  # type: ignore[method-assign]
        env.take_dense_action = take_dense  # type: ignore[method-assign]
        robot.set_arm_joints = set_arm  # type: ignore[method-assign]
        robot.set_gripper = set_grip  # type: ignore[method-assign]
        env.scene.step = stepped  # type: ignore[method-assign]
        robot.left_plan_path = plan(self._orig["left_plan"], "left")  # type: ignore[method-assign]
        robot.right_plan_path = plan(self._orig["right_plan"], "right")  # type: ignore[method-assign]
        if self._orig["left_multi"] is not None:
            robot.left_plan_multi_path = plan(self._orig["left_multi"], "left_multi")  # type: ignore[method-assign]
        if self._orig["right_multi"] is not None:
            robot.right_plan_multi_path = plan(self._orig["right_multi"], "right_multi")  # type: ignore[method-assign]

    def uninstall(self, env: Any) -> None:
        if not self._orig:
            return
        env.take_action = self._orig["take_action"]  # type: ignore[method-assign]
        env.take_dense_action = self._orig["take_dense_action"]  # type: ignore[method-assign]
        env.robot.set_arm_joints = self._orig["set_arm"]  # type: ignore[method-assign]
        env.robot.set_gripper = self._orig["set_grip"]  # type: ignore[method-assign]
        env.scene.step = self._orig["step"]  # type: ignore[method-assign]
        env.robot.left_plan_path = self._orig["left_plan"]  # type: ignore[method-assign]
        env.robot.right_plan_path = self._orig["right_plan"]  # type: ignore[method-assign]
        if self._orig.get("left_multi") is not None:
            env.robot.left_plan_multi_path = self._orig["left_multi"]  # type: ignore[method-assign]
        if self._orig.get("right_multi") is not None:
            env.robot.right_plan_multi_path = self._orig["right_multi"]  # type: ignore[method-assign]


def _path_string(c: dict[str, int]) -> str:
    bits = ["expert"]
    if c["plan_path"]:
        bits += ["EE target", "planner"]
    if c["take_dense_action"]:
        bits += ["dense qpos sequence", "take_dense_action"]
    if c["take_action"]:
        bits += ["take_action"]
    if c["set_arm_joints"] or c["set_gripper"]:
        bits += ["set_arm_joints/set_gripper", "sapien_drive_target"]
    bits += ["physics"]
    return " -> ".join(bits)


def _semantic(A: list[dict[str, np.ndarray]], q_meas: list[np.ndarray], q_js: list[np.ndarray]) -> dict[str, Any]:
    if len(A) < 2:
        return {"n": len(A), "A_minus_q_t": None, "A_minus_q_next": None, "A_minus_jointState": None, "A_eq_q_next": None}
    aq = np.stack([a["q"] for a in A], axis=0)
    qm = np.stack(q_meas, axis=0)
    js = np.stack(q_js, axis=0)
    d_t = np.linalg.norm(aq - qm, axis=1)
    d_n = np.linalg.norm(aq[:-1] - qm[1:], axis=1)
    n_arm = min(aq.shape[1], js.shape[1])
    d_js = np.linalg.norm(aq[:, :n_arm] - js[:, :n_arm], axis=1)

    def pack(d: np.ndarray) -> dict[str, float]:
        return {"mean": float(np.mean(d)), "max": float(np.max(d)), "median": float(np.median(d))}

    return {
        "n": int(len(A)),
        "A_minus_q_t": pack(d_t),
        "A_minus_q_next": pack(d_n),
        "A_minus_jointState": pack(d_js),
        "A_eq_q_next": bool(float(np.max(d_n)) < 1e-6),
        "jointState_is_drive": bool(float(np.max(d_js)) < 1e-6),
    }


def _pick_t0(snaps: list[dict[str, Any]], t_p2: int | None) -> int | None:
    if t_p2 is None or t_p2 < 1:
        return None
    t0 = None
    for i in range(int(t_p2)):
        if float(snaps[i]["dist"]) > ROOT_DIST:
            t0 = i
    return t0


def _collect(cfg: RTWXO0Q0R0DConfig, seed: int, args: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.save_data = False
    env.save_freq = None
    env.step_lim = 20000
    tap = _Tap()
    tap.install(env)
    err = None
    ok = False
    try:
        try:
            env.play_once()
            ok = bool(getattr(env, "plan_success", True))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ok = False
            print(f"[rtwx-o0q0r0d] seed={seed} play_once traceback:\n{traceback.format_exc()}", flush=True)
        t2, t3 = _t_p2(tap.snaps) if tap.snaps else None, _t_p3(tap.snaps) if tap.snaps else None
        print(
            f"[rtwx-o0q0r0d] seed={seed} play_ok={ok} err={err} steps={tap.counts['scene_step']} "
            f"take_action={tap.counts['take_action']} dense={tap.counts['take_dense_action']} "
            f"plan={tap.counts['plan_path']} tP2={t2} tP3={t3}",
            flush=True,
        )
        return {
            "ok": ok,
            "error": err,
            "plan_success": bool(getattr(env, "plan_success", False)),
            "counts": dict(tap.counts),
            "action_types": list(tap.action_types),
            "events": tap.events[:80],
            "path": _path_string(tap.counts),
            "A": tap.A,
            "q_meas": tap.q_meas,
            "ee": tap.ee,
            "snaps": tap.snaps,
            "semantic": _semantic(tap.A, tap.q_meas, tap.q_js),
            "t_p2": t2,
            "t_p3": t3,
            "t0": _pick_t0(tap.snaps, t2),
            "min_dist": min((float(s["dist"]) for s in tap.snaps), default=None),
            "reset": tap.snaps[0] if tap.snaps else None,
        }
    finally:
        tap.uninstall(env)
        try:
            env.close()
        except Exception:
            pass


def _replay(cfg: RTWXO0Q0R0DConfig, seed: int, args: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    A = blob.get("A") or []
    out: dict[str, Any] = {
        "e_q": None,
        "e_ee": None,
        "t_p2": None,
        "t_p3": None,
        "ident": None,
        "n": 0,
    }
    if not A:
        return out
    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.save_data = False
    env.save_freq = None
    env.step_lim = 20000
    try:
        q_rep, ee_rep, snaps = [], [], []
        for a in A:
            _apply_drive(env, a)
            env.scene.step()
            q_rep.append(_meas_q(env))
            ee_rep.append(_ee_p(env))
            snaps.append(_pack_snap(env))
        n = min(len(q_rep), len(blob["q_meas"]))
        if n:
            eq = [float(np.max(np.abs(q_rep[i] - blob["q_meas"][i]))) for i in range(n)]
            ee = [float(np.linalg.norm(ee_rep[i] - blob["ee"][i])) for i in range(n)]
            out["e_q"] = float(max(eq))
            out["e_ee"] = float(max(ee))
            out["n"] = n
        out["t_p2"] = _t_p2(snaps)
        out["t_p3"] = _t_p3(snaps)
        t0 = blob.get("t0")
        if t0 is not None and blob.get("snaps") and int(t0) < len(A):
            root = blob["snaps"][int(t0)]
            suf = A[int(t0) :]
            y1, y2 = [], []
            for _ in range(2):
                _restore(env, root)
                ys = []
                for a in suf:
                    _apply_drive(env, a)
                    env.scene.step()
                    ys.append(_read_yq(env))
                if not y1:
                    y1 = ys
                else:
                    y2 = ys
            if y1 and y2:
                out["ident"] = D_H_pair(y1, y2)
        print(
            f"[rtwx-o0q0r0d] seed={seed} replay n={out['n']} e_q={out['e_q']} e_ee={out['e_ee']} tP2={out['t_p2']} tP3={out['t_p3']} D_H(I)={out['ident']}",
            flush=True,
        )
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


def _numpy_run(_cfg: RTWXO0Q0R0DConfig) -> dict[str, Any]:
    return {
        "channel": "sapien_joint_drive_target",
        "path": "expert -> EE target -> planner -> dense qpos sequence -> take_dense_action -> sapien_drive_target -> physics",
        "n_take_action": 0,
        "ident": [0.0],
        "e_q": [0.0],
        "e_ee": [0.0],
        "n_p2": 1,
        "n_p3": 1,
        "n_ok": 1,
        "n_seed": 1,
        "complete": True,
        "semantic": {"A_eq_q_next": False, "jointState_is_drive": True},
    }


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    need = 1 if smoke else 3
    n_ok = int(raw["n_ok"])
    g0 = str(raw.get("channel")) == "sapien_joint_drive_target"
    g1 = bool(raw.get("complete")) and n_ok >= need
    eq = [float(x) for x in raw["e_q"] if x is not None and np.isfinite(x)]
    ee = [float(x) for x in raw["e_ee"] if x is not None and np.isfinite(x)]
    g2 = bool(eq) and max(eq) < EPS_Q and bool(ee) and max(ee) < EPS_EE
    n_p2, n_p3 = int(raw["n_p2"]), int(raw["n_p3"])
    g3 = n_p2 >= (1 if smoke else N_P2_MIN) and n_p3 >= (1 if smoke else N_P3_MIN)
    ident = [float(x) for x in raw["ident"] if x is not None and np.isfinite(x)]
    g4 = bool(ident) and max(ident) < G0_MAX
    return {
        "G0": {"ok": g0, "channel": raw.get("channel"), "path": raw.get("path"), "n_take_action": raw.get("n_take_action")},
        "G1": {"ok": g1, "n_ok": n_ok, "need": need, "complete": raw.get("complete")},
        "G2": {"ok": g2, "e_q": eq, "e_ee": ee, "eps_q": EPS_Q, "eps_ee": EPS_EE},
        "G3": {"ok": g3, "n_p2": n_p2, "n_p3": n_p3, "need_p2": N_P2_MIN if not smoke else 1, "need_p3": N_P3_MIN if not smoke else 1},
        "G4": {"ok": g4, "D_H_I": ident, "gate": G0_MAX},
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4},
    }


def _channel_of(blobs: list[dict[str, Any]]) -> str:
    if not blobs:
        return "unresolved"
    n_ta = sum(int(b["counts"]["take_action"]) for b in blobs)
    n_drv = sum(int(b["counts"]["set_arm_joints"]) + int(b["counts"]["set_gripper"]) for b in blobs)
    n_st = sum(int(b["counts"]["scene_step"]) for b in blobs)
    if n_st < 1 or n_drv < 1:
        return "unresolved"
    if n_ta > 0:
        return "mixed"
    return "sapien_joint_drive_target"


def run_rtwx_o0q0r0d(output: str | Path, config: RTWXO0Q0R0DConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0R0DConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0R0D",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "no_ik": True,
        "no_hdf5_as_action": True,
        "no_restore_contact": True,
        "no_yaw_science": True,
        "tau": TAU,
        "H_P0_frozen": H_P0,
        "I_ratio_p0_frozen": I_RATIO_P0,
        "P0_W_frozen": P0_W_C.tolist(),
        "eps_q": EPS_Q,
        "eps_ee": EPS_EE,
        "save_freq_dataset": SAVE_FREQ_DOC,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0r0d] backend={cfg.backend} smoke={cfg.smoke}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _numpy_run(cfg)
        demos = []
    else:
        import os
        import sys

        from .task_x0 import _load_task_args

        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_raster_shader()
        args = _load_task_args(repo, TASK, "demo_clean")
        args["task_name"] = TASK
        args["render_freq"] = 0
        args["collect_data"] = False
        blobs, reps = [], []
        for s in cfg.seeds:
            blob = _collect(cfg, int(s), args)
            blobs.append(blob)
            reps.append(_replay(cfg, int(s), args, blob))
            np.savez_compressed(
                root / f"A_star_seed{s}.npz",
                q=np.stack([a["q"] for a in blob["A"]], axis=0) if blob["A"] else np.zeros((0, 1)),
                qd=np.stack([a["qd"] for a in blob["A"]], axis=0) if blob["A"] else np.zeros((0, 1)),
            )
        ch = _channel_of(blobs)
        raw = {
            "channel": ch,
            "path": blobs[0]["path"] if blobs else "",
            "n_take_action": sum(int(b["counts"]["take_action"]) for b in blobs),
            "ident": [r.get("ident") for r in reps],
            "e_q": [r.get("e_q") for r in reps],
            "e_ee": [r.get("e_ee") for r in reps],
            "n_p2": sum(1 for r in reps if r.get("t_p2") is not None),
            "n_p3": sum(1 for r in reps if r.get("t_p3") is not None),
            "n_ok": sum(1 for b in blobs if b.get("ok")),
            "n_seed": len(blobs),
            "complete": all(bool(b.get("ok")) and int(b["counts"]["scene_step"]) == len(b.get("A") or []) and len(b.get("A") or []) > 0 for b in blobs),
            "semantic": blobs[0]["semantic"] if blobs else {},
            "counts": [b["counts"] for b in blobs],
            "errors": [b.get("error") for b in blobs],
            "min_dist": [b.get("min_dist") for b in blobs],
            "action_types": [b.get("action_types") for b in blobs],
            "events": [b.get("events") for b in blobs],
            "t_p2_cap": [b.get("t_p2") for b in blobs],
            "t_p3_cap": [b.get("t_p3") for b in blobs],
        }
        demos = [
            {
                "seed": int(s),
                "ok": blobs[i]["ok"],
                "error": blobs[i]["error"],
                "path": blobs[i]["path"],
                "counts": blobs[i]["counts"],
                "semantic": blobs[i]["semantic"],
                "min_dist": blobs[i]["min_dist"],
                "e_q": reps[i]["e_q"],
                "e_ee": reps[i]["e_ee"],
                "ident": reps[i]["ident"],
                "t_p2": reps[i]["t_p2"],
                "t_p3": reps[i]["t_p3"],
            }
            for i, s in enumerate(cfg.seeds)
        ]

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(g0=g["g0"], g1=g["g1"], g2=g["g2"], g3=g["g3"], g4=g["g4"])
    qual = pattern == "native_control_trace_qualified"
    print(f"[rtwx-o0q0r0d] G0={g['g0']} G1={g['g1']} G2={g['g2']} G3={g['g3']} G4={g['g4']} pattern={pattern}", flush=True)
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G1", "G2", "G3", "G4")},
        "B0": {"path": raw.get("path"), "channel": raw.get("channel")},
        "B2": raw.get("semantic"),
        "demos": demos,
        "errors": raw.get("errors"),
        "unlocks_o0q0r1_prereg": bool(qual),
        "unlocks_o0q1_prereg": False,
        "unlocks_o1": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

"""AC3-P0: official native expert replay via _traj_data + play_once. No HDF5 actions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def demo_dir(repo: Path, task: str) -> Path:
    return Path(repo) / "data" / "demo_clean" / task / "aloha_agilex"


def traj_pkl(repo: Path, task: str, ep_index: int) -> Path:
    return demo_dir(repo, task) / "_traj_data" / f"episode{ep_index}.pkl"


def converted_hdf5(repo: Path, task: str, ep_index: int) -> Path:
    return demo_dir(repo, task) / "data" / f"episode_{ep_index:07d}.hdf5"


def raw_hdf5_candidates(repo: Path, task: str, ep_index: int) -> list[Path]:
    return [
        Path(repo) / "data" / task / "demo_clean" / "data" / f"episode{ep_index}.hdf5",
        demo_dir(repo, task) / "data" / f"episode{ep_index}.hdf5",
    ]


def load_traj(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


class DenseControlTracer:
    def __init__(self) -> None:
        self.sim_step = 0
        self.q_des: list[np.ndarray] = []
        self.qd_des: list[np.ndarray] = []
        self.q_actual: list[np.ndarray] = []
        self.qd_actual: list[np.ndarray] = []
        self.gripper_des: list[np.ndarray] = []
        self.saved_frame_idx: list[int] = []
        self._pending_qd = {"left": None, "right": None}

    def install(self, env: Any) -> None:
        robot = env.robot
        orig_set = robot.set_arm_joints
        orig_step = env.scene.step
        orig_pic = env._take_picture
        tracer = self

        def set_arm(target_position, target_velocity, arm_tag):
            tracer._pending_qd[str(arm_tag)] = np.asarray(target_velocity, dtype=np.float64).reshape(-1)
            return orig_set(target_position, target_velocity, arm_tag)

        def step():
            q_des = np.asarray(robot.get_left_arm_jointState() + robot.get_right_arm_jointState(), dtype=np.float64)
            q_act = np.asarray(robot.get_left_arm_real_jointState() + robot.get_right_arm_real_jointState(), dtype=np.float64)
            lv = tracer._pending_qd["left"]
            rv = tracer._pending_qd["right"]
            if lv is None:
                lv = np.zeros(6)
            if rv is None:
                rv = np.zeros(6)
            qd_des = np.concatenate([lv.reshape(-1)[:6], [0.0], rv.reshape(-1)[:6], [0.0]])
            qd_act = np.zeros_like(q_act)
            try:
                lqv = np.asarray(robot.left_entity.get_qvel(), dtype=np.float64)
                rqv = np.asarray(robot.right_entity.get_qvel(), dtype=np.float64)
                qd_act = np.concatenate([lqv[:6], [0.0], rqv[:6], [0.0]])
            except Exception:
                pass
            g = np.array([robot.get_left_gripper_val(), robot.get_right_gripper_val()], dtype=np.float64)
            tracer.q_des.append(q_des)
            tracer.qd_des.append(qd_des)
            tracer.q_actual.append(q_act)
            tracer.qd_actual.append(qd_act)
            tracer.gripper_des.append(g)
            orig_step()
            tracer.sim_step += 1

        def take_picture():
            tracer.saved_frame_idx.append(int(tracer.sim_step))
            return orig_pic()

        robot.set_arm_joints = set_arm
        env.scene.step = step
        env._take_picture = take_picture

    def as_arrays(self) -> dict[str, np.ndarray]:
        n = len(self.q_des)
        return {
            "sim_step": np.arange(n, dtype=np.int64),
            "q_des": np.stack(self.q_des) if n else np.zeros((0, 14)),
            "qd_des": np.stack(self.qd_des) if n else np.zeros((0, 14)),
            "q_actual": np.stack(self.q_actual) if n else np.zeros((0, 14)),
            "qd_actual": np.stack(self.qd_actual) if n else np.zeros((0, 14)),
            "gripper_des": np.stack(self.gripper_des) if n else np.zeros((0, 2)),
            "saved_frame_idx": np.asarray(self.saved_frame_idx, dtype=np.int64),
        }


def save_trace_h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)


def native_replay_once(repo: Path, task: str, ep_index: int, seed: int, *, record: bool, trace_path: Path | None) -> dict[str, Any]:
    from ..task_x0 import _install_mplib_fallback, _load_task_args, _setup_env

    tp = traj_pkl(repo, task, ep_index)
    traj = load_traj(tp)
    if traj is None:
        return {
            "task": task,
            "ep_index": int(ep_index),
            "seed": int(seed),
            "success": False,
            "safety_violation": False,
            "err": "missing_traj_data",
            "traj_path": str(tp),
        }
    args = _load_task_args(repo, task, "demo_clean")
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    args["save_path"] = str(demo_dir(repo, task))
    env, used, err = _setup_env(repo, task, int(seed), 8, args)
    if env is None:
        return {"task": task, "ep_index": ep_index, "seed": seed, "success": False, "safety_violation": True, "err": str(err)}
    tracer = DenseControlTracer()
    try:
        env.set_path_lst({"need_plan": False, "left_joint_path": traj["left_joint_path"], "right_joint_path": traj["right_joint_path"]})
        env.left_cnt = 0
        env.right_cnt = 0
        if record:
            tracer.install(env)
        env.play_once()
        success = bool(env.check_success()) or bool(getattr(env, "eval_success", False))
        out = {
            "task": task,
            "ep_index": int(ep_index),
            "seed": int(used),
            "success": bool(success),
            "plan_success": bool(getattr(env, "plan_success", False)),
            "safety_violation": False,
            "n_dense": int(tracer.sim_step),
            "n_saved_frames": int(len(tracer.saved_frame_idx)),
        }
        if record and trace_path is not None and tracer.sim_step > 0:
            save_trace_h5(trace_path, tracer.as_arrays())
            out["trace_path"] = str(trace_path)
        return out
    except Exception as exc:
        return {
            "task": task,
            "ep_index": int(ep_index),
            "seed": int(used),
            "success": False,
            "safety_violation": True,
            "err": repr(exc),
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass

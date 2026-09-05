"""Pack oracle structured state. Same schema for all AC0 tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

# q(14)+qd(14)+obj_p(3)+obj_q(4)+goal_p(3)+goal_q(4)+eeL p(3)q(4)+eeR p(3)q(4)+cabinet(1)
STATE_DIM = 57
STATE_SCHEMA = "ac0.state.v1.q14_qd14_obj7_goal7_eeL7_eeR7_cab1"
TASK_SCHEMA = "ac0.task.v1.place_empty_cup,put_object_cabinet,stamp_seal"
TASK_NAMES = ("place_empty_cup", "put_object_cabinet", "stamp_seal")
TASK_TO_ID = {n: i for i, n in enumerate(TASK_NAMES)}


def _quat(x: np.ndarray | None) -> np.ndarray:
    if x is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = np.asarray(x, dtype=np.float64).reshape(-1)
    if q.size >= 7:
        q = q[3:7]
    elif q.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(q[:4]))
    return q[:4] / max(n, 1e-8)


def pack_state(frame: dict[str, np.ndarray]) -> np.ndarray:
    q = np.asarray(frame["q"], dtype=np.float64).reshape(14)
    qd = np.asarray(frame["qd"], dtype=np.float64).reshape(14)
    cab = np.asarray(frame.get("cabinet_q", 0.0), dtype=np.float64).reshape(-1)
    cab1 = np.array([float(cab[0]) if cab.size else 0.0], dtype=np.float64)
    return np.concatenate(
        [
            q,
            qd,
            np.asarray(frame["obj_p"], dtype=np.float64).reshape(3),
            _quat(frame.get("obj_q")),
            np.asarray(frame["goal_p"], dtype=np.float64).reshape(3),
            _quat(frame.get("goal_q")),
            np.asarray(frame["ee_left_p"], dtype=np.float64).reshape(3),
            _quat(frame.get("ee_left_q")),
            np.asarray(frame["ee_right_p"], dtype=np.float64).reshape(3),
            _quat(frame.get("ee_right_q")),
            cab1,
        ]
    )


def frame_from_oracle_extract(fr: dict[str, Any], q: np.ndarray, qd: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q": np.asarray(q, dtype=np.float64).reshape(14),
        "qd": np.asarray(qd, dtype=np.float64).reshape(14),
        "obj_p": np.asarray(fr["obj_p"], dtype=np.float64).reshape(3),
        "obj_q": np.asarray(fr.get("obj_q", [1, 0, 0, 0]), dtype=np.float64).reshape(-1)[:4],
        "goal_p": np.asarray(fr["g"], dtype=np.float64).reshape(3),
        "goal_q": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_left_p": np.asarray(fr["left_ee"], dtype=np.float64).reshape(3),
        "ee_left_q": np.array([1.0, 0.0, 0.0, 0.0]),
        "ee_right_p": np.asarray(fr["right_ee"], dtype=np.float64).reshape(3),
        "ee_right_q": np.array([1.0, 0.0, 0.0, 0.0]),
        "grip_l": np.asarray(fr["grip_l"], dtype=np.float64).reshape(1),
        "grip_r": np.asarray(fr["grip_r"], dtype=np.float64).reshape(1),
        "cabinet_q": np.asarray(fr["s"][-2], dtype=np.float64).reshape(1) if np.asarray(fr["s"]).size >= 2 else np.zeros(1),
    }

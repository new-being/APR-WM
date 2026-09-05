"""Causal process features. No t/T, no episode length, no future state."""

from __future__ import annotations

from typing import Any

import numpy as np

PROCESS_NAMES = ("d_ee_obj", "c_grasp", "d_obj_goal", "c_contact", "c_goal")
PROCESS_BINARY = (False, True, False, True, True)
GOAL_TOL_M = 0.05
GRASP_DIST_M = 0.12
GRIP_CLOSED = 0.50
PROCESS_SCHEMA = "ac0.process.v1.d_ee_obj,c_grasp,d_obj_goal,c_contact,c_goal"


class ProcessEncoder:
    def encode(
        self,
        task_id: str,
        current_state: dict[str, np.ndarray],
        previous_action: np.ndarray | None = None,
        *,
        future_state: Any = None,
        episode_length: Any = None,
        normalized_time: Any = None,
    ) -> np.ndarray:
        assert future_state is None
        assert episode_length is None
        assert normalized_time is None
        _ = task_id, previous_action
        obj = np.asarray(current_state["obj_p"], dtype=np.float64).reshape(3)
        goal = np.asarray(current_state["goal_p"], dtype=np.float64).reshape(3)
        ee_l = np.asarray(current_state["ee_left_p"], dtype=np.float64).reshape(3)
        ee_r = np.asarray(current_state["ee_right_p"], dtype=np.float64).reshape(3)
        gl = float(np.asarray(current_state["grip_l"]).reshape(-1)[0])
        gr = float(np.asarray(current_state["grip_r"]).reshape(-1)[0])
        d_l = float(np.linalg.norm(ee_l - obj))
        d_r = float(np.linalg.norm(ee_r - obj))
        d_ee = min(d_l, d_r)
        grip = gr if d_r <= d_l else gl
        c_grasp = 1.0 if (d_ee < GRASP_DIST_M and grip < GRIP_CLOSED) else 0.0
        d_og = float(np.linalg.norm(obj - goal))
        c_contact = c_grasp
        c_goal = 1.0 if d_og < GOAL_TOL_M else 0.0
        return np.array([d_ee, c_grasp, d_og, c_contact, c_goal], dtype=np.float64)

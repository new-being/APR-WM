"""AC3 provenance Q4 and CLI."""

from __future__ import annotations

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.evaluation.ac3_provenance import alignment_errors, frame_gaps, velocity_keys_present
from aprwm_v0.rtwx_ac3 import r0_pattern


def test_cli_ac3():
    assert build_parser().parse_args(["rtwx-ac3"]).command == "rtwx-ac3"


def test_q4_next_index():
    st = np.arange(5, dtype=np.float64).reshape(5, 1)
    act = np.arange(1, 6, dtype=np.float64).reshape(5, 1)
    r = alignment_errors(st, act)
    assert r["inferred"] == "next_index"
    assert r["e_next"] < r["e_same"]


def test_q4_same_index():
    x = np.arange(5, dtype=np.float64).reshape(5, 1)
    r = alignment_errors(x, x)
    assert r["inferred"] == "same_index"


def test_q2_gaps():
    g = frame_gaps(np.array([0, 15, 30, 45]))
    assert abs(g["median_r"] - 15) < 1e-9
    assert g["max_r"] == 15


def test_q3_velocity_absent():
    assert velocity_keys_present(["action/joint_states", "state/left_arm"]) == []
    assert velocity_keys_present(["action/qvel"]) == ["action/qvel"]


def test_r0_pattern_native_fail():
    z = {"pooled": 0.0, "safety": 0.0, "per_task": {"place_empty_cup": 0, "put_object_cabinet": 0, "stamp_seal": 0}, "n_tasks_gt0": 0}
    assert r0_pattern(False, z, z, z, True) == "native_expert_replay_failure"

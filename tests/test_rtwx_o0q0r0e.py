"""RTWX-O0Q0R0E smoke tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0 import G0_MAX, TAU
from aprwm_v0.rtwx_o0q0r0c import H_P0, I_RATIO_P0
from aprwm_v0.rtwx_o0q0r0e import (
    MIN_STEPS,
    PATTERNS,
    SEEDS,
    RTWXO0Q0R0EConfig,
    _pattern,
    audit_joint_path,
    curobo_available,
    run_rtwx_o0q0r0e,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0r0e"]).command == "rtwx-o0q0r0e"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0r0e(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0R0EConfig(smoke=True))


def test_frozen():
    assert SEEDS == (0, 1, 2)
    assert TAU == 0.05
    assert G0_MAX == 0.02
    assert H_P0 == 6
    assert I_RATIO_P0 == 4.0
    assert MIN_STEPS == 400


def test_patterns():
    ok = dict(g0=True, g1=True, g2=True, g3=True)
    assert _pattern(**ok) == "native_dense_trace_qualified"
    assert _pattern(**{**ok, "g0": False}) == "dense_trace_unavailable"
    assert _pattern(**{**ok, "g1": False}) == "dense_trace_semantics_failure"
    assert _pattern(**{**ok, "g2": False}) == "native_trace_replay_failure"
    assert _pattern(**{**ok, "g3": False}) == "native_trace_replay_failure"


def test_audit_joint_path_sparse_vs_dense(tmp_path: Path):
    sparse = tmp_path / "sparse.pkl"
    sparse.write_bytes(
        pickle.dumps(
            {
                "left_joint_path": [{"status": "Success", "position": np.zeros((1, 6))}],
                "right_joint_path": [],
            }
        )
    )
    a = audit_joint_path(sparse)
    assert a["dense"] is False
    dense = tmp_path / "dense.pkl"
    dense.write_bytes(
        pickle.dumps(
            {
                "left_joint_path": [{"status": "Success", "position": np.zeros((500, 6))}],
                "right_joint_path": [{"status": "Success", "position": np.zeros((500, 6))}],
            }
        )
    )
    b = audit_joint_path(dense)
    assert b["dense"] is True
    assert b["kind"] == "take_dense_action_position"


def test_curobo_probe_does_not_crash():
    info = curobo_available()
    assert "ok" in info


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0r0e(tmp_path / "o0q0r0e", config=RTWXO0Q0R0EConfig(smoke=True, backend="numpy"))
    assert r["unlocks_o0q1_prereg"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_new_planner"] is True
    assert r["header"]["no_p0_p1_reopen"] is True
    assert r["pattern"] in PATTERNS
    if r["pattern"] != "native_dense_trace_qualified":
        assert r["unlocks_o0q0r1_prereg"] is False
    else:
        assert r["unlocks_o0q0r1_prereg"] is True

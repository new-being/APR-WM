"""TASK-X0 tests: recoverability, no future in G, r10_c0 refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aprwm_v0.task_x0 import (
    CABINET_PHASES,
    CUP_PHASES,
    STAMP_PHASES,
    TASKX0Config,
    G_frozen_episode,
    encode_p,
    g_is_prefix_recoverable,
    phases_for,
    run_task_x0,
)


def _cabinet_frames(n: int = 80, seed: int = 0) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    origin = 0.74
    target = np.array([0.0, 0.18, 0.82], dtype=np.float64)
    obj = np.array([0.22, -0.15, origin], dtype=np.float64)
    ee_r = obj + np.array([0.12, -0.04, 0.08])
    ee_l = np.array([-0.18, 0.10, 0.90], dtype=np.float64)
    gl, gr = 1.0, 1.0
    frames = []
    for t in range(n):
        u = t / max(n - 1, 1)
        if u < 0.16:
            ee_r = ee_r + 0.35 * (obj - ee_r) / max(n * 0.16, 1)
            gr = 1.0
        elif u < 0.28:
            ee_r = obj + np.array([0.0, 0.0, 0.02])
            gr = max(0.08, gr - 0.2)
        elif u < 0.48:
            obj = obj + np.array([0.0, 0.0, 0.005])
            ee_r = obj + np.array([0.0, 0.0, 0.02])
            gr = 0.08
        elif u < 0.70:
            obj = obj + 0.09 * (target - obj)
            ee_r = obj + np.array([0.0, 0.0, 0.02])
            gr = 0.08
        elif u < 0.86:
            obj = 0.65 * obj + 0.35 * target
            ee_r = obj + np.array([0.0, 0.0, 0.015])
            gr = 0.08
        else:
            obj = 0.35 * obj + 0.65 * target
            gr = min(1.0, gr + 0.25)
            ee_r = obj + np.array([0.0, 0.06, 0.05])
        ql = np.array([0.0, 0.002 * t, 0.0, 0.0, 0.0, 0.0, gl], dtype=np.float64)
        qr = np.array([0.01 * t, 0.0, 0.0, 0.0, 0.0, 0.0, gr], dtype=np.float64)
        extra = np.array([0.08 * min(u, 1.0), 0.0], dtype=np.float64)
        obj_n = obj + 0.001 * rng.normal(size=3)
        s = np.concatenate(
            [ql, qr, obj_n, np.array([1.0, 0.0, 0.0, 0.0]), ee_l, ee_r, extra]
        )
        frames.append(
            {
                "s": s,
                "a": np.concatenate([ql, qr]),
                "g": target.copy(),
                "obj_p": obj_n.copy(),
                "left_ee": ee_l.copy(),
                "right_ee": ee_r.copy(),
                "grip_l": np.array([gl], dtype=np.float64),
                "grip_r": np.array([gr], dtype=np.float64),
            }
        )
    return frames


def test_r10_c0_path_refused(tmp_path: Path):
    raised = False
    try:
        run_task_x0(tmp_path / "runs" / "r10_c0" / "x", config=TASKX0Config(n_demo=0, seed_attempts=0))
    except RuntimeError as exc:
        raised = "r10_c0" in str(exc)
    assert raised


def test_phase_graphs_exist_for_all_three_tasks():
    assert phases_for("put_object_cabinet") == CABINET_PHASES
    assert phases_for("place_empty_cup") == CUP_PHASES
    assert phases_for("stamp_seal") == STAMP_PHASES
    p = encode_p("insert", np.ones(5), "put_object_cabinet")
    assert p.shape[0] == len(CABINET_PHASES) + 5


def test_g_recoverable_from_prefix_only():
    frames = _cabinet_frames(90, seed=3)
    ks, cs = G_frozen_episode(frames, "put_object_cabinet")
    assert g_is_prefix_recoverable(frames, "put_object_cabinet")
    assert "approach" in ks
    assert "grasp" in set(ks)
    for t in range(len(frames)):
        ks_p, cs_p = G_frozen_episode(frames[: t + 1], "put_object_cabinet")
        assert ks_p[-1] == ks[t]
        assert np.allclose(cs_p[-1], cs[t])


def test_g_does_not_use_future_frames():
    frames = _cabinet_frames(60, seed=1)
    assert g_is_prefix_recoverable(frames, "put_object_cabinet")
    causal = G_frozen_episode(frames[:12], "put_object_cabinet")
    full = G_frozen_episode(frames, "put_object_cabinet")
    assert causal[0] == full[0][:12]
    assert np.allclose(causal[1], full[1][:12])


def test_cup_and_stamp_graphs_are_causal():
    frames = _cabinet_frames(40, seed=2)
    assert g_is_prefix_recoverable(frames, "place_empty_cup")
    assert g_is_prefix_recoverable(frames, "stamp_seal")
    k_c, _ = G_frozen_episode(frames, "place_empty_cup")
    k_s, _ = G_frozen_episode(frames, "stamp_seal")
    assert len(k_c) == len(frames)
    assert len(k_s) == len(frames)


def test_no_physics_predict_in_task_x0_runner():
    src = Path("aprwm_v0/task_x0.py").read_text(encoding="utf-8")
    assert "from .rtwx_x0 import" not in src
    assert "physics_predict_used" in src
    assert "diffusion_train" in src


def _cup_success_frames() -> list[dict[str, np.ndarray]]:
    g = np.array([0.0, 0.0, 0.74], dtype=np.float64)
    cup0 = np.array([0.22, -0.05, 0.74], dtype=np.float64)
    frames: list[dict[str, np.ndarray]] = []

    def add(obj, left, right, gl, gr):
        ql = np.zeros(7, dtype=np.float64)
        qr = np.zeros(7, dtype=np.float64)
        ql[-1], qr[-1] = gl, gr
        obj = np.asarray(obj, dtype=np.float64)
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        s = np.concatenate([ql, qr, obj, np.array([1.0, 0, 0, 0]), left, right, np.zeros(2)])
        frames.append(
            {
                "s": s,
                "a": np.concatenate([ql, qr]),
                "g": g.copy(),
                "obj_p": obj.copy(),
                "left_ee": left.copy(),
                "right_ee": right.copy(),
                "grip_l": np.array([gl], dtype=np.float64),
                "grip_r": np.array([gr], dtype=np.float64),
            }
        )

    for i in range(8):
        add(cup0, [-0.3, 0, 0.9], cup0 + np.array([0.25 - 0.02 * i, 0.0, 0.08]), 1.0, 1.0)
    for i in range(8):
        add(cup0, [-0.3, 0, 0.9], cup0 + np.array([0.02, 0, 0.02]), 1.0, max(0.1, 0.9 - 0.1 * i))
    for i in range(8):
        obj = cup0 + np.array([0.0, 0.0, 0.02 * (i + 1)])
        add(obj, [-0.3, 0, 0.9], obj + np.array([0.02, 0, 0.02]), 1.0, 0.1)
    for i in range(10):
        obj = cup0 + np.array([-0.02 * (i + 1), 0.0, 0.10])
        add(obj, [-0.3, 0, 0.9], obj + np.array([0.02, 0, 0.02]), 1.0, 0.1)
    for i in range(8):
        obj = np.array([0.01, 0.0, 0.74 + 0.04 - 0.005 * i])
        add(obj, [-0.3, 0, 0.9], obj + np.array([0.02, 0, 0.02]), 1.0, 0.15)
    for i in range(8):
        obj = np.array([0.005, 0.0, 0.74])
        add(obj, [-0.3, 0, 0.9], obj + np.array([0.15, 0, 0.08]), 1.0, min(1.0, 0.2 + 0.1 * i))
    return frames


def test_cup_g_recoverable_and_no_future_leak():
    frames = _cup_success_frames()
    assert g_is_prefix_recoverable(frames, "place_empty_cup")
    ks, _ = G_frozen_episode(frames, "place_empty_cup")
    for name in ("approach", "grasp", "lift", "transport"):
        assert name in ks
    assert ks[-1] in {"place", "release", "done"}
    early = G_frozen_episode(frames[:10], "place_empty_cup")[0]
    assert early[-1] not in {"done"}
    full_early = G_frozen_episode(frames, "place_empty_cup")[0][:10]
    assert early == full_early


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_phase_graphs_exist_for_all_three_tasks()
    test_g_recoverable_from_prefix_only()
    test_g_does_not_use_future_frames()
    test_cup_and_stamp_graphs_are_causal()
    test_no_physics_predict_in_task_x0_runner()
    test_cup_g_recoverable_and_no_future_leak()
    with TemporaryDirectory() as d:
        test_r10_c0_path_refused(Path(d))
    print("ALL_TASK_X0_UNIT_OK")

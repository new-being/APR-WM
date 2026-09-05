"""AC3-D0 provenance: native dense vs raw vs AC0 converted HDF5. No new actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .ac3_native import converted_hdf5, raw_hdf5_candidates


def _hdf5_leaf_names(path: Path) -> list[str]:
    import h5py

    names: list[str] = []
    if not path.is_file():
        return names
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: names.append(n.lower()) if hasattr(o, "shape") else None)
    return names


def velocity_keys_present(names: list[str]) -> list[str]:
    hits = []
    for n in names:
        if any(tok in n for tok in ("qvel", "qd_des", "velocity", "qd/", "/qd")):
            hits.append(n)
    return hits


def load_converted_q14(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    import h5py

    if not path.is_file():
        return None, None
    with h5py.File(path, "r") as f:
        st = f["state"] if "state" in f else f
        act = f["action"] if "action" in f else f

        def pack(g):
            if "joint_states" in g:
                return np.asarray(g["joint_states"], dtype=np.float64)
            la, ra = np.asarray(g["left_arm_joint_states"]), np.asarray(g["right_arm_joint_states"])
            lg = np.asarray(g["left_ee_joint_states"]).reshape(len(la), -1)[:, 0]
            rg = np.asarray(g["right_ee_joint_states"]).reshape(len(ra), -1)[:, 0]
            return np.concatenate([la, lg[:, None], ra, rg[:, None]], axis=1)

        try:
            return pack(st), pack(act)
        except Exception:
            return None, None


def load_raw_joint_vector(path: Path) -> np.ndarray | None:
    import h5py

    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        if "joint_action" not in f:
            return None
        j = f["joint_action"]
        if "vector" in j:
            return np.asarray(j["vector"], dtype=np.float64)
        la = np.asarray(j["left_arm"])
        ra = np.asarray(j["right_arm"])
        lg = np.asarray(j["left_gripper"]).reshape(len(la), -1)[:, 0]
        rg = np.asarray(j["right_gripper"]).reshape(len(ra), -1)[:, 0]
        return np.concatenate([la, lg[:, None], ra, rg[:, None]], axis=1)


def alignment_errors(state: np.ndarray, action: np.ndarray) -> dict[str, float]:
    n = min(len(state), len(action))
    if n < 2:
        return {"e_same": float("nan"), "e_next": float("nan"), "n": n}
    e0 = float(np.abs(action[:n] - state[:n]).mean())
    e1 = float(np.abs(action[: n - 1] - state[1:n]).mean())
    return {"e_same": e0, "e_next": e1, "n": n, "inferred": "next_index" if e1 < e0 else "same_index"}


def frame_gaps(saved_frame_idx: np.ndarray) -> dict[str, float]:
    idx = np.asarray(saved_frame_idx, dtype=np.int64)
    if idx.size < 2:
        return {"median_r": float("nan"), "p90_r": float("nan"), "max_r": float("nan"), "n": int(idx.size)}
    r = np.diff(idx)
    r = r[r > 0]
    if r.size == 0:
        return {"median_r": float("nan"), "p90_r": float("nan"), "max_r": float("nan"), "n": int(idx.size)}
    return {
        "median_r": float(np.median(r)),
        "p90_r": float(np.percentile(r, 90)),
        "max_r": float(np.max(r)),
        "mean_r": float(np.mean(r)),
        "n_gaps": int(r.size),
    }


def q1_joint_vs_qdes(joint: np.ndarray | None, q_des: np.ndarray | None, saved_idx: np.ndarray | None) -> dict[str, Any]:
    if joint is None or q_des is None or saved_idx is None or saved_idx.size == 0:
        return {"ok": False, "reason": "missing_arrays"}
    idx = np.clip(saved_idx.astype(int), 0, len(q_des) - 1)
    m = min(len(joint), len(idx))
    if m == 0:
        return {"ok": False, "reason": "empty"}
    err = float(np.abs(joint[:m] - q_des[idx[:m]]).mean())
    return {"ok": err < 1e-3, "mae": err, "n": m}


def audit_episode(repo: Path, task: str, ep_index: int, ac0_a: np.ndarray | None, trace: dict[str, np.ndarray] | None) -> dict[str, Any]:
    conv = converted_hdf5(repo, task, ep_index)
    st, act = load_converted_q14(conv)
    raw = None
    raw_path = None
    for c in raw_hdf5_candidates(repo, task, ep_index):
        raw = load_raw_joint_vector(c)
        if raw is not None:
            raw_path = str(c)
            break
    names = _hdf5_leaf_names(conv)
    vel = velocity_keys_present(names)
    q4 = alignment_errors(st, act) if st is not None and act is not None else {"inferred": "missing"}
    q4_ac0 = None
    if ac0_a is not None and act is not None:
        m = min(len(ac0_a), len(act), len(st) if st is not None else 10**9)
        q4_ac0 = {
            "e0_vs_action_t": float(np.abs(ac0_a[:m] - act[:m]).mean()),
            "e1_vs_action_t1": float(np.abs(ac0_a[: m - 1] - act[1:m]).mean()) if m > 1 else float("nan"),
        }
    q2 = frame_gaps(trace["saved_frame_idx"]) if trace is not None else {"reason": "no_native_trace"}
    q1 = q1_joint_vs_qdes(raw if raw is not None else st, trace["q_des"] if trace else None, trace["saved_frame_idx"] if trace else None)
    return {
        "converted_hdf5": str(conv),
        "converted_exists": conv.is_file(),
        "raw_hdf5": raw_path,
        "Q1": q1,
        "Q2": q2,
        "Q3": {"velocity_keys": vel, "velocity_absent": len(vel) == 0},
        "Q4_converted_state_vs_action": q4,
        "Q4_ac0_vs_converted_action": q4_ac0,
        "n_state": int(len(st) if st is not None else 0),
        "n_action": int(len(act) if act is not None else 0),
    }

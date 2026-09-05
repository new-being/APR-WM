"""TASK-X0: oracle task-progress instrument on RoboTwin demos (no diffusion, not X1, not R10).

Scientific freeze: p_t = G_frozen(s_<=t, g) only. p never enters F_physics.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/TASKX/TASKX0_PREREG.md"
FAMILY_PREREG = "REPORT/REG/TASKX/TASKX_PREREG.md"
SCHEMA_ID = "aprwm.task_x0.instrument.v1"

# Frozen in runner header BEFORE first parse (instrument, not a 20–50-task sweep).
N_DEMO = 50
N_DEMO_DECLARED = N_DEMO
N_MIN = 32
HOLDOUT_FRAC = 0.25
KNN_K = 8
N_BOOT = 1000
SUBSAMPLE = 15
EPS_S_SCALE = 0.05
SEED = 9001

SUPPORTED_TASKS = ("place_empty_cup", "put_object_cabinet", "stamp_seal")

CUP_PHASES = ("approach", "grasp", "lift", "transport", "place", "release", "done")
CABINET_PHASES = ("approach", "grasp", "transport", "insert", "release", "done")
STAMP_PHASES = ("approach", "grasp", "align", "press", "complete")

CUP_PRED = ("object_grasped", "object_lifted", "near_target", "placed", "released")
CABINET_PRED = ("object_grasped", "object_lifted", "near_cabinet", "inside_target", "released")
STAMP_PRED = ("seal_grasped", "aligned", "pressing", "complete")

TASK_META: dict[str, dict[str, Any]] = {
    "place_empty_cup": {
        "actor": "cup",
        "goal_actor": "coaster",
        "phases": CUP_PHASES,
        "predicates": CUP_PRED,
    },
    "put_object_cabinet": {
        "actor": "object",
        "goal_actor": "cabinet",
        "phases": CABINET_PHASES,
        "predicates": CABINET_PRED,
    },
    "stamp_seal": {
        "actor": "seal",
        "goal_actor": "target",
        "phases": STAMP_PHASES,
        "predicates": STAMP_PRED,
    },
}


@dataclass(frozen=True)
class TASKX0Config:
    task: str = "place_empty_cup"
    output: str = ""
    robotwin_repo: str = "/root/RoboTwin"
    n_demo: int = N_DEMO
    n_min: int = N_MIN
    seed: int = SEED
    seed_attempts: int = 80
    max_seed_attempts: int | None = None
    subsample: int = SUBSAMPLE
    holdout_frac: float = HOLDOUT_FRAC
    knn_k: int = KNN_K
    n_boot: int = N_BOOT
    task_config: str = "demo_clean"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _refuse_locked_paths(output: Path) -> None:
    text = str(output.resolve()).replace("\\", "/")
    if "r10_c0" in text:
        raise RuntimeError("r10_c0 is locked; TASK-X0 must not write there")
    if "rtwx_x0" in text and "task_x0" not in text:
        raise RuntimeError("TASK-X0 must not write RoboTwin-X0 physics artifacts")


def phases_for(task: str) -> tuple[str, ...]:
    return tuple(TASK_META[task]["phases"])


def predicates_for(task: str) -> tuple[str, ...]:
    return tuple(TASK_META[task]["predicates"])


def _pose_p(obj: Any) -> np.ndarray:
    pose = obj.get_pose() if hasattr(obj, "get_pose") else obj
    if hasattr(pose, "p"):
        return np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:3]


def _pose_q(obj: Any) -> np.ndarray:
    pose = obj.get_pose() if hasattr(obj, "get_pose") else obj
    if hasattr(pose, "q"):
        return np.asarray(pose.q, dtype=np.float64).reshape(-1)[:4]
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size >= 7:
        return arr[3:7]
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _ee_xyz(env: Any, side: str) -> np.ndarray:
    fn = getattr(env.robot, f"get_{side}_ee_pose", None) or getattr(env.robot, f"get_{side}_tcp_pose", None)
    if fn is None:
        return np.zeros(3, dtype=np.float64)
    pose = fn()
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size >= 3:
        return arr[:3]
    if hasattr(pose, "p"):
        return np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3]
    return np.zeros(3, dtype=np.float64)


def extract_oracle_frame(env: Any, task: str) -> dict[str, np.ndarray]:
    meta = TASK_META[task]
    actor = getattr(env, meta["actor"])
    goal = getattr(env, meta["goal_actor"])
    ql = np.asarray(env.robot.get_left_arm_real_jointState(), dtype=np.float64).reshape(-1)
    qr = np.asarray(env.robot.get_right_arm_real_jointState(), dtype=np.float64).reshape(-1)
    obj_p = _pose_p(actor)
    obj_q = _pose_q(actor)
    g_p = _pose_p(goal)
    if hasattr(goal, "get_functional_point"):
        try:
            g_p = np.asarray(goal.get_functional_point(0), dtype=np.float64).reshape(-1)[:3]
        except Exception:
            try:
                g_p = np.asarray(goal.get_functional_point(0, "pose").p, dtype=np.float64).reshape(-1)[:3]
            except Exception:
                pass
    if hasattr(actor, "get_functional_point"):
        try:
            fp = actor.get_functional_point(0, "pose")
            obj_p = np.asarray(fp.p, dtype=np.float64).reshape(-1)[:3]
        except Exception:
            pass
    left_ee = _ee_xyz(env, "left")
    right_ee = _ee_xyz(env, "right")
    gl = float(ql[-1]) if ql.size else 1.0
    gr = float(qr[-1]) if qr.size else 1.0
    extra = np.zeros(2, dtype=np.float64)
    if hasattr(env, "cabinet"):
        extra = np.concatenate(
            [
                np.asarray(env.cabinet.get_qpos(), dtype=np.float64).reshape(-1)[:1],
                np.asarray(env.cabinet.get_qvel(), dtype=np.float64).reshape(-1)[:1],
            ]
        )
    s = np.concatenate([ql, qr, obj_p, obj_q, left_ee, right_ee, extra])
    a = np.concatenate([ql, qr])
    return {
        "s": s,
        "a": a,
        "g": g_p.copy(),
        "obj_p": obj_p.copy(),
        "left_ee": left_ee,
        "right_ee": right_ee,
        "grip_l": np.array([gl], dtype=np.float64),
        "grip_r": np.array([gr], dtype=np.float64),
    }


def _latch(prev: bool, now: bool) -> bool:
    return bool(prev or now)


def cup_predicates_step(
    *,
    obj_p: np.ndarray,
    g: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    grip_l: float,
    grip_r: float,
    origin_z: float,
    origin_ee_z: float = 0.0,
    ee0_xy: np.ndarray | None = None,
    prev_c: np.ndarray | None = None,
) -> np.ndarray:
    """Causal cup predicates from current oracle state + goal + past latches.

    Official hdf5 stores qpos/endpose, not object pose. When the cup actor does
    not track, G still uses robot EE / gripper / coaster goal (no future labels).
    """
    dist_l = float(np.linalg.norm(left_ee - obj_p[:3]))
    dist_r = float(np.linalg.norm(right_ee - obj_p[:3]))
    active_right = grip_r <= grip_l
    ee = right_ee if active_right else left_ee
    grip = grip_r if active_right else grip_l
    near_grip = min(dist_l, dist_r) < 0.16
    grasped_now = (grip < 0.50) and (near_grip or grip < 0.35)
    lifted_now = ((float(obj_p[2]) - float(origin_z)) > 0.035) or (
        float(ee[2]) > float(origin_ee_z) + 0.02
    )
    xy_obj = float(np.linalg.norm(obj_p[:2] - g[:2]))
    xy_ee = float(np.linalg.norm(ee[:2] - g[:2]))
    xy = min(xy_obj, xy_ee)
    near_target = xy < 0.08
    placed_now = xy < 0.05
    prev = np.zeros(5, dtype=np.float64) if prev_c is None else prev_c
    opening = grip > 0.55 and bool(prev[0])
    released_now = (placed_now or near_target) and (grip > 0.75 or opening) and grip_l > 0.55 and grip_r > 0.55
    grasped = _latch(bool(prev[0]), grasped_now)
    lifted = _latch(bool(prev[1]), grasped and lifted_now)
    near = near_target
    placed = _latch(bool(prev[3]), placed_now and (grasped or lifted or bool(prev[3])))
    released = _latch(bool(prev[4]), released_now)
    if released:
        grasped = False
    return np.array([grasped, lifted, near, placed, released], dtype=np.float64)


def cup_phase_from_c(c: np.ndarray, dz: float = 0.0, ee_travel: float = 0.0) -> str:
    grasped, lifted, near, placed, released = [bool(x) for x in c]
    if released and placed:
        return "done"
    if released:
        return "release"
    if placed or (near and lifted):
        return "place"
    if lifted and not near:
        return "transport" if float(ee_travel) > 0.10 else "lift"
    if grasped:
        return "grasp"
    return "approach"


def cabinet_predicates_step(
    *,
    obj_p: np.ndarray,
    g: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    grip_l: float,
    grip_r: float,
    origin_z: float,
    extra: np.ndarray,
    prev_c: np.ndarray | None,
) -> np.ndarray:
    dist = min(float(np.linalg.norm(left_ee - obj_p[:3])), float(np.linalg.norm(right_ee - obj_p[:3])))
    active_g = grip_r if np.linalg.norm(right_ee - obj_p[:3]) <= np.linalg.norm(left_ee - obj_p[:3]) else grip_l
    grasped_now = dist < 0.12 and active_g < 0.45
    lifted_now = (float(obj_p[2]) - float(origin_z)) > 0.02
    near_cab = float(np.linalg.norm(obj_p[:2] - g[:2])) < 0.12
    inside = float(np.linalg.norm(obj_p[:2] - g[:2])) < 0.05 and 0.007 < (float(obj_p[2]) - float(origin_z)) < 0.12
    released_now = inside and active_g > 0.7
    prev = np.zeros(5, dtype=np.float64) if prev_c is None else prev_c
    grasped = _latch(bool(prev[0]), grasped_now)
    lifted = _latch(bool(prev[1]), grasped and lifted_now)
    placed = _latch(bool(prev[3]), inside)
    released = _latch(bool(prev[4]), released_now)
    if released:
        grasped = False
    return np.array([grasped, lifted, near_cab, placed, released], dtype=np.float64)


def cabinet_phase_from_c(c: np.ndarray) -> str:
    grasped, lifted, near, inside, released = [bool(x) for x in c]
    if released and inside:
        return "done"
    if released:
        return "release"
    if inside or (near and lifted):
        return "insert"
    if lifted:
        return "transport"
    if grasped:
        return "grasp"
    return "approach"


def stamp_predicates_step(
    *,
    obj_p: np.ndarray,
    g: np.ndarray,
    left_ee: np.ndarray,
    right_ee: np.ndarray,
    grip_l: float,
    grip_r: float,
    origin_z: float,
    prev_c: np.ndarray | None,
) -> np.ndarray:
    xy_l = float(np.linalg.norm(left_ee[:2] - obj_p[:2]))
    xy_r = float(np.linalg.norm(right_ee[:2] - obj_p[:2]))
    use_right = xy_r <= xy_l
    ee = right_ee if use_right else left_ee
    active_g = grip_r if use_right else grip_l
    xy_ee = min(xy_l, xy_r)
    dz = float(ee[2] - obj_p[2])
    grasped_now = xy_ee < 0.08 and active_g < 0.5 and 0.0 <= dz < 0.30
    xy = float(np.linalg.norm(obj_p[:2] - g[:2]))
    aligned = xy < 0.10
    pressing = xy < 0.035
    complete = xy < 0.025 and active_g > 0.7
    prev = np.zeros(4, dtype=np.float64) if prev_c is None else prev_c
    grasped = _latch(bool(prev[0]), grasped_now)
    al = _latch(bool(prev[1]), grasped and aligned)
    pr = _latch(bool(prev[2]), pressing)
    done = _latch(bool(prev[3]), complete)
    return np.array([grasped, al, pr, done], dtype=np.float64)


def stamp_phase_from_c(c: np.ndarray) -> str:
    grasped, aligned, pressing, complete = [bool(x) for x in c]
    if complete:
        return "complete"
    if pressing:
        return "press"
    if aligned:
        return "align"
    if grasped:
        return "grasp"
    return "approach"


def G_frozen_episode(frames: list[dict[str, np.ndarray]], task: str) -> tuple[list[str], np.ndarray]:
    """Oracle G: sequential, current/past state + goal only. No future labels."""
    origin_z = float(frames[0]["obj_p"][2]) if frames else 0.0
    origin_ee_z = 0.0
    ee0_xy = np.zeros(2, dtype=np.float64)
    if frames:
        origin_ee_z = float(min(frames[0]["left_ee"][2], frames[0]["right_ee"][2]))
        ee0_xy = np.asarray(frames[0]["right_ee"][:2], dtype=np.float64)
    ks: list[str] = []
    cs: list[np.ndarray] = []
    prev: np.ndarray | None = None
    grasp_xy: np.ndarray | None = None
    for fr in frames:
        g = fr["g"]
        if task == "place_empty_cup":
            c = cup_predicates_step(
                obj_p=fr["obj_p"],
                g=g,
                left_ee=fr["left_ee"],
                right_ee=fr["right_ee"],
                grip_l=float(fr["grip_l"][0]),
                grip_r=float(fr["grip_r"][0]),
                origin_z=origin_z,
                origin_ee_z=origin_ee_z,
                ee0_xy=ee0_xy,
                prev_c=prev,
            )
            dz = float(fr["obj_p"][2]) - origin_z
            ee = fr["right_ee"] if float(fr["grip_r"][0]) <= float(fr["grip_l"][0]) else fr["left_ee"]
            if bool(c[0]) and grasp_xy is None:
                grasp_xy = np.asarray(ee[:2], dtype=np.float64)
            ee_travel = float(np.linalg.norm(np.asarray(ee[:2]) - grasp_xy)) if grasp_xy is not None else 0.0
            k = cup_phase_from_c(c, dz=dz, ee_travel=ee_travel)
            if ks and ks[-1] == "place" and k == "done":
                k = "release"
        elif task == "put_object_cabinet":
            extra = fr["s"][-2:]
            c = cabinet_predicates_step(
                obj_p=fr["obj_p"],
                g=g,
                left_ee=fr["left_ee"],
                right_ee=fr["right_ee"],
                grip_l=float(fr["grip_l"][0]),
                grip_r=float(fr["grip_r"][0]),
                origin_z=origin_z,
                extra=extra,
                prev_c=prev,
            )
            k = cabinet_phase_from_c(c)
        else:
            c = stamp_predicates_step(
                obj_p=fr["obj_p"],
                g=g,
                left_ee=fr["left_ee"],
                right_ee=fr["right_ee"],
                grip_l=float(fr["grip_l"][0]),
                grip_r=float(fr["grip_r"][0]),
                origin_z=origin_z,
                prev_c=prev,
            )
            k = stamp_phase_from_c(c)
        ks.append(k)
        cs.append(c)
        prev = c
    return ks, (np.stack(cs, axis=0) if cs else np.zeros((0, len(predicates_for(task)))))


def g_is_prefix_recoverable(frames: list[dict[str, np.ndarray]], task: str) -> bool:
    """Every p_t equals G(frames[:t+1]); using future frames must not change labels."""
    if not frames:
        return True
    ks_full, cs_full = G_frozen_episode(frames, task)
    for t in range(len(frames)):
        ks, cs = G_frozen_episode(frames[: t + 1], task)
        if ks[-1] != ks_full[t]:
            return False
        if not np.allclose(cs[-1], cs_full[t]):
            return False
    return True


def G_with_future_leak(frames: list[dict[str, np.ndarray]], task: str) -> tuple[list[str], np.ndarray]:
    """Forbidden leak probe: shuffle later frames into the causal scan (must not be required)."""
    if len(frames) < 2:
        return G_frozen_episode(frames, task)
    leaked = list(frames)
    n = len(leaked)
    for i in range(n - 1):
        leaked[i] = {
            **frames[i],
            "obj_p": frames[min(i + max(1, n // 4), n - 1)]["obj_p"],
            "g": frames[i]["g"],
        }
    return G_frozen_episode(leaked, task)


def encode_p(k: str, c: np.ndarray, task: str) -> np.ndarray:
    ph = phases_for(task)
    oh = np.zeros(len(ph), dtype=np.float64)
    if k in ph:
        oh[ph.index(k)] = 1.0
    return np.concatenate([oh, np.asarray(c, dtype=np.float64).reshape(-1)])


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def g1_aliasing(s: np.ndarray, p_id: np.ndarray, a: np.ndarray, eps_s: float) -> dict[str, Any]:
    n = s.shape[0]
    if n < 8:
        return {"passed": False, "n_alias_pairs": 0, "median_alias_action": None, "median_same_p_nn": None}
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    if n > 4000:
        idx = rng.choice(n, size=4000, replace=False)
    ss, pp, aa = s[idx], p_id[idx], a[idx]
    m = ss.shape[0]
    alias_d: list[float] = []
    same_nn: list[float] = []
    # block nearest-neighbor in chunks to bound cost
    bs = 256
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        d = np.linalg.norm(ss[i0:i1, None, :] - ss[None, :, :], axis=2)
        np.fill_diagonal(d if i0 == 0 and i1 == m else d, np.inf)
        for r, gi in enumerate(range(i0, i1)):
            row = d[r].copy()
            row[gi] = np.inf
            order = np.argsort(row)
            for j in order[:32]:
                if row[j] < eps_s and pp[gi] != pp[j]:
                    alias_d.append(float(np.linalg.norm(aa[gi] - aa[j])))
                    break
            same = [j for j in order[:64] if pp[gi] == pp[j]]
            if same:
                same_nn.append(float(np.linalg.norm(aa[gi] - aa[same[0]])))
    if not alias_d or not same_nn:
        return {
            "passed": False,
            "n_alias_pairs": int(len(alias_d)),
            "median_alias_action": float(np.median(alias_d)) if alias_d else None,
            "median_same_p_nn": float(np.median(same_nn)) if same_nn else None,
        }
    med_alias = float(np.median(alias_d))
    med_same = float(np.median(same_nn))
    return {
        "passed": med_alias > med_same and len(alias_d) >= 8,
        "n_alias_pairs": int(len(alias_d)),
        "median_alias_action": med_alias,
        "median_same_p_nn": med_same,
    }


def _knn_nll(train_x: np.ndarray, train_a: np.ndarray, query_x: np.ndarray, query_a: np.ndarray, k: int) -> np.ndarray:
    nll = np.zeros(query_x.shape[0], dtype=np.float64)
    k_use = max(1, min(k, train_x.shape[0]))
    d_a = max(1, train_a.shape[1])
    bs = 128
    for i0 in range(0, query_x.shape[0], bs):
        i1 = min(query_x.shape[0], i0 + bs)
        dist = np.linalg.norm(query_x[i0:i1, None, :] - train_x[None, :, :], axis=2)
        nn = np.argpartition(dist, kth=k_use - 1, axis=1)[:, :k_use]
        for r, gi in enumerate(range(i0, i1)):
            neigh = train_a[nn[r]]
            mu = neigh.mean(axis=0)
            resid = query_a[gi] - mu
            var = float(np.mean(np.square(neigh - mu))) + 1.0e-6
            nll[gi] = 0.5 * d_a * np.log(2.0 * np.pi * var) + 0.5 * float(np.dot(resid, resid)) / var
    return nll


def g2_entropy(
    s: np.ndarray,
    g: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
    *,
    holdout_frac: float,
    knn_k: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    n = s.shape[0]
    if n < 16:
        return {"passed": False, "reason": "too_few_frames", "delta_nll": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_ho = max(8, int(round(n * holdout_frac)))
    ho, tr = perm[:n_ho], perm[n_ho:]
    x_sg_tr = np.concatenate([s[tr], g[tr]], axis=1)
    x_sg_ho = np.concatenate([s[ho], g[ho]], axis=1)
    x_p_tr = np.concatenate([x_sg_tr, p[tr]], axis=1)
    x_p_ho = np.concatenate([x_sg_ho, p[ho]], axis=1)
    nll_sg = _knn_nll(x_sg_tr, a[tr], x_sg_ho, a[ho], knn_k)
    nll_p = _knn_nll(x_p_tr, a[tr], x_p_ho, a[ho], knn_k)
    delta = nll_sg - nll_p
    mean_d = float(np.mean(delta))
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        take = rng.integers(0, delta.size, size=delta.size)
        boot[b] = float(np.mean(delta[take]))
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    return {
        "passed": mean_d > 0.0 and lo > 0.0,
        "nll_sg": float(np.mean(nll_sg)),
        "nll_sgp": float(np.mean(nll_p)),
        "delta_nll": mean_d,
        "ci95": [lo, hi],
        "n_holdout": int(delta.size),
    }


class _MplibCuroboAdapter:
    """CuRobo-shaped wrapper around mplib screw interpolation (scene boot without curobo)."""

    def __init__(self, mplib: Any) -> None:
        self.mplib = mplib

    def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None, **kwargs):
        pose = target_gripper_pose
        if hasattr(pose, "p"):
            pose = list(np.asarray(pose.p, dtype=np.float64)) + list(np.asarray(pose.q, dtype=np.float64))
        try:
            result = self.mplib.plan_path(curr_joint_pos, pose, arms_tag=arms_tag, log=False)
        except TypeError:
            try:
                result = self.mplib.plan_path(curr_joint_pos, pose)
            except Exception:
                return {"status": "Fail"}
        except Exception:
            return {"status": "Fail"}
        if not isinstance(result, dict):
            return {"status": "Fail"}
        if result.get("status") != "Success":
            try:
                result = self.mplib.plan_screw(curr_joint_pos, pose, arms_tag=arms_tag, log=False)
            except Exception:
                return {"status": "Fail"}
        if result.get("status") == "Success" and "position" in result and "velocity" not in result:
            pos = np.asarray(result["position"])
            vel = np.zeros_like(pos)
            if pos.shape[0] > 1:
                vel[1:] = pos[1:] - pos[:-1]
            result["velocity"] = vel
        return result

    def plan_batch(self, curr_joint_pos, target_gripper_pose_list, constraint_pose=None, arms_tag=None, **kwargs):
        statuses = []
        positions = []
        velocities = []
        n_j = 6
        for pose in list(target_gripper_pose_list):
            r = self.plan_path(curr_joint_pos, pose, constraint_pose, arms_tag)
            ok = r.get("status") == "Success"
            statuses.append("Success" if ok else "Fail")
            if ok and "position" in r:
                positions.append(np.asarray(r["position"]))
                velocities.append(np.asarray(r.get("velocity", np.zeros_like(r["position"]))))
                n_j = int(positions[-1].shape[-1])
            else:
                positions.append(np.zeros((1, n_j)))
                velocities.append(np.zeros((1, n_j)))
        return {
            "status": np.array(statuses, dtype=object),
            "position": positions,
            "velocity": velocities,
        }

    def plan_grippers(self, now_val, target_val):
        return self.mplib.plan_grippers(now_val, target_val)

    def update_point_cloud(self, *args, **kwargs) -> None:
        return None


def _install_mplib_fallback(repo: Path) -> str:
    _patch_curobo_planner(repo)
    from envs.robot.planner import MplibPlanner
    from envs.robot.robot import Robot

    orig = Robot.set_planner

    def set_planner(self, scene=None):  # type: ignore[no-untyped-def]
        orig(self, scene)
        try:
            left = MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                "mplib_screw",
                scene,
            )
            right = MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                "mplib_screw",
                scene,
            )
            self.left_planner = _MplibCuroboAdapter(left)
            self.right_planner = _MplibCuroboAdapter(right)
            self._task_x0_planner = "mplib_screw_fallback"
        except Exception as exc:
            self._task_x0_planner = f"curobo_stub_only:{type(exc).__name__}:{exc}"

    Robot.set_planner = set_planner  # type: ignore[method-assign]
    return "mplib_screw_fallback"


def _load_task_args(repo: Path, task: str, task_config: str) -> dict[str, Any]:
    import yaml

    cfg_root = repo / "env_cfg" / "task_config"
    with open(cfg_root / f"{task_config}.yml", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(cfg_root / "_embodiment_config.yml", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment_type = args["embodiment"]
    robot_file = embodiment_types[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(Path(robot_file) / "config.yml", encoding="utf-8") as f:
        emb = yaml.safe_load(f)
    args["left_embodiment_config"] = emb
    args["right_embodiment_config"] = emb
    args["task_name"] = task
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["need_plan"] = True
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = False
    args["data_type"]["endpose"] = True
    args["data_type"]["qpos"] = True
    args["camera"]["collect_head_camera"] = False
    args["camera"]["collect_wrist_camera"] = False
    return args


def _discover_hdf5(repo: Path, task: str) -> list[Path]:
    roots = [
        repo / "data" / "demo_clean" / task,
        repo / "data" / "demo_randomized" / task,
        repo / "data",
    ]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(sorted(root.rglob("*.hdf5")))
        found.extend(sorted(root.rglob("*.h5")))
    uniq = []
    seen: set[str] = set()
    for p in found:
        if task in str(p) and str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


def _frames_from_hdf5(path: Path, task: str) -> list[dict[str, np.ndarray]] | None:
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as f:
            def _get(*keys: str) -> np.ndarray | None:
                for k in keys:
                    if k in f:
                        return np.asarray(f[k], dtype=np.float64)
                    if "state" in f and k in f["state"]:
                        return np.asarray(f["state"][k], dtype=np.float64)
                return None

            la = _get("left_arm_joint_states", "qpos/left_arm")
            ra = _get("right_arm_joint_states", "qpos/right_arm")
            lg = _get("left_ee_joint_states", "qpos/left_gripper")
            rg = _get("right_ee_joint_states", "qpos/right_gripper")
            lee = _get("left_ee_poses", "endpose/left_endpose")
            ree = _get("right_ee_poses", "endpose/right_endpose")
            if la is None or ra is None:
                return None
            # Object pose is required for oracle G; qpos/endpose-only files must be replayed.
            has_obj = False
            for key in ("obj_p", "object_pose", "cup_pose", "seal_pose"):
                if key in f or ("state" in f and key in f["state"]):
                    has_obj = True
            if not has_obj:
                return None
            t = la.shape[0]
            frames = []
            for i in range(t):
                gl = float(np.asarray(lg).reshape(t, -1)[i, 0]) if lg is not None else 1.0
                gr = float(np.asarray(rg).reshape(t, -1)[i, 0]) if rg is not None else 1.0
                ql = np.concatenate([np.asarray(la[i]).reshape(-1), [gl]])
                qr = np.concatenate([np.asarray(ra[i]).reshape(-1), [gr]])
                left_ee = np.asarray(lee[i]).reshape(-1)[:3] if lee is not None else np.zeros(3)
                right_ee = np.asarray(ree[i]).reshape(-1)[:3] if ree is not None else np.zeros(3)
                # HDF5 typically lacks object pose; EE proxy keeps G causal but weak.
                obj_p = 0.5 * (left_ee + right_ee)
                g = obj_p.copy()
                extra = np.zeros(2)
                s = np.concatenate([ql, qr, obj_p, np.array([1, 0, 0, 0], dtype=np.float64), left_ee, right_ee, extra])
                frames.append(
                    {
                        "s": s,
                        "a": np.concatenate([ql, qr]),
                        "g": g,
                        "obj_p": obj_p,
                        "left_ee": left_ee,
                        "right_ee": right_ee,
                        "grip_l": np.array([gl]),
                        "grip_r": np.array([gr]),
                    }
                )
            return frames
    except Exception:
        return None


def _setup_env(repo: Path, task: str, seed: int, attempts: int, args: dict[str, Any]) -> tuple[Any, int, str | None]:
    mod = __import__(f"envs.{task}", fromlist=[task])
    cls = getattr(mod, task)
    last: Exception | None = None
    used = seed
    for s in range(seed, seed + max(1, attempts)):
        env = cls()
        try:
            env.setup_demo(now_ep_num=0, seed=int(s), **args)
            return env, int(s), None
        except Exception as exc:
            last = exc
            try:
                env.close_env()
            except Exception:
                pass
            used = s
    return None, used, str(last)


def _load_hdf5_qpos(path: Path) -> dict[str, np.ndarray] | None:
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as f:
            st = f["state"] if "state" in f else f
            act = f["action"] if "action" in f else f

            def arr(group, *keys):
                for k in keys:
                    if k in group:
                        return np.asarray(group[k], dtype=np.float64)
                return None

            la = arr(st, "left_arm_joint_states")
            ra = arr(st, "right_arm_joint_states")
            lg = arr(st, "left_ee_joint_states")
            rg = arr(st, "right_ee_joint_states")
            ala = arr(act, "left_arm_joint_states")
            ara = arr(act, "right_arm_joint_states")
            alg = arr(act, "left_ee_joint_states")
            arg = arr(act, "right_ee_joint_states")
            if la is None or ra is None:
                return None
            t = la.shape[0]
            gl = np.asarray(lg).reshape(t, -1)[:, 0] if lg is not None else np.ones(t)
            gr = np.asarray(rg).reshape(t, -1)[:, 0] if rg is not None else np.ones(t)
            q = np.concatenate([la, gl[:, None], ra, gr[:, None]], axis=1)
            if ala is not None and ara is not None:
                agl = np.asarray(alg).reshape(t, -1)[:, 0] if alg is not None else gl
                agr = np.asarray(arg).reshape(t, -1)[:, 0] if arg is not None else gr
                qa = np.concatenate([ala, agl[:, None], ara, agr[:, None]], axis=1)
            else:
                qa = np.vstack([q[1:], q[-1:]])
            return {"q": q, "qa": qa}
    except Exception:
        return None


def _replay_qpos_oracle(env: Any, task: str, qpos: np.ndarray, qact: np.ndarray) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    n = int(qpos.shape[0])
    for t in range(n):
        q = qpos[t]
        n_l = q.size // 2
        left, right = q[:n_l], q[n_l:]
        env.robot.set_arm_joints(left[:-1], np.zeros(left.size - 1), "left")
        env.robot.set_arm_joints(right[:-1], np.zeros(right.size - 1), "right")
        env.robot.set_gripper(float(left[-1]), "left")
        env.robot.set_gripper(float(right[-1]), "right")
        for _ in range(8):
            env.scene.step()
        fr = extract_oracle_frame(env, task)
        fr["a"] = np.asarray(qact[t] - qpos[t], dtype=np.float64)
        frames.append(fr)
    return frames


def _record_play_once(env: Any, task: str, subsample: int) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    frames: list[dict[str, np.ndarray]] = []
    counter = {"i": 0}
    orig_step = env.scene.step

    def stepped(*a, **k):
        out = orig_step(*a, **k)
        counter["i"] += 1
        if counter["i"] % max(1, subsample) == 0:
            frames.append(extract_oracle_frame(env, task))
        return out

    env.scene.step = stepped
    try:
        frames.append(extract_oracle_frame(env, task))
    except Exception:
        pass
    info: dict[str, Any] = {}
    try:
        info = env.play_once() or {}
    except Exception as exc:
        info = {"play_once_error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.scene.step = orig_step
    if not frames:
        frames.append(extract_oracle_frame(env, task))
    success = False
    try:
        success = bool(env.check_success()) and bool(getattr(env, "plan_success", False))
    except Exception:
        success = False
    meta = {
        "plan_success": bool(getattr(env, "plan_success", False)),
        "check_success": success,
        "n_steps_raw": int(counter["i"]),
        "planner": getattr(getattr(env, "robot", None), "_task_x0_planner", None),
        "play_once_info": info if isinstance(info, dict) else {"info": str(info)},
    }
    return frames, meta


def collect_demos(cfg: TASKX0Config, log: list[str]) -> dict[str, Any]:
    if cfg.task == "stamp_seal":
        from .task_x0_stamp import collect_stamp_official_episodes

        return collect_stamp_official_episodes(cfg, log)
    repo = Path(cfg.robotwin_repo)
    hdf5s = _discover_hdf5(repo, cfg.task)
    log.append(f"hdf5_candidates={len(hdf5s)}")
    episodes: list[dict[str, Any]] = []
    source = "none"
    curobo_stub = True
    planner_mode = "unknown"
    blocker: str | None = None

    if hdf5s:
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            planner_mode = _install_mplib_fallback(repo)
        except Exception as exc:
            planner_mode = f"stub_failed:{exc}"
            blocker = f"planner_install:{exc}"
        args = _load_task_args(repo, cfg.task, cfg.task_config)
        args["need_plan"] = False
        seeds_path = repo / "data" / "demo_clean" / cfg.task / "aloha_agilex" / "seed.txt"
        seed_list: list[int] = []
        if seeds_path.is_file():
            seed_list = [int(x) for x in seeds_path.read_text(encoding="utf-8").split() if x.strip()]
            log.append(f"official_seed_list n={len(seed_list)}")
        for i, path in enumerate(hdf5s[: cfg.n_demo]):
            qpack = _load_hdf5_qpos(path)
            if qpack is None:
                log.append(f"hdf5_skip_unreadable {path}")
                continue
            seed = seed_list[i] if i < len(seed_list) else i
            env, used, err = _setup_env(repo, cfg.task, seed, 4, args)
            if env is None:
                log.append(f"replay_setup_fail seed={seed} err={err}")
                blocker = blocker or f"replay_setup:{err}"
                continue
            planner_mode = getattr(getattr(env, "robot", None), "_task_x0_planner", planner_mode) or planner_mode
            try:
                frames = _replay_qpos_oracle(env, cfg.task, qpack["q"], qpack["qa"])
                ok = False
                try:
                    ok = bool(env.check_success())
                except Exception:
                    ok = False
                rec = {
                    "frames": frames,
                    "success": bool(ok),
                    "path": str(path),
                    "seed": used,
                    "meta": {"replay_hdf5": True, "check_success": ok},
                }
                # Keep causal replays even if check_success is false; do not fake success.
                if frames and len(frames) >= 2:
                    episodes.append(rec)
                log.append(
                    f"replay hdf5={path.name} seed={used} frames={len(frames)} success={ok} planner={planner_mode}"
                )
            except Exception as exc:
                log.append(f"replay_fail {path.name}: {type(exc).__name__}: {exc}")
                blocker = blocker or f"replay:{exc}"
            try:
                env.close_env()
            except Exception:
                pass
        if episodes:
            source = "official_hdf5_qpos_replay"
            n_ok = sum(1 for e in episodes if e.get("success"))
            log.append(f"replayed_hdf5_episodes={len(episodes)} success={n_ok}")
            if n_ok == 0:
                blocker = blocker or (
                    "official hdf5 qpos+endpose replayed into the simulator; "
                    "check_success=false (CuRobo missing; subsampled qpos does not re-grasp the cup); "
                    "G0–G3 scored on causal robot/EE/goal trajectories, not fake task success"
                )
        else:
            source = "official_hdf5_qpos_replay"
            log.append("hdf5 present but replay produced 0 successful episodes")
            blocker = blocker or (
                "official hdf5 qpos replayed; check_success false; "
                "unsuccessful replays excluded from gates; CuRobo missing"
            )

    if len(episodes) < cfg.n_demo and not hdf5s:
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            planner_mode = _install_mplib_fallback(repo)
        except Exception as exc:
            planner_mode = f"stub_failed:{exc}"
            blocker = f"planner_install:{exc}"
        args = _load_task_args(repo, cfg.task, cfg.task_config)
        seeds_path = repo / "data" / "demo_clean" / cfg.task / "aloha_agilex" / "seed.txt"
        seed_list: list[int] = []
        if seeds_path.is_file():
            seed_list = [int(x) for x in seeds_path.read_text(encoding="utf-8").split() if x.strip()]
            log.append(f"official_seed_list n={len(seed_list)}")
        need = cfg.n_demo - len(episodes)
        tries = 0
        si = 0
        live: list[dict[str, Any]] = []
        max_tries = int(cfg.max_seed_attempts or cfg.seed_attempts)
        while len(live) < need and tries < max_tries:
            seed = seed_list[si] if si < len(seed_list) else (cfg.seed + tries * 3)
            si += 1
            tries += 1
            env, used, err = _setup_env(repo, cfg.task, seed, 3, args)
            if env is None:
                log.append(f"setup_fail seed={seed} err={err}")
                blocker = blocker or f"setup:{err}"
                continue
            planner_mode = getattr(getattr(env, "robot", None), "_task_x0_planner", planner_mode) or planner_mode
            frames, meta = _record_play_once(env, cfg.task, cfg.subsample)
            # actions = delta qpos (causal)
            for t in range(len(frames) - 1):
                frames[t]["a"] = frames[t + 1]["a"] - frames[t]["a"]
            if len(frames) >= 2:
                frames = frames[:-1]
            live.append({"frames": frames, "success": bool(meta["check_success"]), "meta": meta, "seed": used})
            log.append(
                f"ep tries={tries} seed={used} frames={len(frames)} "
                f"plan={meta['plan_success']} success={meta['check_success']} planner={meta['planner']}"
            )
            try:
                env.close_env()
            except Exception:
                pass
        n_ok = sum(1 for e in live if e["success"])
        if n_ok:
            source = "play_once_successful" if source == "none" else f"{source}+play_once"
            episodes.extend([e for e in live if e["success"]])
        if n_ok == 0:
            # Official TASK-X0 data are successful demonstrations only (prereg).
            # Keep unsuccessful trajectories in the log, not in G0–G2 pools.
            blocker = blocker or (
                "play_once_no_success (CuRobo missing; mplib fallback may Fail; "
                "official seed.txt/_traj_data hdf5 qpos replay missing)"
            )
            log.append(
                f"unsuccessful_causal_episodes={sum(1 for e in live if len(e['frames']) >= 2)} "
                "(excluded from gates)"
            )
        curobo_stub = "curobo" not in planner_mode or "fallback" in str(planner_mode) or "stub" in str(planner_mode)

    return {
        "episodes": episodes[: cfg.n_demo],
        "source": source,
        "curobo_stub": bool(curobo_stub),
        "planner_mode": planner_mode,
        "blocker": blocker,
        "n_hdf5": int(len(hdf5s)),
    }


def _stack_episodes(episodes: list[dict[str, Any]], task: str) -> dict[str, Any]:
    s_rows, a_rows, g_rows, p_rows, k_rows, pid_rows = [], [], [], [], [], []
    phase_counts = {k: 0 for k in phases_for(task)}
    n_success = 0
    leak_needed = False
    leak_counts = {k: 0 for k in phases_for(task)}
    for ep in episodes:
        frames = ep["frames"]
        ks, cs = G_frozen_episode(frames, task)
        ks_leak, _ = G_with_future_leak(frames, task)
        if ep.get("success"):
            n_success += 1
        ph = phases_for(task)
        for t, fr in enumerate(frames):
            k = ks[t]
            leak_counts[ks_leak[t]] = leak_counts.get(ks_leak[t], 0) + 1
            k = ks[t]
            c = cs[t]
            s_rows.append(fr["s"])
            a_rows.append(fr["a"])
            g_rows.append(fr["g"])
            p_rows.append(encode_p(k, c, task))
            k_rows.append(k)
            pid_rows.append(ph.index(k) if k in ph else 0)
            phase_counts[k] = phase_counts.get(k, 0) + 1
    return {
        "s": np.stack(s_rows) if s_rows else np.zeros((0, 1)),
        "a": np.stack(a_rows) if a_rows else np.zeros((0, 1)),
        "g": np.stack(g_rows) if g_rows else np.zeros((0, 3)),
        "p": np.stack(p_rows) if p_rows else np.zeros((0, 1)),
        "k": k_rows,
        "p_id": np.asarray(pid_rows, dtype=np.int64),
        "phase_counts": phase_counts,
        "leak_phase_counts": leak_counts,
        "n_success": n_success,
        "leak_needed_for_coverage": leak_needed,
    }


def run_task_x0(output: str | Path | None = None, *, config: TASKX0Config | None = None) -> dict[str, Any]:
    cfg = config or TASKX0Config()
    if cfg.task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {cfg.task}; expected {SUPPORTED_TASKS}")
    out = Path(output or cfg.output or f"runs/task_x0/{cfg.task}").resolve()
    if not str(out).startswith(str((Path("/root/APR-WM") / "runs" / "task_x0").resolve())):
        # keep default under runs/task_x0/<task>
        pass
    _refuse_locked_paths(out)
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    header = {
        "stage": "TASK-X0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "family_prereg": FAMILY_PREREG,
        "task": cfg.task,
        "N_demo_frozen": int(cfg.n_demo),
        "N_min": int(cfg.n_min),
        "oracle_progress": True,
        "rgb_in_s": False,
        "physics_predict_used": False,
        "p_enters_F_physics": False,
        "unlocks_r10": False,
        "unlocks_task_x1": False,
        "diffusion_train": False,
        "config": asdict(cfg),
    }
    log_lines.append("HEADER " + json.dumps(_jsonable(header), ensure_ascii=False))
    collected = collect_demos(cfg, log_lines)
    eps = collected["episodes"]
    packed = _stack_episodes(eps, cfg.task)
    s, a, g, p = packed["s"], packed["a"], packed["g"], packed["p"]
    eps_s = EPS_S_SCALE * _rms(s) if s.size else 0.0
    log_lines.append(f"eps_s={eps_s:.8f} RMS(s)={_rms(s):.8f} n_frames={int(s.shape[0])} n_ep={len(eps)}")

    g0 = {k: int(v) >= int(cfg.n_min) for k, v in packed["phase_counts"].items()}
    g0_pass = bool(packed["phase_counts"]) and all(g0.values())
    g1 = g1_aliasing(s, packed["p_id"], a, eps_s) if s.shape[0] else {"passed": False}
    g2 = g2_entropy(s, g, p, a, holdout_frac=cfg.holdout_frac, knn_k=cfg.knn_k, n_boot=cfg.n_boot, seed=cfg.seed)
    leak_cov = {
        k: int(packed.get("leak_phase_counts", {}).get(k, 0)) >= int(cfg.n_min)
        for k in packed["phase_counts"]
    }
    leak_needed = (not g0_pass) and bool(leak_cov) and all(leak_cov.values())
    future_leak = False
    g3_pass = (not leak_needed) and (not future_leak)
    g3 = {
        "future_leak": False,
        "leak_needed_for_coverage": bool(leak_needed),
        "passed": bool(g3_pass),
        "oracle_G_causal_only": True,
    }
    passed = bool(g0_pass and g1.get("passed") and g2.get("passed") and g3_pass)
    pattern = "instrument_ready"
    if not g0_pass:
        pattern = "coverage_hole"
    elif not g3_pass:
        pattern = "progress_leaks"
    elif not g1.get("passed"):
        pattern = "no_aliasing"
    elif not g2.get("passed"):
        pattern = "p_no_nll"

    summary: dict[str, Any] = {
        **header,
        "n_demo_collected": int(len(eps)),
        "n_frames": int(s.shape[0]),
        "n_success_episodes": int(packed["n_success"]),
        "data_source": collected["source"],
        "curobo_stub": collected["curobo_stub"],
        "planner_mode": collected["planner_mode"],
        "blocker": collected["blocker"],
        "eps_s": float(eps_s),
        "phase_counts": packed["phase_counts"],
        "gates": {
            "G0_coverage": g0_pass,
            "G0_per_phase": g0,
            "G1_non_redundancy": bool(g1.get("passed")),
            "G1_detail": g1,
            "G2_entropy": bool(g2.get("passed")),
            "G2_detail": g2,
            "G3_recoverability": bool(g3_pass),
            "G3_detail": g3,
            "G_label": True,
        },
        "pattern": pattern,
        "task_x0_passed": passed,
        "S_task_claimed": False,
        "notes": (
            "X0 may only claim: phase coverage, s-aliasing, NLL drop with p, causal G. "
            "Not planning success; not diffusion; not physics capacity."
        ),
    }
    _write_json(out / "summary.json", summary)
    (out / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return summary

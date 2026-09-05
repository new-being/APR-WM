"""RTWX-O0G1b: object reference-frame alignment (known δ_O; no backbone; no O0G retune).

B0 = surface centroid vs pose (O0G G1 negative control)
B1 = p_surf - R_BO δ_O vs pose (object-frame known geometry prior)

Does not fit O0G test +5.11cm bias. Does not unlock O1. G2 locked until primary PASS.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args, _quat_fix
from .rtwx_o0d import _cup_ids
from .rtwx_o0g import (
    MASK_PX,
    _opengl_mask_centroid,
    _pos_metrics,
)
from .rtwx_o0d3 import _head_cam
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G1B_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g1b.reference_frame_alignment.v1"
CAMERA = "head_camera"
SEED = 23601
N_EP = 8
N_STEPS = 32
CUP_MODEL = "021_cup"
CUP_MODEL_ID = 0
# same primary gate as O0G G1 — do not loosen
EP_MAX = 0.20
MED_MAX = 0.05
STRONG_MED = 0.02
DELTA_SOURCE = "assets/objects/021_cup/model_data0.json:center*scale"


@dataclass(frozen=True)
class RTWXO0G1BConfig:
    output: str = "runs/rtwx_o0g1b"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_ep: int = N_EP
    n_steps: int = N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G1b must not write there")


def _lock(cfg: RTWXO0G1BConfig) -> RTWXO0G1BConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, n_ep=N_EP, n_steps=N_STEPS, seed=SEED)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Sapien/wxyz quaternion → rotation matrix R_BO (object→world)."""
    q = _quat_fix(np.asarray(q, dtype=np.float64).reshape(1, 4))[0]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_delta_O(robotwin_repo: str | Path) -> dict[str, Any]:
    """Known geometry prior: mesh AABB center in object frame (not test-fit bias)."""
    path = Path(robotwin_repo) / "assets" / "objects" / CUP_MODEL / f"model_data{CUP_MODEL_ID}.json"
    data = json.loads(path.read_text())
    center = np.asarray(data["center"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    delta = center * scale
    return {
        "delta_O": delta,
        "source": str(path.relative_to(Path(robotwin_repo)) if Path(robotwin_repo) in path.parents else path),
        "source_kind": "mesh_aabb_center_times_scale",
        "cup_model": CUP_MODEL,
        "model_id": CUP_MODEL_ID,
        "not_from_o0g_test_bias": True,
    }


def _align_pose(p_surf: np.ndarray, quat: np.ndarray, delta_O: np.ndarray) -> np.ndarray:
    out = np.empty_like(p_surf)
    for i in range(p_surf.shape[0]):
        r = _quat_to_R(quat[i])
        out[i] = p_surf[i] - r @ delta_O
    return out


def _gate(m: dict[str, float]) -> dict[str, Any]:
    primary = bool(m["E_p"] <= EP_MAX and m["median_ep_m"] <= MED_MAX)
    strong = bool(primary and m["median_ep_m"] <= STRONG_MED)
    return {
        "ok_primary": primary,
        "ok_strong": strong,
        "gate": {"E_p_max": EP_MAX, "median_ep_m_max": MED_MAX, "strong_median_ep_m_max": STRONG_MED},
        **m,
    }


def _pattern(b1: dict[str, Any]) -> str:
    if b1["ok_strong"]:
        return "geometry_reference_aligned"
    if b1["ok_primary"]:
        return "geometry_usable_but_coarse"
    return "geometry_alignment_insufficient"


def _numpy_pool(cfg: RTWXO0G1BConfig, rng: np.random.Generator, delta_O: np.ndarray) -> dict[str, Any]:
    """Synthetic upright cups; surface = pose + R δ_O + small noise."""
    n = int(cfg.n_ep) * int(cfg.n_steps)
    # placement qpos from place_empty_cup
    quat = np.tile(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64), (n, 1))
    p = rng.uniform([-0.3, -0.2, 0.74], [0.3, 0.05, 0.74], size=(n, 3))
    p_surf = np.stack([p[i] + _quat_to_R(quat[i]) @ delta_O for i in range(n)])
    p_surf = p_surf + rng.normal(0, 0.003, size=p_surf.shape)
    return {
        "p": p,
        "quat": quat,
        "p_surf": p_surf,
        "in_fov": np.ones(n, dtype=bool),
        "mask_area": np.full(n, 2000, dtype=np.int32),
        "n_ep": int(cfg.n_ep),
        "n_steps": int(cfg.n_steps),
    }


def collect_align_frames(cfg: RTWXO0G1BConfig, *, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep, n_steps = int(cfg.n_ep), int(cfg.n_steps)
    seed0 = int(cfg.seed)
    eps: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(eps) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(eps) % 2 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0g1b] collect valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(n_steps + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            sim_ids = _cup_ids(env)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            tt = np.arange(n_steps) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            cam = _head_cam(env)
            rows: dict[str, list] = {k: [] for k in ("p", "quat", "p_surf", "in_fov", "mask_area")}
            invalid = None
            repaired = None
            for i in range(n_steps):
                p, quat = _cup_pose(env)
                env._update_render()
                env.cameras.update_picture()
                try:
                    seg = np.asarray(cam.get_picture("Segmentation"))
                    actor = np.asarray(seg[..., 1]).astype(np.int32)
                except Exception:
                    actor = np.zeros((480, 640), dtype=np.int32)
                try:
                    position = np.asarray(cam.get_picture("Position"))
                    model = np.asarray(cam.get_model_matrix())
                except Exception:
                    position, model = None, None
                h, w = actor.shape[:2]
                from .rtwx_o0d1 import _project

                k = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
                e = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
                pr = _project(k, e, p, w, h)
                uu = int(np.clip(round(pr["u"]), 0, w - 1))
                vv = int(np.clip(round(pr["v"]), 0, h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                if repaired is None and pr["in_fov"] and id_uv >= 0:
                    repaired = id_uv
                cid = repaired if repaired is not None else (sorted(sim_ids)[0] if sim_ids else -1)
                mask = (actor == cid) if cid >= 0 else np.zeros_like(actor, dtype=bool)
                area = int(mask.sum())
                p_surf = None
                if position is not None and model is not None and area >= MASK_PX:
                    p_surf = _opengl_mask_centroid(position, model, mask)
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["p_surf"].append(p_surf if p_surf is not None else np.full(3, np.nan))
                rows["in_fov"].append(bool(pr["in_fov"]))
                rows["mask_area"].append(area)
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig, _box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "noop"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
            env.close()
            if invalid or len(rows["p"]) != n_steps:
                continue
            eps.append(
                {
                    "p": np.asarray(rows["p"], dtype=np.float64),
                    "quat": np.asarray(rows["quat"], dtype=np.float64),
                    "p_surf": np.asarray(rows["p_surf"], dtype=np.float64),
                    "in_fov": np.asarray(rows["in_fov"], dtype=bool),
                    "mask_area": np.asarray(rows["mask_area"], dtype=np.int32),
                }
            )
        except Exception as exc:
            print(f"[rtwx-o0g1b] collect fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"collect:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError("O0G1b collect: 0 episodes")
    out: dict[str, Any] = {
        k: np.concatenate([e[k] for e in eps], axis=0) for k in ("p", "quat", "p_surf", "in_fov", "mask_area")
    }
    out["n_ep"] = len(eps)
    out["n_steps"] = n_steps
    return out


def _save(path: Path, pool: dict[str, Any]) -> None:
    payload = {k: pool[k] for k in ("p", "quat", "p_surf", "in_fov", "mask_area")}
    payload["n_ep"] = np.array([pool["n_ep"]])
    payload["n_steps"] = np.array([pool["n_steps"]])
    np.savez_compressed(path, **payload)


def _load(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if "p_surf" not in z:
        return None
    out = {k: np.asarray(z[k]) for k in ("p", "quat", "p_surf", "in_fov", "mask_area")}
    out["n_ep"] = int(z["n_ep"][0])
    out["n_steps"] = int(z["n_steps"][0])
    return out


def _eval_pair(p_hat: np.ndarray, pool: dict[str, Any]) -> dict[str, Any]:
    p = pool["p"]
    valid = np.isfinite(p_hat).all(axis=1) & (pool["mask_area"] >= MASK_PX) & pool["in_fov"]
    n_valid = int(valid.sum())
    if n_valid < max(8, int(0.5 * p.shape[0])):
        return {
            "ok_primary": False,
            "ok_strong": False,
            "reason": "insufficient_valid",
            "n_valid": n_valid,
            "n": int(p.shape[0]),
            "gate": {"E_p_max": EP_MAX, "median_ep_m_max": MED_MAX, "strong_median_ep_m_max": STRONG_MED},
        }
    m = _pos_metrics(p_hat[valid], p[valid])
    out = _gate(m)
    out["n_valid"] = n_valid
    out["frac_valid"] = float(n_valid / p.shape[0])
    e = p_hat[valid] - p[valid]
    out["bias_xyz_m"] = e.mean(0).tolist()
    return out


def run_rtwx_o0g1b(output: str | Path, config: RTWXO0G1BConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G1BConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    prior = load_delta_O(cfg.robotwin_repo if cfg.backend == "robotwin" else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    header = {
        "stage": "RTWX-O0G1b",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "camera": CAMERA,
        "seed": cfg.seed,
        "depends_on": "O0G=geometry_insufficient",
        "delta_O": delta_O.tolist(),
        "delta_source": DELTA_SOURCE,
        "delta_meta": prior,
        "not_from_o0g_test_bias": True,
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    if cfg.backend == "numpy":
        pool = _numpy_pool(cfg, np.random.default_rng(cfg.seed), delta_O)
    else:
        stop: list[str] = []
        cache = root / "cache_o0g1b.npz"
        pool = _load(cache)
        if pool is None:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_align_frames(cfg, rng=np.random.default_rng(cfg.seed), stop=stop)
            _save(cache, pool)
        else:
            print("[rtwx-o0g1b] cache hit", flush=True)
        if stop:
            raise RuntimeError(f"O0G1b collect stop: {stop}")

    b0 = _eval_pair(pool["p_surf"], pool)
    p_pose = _align_pose(pool["p_surf"], pool["quat"], delta_O)
    b1 = _eval_pair(p_pose, pool)
    # secondary audit only: world-z constant equal to (R δ_O)_z under mean upright R
    r0 = _quat_to_R(pool["quat"][0])
    dz = float((r0 @ delta_O)[2])
    p_wz = pool["p_surf"].copy()
    p_wz[:, 2] = p_wz[:, 2] - dz
    b1_wz = _eval_pair(p_wz, pool)
    print(
        f"[rtwx-o0g1b] B0 med={b0.get('median_ep_m', float('nan')):.4f} "
        f"B1 med={b1.get('median_ep_m', float('nan')):.4f} strong={b1.get('ok_strong')}",
        flush=True,
    )
    pattern = _pattern(b1)
    unlock_g2 = bool(b1.get("ok_primary"))
    summary = {
        "header": header,
        "pattern": pattern,
        "B0_surface_vs_pose": b0,
        "B1_aligned_vs_pose": b1,
        "B1_world_z_secondary": {**b1_wz, "does_not_override_primary": True, "dz_m": dz},
        "delta_O_m": delta_O.tolist(),
        "unlocks_g2_prereg": unlock_g2,
        "unlocks_o1": False,
        "does_not_rewrite_o0g": True,
        "does_not_fit_o0g_test_bias": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "B0_surface_vs_pose": b0,
            "B1_aligned_vs_pose": b1,
            "B1_world_z_secondary": summary["B1_world_z_secondary"],
            "unlocks_g2_prereg": unlock_g2,
        },
    )
    return summary

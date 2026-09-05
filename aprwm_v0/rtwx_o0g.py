"""RTWX-O0G: geometry-mediated object position (G0 oracle → G1 mask → G2 learned).

Pause whole-image RGB→p_B. Known camera geometry is explicit; network must not relearn it.
Does not rewrite O0R. Does not unlock O1. Not R10.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args
from .rtwx_o0d import _cup_ids
from .rtwx_o0d1 import _project
from .rtwx_o0d3 import _head_cam
from .rtwx_x0 import _nrmse
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g.geometry_mediated_position.v1"
CAMERA = "head_camera"
SEED = 22601
N_EP = 8
N_STEPS = 32
MASK_PX = 50
# G0: near numerical precision on oracle (u,v,z_C) round-trip
G0_EP_MAX = 0.05
G0_MED_M_MAX = 0.005
# G1: pose origin vs visual centroid may differ by a few cm
G1_EP_MAX = 0.20
G1_MED_M_MAX = 0.05


@dataclass(frozen=True)
class RTWXO0GConfig:
    output: str = "runs/rtwx_o0g"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    stage: str = "g0"  # g0 | g1 | g01 | all (all=G0→G1; G2 deferred)
    n_ep: int = N_EP
    n_steps: int = N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G must not write there")


def _lock(cfg: RTWXO0GConfig) -> RTWXO0GConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, n_ep=N_EP, n_steps=N_STEPS, seed=SEED)


def _as_T44(E: np.ndarray) -> np.ndarray:
    e = np.asarray(E, dtype=np.float64)
    if e.shape == (4, 4):
        return e
    if e.shape == (3, 4):
        t = np.eye(4, dtype=np.float64)
        t[:3, :] = e
        return t
    raise ValueError(f"bad extrinsic shape {e.shape}")


def _unproject_uvd(K: np.ndarray, E: np.ndarray, u: float, v: float, d: float) -> np.ndarray:
    """p_C = d K^{-1}[u,v,1]^T ; p_B = T_BC p_C. d = camera-frame z (OpenCV)."""
    k = np.asarray(K, dtype=np.float64)
    ray = np.linalg.inv(k) @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    p_c = float(d) * ray
    t_cw = _as_T44(E)
    t_wc = np.linalg.inv(t_cw)
    p_w = t_wc @ np.array([p_c[0], p_c[1], p_c[2], 1.0], dtype=np.float64)
    return p_w[:3].astype(np.float64)


def _pos_metrics(phat: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(np.asarray(phat) - np.asarray(p), axis=1)
    return {
        "E_p": float(_nrmse(np.asarray(phat), np.asarray(p))),
        "median_ep_m": float(np.median(err)),
        "mean_ep_m": float(np.mean(err)),
        "p90_ep_m": float(np.percentile(err, 90)),
        "max_ep_m": float(np.max(err)),
        "n": int(p.shape[0]),
    }


def _opengl_mask_centroid(position: np.ndarray, model: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Sapien Position (OpenGL) + model_matrix → world median of masked points."""
    m = np.asarray(mask, dtype=bool)
    pos = np.asarray(position)
    if pos.ndim != 3 or pos.shape[-1] < 3 or not m.any():
        return None
    pts = pos[m][:, :3]
    if pos.shape[-1] >= 4:
        pts = pts[pos[m][:, 3] < 1.0]
    if pts.shape[0] < 8:
        return None
    mm = np.asarray(model, dtype=np.float64)
    r = mm[:3, :3]
    t = mm[:3, 3]
    world = pts @ r.T + t
    return np.median(world, axis=0).astype(np.float64)


def _numpy_g0(cfg: RTWXO0GConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic OpenCV K/E round-trip; must pass G0."""
    fx, fy, cx, cy = 400.0, 400.0, 320.0, 240.0
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    # Camera looks at origin from +z in world-ish setup
    t_cw = np.eye(4, dtype=np.float64)
    t_cw[:3, 3] = np.array([0.05, -0.02, 0.8])
    ps, phats = [], []
    for _ in range(int(cfg.n_ep) * int(cfg.n_steps)):
        p = rng.uniform([-0.3, -0.2, 0.7], [0.3, 0.05, 0.75])
        pr = _project(k, t_cw, p, 640, 480)
        phat = _unproject_uvd(k, t_cw, pr["u"], pr["v"], pr["z"])
        ps.append(p)
        phats.append(phat)
    p = np.stack(ps)
    phat = np.stack(phats)
    m = _pos_metrics(phat, p)
    ok = bool(m["E_p"] <= G0_EP_MAX and m["median_ep_m"] <= G0_MED_M_MAX)
    return {"ok": ok, "gate": {"E_p_max": G0_EP_MAX, "median_ep_m_max": G0_MED_M_MAX}, **m, "backend": "numpy"}


def _numpy_g1(cfg: RTWXO0GConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic masked depth centroid via same unproject grid — passes if geometry ok."""
    fx, fy, cx, cy = 400.0, 400.0, 32.0, 32.0
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    t_cw = np.eye(4, dtype=np.float64)
    t_cw[:3, 3] = np.array([0.0, 0.0, 1.0])
    size = 64
    ps, phats = [], []
    for _ in range(int(cfg.n_ep) * int(cfg.n_steps)):
        p = rng.uniform([-0.15, -0.1, 0.7], [0.15, 0.05, 0.75])
        pr = _project(k, t_cw, p, size, size)
        uu, vv = int(round(pr["u"])), int(round(pr["v"]))
        mask = np.zeros((size, size), dtype=bool)
        r = 4
        mask[max(0, vv - r) : min(size, vv + r + 1), max(0, uu - r) : min(size, uu + r + 1)] = True
        ys, xs = np.where(mask)
        pts = []
        for u, v in zip(xs.tolist(), ys.tolist()):
            pts.append(_unproject_uvd(k, t_cw, float(u) + 0.5, float(v) + 0.5, pr["z"]))
        phat = np.median(np.stack(pts), axis=0)
        ps.append(p)
        phats.append(phat)
    p = np.stack(ps)
    phat = np.stack(phats)
    m = _pos_metrics(phat, p)
    ok = bool(m["E_p"] <= G1_EP_MAX and m["median_ep_m"] <= G1_MED_M_MAX)
    return {"ok": ok, "gate": {"E_p_max": G1_EP_MAX, "median_ep_m_max": G1_MED_M_MAX}, **m, "backend": "numpy"}


def collect_geom_frames(cfg: RTWXO0GConfig, *, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    """Collect GT p + K,E + oracle (u,v,z_C) + mask + OpenGL Position/model for G0/G1."""
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
            print(f"[rtwx-o0g] collect valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
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
            keys = (
                "p",
                "u",
                "v",
                "z_c",
                "p_g0",
                "p_g1",
                "in_fov",
                "mask_area",
                "K",
                "E",
            )
            rows: dict[str, list] = {k: [] for k in keys}
            invalid = None
            repaired = None
            for i in range(n_steps):
                p, _quat = _cup_pose(env)
                env._update_render()
                env.cameras.update_picture()
                try:
                    seg = np.asarray(cam.get_picture("Segmentation"))
                    actor = np.asarray(seg[..., 1]).astype(np.int32)
                except Exception:
                    actor = np.zeros((480, 640), dtype=np.int32)
                try:
                    position = np.asarray(cam.get_picture("Position"))
                except Exception:
                    position = None
                try:
                    model = np.asarray(cam.get_model_matrix())
                except Exception:
                    model = None
                h, w = actor.shape[:2]
                k = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
                e = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
                pr = _project(k, e, p, w, h)
                p_g0 = _unproject_uvd(k, e, pr["u"], pr["v"], pr["z"])
                uu = int(np.clip(round(pr["u"]), 0, w - 1))
                vv = int(np.clip(round(pr["v"]), 0, h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                if repaired is None and pr["in_fov"] and id_uv >= 0:
                    repaired = id_uv
                cid = repaired if repaired is not None else (sorted(sim_ids)[0] if sim_ids else -1)
                mask = (actor == cid) if cid >= 0 else np.zeros_like(actor, dtype=bool)
                area = int(mask.sum())
                p_g1 = None
                if position is not None and model is not None and area >= MASK_PX:
                    p_g1 = _opengl_mask_centroid(position, model, mask)
                rows["p"].append(p)
                rows["u"].append(float(pr["u"]))
                rows["v"].append(float(pr["v"]))
                rows["z_c"].append(float(pr["z"]))
                rows["p_g0"].append(p_g0)
                rows["p_g1"].append(p_g1 if p_g1 is not None else np.full(3, np.nan))
                rows["in_fov"].append(bool(pr["in_fov"]))
                rows["mask_area"].append(area)
                rows["K"].append(k)
                rows["E"].append(_as_T44(e))
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "noop"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
                _ = box
            env.close()
            if invalid or len(rows["p"]) != n_steps:
                continue
            eps.append(
                {
                    "p": np.asarray(rows["p"], dtype=np.float64),
                    "u": np.asarray(rows["u"], dtype=np.float64),
                    "v": np.asarray(rows["v"], dtype=np.float64),
                    "z_c": np.asarray(rows["z_c"], dtype=np.float64),
                    "p_g0": np.asarray(rows["p_g0"], dtype=np.float64),
                    "p_g1": np.asarray(rows["p_g1"], dtype=np.float64),
                    "in_fov": np.asarray(rows["in_fov"], dtype=bool),
                    "mask_area": np.asarray(rows["mask_area"], dtype=np.int32),
                    "K": np.stack(rows["K"]),
                    "E": np.stack(rows["E"]),
                }
            )
        except Exception as exc:
            print(f"[rtwx-o0g] collect fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"collect:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError("O0G collect: 0 episodes")
    out: dict[str, Any] = {
        k: np.concatenate([e[k] for e in eps], axis=0)
        for k in ("p", "u", "v", "z_c", "p_g0", "p_g1", "in_fov", "mask_area", "K", "E")
    }
    out["n_ep"] = len(eps)
    out["n_steps"] = n_steps
    return out


def _save_cache(path: Path, pool: dict[str, Any]) -> None:
    payload = {k: pool[k] for k in ("p", "u", "v", "z_c", "p_g0", "p_g1", "in_fov", "mask_area", "K", "E")}
    payload["n_ep"] = np.array([pool["n_ep"]])
    payload["n_steps"] = np.array([pool["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if "p" not in z:
        return None
    out = {k: np.asarray(z[k]) for k in ("p", "u", "v", "z_c", "p_g0", "p_g1", "in_fov", "mask_area", "K", "E")}
    out["n_ep"] = int(z["n_ep"][0])
    out["n_steps"] = int(z["n_steps"][0])
    return out


def _eval_g0(pool: dict[str, Any]) -> dict[str, Any]:
    m = _pos_metrics(pool["p_g0"], pool["p"])
    ok = bool(m["E_p"] <= G0_EP_MAX and m["median_ep_m"] <= G0_MED_M_MAX)
    return {"ok": ok, "gate": {"E_p_max": G0_EP_MAX, "median_ep_m_max": G0_MED_M_MAX}, **m}


def _eval_g1(pool: dict[str, Any]) -> dict[str, Any]:
    p = pool["p"]
    phat = pool["p_g1"]
    valid = np.isfinite(phat).all(axis=1) & (pool["mask_area"] >= MASK_PX) & pool["in_fov"]
    n_valid = int(valid.sum())
    if n_valid < max(8, int(0.5 * p.shape[0])):
        return {
            "ok": False,
            "reason": "insufficient_masked_depth",
            "n_valid": n_valid,
            "n": int(p.shape[0]),
            "frac_valid": float(n_valid / max(p.shape[0], 1)),
            "gate": {"E_p_max": G1_EP_MAX, "median_ep_m_max": G1_MED_M_MAX},
        }
    m = _pos_metrics(phat[valid], p[valid])
    ok = bool(m["E_p"] <= G1_EP_MAX and m["median_ep_m"] <= G1_MED_M_MAX)
    return {
        "ok": ok,
        "gate": {"E_p_max": G1_EP_MAX, "median_ep_m_max": G1_MED_M_MAX},
        "n_valid": n_valid,
        "frac_valid": float(n_valid / p.shape[0]),
        **m,
    }


def _pattern(*, g0: bool | None, g1: bool | None, g2: bool | None) -> str:
    if g0 is False:
        return "camera_geometry_contract_failure"
    if g0 is True and g1 is False:
        return "geometry_insufficient"
    if g0 is True and g1 is True and g2 is True:
        return "geometry_mediated_position_supported"
    # G2 False (failed/deferred) or not run yet → localization still pending after G0∧G1
    if g0 is True and g1 is True:
        return "geometry_feasible_localization_pending"
    if g0 is True and g1 is None:
        return "g0_pass_g1_pending"
    return "incomplete"


def run_rtwx_o0g(output: str | Path, config: RTWXO0GConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0GConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    stage = str(cfg.stage).lower()
    header = {
        "stage": "RTWX-O0G",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "camera": CAMERA,
        "seed": cfg.seed,
        "run_stage": stage,
        "depends_on": "O0R=object_pose_failure",
        "pauses_rgb_to_p": True,
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    g0_res: dict[str, Any] | None = None
    g1_res: dict[str, Any] | None = None
    g2_res: dict[str, Any] | None = None
    want_g0 = stage in {"g0", "g01", "all"}
    want_g1 = stage in {"g1", "g01", "all"}
    want_g2 = stage in {"g2", "all"}

    if cfg.backend == "numpy":
        rng = np.random.default_rng(cfg.seed)
        if want_g0:
            g0_res = _numpy_g0(cfg, rng)
            print(f"[rtwx-o0g] G0 E_p={g0_res['E_p']:.6g} med={g0_res['median_ep_m']:.3e} ok={g0_res['ok']}", flush=True)
        if want_g1 and (g0_res is None or g0_res.get("ok")):
            g1_res = _numpy_g1(cfg, rng)
            print(f"[rtwx-o0g] G1 E_p={g1_res['E_p']:.6g} med={g1_res['median_ep_m']:.3e} ok={g1_res['ok']}", flush=True)
        elif want_g1 and g0_res is not None and not g0_res.get("ok"):
            g1_res = {"ok": False, "skipped": True, "reason": "g0_failed"}
        if want_g2:
            g2_res = {"ok": False, "deferred": True, "reason": "g2_not_implemented_pending_g0_g1"}
    else:
        stop: list[str] = []
        cache = root / "cache_o0g_geom.npz"
        pool = _load_cache(cache)
        if pool is None:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_geom_frames(cfg, rng=np.random.default_rng(cfg.seed), stop=stop)
            _save_cache(cache, pool)
        else:
            print("[rtwx-o0g] cache hit", flush=True)
        if stop:
            raise RuntimeError(f"O0G collect stop: {stop}")
        if want_g0:
            g0_res = _eval_g0(pool)
            print(
                f"[rtwx-o0g] G0 E_p={g0_res['E_p']:.6g} med_m={g0_res['median_ep_m']:.3e} ok={g0_res['ok']}",
                flush=True,
            )
        if want_g1:
            if g0_res is not None and not g0_res.get("ok"):
                g1_res = {"ok": False, "skipped": True, "reason": "g0_failed"}
            else:
                g1_res = _eval_g1(pool)
                print(
                    f"[rtwx-o0g] G1 E_p={g1_res.get('E_p', float('nan')):.6g} "
                    f"med_m={g1_res.get('median_ep_m', float('nan')):.3e} ok={g1_res['ok']}",
                    flush=True,
                )
        if want_g2:
            g2_res = {"ok": False, "deferred": True, "reason": "g2_not_implemented_pending_g0_g1"}

    g0_ok = None if g0_res is None else bool(g0_res.get("ok"))
    g1_ok = None if g1_res is None else bool(g1_res.get("ok"))
    g2_ok = None if g2_res is None else (False if g2_res.get("deferred") else bool(g2_res.get("ok")))
    pattern = _pattern(g0=g0_ok, g1=g1_ok, g2=g2_ok)
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": g0_res,
        "G1": g1_res,
        "G2": g2_res,
        "unlocks_o1": False,
        "pauses_rgb_to_p": True,
        "does_not_rewrite_o0r": True,
        "does_not_claim_real_camera": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {"pattern": pattern, "G0": g0_res, "G1": g1_res, "G2": g2_res})
    return summary

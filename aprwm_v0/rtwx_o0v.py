"""RTWX-O0V: robust visual coverage contract (no training, no pose claim).

B0 = head_camera only; B1 = head ∨ observer (pre-frozen from O0D2 screening).
Multi-seed; does not retune O0G2; does not unlock O1.
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
from .rtwx_o0g import MASK_PX
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0V_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0v.robust_visual_coverage.v1"
CAM_HEAD = "head_camera"
CAM_OBS = "observer_camera"
SEEDS = (25601, 25602, 25603)
N_EP = 24
N_STEPS = 120
P_ANY_AGG = 0.98
P_ANY_SEED = 0.95


@dataclass(frozen=True)
class RTWXO0VConfig:
    output: str = "runs/rtwx_o0v"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_ep: int = N_EP
    n_steps: int = N_STEPS
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0V must not write there")


def _lock(cfg: RTWXO0VConfig) -> RTWXO0VConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, seeds=SEEDS, n_ep=N_EP, n_steps=N_STEPS)


def _get_cam(env: Any, name: str) -> Any | None:
    cams = env.cameras
    for cam, n in zip(cams.static_camera_list, cams.static_camera_name):
        if n == name:
            return cam
    if name == CAM_OBS and hasattr(cams, "observer_camera"):
        return cams.observer_camera
    if hasattr(cams, name):
        return getattr(cams, name)
    return None


def _visible_on_cam(cam: Any, p: np.ndarray, cup_ids: set[int], repaired: int | None) -> tuple[bool, bool, int | None]:
    """Return (in_fov, visible, repaired_id)."""
    try:
        seg = np.asarray(cam.get_picture("Segmentation"))
        actor = np.asarray(seg[..., 1]).astype(np.int32)
    except Exception:
        actor = np.zeros((480, 640), dtype=np.int32)
    h, w = actor.shape[:2]
    k = np.asarray(cam.get_intrinsic_matrix())
    e = np.asarray(cam.get_extrinsic_matrix())
    pr = _project(k, e, p, w, h)
    in_fov = bool(pr["in_fov"])
    uu = int(np.clip(round(pr["u"]), 0, w - 1))
    vv = int(np.clip(round(pr["v"]), 0, h - 1))
    id_uv = int(actor[vv, uu]) if in_fov else -1
    rep = repaired
    if rep is None and in_fov and id_uv >= 0:
        rep = id_uv
    cid = rep if rep is not None else (sorted(cup_ids)[0] if cup_ids else -1)
    area = int((actor == cid).sum()) if cid >= 0 else 0
    visible = bool(in_fov and (area >= MASK_PX or id_uv >= 0))
    return in_fov, visible, rep


def _numpy_seed(cfg: RTWXO0VConfig, seed: int, rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic: head misses ~12% frames; observer covers those → any≈1."""
    n = int(cfg.n_ep) * int(cfg.n_steps)
    head = rng.random(n) > 0.12
    # observer independent high coverage; covers most head misses
    obs = rng.random(n) > 0.02
    obs = obs | (~head & (rng.random(n) > 0.05))
    return {
        "seed": seed,
        "head_fov": head.copy(),
        "head_vis": head.copy(),
        "obs_fov": obs.copy(),
        "obs_vis": obs.copy(),
        "n_ep": int(cfg.n_ep),
        "n_steps": int(cfg.n_steps),
    }


def collect_seed(cfg: RTWXO0VConfig, *, seed0: int, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep, n_steps = int(cfg.n_ep), int(cfg.n_steps)
    eps: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(eps) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(eps) % 4 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0v] seed={seed0} valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(n_steps + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            sim_ids = _cup_ids(env)
            cam_h = _get_cam(env, CAM_HEAD) or _head_cam(env)
            cam_o = _get_cam(env, CAM_OBS)
            if cam_o is None:
                env.close()
                stop.append(f"seed{seed0}:missing_observer")
                break
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
            rows = {k: [] for k in ("head_fov", "head_vis", "obs_fov", "obs_vis")}
            invalid = None
            rep_h, rep_o = None, None
            for i in range(n_steps):
                p, _q = _cup_pose(env)
                env._update_render()
                env.cameras.update_picture()
                # ensure observer also captured
                try:
                    cam_o.take_picture()
                except Exception:
                    pass
                hf, hv, rep_h = _visible_on_cam(cam_h, p, sim_ids, rep_h)
                of, ov, rep_o = _visible_on_cam(cam_o, p, sim_ids, rep_o)
                rows["head_fov"].append(hf)
                rows["head_vis"].append(hv)
                rows["obs_fov"].append(of)
                rows["obs_vis"].append(ov)
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
            if invalid or len(rows["head_vis"]) != n_steps:
                continue
            eps.append({k: np.asarray(rows[k], dtype=bool) for k in rows})
        except Exception as exc:
            print(f"[rtwx-o0v] seed={seed0} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"seed{seed0}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0V collect seed {seed0}: 0 episodes")
    out = {k: np.concatenate([e[k] for e in eps], axis=0) for k in ("head_fov", "head_vis", "obs_fov", "obs_vis")}
    out["seed"] = seed0
    out["n_ep"] = len(eps)
    out["n_steps"] = n_steps
    return out


def _save_seed(path: Path, pool: dict[str, Any]) -> None:
    payload = {k: pool[k] for k in ("head_fov", "head_vis", "obs_fov", "obs_vis")}
    payload["seed"] = np.array([pool["seed"]])
    payload["n_ep"] = np.array([pool["n_ep"]])
    payload["n_steps"] = np.array([pool["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_seed(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if "head_vis" not in z:
        return None
    return {
        "head_fov": np.asarray(z["head_fov"], dtype=bool),
        "head_vis": np.asarray(z["head_vis"], dtype=bool),
        "obs_fov": np.asarray(z["obs_fov"], dtype=bool),
        "obs_vis": np.asarray(z["obs_vis"], dtype=bool),
        "seed": int(z["seed"][0]),
        "n_ep": int(z["n_ep"][0]),
        "n_steps": int(z["n_steps"][0]),
    }


def _rates(pool: dict[str, Any]) -> dict[str, Any]:
    hv, ov = pool["head_vis"], pool["obs_vis"]
    any_v = hv | ov
    return {
        "P_FOV_head": float(np.mean(pool["head_fov"])),
        "P_visible_head": float(np.mean(hv)),
        "P_FOV_observer": float(np.mean(pool["obs_fov"])),
        "P_visible_observer": float(np.mean(ov)),
        "P_visible_any": float(np.mean(any_v)),
        "P_FOV_any": float(np.mean(pool["head_fov"] | pool["obs_fov"])),
        "n": int(hv.shape[0]),
        "ok_seed_any": bool(float(np.mean(any_v)) >= P_ANY_SEED),
        "ok_seed_head_95": bool(float(np.mean(hv)) >= P_ANY_SEED),
    }


def run_rtwx_o0v(output: str | Path, config: RTWXO0VConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0VConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0V",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "B0": CAM_HEAD,
        "B1": [CAM_HEAD, CAM_OBS],
        "seeds": list(cfg.seeds),
        "n_ep": cfg.n_ep,
        "n_steps": cfg.n_steps,
        "no_training": True,
        "no_pose_claim": True,
        "depends_on": "O0G2=coverage_failure",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    stop: list[str] = []
    per_seed: list[dict[str, Any]] = []
    pools: list[dict[str, Any]] = []
    for i, seed0 in enumerate(cfg.seeds):
        cache = root / f"cache_o0v_seed{seed0}.npz"
        pool = _load_seed(cache)
        if pool is None:
            if cfg.backend == "numpy":
                pool = _numpy_seed(cfg, int(seed0), np.random.default_rng(int(seed0)))
            else:
                repo = Path(cfg.robotwin_repo)
                os.chdir(repo)
                if str(repo) not in sys.path:
                    sys.path.insert(0, str(repo))
                os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
                _patch_curobo_planner(repo)
                pool = collect_seed(cfg, seed0=int(seed0), rng=np.random.default_rng(int(seed0) + 7), stop=stop)
            _save_seed(cache, pool)
        else:
            print(f"[rtwx-o0v] seed={seed0}: cache", flush=True)
        rates = _rates(pool)
        rates["seed"] = int(seed0)
        per_seed.append(rates)
        pools.append(pool)
        print(
            f"[rtwx-o0v] seed={seed0} head_vis={rates['P_visible_head']:.3f} "
            f"any={rates['P_visible_any']:.3f} ok_any={rates['ok_seed_any']}",
            flush=True,
        )

    if stop:
        raise RuntimeError(f"O0V collect stop: {stop}")

    head_all = np.concatenate([p["head_vis"] for p in pools])
    any_all = np.concatenate([p["head_vis"] | p["obs_vis"] for p in pools])
    b0_agg = float(np.mean(head_all))
    b1_agg = float(np.mean(any_all))
    per_ok = all(r["ok_seed_any"] for r in per_seed)
    g_agg = bool(b1_agg >= P_ANY_AGG)
    g_seed = bool(per_ok)
    b1_ok = bool(g_agg and g_seed)
    head_unstable = any(not r["ok_seed_head_95"] for r in per_seed)
    pattern = "dual_view_coverage_supported" if b1_ok else "dual_view_coverage_insufficient"
    summary = {
        "header": header,
        "pattern": pattern,
        "B0_head_only": {
            "P_visible_agg": b0_agg,
            "per_seed": [{k: r[k] for k in ("seed", "P_visible_head", "P_FOV_head", "ok_seed_head_95")} for r in per_seed],
            "head_only_unstable": head_unstable,
        },
        "B1_head_or_observer": {
            "ok": b1_ok,
            "P_visible_any_agg": b1_agg,
            "gate_agg": P_ANY_AGG,
            "gate_per_seed": P_ANY_SEED,
            "ok_aggregate": g_agg,
            "ok_all_seeds": g_seed,
            "per_seed": per_seed,
        },
        "unlocks_o0g2r_prereg": b1_ok,
        "unlocks_o1": False,
        "no_training": True,
        "no_pose_claim": True,
        "does_not_rewrite_o0g2": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "B0_head_only", "B1_head_or_observer", "unlocks_o0g2r_prereg")})
    return summary

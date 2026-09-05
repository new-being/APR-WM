"""RTWX-O0G3 G0: oracle dense correspondence → Kabsch orientation geometry ceiling.

No training. No ICP. Dual-view head+observer. Unlocks O0G3R prereg on PASS. O1 locked.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _e_R_deg, _o0_args, _quat_fix
from .rtwx_o0d import _cup_ids
from .rtwx_o0d1 import _project
from .rtwx_o0d3 import _head_cam
from .rtwx_o0g import MASK_PX
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0c import _R_to_quat
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, P_ANY_SEED, _get_cam
from .rtwx_x0c import FORMAL_N_STEPS, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G3_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g3.oracle_orientation_geometry.v1"
SEEDS = (28601, 28602, 28603)
N_EP = 24
N_STEPS = FORMAL_N_STEPS
N_MIN = 50
MED_ER_MAX = 15.0
P90_ER_MAX = 30.0
MAX_PAIRS = 4000  # subsample for Kabsch speed


@dataclass(frozen=True)
class RTWXO0G3Config:
    output: str = "runs/rtwx_o0g3"
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
        raise RuntimeError("r10_c0 is locked; RTWX-O0G3 must not write there")


def _lock(cfg: RTWXO0G3Config) -> RTWXO0G3Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(cfg, seeds=SEEDS, n_ep=N_EP, n_steps=N_STEPS)


def _kabsch(X_O: np.ndarray, X_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map object-frame points → base: X_B ≈ R @ X_O + t. Returns R(3,3), t(3,), rms."""
    A = np.asarray(X_O, dtype=np.float64)
    B = np.asarray(X_B, dtype=np.float64)
    if A.shape[0] < 3:
        return np.full((3, 3), np.nan), np.full(3, np.nan), float("nan")
    mu_a = A.mean(0)
    mu_b = B.mean(0)
    Ac = A - mu_a
    Bc = B - mu_b
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_b - R @ mu_a
    resid = B - (Ac @ R.T + mu_b)  # equivalent: (R@A.T).T + t
    # clearer residual:
    pred = (R @ A.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - B) ** 2, axis=1))))
    return R, t, rms


def _pairs_from_cam(cam: Any, p: np.ndarray, quat: np.ndarray, cup_ids: set[int], repaired: int | None) -> dict[str, Any]:
    try:
        seg = np.asarray(cam.get_picture("Segmentation"))
        actor = np.asarray(seg[..., 1]).astype(np.int32)
    except Exception:
        actor = np.zeros((480, 640), dtype=np.int32)
    try:
        position = np.asarray(cam.get_picture("Position"))
        model = np.asarray(cam.get_model_matrix(), dtype=np.float64)
    except Exception:
        return {"visible": False, "X_O": np.zeros((0, 3)), "X_B": np.zeros((0, 3)), "repaired": repaired, "n": 0}
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
    mask = (actor == cid) if cid >= 0 else np.zeros_like(actor, dtype=bool)
    area = int(mask.sum())
    visible = bool(in_fov and (area >= MASK_PX or id_uv >= 0))
    if not visible or position is None:
        return {"visible": visible, "X_O": np.zeros((0, 3)), "X_B": np.zeros((0, 3)), "repaired": rep, "n": 0}
    pts = position[..., :3].astype(np.float64)
    world = pts @ model[:3, :3].T + model[:3, 3]
    valid = mask & np.isfinite(world).all(axis=-1)
    if position.shape[-1] >= 4:
        valid = valid & (position[..., 3] < 1.0)
    X_B = world[valid]
    if X_B.shape[0] == 0:
        return {"visible": visible, "X_O": np.zeros((0, 3)), "X_B": np.zeros((0, 3)), "repaired": rep, "n": 0}
    R = _quat_to_R(quat)
    X_O = (X_B - p.reshape(1, 3)) @ R  # row: R.T @ (x-p)
    return {"visible": visible, "X_O": X_O, "X_B": X_B, "repaired": rep, "n": int(X_B.shape[0])}


def _subsample_pairs(X_O: np.ndarray, X_B: np.ndarray, rng: np.random.Generator, cap: int = MAX_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    n = X_O.shape[0]
    if n <= cap:
        return X_O, X_B
    idx = rng.choice(n, size=cap, replace=False)
    return X_O[idx], X_B[idx]


def _numpy_seed(cfg: RTWXO0G3Config, seed: int, rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic non-degenerate clouds; Kabsch should recover GT R."""
    n = int(cfg.n_ep) * int(cfg.n_steps)
    er_f, er_h, er_o, n_pts, vis_h, vis_o, resid, dviews = [], [], [], [], [], [], [], []
    for i in range(n):
        ang = 0.2 * np.sin(i / 11.0 + 0.01 * seed)
        quat = _quat_fix(np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0]))[0]
        R = _quat_to_R(quat)
        p = np.array([-0.2 + 0.01 * (i % 17), -0.05, 0.74])
        X_O = rng.normal(size=(200, 3)) * np.array([0.04, 0.05, 0.04])
        X_B = (R @ X_O.T).T + p
        Rh, th, rmsh = _kabsch(X_O[:120], X_B[:120])
        Ro, to, rmso = _kabsch(X_O[80:], X_B[80:])
        Rf, tf, rmsf = _kabsch(X_O, X_B)
        qh, qo, qf = _R_to_quat(Rh), _R_to_quat(Ro), _R_to_quat(Rf)
        er_h.append(float(_e_R_deg(qh.reshape(1, 4), quat.reshape(1, 4))[0]))
        er_o.append(float(_e_R_deg(qo.reshape(1, 4), quat.reshape(1, 4))[0]))
        er_f.append(float(_e_R_deg(qf.reshape(1, 4), quat.reshape(1, 4))[0]))
        dviews.append(float(_e_R_deg(qh.reshape(1, 4), qo.reshape(1, 4))[0]))
        n_pts.append(200)
        vis_h.append(True)
        vis_o.append(True)
        resid.append(rmsf)
    return {
        "e_R_fusion": np.asarray(er_f),
        "e_R_head": np.asarray(er_h),
        "e_R_obs": np.asarray(er_o),
        "n_pairs": np.asarray(n_pts),
        "vis_h": np.asarray(vis_h, dtype=bool),
        "vis_o": np.asarray(vis_o, dtype=bool),
        "residual_rms": np.asarray(resid),
        "d_view": np.asarray(dviews),
        "n_ep": int(cfg.n_ep),
        "n_steps": int(cfg.n_steps),
        "seed": int(seed),
    }


def collect_seed(cfg: RTWXO0G3Config, *, seed: int, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep = int(cfg.n_ep)
    seed0 = int(seed)
    rows = {k: [] for k in ("e_R_fusion", "e_R_head", "e_R_obs", "n_pairs", "vis_h", "vis_o", "residual_rms", "d_view")}
    eps_ok, attempts, slot = 0, 0, 0
    while eps_ok < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        ep_seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if eps_ok % 4 == 0 or eps_ok + 1 == n_ep:
            print(f"[rtwx-o0g3] seed={seed} valid {eps_ok}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, ep_seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            sim_ids = _cup_ids(env)
            cam_h = _get_cam(env, CAM_HEAD) or _head_cam(env)
            cam_o = _get_cam(env, CAM_OBS)
            if cam_o is None:
                env.close()
                stop.append(f"{seed}:missing_observer")
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
            tt = np.arange(cfg.n_steps) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            invalid = None
            rep_h, rep_o = None, None
            frame_buf = []
            for i in range(int(cfg.n_steps)):
                p, quat = _cup_pose(env)
                env._update_render()
                env.cameras.update_picture()
                ph = _pairs_from_cam(cam_h, p, quat, sim_ids, rep_h)
                try:
                    cam_o.take_picture()
                except Exception:
                    pass
                po = _pairs_from_cam(cam_o, p, quat, sim_ids, rep_o)
                rep_h, rep_o = ph["repaired"], po["repaired"]
                X_O = np.concatenate([ph["X_O"], po["X_O"]], axis=0) if ph["n"] + po["n"] else np.zeros((0, 3))
                X_B = np.concatenate([ph["X_B"], po["X_B"]], axis=0) if ph["n"] + po["n"] else np.zeros((0, 3))
                X_O, X_B = _subsample_pairs(X_O, X_B, rng)
                Rf, tf, rmsf = _kabsch(X_O, X_B)
                qf = _R_to_quat(Rf) if np.isfinite(Rf).all() else np.full(4, np.nan)
                er_f = float(_e_R_deg(qf.reshape(1, 4), quat.reshape(1, 4))[0]) if np.isfinite(qf).all() else float("nan")
                # per-view
                er_h = er_o = d_view = float("nan")
                qh = qo = None
                if ph["n"] >= 3:
                    Rh, _, _ = _kabsch(*_subsample_pairs(ph["X_O"], ph["X_B"], rng))
                    qh = _R_to_quat(Rh) if np.isfinite(Rh).all() else None
                    if qh is not None:
                        er_h = float(_e_R_deg(qh.reshape(1, 4), quat.reshape(1, 4))[0])
                if po["n"] >= 3:
                    Ro, _, _ = _kabsch(*_subsample_pairs(po["X_O"], po["X_B"], rng))
                    qo = _R_to_quat(Ro) if np.isfinite(Ro).all() else None
                    if qo is not None:
                        er_o = float(_e_R_deg(qo.reshape(1, 4), quat.reshape(1, 4))[0])
                if qh is not None and qo is not None:
                    d_view = float(_e_R_deg(qh.reshape(1, 4), qo.reshape(1, 4))[0])
                frame_buf.append(
                    {
                        "e_R_fusion": er_f,
                        "e_R_head": er_h,
                        "e_R_obs": er_o,
                        "n_pairs": int(X_O.shape[0]),
                        "vis_h": bool(ph["visible"]),
                        "vis_o": bool(po["visible"]),
                        "residual_rms": rmsf,
                        "d_view": d_view,
                    }
                )
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
            if invalid or len(frame_buf) != int(cfg.n_steps):
                continue
            for fr in frame_buf:
                for k, v in fr.items():
                    rows[k].append(v)
            eps_ok += 1
        except Exception as exc:
            print(f"[rtwx-o0g3] seed={seed} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if eps_ok < n_ep:
        stop.append(f"{seed}:only_{eps_ok}_of_{n_ep}")
    if eps_ok == 0:
        raise RuntimeError(f"O0G3 collect seed={seed}: 0 episodes")
    out = {k: np.asarray(rows[k]) for k in rows}
    out["vis_h"] = out["vis_h"].astype(bool)
    out["vis_o"] = out["vis_o"].astype(bool)
    out["n_ep"] = eps_ok
    out["n_steps"] = int(cfg.n_steps)
    out["seed"] = int(seed)
    return out


def _score_ori_arr(er: np.ndarray) -> dict[str, float]:
    e = np.asarray(er, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"median_e_R_deg": float("nan"), "p90_e_R_deg": float("nan"), "n": 0, "frac_near_90": float("nan")}
    return {
        "median_e_R_deg": float(np.median(e)),
        "p90_e_R_deg": float(np.percentile(e, 90)),
        "mean_e_R_deg": float(np.mean(e)),
        "n": int(e.size),
        "frac_near_90": float(np.mean((e >= 75.0) & (e <= 105.0))),
    }


def run_rtwx_o0g3(output: str | Path, config: RTWXO0G3Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G3Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0G3",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "method": "oracle_correspondence_Kabsch",
        "x_O": "R_GT.T @ (x_B - p_GT) pose-frame",
        "seeds": list(cfg.seeds),
        "n_ep": cfg.n_ep,
        "n_steps": cfg.n_steps,
        "N_min": N_MIN,
        "unlocks_o1": False,
        "unlocks_o0g3r_prereg": True,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    per_seed = []
    all_er, all_nh, all_no, all_dv, all_np, all_vis_any, all_rms = [], [], [], [], [], [], []
    stop: list[str] = []
    for seed in cfg.seeds:
        cache = root / f"cache_o0g3_s{seed}.npz"
        if cache.is_file():
            z = np.load(cache)
            pool = {k: np.asarray(z[k]) for k in z.files}
            print(f"[rtwx-o0g3] seed={seed} cache", flush=True)
        else:
            rng = np.random.default_rng(int(seed))
            if cfg.backend == "numpy":
                pool = _numpy_seed(cfg, int(seed), rng)
            else:
                repo = Path(cfg.robotwin_repo)
                os.chdir(repo)
                if str(repo) not in sys.path:
                    sys.path.insert(0, str(repo))
                os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
                _patch_curobo_planner(repo)
                pool = collect_seed(cfg, seed=int(seed), rng=rng, stop=stop)
            np.savez_compressed(
                cache,
                **{k: pool[k] for k in ("e_R_fusion", "e_R_head", "e_R_obs", "n_pairs", "vis_h", "vis_o", "residual_rms", "d_view")},
                n_ep=np.array([pool["n_ep"]]),
                n_steps=np.array([pool["n_steps"]]),
                seed=np.array([pool["seed"]]),
            )

        vis_any = pool["vis_h"] | pool["vis_o"]
        enough = pool["n_pairs"] >= N_MIN
        p_any = float(np.mean(vis_any))
        p_enough = float(np.mean(enough & vis_any)) if vis_any.any() else 0.0
        sc = _score_ori_arr(pool["e_R_fusion"])
        cov_ok = bool(p_any >= P_ANY_SEED)
        print(
            f"[rtwx-o0g3] seed={seed} P_any={p_any:.3f} med={sc['median_e_R_deg']:.4f} "
            f"p90={sc['p90_e_R_deg']:.4f} near90={sc['frac_near_90']:.4f}",
            flush=True,
        )
        per_seed.append(
            {
                "seed": int(seed),
                "P_visible_any": p_any,
                "P_enough_pairs": p_enough,
                "cov_ok": cov_ok,
                **sc,
                "median_residual_rms": float(np.nanmedian(pool["residual_rms"])),
                "median_d_view": float(np.nanmedian(pool["d_view"])),
            }
        )
        all_er.append(pool["e_R_fusion"])
        all_nh.append(pool["e_R_head"])
        all_no.append(pool["e_R_obs"])
        all_dv.append(pool["d_view"])
        all_np.append(pool["n_pairs"])
        all_vis_any.append(vis_any)
        all_rms.append(pool["residual_rms"])

    if stop and cfg.backend != "numpy":
        raise RuntimeError(f"O0G3 collect stop: {stop}")

    er = np.concatenate(all_er)
    vis = np.concatenate(all_vis_any)
    npairs = np.concatenate(all_np)
    agg = _score_ori_arr(er)
    p_any_agg = float(np.mean(vis))
    p_enough_agg = float(np.mean((npairs >= N_MIN) & vis))
    g0_cov = bool(p_any_agg >= P_ANY_AGG and all(r["cov_ok"] for r in per_seed) and p_enough_agg >= 0.95)
    g0_ori = bool(
        np.isfinite(agg["median_e_R_deg"])
        and agg["median_e_R_deg"] <= MED_ER_MAX
        and agg["p90_e_R_deg"] <= P90_ER_MAX
    )
    pattern = "oracle_orientation_geometry_supported" if (g0_cov and g0_ori) else "orientation_geometry_insufficient"
    unlock_g3r = pattern == "oracle_orientation_geometry_supported"
    print(f"[rtwx-o0g3] pattern={pattern} near90={agg['frac_near_90']:.4f}", flush=True)

    # secondary disagreement curve on frames with finite d_view
    dv = np.concatenate(all_dv)
    curve = []
    both = np.isfinite(dv) & np.isfinite(er)
    if both.any():
        for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 90), (90, 180)]:
            m = both & (dv >= lo) & (dv < hi if hi < 180 else dv <= hi)
            curve.append(
                {
                    "d_lo": lo,
                    "d_hi": hi,
                    "n": int(m.sum()),
                    "P_ef_gt_30": float(np.mean(er[m] > 30.0)) if m.any() else float("nan"),
                }
            )

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_coverage": {
            "ok": g0_cov,
            "P_visible_any_agg": p_any_agg,
            "P_enough_pairs_agg": p_enough_agg,
            "gate_agg": P_ANY_AGG,
            "gate_seed": P_ANY_SEED,
            "N_min": N_MIN,
        },
        "G0_orientation": {
            "ok": g0_ori,
            **agg,
            "gate": {"med": MED_ER_MAX, "p90": P90_ER_MAX},
        },
        "mode_mass": {"P_eR_in_75_105": agg["frac_near_90"]},
        "secondary": {
            "head": _score_ori_arr(np.concatenate(all_nh)),
            "observer": _score_ori_arr(np.concatenate(all_no)),
            "d_view_curve": curve,
            "median_d_view": float(np.nanmedian(dv)),
            "median_residual_rms": float(np.nanmedian(np.concatenate(all_rms))),
        },
        "per_seed": per_seed,
        "unlocks_o0g3r_prereg": unlock_g3r,
        "unlocks_o1": False,
        "does_not_rewrite_o0c": True,
        "icp_not_primary": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {k: summary[k] for k in ("pattern", "G0_coverage", "G0_orientation", "mode_mass", "secondary", "per_seed", "unlocks_o0g3r_prereg")},
    )
    return summary

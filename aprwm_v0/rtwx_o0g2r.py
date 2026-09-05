"""RTWX-O0G2R: dual-view (head∨observer) geometry-mediated position confirmation.

Frozen O0V cameras + O0G2 U-Net + O0G1b δ_O + per-axis median fusion.
R_BO = GT only (nuisance for reference correction). No RGB→p. Unlocks O0C prereg on PASS.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args
from .rtwx_o0d import _cup_ids, _resize_mask
from .rtwx_o0d1 import _project
from .rtwx_o0d3 import _head_cam
from .rtwx_o0g import MASK_PX
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_o0g2 import (
    EPOCHS,
    INFO_RATIO,
    MED_MAX,
    RGB_SIZE,
    STRONG_MED,
    EP_MAX,
    _iou,
    _pose_from_masks,
    _predict_masks,
    _score_pose,
    _train_unet,
    _xyz64_from_position,
)
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, _get_cam
from .rtwx_x0c import FORMAL_N_STEPS, FORMAL_N_TEST, FORMAL_N_TRAIN, FORMAL_N_VAL, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader, _resize_rgb

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G2R_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g2r.dual_view_geometry_position.v1"
SEED = 26601
ETA_MAX = 1.5
DELTA_E_MAX = 0.05
JOINT_DET_MIN = 0.95


@dataclass(frozen=True)
class RTWXO0G2RConfig:
    output: str = "runs/rtwx_o0g2r"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    epochs: int = EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G2R must not write there")


def _lock(cfg: RTWXO0G2RConfig) -> RTWXO0G2RConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=SEED,
        rgb_size=RGB_SIZE,
        epochs=EPOCHS,
    )


def _capture_pair_rgb(env: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    env._update_render()
    env.cameras.update_picture()
    rgb_map = env.cameras.get_rgb()
    head = rgb_map.get(CAM_HEAD)
    if head is None or head.get("rgb") is None:
        raise RuntimeError(f"missing {CAM_HEAD} rgb")
    obs = env.cameras.get_observer_rgb()
    return _resize_rgb(head["rgb"], size), _resize_rgb(obs, size)


def _cam_pack(cam: Any, p: np.ndarray, cup_ids: set[int], repaired: int | None, size: int) -> dict[str, Any]:
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
    mask_n = (actor == cid) if cid >= 0 else np.zeros_like(actor, dtype=bool)
    area = int(mask_n.sum())
    visible = bool(in_fov and (area >= MASK_PX or id_uv >= 0))
    mask64 = _resize_mask(mask_n, size)
    xyz = _xyz64_from_position(position, model, size) if position is not None and model is not None else np.full((size, size, 3), np.nan, np.float32)
    return {
        "mask": mask64,
        "xyz": xyz,
        "in_fov": in_fov,
        "visible": visible,
        "repaired": rep,
    }


def _fuse_pose(
    masks: list[np.ndarray],
    xyzs: list[np.ndarray],
    quat: np.ndarray,
    delta_O: np.ndarray,
) -> np.ndarray:
    """Union masked points → per-axis median → reference correction. Empty → NaN."""
    chunks = []
    for m, xyz in zip(masks, xyzs):
        mm = np.asarray(m, dtype=bool)
        if not mm.any():
            continue
        pts = np.asarray(xyz, dtype=np.float64)[mm]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] >= 1:
            chunks.append(pts)
    if not chunks:
        return np.full(3, np.nan, dtype=np.float64)
    cloud = np.concatenate(chunks, axis=0)
    if cloud.shape[0] < 8:
        return np.full(3, np.nan, dtype=np.float64)
    surf = np.median(cloud, axis=0)
    return surf - _quat_to_R(quat) @ delta_O


def _batch_fuse(masks_h, masks_o, xyz_h, xyz_o, quat, delta_O) -> np.ndarray:
    n = masks_h.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for i in range(n):
        out[i] = _fuse_pose([masks_h[i], masks_o[i]], [xyz_h[i], xyz_o[i]], quat[i], delta_O)
    return out


def _numpy_split(cfg: RTWXO0G2RConfig, split: str, rng: np.random.Generator, delta_O: np.ndarray) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    size = int(cfg.rgb_size)
    quat0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    rows: dict[str, list] = {k: [] for k in (
        "p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o"
    )}
    for ep in range(n_ep):
        p0 = np.array([-0.25 + 0.5 * (ep / max(n_ep - 1, 1)), rng.uniform(-0.15, 0.05), 0.74])
        for t in range(int(cfg.n_steps)):
            p = p0 + np.array([0.001 * np.sin(t / 10.0), 0.0, 0.0])
            # head always sees; observer always sees in numpy
            def blob(u0, v0, color):
                rgb = np.full((size, size, 3), 18, dtype=np.uint8)
                mask = np.zeros((size, size), dtype=bool)
                r = max(3, size // 12)
                mask[v0 - r : v0 + r + 1, u0 - r : u0 + r + 1] = True
                rgb[mask] = color
                rgb[0, (ep + t) % size, 2] = 50 + ((ep * t) % 180)
                xyz = np.full((size, size, 3), np.nan, np.float32)
                surf = p + _quat_to_R(quat0) @ delta_O
                ys, xs = np.where(mask)
                for yy, xx in zip(ys, xs):
                    xyz[yy, xx] = surf + np.array([(xx - u0) * 0.001, (yy - v0) * 0.001, 0.0])
                return rgb, mask, xyz

            uh = int(np.clip((p[0] + 0.35) / 0.70 * (size - 1), 4, size - 5))
            vh = int(np.clip((p[1] + 0.25) / 0.50 * (size - 1), 4, size - 5))
            uo = int(np.clip(size - 1 - uh, 4, size - 5))
            vo = int(np.clip(vh + 2, 4, size - 5))
            rh, mh, xh = blob(uh, vh, (210, 80, 40))
            ro, mo, xo = blob(uo, vo, (200, 70, 35))
            rows["p"].append(p)
            rows["quat"].append(quat0)
            rows["rgb_h"].append(rh)
            rows["rgb_o"].append(ro)
            rows["mask_h"].append(mh)
            rows["mask_o"].append(mo)
            rows["xyz_h"].append(xh)
            rows["xyz_o"].append(xo)
            rows["vis_h"].append(True)
            rows["vis_o"].append(True)
    return {
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "mask_h": np.stack(rows["mask_h"]),
        "mask_o": np.stack(rows["mask_o"]),
        "xyz_h": np.stack(rows["xyz_h"]),
        "xyz_o": np.stack(rows["xyz_o"]),
        "vis_h": np.asarray(rows["vis_h"]),
        "vis_o": np.asarray(rows["vis_o"]),
        "n_ep": n_ep,
        "n_steps": int(cfg.n_steps),
    }


def collect_split(cfg: RTWXO0G2RConfig, *, split: str, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = cfg.seed + {"train": 0, "val": 10_000, "test": 20_000}[split]
    size = int(cfg.rgb_size)
    eps: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(eps) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(eps) % 4 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0g2r] {split} valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
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
                stop.append(f"{split}:missing_observer")
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
            keys = ("p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o")
            rows: dict[str, list] = {k: [] for k in keys}
            invalid = None
            rep_h, rep_o = None, None
            for i in range(int(cfg.n_steps)):
                p, quat = _cup_pose(env)
                rgb_h, rgb_o = _capture_pair_rgb(env, size)
                # observer already pictured in get_observer_rgb; refresh head packs from update_picture
                ph = _cam_pack(cam_h, p, sim_ids, rep_h, size)
                # observer needs take_picture again for seg/position after Color
                try:
                    cam_o.take_picture()
                except Exception:
                    pass
                po = _cam_pack(cam_o, p, sim_ids, rep_o, size)
                rep_h, rep_o = ph["repaired"], po["repaired"]
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["rgb_h"].append(rgb_h)
                rows["rgb_o"].append(rgb_o)
                rows["mask_h"].append(ph["mask"])
                rows["mask_o"].append(po["mask"])
                rows["xyz_h"].append(ph["xyz"])
                rows["xyz_o"].append(po["xyz"])
                rows["vis_h"].append(ph["visible"])
                rows["vis_o"].append(po["visible"])
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
            if invalid or len(rows["p"]) != int(cfg.n_steps):
                continue
            eps.append(
                {
                    "p": np.asarray(rows["p"], dtype=np.float64),
                    "quat": np.asarray(rows["quat"], dtype=np.float64),
                    "rgb_h": np.stack(rows["rgb_h"]),
                    "rgb_o": np.stack(rows["rgb_o"]),
                    "mask_h": np.stack(rows["mask_h"]),
                    "mask_o": np.stack(rows["mask_o"]),
                    "xyz_h": np.stack(rows["xyz_h"]),
                    "xyz_o": np.stack(rows["xyz_o"]),
                    "vis_h": np.asarray(rows["vis_h"], dtype=bool),
                    "vis_o": np.asarray(rows["vis_o"], dtype=bool),
                }
            )
        except Exception as exc:
            print(f"[rtwx-o0g2r] {split} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0G2R collect {split}: 0 episodes")
    keys = ("p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o")
    out: dict[str, Any] = {k: np.concatenate([e[k] for e in eps], axis=0) for k in keys}
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    return out


def _save(path: Path, name: str, p: dict[str, Any]) -> None:
    keys = ("p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o")
    payload = {f"{name}_{k}": p[k] for k in keys}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    np.savez_compressed(path, **payload)


def _load(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    keys = ("p", "quat", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o")
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in keys}
    p["rgb_h"] = np.asarray(z[f"{name}_rgb_h"], dtype=np.uint8)
    p["rgb_o"] = np.asarray(z[f"{name}_rgb_o"], dtype=np.uint8)
    p["mask_h"] = np.asarray(p["mask_h"], dtype=bool)
    p["mask_o"] = np.asarray(p["mask_o"], dtype=bool)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect(cfg: RTWXO0G2RConfig, root: Path, delta_O: np.ndarray) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(cfg.seed),
        "val": np.random.default_rng(cfg.seed + 1),
        "test": np.random.default_rng(cfg.seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0g2r_{name}.npz"
        hit = _load(part, name)
        if hit is not None:
            print(f"[rtwx-o0g2r] {name}: cache", flush=True)
            splits[name] = hit
            continue
        if cfg.backend == "numpy":
            pool = _numpy_split(cfg, name, rng, delta_O)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_split(cfg, split=name, rng=rng, stop=stop)
        splits[name] = pool
        _save(part, name, pool)
    if stop:
        raise RuntimeError(f"O0G2R collect stop: {stop}")
    return splits


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "dual_view_coverage_failure"
    if not g1:
        return "oracle_fusion_geometry_failure"
    if not g2:
        return "dual_view_localization_failure"
    if not g3 or not g4:
        return "dual_view_position_failure"
    return "dual_view_geometry_position_supported"


def run_rtwx_o0g2r(output: str | Path, config: RTWXO0G2RConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G2RConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    header = {
        "stage": "RTWX-O0G2R",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "rgb_size": RGB_SIZE,
        "encoder": "small_UNet_shared",
        "fusion": "union_masked_points_per_axis_median",
        "R_BO": "GT_nuisance_for_delta_O_only",
        "seed": cfg.seed,
        "delta_O": delta_O.tolist(),
        "depends_on": "O0V=dual_view_coverage_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    splits = _collect(cfg, root, delta_O)
    train, val, test = splits["train"], splits["val"], splits["test"]

    vis_any = test["vis_h"] | test["vis_o"]
    p_any = float(np.mean(vis_any))
    g0 = bool(p_any >= P_ANY_AGG)
    print(f"[rtwx-o0g2r] G0 P_any={p_any:.3f} ok={g0}", flush=True)

    b0 = _score_pose(np.repeat(train["p"].mean(0, keepdims=True), test["p"].shape[0], 0), test["p"])
    print(f"[rtwx-o0g2r] B0 E_p={b0['E_p']:.4f} med={b0['median_ep_m']:.4f}", flush=True)

    p_b1 = _batch_fuse(test["mask_h"], test["mask_o"], test["xyz_h"], test["xyz_o"], test["quat"], delta_O)
    b1 = _score_pose(p_b1, test["p"])
    g1 = bool(b1["E_p"] <= EP_MAX and b1["median_ep_m"] <= MED_MAX)
    b1_strong = bool(g1 and b1["median_ep_m"] <= STRONG_MED)
    print(f"[rtwx-o0g2r] B1 E_p={b1['E_p']:.4f} med={b1['median_ep_m']:.4f} ok={g1} strong={b1_strong}", flush=True)

    # shared U-Net on pooled cameras
    rgb_tr = np.concatenate([train["rgb_h"], train["rgb_o"]], axis=0)
    m_tr = np.concatenate([train["mask_h"], train["mask_o"]], axis=0)
    rgb_va = np.concatenate([val["rgb_h"], val["rgb_o"]], axis=0)
    m_va = np.concatenate([val["mask_h"], val["mask_o"]], axis=0)
    epochs = 4 if cfg.smoke else cfg.epochs
    model = _train_unet(rgb_tr, m_tr, rgb_va, m_va, seed=cfg.seed, epochs=epochs)
    pred_h = _predict_masks(model, test["rgb_h"])
    pred_o = _predict_masks(model, test["rgb_o"])

    ious_h = [_iou(pred_h[i], test["mask_h"][i]) for i in range(pred_h.shape[0])]
    ious_o = [_iou(pred_o[i], test["mask_o"][i]) for i in range(pred_o.shape[0])]
    det_any = pred_h.any(axis=(1, 2)) | pred_o.any(axis=(1, 2))
    joint = float(np.mean(det_any[vis_any])) if vis_any.any() else 0.0
    g2 = bool(joint >= JOINT_DET_MIN)
    loc = {
        "median_iou_head": float(np.median(ious_h)),
        "median_iou_observer": float(np.median(ious_o)),
        "P_det_any_given_Vany": joint,
        "ok": g2,
        "gate": {"joint_det_min": JOINT_DET_MIN},
    }
    print(
        f"[rtwx-o0g2r] G2 IoU_h={loc['median_iou_head']:.3f} IoU_o={loc['median_iou_observer']:.3f} "
        f"joint_det={joint:.3f} ok={g2}",
        flush=True,
    )

    p_b2 = _batch_fuse(pred_h, pred_o, test["xyz_h"], test["xyz_o"], test["quat"], delta_O)
    b2 = _score_pose(p_b2, test["p"])
    g3 = bool(b2["E_p"] <= EP_MAX and b2["median_ep_m"] <= MED_MAX)
    b2_strong = bool(g3 and b2["median_ep_m"] <= STRONG_MED)
    g4 = bool(b2["E_p"] <= INFO_RATIO * b0["E_p"]) if np.isfinite(b0["E_p"]) else False
    eta = float(b2["E_p"] / b1["E_p"]) if np.isfinite(b2["E_p"]) and b1["E_p"] > 1e-12 else float("nan")
    d_loc = float(b2["E_p"] - b1["E_p"]) if np.isfinite(b2["E_p"]) and np.isfinite(b1["E_p"]) else float("nan")
    g5 = bool((np.isfinite(eta) and eta <= ETA_MAX) or (np.isfinite(d_loc) and d_loc <= DELTA_E_MAX))
    print(
        f"[rtwx-o0g2r] B2 E_p={b2['E_p']:.4f} med={b2['median_ep_m']:.4f} g3={g3} g4={g4} "
        f"eta={eta:.3f} g5={g5}",
        flush=True,
    )

    # B3 head-only learned (descriptive)
    p_b3 = _pose_from_masks(test["xyz_h"], pred_h, test["quat"], delta_O)
    b3 = _score_pose(p_b3, test["p"])

    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3, g4=g4)
    unlock_o0c = pattern == "dual_view_geometry_position_supported"
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": g0, "P_visible_any": p_any, "gate": P_ANY_AGG},
        "G1_oracle_fusion": {"ok": g1, "strong": b1_strong, **b1},
        "G2_localization": loc,
        "G3_pose": {"ok": g3, "strong": b2_strong, **b2},
        "G4_info": {"ok": g4, "ratio_to_B0": float(b2["E_p"] / max(b0["E_p"], 1e-12)), "INFO_RATIO": INFO_RATIO},
        "G5_near_oracle": {"ok": g5, "eta_loc": eta, "Delta_E_loc": d_loc, "localization_near_oracle": g5},
        "B0_mean": b0,
        "B1_oracle_dual": b1,
        "B2_learned_dual": b2,
        "B3_learned_head_only_secondary": {**b3, "does_not_override_primary": True},
        "unlocks_o0c_prereg": unlock_o0c,
        "unlocks_o1": False,
        "R_BO_is_GT_nuisance": True,
        "does_not_rewrite_o0v": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            k: summary[k]
            for k in (
                "pattern",
                "G0",
                "G1_oracle_fusion",
                "G2_localization",
                "G3_pose",
                "G4_info",
                "G5_near_oracle",
                "B0_mean",
                "B1_oracle_dual",
                "B2_learned_dual",
                "B3_learned_head_only_secondary",
                "unlocks_o0c_prereg",
            )
        },
    )
    return summary

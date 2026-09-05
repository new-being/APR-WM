"""RTWX-O0C: composite non-oracle object pose T_BO=(R_BO, p_BO).

Dual-view U-Net position (O0G2R) + CoordConv camera-frame orientation fused by SO(3)
chordal mean. Reference correction uses learned R̂ (no R_BO^GT). Unlocks O1 on PASS.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _e_R_deg, _o0_args, _quat_fix
from .rtwx_o0d import _cup_ids, _resize_mask
from .rtwx_o0d1 import _project
from .rtwx_o0d3 import _coord_net, _head_cam, _nchw
from .rtwx_o0g import MASK_PX, _as_T44
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_o0g2 import (
    EPOCHS as UNET_EPOCHS,
    INFO_RATIO,
    MED_MAX,
    RGB_SIZE,
    STRONG_MED,
    EP_MAX,
    _iou,
    _predict_masks,
    _score_pose,
    _train_unet,
    _xyz64_from_position,
)
from .rtwx_o0g2r import _batch_fuse, _capture_pair_rgb
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX, EPOCHS as ORI_EPOCHS, PATIENCE as ORI_PATIENCE, LR as ORI_LR
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, _get_cam
from .rtwx_x0c import FORMAL_N_STEPS, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0C_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0c.composite_non_oracle_object_pose.v1"
SEEDS = (27601, 27602, 27603)
SEED_PRIMARY = SEEDS[0]
N_TRAIN_EP = 24
N_VAL_EP = 12
N_TEST_EP = 12
P_SEED_ANY = 0.95
JOINT_DET_MIN = 0.95
ETA_POSE_MAX = 1.5
DELTA_E_MAX = 0.05
ORI_OUT = 4


@dataclass(frozen=True)
class RTWXO0CConfig:
    output: str = "runs/rtwx_o0c"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = N_TRAIN_EP
    n_val_ep: int = N_VAL_EP
    n_test_ep: int = N_TEST_EP
    n_steps: int = FORMAL_N_STEPS
    seeds: tuple[int, ...] = SEEDS
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    epochs_unet: int = UNET_EPOCHS
    epochs_ori: int = ORI_EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0C must not write there")


def _lock(cfg: RTWXO0CConfig) -> RTWXO0CConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=N_TRAIN_EP,
        n_val_ep=N_VAL_EP,
        n_test_ep=N_TEST_EP,
        n_steps=FORMAL_N_STEPS,
        seeds=SEEDS,
        rgb_size=RGB_SIZE,
        epochs_unet=UNET_EPOCHS,
        epochs_ori=ORI_EPOCHS,
    )


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → Sapien wxyz quaternion."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return _quat_fix(np.array([w, x, y, z], dtype=np.float64))[0]


def _R_BC_from_extrinsic(E: np.ndarray) -> np.ndarray:
    t_cw = _as_T44(E)
    return np.linalg.inv(t_cw)[:3, :3].astype(np.float64)


def _quat_CO_from_BO(quat_BO: np.ndarray, R_BC: np.ndarray) -> np.ndarray:
    return _R_to_quat(R_BC.T @ _quat_to_R(quat_BO))


def _quat_BO_from_CO(quat_CO: np.ndarray, R_BC: np.ndarray) -> np.ndarray:
    return _R_to_quat(R_BC @ _quat_to_R(quat_CO))


def _geomedian_R(Rs: list[np.ndarray]) -> np.ndarray:
    """Equal-weight SO(3) chordal mean (SVD); GeoMedian for |V|<=2 equal weights."""
    if not Rs:
        return np.eye(3, dtype=np.float64)
    if len(Rs) == 1:
        return np.asarray(Rs[0], dtype=np.float64)
    M = sum(np.asarray(R, dtype=np.float64) for R in Rs)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _fuse_R_BO(quat_CO_h, quat_CO_o, R_BC_h, R_BC_o, valid_h: bool, valid_o: bool) -> np.ndarray:
    Rs = []
    if valid_h:
        Rs.append(_quat_to_R(_quat_BO_from_CO(quat_CO_h, R_BC_h)))
    if valid_o:
        Rs.append(_quat_to_R(_quat_BO_from_CO(quat_CO_o, R_BC_o)))
    if not Rs:
        return np.full(4, np.nan, dtype=np.float64)
    return _R_to_quat(_geomedian_R(Rs))


KEYS = (
    "p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o",
    "vis_h", "vis_o", "R_BC_h", "R_BC_o", "quat_CO_h", "quat_CO_o",
)


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
    R_BC = _R_BC_from_extrinsic(e)
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
    xyz = (
        _xyz64_from_position(position, model, size)
        if position is not None and model is not None
        else np.full((size, size, 3), np.nan, np.float32)
    )
    return {"mask": mask64, "xyz": xyz, "in_fov": in_fov, "visible": visible, "repaired": rep, "R_BC": R_BC}


def _numpy_split(cfg: RTWXO0CConfig, split: str, rng: np.random.Generator, delta_O: np.ndarray, seed: int) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    size = int(cfg.rgb_size)
    R_BC = np.eye(3, dtype=np.float64)
    rows: dict[str, list] = {k: [] for k in KEYS}
    for ep in range(n_ep):
        p0 = np.array([-0.25 + 0.5 * (ep / max(n_ep - 1, 1)), rng.uniform(-0.15, 0.05), 0.74])
        for t in range(int(cfg.n_steps)):
            p = p0 + np.array([0.001 * np.sin(t / 10.0), 0.0, 0.0])
            ang = 0.15 * np.sin(t / 8.0 + ep + 0.01 * seed)
            quat = _quat_fix(np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)]))[0]
            q_co = _quat_CO_from_BO(quat, R_BC)

            def blob(u0, v0, color):
                rgb = np.full((size, size, 3), 18, dtype=np.uint8)
                mask = np.zeros((size, size), dtype=bool)
                r = max(3, size // 12)
                mask[v0 - r : v0 + r + 1, u0 - r : u0 + r + 1] = True
                rgb[mask] = color
                rgb[1, (ep + t) % size, 0] = int(np.clip(128 + 80 * np.sin(ang), 0, 255))
                rgb[0, (ep + t) % size, 2] = 50 + ((ep * t + seed) % 180)
                xyz = np.full((size, size, 3), np.nan, np.float32)
                surf = p + _quat_to_R(quat) @ delta_O
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
            rows["quat"].append(quat)
            rows["rgb_h"].append(rh)
            rows["rgb_o"].append(ro)
            rows["mask_h"].append(mh)
            rows["mask_o"].append(mo)
            rows["xyz_h"].append(xh)
            rows["xyz_o"].append(xo)
            rows["vis_h"].append(True)
            rows["vis_o"].append(True)
            rows["R_BC_h"].append(R_BC)
            rows["R_BC_o"].append(R_BC)
            rows["quat_CO_h"].append(q_co)
            rows["quat_CO_o"].append(q_co)
    out = {k: np.stack(rows[k]) for k in KEYS}
    out["n_ep"] = n_ep
    out["n_steps"] = int(cfg.n_steps)
    out["seed"] = int(seed)
    return out


def collect_split(
    cfg: RTWXO0CConfig, *, split: str, seed: int, rng: np.random.Generator, stop: list[str]
) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = int(seed) + {"train": 0, "val": 10_000, "test": 20_000}[split]
    size = int(cfg.rgb_size)
    eps: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(eps) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        ep_seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(eps) % 4 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0c] seed={seed} {split} valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
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
                stop.append(f"{seed}:{split}:missing_observer")
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
            rows: dict[str, list] = {k: [] for k in KEYS}
            invalid = None
            rep_h, rep_o = None, None
            for i in range(int(cfg.n_steps)):
                p, quat = _cup_pose(env)
                rgb_h, rgb_o = _capture_pair_rgb(env, size)
                ph = _cam_pack(cam_h, p, sim_ids, rep_h, size)
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
                rows["R_BC_h"].append(ph["R_BC"])
                rows["R_BC_o"].append(po["R_BC"])
                rows["quat_CO_h"].append(_quat_CO_from_BO(quat, ph["R_BC"]))
                rows["quat_CO_o"].append(_quat_CO_from_BO(quat, po["R_BC"]))
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
            eps.append({k: np.asarray(rows[k]) for k in KEYS})
        except Exception as exc:
            print(f"[rtwx-o0c] seed={seed} {split} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"{seed}:{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0C collect seed={seed} {split}: 0 episodes")
    out: dict[str, Any] = {k: np.concatenate([e[k] for e in eps], axis=0) for k in KEYS}
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    out["seed"] = int(seed)
    return out


def _save(path: Path, name: str, p: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": p[k] for k in KEYS}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    payload[f"{name}_seed"] = np.array([p["seed"]])
    np.savez_compressed(path, **payload)


def _load(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in KEYS}
    p["rgb_h"] = np.asarray(p["rgb_h"], dtype=np.uint8)
    p["rgb_o"] = np.asarray(p["rgb_o"], dtype=np.uint8)
    p["mask_h"] = np.asarray(p["mask_h"], dtype=bool)
    p["mask_o"] = np.asarray(p["mask_o"], dtype=bool)
    p["vis_h"] = np.asarray(p["vis_h"], dtype=bool)
    p["vis_o"] = np.asarray(p["vis_o"], dtype=bool)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    p["seed"] = int(z[f"{name}_seed"][0])
    return p


def _collect_seed(cfg: RTWXO0CConfig, root: Path, delta_O: np.ndarray, seed: int) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(seed),
        "val": np.random.default_rng(seed + 1),
        "test": np.random.default_rng(seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0c_s{seed}_{name}.npz"
        hit = _load(part, name)
        if hit is not None:
            print(f"[rtwx-o0c] seed={seed} {name}: cache", flush=True)
            splits[name] = hit
            continue
        if cfg.backend == "numpy":
            pool = _numpy_split(cfg, name, rng, delta_O, seed)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_split(cfg, split=name, seed=seed, rng=rng, stop=stop)
        splits[name] = pool
        _save(part, name, pool)
    if stop:
        raise RuntimeError(f"O0C collect stop: {stop}")
    return splits


def _concat_pools(pools: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {k: np.concatenate([p[k] for p in pools], axis=0) for k in KEYS}
    out["n_ep"] = int(sum(p["n_ep"] for p in pools))
    out["n_steps"] = int(pools[0]["n_steps"])
    out["seed"] = -1
    return out


def _train_ori(model: Any, rgb: np.ndarray, quat_CO: np.ndarray, rgb_va: np.ndarray, quat_va: np.ndarray, *, seed: int, epochs: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    mu = quat_CO.mean(0)
    sd = quat_CO.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    y_tr = ((quat_CO - mu) / sd).astype(np.float32)
    y_va = ((quat_va - mu) / sd).astype(np.float32)
    x_tr = _nchw(rgb)
    x_va = _nchw(rgb_va)
    tr = [(x_tr[i : i + 16], y_tr[i : i + 16]) for i in range(0, rgb.shape[0], 16)]
    va = [(x_va[i : i + 16], y_va[i : i + 16]) for i in range(0, rgb_va.shape[0], 16)]
    opt = torch.optim.Adam(model.parameters(), lr=ORI_LR)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(tr))
        for i in order:
            xb, yb = tr[int(i)]
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(xb).to(dev)), torch.from_numpy(yb).float().to(dev)).backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va:
                vals.append(float(loss_fn(model(torch.from_numpy(xb).to(dev)), torch.from_numpy(yb).float().to(dev))))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= ORI_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model._o0c_mu = mu
    model._o0c_sd = sd
    return model


def _predict_quat_CO(model: Any, rgb: np.ndarray) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    mu = np.asarray(model._o0c_mu, dtype=np.float64)
    sd = np.asarray(model._o0c_sd, dtype=np.float64)
    x = _nchw(rgb)
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            rows.append(model(torch.from_numpy(x[i : i + 16]).to(dev)).cpu().numpy())
    y = np.concatenate(rows, axis=0)
    return _quat_fix(y * sd + mu)


def _batch_fuse_R(pool: dict[str, Any], quat_CO_h: np.ndarray, quat_CO_o: np.ndarray) -> np.ndarray:
    n = pool["p"].shape[0]
    out = np.full((n, 4), np.nan, dtype=np.float64)
    for i in range(n):
        out[i] = _fuse_R_BO(
            quat_CO_h[i], quat_CO_o[i], pool["R_BC_h"][i], pool["R_BC_o"][i],
            bool(pool["vis_h"][i]), bool(pool["vis_o"][i]),
        )
    return out


def _score_ori(qhat: np.ndarray, q: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(qhat).all(axis=1) & np.isfinite(q).all(axis=1)
    if not ok.any():
        return {"median_e_R_deg": float("nan"), "p90_e_R_deg": float("nan"), "n_finite": 0, "frac_finite": 0.0}
    er = _e_R_deg(qhat[ok], q[ok])
    return {
        "median_e_R_deg": float(np.median(er)),
        "p90_e_R_deg": float(np.percentile(er, 90)),
        "mean_e_R_deg": float(np.mean(er)),
        "n_finite": int(ok.sum()),
        "frac_finite": float(ok.mean()),
    }


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool) -> str:
    if not g0:
        return "coverage_failure"
    if not g1:
        return "orientation_failure"
    if not g2:
        return "orientation_position_interface_failure"
    if not g3:
        return "composite_pose_failure"
    return "composite_object_pose_supported"


def run_rtwx_o0c(output: str | Path, config: RTWXO0CConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0CConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    header = {
        "stage": "RTWX-O0C",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "rgb_size": RGB_SIZE,
        "state": "s^O,pose = T_BO = (R_BO, p_BO)",
        "R_BO": "learned_non_oracle",
        "fusion_R": "SO3_chordal_SVD_geomedian",
        "fusion_p": "union_masked_points_per_axis_median",
        "seeds": list(cfg.seeds),
        "delta_O": delta_O.tolist(),
        "depends_on": "O0G2R=dual_view_geometry_position_supported",
        "unlocks_o1": True,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    per_seed_splits: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in cfg.seeds:
        print(f"[rtwx-o0c] === collect seed {seed} ===", flush=True)
        per_seed_splits[int(seed)] = _collect_seed(cfg, root, delta_O, int(seed))

    train = _concat_pools([per_seed_splits[s]["train"] for s in cfg.seeds])
    val = _concat_pools([per_seed_splits[s]["val"] for s in cfg.seeds])
    test = _concat_pools([per_seed_splits[s]["test"] for s in cfg.seeds])

    rgb_tr = np.concatenate([train["rgb_h"], train["rgb_o"]], axis=0)
    m_tr = np.concatenate([train["mask_h"], train["mask_o"]], axis=0)
    rgb_va = np.concatenate([val["rgb_h"], val["rgb_o"]], axis=0)
    m_va = np.concatenate([val["mask_h"], val["mask_o"]], axis=0)
    ep_u = 4 if cfg.smoke else cfg.epochs_unet
    # Collection (Sapien/Vulkan) can leave GPU fragmented; reclaim before torch.
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[rtwx-o0c] cuda free={torch.cuda.mem_get_info()[0]/1e9:.2f}GB", flush=True)
    except Exception as exc:
        print(f"[rtwx-o0c] cuda reclaim skip: {exc}", flush=True)
    print(f"[rtwx-o0c] train U-Net epochs={ep_u}", flush=True)
    unet = _train_unet(rgb_tr, m_tr, rgb_va, m_va, seed=SEED_PRIMARY, epochs=ep_u)

    q_tr = np.concatenate([train["quat_CO_h"], train["quat_CO_o"]], axis=0)
    q_va = np.concatenate([val["quat_CO_h"], val["quat_CO_o"]], axis=0)
    ep_o = 4 if cfg.smoke else cfg.epochs_ori
    print(f"[rtwx-o0c] train Ori CoordConv epochs={ep_o}", flush=True)
    ori_net = _coord_net(ORI_OUT, int(cfg.rgb_size))
    ori_net = _train_ori(ori_net, rgb_tr, q_tr, rgb_va, q_va, seed=SEED_PRIMARY + 7, epochs=ep_o)

    def eval_pool(pool: dict[str, Any], *, label: str) -> dict[str, Any]:
        vis_any = pool["vis_h"] | pool["vis_o"]
        p_any = float(np.mean(vis_any))
        pred_h = _predict_masks(unet, pool["rgb_h"])
        pred_o = _predict_masks(unet, pool["rgb_o"])
        det_any = pred_h.any(axis=(1, 2)) | pred_o.any(axis=(1, 2))
        joint = float(np.mean(det_any[vis_any])) if vis_any.any() else 0.0
        pred_co_h = _predict_quat_CO(ori_net, pool["rgb_h"])
        pred_co_o = _predict_quat_CO(ori_net, pool["rgb_o"])
        q_hat = _batch_fuse_R(pool, pred_co_h, pred_co_o)

        b0_p = np.repeat(train["p"].mean(0, keepdims=True), pool["p"].shape[0], 0)
        b0_q = np.repeat(_quat_fix(train["quat"].mean(0, keepdims=True)), pool["p"].shape[0], 0)
        s_b0_p = _score_pose(b0_p, pool["p"])
        s_b0_r = _score_ori(b0_q, pool["quat"])

        p_b1 = _batch_fuse(pool["mask_h"], pool["mask_o"], pool["xyz_h"], pool["xyz_o"], pool["quat"], delta_O)
        s_b1 = _score_pose(p_b1, pool["p"])

        p_b2 = _batch_fuse(pool["mask_h"], pool["mask_o"], pool["xyz_h"], pool["xyz_o"], q_hat, delta_O)
        s_b2_p = _score_pose(p_b2, pool["p"])
        s_b2_r = _score_ori(q_hat, pool["quat"])

        p_b3 = _batch_fuse(pred_h, pred_o, pool["xyz_h"], pool["xyz_o"], q_hat, delta_O)
        s_b3_p = _score_pose(p_b3, pool["p"])
        s_b3_r = _score_ori(q_hat, pool["quat"])

        d_er = float(s_b2_p["E_p"] - s_b1["E_p"]) if np.isfinite(s_b2_p["E_p"]) and np.isfinite(s_b1["E_p"]) else float("nan")
        eta = float(s_b3_p["E_p"] / s_b1["E_p"]) if np.isfinite(s_b3_p["E_p"]) and s_b1["E_p"] > 1e-12 else float("nan")
        d_pose = float(s_b3_p["E_p"] - s_b1["E_p"]) if np.isfinite(s_b3_p["E_p"]) and np.isfinite(s_b1["E_p"]) else float("nan")

        print(
            f"[rtwx-o0c] {label} P_any={p_any:.3f} joint={joint:.3f} "
            f"eR_med={s_b3_r['median_e_R_deg']:.2f} Ep_B3={s_b3_p['E_p']:.4f} med={s_b3_p['median_ep_m']:.4f}",
            flush=True,
        )
        return {
            "P_visible_any": p_any,
            "joint_det": joint,
            "iou_h": float(np.median([_iou(pred_h[i], pool["mask_h"][i]) for i in range(pred_h.shape[0])])),
            "iou_o": float(np.median([_iou(pred_o[i], pool["mask_o"][i]) for i in range(pred_o.shape[0])])),
            "B0_pose": s_b0_p,
            "B0_ori": s_b0_r,
            "B1_GT_R": s_b1,
            "B2_learned_R_GT_mask": {**s_b2_p, "ori": s_b2_r, "Delta_E_R": d_er},
            "B3_fully_learned": {**s_b3_p, "ori": s_b3_r},
            "eta_pose": eta,
            "Delta_E_pose": d_pose,
            "info_ratio": float(s_b3_p["E_p"] / max(s_b0_p["E_p"], 1e-12)),
            "ori_beats_mean": bool(
                np.isfinite(s_b3_r["median_e_R_deg"])
                and np.isfinite(s_b0_r["median_e_R_deg"])
                and s_b3_r["median_e_R_deg"] < s_b0_r["median_e_R_deg"]
            ),
        }

    agg = eval_pool(test, label="AGG")
    per_seed: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        ev = eval_pool(per_seed_splits[int(seed)]["test"], label=f"seed={seed}")
        pool = per_seed_splits[int(seed)]["test"]
        tr_s = per_seed_splits[int(seed)]["train"]
        b0s = _score_pose(np.repeat(tr_s["p"].mean(0, keepdims=True), pool["p"].shape[0], 0), pool["p"])
        collapse_ok = bool(ev["B3_fully_learned"]["E_p"] <= INFO_RATIO * b0s["E_p"])
        cov_ok = bool(ev["P_visible_any"] >= P_SEED_ANY)
        per_seed.append(
            {
                "seed": int(seed),
                "P_visible_any": ev["P_visible_any"],
                "cov_ok": cov_ok,
                "E_p_B3": ev["B3_fully_learned"]["E_p"],
                "E_p_B0_seed": b0s["E_p"],
                "no_collapse": collapse_ok,
                "median_e_R_deg": ev["B3_fully_learned"]["ori"]["median_e_R_deg"],
                "median_ep_m": ev["B3_fully_learned"]["median_ep_m"],
            }
        )

    g0_agg = bool(agg["P_visible_any"] >= P_ANY_AGG and agg["joint_det"] >= JOINT_DET_MIN)
    g0_seeds = bool(all(r["cov_ok"] for r in per_seed))
    g0 = bool(g0_agg and g0_seeds)

    b3 = agg["B3_fully_learned"]
    ori = b3["ori"]
    g1 = bool(
        np.isfinite(ori["median_e_R_deg"])
        and ori["median_e_R_deg"] <= MED_ER_MAX
        and ori["p90_e_R_deg"] <= P90_ER_MAX
    )
    g2 = bool(b3["E_p"] <= EP_MAX and b3["median_ep_m"] <= MED_MAX)
    g2_strong = bool(g2 and b3["median_ep_m"] <= STRONG_MED)
    g3 = bool(agg["info_ratio"] <= INFO_RATIO and agg["ori_beats_mean"] and all(r["no_collapse"] for r in per_seed))
    g4 = bool(
        (np.isfinite(agg["eta_pose"]) and agg["eta_pose"] <= ETA_POSE_MAX)
        or (np.isfinite(agg["Delta_E_pose"]) and agg["Delta_E_pose"] <= DELTA_E_MAX)
    )

    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3)
    unlock_o1 = pattern == "composite_object_pose_supported"
    print(f"[rtwx-o0c] pattern={pattern} unlocks_o1={unlock_o1}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0": {
            "ok": g0,
            "P_visible_any_agg": agg["P_visible_any"],
            "joint_det_agg": agg["joint_det"],
            "ok_aggregate": g0_agg,
            "ok_all_seeds": g0_seeds,
            "gate_agg": P_ANY_AGG,
            "gate_seed": P_SEED_ANY,
            "gate_joint_det": JOINT_DET_MIN,
        },
        "G1_orientation": {"ok": g1, **ori, "gate": {"med": MED_ER_MAX, "p90": P90_ER_MAX}},
        "G2_position": {
            "ok": g2,
            "strong": g2_strong,
            "E_p": b3["E_p"],
            "median_ep_m": b3["median_ep_m"],
            **{k: b3[k] for k in ("mean_ep_m", "p90_ep_m", "n_finite", "frac_finite", "n") if k in b3},
        },
        "G3_info": {
            "ok": g3,
            "ratio_to_B0": agg["info_ratio"],
            "ori_beats_mean": agg["ori_beats_mean"],
            "no_seed_collapse": all(r["no_collapse"] for r in per_seed),
            "INFO_RATIO": INFO_RATIO,
        },
        "G4_near_oracle": {
            "ok": g4,
            "eta_pose": agg["eta_pose"],
            "Delta_E_pose": agg["Delta_E_pose"],
            "near_oracle_position": g4,
        },
        "B0_mean": {"pose": agg["B0_pose"], "ori": agg["B0_ori"]},
        "B1_GT_R_ceiling": agg["B1_GT_R"],
        "B2_learned_R_oracle_mask": agg["B2_learned_R_GT_mask"],
        "B3_fully_learned_primary": b3,
        "orientation_to_position_propagation": {
            "E_p_GT_R": agg["B1_GT_R"]["E_p"],
            "E_p_learned_R": agg["B2_learned_R_GT_mask"]["E_p"],
            "Delta_E_R": agg["B2_learned_R_GT_mask"]["Delta_E_R"],
        },
        "per_seed": per_seed,
        "localization": {"median_iou_head": agg["iou_h"], "median_iou_observer": agg["iou_o"]},
        "unlocks_o1": unlock_o1,
        "unlocks_o1_prereg": unlock_o1,
        "R_BO_is_GT_nuisance": False,
        "does_not_rewrite_o0g2r": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            k: summary[k]
            for k in (
                "pattern", "G0", "G1_orientation", "G2_position", "G3_info", "G4_near_oracle",
                "B0_mean", "B1_GT_R_ceiling", "B2_learned_R_oracle_mask", "B3_fully_learned_primary",
                "orientation_to_position_propagation", "per_seed", "unlocks_o1",
            )
        },
    )
    return summary

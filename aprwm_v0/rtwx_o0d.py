"""RTWX-O0D: perception instrument audit. No O1 unlock. No observability claim."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import (
    L_HIST,
    RGB_SIZE,
    TASK,
    _build_encoder,
    _e_R_deg,
    _fill_twist_fd,
    _o0_args,
    _quat_fix,
    _stats,
)
from .rtwx_x0 import _nrmse
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import CAMERA, _capture_rgb_step, _patch_curobo_planner, _patch_raster_shader, _resize_rgb

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0D_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0d.perception_instrument_audit.v1"
SEED_AUDIT = 17601
O0_CACHE_DEFAULT = "runs/rtwx_o0"
MEM_N = 256
MEM_EPOCHS = 200
E_P_MEM_MAX = 0.15
E_P_MAX = 0.30
INFO_RATIO = 0.85
D0_VIS_MIN = 0.001
D0_TRACK_MIN = 0.25
N_OVERLAY = 32
N_TRAIN_EP = 16
N_VAL_EP = 8
N_TEST_EP = 8
N_STEPS = 40


def _score_pose(pred: dict[str, np.ndarray], pool: dict[str, Any]) -> dict[str, Any]:
    er = _e_R_deg(pred["quat"], pool["quat"])
    ep_cm = np.linalg.norm(pred["p"] - pool["p"], axis=1) * 100.0
    return {
        "E_p": _nrmse(pred["p"], pool["p"]),
        "median_ep_cm": float(np.median(ep_cm)),
        "mean_ep_cm": float(np.mean(ep_cm)),
        "median_e_R_deg": float(np.median(er)),
        "p90_e_R_deg": float(np.percentile(er, 90)),
    }


@dataclass(frozen=True)
class RTWXO0DConfig:
    output: str = "runs/rtwx_o0d"
    o0_cache: str = O0_CACHE_DEFAULT
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = N_TRAIN_EP
    n_val_ep: int = N_VAL_EP
    n_test_ep: int = N_TEST_EP
    n_steps: int = N_STEPS
    seed: int = SEED_AUDIT
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    mem_n: int = MEM_N
    mem_epochs: int = MEM_EPOCHS
    vis_epochs: int = 40


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0D must not write there")


def _lock(cfg: RTWXO0DConfig) -> RTWXO0DConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=N_TRAIN_EP,
        n_val_ep=N_VAL_EP,
        n_test_ep=N_TEST_EP,
        n_steps=N_STEPS,
        seed=SEED_AUDIT,
        rgb_size=RGB_SIZE,
        mem_n=MEM_N,
        mem_epochs=MEM_EPOCHS,
        vis_epochs=40,
    )


def architecture_audit() -> dict[str, Any]:
    """O0 encoder: ResNet children()[:-2] keeps 7×7; flatten; no GAP; no coord conv."""
    return {
        "o0_l_hist_frozen": int(L_HIST),
        "drops_resnet_avgpool_and_fc": True,
        "uses_global_average_pool": False,
        "spatial_map_then_flatten": True,
        "explicit_coord_conv": False,
        "predicts_world_p_from_global_linear": True,
        "note": "7×7 flatten is not GAP, but still no explicit (u,v); whole-image regression.",
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    return arr[ys][:, xs]


def _z2d_from_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    ys, xs = np.where(m)
    vis = float(xs.size) / float(h * w)
    if xs.size == 0:
        return np.full(5, np.nan), 0.0
    z = np.array(
        [float(xs.mean()), float(ys.mean()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), vis],
        dtype=np.float64,
    )
    return z, vis


def _load_o0_train(cache: Path) -> dict[str, Any] | None:
    part = Path(cache) / "cache_o0_train.npz"
    if not part.is_file():
        return None
    z = np.load(part)
    if "train_p" not in z:
        return None
    return {
        "p": np.asarray(z["train_p"]),
        "quat": np.asarray(z["train_quat"]),
        "rgb": np.asarray(z["train_rgb"], dtype=np.uint8),
        "n_ep": int(z["train_n_ep"][0]),
        "n_steps": int(z["train_n_steps"][0]),
    }


def _cup_ids(env: Any) -> set[int]:
    ids: set[int] = set()
    cup = env.cup
    for obj in (cup, getattr(cup, "actor", None), getattr(cup, "entity", None)):
        if obj is None:
            continue
        if hasattr(obj, "get_id"):
            try:
                ids.add(int(obj.get_id()))
            except Exception:
                pass
        for attr in ("per_scene_id", "id"):
            if hasattr(obj, attr):
                try:
                    ids.add(int(getattr(obj, attr)))
                except Exception:
                    pass
    if hasattr(cup, "get_links"):
        try:
            for lk in cup.get_links():
                if hasattr(lk, "get_id"):
                    ids.add(int(lk.get_id()))
        except Exception:
            pass
    return ids


def _front_sapien_cam(env: Any) -> Any:
    cams = env.cameras
    for cam, name in zip(cams.static_camera_list, cams.static_camera_name):
        if name == CAMERA or (name == "head_camera" and CAMERA not in cams.static_camera_name):
            return cam
    if cams.static_camera_list:
        return cams.static_camera_list[0]
    raise RuntimeError("no static camera")


def _capture_rgb_mask(env: Any, size: int, cup_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    rgb = _capture_rgb_step(env, size)
    cam = _front_sapien_cam(env)
    try:
        seg = np.asarray(cam.get_picture("Segmentation"))
        actor = np.asarray(seg[..., 1]).astype(np.int32)
        mask = np.isin(actor, list(cup_ids)) if cup_ids else np.zeros(actor.shape, dtype=bool)
        if not mask.any() and seg.shape[-1] >= 1:
            mesh = np.asarray(seg[..., 0]).astype(np.int32)
            mask = np.isin(mesh, list(cup_ids))
    except Exception:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
    return rgb, _resize_mask(mask, size)


def _blob_frame(p: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    u = int(np.clip((float(p[0]) + 0.35) / 0.70 * (size - 1), 1, size - 2))
    v = int(np.clip((float(p[1]) + 0.25) / 0.50 * (size - 1), 1, size - 2))
    r = max(3, size // 16)
    y0, y1 = max(0, v - r), min(size, v + r + 1)
    x0, x1 = max(0, u - r), min(size, u + r + 1)
    img[y0:y1, x0:x1] = (210, 70, 40)
    mask = img[:, :, 0] > 100
    return img, mask


def _collect_numpy(cfg: RTWXO0DConfig, split: str, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    ps, qs, rgbs, masks, zs, vis = [], [], [], [], [], []
    for _ep in range(n_ep):
        p0 = rng.uniform([-0.3, -0.2, 0.70], [0.3, 0.05, 0.72])
        for t in range(n_steps):
            p = p0 + np.array([0.0, 0.0, 0.0])
            quat = _quat_fix(np.array([1.0, 0.0, 0.0, 0.0]))[0]
            rgb, mask = _blob_frame(p, cfg.rgb_size)
            z, vf = _z2d_from_mask(mask)
            ps.append(p)
            qs.append(quat)
            rgbs.append(rgb)
            masks.append(mask)
            zs.append(z)
            vis.append(vf)
    return {
        "p": np.stack(ps),
        "quat": np.stack(qs),
        "rgb": np.stack(rgbs),
        "mask": np.stack(masks),
        "z2d": np.stack(zs),
        "vis": np.asarray(vis),
        "n_ep": n_ep,
        "n_steps": n_steps,
    }


def collect_audit_masks(cfg: RTWXO0DConfig, *, split: str, rng: np.random.Generator, stop_mode: list[str]) -> dict[str, Any]:
    from .rtwx_o0 import _cup_pose, _cup_twist
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _drive_pack, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = cfg.seed + {"train": 0, "val": 10_000, "test": 20_000}[split]
    eps: list[dict[str, Any]] = []
    attempts_used = 0
    slot = 0
    while len(eps) < n_ep and attempts_used < n_ep * max(2, cfg.max_resample):
        attempts_used += 1
        seed = int(seed0 + slot * 17 + attempts_used)
        slot += 1
        if len(eps) % 2 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0d] {split} valid {len(eps)}/{n_ep} attempt {attempts_used}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            if getattr(env, "step_lim", 1000) is None or int(env.step_lim) <= int(cfg.n_steps):
                env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup"):
                env.close()
                continue
            ids = _cup_ids(env)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            if _contact_invalid(env):
                env.close()
                continue
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            tt = np.arange(cfg.n_steps, dtype=np.float64) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            rows: dict[str, list] = {k: [] for k in ("p", "quat", "v", "omega", "rgb", "mask", "dt")}
            invalid = None
            for i in range(int(cfg.n_steps)):
                drv_pre = _drive_pack(jl + jr)
                p, quat = _cup_pose(env)
                lin, ang = _cup_twist(env)
                rgb, mask = _capture_rgb_mask(env, cfg.rgb_size, ids)
                rows["rgb"].append(rgb)
                rows["mask"].append(mask)
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["v"].append(lin)
                rows["omega"].append(ang)
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig_step, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig_step
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "take_action_noop"
                    break
                n_int = int(box[0])
                rows["dt"].append(max(n_int * scene_dt, scene_dt))
                drv_post = _drive_pack(jl + jr)
                n_l = q0_l.size
                cmd_l = np.asarray(action[:n_l], dtype=np.float64)
                cmd_r = np.asarray(action[n_l + 1 : n_l + 1 + q0_r.size], dtype=np.float64)
                moved_l = not np.allclose(drv_post["q_tar"][:n_l], drv_pre["q_tar"][:n_l], atol=1e-5)
                moved_r = not np.allclose(drv_post["q_tar"][n_l:], drv_pre["q_tar"][n_l:], atol=1e-5)
                match_l = np.allclose(drv_post["q_tar"][:n_l], cmd_l, atol=5e-3)
                match_r = np.allclose(drv_post["q_tar"][n_l:], cmd_r, atol=5e-3)
                if (not moved_l and not match_l) or (not moved_r and not match_r):
                    invalid = "topp_skip_drive_not_updated"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
            try:
                env.close()
            except Exception:
                pass
            if invalid is not None:
                continue
            if len(rows["p"]) != int(cfg.n_steps):
                continue
            ep = {
                "p": np.asarray(rows["p"], dtype=np.float64),
                "quat": np.asarray(rows["quat"], dtype=np.float64),
                "v": np.asarray(rows["v"], dtype=np.float64),
                "omega": np.asarray(rows["omega"], dtype=np.float64),
                "rgb": np.stack(rows["rgb"]),
                "mask": np.stack(rows["mask"]),
                "dt": np.asarray(rows["dt"], dtype=np.float64),
            }
            if not np.isfinite(ep["v"]).all():
                ep["v"], ep["omega"] = _fill_twist_fd(ep["p"], ep["quat"], ep["dt"])
            eps.append(ep)
        except Exception as exc:
            print(f"[rtwx-o0d] {split} fail {attempts_used}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop_mode.append(f"{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0D collect {split}: 0 episodes")
    out: dict[str, Any] = {k: np.concatenate([e[k] for e in eps], axis=0) for k in ("p", "quat", "rgb", "mask")}
    zs, vis = [], []
    for m in out["mask"]:
        z, vf = _z2d_from_mask(m)
        zs.append(z)
        vis.append(vf)
    out["z2d"] = np.stack(zs)
    out["vis"] = np.asarray(vis)
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    return out


def _save_audit(path: Path, name: str, p: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": p[k] for k in ("p", "quat", "rgb", "mask", "z2d", "vis")}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_audit(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in ("p", "quat", "mask", "z2d", "vis")}
    p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect_audit(cfg: RTWXO0DConfig, root: Path) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(cfg.seed),
        "val": np.random.default_rng(cfg.seed + 1),
        "test": np.random.default_rng(cfg.seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0d_{name}.npz"
        hit = _load_audit(part, name)
        if hit is not None:
            print(f"[rtwx-o0d] {name}: cache", flush=True)
            splits[name] = hit
            continue
        if cfg.backend == "numpy":
            pool = _collect_numpy(cfg, name, rng)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_audit_masks(cfg, split=name, rng=rng, stop_mode=stop)
        splits[name] = pool
        _save_audit(part, name, pool)
    if stop:
        raise RuntimeError(f"O0D collect stop: {stop}")
    return splits


def _clip1(rgb: np.ndarray, i: int) -> np.ndarray:
    x = np.asarray(rgb[i], dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))


def _xy_pose(pool: dict[str, Any], stats: dict[str, np.ndarray], idx: np.ndarray, *, masked: bool) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(pool["rgb"])
    if masked:
        m = np.asarray(pool["mask"], dtype=np.float32)[..., None]
        rgb = (rgb.astype(np.float32) * m).astype(np.uint8)
    y = np.concatenate(
        [(pool["p"] - stats["mu_p"]) / stats["sd_p"], (pool["quat"] - stats["mu_q"]) / stats["sd_q"]],
        axis=1,
    ).astype(np.float32)
    x = np.stack([_clip1(rgb, int(i))[None] for i in idx])
    return x, y[idx]


def _batches(x: np.ndarray, y: np.ndarray, bs: int) -> list:
    return [(x[i : i + bs], y[i : i + bs]) for i in range(0, x.shape[0], bs)]


def _train_overfit(model: Any, x: np.ndarray, y: np.ndarray, *, seed: int, epochs: int, wd: float) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=wd)
    loss_fn = nn.MSELoss()
    batches = _batches(x, y, 16 if x.shape[0] > 16 else max(1, x.shape[0]))
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(batches))
        for i in order:
            xb, yb = batches[int(i)]
            opt.zero_grad()
            pred = model(torch.from_numpy(xb).to(dev))
            loss_fn(pred, torch.from_numpy(yb).float().to(dev)).backward()
            opt.step()
    model.eval()
    return model


def _train_val(model: Any, tr: list, va: list, *, seed: int, epochs: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
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
            if bad >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_pose(model: Any, x: np.ndarray, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            rows.append(model(torch.from_numpy(x[i : i + 16]).to(dev)).cpu().numpy())
    y = np.concatenate(rows, axis=0)
    return {"p": y[:, :3] * stats["sd_p"] + stats["mu_p"], "quat": _quat_fix(y[:, 3:7] * stats["sd_q"] + stats["mu_q"])}


def _fit_z2d(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.isfinite(z).all(1) & np.isfinite(p).all(1)
    x = np.concatenate([np.ones((int(m.sum()), 1)), z[m]], axis=1)
    w, *_ = np.linalg.lstsq(x, p[m], rcond=None)
    return w


def _pred_z2d(w: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.ones((z.shape[0], 1)), np.nan_to_num(z, nan=0.0)], axis=1)
    return x @ w


def _overlay(rgb: np.ndarray, mask: np.ndarray, p: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = np.asarray(rgb, dtype=np.uint8).copy()
    m = np.asarray(mask, dtype=bool)
    img[m] = (0.55 * img[m] + np.array([0, 140, 0])).clip(0, 255).astype(np.uint8)
    ys, xs = np.where(m)
    if xs.size:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        img[y0 : y1 + 1, x0] = (0, 255, 40)
        img[y0 : y1 + 1, x1] = (0, 255, 40)
        img[y0, x0 : x1 + 1] = (0, 255, 40)
        img[y1, x0 : x1 + 1] = (0, 255, 40)
    Image.fromarray(img).save(path)


def _d0(audit: dict[str, Any], root: Path) -> dict[str, Any]:
    vis = np.asarray(audit["vis"], dtype=np.float64)
    z = np.asarray(audit["z2d"])
    p = np.asarray(audit["p"])
    ov = root / "overlays"
    ov.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED_AUDIT)
    n = min(N_OVERLAY, p.shape[0])
    pick = rng.choice(p.shape[0], size=n, replace=False)
    for k, i in enumerate(pick):
        _overlay(audit["rgb"][int(i)], audit["mask"][int(i)], p[int(i)], ov / f"overlay_{k:02d}.png")
    rho = {
        "u_px": _spearman(z[:, 0], p[:, 0]),
        "u_py": _spearman(z[:, 0], p[:, 1]),
        "v_px": _spearman(z[:, 1], p[:, 0]),
        "v_py": _spearman(z[:, 1], p[:, 1]),
    }
    track = float(np.nanmax(np.abs([rho[k] for k in rho])))
    return {
        "n_overlay": n,
        "median_vis": float(np.median(vis)),
        "mean_vis": float(np.mean(vis)),
        "frac_visible": float(np.mean(vis > 0)),
        "median_area_px": float(np.median(vis * (audit["rgb"].shape[1] ** 2))),
        "spearman": rho,
        "D0_vis": bool(np.median(vis) >= D0_VIS_MIN),
        "D0_track": bool(np.isfinite(track) and track >= D0_TRACK_MIN),
        "same_step_rgb_and_pose": True,
        "note": "RGB and s^O recorded in the same loop iteration before take_action.",
    }


def _pose_stats(pool: dict[str, Any]) -> dict[str, np.ndarray]:
    st = _stats({"p": pool["p"], "quat": pool["quat"], "v": pool["p"], "omega": pool["p"]})
    return {"mu_p": st["mu_p"], "sd_p": st["sd_p"], "mu_q": st["mu_q"], "sd_q": st["sd_q"]}


def _b0_p(train: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    return {
        "p": np.repeat(train["p"].mean(0, keepdims=True), n, 0),
        "quat": np.repeat(_quat_fix(train["quat"].mean(0, keepdims=True)), n, 0),
    }


def _d1(cfg: RTWXO0DConfig, o0: dict[str, Any] | None, synth: dict[str, Any] | None) -> dict[str, Any]:
    src = o0 if o0 is not None else synth
    if src is None:
        return {"D1_ok": False, "reason": "no_o0_cache_and_no_synth"}
    n = min(int(cfg.mem_n), int(src["p"].shape[0]))
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(src["p"].shape[0], size=n, replace=False))
    sub = {"p": src["p"][idx], "quat": src["quat"][idx], "rgb": src["rgb"][idx]}
    dummy_mask = np.ones(sub["rgb"].shape[:3], dtype=bool)
    sub["mask"] = dummy_mask
    st = _pose_stats(sub)
    x, y = _xy_pose(sub, st, np.arange(n), masked=False)
    model = _train_overfit(
        _build_encoder(n_out=7, l_hist=1, smoke=cfg.smoke),
        x,
        y,
        seed=cfg.seed,
        epochs=int(cfg.mem_epochs),
        wd=0.0,
    )
    pred = _predict_pose(model, x, st)
    sc = _score_pose(pred, {"p": sub["p"], "quat": sub["quat"]})
    b0 = _score_pose(_b0_p(sub, n), {"p": sub["p"], "quat": sub["quat"]})
    return {
        "n": n,
        "source": "o0_train_cache" if o0 is not None else "numpy_synth",
        "E_p_mem": sc["E_p"],
        "median_ep_cm": sc["median_ep_cm"],
        "median_e_R_deg": sc["median_e_R_deg"],
        "B0_E_p": b0["E_p"],
        "D1_ok": bool(sc["E_p"] <= E_P_MEM_MAX),
        "epochs": int(cfg.mem_epochs),
        "wd": 0.0,
        "L": 1,
    }


def _d2(cfg: RTWXO0DConfig, splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tr, va, te = splits["train"], splits["val"], splits["test"]
    st = _pose_stats(tr)
    xtr, ytr = _xy_pose(tr, st, np.arange(tr["p"].shape[0]), masked=True)
    xva, yva = _xy_pose(va, st, np.arange(va["p"].shape[0]), masked=True)
    xte, _ = _xy_pose(te, st, np.arange(te["p"].shape[0]), masked=True)
    model = _train_val(
        _build_encoder(n_out=7, l_hist=1, smoke=cfg.smoke),
        _batches(xtr, ytr, 12),
        _batches(xva, yva, 12),
        seed=cfg.seed,
        epochs=4 if cfg.smoke else cfg.vis_epochs,
    )
    pred = _predict_pose(model, xte, st)
    sc = _score_pose(pred, te)
    b0 = _score_pose(_b0_p(tr, te["p"].shape[0]), te)
    ok = bool(sc["E_p"] <= E_P_MAX and sc["E_p"] <= INFO_RATIO * b0["E_p"])
    return {"B2_mask": sc, "B0": b0, "D2_ok": ok}


def _d3(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tr, te = splits["train"], splits["test"]
    w = _fit_z2d(tr["z2d"], tr["p"])
    pred_p = _pred_z2d(w, te["z2d"])
    dummy_q = np.repeat(_quat_fix(tr["quat"].mean(0, keepdims=True)), te["p"].shape[0], 0)
    sc = _score_pose({"p": pred_p, "quat": dummy_q}, te)
    b0 = _score_pose(_b0_p(tr, te["p"].shape[0]), te)
    return {"E_p": sc["E_p"], "median_ep_cm": sc["median_ep_cm"], "B0_E_p": b0["E_p"], "D3_ok": bool(sc["E_p"] <= E_P_MAX)}


def _pattern(d1: dict[str, Any], d2: dict[str, Any], d3: dict[str, Any]) -> str:
    if not d1.get("D1_ok"):
        return "perception_instrument_failure"
    if d2.get("D2_ok") or d3.get("D3_ok"):
        return "localization_bottleneck"
    return "pose_geometry_unresolved"


def run_rtwx_o0d(output: str | Path, config: RTWXO0DConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0DConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0D",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "does_not_unlock_o1": True,
        "o0_frozen": "object_pose_failure",
        "seed": cfg.seed,
        "config": asdict(cfg),
        "architecture": architecture_audit(),
    }
    _write_json(root / "header.json", header)
    o0 = None if cfg.smoke else _load_o0_train(Path(cfg.o0_cache))
    if o0 is None and cfg.backend == "robotwin" and not cfg.smoke:
        print("[rtwx-o0d] warning: O0 train cache missing; D1 will fail closed", flush=True)
    splits = _collect_audit(cfg, root)
    d0 = _d0(splits["train"], root)
    d1 = _d1(cfg, o0, splits["train"] if o0 is None else None)
    print(f"[rtwx-o0d] D1 E_p_mem={d1.get('E_p_mem')} D1_ok={d1.get('D1_ok')}", flush=True)
    d2 = _d2(cfg, splits)
    print(f"[rtwx-o0d] D2 E_p={d2['B2_mask']['E_p']:.4f} D2_ok={d2['D2_ok']}", flush=True)
    d3 = _d3(splits)
    print(f"[rtwx-o0d] D3 E_p={d3['E_p']:.4f} D3_ok={d3['D3_ok']}", flush=True)
    pattern = _pattern(d1, d2, d3)
    summary = {
        "header": header,
        "pattern": pattern,
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "does_not_unlock_o1": True,
        "no_observability_claim": True,
        "velocity_secondary": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {"pattern": pattern, "D0": d0, "D1": d1, "D2": d2, "D3": d3})
    return summary

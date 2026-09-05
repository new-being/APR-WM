"""RTWX-O0G2: learned RGB→mask + depth/geometry → p_pose. No RGB→p regression.

Frozen: head_camera 64², small U-Net, δ_O from O0G1b asset prior, seed 24601.
Does not unlock O1. O0C locked until this cell PASS. Not R10.
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
from .rtwx_o0d3 import _capture_head, _head_cam
from .rtwx_o0g import MASK_PX
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_x0 import _nrmse
from .rtwx_x0c import FORMAL_N_STEPS, FORMAL_N_TEST, FORMAL_N_TRAIN, FORMAL_N_VAL, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g2.learned_localization_geometry_position.v1"
CAMERA = "head_camera"
SEED = 24601
RGB_SIZE = 64
G0_FOV = 0.95
G0_VIS = 0.90
EP_MAX = 0.20
MED_MAX = 0.05
STRONG_MED = 0.02
IOU_MED_MIN = 0.50
EMPTY_RATE_MAX = 0.05
INFO_RATIO = 0.85
EPOCHS = 25
PATIENCE = 6
LR = 1.0e-3
MASK_THRESH = 0.5


@dataclass(frozen=True)
class RTWXO0G2Config:
    output: str = "runs/rtwx_o0g2"
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
        raise RuntimeError("r10_c0 is locked; RTWX-O0G2 must not write there")


def _lock(cfg: RTWXO0G2Config) -> RTWXO0G2Config:
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


def _small_unet(size: int = 64):
    from torch import nn
    import torch.nn.functional as F

    class ConvBN(nn.Module):
        def __init__(self, a: int, b: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e1 = ConvBN(3, 16)
            self.e2 = ConvBN(16, 32)
            self.e3 = ConvBN(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d2 = ConvBN(64, 32)
            self.u1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.d1 = ConvBN(32, 16)
            self.out = nn.Conv2d(16, 1, 1)
            self.size = size

        def forward(self, x):
            x1 = self.e1(x)
            x2 = self.e2(self.pool(x1))
            x3 = self.e3(self.pool(x2))
            y2 = self.u2(x3)
            if y2.shape[-2:] != x2.shape[-2:]:
                y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            y2 = self.d2(torch_cat(y2, x2))
            y1 = self.u1(y2)
            if y1.shape[-2:] != x1.shape[-2:]:
                y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            y1 = self.d1(torch_cat(y1, x1))
            return self.out(y1)

    def torch_cat(a, b):
        import torch

        return torch.cat([a, b], dim=1)

    return UNet()


def _nchw_rgb(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(x, (0, 3, 1, 2))


def _xyz64_from_position(position: np.ndarray, model: np.ndarray, size: int) -> np.ndarray:
    pos = np.asarray(position)
    h, w = pos.shape[:2]
    pts = pos[..., :3].astype(np.float64)
    mm = np.asarray(model, dtype=np.float64)
    world = pts @ mm[:3, :3].T + mm[:3, 3]
    if pos.shape[-1] >= 4:
        world = world.copy()
        world[pos[..., 3] >= 1.0] = np.nan
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    return world[ys][:, xs].astype(np.float32)


def _centroid_xyz(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(3, np.nan, dtype=np.float64)
    pts = np.asarray(xyz, dtype=np.float64)[m]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 8:
        return np.full(3, np.nan, dtype=np.float64)
    return np.median(pts, axis=0)


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _dice_bce(logits, target):
    import torch
    from torch.nn import functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    den = prob.sum(dims) + target.sum(dims)
    dice = 1.0 - (2 * inter + 1.0) / (den + 1.0)
    return bce + dice.mean()


def _train_unet(rgb: np.ndarray, mask: np.ndarray, rgb_va: np.ndarray, mask_va: np.ndarray, *, seed: int, epochs: int) -> Any:
    import torch

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _small_unet(int(rgb.shape[1])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xtr = torch.from_numpy(_nchw_rgb(rgb))
    ytr = torch.from_numpy(np.asarray(mask, dtype=np.float32)[:, None])
    xva = torch.from_numpy(_nchw_rgb(rgb_va))
    yva = torch.from_numpy(np.asarray(mask_va, dtype=np.float32)[:, None])
    bs = 16
    best, best_state, bad = float("inf"), None, 0
    n = xtr.shape[0]
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(n)
        for i in range(0, n, bs):
            idx = order[i : i + bs]
            xb = xtr[idx].to(dev)
            yb = ytr[idx].to(dev)
            opt.zero_grad()
            loss = _dice_bce(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, xva.shape[0], bs):
                vals.append(float(_dice_bce(model(xva[i : i + bs].to(dev)), yva[i : i + bs].to(dev))))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 5 == 0 or ep + 1 == epochs:
            print(f"[rtwx-o0g2] unet ep={ep} val={v:.4f} best={best:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_masks(model: Any, rgb: np.ndarray) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    x = _nchw_rgb(rgb)
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            logits = model(torch.from_numpy(x[i : i + 16]).to(dev))
            outs.append((torch.sigmoid(logits).cpu().numpy()[:, 0] >= MASK_THRESH))
    return np.concatenate(outs, axis=0)


def _pose_from_masks(xyz: np.ndarray, masks: np.ndarray, quat: np.ndarray, delta_O: np.ndarray) -> np.ndarray:
    n = masks.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for i in range(n):
        surf = _centroid_xyz(xyz[i], masks[i])
        if not np.isfinite(surf).all():
            continue
        out[i] = surf - _quat_to_R(quat[i]) @ delta_O
    return out


def _numpy_split(cfg: RTWXO0G2Config, split: str, rng: np.random.Generator, delta_O: np.ndarray) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    size = int(cfg.rgb_size)
    rows = {k: [] for k in ("p", "quat", "rgb", "mask", "xyz", "in_fov", "visible")}
    quat0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    for ep in range(n_ep):
        p0 = np.array([-0.25 + 0.5 * (ep / max(n_ep - 1, 1)), rng.uniform(-0.15, 0.05), 0.74])
        for t in range(int(cfg.n_steps)):
            p = p0 + np.array([0.001 * np.sin(t / 10.0), 0.0, 0.0])
            rgb = np.full((size, size, 3), 20, dtype=np.uint8)
            # project-ish blob from p
            u = int(np.clip((p[0] + 0.35) / 0.70 * (size - 1), 2, size - 3))
            v = int(np.clip((p[1] + 0.25) / 0.50 * (size - 1), 2, size - 3))
            r = max(3, size // 12)
            mask = np.zeros((size, size), dtype=bool)
            mask[v - r : v + r + 1, u - r : u + r + 1] = True
            rgb[mask] = (210, 80, 40)
            rgb[0, (ep * 3 + t) % size, 2] = 40 + ((ep + t) % 200)
            xyz = np.full((size, size, 3), np.nan, dtype=np.float32)
            surf = p + _quat_to_R(quat0) @ delta_O
            # fill mask pixels with surf + small spatial variation
            ys, xs = np.where(mask)
            for yy, xx in zip(ys, xs):
                xyz[yy, xx] = surf + np.array([(xx - u) * 0.001, (yy - v) * 0.001, 0.0])
            rows["p"].append(p)
            rows["quat"].append(quat0)
            rows["rgb"].append(rgb)
            rows["mask"].append(mask)
            rows["xyz"].append(xyz)
            rows["in_fov"].append(True)
            rows["visible"].append(True)
    return {
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "rgb": np.stack(rows["rgb"]),
        "mask": np.stack(rows["mask"]),
        "xyz": np.stack(rows["xyz"]),
        "in_fov": np.asarray(rows["in_fov"]),
        "visible": np.asarray(rows["visible"]),
        "n_ep": n_ep,
        "n_steps": int(cfg.n_steps),
    }


def collect_split(cfg: RTWXO0G2Config, *, split: str, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
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
            print(f"[rtwx-o0g2] {split} valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(int(cfg.n_steps) + 50, 1000)
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
            tt = np.arange(cfg.n_steps) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            cam = _head_cam(env)
            rows: dict[str, list] = {k: [] for k in ("p", "quat", "rgb", "mask", "xyz", "in_fov", "visible")}
            invalid = None
            repaired = None
            for i in range(int(cfg.n_steps)):
                p, quat = _cup_pose(env)
                rgb = _capture_head(env, size)
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
                uu = int(np.clip(round(pr["u"]), 0, w - 1))
                vv = int(np.clip(round(pr["v"]), 0, h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                if repaired is None and pr["in_fov"] and id_uv >= 0:
                    repaired = id_uv
                cid = repaired if repaired is not None else (sorted(sim_ids)[0] if sim_ids else -1)
                mask_n = (actor == cid) if cid >= 0 else np.zeros_like(actor, dtype=bool)
                area = int(mask_n.sum())
                in_fov = bool(pr["in_fov"])
                visible = bool(in_fov and (area >= MASK_PX or id_uv >= 0))
                mask64 = _resize_mask(mask_n, size)
                if position is not None and model is not None:
                    xyz = _xyz64_from_position(position, model, size)
                else:
                    xyz = np.full((size, size, 3), np.nan, dtype=np.float32)
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["rgb"].append(rgb)
                rows["mask"].append(mask64)
                rows["xyz"].append(xyz)
                rows["in_fov"].append(in_fov)
                rows["visible"].append(visible)
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
                    "rgb": np.stack(rows["rgb"]),
                    "mask": np.stack(rows["mask"]),
                    "xyz": np.stack(rows["xyz"]),
                    "in_fov": np.asarray(rows["in_fov"], dtype=bool),
                    "visible": np.asarray(rows["visible"], dtype=bool),
                }
            )
        except Exception as exc:
            print(f"[rtwx-o0g2] {split} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0G2 collect {split}: 0 episodes")
    out: dict[str, Any] = {
        k: np.concatenate([e[k] for e in eps], axis=0) for k in ("p", "quat", "rgb", "mask", "xyz", "in_fov", "visible")
    }
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    return out


def _save(path: Path, name: str, p: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": p[k] for k in ("p", "quat", "rgb", "mask", "xyz", "in_fov", "visible")}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    np.savez_compressed(path, **payload)


def _load(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in ("p", "quat", "mask", "xyz", "in_fov", "visible")}
    p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
    p["mask"] = np.asarray(p["mask"], dtype=bool)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect(cfg: RTWXO0G2Config, root: Path, delta_O: np.ndarray) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(cfg.seed),
        "val": np.random.default_rng(cfg.seed + 1),
        "test": np.random.default_rng(cfg.seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0g2_{name}.npz"
        hit = _load(part, name)
        if hit is not None:
            print(f"[rtwx-o0g2] {name}: cache", flush=True)
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
        raise RuntimeError(f"O0G2 collect stop: {stop}")
    return splits


def _score_pose(phat: np.ndarray, p: np.ndarray, *, require_all: bool = True) -> dict[str, Any]:
    """Empty/NaN predictions count as failures (use train-mean distance proxy via large err)."""
    finite = np.isfinite(phat).all(axis=1)
    if require_all and not finite.any():
        return {"E_p": float("inf"), "median_ep_m": float("inf"), "n_finite": 0, "frac_finite": 0.0, "n": int(p.shape[0])}
    # for non-finite: treat as max of observed errors or 1.0 m floor so they hurt median
    err = np.linalg.norm(phat - p, axis=1)
    fill = float(np.nanmax(err[finite])) if finite.any() else 1.0
    fill = max(fill, 1.0)
    err = np.where(finite, err, fill)
    ph = np.where(finite[:, None], phat, p + fill / np.sqrt(3.0))
    return {
        "E_p": float(_nrmse(ph, p)),
        "median_ep_m": float(np.median(err)),
        "mean_ep_m": float(np.mean(err)),
        "p90_ep_m": float(np.percentile(err, 90)),
        "n_finite": int(finite.sum()),
        "frac_finite": float(finite.mean()),
        "n": int(p.shape[0]),
    }


def _loc_metrics(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    ious, euvs = [], []
    empty_vis = 0
    n_vis = int(visible.sum())
    for i in range(pred.shape[0]):
        ious.append(_iou(pred[i], gt[i]))
        pu, pv = _mask_center(pred[i])
        gu, gv = _mask_center(gt[i])
        if visible[i] and not pred[i].any():
            empty_vis += 1
        if np.isfinite(pu) and np.isfinite(gu):
            euvs.append(float(np.hypot(pu - gu, pv - gv)))
    empty_rate = float(empty_vis / max(n_vis, 1))
    return {
        "median_iou": float(np.median(ious)),
        "mean_iou": float(np.mean(ious)),
        "median_e_uv_px": float(np.median(euvs)) if euvs else float("inf"),
        "empty_given_visible": empty_rate,
        "n_visible": n_vis,
        "ok": bool(float(np.median(ious)) >= IOU_MED_MIN and empty_rate <= EMPTY_RATE_MAX),
        "gate": {"median_iou_min": IOU_MED_MIN, "empty_rate_max": EMPTY_RATE_MAX},
    }


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "coverage_failure"
    if not g1:
        return "oracle_geometry_failure"
    if not g2:
        return "learned_localization_failure"
    if not g3 or not g4:
        return "localization_geometry_failure"
    return "geometry_mediated_position_supported"


def run_rtwx_o0g2(output: str | Path, config: RTWXO0G2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G2Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    header = {
        "stage": "RTWX-O0G2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "camera": CAMERA,
        "rgb_size": RGB_SIZE,
        "encoder": "small_UNet_binary_mask",
        "seed": cfg.seed,
        "delta_O": delta_O.tolist(),
        "delta_meta": prior,
        "no_rgb_to_p_regression": True,
        "unlocks_o1": False,
        "unlocks_o0c_prereg_if_pass": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    splits = _collect(cfg, root, delta_O)
    train, val, test = splits["train"], splits["val"], splits["test"]

    p_fov = float(np.mean(test["in_fov"]))
    p_vis = float(np.mean(test["visible"]))
    g0 = bool(p_fov >= G0_FOV and p_vis >= G0_VIS)
    print(f"[rtwx-o0g2] G0 FOV={p_fov:.3f} vis={p_vis:.3f} ok={g0}", flush=True)

    # B0 train-mean
    p_mean = train["p"].mean(0, keepdims=True)
    b0 = _score_pose(np.repeat(p_mean, test["p"].shape[0], 0), test["p"])
    print(f"[rtwx-o0g2] B0 E_p={b0['E_p']:.4f} med={b0['median_ep_m']:.4f}", flush=True)

    # B1 oracle
    p_b1 = _pose_from_masks(test["xyz"], test["mask"], test["quat"], delta_O)
    b1 = _score_pose(p_b1, test["p"])
    g1 = bool(b1["E_p"] <= EP_MAX and b1["median_ep_m"] <= MED_MAX)
    b1_strong = bool(g1 and b1["median_ep_m"] <= STRONG_MED)
    print(f"[rtwx-o0g2] B1 E_p={b1['E_p']:.4f} med={b1['median_ep_m']:.4f} ok={g1} strong={b1_strong}", flush=True)

    # B2 learned
    epochs = 4 if cfg.smoke else cfg.epochs
    model = _train_unet(train["rgb"], train["mask"], val["rgb"], val["mask"], seed=cfg.seed, epochs=epochs)
    pred_m = _predict_masks(model, test["rgb"])
    loc = _loc_metrics(pred_m, test["mask"], test["visible"])
    g2 = bool(loc["ok"])
    print(
        f"[rtwx-o0g2] G2 IoU_med={loc['median_iou']:.3f} empty|vis={loc['empty_given_visible']:.3f} ok={g2}",
        flush=True,
    )
    p_b2 = _pose_from_masks(test["xyz"], pred_m, test["quat"], delta_O)
    b2 = _score_pose(p_b2, test["p"])
    g3 = bool(b2["E_p"] <= EP_MAX and b2["median_ep_m"] <= MED_MAX)
    b2_strong = bool(g3 and b2["median_ep_m"] <= STRONG_MED)
    g4 = bool(b2["E_p"] <= INFO_RATIO * b0["E_p"]) if np.isfinite(b0["E_p"]) else False
    print(
        f"[rtwx-o0g2] B2 E_p={b2['E_p']:.4f} med={b2['median_ep_m']:.4f} g3={g3} g4={g4} ({b2['E_p']/max(b0['E_p'],1e-12):.3f}×B0)",
        flush=True,
    )

    d_loc = float(b2["E_p"] - b1["E_p"]) if np.isfinite(b2["E_p"]) and np.isfinite(b1["E_p"]) else float("nan")
    eta = float(b2["E_p"] / b1["E_p"]) if np.isfinite(b2["E_p"]) and b1["E_p"] > 1e-12 else float("nan")
    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3, g4=g4)
    unlock_o0c = pattern == "geometry_mediated_position_supported"
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": g0, "P_FOV": p_fov, "P_visible": p_vis},
        "G1_oracle": {"ok": g1, "strong": b1_strong, **b1},
        "G2_localization": loc,
        "G3_pose": {"ok": g3, "strong": b2_strong, **b2},
        "G4_info": {"ok": g4, "ratio_to_B0": float(b2["E_p"] / max(b0["E_p"], 1e-12)), "INFO_RATIO": INFO_RATIO},
        "B0_mean": b0,
        "B1_oracle": b1,
        "B2_learned": b2,
        "decomp": {"Delta_E_loc": d_loc, "eta_geom": eta},
        "unlocks_o0c_prereg": unlock_o0c,
        "unlocks_o1": False,
        "no_rgb_to_p_regression": True,
        "does_not_rewrite_o0g1b": True,
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
                "G1_oracle",
                "G2_localization",
                "G3_pose",
                "G4_info",
                "B0_mean",
                "B1_oracle",
                "B2_learned",
                "decomp",
                "unlocks_o0c_prereg",
            )
        },
    )
    return summary

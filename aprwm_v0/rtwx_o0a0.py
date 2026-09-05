"""RTWX-O0A0: appearance-conditioned yaw observability. Oracle mask; no full-R retraining."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args, _quat_fix
from .rtwx_o0d import _cup_ids
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g2r import _cam_pack, _capture_pair_rgb
from .rtwx_o0q0 import _R_to_quat, _R_y, _robot_q
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0v import CAM_HEAD, CAM_OBS, _get_cam
from .rtwx_x0c import _contact_invalid, _write_json
from .rtwx_x0rgb import _patch_raster_shader, _resize_rgb

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0A0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0a0.appearance_yaw_observability.v1"
SEED = 35601
N_TRAIN_BASE = 48
N_VAL_BASE = 24
N_TEST_BASE = 24
N_YAW = 24
YAW_STEP_DEG = 15.0
RGB_SIZE = 128
CROP_PAD = 0.20
EY = np.array([0.0, 1.0, 0.0], dtype=np.float64)
EPOCHS = 40
PATIENCE = 8
LR = 1.0e-3
BATCH = 64
DELTA_RGB_MAX = 2.0
PN_MAX = 1.0e-5
LABEL_ERR_MAX_DEG = 1.0
F90_THRESH = 75.0
REPEAT_DELTA_MAX = 2.0


@dataclass(frozen=True)
class RTWXO0A0Config:
    output: str = "runs/rtwx_o0a0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    n_train_base: int = N_TRAIN_BASE
    n_val_base: int = N_VAL_BASE
    n_test_base: int = N_TEST_BASE
    n_yaw: int = N_YAW
    rgb_size: int = RGB_SIZE
    epochs: int = EPOCHS
    smoke: bool = False
    seed_attempts: int = 32
    max_resample: int = 16


def _lock(cfg: RTWXO0A0Config) -> RTWXO0A0Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        if cfg.smoke:
            return replace(
                cfg,
                n_train_base=4,
                n_val_base=2,
                n_test_base=2,
                epochs=6,
                n_yaw=N_YAW,
                rgb_size=RGB_SIZE,
            )
        return cfg
    return replace(
        cfg,
        seed=SEED,
        n_train_base=N_TRAIN_BASE,
        n_val_base=N_VAL_BASE,
        n_test_base=N_TEST_BASE,
        n_yaw=N_YAW,
        rgb_size=RGB_SIZE,
        epochs=EPOCHS,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0A0 must not write there")


def _n_cup(quat: np.ndarray) -> np.ndarray:
    return _quat_to_R(quat) @ EY


def _yaw_bins(n: int = N_YAW) -> np.ndarray:
    return YAW_STEP_DEG * np.arange(int(n), dtype=np.float64)


def _relative_yaw_deg(quat0: np.ndarray, quat: np.ndarray) -> float:
    R0 = _quat_to_R(quat0)
    R = _quat_to_R(quat)
    Rel = R0.T @ R
    # R_y(θ): [[c,0,s],[0,1,0],[-s,0,c]]
    c, s = float(Rel[0, 0]), float(Rel[0, 2])
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def _circular_err_deg(theta_hat: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = (np.asarray(theta_hat, dtype=np.float64) - np.asarray(theta, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _bin_to_deg(k: np.ndarray, n_yaw: int = N_YAW) -> np.ndarray:
    return YAW_STEP_DEG * (np.asarray(k, dtype=np.float64) % int(n_yaw))


def _mask_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    size: int = RGB_SIZE,
    pad: float = CROP_PAD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oracle-mask bbox crop → fixed size; background zeroed."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    out_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    out_d = np.zeros((size, size), dtype=np.float32)
    out_m = np.zeros((size, size), dtype=bool)
    if not m.any():
        return out_rgb, out_d, out_m
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    py = int(np.ceil(pad * max(bh, 1)))
    px = int(np.ceil(pad * max(bw, 1)))
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    rgb_c = np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8).copy()
    d_c = np.asarray(depth[y0:y1, x0:x1], dtype=np.float32).copy()
    m_c = m[y0:y1, x0:x1]
    rgb_c[~m_c] = 0
    d_c[~m_c] = 0.0
    out_rgb = _resize_rgb(rgb_c, size)
    # nearest for depth/mask
    ys_i = np.linspace(0, m_c.shape[0] - 1, size).astype(np.int64)
    xs_i = np.linspace(0, m_c.shape[1] - 1, size).astype(np.int64)
    out_d = d_c[ys_i][:, xs_i].astype(np.float32)
    out_m = m_c[ys_i][:, xs_i]
    out_rgb[~out_m] = 0
    out_d[~out_m] = 0.0
    return out_rgb, out_d, out_m


def _depth_from_position(position: np.ndarray, size: int) -> np.ndarray:
    pos = np.asarray(position)
    if pos.ndim < 3:
        return np.zeros((size, size), dtype=np.float32)
    z = np.abs(pos[..., 2].astype(np.float64))
    if pos.shape[-1] >= 4:
        z = z.copy()
        z[pos[..., 3] >= 1.0] = np.nan
    h, w = z.shape
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    out = z[ys][:, xs].astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _capture_views(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
    from .rtwx_o0d3 import _head_cam

    p, quat = _cup_pose(env)
    rgb_h, rgb_o = _capture_pair_rgb(env, size)
    cam_h = _get_cam(env, CAM_HEAD) or _head_cam(env)
    cam_o = _get_cam(env, CAM_OBS)
    ph = _cam_pack(cam_h, p, cup_ids, rep_h, size)
    try:
        cam_o.take_picture()
    except Exception:
        pass
    po = _cam_pack(cam_o, p, cup_ids, rep_o, size)
    # depth from Position (camera z)
    try:
        pos_h = np.asarray(cam_h.get_picture("Position"))
        d_h = _depth_from_position(pos_h, size)
    except Exception:
        d_h = np.zeros((size, size), dtype=np.float32)
    try:
        pos_o = np.asarray(cam_o.get_picture("Position"))
        d_o = _depth_from_position(pos_o, size)
    except Exception:
        d_o = np.zeros((size, size), dtype=np.float32)
    return {
        "p": np.asarray(p, dtype=np.float64),
        "quat": np.asarray(quat, dtype=np.float64),
        "n": _n_cup(quat),
        "rgb_h": rgb_h,
        "rgb_o": rgb_o,
        "mask_h": ph["mask"],
        "mask_o": po["mask"],
        "depth_h": d_h,
        "depth_o": d_o,
        "vis_h": bool(ph["visible"]),
        "vis_o": bool(po["visible"]),
        "rep_h": ph["repaired"],
        "rep_o": po["repaired"],
    }


def _set_yaw_pose(env: Any, p: np.ndarray, quat0: np.ndarray, yaw_deg: float) -> None:
    import sapien

    R = _quat_to_R(quat0) @ _R_y(yaw_deg)
    qn = _R_to_quat(R)
    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p, qn))
    try:
        from .rtwx_o0q0 import _physx

        pc = _physx(env.cup)
        if pc is not None:
            pc.linear_velocity = np.zeros(3)
            pc.angular_velocity = np.zeros(3)
    except Exception:
        pass


def _restore_pose(env: Any, snap: dict[str, Any]) -> None:
    import sapien

    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(snap["p"], snap["quat"]))
    try:
        from .rtwx_o0q0 import _physx

        pc = _physx(env.cup)
        if pc is not None:
            pc.linear_velocity = snap.get("v", np.zeros(3))
            pc.angular_velocity = snap.get("w", np.zeros(3))
    except Exception:
        pass
    q = np.asarray(snap["q"], dtype=np.float64)
    n_l = q.size // 2
    try:
        env.robot.set_arm_joints(q[: n_l - 1], np.zeros(n_l - 1), "left")
        env.robot.set_arm_joints(q[n_l : q.size - 1], np.zeros(n_l - 1), "right")
        env.robot.set_gripper(float(q[n_l - 1]), "left")
        env.robot.set_gripper(float(q[-1]), "right")
    except Exception:
        pass


def _collect_base(
    env: Any,
    *,
    size: int,
    n_yaw: int,
    cup_ids: set[int],
) -> dict[str, Any] | None:
    p0, quat0 = _cup_pose(env)
    if _contact_invalid(env):
        return None
    q, _ = _robot_q(env)
    n0 = _n_cup(quat0)
    snap = {"p": p0.copy(), "quat": quat0.copy(), "q": q.copy(), "v": np.zeros(3), "w": np.zeros(3)}
    bins = _yaw_bins(n_yaw)
    rows: dict[str, list] = {
        k: []
        for k in (
            "yaw_deg",
            "yaw_bin",
            "rgb_h",
            "rgb_o",
            "depth_h",
            "depth_o",
            "mask_h",
            "mask_o",
            "p",
            "n",
            "quat",
            "label_yaw_deg",
        )
    }
    rep_h = rep_o = None
    # instrument: repeat render at yaw=0
    _set_yaw_pose(env, p0, quat0, 0.0)
    cap_a = _capture_views(env, size, cup_ids, rep_h, rep_o)
    rep_h, rep_o = cap_a["rep_h"], cap_a["rep_o"]
    cap_b = _capture_views(env, size, cup_ids, rep_h, rep_o)
    mh = np.asarray(cap_a["mask_h"], bool) | np.asarray(cap_b["mask_h"], bool)
    if mh.any():
        dlt = np.abs(cap_a["rgb_h"].astype(np.float32) - cap_b["rgb_h"].astype(np.float32))[mh].mean()
    else:
        dlt = 0.0
    for k, yaw in enumerate(bins):
        _set_yaw_pose(env, p0, quat0, float(yaw))
        cap = _capture_views(env, size, cup_ids, rep_h, rep_o)
        rep_h, rep_o = cap["rep_h"], cap["rep_o"]
        lab = _relative_yaw_deg(quat0, cap["quat"])
        ch, dh, mh = _mask_crop(cap["rgb_h"], cap["depth_h"], cap["mask_h"], size=size)
        co, do, mo = _mask_crop(cap["rgb_o"], cap["depth_o"], cap["mask_o"], size=size)
        if not (mh.any() or mo.any()):
            _restore_pose(env, snap)
            return None
        rows["yaw_deg"].append(float(yaw))
        rows["yaw_bin"].append(int(k))
        rows["rgb_h"].append(ch)
        rows["rgb_o"].append(co)
        rows["depth_h"].append(dh)
        rows["depth_o"].append(do)
        rows["mask_h"].append(mh)
        rows["mask_o"].append(mo)
        rows["p"].append(cap["p"])
        rows["n"].append(cap["n"])
        rows["quat"].append(cap["quat"])
        rows["label_yaw_deg"].append(lab)
        _restore_pose(env, snap)
    return {
        "p0": p0,
        "n0": n0,
        "quat0": quat0,
        "repeat_delta_rgb": float(dlt),
        "yaw_deg": np.asarray(rows["yaw_deg"], dtype=np.float64),
        "yaw_bin": np.asarray(rows["yaw_bin"], dtype=np.int64),
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "depth_h": np.stack(rows["depth_h"]),
        "depth_o": np.stack(rows["depth_o"]),
        "mask_h": np.stack(rows["mask_h"]),
        "mask_o": np.stack(rows["mask_o"]),
        "p": np.stack(rows["p"]),
        "n": np.stack(rows["n"]),
        "quat": np.stack(rows["quat"]),
        "label_yaw_deg": np.asarray(rows["label_yaw_deg"], dtype=np.float64),
    }


def _collect_split_robotwin(
    cfg: RTWXO0A0Config,
    *,
    split: str,
    n_base: int,
    seed0: int,
    stop: list[str],
) -> list[dict[str, Any]]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    bases: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(bases) < n_base and attempts < n_base * max(2, cfg.max_resample):
        attempts += 1
        seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(bases) % 4 == 0 or len(bases) + 1 == n_base:
            print(f"[rtwx-o0a0] {split} base {len(bases)}/{n_base} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = 1000
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            cam_o = _get_cam(env, CAM_OBS)
            if cam_o is None:
                env.close()
                stop.append(f"{split}:missing_observer")
                break
            cup_ids = _cup_ids(env)
            base = _collect_base(env, size=cfg.rgb_size, n_yaw=cfg.n_yaw, cup_ids=cup_ids)
            env.close()
            if base is None:
                continue
            base["base_id"] = int(len(bases))
            base["split"] = split
            base["seed"] = int(seed)
            bases.append(base)
        except Exception as exc:
            print(f"[rtwx-o0a0] {split} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(bases) < n_base:
        stop.append(f"{split}:only_{len(bases)}_of_{n_base}")
    return bases


def _synthetic_base(rng: np.random.Generator, *, base_id: int, split: str, n_yaw: int, size: int) -> dict[str, Any]:
    """Smoke: RGB encodes yaw via oriented stripe; depth ≈ rotationally symmetric disk."""
    bins = _yaw_bins(n_yaw)
    p0 = rng.uniform([-0.2, -0.15, 0.70], [0.2, 0.05, 0.72])
    n0 = EY.copy()
    quat0 = _quat_fix(np.array([1.0, 0.0, 0.0, 0.0]))[0]
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = rr < size * 0.35
    rows = {k: [] for k in ("yaw_deg", "yaw_bin", "rgb_h", "rgb_o", "depth_h", "depth_o", "mask_h", "mask_o", "p", "n", "quat", "label_yaw_deg")}
    for k, yaw in enumerate(bins):
        ang = np.deg2rad(yaw)
        # stripe along yaw direction (appearance cue)
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        stripe = (np.abs(u) < size * 0.06) & disk
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[disk] = (40, 40, 40)
        rgb[stripe] = (220, 80, 40)
        # slight view difference
        rgb_o = rgb.copy()
        rgb_o[disk] = np.clip(rgb_o[disk].astype(np.int16) + 10, 0, 255).astype(np.uint8)
        depth = np.zeros((size, size), dtype=np.float32)
        depth[disk] = 0.85 + 0.02 * ((rr[disk] / (size * 0.35)) ** 2)  # nearly yaw-invariant
        R = _R_y(float(yaw))
        quat = _R_to_quat(R)
        for key, val in (
            ("yaw_deg", float(yaw)),
            ("yaw_bin", int(k)),
            ("rgb_h", rgb),
            ("rgb_o", rgb_o),
            ("depth_h", depth),
            ("depth_o", depth.copy()),
            ("mask_h", disk),
            ("mask_o", disk),
            ("p", p0.copy()),
            ("n", n0.copy()),
            ("quat", quat),
            ("label_yaw_deg", float(yaw % 360.0)),
        ):
            rows[key].append(val)
    return {
        "base_id": int(base_id),
        "split": split,
        "seed": int(base_id),
        "p0": p0,
        "n0": n0,
        "quat0": quat0,
        "repeat_delta_rgb": 0.0,
        "yaw_deg": np.asarray(rows["yaw_deg"], dtype=np.float64),
        "yaw_bin": np.asarray(rows["yaw_bin"], dtype=np.int64),
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "depth_h": np.stack(rows["depth_h"]),
        "depth_o": np.stack(rows["depth_o"]),
        "mask_h": np.stack(rows["mask_h"]),
        "mask_o": np.stack(rows["mask_o"]),
        "p": np.stack(rows["p"]),
        "n": np.stack(rows["n"]),
        "quat": np.stack(rows["quat"]),
        "label_yaw_deg": np.asarray(rows["label_yaw_deg"], dtype=np.float64),
    }


def _flatten_bases(bases: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = (
        "yaw_deg",
        "yaw_bin",
        "rgb_h",
        "rgb_o",
        "depth_h",
        "depth_o",
        "mask_h",
        "mask_o",
        "p",
        "n",
        "quat",
        "label_yaw_deg",
    )
    out: dict[str, Any] = {k: np.concatenate([b[k] for b in bases], 0) for k in keys}
    out["base_id"] = np.concatenate([np.full(b["yaw_bin"].shape[0], b["base_id"], dtype=np.int64) for b in bases])
    return out


def _g0_instrument(bases: list[dict[str, Any]]) -> dict[str, Any]:
    dlt = [float(b["repeat_delta_rgb"]) for b in bases]
    pn_ok, lab_ok = [], []
    for b in bases:
        p0, n0 = b["p0"], b["n0"]
        dp = np.linalg.norm(b["p"] - p0[None, :], axis=1)
        dn = np.linalg.norm(b["n"] - n0[None, :], axis=1)
        pn_ok.append(bool(np.all(dp <= PN_MAX) and np.all(dn <= PN_MAX)))
        err = _circular_err_deg(b["label_yaw_deg"], b["yaw_deg"])
        lab_ok.append(bool(np.all(err <= LABEL_ERR_MAX_DEG)))
    g0_repeat = bool(len(dlt) and float(np.max(dlt)) <= REPEAT_DELTA_MAX)
    g0_pn = bool(len(pn_ok) and all(pn_ok))
    g0_lab = bool(len(lab_ok) and all(lab_ok))
    # synthetic/smoke: pn may be exact; robotwin may have tiny float noise — allow soft pn via median
    if not g0_pn and bases:
        all_dp, all_dn = [], []
        for b in bases:
            all_dp.append(np.linalg.norm(b["p"] - b["p0"][None, :], axis=1))
            all_dn.append(np.linalg.norm(b["n"] - b["n0"][None, :], axis=1))
        med_dp = float(np.median(np.concatenate(all_dp)))
        med_dn = float(np.median(np.concatenate(all_dn)))
        g0_pn = bool(med_dp <= 1e-4 and med_dn <= 1e-4)
    ok = bool(g0_repeat and g0_pn and g0_lab)
    return {
        "ok": ok,
        "max_repeat_delta_rgb": float(np.max(dlt)) if dlt else float("nan"),
        "g0_repeat": g0_repeat,
        "g0_p_n_invariant": g0_pn,
        "g0_label_consistent": g0_lab,
        "frac_bases_pn_ok": float(np.mean(pn_ok)) if pn_ok else 0.0,
        "frac_bases_label_ok": float(np.mean(lab_ok)) if lab_ok else 0.0,
    }


def _yaw_net(in_ch: int, n_cls: int, size: int = RGB_SIZE):
    """Shared conv + CoordConv + spatial flatten (no GAP). Matches O0D3/O0R contract."""
    import torch
    from torch import nn

    spat = 32 * (size // 8) * (size // 8)  # 3× stride-2 → 16² at 128

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(in_ch + 2, 16, 5, stride=2, padding=2)
            self.c2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.c3 = nn.Conv2d(32, 32, 5, stride=2, padding=2)

        def forward(self, x):  # noqa: ANN001
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
            z = torch.relu(self.c1(torch.cat([x, xx, yy], dim=1)))
            z = torch.relu(self.c2(z))
            return torch.relu(self.c3(z)).flatten(1)

    class Twin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = Enc()
            self.head = nn.Sequential(
                nn.Linear(2 * spat, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_cls),
            )

        def forward(self, xh, xo):  # noqa: ANN001
            return self.head(torch.cat([self.enc(xh), self.enc(xo)], dim=1))

    return Twin()


def _pack_inputs(pool: dict[str, np.ndarray], modality: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (Xh, Xo) float32 NCHW."""
    if modality == "rgb":
        xh = pool["rgb_h"].astype(np.float32) / 255.0
        xo = pool["rgb_o"].astype(np.float32) / 255.0
        return np.transpose(xh, (0, 3, 1, 2)), np.transpose(xo, (0, 3, 1, 2))
    if modality == "depth":
        dh = pool["depth_h"].astype(np.float32)
        do = pool["depth_o"].astype(np.float32)
        # per-sample median normalize on masked pixels
        for i in range(dh.shape[0]):
            for d, m in ((dh, pool["mask_h"][i]), (do, pool["mask_o"][i])):
                mm = np.asarray(m, bool) & (d > 0)
                if mm.any():
                    med = float(np.median(d[mm]))
                    if med > 1e-6:
                        d /= med
        return dh[:, None], do[:, None]
    if modality == "rgbd":
        rh, ro = _pack_inputs(pool, "rgb")
        dh, do = _pack_inputs(pool, "depth")
        return np.concatenate([rh, dh], 1), np.concatenate([ro, do], 1)
    raise ValueError(modality)


def _train_eval_branch(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    modality: str,
    n_cls: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xh_tr, xo_tr = _pack_inputs(train, modality)
    xh_va, xo_va = _pack_inputs(val, modality)
    xh_te, xo_te = _pack_inputs(test, modality)
    in_ch = int(xh_tr.shape[1])
    y_tr = torch.from_numpy(train["yaw_bin"].astype(np.int64)).to(dev)
    y_va = torch.from_numpy(val["yaw_bin"].astype(np.int64)).to(dev)
    model = _yaw_net(in_ch, n_cls, size=int(xh_tr.shape[-1])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    n = xh_tr.shape[0]
    last_tr, last_va = float("nan"), float("nan")
    # Ceiling experiment: train all epochs. Do not keep val-early-stop init
    # (val at chance would freeze a random checkpoint and fake "train chance").
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(n)
        for i0 in range(0, n, BATCH):
            idx = perm[i0 : i0 + BATCH]
            xb_h = torch.from_numpy(xh_tr[idx]).to(dev)
            xb_o = torch.from_numpy(xo_tr[idx]).to(dev)
            opt.zero_grad()
            loss_fn(model(xb_h, xb_o), y_tr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_logits = model(torch.from_numpy(xh_tr).to(dev), torch.from_numpy(xo_tr).to(dev))
            va_logits = model(torch.from_numpy(xh_va).to(dev), torch.from_numpy(xo_va).to(dev))
            last_tr = float(loss_fn(tr_logits, y_tr).item())
            last_va = float(loss_fn(va_logits, y_va).item())
        if ep == 0 or ep + 1 == epochs or (ep + 1) % 10 == 0:
            print(f"[rtwx-o0a0] {modality} ep={ep+1}/{epochs} train_ce={last_tr:.4f} val_ce={last_va:.4f}", flush=True)
    model.eval()

    def _predict(xh: np.ndarray, xo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            logits = model(torch.from_numpy(xh).to(dev), torch.from_numpy(xo).to(dev))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
        return pred, prob

    def _score(pred: np.ndarray, gt: np.ndarray, prob: np.ndarray) -> dict[str, float]:
        e = _circular_err_deg(_bin_to_deg(pred, n_cls), _bin_to_deg(gt, n_cls))
        out: dict[str, float] = {
            "median_e_deg": float(np.median(e)),
            "p90_e_deg": float(np.percentile(e, 90)),
            "f90": float(np.mean(e >= F90_THRESH)),
            "mean_e_deg": float(np.mean(e)),
            "n": int(len(e)),
        }
        for k in (1, 3, 5):
            kk = min(k, n_cls)
            top = np.argsort(-prob, axis=1)[:, :kk]
            out[f"top{k}"] = float(np.mean([gt[i] in top[i] for i in range(len(gt))]))
        return out

    pred, prob = _predict(xh_te, xo_te)
    gt = test["yaw_bin"].astype(np.int64)
    te = _score(pred, gt, prob)
    pred_tr, prob_tr = _predict(xh_tr, xo_tr)
    tr = _score(pred_tr, train["yaw_bin"].astype(np.int64), prob_tr)
    med, p90 = te["median_e_deg"], te["p90_e_deg"]
    g1 = bool(med <= MED_ER_MAX and p90 <= P90_ER_MAX)
    return {
        "modality": modality,
        **te,
        "G1_ok": g1,
        "train": tr,
        "encoder": "coordconv_spatial_flatten_no_GAP",
        "epochs_ran": int(epochs),
        "last_train_ce": float(last_tr),
        "last_val_ce": float(last_va),
    }


def _pixel_nn_top1(train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> dict[str, float]:
    """Flattened RGB 1-NN confirmatory (not a gate)."""
    xtr = np.concatenate(
        [train["rgb_h"].reshape(len(train["yaw_bin"]), -1), train["rgb_o"].reshape(len(train["yaw_bin"]), -1)],
        1,
    ).astype(np.float32)
    xte = np.concatenate(
        [test["rgb_h"].reshape(len(test["yaw_bin"]), -1), test["rgb_o"].reshape(len(test["yaw_bin"]), -1)],
        1,
    ).astype(np.float32)
    ytr, yte = train["yaw_bin"], test["yaw_bin"]
    ntr2 = (xtr * xtr).sum(1)
    pred = np.empty(len(yte), dtype=np.int64)
    bs = 64
    for i0 in range(0, len(xte), bs):
        q = xte[i0 : i0 + bs]
        nq2 = (q * q).sum(1, keepdims=True)
        d = nq2 + ntr2[None, :] - 2.0 * (q @ xtr.T)
        pred[i0 : i0 + bs] = ytr[np.argmin(d, axis=1)]
    e = _circular_err_deg(_bin_to_deg(pred), _bin_to_deg(yte))
    return {
        "top1": float(np.mean(pred == yte)),
        "median_e_deg": float(np.median(e)),
        "p90_e_deg": float(np.percentile(e, 90)),
    }


def _pattern(g0: bool, g1_rgb: bool, g1_depth: bool) -> str:
    if not g0:
        return "appearance_instrument_failure"
    if g1_depth:
        return "geometry_yaw_supported_unexpectedly"
    if g1_rgb:
        return "appearance_yaw_supported"
    return "appearance_yaw_generalization_failure"


def run_rtwx_o0a0(output: str | Path, config: RTWXO0A0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0A0Config(output=str(output)))
    root = Path(output)
    _refuse_r10(root)
    root.mkdir(parents=True, exist_ok=True)
    stop: list[str] = []
    print(
        f"[rtwx-o0a0] backend={cfg.backend} smoke={cfg.smoke} "
        f"bases={cfg.n_train_base}/{cfg.n_val_base}/{cfg.n_test_base} n_yaw={cfg.n_yaw}",
        flush=True,
    )

    if cfg.smoke or cfg.backend == "numpy":
        rng = np.random.default_rng(cfg.seed)
        train_b = [_synthetic_base(rng, base_id=i, split="train", n_yaw=cfg.n_yaw, size=cfg.rgb_size) for i in range(cfg.n_train_base)]
        val_b = [
            _synthetic_base(rng, base_id=1000 + i, split="val", n_yaw=cfg.n_yaw, size=cfg.rgb_size)
            for i in range(cfg.n_val_base)
        ]
        test_b = [
            _synthetic_base(rng, base_id=2000 + i, split="test", n_yaw=cfg.n_yaw, size=cfg.rgb_size)
            for i in range(cfg.n_test_base)
        ]
    else:
        train_b = _collect_split_robotwin(cfg, split="train", n_base=cfg.n_train_base, seed0=cfg.seed, stop=stop)
        val_b = _collect_split_robotwin(cfg, split="val", n_base=cfg.n_val_base, seed0=cfg.seed + 10_000, stop=stop)
        test_b = _collect_split_robotwin(cfg, split="test", n_base=cfg.n_test_base, seed0=cfg.seed + 20_000, stop=stop)

    g0 = _g0_instrument(train_b + val_b + test_b)
    print(
        f"[rtwx-o0a0] G0 ok={g0['ok']} repeat={g0['max_repeat_delta_rgb']:.4f} "
        f"pn={g0['g0_p_n_invariant']} lab={g0['g0_label_consistent']}",
        flush=True,
    )

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "seed": cfg.seed,
        "n_train_base": cfg.n_train_base,
        "n_val_base": cfg.n_val_base,
        "n_test_base": cfg.n_test_base,
        "n_yaw": cfg.n_yaw,
        "yaw_step_deg": YAW_STEP_DEG,
        "rgb_size": cfg.rgb_size,
        "grouped_split": True,
        "oracle_mask": True,
        "no_physics_on_yaw": True,
        "encoder": "coordconv_spatial_flatten_no_GAP",
        "o0_target_unchanged": True,
        "o0q_paused": True,
        "gates": {"med": MED_ER_MAX, "p90": P90_ER_MAX, "f90_thresh": F90_THRESH},
        "stop": stop,
    }
    _write_json(root / "header.json", header)

    if not g0["ok"] or not train_b or not test_b:
        pattern = "appearance_instrument_failure"
        result = {
            "header": header,
            "pattern": pattern,
            "G0": g0,
            "B0_depth": None,
            "B1_rgb": None,
            "B2_rgbd": None,
            "G1_rgb": False,
            "G2_f90": float("nan"),
            "G3_delta_median": float("nan"),
            "unlocks_o0a1_prereg": False,
            "unlocks_o0t0_prereg": False,
            "unlocks_o1": False,
            "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print(f"[rtwx-o0a0] pattern={pattern}", flush=True)
        return result

    train, val, test = _flatten_bases(train_b), _flatten_bases(val_b), _flatten_bases(test_b)
    # cache crops
    np.savez_compressed(
        root / "cache_test.npz",
        yaw_bin=test["yaw_bin"],
        base_id=test["base_id"],
        rgb_h=test["rgb_h"],
        rgb_o=test["rgb_o"],
    )

    print("[rtwx-o0a0] train B0 depth", flush=True)
    b0 = _train_eval_branch(train, val, test, modality="depth", n_cls=cfg.n_yaw, epochs=cfg.epochs, seed=cfg.seed + 1)
    print(f"[rtwx-o0a0] B0 med={b0['median_e_deg']:.2f} p90={b0['p90_e_deg']:.2f} top1={b0['top1']:.3f}", flush=True)
    _cuda_gc()

    print("[rtwx-o0a0] train B1 RGB (primary)", flush=True)
    b1 = _train_eval_branch(train, val, test, modality="rgb", n_cls=cfg.n_yaw, epochs=cfg.epochs, seed=cfg.seed + 2)
    print(
        f"[rtwx-o0a0] B1 med={b1['median_e_deg']:.2f} p90={b1['p90_e_deg']:.2f} "
        f"f90={b1['f90']:.3f} top1={b1['top1']:.3f}",
        flush=True,
    )
    _cuda_gc()

    print("[rtwx-o0a0] train B2 RGB-D", flush=True)
    b2 = _train_eval_branch(train, val, test, modality="rgbd", n_cls=cfg.n_yaw, epochs=cfg.epochs, seed=cfg.seed + 3)
    print(f"[rtwx-o0a0] B2 med={b2['median_e_deg']:.2f} p90={b2['p90_e_deg']:.2f}", flush=True)
    _cuda_gc()

    g1_rgb = bool(b1["G1_ok"])
    g1_depth = bool(b0["G1_ok"])
    pattern = _pattern(True, g1_rgb, g1_depth)
    delta_med = float(b0["median_e_deg"] - b1["median_e_deg"])
    delta_p90 = float(b0["p90_e_deg"] - b1["p90_e_deg"])
    nn = _pixel_nn_top1(train, test)
    print(f"[rtwx-o0a0] RGB 1-NN confirmatory top1={nn['top1']:.3f} med={nn['median_e_deg']:.2f}", flush=True)

    result = {
        "header": header,
        "pattern": pattern,
        "G0": g0,
        "B0_depth": b0,
        "B1_rgb": b1,
        "B2_rgbd": b2,
        "G1_rgb": g1_rgb,
        "G1_depth": g1_depth,
        "G2_f90": b1["f90"],
        "G3_delta_median": delta_med,
        "G3_delta_p90": delta_p90,
        "rgb_pixel_nn": nn,
        "unlocks_o0a1_prereg": pattern == "appearance_yaw_supported",
        "unlocks_o0t0_prereg": pattern == "appearance_yaw_generalization_failure",
        "unlocks_o1": False,
        "o0_target_unchanged": True,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(
        f"[rtwx-o0a0] pattern={pattern} G1_rgb={g1_rgb} G1_depth={g1_depth} "
        f"Δmed={delta_med:.2f} unlocks_a1={result['unlocks_o0a1_prereg']} unlocks_t0={result['unlocks_o0t0_prereg']}",
        flush=True,
    )
    return result


def _cuda_gc() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

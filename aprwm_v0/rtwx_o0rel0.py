"""RTWX-O0REL0: relative effective-pose tracking probe (2×2 factorial C0/C1/C2)."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry.center_axis_inference import VisObs, run_a1_center_first
from .geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from .geometry.relative_rigid_registration import (
    REL_ICP_CONFIG_HASH,
    RelativeICPConfig,
    delta_T_from_poses,
    estimate_relative_rigid,
    fuse_voxel_clouds,
)
from .geometry.visibility_reference import AxialVisibilityReference, camera_center_world
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    N_OBS,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
    fuse_points,
)
from .rtwx_o0e0p0 import (
    RGB_SIZE,
    RTWXO0E0P0Config,
    TILT_DEG,
    _capture_one,
    _set_static_pose,
    _stratified_betas,
    sample_pose,
)
from .rtwx_o0e0r1 import _axis_diag_pass, _axis_stats, _ep_stats
from .rtwx_o0e0r2 import _load_p0_split, _load_seg_split
from .rtwx_o0e0r5 import FRESH_TEST_SEED as R5_SEED, S0_TRAIN_SEED
from .rtwx_o0e0r4 import _set_robot_q
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import EP_MAX, MED_MAX, _predict_masks, _train_unet
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0q0 import _robot_q
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0v import CAM_HEAD, CAM_OBS, _get_cam
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0REL0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0rel0.relative_tracking_probe.v1"
SEED = 45601
FRESH_PAIR_SEED = 37606
CAM_Q_SEED = 37607
N_PAIRS = 200
DELTA_BETA_DEG = (10.0, 20.0, 35.0)
DELTA_TRANS_M = (0.0, 0.03, 0.06)
TRANS_DIR_CATALOG = fibonacci_sphere(6, 37608)
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n", "delta_T_gt"})
REGIMES = ("c0", "c1", "c2")
C0_AXIS_MED = 5.0
C0_AXIS_P90 = 10.0
C0_POS_MED = 0.02
C0_POS_P90 = 0.04
ICP_CFG = RelativeICPConfig()
S0_CONFIG_HASH = hashlib.sha256(f"s0_unet_epochs={UNET_EPOCHS}".encode()).hexdigest()
_APR_ROOT = Path(__file__).resolve().parent.parent


def _runs_store() -> Path:
    """Large run artifacts live on the data disk when present."""
    env = os.environ.get("APRWM_RUNS_ROOT")
    if env:
        return Path(env)
    data = Path("/root/autodl-tmp/APR-WM/runs")
    if data.exists():
        return data
    return _APR_ROOT / "runs"


def _resolve_apr(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == "runs":
        rest = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        return (_runs_store() / rest).resolve()
    return (_APR_ROOT / p).resolve()


@dataclass(frozen=True)
class RTWXO0REL0Config:
    output: str = "runs/rtwx_o0rel0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    pair_seed: int = FRESH_PAIR_SEED
    n_pairs: int = N_PAIRS
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0REL0Config) -> RTWXO0REL0Config:
    if cfg.smoke:
        return replace(cfg, n_pairs=40, epochs_unet=4, backend="numpy")
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0REL0 must not write there")


def assert_obs_no_gt(obs: dict[str, Any]) -> None:
    bad = OBS_FORBIDDEN.intersection(obs.keys())
    if bad:
        raise AssertionError(f"oracle leak into estimator obs keys: {sorted(bad)}")


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = a / n
    x, y, z = u
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def _yaw_octant_gamma(rng: np.random.Generator) -> float:
    k = int(rng.integers(0, 8))
    return float((k + rng.uniform(0.0, 1.0)) * (2.0 * np.pi / 8.0))


def _apply_object_motion(
    R0: np.ndarray,
    p0: np.ndarray,
    n0: np.ndarray,
    *,
    delta_beta_deg: float,
    delta_gamma: float,
    trans_dir: np.ndarray,
    trans_mag: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = rng.standard_normal(3)
    u = u - np.dot(u, n0) * n0
    if np.linalg.norm(u) < 1e-9:
        u = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), n0) * n0
    u = u / np.linalg.norm(u)
    r_tilt = _rodrigues(u, np.radians(delta_beta_deg))
    n_tilt = r_tilt @ n0
    r_yaw = _rodrigues(n_tilt, delta_gamma)
    r1 = r_yaw @ r_tilt @ R0
    p1 = p0 + float(trans_mag) * np.asarray(trans_dir, dtype=np.float64)
    n1 = r1 @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n1 = n1 / np.linalg.norm(n1)
    return r1, p1, n1, _R_to_quat_local(r1)


def _R_to_quat_local(R: np.ndarray) -> np.ndarray:
    from .rtwx_o0q0 import _R_to_quat

    return _R_to_quat(R)


def generate_pair_specs(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    betas = _stratified_betas(n)
    rng.shuffle(betas)
    specs: list[dict[str, Any]] = []
    for j, beta in enumerate(betas):
        pose0 = sample_pose(rng, float(beta))
        r0 = np.asarray(pose0["R"], dtype=np.float64)
        p0 = np.asarray(pose0["p"], dtype=np.float64)
        n0 = np.asarray(pose0["n"], dtype=np.float64)
        db = float(rng.choice(DELTA_BETA_DEG))
        tm = float(rng.choice(DELTA_TRANS_M))
        td = TRANS_DIR_CATALOG[j % len(TRANS_DIR_CATALOG)]
        dg = _yaw_octant_gamma(rng)
        r1, p1, n1, q1 = _apply_object_motion(
            r0, p0, n0, delta_beta_deg=db, delta_gamma=dg, trans_dir=td, trans_mag=tm, rng=rng,
        )
        specs.append({
            "pair_id": j,
            "beta0_deg": float(beta),
            "delta_beta_deg": db,
            "delta_trans_m": tm,
            "delta_gamma_rad": dg,
            "trans_dir": td,
            "cam_catalog_idx": j % 12,
            "p0": p0,
            "quat0": np.asarray(pose0["quat"], dtype=np.float64),
            "R0": r0,
            "n0": n0,
            "p1": p1,
            "quat1": q1,
            "R1": r1,
            "n1": n1,
        })
    return specs


def _camera_q1(q0: np.ndarray, catalog_idx: int) -> np.ndarray:
    rng = np.random.default_rng(CAM_Q_SEED + int(catalog_idx))
    delta = np.zeros_like(q0)
    t_m = [0.02, 0.04][catalog_idx % 2]
    theta = np.radians([5.0, 10.0][(catalog_idx // 2) % 2])
    joint = catalog_idx % max(q0.size - 2, 1)
    delta[joint] = theta
    if joint + 1 < q0.size:
        delta[joint + 1] = t_m * 4.0
    delta += rng.normal(0.0, 0.01, size=q0.shape)
    return np.asarray(q0 + delta, dtype=np.float64)


def _extrinsic_pack(cam: Any) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
    k = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
    return e, k


def _capture_frame(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
    cap = _capture_one(env, size, cup_ids, rep_h, rep_o)
    cam_h = _get_cam(env, CAM_HEAD)
    cam_o = _get_cam(env, CAM_OBS)
    e_h, k_h = _extrinsic_pack(cam_h)
    e_o, k_o = _extrinsic_pack(cam_o)
    cap["E_h"] = e_h
    cap["E_o"] = e_o
    cap["K_h"] = k_h
    cap["K_o"] = k_o
    cap["cam_h"] = camera_center_world(e_h)
    cap["cam_o"] = camera_center_world(e_o)
    return cap


def collect_regime_robotwin(
    cfg: RTWXO0REL0Config,
    regime: Literal["c0", "c1", "c2"],
    specs: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rtwx_o0 import TASK, _o0_args
    from .rtwx_o0d import _cup_ids
    from .rtwx_x0 import _setup_env

    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    p0cfg = RTWXO0E0P0Config(robotwin_repo=str(repo), rgb_size=RGB_SIZE)
    env = None
    obs_rows: dict[str, list] = {k: [] for k in (
        "rgb0_h", "rgb0_o", "xyz0_h", "xyz0_o", "E0_h", "E0_o", "K0_h", "K0_o",
        "rgb1_h", "rgb1_o", "xyz1_h", "xyz1_o", "E1_h", "E1_o", "K1_h", "K1_o",
        "vis0_h", "vis0_o", "vis1_h", "vis1_o",
    )}
    gt_rows: dict[str, list] = {k: [] for k in (
        "p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1",
        "delta_beta_deg", "delta_trans_m", "mask0_h", "mask0_o", "mask1_h", "mask1_o",
        "pair_id", "cam_catalog_idx",
    )}
    try:
        env = _setup_env(repo, TASK, seed, 32, args)
        env.check_success = lambda *a, **k: False
        cup_ids = _cup_ids(env)
        q0, _ = _robot_q(env)
        rep_h = rep_o = None
        n = len(specs)
        for i, sp in enumerate(specs):
            if i % 25 == 0 or i + 1 == n:
                print(f"[rtwx-o0rel0] {regime} collect {i}/{n}", flush=True)
            q1 = _camera_q1(q0, int(sp["cam_catalog_idx"]))
            _set_static_pose(env, sp["p0"], sp["quat0"])
            if regime in ("c0", "c2"):
                _set_robot_q(env, q0)
            else:
                _set_robot_q(env, q0)
            cap0 = _capture_frame(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap0["rep_h"], cap0["rep_o"]
            if regime == "c0":
                _set_static_pose(env, sp["p0"], sp["quat0"])
                _set_robot_q(env, q1)
            elif regime == "c1":
                _set_static_pose(env, sp["p1"], sp["quat1"])
                _set_robot_q(env, q0)
            else:
                _set_static_pose(env, sp["p1"], sp["quat1"])
                _set_robot_q(env, q1)
            cap1 = _capture_frame(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap1["rep_h"], cap1["rep_o"]
            for prefix, cap in (("0", cap0), ("1", cap1)):
                obs_rows[f"rgb{prefix}_h"].append(cap["rgb_h"])
                obs_rows[f"rgb{prefix}_o"].append(cap["rgb_o"])
                obs_rows[f"xyz{prefix}_h"].append(cap["xyz_h"])
                obs_rows[f"xyz{prefix}_o"].append(cap["xyz_o"])
                obs_rows[f"E{prefix}_h"].append(cap["E_h"])
                obs_rows[f"E{prefix}_o"].append(cap["E_o"])
                obs_rows[f"K{prefix}_h"].append(cap["K_h"])
                obs_rows[f"K{prefix}_o"].append(cap["K_o"])
                obs_rows[f"vis{prefix}_h"].append(cap["vis_h"])
                obs_rows[f"vis{prefix}_o"].append(cap["vis_o"])
            gt_rows["p0"].append(sp["p0"])
            gt_rows["quat0"].append(sp["quat0"])
            gt_rows["R0"].append(sp["R0"])
            gt_rows["n0"].append(sp["n0"])
            gt_rows["p1"].append(sp["p1"])
            gt_rows["quat1"].append(sp["quat1"])
            gt_rows["R1"].append(sp["R1"])
            gt_rows["n1"].append(sp["n1"])
            gt_rows["delta_beta_deg"].append(sp["delta_beta_deg"])
            gt_rows["delta_trans_m"].append(sp["delta_trans_m"])
            gt_rows["mask0_h"].append(cap0["mask_h"])
            gt_rows["mask0_o"].append(cap0["mask_o"])
            gt_rows["mask1_h"].append(cap1["mask_h"])
            gt_rows["mask1_o"].append(cap1["mask_o"])
            gt_rows["pair_id"].append(sp["pair_id"])
            gt_rows["cam_catalog_idx"].append(sp["cam_catalog_idx"])
        env.close()
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    obs = {k: (np.stack(v) if k.startswith(("rgb", "xyz", "E", "K")) else np.asarray(v)) for k, v in obs_rows.items()}
    gt = {
        "p0": np.stack(gt_rows["p0"]),
        "quat0": np.stack(gt_rows["quat0"]),
        "R0": np.stack(gt_rows["R0"]),
        "n0": np.stack(gt_rows["n0"]),
        "p1": np.stack(gt_rows["p1"]),
        "quat1": np.stack(gt_rows["quat1"]),
        "R1": np.stack(gt_rows["R1"]),
        "n1": np.stack(gt_rows["n1"]),
        "delta_beta_deg": np.asarray(gt_rows["delta_beta_deg"], np.float64),
        "delta_trans_m": np.asarray(gt_rows["delta_trans_m"], np.float64),
        "mask0_h": np.stack(gt_rows["mask0_h"]),
        "mask0_o": np.stack(gt_rows["mask0_o"]),
        "mask1_h": np.stack(gt_rows["mask1_h"]),
        "mask1_o": np.stack(gt_rows["mask1_o"]),
        "pair_id": np.asarray(gt_rows["pair_id"], np.int32),
        "cam_catalog_idx": np.asarray(gt_rows["cam_catalog_idx"], np.int32),
    }
    assert_obs_no_gt(obs)
    return obs, gt


def _numpy_disk_cloud(pose_p: np.ndarray, size: int, rng: np.random.Generator, shift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.28) ** 2
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[disk] = (180, 90, 40)
    xyz = np.full((size, size, 3), np.nan, np.float32)
    base = np.asarray(pose_p, dtype=np.float64) + (shift if shift is not None else 0.0)
    ys, xs = np.where(disk)
    for u, v in zip(xs[: min(96, xs.size)], ys[: min(96, ys.size)]):
        xyz[v, u] = (base + rng.normal(0, 0.003, 3)).astype(np.float32)
    return rgb, disk, xyz


def collect_regime_numpy(
    regime: Literal["c0", "c1", "c2"],
    specs: list[dict[str, Any]],
    *,
    size: int = RGB_SIZE,
    seed: int = FRESH_PAIR_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed + {"c0": 0, "c1": 1, "c2": 2}[regime])
    obs_rows: dict[str, list] = {k: [] for k in (
        "rgb0_h", "rgb0_o", "xyz0_h", "xyz0_o", "rgb1_h", "rgb1_o", "xyz1_h", "xyz1_o",
        "vis0_h", "vis0_o", "vis1_h", "vis1_o",
    )}
    gt_rows: dict[str, list] = {k: [] for k in (
        "p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1",
        "delta_beta_deg", "delta_trans_m", "mask0_h", "mask0_o", "mask1_h", "mask1_o",
        "pair_id", "cam_catalog_idx",
    )}
    cam_shift = np.array([0.02, 0.0, 0.0], dtype=np.float64)
    for sp in specs:
        rgb0_h, m0h, xyz0_h = _numpy_disk_cloud(sp["p0"], size, rng)
        rgb0_o, m0o, xyz0_o = _numpy_disk_cloud(sp["p0"], size, rng, shift=np.array([0.01, 0.0, 0.0]))
        if regime == "c0":
            p1_vis, q1_vis = sp["p0"], sp["quat0"]
            view_shift = cam_shift
        elif regime == "c1":
            p1_vis, q1_vis = sp["p1"], sp["quat1"]
            view_shift = np.zeros(3)
        else:
            p1_vis, q1_vis = sp["p1"], sp["quat1"]
            view_shift = cam_shift
        rgb1_h, m1h, xyz1_h = _numpy_disk_cloud(p1_vis, size, rng, shift=view_shift)
        rgb1_o, m1o, xyz1_o = _numpy_disk_cloud(p1_vis, size, rng, shift=view_shift + np.array([0.01, 0.0, 0.0]))
        for k, v in (
            ("rgb0_h", rgb0_h), ("rgb0_o", rgb0_o), ("xyz0_h", xyz0_h), ("xyz0_o", xyz0_o),
            ("rgb1_h", rgb1_h), ("rgb1_o", rgb1_o), ("xyz1_h", xyz1_h), ("xyz1_o", xyz1_o),
            ("vis0_h", True), ("vis0_o", True), ("vis1_h", True), ("vis1_o", True),
        ):
            obs_rows[k].append(v)
        gt_rows["mask0_h"].append(m0h)
        gt_rows["mask0_o"].append(m0o)
        gt_rows["mask1_h"].append(m1h)
        gt_rows["mask1_o"].append(m1o)
        for k in ("p0", "quat0", "R0", "n0", "p1", "quat1", "R1", "n1"):
            gt_rows[k].append(sp[k])
        gt_rows["delta_beta_deg"].append(sp["delta_beta_deg"])
        gt_rows["delta_trans_m"].append(sp["delta_trans_m"])
        gt_rows["pair_id"].append(sp["pair_id"])
        gt_rows["cam_catalog_idx"].append(sp["cam_catalog_idx"])
    obs = {k: np.stack(v) if k.startswith(("rgb", "xyz")) else np.asarray(v, bool) for k, v in obs_rows.items()}
    gt = {
        "p0": np.stack(gt_rows["p0"]),
        "quat0": np.stack(gt_rows["quat0"]),
        "R0": np.stack(gt_rows["R0"]),
        "n0": np.stack(gt_rows["n0"]),
        "p1": np.stack(gt_rows["p1"]),
        "quat1": np.stack(gt_rows["quat1"]),
        "R1": np.stack(gt_rows["R1"]),
        "n1": np.stack(gt_rows["n1"]),
        "delta_beta_deg": np.asarray(gt_rows["delta_beta_deg"], np.float64),
        "delta_trans_m": np.asarray(gt_rows["delta_trans_m"], np.float64),
        "mask0_h": np.stack(gt_rows["mask0_h"]),
        "mask0_o": np.stack(gt_rows["mask0_o"]),
        "mask1_h": np.stack(gt_rows["mask1_h"]),
        "mask1_o": np.stack(gt_rows["mask1_o"]),
        "pair_id": np.asarray(gt_rows["pair_id"], np.int32),
        "cam_catalog_idx": np.asarray(gt_rows["cam_catalog_idx"], np.int32),
    }
    assert_obs_no_gt(obs)
    return obs, gt


def _write_cache(root: Path, regime: str, obs: dict[str, Any], gt: dict[str, Any]) -> None:
    d = root / "cache" / regime
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "obs.npz", **obs)
    np.savez_compressed(d / "gt.npz", **gt)


def _load_cache(root: Path, regime: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = dict(np.load(root / "cache" / regime / "obs.npz"))
    gt = dict(np.load(root / "cache" / regime / "gt.npz"))
    return obs, gt


def _build_frame_clouds(
    obs: dict[str, Any],
    gt_masks: dict[str, np.ndarray] | None,
    s0: Any,
    pair_i: int,
    *,
    rng: np.random.Generator,
    use_gt_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {}
    clouds: list[np.ndarray] = []
    for t in (0, 1):
        if use_gt_mask and gt_masks is not None:
            mh = gt_masks[f"mask{t}_h"][pair_i]
            mo = gt_masks[f"mask{t}_o"][pair_i]
        else:
            rh = obs[f"rgb{t}_h"][pair_i]
            ro = obs[f"rgb{t}_o"][pair_i]
            mh = _predict_masks(s0, rh[None])[0] > 0.5
            mo = _predict_masks(s0, ro[None])[0] > 0.5
        fused = fuse_points(
            obs[f"xyz{t}_h"][pair_i], mh,
            obs[f"xyz{t}_o"][pair_i], mo,
            rng=rng,
        )
        clouds.append(fused)
        meta[f"mask{t}_empty"] = bool(fused.shape[0] == 0)
        meta[f"mask{t}_area"] = int(mh.sum() + mo.sum())
    return clouds[0], clouds[1], meta


def _eval_pair(
    cloud0: np.ndarray,
    cloud1: np.ndarray,
    *,
    R0: np.ndarray,
    p0: np.ndarray,
    n0: np.ndarray,
    R1: np.ndarray,
    p1: np.ndarray,
    n1: np.ndarray,
    D_O: float,
    identity: bool = False,
) -> dict[str, Any]:
    p0v = voxel_once(cloud0, D_O)
    p1v = voxel_once(cloud1, D_O)

    if identity:
        r_hat = np.eye(3)
        t_hat = np.zeros(3)
        valid = True
        n_corr = 0
    else:
        res = estimate_relative_rigid(p0v, p1v, D_O, ICP_CFG)
        r_hat, t_hat, valid, n_corr = res.R, res.t, res.valid, res.n_corr
    n_hat = r_hat @ n0
    n_hat = n_hat / max(np.linalg.norm(n_hat), 1e-12)
    p_hat = r_hat @ p0 + t_hat
    r_gt, t_gt = delta_T_from_poses(R0, p0, R1, p1)
    return {
        "valid": bool(valid),
        "n_corr": int(n_corr),
        "e_delta_n_deg": float(e_axis_deg(n_hat, n1)),
        "e_delta_p_m": float(np.linalg.norm(p_hat - p1)),
        "R_hat": r_hat,
        "t_hat": t_hat,
        "R_gt": r_gt,
        "t_gt": t_gt,
    }


def voxel_once(cloud: np.ndarray, d_o: float) -> np.ndarray:
    from .geometry.relative_rigid_registration import voxel_reduce

    h = 0.02 * d_o
    return voxel_reduce(np.asarray(cloud, dtype=np.float64), h)


def _overlap_ratio(cloud0: np.ndarray, cloud1: np.ndarray, r_gt: np.ndarray, t_gt: np.ndarray, d_o: float) -> float:
    from scipy.spatial import cKDTree

    p0 = voxel_once(cloud0, d_o)
    p1 = voxel_once(cloud1, d_o)
    if p0.shape[0] < 4 or p1.shape[0] < 4:
        return 0.0
    p0w = (r_gt @ p0.T).T + t_gt
    d, _ = cKDTree(p1).query(p0w, k=1)
    return float((d <= 0.05 * d_o).mean())


def _observation_gate(obs: dict[str, Any], mask_meta: list[dict[str, Any]]) -> dict[str, Any]:
    v0 = np.asarray(obs["vis0_h"], bool) | np.asarray(obs["vis0_o"], bool)
    v1 = np.asarray(obs["vis1_h"], bool) | np.asarray(obs["vis1_o"], bool)
    both = v0 & v1
    p_both = float(both.mean())
    nonempty = []
    for m in mask_meta:
        if m.get("skip"):
            continue
        ok = not (m["mask0_empty"] or m["mask1_empty"])
        nonempty.append(ok)
    nonempty = np.asarray(nonempty, bool)
    p_mask = float(nonempty.mean()) if nonempty.size else 0.0
    ok = bool(p_both >= 0.98 and p_mask >= 0.95)
    return {"ok": ok, "p_visibility_both": p_both, "p_mask_nonempty": p_mask}


def _excitation_gate(e_axis_c1c2: np.ndarray) -> dict[str, Any]:
    stats = _axis_stats(e_axis_c1c2)
    # Identity must fail the formal axis gate (motion sufficiently excites Δn).
    ok = bool(stats["median_e_axis_deg"] > MED_ER_MAX or stats["p90_e_axis_deg"] > P90_ER_MAX)
    return {"ok": ok, "identity_baseline": stats}


def _regime_science_pass(axis_deg: np.ndarray, pos_m: np.ndarray, *, c0: bool = False) -> dict[str, Any]:
    ax = _axis_stats(axis_deg)
    ps = _ep_stats(pos_m)
    ep = float(np.nanmean(pos_m))
    if c0:
        ok = bool(
            ax["median_e_axis_deg"] <= C0_AXIS_MED
            and ax["p90_e_axis_deg"] <= C0_AXIS_P90
            and ps["median_ep_m"] <= C0_POS_MED
            and ps["p90_ep_m"] <= C0_POS_P90
        )
    else:
        ok = bool(
            ax["median_e_axis_deg"] <= MED_ER_MAX
            and ax["p90_e_axis_deg"] <= P90_ER_MAX
            and ep <= EP_MAX
            and ps["median_ep_m"] <= MED_MAX
        )
    return {"ok": ok, "axis": ax, "position": {**ps, "E_p": ep}}


def _pattern(
    *,
    data_ok: bool,
    excite_ok: bool,
    c0_ok: bool,
    c1_ok: bool,
    c2_ok: bool,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    if not data_ok:
        return "relative_tracking_observation_failure", tags
    if not excite_ok:
        return "relative_motion_excitation_failure", tags
    if not c0_ok:
        return "ego_motion_compensation_failure", tags
    if c1_ok and c2_ok:
        return "relative_tracking_supported", tags
    if not c1_ok and not c2_ok:
        return "relative_tracking_insufficient", tags
    if c1_ok and not c2_ok:
        tags.append("camera_view_change_coupled")
    return "relative_tracking_insufficient", tags


def _eval_regime(
    obs: dict[str, Any],
    gt: dict[str, Any],
    s0: Any,
    *,
    D_O: float,
    rng: np.random.Generator,
    lut: AxialVisibilityReference | None = None,
    b2: FrozenQuotientB2 | None = None,
    n_const: np.ndarray | None = None,
    identity: bool = False,
    use_gt_mask: bool = False,
) -> dict[str, Any]:
    n_inst = gt["p0"].shape[0]
    per_pair: list[dict[str, Any]] = []
    e_n, e_p = [], []
    e_n_a1: list[float] = []
    overlaps: list[float] = []
    mask_meta: list[dict[str, Any]] = []
    for i in range(n_inst):
        c0, c1, mmeta = _build_frame_clouds(
            obs, gt if use_gt_mask else None, s0, i, rng=rng, use_gt_mask=use_gt_mask,
        )
        mask_meta.append(mmeta)
        if c0.shape[0] < 8 or c1.shape[0] < 8:
            per_pair.append({"pair_id": int(gt["pair_id"][i]), "valid": False, "skip": True})
            continue
        ev = _eval_pair(
            c0, c1,
            R0=gt["R0"][i], p0=gt["p0"][i], n0=gt["n0"][i],
            R1=gt["R1"][i], p1=gt["p1"][i], n1=gt["n1"][i],
            D_O=D_O, identity=identity,
        )
        ev["pair_id"] = int(gt["pair_id"][i])
        ev["delta_beta_deg"] = float(gt["delta_beta_deg"][i])
        ev["delta_trans_m"] = float(gt["delta_trans_m"][i])
        ev["rho_overlap"] = _overlap_ratio(c0, c1, ev["R_gt"], ev["t_gt"], D_O)
        overlaps.append(ev["rho_overlap"])
        if ev["valid"] or identity:
            e_n.append(ev["e_delta_n_deg"])
            e_p.append(ev["e_delta_p_m"])
        if lut is not None and b2 is not None and n_const is not None and not identity:
            fo = VisObs(
                fused_cloud=c1,
                c_h=cloud_centroid_safe(c1),
                c_o=cloud_centroid_safe(c1),
                cam_h=np.zeros(3),
                cam_o=np.zeros(3),
            )
            if "E1_h" in obs:
                fo = VisObs(
                    fused_cloud=c1,
                    c_h=cloud_centroid_safe(c1),
                    c_o=cloud_centroid_safe(c1),
                    cam_h=camera_center_world(obs["E1_h"][i]),
                    cam_o=camera_center_world(obs["E1_o"][i]),
                )
            a1 = run_a1_center_first(fo, n_const, lut, b2)
            e_n_a1.append(float(e_axis_deg(a1.n1, gt["n1"][i])))
        per_pair.append({k: v for k, v in ev.items() if k not in ("R_hat", "t_hat", "R_gt", "t_gt")})
    return {
        "per_pair": per_pair,
        "mask_meta": mask_meta,
        "axis": _axis_stats(np.asarray(e_n, dtype=np.float64)) if e_n else _axis_stats(np.array([np.nan])),
        "position": _ep_stats(np.asarray(e_p, dtype=np.float64)) if e_p else _ep_stats(np.array([np.nan])),
        "E_p": float(np.nanmean(e_p)) if e_p else float("nan"),
        "a1_n1_axis": _axis_stats(np.asarray(e_n_a1, dtype=np.float64)) if e_n_a1 else None,
        "overlap_median": float(np.median(overlaps)) if overlaps else 0.0,
    }


def cloud_centroid_safe(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape[0] == 0:
        return np.zeros(3)
    return p.mean(0)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_rtwx_o0rel0(output: str | Path, config: RTWXO0REL0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0REL0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    natural_root = _resolve_apr(cfg.natural_cache)
    p0_root = _resolve_apr(cfg.p0_cache)
    r5_root = _resolve_apr(cfg.r5_run)
    repo = Path(cfg.robotwin_repo)
    if not repo.is_absolute():
        repo = Path(cfg.robotwin_repo).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    specs = generate_pair_specs(cfg.n_pairs, cfg.pair_seed)
    _write_json(
        root / "pair_specs.json",
        {
            "seed": cfg.pair_seed,
            "n": cfg.n_pairs,
            "delta_beta_deg": list(DELTA_BETA_DEG),
            "delta_trans_m": list(DELTA_TRANS_M),
        },
    )

    for ri, regime in enumerate(REGIMES):
        cache_obs = root / "cache" / regime / "obs.npz"
        if not cache_obs.is_file():
            print(f"[rtwx-o0rel0] collecting {regime} backend={cfg.backend}", flush=True)
            if cfg.backend == "numpy":
                obs, gt = collect_regime_numpy(regime, specs, seed=cfg.pair_seed + ri)
            else:
                obs, gt = collect_regime_robotwin(cfg, regime, specs, seed=cfg.pair_seed + 1000 * ri)
            _write_cache(root, regime, obs, gt)

    D_O = load_D_O(str(repo) if repo.is_dir() else "/root/RoboTwin")

    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    epochs = cfg.epochs_unet
    print("[rtwx-o0rel0] train S0", flush=True)
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=epochs,
    )

    lut = b2 = n_const = None
    lut_path = r5_root / "visibility_lut.npz"
    if lut_path.is_file():
        lut = AxialVisibilityReference.load(lut_path)
        cad_glb = repo / "assets/objects/021_cup/visual/base0.glb"
        if cad_glb.is_file():
            cad_pts = _fps(load_scaled_cad(str(repo), 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
        else:
            hs = np.linspace(0, 0.088, 64)
            cad_pts = np.array(
                [[0.02 + 0.22 * (h / 0.088) * np.cos(th), h, 0.02 + 0.22 * (h / 0.088) * np.sin(th)]
                 for h in hs for th in np.linspace(0, 2 * np.pi, 32, endpoint=False)],
                dtype=np.float64,
            )
        from scipy.spatial import cKDTree

        tree = cKDTree(cad_hr_profile(cad_pts))
        b2 = FrozenQuotientB2(tree, fibonacci_sphere(K_SPHERE, SPHERE_SEED))
        _, gt_tr = _load_p0_split(p0_root, "train")
        n_const = gt_tr["n_gt"].sum(0)
        n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "pair_seed": cfg.pair_seed,
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "s0_config_hash": S0_CONFIG_HASH,
        "b2_config_hash": R5_B2_CONFIG_HASH,
    }
    _write_json(root / "header.json", header)

    rng = np.random.default_rng(cfg.seed)
    regime_results: dict[str, Any] = {}
    all_mask_meta: list[dict[str, Any]] = []
    identity_e: list[float] = []

    for regime in REGIMES:
        obs, gt = _load_cache(root, regime)
        print(f"[rtwx-o0rel0] eval {regime}", flush=True)
        res = _eval_regime(obs, gt, s0, D_O=D_O, rng=rng, lut=lut, b2=b2, n_const=n_const)
        regime_results[regime] = res
        all_mask_meta.extend(res["mask_meta"])
        if regime in ("c1", "c2"):
            id_res = _eval_regime(obs, gt, s0, D_O=D_O, rng=rng, identity=True)
            identity_e.extend([p["e_delta_n_deg"] for p in id_res["per_pair"] if not p.get("skip")])

    obs0, _ = _load_cache(root, "c0")
    data_gate = _observation_gate(obs0, all_mask_meta)
    excite_gate = _excitation_gate(np.asarray(identity_e, dtype=np.float64))
    c0_pass = _regime_science_pass(
        np.array([p["e_delta_n_deg"] for p in regime_results["c0"]["per_pair"] if not p.get("skip")]),
        np.array([p["e_delta_p_m"] for p in regime_results["c0"]["per_pair"] if not p.get("skip")]),
        c0=True,
    )
    c1_pass = _regime_science_pass(
        np.array([p["e_delta_n_deg"] for p in regime_results["c1"]["per_pair"] if not p.get("skip")]),
        np.array([p["e_delta_p_m"] for p in regime_results["c1"]["per_pair"] if not p.get("skip")]),
    )
    c2_pass = _regime_science_pass(
        np.array([p["e_delta_n_deg"] for p in regime_results["c2"]["per_pair"] if not p.get("skip")]),
        np.array([p["e_delta_p_m"] for p in regime_results["c2"]["per_pair"] if not p.get("skip")]),
    )

    pattern, tags = _pattern(
        data_ok=data_gate["ok"],
        excite_ok=excite_gate["ok"],
        c0_ok=c0_pass["ok"],
        c1_ok=c1_pass["ok"],
        c2_ok=c2_pass["ok"],
    )
    supported = pattern == "relative_tracking_supported"

    per_pair_out: dict[str, list] = {r: regime_results[r]["per_pair"] for r in REGIMES}
    _write_json(root / "per_pair.json", per_pair_out)

    summary = {
        "pattern": pattern,
        "tags": tags,
        "unlocks_relative_tracking_branch": supported,
        "unlocks_o0e1": False,
        "unlocks_o1": False,
        "G_L1a_observation": data_gate,
        "G_L1b_excitation": excite_gate,
        "G_L2_C0": c0_pass,
        "G_L3_C1": c1_pass,
        "G_L3_C2": c2_pass,
        "regimes": {
            r: {
                "axis": regime_results[r]["axis"],
                "position": {**regime_results[r]["position"], "E_p": regime_results[r]["E_p"]},
                "overlap_median": regime_results[r]["overlap_median"],
                "a1_baseline_n1": regime_results[r].get("a1_n1_axis"),
            }
            for r in REGIMES
        },
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})

    print(f"[rtwx-o0rel0] pattern={pattern}", flush=True)
    return summary

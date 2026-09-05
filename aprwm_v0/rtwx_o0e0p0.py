"""RTWX-O0E0P0: controlled-pose observation qualification. No B2 science."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args
from .rtwx_o0d import _cup_ids
from .rtwx_o0d3 import _head_cam
from .rtwx_o0e0 import R0_from_n, e_axis_deg, n_from_R
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g2r import _cam_pack, _capture_pair_rgb
from .rtwx_o0q0 import _R_to_quat, _R_y
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, _get_cam
from .rtwx_x0 import _setup_env
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0P0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0p0.controlled_pose_qualification.v1"
SEED = 37601
RGB_SIZE = 128
TILT_DEG = (0.0, 10.0, 20.0, 35.0, 50.0)
N_TRAIN = 250
N_VAL = 100
N_TEST = 200
P_XY = ((-0.15, 0.15), (-0.20, 0.05))
P_Z = 0.74
EXCITE_ALPHA_DEG = 15.0
EXCITE_RHO = 0.30
YAW_OCTANTS = 8
YAW_OCTANTS_MIN = 6
EZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class RTWXO0E0P0Config:
    output: str = "runs/rtwx_o0e0p0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    n_test: int = N_TEST
    rgb_size: int = RGB_SIZE
    smoke: bool = False
    seed_attempts: int = 32


def _lock(cfg: RTWXO0E0P0Config) -> RTWXO0E0P0Config:
    if cfg.smoke:
        return replace(cfg, n_train=10, n_val=5, n_test=10, rgb_size=RGB_SIZE)
    if cfg.backend != "robotwin":
        return cfg
    return replace(cfg, seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, rgb_size=RGB_SIZE)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0P0 must not write there")


def n_from_tilt(beta_deg: float, alpha_rad: float) -> np.ndarray:
    b = np.radians(float(beta_deg))
    return np.array([np.sin(b) * np.cos(alpha_rad), np.sin(b) * np.sin(alpha_rad), np.cos(b)], dtype=np.float64)


def sample_pose(rng: np.random.Generator, beta_deg: float) -> dict[str, np.ndarray]:
    alpha = float(rng.uniform(0.0, 2.0 * np.pi))
    gamma = float(rng.uniform(0.0, 2.0 * np.pi))
    n = n_from_tilt(beta_deg, alpha)
    R = R0_from_n(n) @ _R_y(np.degrees(gamma))
    p = np.array(
        [rng.uniform(*P_XY[0]), rng.uniform(*P_XY[1]), P_Z],
        dtype=np.float64,
    )
    return {
        "p": p,
        "R": R,
        "n": n_from_R(R),
        "quat": _R_to_quat(R),
        "beta_deg": np.array(float(beta_deg)),
        "alpha_rad": np.array(alpha),
        "gamma_rad": np.array(gamma),
    }


def _set_static_pose(env: Any, p: np.ndarray, quat: np.ndarray) -> None:
    import sapien

    getattr(env.cup, "actor", env.cup).set_pose(sapien.Pose(p, quat))
    try:
        from .rtwx_o0q0 import _physx

        pc = _physx(env.cup)
        if pc is not None:
            pc.linear_velocity = np.zeros(3)
            pc.angular_velocity = np.zeros(3)
    except Exception:
        pass


def _capture_one(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
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
    return {
        "rgb_h": rgb_h,
        "rgb_o": rgb_o,
        "mask_h": ph["mask"],
        "mask_o": po["mask"],
        "xyz_h": ph["xyz"],
        "xyz_o": po["xyz"],
        "vis_h": bool(ph["visible"]),
        "vis_o": bool(po["visible"]),
        "fov_h": bool(ph["in_fov"]),
        "fov_o": bool(po["in_fov"]),
        "area_h": int(np.asarray(ph["mask"]).sum()),
        "area_o": int(np.asarray(po["mask"]).sum()),
        "p_read": np.asarray(p, dtype=np.float64),
        "quat_read": np.asarray(quat, dtype=np.float64),
        "rep_h": ph["repaired"],
        "rep_o": po["repaired"],
    }


def _stratified_betas(n: int) -> np.ndarray:
    k = len(TILT_DEG)
    reps = np.array(TILT_DEG * (n // k + 1), dtype=np.float64)[:n]
    return reps


def collect_split_robotwin(cfg: RTWXO0E0P0Config, split: str, n: int, seed: int, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0rgb import _patch_curobo_planner

    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    rng = np.random.default_rng(seed)
    betas = _stratified_betas(n)
    rng.shuffle(betas)
    env = None
    try:
        env = _setup_env(repo, TASK, seed, cfg.seed_attempts, args)
        env.check_success = lambda *a, **k: False
        env.step_lim = 1000
        cam_o = _get_cam(env, CAM_OBS)
        if cam_o is None:
            stop.append(f"{split}:missing_observer")
            env.close()
            raise RuntimeError("missing observer")
        cup_ids = _cup_ids(env)
        rows: dict[str, list] = {k: [] for k in (
            "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o",
            "vis_h", "vis_o", "fov_h", "fov_o", "area_h", "area_o",
            "p", "quat", "n", "beta_deg", "alpha_rad", "gamma_rad",
        )}
        rep_h = rep_o = None
        for i, beta in enumerate(betas):
            if i % 25 == 0 or i + 1 == n:
                print(f"[rtwx-o0e0p0] {split} {i}/{n}", flush=True)
            pose = sample_pose(rng, float(beta))
            _set_static_pose(env, pose["p"], pose["quat"])
            cap = _capture_one(env, cfg.rgb_size, cup_ids, rep_h, rep_o)
            rep_h, rep_o = cap["rep_h"], cap["rep_o"]
            rows["rgb_h"].append(cap["rgb_h"])
            rows["rgb_o"].append(cap["rgb_o"])
            rows["mask_h"].append(cap["mask_h"])
            rows["mask_o"].append(cap["mask_o"])
            rows["xyz_h"].append(cap["xyz_h"])
            rows["xyz_o"].append(cap["xyz_o"])
            rows["vis_h"].append(cap["vis_h"])
            rows["vis_o"].append(cap["vis_o"])
            rows["fov_h"].append(cap["fov_h"])
            rows["fov_o"].append(cap["fov_o"])
            rows["area_h"].append(cap["area_h"])
            rows["area_o"].append(cap["area_o"])
            rows["p"].append(pose["p"])
            rows["quat"].append(pose["quat"])
            rows["n"].append(pose["n"])
            rows["beta_deg"].append(float(beta))
            rows["alpha_rad"].append(float(pose["alpha_rad"]))
            rows["gamma_rad"].append(float(pose["gamma_rad"]))
        env.close()
    except Exception as exc:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    out: dict[str, Any] = {
        "rgb_h": np.stack(rows["rgb_h"]),
        "rgb_o": np.stack(rows["rgb_o"]),
        "mask_h": np.stack(rows["mask_h"]),
        "mask_o": np.stack(rows["mask_o"]),
        "xyz_h": np.stack(rows["xyz_h"]),
        "xyz_o": np.stack(rows["xyz_o"]),
        "vis_h": np.asarray(rows["vis_h"], bool),
        "vis_o": np.asarray(rows["vis_o"], bool),
        "fov_h": np.asarray(rows["fov_h"], bool),
        "fov_o": np.asarray(rows["fov_o"], bool),
        "area_h": np.asarray(rows["area_h"], np.int32),
        "area_o": np.asarray(rows["area_o"], np.int32),
        "p": np.stack(rows["p"]),
        "quat": np.stack(rows["quat"]),
        "n": np.stack(rows["n"]),
        "beta_deg": np.asarray(rows["beta_deg"], np.float64),
        "alpha_rad": np.asarray(rows["alpha_rad"], np.float64),
        "gamma_rad": np.asarray(rows["gamma_rad"], np.float64),
    }
    return out


def collect_split_numpy(cfg: RTWXO0E0P0Config, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    betas = _stratified_betas(n)
    rng.shuffle(betas)
    size = cfg.rgb_size
    rows: dict[str, list] = {k: [] for k in (
        "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o",
        "vis_h", "vis_o", "fov_h", "fov_o", "area_h", "area_o",
        "p", "quat", "n", "beta_deg", "alpha_rad", "gamma_rad",
    )}
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.28) ** 2
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[disk] = (180, 90, 40)
    for beta in betas:
        pose = sample_pose(rng, float(beta))
        xyz = np.full((size, size, 3), np.nan, np.float32)
        ys, xs = np.where(disk)
        for j, (u, v) in enumerate(zip(xs[:80], ys[:80])):
            xyz[v, u] = pose["p"]
        rows["rgb_h"].append(rgb)
        rows["rgb_o"].append(rgb.copy())
        rows["mask_h"].append(disk)
        rows["mask_o"].append(disk.copy())
        rows["xyz_h"].append(xyz)
        rows["xyz_o"].append(xyz.copy())
        rows["vis_h"].append(True)
        rows["vis_o"].append(True)
        rows["fov_h"].append(True)
        rows["fov_o"].append(True)
        rows["area_h"].append(int(disk.sum()))
        rows["area_o"].append(int(disk.sum()))
        rows["p"].append(pose["p"])
        rows["quat"].append(pose["quat"])
        rows["n"].append(pose["n"])
        rows["beta_deg"].append(float(beta))
        rows["alpha_rad"].append(float(pose["alpha_rad"]))
        rows["gamma_rad"].append(float(pose["gamma_rad"]))
    return {k: (np.stack(rows[k]) if k not in ("vis_h", "vis_o", "fov_h", "fov_o", "beta_deg", "alpha_rad", "gamma_rad", "area_h", "area_o") else np.asarray(rows[k])) for k in rows}


def _write_isolated(root: Path, split: str, pool: dict[str, Any]) -> None:
    d = root / "cache" / split
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "obs.npz",
        rgb_h=pool["rgb_h"],
        rgb_o=pool["rgb_o"],
        xyz_h=pool["xyz_h"],
        xyz_o=pool["xyz_o"],
        mask_h=pool["mask_h"],
        mask_o=pool["mask_o"],
        vis_h=pool["vis_h"],
        vis_o=pool["vis_o"],
        fov_h=pool["fov_h"],
        fov_o=pool["fov_o"],
        area_h=pool["area_h"],
        area_o=pool["area_o"],
    )
    np.savez_compressed(
        d / "gt.npz",
        p_gt=pool["p"],
        quat_gt=pool["quat"],
        n_gt=pool["n"],
        beta_deg=pool["beta_deg"],
        alpha_rad=pool["alpha_rad"],
        gamma_rad=pool["gamma_rad"],
    )


def _pattern(*, g0a: bool, g0b: bool, g0c: bool) -> str:
    if not g0a:
        return "observation_support_failure"
    if not g0b:
        return "axis_excitation_failure"
    if not g0c:
        return "yaw_nuisance_coverage_failure"
    return "controlled_pose_observation_qualified"


def run_rtwx_o0e0p0(output: str | Path, config: RTWXO0E0P0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0P0Config(output=str(output)))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0p0] backend={cfg.backend} smoke={cfg.smoke} n={cfg.n_train}/{cfg.n_val}/{cfg.n_test}", flush=True)
    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "seed": cfg.seed,
        "tilt_deg": list(TILT_DEG),
        "cameras": [CAM_HEAD, CAM_OBS],
        "no_b2": True,
        "perception_only_static": True,
        "not_natural_task_claim": True,
        "b2_untested": True,
        "gates": {"P_any": P_ANY_AGG, "r_excite": EXCITE_RHO, "yaw_octants_min": YAW_OCTANTS_MIN},
    }
    _write_json(root / "header.json", header)
    stop: list[str] = []
    splits = {}
    ns = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}
    seeds = {"train": cfg.seed, "val": cfg.seed + 1000, "test": cfg.seed + 2000}
    for split in ("train", "val", "test"):
        cache = root / "cache" / split / "gt.npz"
        if cache.is_file() and (root / "cache" / split / "obs.npz").is_file():
            print(f"[rtwx-o0e0p0] {split}: cache", flush=True)
            gt = np.load(cache)
            obs = np.load(root / "cache" / split / "obs.npz")
            splits[split] = {
                "vis_h": np.asarray(obs["vis_h"]),
                "vis_o": np.asarray(obs["vis_o"]),
                "fov_h": np.asarray(obs["fov_h"]),
                "fov_o": np.asarray(obs["fov_o"]),
                "area_h": np.asarray(obs["area_h"]),
                "area_o": np.asarray(obs["area_o"]),
                "n": np.asarray(gt["n_gt"]),
                "beta_deg": np.asarray(gt["beta_deg"]),
                "gamma_rad": np.asarray(gt["gamma_rad"]),
            }
            continue
        print(f"[rtwx-o0e0p0] collect {split}", flush=True)
        if cfg.backend == "numpy" or cfg.smoke:
            pool = collect_split_numpy(cfg, ns[split], seeds[split])
        else:
            pool = collect_split_robotwin(cfg, split, ns[split], seeds[split], stop)
        _write_isolated(root, split, pool)
        splits[split] = pool
        # drop rgb from RAM
        pool.pop("rgb_h", None)
        pool.pop("rgb_o", None)

    te, tr = splits["test"], splits["train"]
    vis_any = te["vis_h"] | te["vis_o"]
    p_h = float(np.mean(te["vis_h"]))
    p_o = float(np.mean(te["vis_o"]))
    p_any = float(np.mean(vis_any))
    g0a = bool(p_any >= P_ANY_AGG)
    fail = ~vis_any
    diag_fail = {
        "n_fail": int(fail.sum()),
        "frac_fov_h": float(np.mean(te["fov_h"][fail])) if fail.any() else float("nan"),
        "frac_fov_o": float(np.mean(te["fov_o"][fail])) if fail.any() else float("nan"),
        "median_area_h_fail": float(np.median(te["area_h"][fail])) if fail.any() else float("nan"),
        "median_area_o_fail": float(np.median(te["area_o"][fail])) if fail.any() else float("nan"),
    }
    print(f"[rtwx-o0e0p0] G0a P_any={p_any:.3f} P_h={p_h:.3f} P_o={p_o:.3f} ok={g0a}", flush=True)

    n_sum = np.asarray(tr["n"], dtype=np.float64).sum(0)
    n_const = n_sum / max(np.linalg.norm(n_sum), 1e-12)
    ang = np.array([e_axis_deg(te["n"][i], n_const) for i in range(te["n"].shape[0])])
    r_excite = float(np.mean(ang > EXCITE_ALPHA_DEG))
    g0b = bool(r_excite >= EXCITE_RHO)
    # generator check: equal tilt bins
    beta_frac = {float(b): float(np.mean(np.isclose(te["beta_deg"], b))) for b in TILT_DEG}
    print(f"[rtwx-o0e0p0] G0b r_excite={r_excite:.3f} n_const={n_const.tolist()} ok={g0b}", flush=True)

    octant = (np.asarray(te["gamma_rad"]) % (2 * np.pi) / (2 * np.pi / YAW_OCTANTS)).astype(int) % YAW_OCTANTS
    per_beta_oct = {}
    g0c_ok = True
    for b in TILT_DEG:
        m = np.isclose(te["beta_deg"], b)
        occ = int(len(np.unique(octant[m]))) if m.any() else 0
        per_beta_oct[str(b)] = occ
        if occ < YAW_OCTANTS_MIN:
            g0c_ok = False
    print(f"[rtwx-o0e0p0] G0c yaw octants {per_beta_oct} ok={g0c_ok}", flush=True)

    if not g0a:
        pattern = "observation_support_failure"
    elif not g0b:
        pattern = "axis_excitation_failure"
    elif not g0c_ok:
        pattern = "yaw_nuisance_coverage_failure"
    else:
        pattern = "controlled_pose_observation_qualified"

    result = {
        "header": header,
        "pattern": pattern,
        "G0a": {"ok": g0a, "P_any": p_any, "P_head": p_h, "P_observer": p_o, "fail": diag_fail},
        "G0b": {
            "ok": g0b,
            "r_excite": r_excite,
            "n_const": n_const.tolist(),
            "median_ang_deg": float(np.median(ang)),
            "beta_frac_test": beta_frac,
        },
        "G0c": {"ok": g0c_ok, "octants_per_beta": per_beta_oct, "min_required": YAW_OCTANTS_MIN},
        "B2_not_run": True,
        "b2_untested": True,
        "unlocks_o0e0_science_on_p0_cache": pattern == "controlled_pose_observation_qualified",
        "unlocks_o0e1": False,
        "unlocks_o1": False,
        "stop": stop,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    print(f"[rtwx-o0e0p0] pattern={pattern} unlocks_science={result['unlocks_o0e0_science_on_p0_cache']}", flush=True)
    return result

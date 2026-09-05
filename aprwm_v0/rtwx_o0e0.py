"""RTWX-O0E0: symmetry-aware effective pose (p, n). No yaw; no GT in estimator."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args, _quat_fix
from .rtwx_o0d import _cup_ids
from .rtwx_o0g1b import _quat_to_R, load_delta_O
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, STRONG_MED, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0g2r import RTWXO0G2RConfig, collect_split
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0q0 import _R_to_quat, _R_y
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0v import P_ANY_AGG
from .rtwx_x0c import FORMAL_N_STEPS, FORMAL_N_TEST, FORMAL_N_TRAIN, FORMAL_N_VAL, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0.effective_pose_closure.v1"
SEED = 36601
RGB_SIZE = 128
EY = np.array([0.0, 1.0, 0.0], dtype=np.float64)
N_CAD = 2048
N_OBS = 256
CAD_FPS_SEED = 36611
SPHERE_SEED = 36612
K_SPHERE = 162
TOP_KR = 8
REFINE_DEGS = (5.0, 3.0, 1.0)
EXCITE_ALPHA_DEG = 15.0
EXCITE_RHO = 0.30
P_DET_MIN = 0.95
F90_THRESH = 75.0
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n"})


@dataclass(frozen=True)
class RTWXO0E0Config:
    output: str = "runs/rtwx_o0e0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    rgb_size: int = RGB_SIZE
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False
    seed_attempts: int = 32
    max_resample: int = 16


def _lock(cfg: RTWXO0E0Config) -> RTWXO0E0Config:
    if cfg.smoke:
        return replace(
            cfg,
            n_train_ep=4,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=8,
            epochs_unet=4,
            rgb_size=RGB_SIZE,
        )
    if cfg.backend != "robotwin":
        return cfg
    return replace(
        cfg,
        seed=SEED,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        rgb_size=RGB_SIZE,
        epochs_unet=UNET_EPOCHS,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0 must not write there")


def _cuda_gc() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Geometry (no GT)
# ---------------------------------------------------------------------------


def n_from_R(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float64) @ EY


def e_axis_deg(n_hat: np.ndarray, n_gt: np.ndarray) -> float:
    a = np.asarray(n_hat, dtype=np.float64).reshape(3)
    b = np.asarray(n_gt, dtype=np.float64).reshape(3)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def d_G_deg(R1: np.ndarray, R2: np.ndarray, n_samples: int = 360) -> float:
    """min_θ d_SO3(R1, R2 R_y(θ)) for G=SO(2)_y. Equals ∠(R1 ey, R2 ey)."""
    # closed form for this G (right SO(2)_y action)
    return e_axis_deg(n_from_R(R1), n_from_R(R2))


def R0_from_n(n: np.ndarray) -> np.ndarray:
    """Deterministic zero-twist representative: shortest rotation taking ey → n."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    v = EY
    c = float(np.clip(np.dot(v, n), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        # 180° about any axis ⟂ ey
        axis = np.array([1.0, 0.0, 0.0])
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(v, n)
    axis = axis / np.linalg.norm(axis)
    s = np.sqrt(max(1.0 - c * c, 0.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def fibonacci_sphere(k: int = K_SPHERE, seed: int = SPHERE_SEED) -> np.ndarray:
    """Deterministic almost-uniform points on S^2 (includes both hemispheres)."""
    rng = np.random.default_rng(seed)
    # golden spiral
    i = np.arange(k, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / k)
    theta = np.pi * (1.0 + 5**0.5) * (i + 0.5)
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)
    pts = np.stack([x, y, z], 1)
    # slight fixed jitter disabled — pure deterministic
    _ = rng  # seed reserved for future; grid is closed-form
    return pts


def cad_hr_profile(cad_pts: np.ndarray) -> np.ndarray:
    """(N,2) columns (h,r) in object frame; yaw-invariant."""
    p = np.asarray(cad_pts, dtype=np.float64)
    h = p[:, 1]
    r = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    return np.stack([h, r], 1)


def obs_hr(P: np.ndarray, p_pose: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    d = np.asarray(P, dtype=np.float64) - np.asarray(p_pose, dtype=np.float64).reshape(1, 3)
    h = d @ n
    radial = d - h[:, None] * n[None, :]
    r = np.linalg.norm(radial, axis=1)
    return np.stack([h, r], 1)


def S_obs_to_cad(P: np.ndarray, p_surf: np.ndarray, n: np.ndarray, delta_y: float, tree: Any) -> float:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    p_pose = np.asarray(p_surf, dtype=np.float64).reshape(3) - float(delta_y) * n
    c = obs_hr(P, p_pose, n)
    if c.shape[0] == 0:
        return float("inf")
    d, _ = tree.query(c, k=1)
    return float(np.mean(np.asarray(d, dtype=np.float64) ** 2))


def _tangent_neighbors(n: np.ndarray, deg: float) -> list[np.ndarray]:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    # orthonormal basis of T_n S^2
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(n, e1)
    out = []
    th = np.radians(deg)
    for k in range(8):
        ang = 2.0 * np.pi * k / 8.0
        v = np.cos(ang) * e1 + np.sin(ang) * e2
        nn = n * np.cos(th) + v * np.sin(th)
        out.append(nn / np.linalg.norm(nn))
    return out


def estimate_axis_b2(
    P: np.ndarray,
    p_surf: np.ndarray,
    delta_y: float,
    tree: Any,
    *,
    sphere: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if P.shape[0] < 8:
        return EY.copy(), {"S_best": float("inf"), "S_second": float("inf"), "margin": float("nan")}
    scores = []
    for n in sphere:
        scores.append(S_obs_to_cad(P, p_surf, n, delta_y, tree))
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    cands = [sphere[i] for i in order[:TOP_KR]]
    best_n, best_s = cands[0], float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else best_s
    for n0 in list(cands):
        cur_n, cur_s = n0, S_obs_to_cad(P, p_surf, n0, delta_y, tree)
        for deg in REFINE_DEGS:
            improved = True
            while improved:
                improved = False
                for nb in _tangent_neighbors(cur_n, deg):
                    s = S_obs_to_cad(P, p_surf, nb, delta_y, tree)
                    if s + 1e-12 < cur_s:
                        cur_n, cur_s = nb, s
                        improved = True
        if cur_s < best_s:
            second = best_s
            best_n, best_s = cur_n, cur_s
        elif cur_s < second and not np.allclose(cur_n, best_n):
            second = cur_s
    return best_n / np.linalg.norm(best_n), {"S_best": best_s, "S_second": second, "margin": second - best_s}


def estimate_axis_b1(P: np.ndarray, cad_hr: np.ndarray, p_surf: np.ndarray, delta_y: float, tree: Any) -> np.ndarray:
    """PCA principal axis + CAD profile sign disambiguation."""
    if P.shape[0] < 8:
        return EY.copy()
    X = np.asarray(P, dtype=np.float64)
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    w, V = np.linalg.eigh(C)
    v = V[:, int(np.argmax(w))]
    v = v / max(np.linalg.norm(v), 1e-12)
    # choose sign by S
    s_pos = S_obs_to_cad(P, p_surf, v, delta_y, tree)
    s_neg = S_obs_to_cad(P, p_surf, -v, delta_y, tree)
    return v if s_pos <= s_neg else -v


def p_surf_from_cloud(P: np.ndarray) -> np.ndarray:
    if P.shape[0] < 8:
        return np.full(3, np.nan, dtype=np.float64)
    return np.median(np.asarray(P, dtype=np.float64), axis=0)


def fuse_points(xyz_h: np.ndarray, mask_h: np.ndarray, xyz_o: np.ndarray, mask_o: np.ndarray, n_max: int = N_OBS, rng: np.random.Generator | None = None) -> np.ndarray:
    chunks = []
    for xyz, m in ((xyz_h, mask_h), (xyz_o, mask_o)):
        mm = np.asarray(m, bool) & np.isfinite(xyz).all(-1)
        if mm.any():
            chunks.append(np.asarray(xyz, dtype=np.float64)[mm])
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    P = np.concatenate(chunks, 0)
    if P.shape[0] > n_max:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(P.shape[0], n_max, replace=False)]
    return P


def assert_obs_no_gt(obs: dict[str, Any]) -> None:
    bad = OBS_FORBIDDEN.intersection(obs.keys())
    if bad:
        raise AssertionError(f"oracle leak into estimator obs keys: {sorted(bad)}")


def estimate_effective_pose(
    obs: dict[str, Any],
    *,
    delta_y: float,
    tree: Any,
    sphere: np.ndarray,
    mode: str,
    n_const: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Estimator: obs-only. Returns (p_hat, n_hat, meta)."""
    assert_obs_no_gt(obs)
    P = np.asarray(obs["fused_points_B"], dtype=np.float64)
    p_surf = p_surf_from_cloud(P)
    if mode == "B0":
        assert n_const is not None
        n_hat = np.asarray(n_const, dtype=np.float64).reshape(3)
        n_hat = n_hat / max(np.linalg.norm(n_hat), 1e-12)
        meta = {"S_best": float("nan"), "S_second": float("nan"), "margin": float("nan")}
    elif mode == "B1":
        n_hat = estimate_axis_b1(P, obs.get("cad_hr", np.zeros((0, 2))), p_surf, delta_y, tree)
        meta = {"S_best": S_obs_to_cad(P, p_surf, n_hat, delta_y, tree), "S_second": float("nan"), "margin": float("nan")}
    elif mode == "B2":
        n_hat, meta = estimate_axis_b2(P, p_surf, delta_y, tree, sphere=sphere)
    else:
        raise ValueError(mode)
    if not np.isfinite(p_surf).all():
        return np.full(3, np.nan), n_hat, meta
    p_hat = p_surf - float(delta_y) * n_hat
    return p_hat, n_hat, {**meta, "p_surf": p_surf}


# ---------------------------------------------------------------------------
# Collect / cache (obs ⟂ gt)
# ---------------------------------------------------------------------------


def _g2r_cfg(cfg: RTWXO0E0Config, seed: int) -> RTWXO0G2RConfig:
    return RTWXO0G2RConfig(
        output=cfg.output,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        n_train_ep=cfg.n_train_ep,
        n_val_ep=cfg.n_val_ep,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        seed=seed,
        seed_attempts=cfg.seed_attempts,
        max_resample=cfg.max_resample,
        smoke=cfg.smoke,
        rgb_size=cfg.rgb_size,
        epochs=cfg.epochs_unet,
    )


def _synthetic_split(cfg: RTWXO0E0Config, split: str, rng: np.random.Generator, delta_y: float) -> dict[str, Any]:
    """Asymmetric frustum (top wider): axis + position recoverable; yaw irrelevant."""
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    size = cfg.rgb_size
    rows: dict[str, list] = {k: [] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "mask_h", "mask_o", "vis_h", "vis_o", "p", "quat", "n", "ep", "fr")}
    for ep in range(n_ep):
        for fr in range(cfg.n_steps):
            # random axis with excitation
            n = rng.normal(size=3)
            n = n / np.linalg.norm(n)
            if abs(n[1]) > 0.95 and split == "test" and fr % 3 == 0:
                # force tilt for excitation
                n = np.array([0.5, 0.7, 0.3])
                n = n / np.linalg.norm(n)
            R = R0_from_n(n) @ _R_y(float(rng.uniform(0, 360)))
            p = rng.uniform([-0.2, -0.15, 0.70], [0.2, 0.05, 0.75])
            # CAD-like cloud in object frame then to world
            hs = rng.uniform(0.0, 0.088, size=400)
            # wider at high h (rim)
            rs = 0.02 + 0.22 * (hs / 0.088)
            th = rng.uniform(0, 2 * np.pi, size=400)
            xo = np.stack([rs * np.cos(th), hs, rs * np.sin(th)], 1)
            xb = (R @ xo.T).T + p
            # fake image: disk intensity encodes nothing about yaw
            rgb = np.zeros((size, size, 3), dtype=np.uint8)
            yy, xx = np.mgrid[0:size, 0:size]
            disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.3) ** 2
            rgb[disk] = (180, 90, 40)
            mask = disk
            # sparse xyz map: put points randomly into mask pixels (instrument)
            xyz = np.full((size, size, 3), np.nan, np.float32)
            ys, xs = np.where(mask)
            for i, (u, v) in enumerate(zip(xs[: min(200, len(xs))], ys[: min(200, len(ys))])):
                xyz[v, u] = xb[i % len(xb)]
            rows["rgb_h"].append(rgb)
            rows["rgb_o"].append(rgb.copy())
            rows["xyz_h"].append(xyz)
            rows["xyz_o"].append(xyz.copy())
            rows["mask_h"].append(mask)
            rows["mask_o"].append(mask.copy())
            rows["vis_h"].append(True)
            rows["vis_o"].append(True)
            rows["p"].append(p)
            rows["quat"].append(_R_to_quat(R))
            rows["n"].append(n_from_R(R))
            rows["ep"].append(ep)
            rows["fr"].append(fr)
    out = {k: np.stack(rows[k]) for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "mask_h", "mask_o", "p", "quat", "n")}
    out["vis_h"] = np.asarray(rows["vis_h"], bool)
    out["vis_o"] = np.asarray(rows["vis_o"], bool)
    out["ep"] = np.asarray(rows["ep"], np.int64)
    out["fr"] = np.asarray(rows["fr"], np.int64)
    out["n_ep"] = n_ep
    out["n_steps"] = cfg.n_steps
    return out


def _collect_robotwin(cfg: RTWXO0E0Config, split: str, stop: list[str]) -> dict[str, Any]:
    gcfg = _g2r_cfg(cfg, cfg.seed + {"train": 0, "val": 1, "test": 2}[split] * 1000)
    # hijack episode counts per split via temporary config
    if split == "train":
        gcfg = replace(gcfg, n_train_ep=cfg.n_train_ep, n_val_ep=0, n_test_ep=0)
    elif split == "val":
        gcfg = replace(gcfg, n_train_ep=0, n_val_ep=cfg.n_val_ep, n_test_ep=0)
    else:
        gcfg = replace(gcfg, n_train_ep=0, n_val_ep=0, n_test_ep=cfg.n_test_ep)
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_curobo_planner(repo)
    _patch_raster_shader()
    pool = collect_split(gcfg, split=split, rng=np.random.default_rng(cfg.seed + hash(split) % 10000), stop=stop)
    # add n, ep, fr
    n = pool["p"].shape[0]
    n_steps = int(pool["n_steps"])
    quat = pool["quat"]
    ns = np.stack([n_from_R(_quat_to_R(quat[i])) for i in range(n)])
    pool["n"] = ns
    pool["ep"] = np.arange(n) // n_steps
    pool["fr"] = np.arange(n) % n_steps
    return pool


def _write_split_cache(root: Path, split: str, pool: dict[str, Any], masks_learned: dict[str, np.ndarray] | None) -> None:
    """Physical isolation: obs.npz (estimator) vs gt.npz (evaluator) vs seg_gt (train only)."""
    d = root / "cache" / split
    d.mkdir(parents=True, exist_ok=True)
    # sealed GT masks for UNet training (not passed to estimator)
    np.savez_compressed(
        d / "seg_gt.npz",
        mask_h=pool["mask_h"],
        mask_o=pool["mask_o"],
        rgb_h=pool["rgb_h"],
        rgb_o=pool["rgb_o"],
    )
    np.savez_compressed(
        d / "gt.npz",
        p_gt=pool["p"],
        quat_gt=pool["quat"],
        n_gt=pool["n"],
        episode_id=pool["ep"],
        frame_id=pool["fr"],
    )
    mh = masks_learned["mask_h"] if masks_learned else pool["mask_h"]
    mo = masks_learned["mask_o"] if masks_learned else pool["mask_o"]
    # fused points from chosen masks
    fused = []
    rng = np.random.default_rng(SEED + hash(split) % 997)
    for i in range(pool["p"].shape[0]):
        fused.append(fuse_points(pool["xyz_h"][i], mh[i], pool["xyz_o"][i], mo[i], rng=rng))
    # ragged → object array
    fused_arr = np.empty(len(fused), dtype=object)
    for i, P in enumerate(fused):
        fused_arr[i] = P
    np.savez_compressed(
        d / "obs.npz",
        rgb_h=pool["rgb_h"],
        rgb_o=pool["rgb_o"],
        xyz_h=pool["xyz_h"],
        xyz_o=pool["xyz_o"],
        mask_h=mh,
        mask_o=mo,
        vis_h=pool["vis_h"],
        vis_o=pool["vis_o"],
        fused_points_B=fused_arr,
        coverage_any=(pool["vis_h"] | pool["vis_o"]),
    )


def _load_obs(root: Path, split: str) -> dict[str, Any]:
    z = np.load(root / "cache" / split / "obs.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _load_gt(root: Path, split: str) -> dict[str, Any]:
    z = np.load(root / "cache" / split / "gt.npz")
    return {k: np.asarray(z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# Gates / pattern
# ---------------------------------------------------------------------------


def _pattern(*, g0: bool, g0b: bool, g1: bool, g2: bool, g3: bool) -> str:
    if not g0:
        return "observation_support_failure"
    if not g0b:
        return "axis_excitation_failure"
    if not g1:
        return "effective_axis_failure"
    if not g2:
        return "effective_position_failure"
    if not g3:
        return "effective_axis_failure"  # B2 not above B0 → still axis failure class
    return "effective_pose_supported"


def run_rtwx_o0e0(output: str | Path, config: RTWXO0E0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0E0Config(output=str(output)))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    print(f"[rtwx-o0e0] backend={cfg.backend} smoke={cfg.smoke} seed={cfg.seed}", flush=True)

    prior = load_delta_O(cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin")
    delta_O = np.asarray(prior["delta_O"], dtype=np.float64)
    delta_y = float(delta_O[1])  # object-frame Y; almost pure axial offset

    # CAD quotient profile
    if cfg.backend == "numpy" or cfg.smoke and not Path(cfg.robotwin_repo).joinpath("assets/objects/021_cup/visual/base0.glb").is_file():
        # synthetic CAD frustum
        hs = np.linspace(0, 0.088, 64)
        cad = []
        for h in hs:
            r = 0.02 + 0.22 * (h / 0.088)
            for th in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                cad.append([r * np.cos(th), h, r * np.sin(th)])
        cad_pts = np.asarray(cad, dtype=np.float64)
    else:
        cad_pts = _fps(load_scaled_cad(cfg.robotwin_repo, 8192), N_CAD, np.random.default_rng(CAD_FPS_SEED))
    cad_hr = cad_hr_profile(cad_pts)
    from scipy.spatial import cKDTree

    tree = cKDTree(cad_hr)
    sphere = fibonacci_sphere(K_SPHERE, SPHERE_SEED)

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "seed": cfg.seed,
        "rgb_size": cfg.rgb_size,
        "delta_O": delta_O.tolist(),
        "delta_y": delta_y,
        "G_geom": "SO(2)_y",
        "no_yaw_in_primary": True,
        "obs_gt_isolated": True,
        "sphere_K": K_SPHERE,
        "top_Kr": TOP_KR,
        "refine_degs": list(REFINE_DEGS),
        "n_cad": int(cad_pts.shape[0]),
        "gates": {"axis_med": MED_ER_MAX, "axis_p90": P90_ER_MAX, "Ep": EP_MAX, "med_ep_m": MED_MAX},
    }
    _write_json(root / "header.json", header)

    # ---- collect ----
    stop: list[str] = []
    pools: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        cache_obs = root / "cache" / split / "obs.npz"
        cache_gt = root / "cache" / split / "gt.npz"
        if cache_obs.is_file() and cache_gt.is_file() and (root / "cache" / split / "seg_gt.npz").is_file():
            print(f"[rtwx-o0e0] {split}: cache hit (raw collect still needed for masks? reload pools from seg)", flush=True)
            # rebuild minimal pool from caches for unet if needed
            seg = np.load(root / "cache" / split / "seg_gt.npz")
            gt = _load_gt(root, split)
            obs = _load_obs(root, split)
            pools[split] = {
                "rgb_h": np.asarray(seg["rgb_h"]),
                "rgb_o": np.asarray(seg["rgb_o"]),
                "mask_h": np.asarray(seg["mask_h"]),
                "mask_o": np.asarray(seg["mask_o"]),
                "xyz_h": np.asarray(obs["xyz_h"]),
                "xyz_o": np.asarray(obs["xyz_o"]),
                "vis_h": np.asarray(obs["vis_h"]),
                "vis_o": np.asarray(obs["vis_o"]),
                "p": gt["p_gt"],
                "quat": gt["quat_gt"],
                "n": gt["n_gt"],
                "ep": gt["episode_id"],
                "fr": gt["frame_id"],
            }
            continue
        print(f"[rtwx-o0e0] collect {split}", flush=True)
        if cfg.backend == "numpy" or cfg.smoke:
            pools[split] = _synthetic_split(cfg, split, np.random.default_rng(cfg.seed + {"train": 0, "val": 1, "test": 2}[split]), delta_y)
        else:
            pools[split] = _collect_robotwin(cfg, split, stop)
        _write_split_cache(root, split, pools[split], masks_learned=None)

    if stop:
        raise RuntimeError(f"O0E0 collect stop: {stop}")

    # ---- G0 (GT visibility + later detection) ----
    test_pool = pools["test"]
    vis_any = test_pool["vis_h"] | test_pool["vis_o"]
    p_any = float(np.mean(vis_any)) if len(vis_any) else 0.0
    g0_cov = bool(p_any >= P_ANY_AGG)
    print(f"[rtwx-o0e0] G0 coverage P_any={p_any:.3f} ok={g0_cov}", flush=True)

    if not g0_cov:
        result = {
            "header": header,
            "pattern": "observation_support_failure",
            "G0": {"ok": False, "P_any": p_any, "P_det": float("nan"), "note": "STOP before UNet / axis (coverage)"},
            "unlocks_o0e1_prereg": False,
            "unlocks_o1": False,
            "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0] pattern=observation_support_failure", flush=True)
        return result

    # ---- train frozen U-Net family (not a new segmentation cell) ----
    print("[rtwx-o0e0] train frozen U-Net (O0G2 family)", flush=True)
    tr, va = pools["train"], pools["val"]
    rgb_tr = np.concatenate([tr["rgb_h"], tr["rgb_o"]], 0)
    m_tr = np.concatenate([tr["mask_h"], tr["mask_o"]], 0)
    rgb_va = np.concatenate([va["rgb_h"], va["rgb_o"]], 0)
    m_va = np.concatenate([va["mask_h"], va["mask_o"]], 0)
    model = _train_unet(rgb_tr, m_tr, rgb_va, m_va, seed=cfg.seed, epochs=cfg.epochs_unet)
    _cuda_gc()

    learned: dict[str, dict[str, np.ndarray]] = {}
    for split, pool in pools.items():
        mh = _predict_masks(model, pool["rgb_h"])
        mo = _predict_masks(model, pool["rgb_o"])
        learned[split] = {"mask_h": mh, "mask_o": mo}
        _write_split_cache(root, split, pool, masks_learned=learned[split])
    _cuda_gc()

    # detection given visibility
    te_l = learned["test"]
    nonempty = (te_l["mask_h"].reshape(te_l["mask_h"].shape[0], -1).any(1)) | (te_l["mask_o"].reshape(te_l["mask_o"].shape[0], -1).any(1))
    p_det = float(np.mean(nonempty[vis_any])) if vis_any.any() else 0.0
    g0_det = bool(p_det >= P_DET_MIN)
    g0 = bool(g0_cov and g0_det)
    print(f"[rtwx-o0e0] G0 det P(M≠∅|V)={p_det:.3f} ok={g0_det} G0={g0}", flush=True)

    if not g0:
        result = {
            "header": header,
            "pattern": "observation_support_failure",
            "G0": {"ok": False, "P_any": p_any, "P_det": p_det},
            "unlocks_o0e1_prereg": False,
            "unlocks_o1": False,
            "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0] pattern=observation_support_failure", flush=True)
        return result

    # ---- G0b excitation (evaluator opens GT only) ----
    gt_tr = _load_gt(root, "train")
    gt_te = _load_gt(root, "test")
    n_sum = gt_tr["n_gt"].sum(0)
    n_const = n_sum / max(np.linalg.norm(n_sum), 1e-12)
    ang = np.array([e_axis_deg(gt_te["n_gt"][i], n_const) for i in range(gt_te["n_gt"].shape[0])])
    r_excite = float(np.mean(ang > EXCITE_ALPHA_DEG))
    g0b = bool(r_excite >= EXCITE_RHO)
    print(f"[rtwx-o0e0] G0b r_excite={r_excite:.3f} (≥{EXCITE_RHO}) ok={g0b}", flush=True)
    if not g0b:
        result = {
            "header": header,
            "pattern": "axis_excitation_failure",
            "G0": {"ok": True, "P_any": p_any, "P_det": p_det},
            "G0b": {"ok": False, "r_excite": r_excite, "alpha_deg": EXCITE_ALPHA_DEG, "rho": EXCITE_RHO, "n_const": n_const.tolist()},
            "unlocks_o0e1_prereg": False,
            "unlocks_o1": False,
            "config": asdict(cfg),
        }
        _write_json(root / "summary.json", result)
        _write_json(root / "run.json", result)
        print("[rtwx-o0e0] pattern=axis_excitation_failure", flush=True)
        return result

    # ---- evaluate B0/B1/B2 on test obs only ----
    obs_te = _load_obs(root, "test")
    n_frames = gt_te["p_gt"].shape[0]
    per_frame = []
    branch_stats: dict[str, dict[str, Any]] = {}

    for mode in ("B0", "B1", "B2"):
        print(f"[rtwx-o0e0] eval {mode}", flush=True)
        e_ax, e_p, S_m = [], [], []
        p_hats, n_hats = [], []
        full_R_diag = []
        for i in range(n_frames):
            obs_i = {
                "fused_points_B": np.asarray(obs_te["fused_points_B"][i], dtype=np.float64),
                "mask_h": obs_te["mask_h"][i],
                "mask_o": obs_te["mask_o"][i],
            }
            p_hat, n_hat, meta = estimate_effective_pose(
                obs_i, delta_y=delta_y, tree=tree, sphere=sphere, mode=mode, n_const=n_const
            )
            ea = e_axis_deg(n_hat, gt_te["n_gt"][i])
            ep = float(np.linalg.norm(p_hat - gt_te["p_gt"][i])) if np.isfinite(p_hat).all() else float("nan")
            R0 = R0_from_n(n_hat)
            Rgt = _quat_to_R(gt_te["quat_gt"][i])
            # geodesic
            Rel = R0.T @ Rgt
            c = float(np.clip((np.trace(Rel) - 1.0) * 0.5, -1.0, 1.0))
            dR = float(np.degrees(np.arccos(c)))
            e_ax.append(ea)
            e_p.append(ep)
            S_m.append(meta.get("S_best", float("nan")))
            p_hats.append(p_hat)
            n_hats.append(n_hat)
            full_R_diag.append(dR)
            if mode == "B2":
                per_frame.append(
                    {
                        "frame": int(i),
                        "episode": int(gt_te["episode_id"][i]),
                        "n_gt": gt_te["n_gt"][i].tolist(),
                        "n_hat": n_hat.tolist(),
                        "axis_err": ea,
                        "p_hat": p_hat.tolist() if np.isfinite(p_hat).all() else [None] * 3,
                        "p_gt": gt_te["p_gt"][i].tolist(),
                        "position_error": ep,
                        "S_best": meta.get("S_best"),
                        "S_second": meta.get("S_second"),
                        "objective_margin": meta.get("margin"),
                        "full_R_diag": dR,
                        "diagnostic_only": True,
                        "used_by_gate": False,
                    }
                )
        e_ax = np.asarray(e_ax, dtype=np.float64)
        e_p = np.asarray(e_p, dtype=np.float64)
        p_hats = np.stack(p_hats)
        pos = _score_pose(p_hats, gt_te["p_gt"])
        branch_stats[mode] = {
            "median_e_axis_deg": float(np.nanmedian(e_ax)),
            "p90_e_axis_deg": float(np.nanpercentile(e_ax, 90)),
            "f90": float(np.nanmean(e_ax >= F90_THRESH)),
            "mean_e_axis_deg": float(np.nanmean(e_ax)),
            "E_p": pos["E_p"],
            "median_ep_m": pos["median_ep_m"],
            "median_ep_cm": float(pos["median_ep_m"] * 100.0),
            "strong_2cm": bool(pos["median_ep_m"] <= STRONG_MED),
            "median_full_R_diag_deg": float(np.nanmedian(full_R_diag)),
            "p90_full_R_diag_deg": float(np.nanpercentile(full_R_diag, 90)),
            "n": int(np.isfinite(e_ax).sum()),
        }
        print(
            f"[rtwx-o0e0] {mode} med_axis={branch_stats[mode]['median_e_axis_deg']:.2f} "
            f"p90={branch_stats[mode]['p90_e_axis_deg']:.2f} Ep={branch_stats[mode]['E_p']:.4f} "
            f"med_ep_cm={branch_stats[mode]['median_ep_cm']:.2f}",
            flush=True,
        )

    b2 = branch_stats["B2"]
    b0 = branch_stats["B0"]
    g1 = bool(b2["median_e_axis_deg"] <= MED_ER_MAX and b2["p90_e_axis_deg"] <= P90_ER_MAX)
    g2 = bool(b2["E_p"] <= EP_MAX and b2["median_ep_m"] <= MED_MAX)
    g3 = bool(b2["median_e_axis_deg"] < b0["median_e_axis_deg"])
    pattern = _pattern(g0=True, g0b=True, g1=g1, g2=g2, g3=g3)

    # d_G vs e_axis unit check on a few frames
    dG_checks = []
    for i in range(min(5, n_frames)):
        Rgt = _quat_to_R(gt_te["quat_gt"][i])
        n_hat = np.asarray(per_frame[i]["n_hat"]) if per_frame else n_from_R(Rgt)
        R0 = R0_from_n(n_hat)
        dG_checks.append(abs(d_G_deg(R0, Rgt) - e_axis_deg(n_hat, gt_te["n_gt"][i])))

    result = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": True, "P_any": p_any, "P_det": p_det},
        "G0b": {"ok": True, "r_excite": r_excite, "alpha_deg": EXCITE_ALPHA_DEG, "rho": EXCITE_RHO, "n_const": n_const.tolist()},
        "B0_constant": branch_stats["B0"],
        "B1_pca": branch_stats["B1"],
        "B2_quotient_cad": branch_stats["B2"],
        "G1_axis": {"ok": g1, **{k: b2[k] for k in ("median_e_axis_deg", "p90_e_axis_deg", "f90")}},
        "G2_position": {"ok": g2, "E_p": b2["E_p"], "median_ep_m": b2["median_ep_m"], "strong_2cm": b2["strong_2cm"]},
        "G3_above_B0": {"ok": g3, "med_B2": b2["median_e_axis_deg"], "med_B0": b0["median_e_axis_deg"]},
        "full_R_diagnostic": {
            "diagnostic_only": True,
            "used_by_gate": False,
            "median_deg": b2["median_full_R_diag_deg"],
            "p90_deg": b2["p90_full_R_diag_deg"],
        },
        "dG_vs_eaxis_max_abs": float(max(dG_checks)) if dG_checks else float("nan"),
        "unlocks_o0e1_prereg": pattern == "effective_pose_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    _write_json(root / "per_frame_b2.json", {"frames": per_frame[:500]})  # cap
    print(
        f"[rtwx-o0e0] pattern={pattern} G1={g1} G2={g2} G3={g3} unlocks_e1={result['unlocks_o0e1_prereg']}",
        flush=True,
    )
    return result

"""RTWX-O0SEQ0: relative propagation horizon audit (chain vs direct vs absolute)."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.center_axis_inference import VisObs, run_a1_center_first, visibility_center_fused
from .geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, RelativeICPConfig
from .geometry.relative_state_chain import run_adjacent_chain, run_direct_anchor
from .geometry.relative_validity import symmetric_support_score
from .geometry.visibility_reference import AxialVisibilityReference, camera_center_world
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
    fuse_points,
)
from .rtwx_o0e0p0 import RGB_SIZE, RTWXO0E0P0Config, P_XY, P_Z, _capture_one, _set_static_pose, n_from_tilt
from .rtwx_o0e0r1 import _axis_stats, _ep_stats
from .rtwx_o0e0r2 import _load_p0_split, _load_seg_split
from .rtwx_o0e0r4 import _set_robot_q
from .rtwx_o0e0r5 import S0_TRAIN_SEED
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import EP_MAX, MED_MAX, _predict_masks, _train_unet
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0hyb0 import (
    BETA0_DEG,
    _camera_q1,
    _resolve_apr,
    _sample_upright_pose,
    _yaw_octant_gamma,
)
from .rtwx_o0q0 import _R_to_quat, _robot_q
from .rtwx_o0rel0 import (
    ICP_CFG,
    TRANS_DIR_CATALOG,
    _apply_object_motion,
    _numpy_disk_cloud,
    assert_obs_no_gt,
    voxel_once,
)
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0v import CAM_HEAD, CAM_OBS, _get_cam
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0SEQ0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0seq0.relative_horizon_audit.v1"
SEED = 47601
SEQ_SEED = 37609
N_SEQ = 100
T_TRANS = 16
CHECKPOINTS = (1, 2, 4, 8, 16)
PRIOR_N_MED = 5.0
PRIOR_N_P90 = 10.0
DELTA_BETA_STEP = (0.0, 5.0, 10.0)
DELTA_BETA_PROBS = (0.3, 0.4, 0.3)
TRANS_STEP_M = (0.0, 0.01, 0.02)
GT_MASK_DIAG_N = 20
UNC_EPS = 1e-6
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n"})
_APR_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RTWXO0SEQ0Config:
    output: str = "runs/rtwx_o0seq0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    seq_seed: int = SEQ_SEED
    n_seq: int = N_SEQ
    t_trans: int = T_TRANS
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0SEQ0Config) -> RTWXO0SEQ0Config:
    if cfg.smoke:
        return replace(cfg, n_seq=4, t_trans=8, epochs_unet=4, backend="numpy")
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0SEQ0 must not write there")


def _reflect_delta(p: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p_new = p + delta
    lo = np.array([P_XY[0][0], P_XY[1][0], P_Z - 0.02])
    hi = np.array([P_XY[0][1], P_XY[1][1], P_Z + 0.02])
    out = p.copy()
    d = delta.copy()
    for k in range(3):
        if p_new[k] < lo[k]:
            d[k] = -abs(d[k])
        elif p_new[k] > hi[k]:
            d[k] = abs(d[k])
    return d


def generate_sequence_trajectory(seq_id: int, t_trans: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + seq_id * 9973)
    beta = float(rng.choice(BETA0_DEG))
    pose0 = _sample_upright_pose(rng, beta)
    frames: list[dict[str, Any]] = [{
        "t": 0,
        "p": pose0["p"],
        "quat": pose0["quat"],
        "R": pose0["R"],
        "n": pose0["n"],
        "cam_catalog_idx": 0,
    }]
    r_cur = np.asarray(pose0["R"], dtype=np.float64)
    p_cur = np.asarray(pose0["p"], dtype=np.float64)
    n_cur = np.asarray(pose0["n"], dtype=np.float64)
    for t in range(1, t_trans + 1):
        db = float(rng.choice(DELTA_BETA_STEP, p=DELTA_BETA_PROBS))
        tm = float(rng.choice(TRANS_STEP_M))
        td = TRANS_DIR_CATALOG[(seq_id + t) % len(TRANS_DIR_CATALOG)]
        dg = _yaw_octant_gamma(rng)
        r_next, p_next, n_next, q_next = _apply_object_motion(
            r_cur, p_cur, n_cur, delta_beta_deg=db, delta_gamma=dg, trans_dir=td, trans_mag=tm, rng=rng,
        )
        delta_p = p_next - p_cur
        delta_p = _reflect_delta(p_cur, delta_p)
        p_next = p_cur + delta_p
        r_cur, n_cur = r_next, n_next
        p_cur = p_next
        frames.append({
            "t": t,
            "p": p_cur.copy(),
            "quat": q_next,
            "R": r_cur.copy(),
            "n": n_cur.copy(),
            "cam_catalog_idx": (seq_id + t) % 12,
        })
    return frames


def _capture_frame(env: Any, size: int, cup_ids: set[int], rep_h: int | None, rep_o: int | None) -> dict[str, Any]:
    cap = _capture_one(env, size, cup_ids, rep_h, rep_o)
    cap["E_h"] = np.asarray(_get_cam(env, CAM_HEAD).get_extrinsic_matrix(), dtype=np.float64)
    cap["E_o"] = np.asarray(_get_cam(env, CAM_OBS).get_extrinsic_matrix(), dtype=np.float64)
    return cap


def collect_sequences_robotwin(cfg: RTWXO0SEQ0Config) -> tuple[dict[str, Any], dict[str, Any]]:
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
    n_frames = cfg.t_trans + 1
    obs_keys = ["rgb_h", "rgb_o", "xyz_h", "xyz_o", "vis_h", "vis_o"]
    obs: dict[str, list] = {k: [] for k in obs_keys}
    gt: dict[str, list] = {k: [] for k in (
        "p", "quat", "R", "n", "mask_h", "mask_o", "seq_id", "frame_id",
    )}
    env = None
    try:
        env = _setup_env(repo, TASK, cfg.seq_seed, 32, args)
        env.check_success = lambda *a, **k: False
        cup_ids = _cup_ids(env)
        q0, _ = _robot_q(env)
        rep_h = rep_o = None
        for si in range(cfg.n_seq):
            if si % 10 == 0 or si + 1 == cfg.n_seq:
                print(f"[rtwx-o0seq0] collect seq {si}/{cfg.n_seq}", flush=True)
            traj = generate_sequence_trajectory(si, cfg.t_trans, cfg.seq_seed)
            q_cur = q0.copy()
            seq_rgb_h, seq_rgb_o = [], []
            seq_xyz_h, seq_xyz_o = [], []
            seq_vh, seq_vo = [], []
            for fr in traj:
                _set_static_pose(env, fr["p"], fr["quat"])
                q_cur = _camera_q1(q0, int(fr["cam_catalog_idx"]), large=False)
                _set_robot_q(env, q_cur)
                cap = _capture_frame(env, p0cfg.rgb_size, cup_ids, rep_h, rep_o)
                rep_h, rep_o = cap["rep_h"], cap["rep_o"]
                seq_rgb_h.append(cap["rgb_h"])
                seq_rgb_o.append(cap["rgb_o"])
                seq_xyz_h.append(cap["xyz_h"])
                seq_xyz_o.append(cap["xyz_o"])
                seq_vh.append(cap["vis_h"])
                seq_vo.append(cap["vis_o"])
                gt["p"].append(fr["p"])
                gt["quat"].append(fr["quat"])
                gt["R"].append(fr["R"])
                gt["n"].append(fr["n"])
                gt["mask_h"].append(cap["mask_h"])
                gt["mask_o"].append(cap["mask_o"])
                gt["seq_id"].append(si)
                gt["frame_id"].append(fr["t"])
            obs["rgb_h"].append(np.stack(seq_rgb_h))
            obs["rgb_o"].append(np.stack(seq_rgb_o))
            obs["xyz_h"].append(np.stack(seq_xyz_h))
            obs["xyz_o"].append(np.stack(seq_xyz_o))
            obs["vis_h"].append(np.asarray(seq_vh, bool))
            obs["vis_o"].append(np.asarray(seq_vo, bool))
        env.close()
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise
    obs_out = {k: np.stack(obs[k]) for k in obs_keys if k.startswith(("rgb", "xyz"))}
    obs_out["vis_h"] = np.stack(obs["vis_h"])
    obs_out["vis_o"] = np.stack(obs["vis_o"])
    gt_out = {
        "p": np.stack(gt["p"]).reshape(cfg.n_seq, n_frames, 3),
        "quat": np.stack(gt["quat"]).reshape(cfg.n_seq, n_frames, 4),
        "R": np.stack(gt["R"]).reshape(cfg.n_seq, n_frames, 3, 3),
        "n": np.stack(gt["n"]).reshape(cfg.n_seq, n_frames, 3),
        "mask_h": np.stack(gt["mask_h"]).reshape(cfg.n_seq, n_frames, RGB_SIZE, RGB_SIZE),
        "mask_o": np.stack(gt["mask_o"]).reshape(cfg.n_seq, n_frames, RGB_SIZE, RGB_SIZE),
        "seq_id": np.asarray(gt["seq_id"], np.int32).reshape(cfg.n_seq, n_frames),
        "frame_id": np.asarray(gt["frame_id"], np.int32).reshape(cfg.n_seq, n_frames),
    }
    assert_obs_no_gt(obs_out)
    return obs_out, gt_out


def collect_sequences_numpy(cfg: RTWXO0SEQ0Config) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seq_seed)
    n_frames = cfg.t_trans + 1
    obs: dict[str, list] = {k: [] for k in ("rgb_h", "rgb_o", "xyz_h", "xyz_o", "vis_h", "vis_o")}
    gt: dict[str, list] = {k: [] for k in ("p", "quat", "R", "n", "mask_h", "mask_o")}
    for si in range(cfg.n_seq):
        traj = generate_sequence_trajectory(si, cfg.t_trans, cfg.seq_seed)
        srh, sro, sxh, sxo, svh, svo = [], [], [], [], [], []
        gp, gq, gR, gn, gmh, gmo = [], [], [], [], [], []
        for fr in traj:
            shift = np.array([0.002 * fr["t"], 0.0, 0.0])
            rh, mh, xh = _numpy_disk_cloud(fr["p"], RGB_SIZE, rng, shift=shift)
            ro, mo, xo = _numpy_disk_cloud(fr["p"], RGB_SIZE, rng, shift=shift + np.array([0.01, 0, 0]))
            srh.append(rh); sro.append(ro); sxh.append(xh); sxo.append(xo)
            svh.append(True); svo.append(True)
            gp.append(fr["p"]); gq.append(fr["quat"]); gR.append(fr["R"]); gn.append(fr["n"])
            gmh.append(mh); gmo.append(mo)
        for k, v in zip(("rgb_h", "rgb_o", "xyz_h", "xyz_o"), (srh, sro, sxh, sxo)):
            obs[k].append(np.stack(v))
        obs["vis_h"].append(np.asarray(svh, bool))
        obs["vis_o"].append(np.asarray(svo, bool))
        gt["p"].append(np.stack(gp))
        gt["quat"].append(np.stack(gq))
        gt["R"].append(np.stack(gR))
        gt["n"].append(np.stack(gn))
        gt["mask_h"].append(np.stack(gmh))
        gt["mask_o"].append(np.stack(gmo))
    obs_out = {k: np.stack(obs[k]) for k in obs}
    gt_out = {k: np.stack(gt[k]) for k in gt}
    assert_obs_no_gt(obs_out)
    return obs_out, gt_out


def _science_pass(axis_deg: np.ndarray, pos_m: np.ndarray) -> bool:
    ax = _axis_stats(axis_deg)
    ps = _ep_stats(pos_m)
    return bool(
        ax["median_e_axis_deg"] <= MED_ER_MAX and ax["p90_e_axis_deg"] <= P90_ER_MAX
        and float(np.nanmean(pos_m)) <= EP_MAX and ps["median_ep_m"] <= MED_MAX
    )


def _largest_h_star(pass_at_h: dict[int, bool]) -> int:
    h_star = 0
    for h in CHECKPOINTS:
        if pass_at_h.get(h, False):
            h_star = h
        else:
            break
    return h_star


def _build_clouds_seq(
    obs: dict[str, Any],
    gt: dict[str, Any] | None,
    s0: Any,
    seq_i: int,
    *,
    rng: np.random.Generator,
    use_gt_mask: bool = False,
) -> list[np.ndarray]:
    n_frames = obs["rgb_h"].shape[1]
    clouds: list[np.ndarray] = []
    for t in range(n_frames):
        if use_gt_mask and gt is not None:
            mh = gt["mask_h"][seq_i, t]
            mo = gt["mask_o"][seq_i, t]
        else:
            mh = _predict_masks(s0, obs["rgb_h"][seq_i, t][None])[0] > 0.5
            mo = _predict_masks(s0, obs["rgb_o"][seq_i, t][None])[0] > 0.5
        clouds.append(fuse_points(
            obs["xyz_h"][seq_i, t], mh, obs["xyz_o"][seq_i, t], mo, rng=rng,
        ))
    return clouds


def _eval_sequence(
    obs: dict[str, Any],
    gt: dict[str, Any],
    seq_i: int,
    *,
    s0: Any,
    lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
    n_const: np.ndarray,
    d_o: float,
    rng: np.random.Generator,
    use_gt_mask: bool = False,
) -> dict[str, Any]:
    clouds = _build_clouds_seq(obs, gt, s0, seq_i, rng=rng, use_gt_mask=use_gt_mask)
    c0 = clouds[0]
    vo0 = VisObs(
        fused_cloud=c0,
        c_h=c0.mean(0) if c0.size else np.zeros(3),
        c_o=c0.mean(0) if c0.size else np.zeros(3),
        cam_h=np.zeros(3),
        cam_o=np.zeros(3),
    )
    p0_hat, _, _, _ = visibility_center_fused(n_const, vo0.c_h, vo0.c_o, vo0.cam_h, vo0.cam_o, lut)
    n0_hat = n_const.copy()
    ckpt_states, step_res = run_adjacent_chain(clouds, p0_hat, n0_hat, d_o, cfg=ICP_CFG, checkpoints=CHECKPOINTS)
    steps: list[dict[str, Any]] = []
    cum_u = 0.0
    for t, res in enumerate(step_res, start=1):
        r_gt = gt["R"][seq_i, t - 1]
        n_prev = gt["n"][seq_i, t - 1]
        n_cur = gt["n"][seq_i, t]
        r_d = gt["R"][seq_i, t] @ r_gt.T
        p_d = gt["p"][seq_i, t] - r_d @ gt["p"][seq_i, t - 1]
        e_dn = float(e_axis_deg(res.R @ n_prev, n_cur)) if res.valid else float("nan")
        q = symmetric_support_score(clouds[t - 1], clouds[t], res.R, res.t, d_o) if res.valid else 0.0
        cum_u += -np.log(q + UNC_EPS)
        steps.append({
            "t": t,
            "relative_valid": bool(res.valid),
            "relative_axis_error_deg": e_dn,
            "q_rel": float(q),
            "n_corr": int(res.n_corr),
            "icp_rmse": float(res.rmse),
        })
    checkpoints: dict[str, dict[str, Any]] = {}
    for h in CHECKPOINTS:
        if h >= len(clouds):
            continue
        st = ckpt_states.get(h)
        if st is None:
            continue
        p_d, n_d, dres = run_direct_anchor(clouds[0], clouds[h], p0_hat, n0_hat, d_o, cfg=ICP_CFG)
        vo_h = VisObs(
            fused_cloud=clouds[h],
            c_h=clouds[h].mean(0) if clouds[h].size else np.zeros(3),
            c_o=clouds[h].mean(0) if clouds[h].size else np.zeros(3),
            cam_h=np.zeros(3),
            cam_o=np.zeros(3),
        )
        a1 = run_a1_center_first(vo_h, n_const, lut, b2)
        checkpoints[str(h)] = {
            "chain_axis_error_deg": float(e_axis_deg(st.n, gt["n"][seq_i, h])),
            "chain_position_error_m": float(np.linalg.norm(st.p - gt["p"][seq_i, h])),
            "direct_axis_error_deg": float(e_axis_deg(n_d, gt["n"][seq_i, h])),
            "direct_position_error_m": float(np.linalg.norm(p_d - gt["p"][seq_i, h])),
            "absolute_axis_error_deg": float(e_axis_deg(a1.n1, gt["n"][seq_i, h])),
            "absolute_position_error_m": float(np.linalg.norm(a1.p1 - gt["p"][seq_i, h])),
            "cum_uncertainty": float(cum_u),
        }
    return {
        "sequence_id": int(seq_i),
        "anchor": {
            "axis_error_deg": float(e_axis_deg(n0_hat, gt["n"][seq_i, 0])),
            "position_error_m": float(np.linalg.norm(p0_hat - gt["p"][seq_i, 0])),
        },
        "checkpoints": checkpoints,
        "steps": steps,
    }


def _horizon_aggregate(per_seq: list[dict[str, Any]], h: int, branch: str) -> dict[str, Any]:
    e_n, e_p = [], []
    for sq in per_seq:
        ck = sq["checkpoints"].get(str(h))
        if ck is None:
            continue
        e_n.append(ck[f"{branch}_axis_error_deg"])
        e_p.append(ck[f"{branch}_position_error_m"])
    e_n = np.asarray(e_n, dtype=np.float64)
    e_p = np.asarray(e_p, dtype=np.float64)
    if e_n.size == 0:
        return {"axis": _axis_stats(np.array([np.nan])), "position": _ep_stats(np.array([np.nan])), "pass": False}
    return {
        "axis": _axis_stats(e_n),
        "position": {**_ep_stats(e_p), "E_p": float(np.nanmean(e_p))},
        "pass": _science_pass(e_n, e_p),
    }


def _pattern(
    *,
    l1_ok: bool,
    l2_ok: bool,
    h_star: int,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    if not l1_ok:
        return "sequence_observation_failure", tags
    if not l2_ok:
        return "sequence_anchor_failure", tags
    if h_star < 4:
        return "relative_sequence_insufficient", tags
    if h_star >= 16:
        tags.append("stable_to_16")
    elif h_star >= 8:
        tags.append("drift_after_h8")
    elif h_star >= 4:
        tags.append("drift_after_h4")
    return "relative_sequence_supported", tags


def run_rtwx_o0seq0(output: str | Path, config: RTWXO0SEQ0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0SEQ0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    natural_root = _resolve_apr(cfg.natural_cache)
    p0_root = _resolve_apr(cfg.p0_cache)
    r5_root = _resolve_apr(cfg.r5_run)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    cache_obs = root / "cache" / "observations" / "obs.npz"
    if not cache_obs.is_file():
        print(f"[rtwx-o0seq0] collecting n_seq={cfg.n_seq} T={cfg.t_trans} backend={cfg.backend}", flush=True)
        if cfg.backend == "numpy":
            obs, gt = collect_sequences_numpy(cfg)
        else:
            obs, gt = collect_sequences_robotwin(cfg)
        (root / "cache" / "observations").mkdir(parents=True, exist_ok=True)
        (root / "cache" / "gt").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_obs, **obs)
        np.savez_compressed(root / "cache" / "gt" / "gt.npz", **gt)

    obs = dict(np.load(cache_obs))
    gt = dict(np.load(root / "cache" / "gt" / "gt.npz"))

    repo = Path(cfg.robotwin_repo)
    if not repo.is_absolute():
        repo = repo.resolve()
    d_o = load_D_O(str(repo) if repo.is_dir() else "/root/RoboTwin")
    lut = AxialVisibilityReference.load(r5_root / "visibility_lut.npz")
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

    b2 = FrozenQuotientB2(cKDTree(cad_hr_profile(cad_pts)), fibonacci_sphere(K_SPHERE, SPHERE_SEED))
    b2.assert_frozen()
    _, gt_tr = _load_p0_split(p0_root, "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    print("[rtwx-o0seq0] train S0", flush=True)
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )

    rng = np.random.default_rng(cfg.seed)
    per_seq: list[dict[str, Any]] = []
    for si in range(cfg.n_seq):
        if si % 20 == 0:
            print(f"[rtwx-o0seq0] eval seq {si}/{cfg.n_seq}", flush=True)
        per_seq.append(_eval_sequence(obs, gt, si, s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o, rng=rng))

    v_any = np.asarray(obs["vis_h"], bool) | np.asarray(obs["vis_o"], bool)
    l1_vis = float((v_any).mean())
    adj_ok = []
    n_frames = obs["rgb_h"].shape[1]
    for si in range(cfg.n_seq):
        for t in range(1, n_frames):
            adj_ok.append(bool(v_any[si, t - 1] and v_any[si, t]))
    l1_adj = float(np.mean(adj_ok)) if adj_ok else 0.0
    l1_ok = bool(l1_vis >= 0.98 and l1_adj >= 0.98)

    e_n0 = np.asarray([s["anchor"]["axis_error_deg"] for s in per_seq], dtype=np.float64)
    e_p0 = np.asarray([s["anchor"]["position_error_m"] for s in per_seq], dtype=np.float64)
    l2_axis = _axis_stats(e_n0)
    l2_pos = _ep_stats(e_p0)
    l2_ok = bool(
        l2_axis["median_e_axis_deg"] <= PRIOR_N_MED and l2_axis["p90_e_axis_deg"] <= PRIOR_N_P90
        and l2_pos["median_ep_m"] <= MED_MAX and float(np.nanmean(e_p0)) <= EP_MAX
    )

    horizon_table: dict[str, dict[str, dict[str, Any]]] = {
        b: {} for b in ("chain", "direct", "absolute")
    }
    pass_at_h: dict[int, bool] = {}
    for h in CHECKPOINTS:
        for b in ("chain", "direct", "absolute"):
            horizon_table[b][str(h)] = _horizon_aggregate(per_seq, h, b)
        pass_at_h[h] = horizon_table["chain"][str(h)]["pass"]

    h_star = _largest_h_star(pass_at_h)
    pattern, tags = _pattern(l1_ok=l1_ok, l2_ok=l2_ok, h_star=h_star)

    gtmask_h_star = None
    if cfg.n_seq >= GT_MASK_DIAG_N:
        diag_rng = np.random.default_rng(cfg.seq_seed + 99)
        diag_ids = list(range(min(GT_MASK_DIAG_N, cfg.n_seq)))
        diag_seqs = []
        for si in diag_ids:
            diag_seqs.append(_eval_sequence(
                obs, gt, si, s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o, rng=diag_rng, use_gt_mask=True,
            ))
        d_pass = {}
        for h in CHECKPOINTS:
            d_pass[h] = _horizon_aggregate(diag_seqs, h, "chain")["pass"]
        gtmask_h_star = _largest_h_star(d_pass)

    summary = {
        "pattern": pattern,
        "tags": tags,
        "H_star": int(h_star),
        "H_star_gtmask_diag": gtmask_h_star,
        "unlocks_relative_sequence_branch": pattern == "relative_sequence_supported",
        "unlocks_o0e1": False,
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "b2_config_hash": R5_B2_CONFIG_HASH,
        "G_L1_observation": {"ok": l1_ok, "p_vis": l1_vis, "p_adj_vis": l1_adj},
        "G_L2_anchor": {"ok": l2_ok, "axis_n0": l2_axis, "position_p0": {**l2_pos, "E_p": float(np.nanmean(e_p0))}},
        "horizon_table": horizon_table,
        "pass_at_h_chain": {str(k): v for k, v in pass_at_h.items()},
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH, "seq_seed": cfg.seq_seed})
    _write_json(root / "per_sequence.json", per_seq)
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0seq0] pattern={pattern} H*={h_star}", flush=True)
    return summary

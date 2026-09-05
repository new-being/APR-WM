"""RTWX-O0E0R6: finite-step alternating center–axis inference (inference-only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.center_axis_inference import (
    VisObs,
    aggregate_candidate_diagnostics,
    candidate_diagnostics,
    result_to_dict,
    run_a0_joint,
    run_a1_center_first,
    run_a2_consensus,
)
from .geometry.frozen_quotient_b2 import R5_B2_CONFIG_HASH, FrozenQuotientB2
from .geometry.visibility_reference import AxialVisibilityReference, cloud_centroid
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
from .rtwx_o0e0r1 import _axis_diag_pass, _axis_stats, _ep_stats
from .rtwx_o0e0r2 import _load_p0_split
from .rtwx_o0e0r5 import FRESH_TEST_SEED as R5_SEED, S0_TRAIN_SEED, _per_view_clouds
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import MED_MAX, EP_MAX, _predict_masks, _score_pose, _train_unet
from .rtwx_o0e0r2 import _load_seg_split
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0E0R6_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0e0r6.center_axis_inference.v1"
SEED = 44601
A0_SANITY_MEDIAN_TOL_DEG = 0.5
OBS_FORBIDDEN = frozenset({"p_gt", "R_gt", "n_gt", "p", "quat", "R", "n"})


@dataclass(frozen=True)
class RTWXO0E0R6Config:
    output: str = "runs/rtwx_o0e0r6"
    r5_run: str = "runs/rtwx_o0e0r5"
    p0_cache: str = "runs/rtwx_o0e0p0"
    natural_cache: str = "runs/rtwx_o0e0"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0E0R6 must not write there")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_r5_test(r5_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    obs_path = r5_root / "cache" / "test" / "obs.npz"
    gt_path = r5_root / "cache" / "test" / "gt.npz"
    if not obs_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"R5 cache missing under {r5_root}")
    obs = np.load(obs_path)
    gt = np.load(gt_path)
    return {
        "rgb_h": obs["rgb_h"], "rgb_o": obs["rgb_o"],
        "xyz_h": obs["xyz_h"], "xyz_o": obs["xyz_o"],
        "cam_h": obs["cam_h"], "cam_o": obs["cam_o"],
        "vis_h": obs["vis_h"], "vis_o": obs["vis_o"],
    }, {
        "p_gt": np.asarray(gt["p_gt"]),
        "n_gt": np.asarray(gt["n_gt"]),
        "beta_deg": np.asarray(gt["beta_deg"]) if "beta_deg" in gt else np.zeros(gt["p_gt"].shape[0]),
    }


def _build_frames(
    obs: dict[str, Any],
    s0: Any,
    *,
    seed: int,
) -> list[VisObs]:
    masks = {
        "mask_h": _predict_masks(s0, obs["rgb_h"]),
        "mask_o": _predict_masks(s0, obs["rgb_o"]),
    }
    rng = np.random.default_rng(seed)
    frames: list[VisObs] = []
    n = obs["rgb_h"].shape[0]
    for i in range(n):
        Ph = _per_view_clouds(obs["xyz_h"][i], masks["mask_h"][i], rng=rng)
        Po = _per_view_clouds(obs["xyz_o"][i], masks["mask_o"][i], rng=rng)
        Pf = fuse_points(
            obs["xyz_h"][i], masks["mask_h"][i],
            obs["xyz_o"][i], masks["mask_o"][i],
            rng=rng,
        )
        frames.append(VisObs(
            fused_cloud=Pf,
            c_h=cloud_centroid(Ph),
            c_o=cloud_centroid(Po),
            cam_h=np.asarray(obs["cam_h"][i], dtype=np.float64),
            cam_o=np.asarray(obs["cam_o"][i], dtype=np.float64),
        ))
    return frames


def _eval_branch(
    p_hats: list[np.ndarray],
    n_hats: list[np.ndarray],
    gt: dict[str, Any],
) -> dict[str, Any]:
    e_ax = np.asarray([e_axis_deg(n_hats[i], gt["n_gt"][i]) for i in range(len(n_hats))], dtype=np.float64)
    pos = _score_pose(np.stack(p_hats), gt["p_gt"])
    ax = _axis_stats(e_ax)
    g1 = _axis_diag_pass(ax["median_e_axis_deg"], ax["p90_e_axis_deg"])
    g2 = bool(pos["E_p"] <= EP_MAX and pos["median_ep_m"] <= MED_MAX)
    return {
        "axis": ax,
        "G1": g1,
        "G2": g2,
        "E_p": pos["E_p"],
        "median_ep_m": pos["median_ep_m"],
        "median_ep_cm": float(pos["median_ep_m"] * 100.0),
        "formal_pass": bool(g1 and g2),
    }


def _pattern(*, a1_pass: bool, a2_pass: bool) -> tuple[str, list[str]]:
    tags: list[str] = []
    if a1_pass or a2_pass:
        return "joint_inference_supported", tags
    tags.extend(["a1_failed", "a2_failed", "persistent_surface_next"])
    return "joint_inference_insufficient", tags


def run_rtwx_o0e0r6(output: str | Path, config: RTWXO0E0R6Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXO0E0R6Config(output=str(output))
    root = Path(output).resolve()
    r5_root = Path(cfg.r5_run).resolve()
    p0_root = Path(cfg.p0_cache).resolve()
    natural_root = Path(cfg.natural_cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    lut_src = r5_root / "visibility_lut.npz"
    if not lut_src.is_file():
        raise FileNotFoundError(f"R5 visibility_lut missing: {lut_src}")
    lut_dst = root / "visibility_lut.npz"
    if not lut_dst.is_file():
        import shutil
        shutil.copy2(lut_src, lut_dst)
    lut = AxialVisibilityReference.load(lut_dst)

    obs_path = r5_root / "cache" / "test" / "obs.npz"
    cache_hashes = {
        "r5_obs_cache_sha256": _file_sha256(obs_path),
        "visibility_lut_sha256": _file_sha256(lut_dst),
        "b2_config_hash": R5_B2_CONFIG_HASH,
        "r5_fresh_test_seed": R5_SEED,
    }
    print(f"[rtwx-o0e0r6] R5 cache seed={R5_SEED} b2_hash={R5_B2_CONFIG_HASH[:12]}", flush=True)

    repo = Path(cfg.robotwin_repo)
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
    sphere = fibonacci_sphere(K_SPHERE, SPHERE_SEED)
    b2 = FrozenQuotientB2(tree, sphere)
    b2.assert_frozen()

    header = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "inference_only": True,
        "r5_run": str(r5_root),
        **cache_hashes,
    }
    _write_json(root / "header.json", header)

    obs_te, gt_te = _load_r5_test(r5_root)
    _, gt_tr = _load_p0_split(p0_root, "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)

    print("[rtwx-o0e0r6] build S0 clouds from R5 obs", flush=True)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    epochs = 4 if cfg.smoke else cfg.epochs_unet
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=epochs,
    )
    frames = _build_frames(obs_te, s0, seed=cfg.seed)
    n_inst = gt_te["p_gt"].shape[0]

    per_frame: list[dict[str, Any]] = []
    a0_p, a0_n, a1_p, a1_n, a2_p, a2_n, a1_n1_list, a1_p0_list = [], [], [], [], [], [], [], []
    cand_diags: list[dict[str, Any]] = []

    from .geometry.center_axis_inference import d_ho_at_axis

    for i in range(n_inst):
        fo = frames[i]
        a0 = run_a0_joint(fo, lut, b2)
        a1 = run_a1_center_first(fo, n_const, lut, b2)
        a2 = run_a2_consensus(fo, n_const, lut, b2)
        a0_p.append(a0.p)
        a0_n.append(a0.n)
        a1_p.append(a1.p)
        a1_n.append(a1.n)
        a1_n1_list.append(a1.n1)
        a1_p0_list.append(a1.p0)
        a2_p.append(a2.p)
        a2_n.append(a2.n)
        cd = candidate_diagnostics(fo, lut, b2, gt_te["n_gt"][i])
        cand_diags.append(cd)
        per_frame.append({
            "frame_id": i,
            "a0": {
                **result_to_dict(a0),
                "axis_error_deg": e_axis_deg(a0.n, gt_te["n_gt"][i]),
                "position_error_m": float(np.linalg.norm(a0.p - gt_te["p_gt"][i])),
            },
            "a1": {
                **result_to_dict(a1),
                "axis_error_n1_deg": e_axis_deg(a1.n1, gt_te["n_gt"][i]),
                "axis_error_n2_deg": e_axis_deg(a1.n2, gt_te["n_gt"][i]),
                "position_error_p1_m": float(np.linalg.norm(a1.p1 - gt_te["p_gt"][i])),
            },
            "a2": {
                **result_to_dict(a2),
                "axis_error_deg": e_axis_deg(a2.n, gt_te["n_gt"][i]),
                "position_error_m": float(np.linalg.norm(a2.p - gt_te["p_gt"][i])),
            },
            "d_ho_gt": d_ho_at_axis(gt_te["n_gt"][i], fo, lut),
            "d_ho_a0": a0.d_ho,
        })

    formal_a0 = _eval_branch(a0_p, a0_n, gt_te)
    formal_a1 = _eval_branch(a1_p, a1_n, gt_te)
    formal_a1_n1 = _eval_branch(a1_p0_list, a1_n1_list, gt_te)
    formal_a2 = _eval_branch(a2_p, a2_n, gt_te)

    print(
        f"[rtwx-o0e0r6] A0 med={formal_a0['axis']['median_e_axis_deg']:.2f}° "
        f"A1 n2={formal_a1['axis']['median_e_axis_deg']:.2f}° "
        f"A1 n1={formal_a1_n1['axis']['median_e_axis_deg']:.2f}° "
        f"A2={formal_a2['axis']['median_e_axis_deg']:.2f}°",
        flush=True,
    )

    a0_sanity: dict[str, Any] = {"tolerance_deg": A0_SANITY_MEDIAN_TOL_DEG}
    r5_summary = r5_root / "summary.json"
    if r5_summary.is_file():
        r5_med = float(json.loads(r5_summary.read_text())["L3_formal_vis"]["axis"]["median_e_axis_deg"])
        delta = abs(formal_a0["axis"]["median_e_axis_deg"] - r5_med)
        a0_sanity["r5_median_axis_deg"] = r5_med
        a0_sanity["a0_median_axis_deg"] = formal_a0["axis"]["median_e_axis_deg"]
        a0_sanity["delta_median_deg"] = delta
        a0_sanity["within_tolerance"] = bool(delta <= A0_SANITY_MEDIAN_TOL_DEG)

    cand_agg = aggregate_candidate_diagnostics(cand_diags)
    pattern, tags = _pattern(a1_pass=formal_a1["formal_pass"], a2_pass=formal_a2["formal_pass"])

    mechanism_table = {
        "A0_joint": formal_a0["axis"],
        "A1_n1": formal_a1_n1["axis"],
        "A1_n2_formal": formal_a1["axis"],
        "A2_formal": formal_a2["axis"],
    }

    result = {
        "header": header,
        "pattern": pattern,
        "patterns": tags,
        "a0_sanity": a0_sanity,
        "formal": {"A0": formal_a0, "A1": formal_a1, "A1_n1_stage": formal_a1_n1, "A2": formal_a2},
        "mechanism_table": mechanism_table,
        "candidate_diagnostics": cand_agg,
        "science_ran": True,
        "unlocks_o0e1_prereg": pattern == "joint_inference_supported",
        "unlocks_o1": False,
        "config": asdict(cfg),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "run.json", result)
    _write_json(root / "per_frame.json", {"frames": per_frame})
    print(f"[rtwx-o0e0r6] pattern={pattern}", flush=True)
    return result

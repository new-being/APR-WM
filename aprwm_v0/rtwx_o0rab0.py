"""RTWX-O0RAB0: re-anchor breadth bakeoff (P0 architecture selection, not scientific)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .geometry.center_axis_inference import VisObs, run_a1_center_first, visibility_center_fused
from .geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from .geometry.periodic_reanchor import (
    ReanchorConfig,
    copy_state,
    is_scheduled,
    make_policy,
)
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, estimate_relative_rigid
from .geometry.relative_state_chain import ChainState, propagate_state
from .geometry.relative_validity import symmetric_support_score
from .geometry.visibility_reference import AxialVisibilityReference
from .rtwx_o0bel0 import CHECKPOINTS, FORMAL_SEED as BEL0_FORMAL_SEED, T_TRANS as BEL0_T
from .rtwx_o0bel0 import N_SEQ as BEL0_N
from .rtwx_o0bel0 import _chain_horizon_table, _largest_h_star
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
)
from .rtwx_o0e0r1 import _axis_stats
from .rtwx_o0e0r2 import _load_p0_split, _load_seg_split
from .rtwx_o0e0r5 import S0_TRAIN_SEED
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import _predict_masks, _train_unet
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_o0rel0 import ICP_CFG
from .rtwx_o0seq0 import RTWXO0SEQ0Config, _build_clouds_seq, _science_pass, collect_sequences_numpy
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0RAB0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0rab0.reanchor_breadth.v1"
PERIODS = (16, 32)
KINDS = ("absolute", "initial", "rolling")
COMPLEXITY_RANK = {"absolute": 0, "initial": 1, "rolling": 2}
POST_RESET_AGES = (0, 1, 2, 4, 8, 16)
CHECKPOINT_SET = frozenset(CHECKPOINTS)


@dataclass(frozen=True)
class RTWXO0RAB0Config:
    output: str = "runs/rtwx_o0rab0"
    bel0_cache: str = "runs/rtwx_o0bel0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = 47603
    n_sequences: int = BEL0_N
    n_transitions: int = BEL0_T
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0RAB0Config) -> RTWXO0RAB0Config:
    if cfg.smoke:
        return replace(cfg, n_sequences=4, n_transitions=16, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0RAB0 must not write there")


def winner_config_hash(kind: str, period: int) -> str:
    payload = f"rab0.winner.v1.kind={kind}.period={period}".encode()
    return hashlib.sha256(payload).hexdigest()


def _vis_obs(cloud: np.ndarray) -> VisObs:
    c = cloud.mean(0) if cloud.size else np.zeros(3)
    return VisObs(
        fused_cloud=cloud,
        c_h=c,
        c_o=c,
        cam_h=np.zeros(3),
        cam_o=np.zeros(3),
    )


def make_abs_fn(n_const: np.ndarray, lut: AxialVisibilityReference, b2: FrozenQuotientB2):
    def _fn(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a1 = run_a1_center_first(_vis_obs(cloud), n_const, lut, b2)
        return a1.p1, a1.n1

    return _fn


def init_anchor_state(cloud0: np.ndarray, n_const: np.ndarray, lut: AxialVisibilityReference) -> ChainState:
    vo0 = _vis_obs(cloud0)
    p0_hat, _, _, _ = visibility_center_fused(n_const, vo0.c_h, vo0.c_o, vo0.cam_h, vo0.cam_o, lut)
    n0 = np.asarray(n_const, dtype=np.float64).reshape(3)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    return ChainState(p=np.asarray(p0_hat, dtype=np.float64).reshape(3), n=n0, t_accum=np.eye(4))


def run_sequence_with_policy(
    clouds: list[np.ndarray],
    gt: dict[str, Any],
    seq_i: int,
    *,
    policy: Any,
    n_const: np.ndarray,
    lut: AxialVisibilityReference,
    d_o: float,
) -> dict[str, Any]:
    state = init_anchor_state(clouds[0], n_const, lut)
    policy.initialize(0, clouds[0], state)
    last_applied_t = 0
    steps: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, float]] = {}
    gains_n: list[float] = []
    gains_p: list[float] = []
    age_errors: dict[int, list[float]] = {a: [] for a in POST_RESET_AGES}

    for t in range(1, len(clouds)):
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], d_o, ICP_CFG)
        if res.valid:
            state = propagate_state(state, res.R, res.t)
        state_before = copy_state(state)
        rr = policy.maybe_reanchor(t, state, clouds[t], d_o, icp_cfg=ICP_CFG)
        state = rr.state
        e_n = float(e_axis_deg(state.n, gt["n"][seq_i, t]))
        e_p = float(np.linalg.norm(state.p - gt["p"][seq_i, t]))
        e_n_before = float(e_axis_deg(state_before.n, gt["n"][seq_i, t]))
        e_p_before = float(np.linalg.norm(state_before.p - gt["p"][seq_i, t]))
        if rr.attempted:
            gains_n.append(e_n_before - e_n)
            gains_p.append(e_p_before - e_p)
        if rr.applied:
            last_applied_t = t
        age = t - last_applied_t
        if age in age_errors:
            age_errors[age].append(e_n)
        q_adj = (
            symmetric_support_score(clouds[t - 1], clouds[t], res.R, res.t, d_o) if res.valid else 0.0
        )
        steps.append({
            "t": int(t),
            "reanchor_attempted": bool(rr.attempted),
            "reanchor_valid": bool(rr.valid),
            "reanchor_applied": bool(rr.applied),
            "source_frame": rr.source_frame,
            "q_anchor": rr.q_support,
            "n_corr": rr.n_corr,
            "icp_rmse": rr.icp_rmse,
            "q_rel_adj": float(q_adj),
            "evaluator": {
                "axis_error_deg": e_n,
                "position_error_m": e_p,
                "axis_error_before_deg": e_n_before,
                "position_error_before_m": e_p_before,
                "reset_gain_axis_deg": (e_n_before - e_n) if rr.attempted else None,
            },
        })
        if t in CHECKPOINT_SET:
            checkpoints[str(t)] = {
                "chain_axis_error_deg": e_n,
                "chain_position_error_m": e_p,
            }

    gn = np.asarray(gains_n, dtype=np.float64)
    gp = np.asarray(gains_p, dtype=np.float64)
    return {
        "sequence_id": int(seq_i),
        "checkpoints": checkpoints,
        "steps": steps,
        "reset_gain": {
            "median_Gn": float(np.median(gn)) if gn.size else float("nan"),
            "P_Gn_gt0": float(np.mean(gn > 0)) if gn.size else float("nan"),
            "P_Gn_lt_m5": float(np.mean(gn < -5.0)) if gn.size else float("nan"),
            "median_Gp": float(np.median(gp)) if gp.size else float("nan"),
            "n": int(gn.size),
        },
        "post_reset_age": {
            str(a): (float(np.median(v)) if v else float("nan"))
            for a, v in age_errors.items()
        },
    }


def _cell_row(per_seq: list[dict[str, Any]], kind: str, period: int) -> dict[str, Any]:
    horizon, pass_at_h = _chain_horizon_table(per_seq)
    h_star = _largest_h_star(pass_at_h)
    h64 = horizon.get("64", {})
    axis64 = h64.get("axis", {})
    gn = np.asarray([sq["reset_gain"]["median_Gn"] for sq in per_seq], dtype=np.float64)
    all_g = []
    for sq in per_seq:
        for st in sq["steps"]:
            g = st["evaluator"].get("reset_gain_axis_deg")
            if g is not None:
                all_g.append(g)
    g_all = np.asarray(all_g, dtype=np.float64)
    attempts = [st for sq in per_seq for st in sq["steps"] if st["reanchor_attempted"]]
    applied = [st for st in attempts if st["reanchor_applied"]]
    return {
        "kind": kind,
        "period": int(period),
        "H_star": int(h_star),
        "h64_median": float(axis64.get("median_e_axis_deg", float("nan"))),
        "h64_p90": float(axis64.get("p90_e_axis_deg", float("nan"))),
        "horizon": horizon,
        "pass_at_h": {str(k): v for k, v in pass_at_h.items()},
        "median_reset_gain": float(np.nanmedian(g_all)) if g_all.size else float("nan"),
        "improve_rate": float(np.mean(g_all > 0)) if g_all.size else float("nan"),
        "P_Gn_lt_m5": float(np.mean(g_all < -5.0)) if g_all.size else float("nan"),
        "n_attempted": len(attempts),
        "n_applied": len(applied),
        "apply_rate": float(len(applied) / len(attempts)) if attempts else float("nan"),
        "post_reset_age_median": {
            a: float(np.nanmedian([sq["post_reset_age"][str(a)] for sq in per_seq]))
            for a in POST_RESET_AGES
        },
        "per_seq_median_Gn": float(np.nanmedian(gn)) if gn.size else float("nan"),
    }


def select_winner(cells: list[dict[str, Any]], b0: dict[str, Any]) -> dict[str, Any] | None:
    h0 = int(b0["H_star"])
    cands = [c for c in cells if c["kind"] != "none" and int(c["H_star"]) > h0]
    if not cands:
        return None

    def _key(c: dict[str, Any]) -> tuple:
        p90 = c["h64_p90"]
        med = c["h64_median"]
        p90_k = p90 if np.isfinite(p90) else 1.0e9
        med_k = med if np.isfinite(med) else 1.0e9
        return (
            -int(c["H_star"]),
            p90_k,
            med_k,
            -int(c["period"]),
            COMPLEXITY_RANK.get(c["kind"], 9),
        )

    return min(cands, key=_key)


def cells_spec() -> list[ReanchorConfig]:
    out = [ReanchorConfig(kind="none", period=0)]
    for kind in KINDS:
        for k in PERIODS:
            out.append(ReanchorConfig(kind=kind, period=k))
    return out


def scheduled_times(period: int, t_max: int) -> list[int]:
    return [t for t in range(1, t_max + 1) if is_scheduled(t, period)]


def _load_assets(cfg: RTWXO0RAB0Config):
    repo = Path(cfg.robotwin_repo)
    if not repo.is_absolute():
        repo = repo.resolve()
    d_o = load_D_O(str(repo) if repo.is_dir() else "/root/RoboTwin")
    lut = AxialVisibilityReference.load(_resolve_apr(cfg.r5_run) / "visibility_lut.npz")
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
    _, gt_tr = _load_p0_split(_resolve_apr(cfg.p0_cache), "train")
    n_const = gt_tr["n_gt"].sum(0)
    n_const = n_const / max(np.linalg.norm(n_const), 1e-12)
    return d_o, lut, b2, n_const, repo


def _train_s0(cfg: RTWXO0RAB0Config):
    natural_root = _resolve_apr(cfg.natural_cache)
    rh_tr, ro_tr, mh_tr, mo_tr = _load_seg_split(natural_root, "train")
    rh_va, ro_va, mh_va, mo_va = _load_seg_split(natural_root, "val")
    print("[rtwx-o0rab0] train S0", flush=True)
    return _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )


def load_bel0_formal_cache(bel0_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    obs_p = bel0_root / "cache" / "formal" / "observations" / "obs.npz"
    gt_p = bel0_root / "cache" / "gt" / "gt.npz"
    if not gt_p.is_file():
        gt_p = bel0_root / "cache" / "formal" / "gt" / "gt.npz"
    if not obs_p.is_file() or not gt_p.is_file():
        raise FileNotFoundError(f"BEL0 formal cache missing under {bel0_root}")
    return dict(np.load(obs_p)), dict(np.load(gt_p))


def eval_cell(
    obs: dict[str, Any],
    gt: dict[str, Any],
    *,
    kind: str,
    period: int,
    s0: Any,
    lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
    n_const: np.ndarray,
    d_o: float,
    n_seq: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    abs_fn = make_abs_fn(n_const, lut, b2)
    per_seq: list[dict[str, Any]] = []
    for si in range(n_seq):
        policy = make_policy(kind, period, abs_fn=abs_fn)
        clouds = _build_clouds_seq(obs, gt, s0, si, rng=rng, use_gt_mask=False)
        per_seq.append(run_sequence_with_policy(
            clouds, gt, si, policy=policy, n_const=n_const, lut=lut, d_o=d_o,
        ))
    row = _cell_row(per_seq, kind, period)
    return per_seq, row


def run_rtwx_o0rab0(output: str | Path, config: RTWXO0RAB0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0RAB0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

    bel0_root = _resolve_apr(cfg.bel0_cache)
    if cfg.smoke:
        sub = RTWXO0SEQ0Config(
            output=str(root / "smoke_cache"),
            natural_cache=cfg.natural_cache,
            p0_cache=cfg.p0_cache,
            r5_run=cfg.r5_run,
            robotwin_repo=cfg.robotwin_repo,
            backend="numpy",
            seed=cfg.seed,
            seq_seed=BEL0_FORMAL_SEED,
            n_seq=cfg.n_sequences,
            t_trans=cfg.n_transitions,
            epochs_unet=cfg.epochs_unet,
            smoke=False,
        )
        print("[rtwx-o0rab0] smoke numpy collect", flush=True)
        obs, gt = collect_sequences_numpy(sub)
    else:
        obs, gt = load_bel0_formal_cache(bel0_root)
        n_avail = int(obs["rgb_h"].shape[0])
        t_avail = int(obs["rgb_h"].shape[1]) - 1
        cfg = replace(
            cfg,
            n_sequences=min(cfg.n_sequences, n_avail),
            n_transitions=min(cfg.n_transitions, t_avail),
        )

    d_o, lut, b2, n_const, _repo = _load_assets(cfg)
    s0 = _train_s0(cfg)
    rng = np.random.default_rng(cfg.seed)

    table: list[dict[str, Any]] = []
    per_cell: dict[str, Any] = {}
    for spec in cells_spec():
        key = f"{spec.kind}_K{spec.period}" if spec.kind != "none" else "B0"
        print(f"[rtwx-o0rab0] eval {key}", flush=True)
        per_seq, row = eval_cell(
            obs, gt, kind=spec.kind, period=spec.period,
            s0=s0, lut=lut, b2=b2, n_const=n_const, d_o=d_o,
            n_seq=cfg.n_sequences, rng=rng,
        )
        table.append({k: v for k, v in row.items() if k not in ("horizon",)})
        per_cell[key] = {"row": row, "n_seq": len(per_seq)}
        _write_json(root / f"per_sequence_{key}.json", per_seq)

    b0 = next(r for r in table if r["kind"] == "none")
    winner = select_winner(table, b0)
    if winner is None:
        summary = {
            "pattern": "reanchor_breadth_selection_complete",
            "scientific_result": False,
            "winner": None,
            "winner_id": None,
            "winner_config_hash": None,
            "next_frontier": ["history_retrieval", "persistent_surface"],
            "H_star_B0": int(b0["H_star"]),
            "table": table,
            "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
            "b2_config_hash": R5_B2_CONFIG_HASH,
            "bel0_formal_seed": BEL0_FORMAL_SEED,
        }
    else:
        wid = f"{winner['kind']}_K{winner['period']}"
        summary = {
            "pattern": "reanchor_breadth_selection_complete",
            "scientific_result": False,
            "winner": {"kind": winner["kind"], "period": int(winner["period"])},
            "winner_id": wid,
            "winner_config_hash": winner_config_hash(winner["kind"], int(winner["period"])),
            "next_frontier": [],
            "H_star_B0": int(b0["H_star"]),
            "H_star_winner_p0": int(winner["H_star"]),
            "table": table,
            "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
            "b2_config_hash": R5_B2_CONFIG_HASH,
            "bel0_formal_seed": BEL0_FORMAL_SEED,
        }

    _write_json(root / "header.json", {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "scientific_result": False,
        "purpose": "architecture_selection",
    })
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0rab0] winner={summary.get('winner_id')}", flush=True)
    return summary

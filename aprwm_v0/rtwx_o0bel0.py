"""RTWX-O0BEL0: shadow relative belief / re-anchor observability audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry.center_axis_inference import VisObs, visibility_center_fused
from .geometry.frozen_quotient_b2 import FrozenQuotientB2, R5_B2_CONFIG_HASH
from .geometry.relative_belief import (
    RelativeBeliefState,
    mean_surprise,
    recent_hazard,
    registration_surprise,
    update_belief,
)
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, estimate_relative_rigid
from .geometry.relative_state_chain import ChainState, propagate_state
from .geometry.relative_validity import relative_safe_label, symmetric_support_score
from .geometry.visibility_reference import AxialVisibilityReference
from .rtwx_o0e0 import (
    CAD_FPS_SEED,
    K_SPHERE,
    N_CAD,
    SPHERE_SEED,
    cad_hr_profile,
    e_axis_deg,
    fibonacci_sphere,
)
from .rtwx_o0e0r1 import _axis_stats, _ep_stats
from .rtwx_o0e0r2 import _load_p0_split, _load_seg_split
from .rtwx_o0e0r5 import S0_TRAIN_SEED
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0g2 import EP_MAX, MED_MAX, _predict_masks, _train_unet
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5b import _fps, load_scaled_cad
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_o0rel0 import ICP_CFG
from .rtwx_o0r import MED_ER_MAX, P90_ER_MAX
from .rtwx_o0seq0 import (
    RTWXO0SEQ0Config,
    _build_clouds_seq,
    _science_pass,
    collect_sequences_numpy,
    collect_sequences_robotwin,
)
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0BEL0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0bel0.relative_belief_audit.v1"
SEED = 47602
CAL_SEED = 37610
FORMAL_SEED = 37611
N_SEQ = 100
T_TRANS = 64
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64)
REPLICATION_CHECKPOINTS = (1, 2, 4, 8, 16)
FAILURE_HORIZON = 4
BELIEF_EPS = 1e-6
RECENT_WINDOW = 4
MIN_FAILURE_SEQUENCE_RATE = 0.10
PRIMARY_AUC_MIN = 0.75
AUC_GAIN_OVER_TIME_MIN = 0.05
TRIGGER_PRECISION_TARGET = 0.90
REPLICATION_H16_TOLERANCE_DEG = 2.0
SEQ0_H16_AXIS_MEDIAN_REF = 8.0
PERIODIC_RESET_K = (8, 16, 32)
SPLITS = ("calibration", "formal")
EXPECTED_REL_ICP_HASH = REL_ICP_CONFIG_HASH
CHECKPOINT_SET = frozenset(CHECKPOINTS)


@dataclass(frozen=True)
class RTWXO0BEL0Config:
    output: str = "runs/rtwx_o0bel0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seed: int = SEED
    cal_seed: int = CAL_SEED
    formal_seed: int = FORMAL_SEED
    n_sequences: int = N_SEQ
    n_transitions: int = T_TRANS
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0BEL0Config) -> RTWXO0BEL0Config:
    if cfg.smoke:
        return replace(cfg, n_sequences=4, n_transitions=16, epochs_unet=4, backend="numpy")
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0BEL0 must not write there")


def _assert_frozen_hashes() -> None:
    if REL_ICP_CONFIG_HASH != EXPECTED_REL_ICP_HASH:
        raise RuntimeError("REL ICP config hash drift; BEL0 requires frozen SEQ0/REL0 ICP")


def _largest_h_star(pass_at_h: dict[int, bool]) -> int:
    h_star = 0
    for h in CHECKPOINTS:
        if pass_at_h.get(h, False):
            h_star = h
        else:
            break
    return h_star


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    tpr = tps / max(y.sum(), 1)
    fpr = fps / max((~y).sum(), 1)
    return float(np.trapz(tpr, fpr))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0 or not np.any(y):
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    rec = tp / max(y.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.trapz(prec, rec))


def future_failure_labels(unsafe: list[bool] | np.ndarray, horizon: int = FAILURE_HORIZON) -> np.ndarray:
    u = np.asarray(unsafe, dtype=bool)
    labels = np.full(u.shape[0], np.nan, dtype=np.float64)
    for t in range(u.shape[0] - horizon):
        labels[t] = float(any(u[t + 1: t + 1 + horizon]))
    return labels


def _seq_cfg_for_split(cfg: RTWXO0BEL0Config, split: Literal["calibration", "formal"]) -> RTWXO0SEQ0Config:
    seq_seed = cfg.cal_seed if split == "calibration" else cfg.formal_seed
    return RTWXO0SEQ0Config(
        output=cfg.output,
        natural_cache=cfg.natural_cache,
        p0_cache=cfg.p0_cache,
        r5_run=cfg.r5_run,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        seed=cfg.seed,
        seq_seed=seq_seed,
        n_seq=cfg.n_sequences,
        t_trans=cfg.n_transitions,
        epochs_unet=cfg.epochs_unet,
        smoke=cfg.smoke,
    )


def _load_or_collect_split(root: Path, split: Literal["calibration", "formal"], cfg: RTWXO0BEL0Config) -> tuple[dict[str, Any], dict[str, Any]]:
    split_key = "cal" if split == "calibration" else "formal"
    cache_obs = root / "cache" / split_key / "observations" / "obs.npz"
    cache_gt = root / "cache" / split_key / "gt" / "gt.npz"
    if cache_obs.is_file() and cache_gt.is_file():
        return dict(np.load(cache_obs)), dict(np.load(cache_gt))
    sub = _seq_cfg_for_split(cfg, split)
    print(f"[rtwx-o0bel0] collecting {split} n_seq={sub.n_seq} T={sub.t_trans} backend={sub.backend}", flush=True)
    if sub.backend == "numpy":
        obs, gt = collect_sequences_numpy(sub)
    else:
        obs, gt = collect_sequences_robotwin(sub)
    cache_obs.parent.mkdir(parents=True, exist_ok=True)
    cache_gt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_obs, **obs)
    np.savez_compressed(cache_gt, **gt)
    return obs, gt


def _eval_sequence_bel0(
    obs: dict[str, Any],
    gt: dict[str, Any],
    seq_i: int,
    *,
    s0: Any,
    lut: AxialVisibilityReference,
    n_const: np.ndarray,
    d_o: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    clouds = _build_clouds_seq(obs, gt, s0, seq_i, rng=rng, use_gt_mask=False)
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

    state = ChainState(
        p=np.asarray(p0_hat, dtype=np.float64).reshape(3),
        n=np.asarray(n0_hat, dtype=np.float64).reshape(3) / max(np.linalg.norm(n0_hat), 1e-12),
        t_accum=np.eye(4),
    )
    belief = RelativeBeliefState()
    q_history: list[float] = []
    steps: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, float]] = {}

    for t in range(1, len(clouds)):
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], d_o, ICP_CFG)
        if res.valid:
            state = propagate_state(state, res.R, res.t)
        q = symmetric_support_score(clouds[t - 1], clouds[t], res.R, res.t, d_o) if res.valid else 0.0
        belief = update_belief(belief, q, eps=BELIEF_EPS)
        q_history.append(q)
        e_n = float(e_axis_deg(state.n, gt["n"][seq_i, t]))
        e_p = float(np.linalg.norm(state.p - gt["p"][seq_i, t]))
        steps.append({
            "sequence_id": int(seq_i),
            "t": int(t),
            "online": {
                "q_rel": float(q),
                "icp_rmse": float(res.rmse),
                "n_corr": int(res.n_corr),
                "relative_valid": bool(res.valid),
                "surprise": float(registration_surprise(q, BELIEF_EPS)),
                "cumulative_surprise": float(belief.cumulative_surprise),
                "mean_surprise": float(mean_surprise(belief.cumulative_surprise, t)),
                "recent_hazard": float(recent_hazard(q_history, RECENT_WINDOW, BELIEF_EPS)),
                "trigger_shadow": False,
            },
            "evaluator": {
                "axis_error_deg": e_n,
                "position_error_m": e_p,
                "unsafe_now": not relative_safe_label(e_n, e_p),
                "unsafe_within_4": None,
            },
        })
        if t in CHECKPOINT_SET:
            checkpoints[str(t)] = {
                "chain_axis_error_deg": e_n,
                "chain_position_error_m": e_p,
            }

    unsafe = [bool(s["evaluator"]["unsafe_now"]) for s in steps]
    fut = future_failure_labels(unsafe, FAILURE_HORIZON)
    for i, s in enumerate(steps):
        if np.isfinite(fut[i]):
            s["evaluator"]["unsafe_within_4"] = bool(fut[i])

    return {
        "sequence_id": int(seq_i),
        "anchor": {
            "axis_error_deg": float(e_axis_deg(n0_hat, gt["n"][seq_i, 0])),
            "position_error_m": float(np.linalg.norm(p0_hat - gt["p"][seq_i, 0])),
        },
        "checkpoints": checkpoints,
        "steps": steps,
        "has_failure": bool(any(unsafe)),
        "first_failure_t": int(next((s["t"] for s in steps if s["evaluator"]["unsafe_now"]), -1)),
    }


def _chain_horizon_table(per_seq: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, bool]]:
    adapted = []
    for sq in per_seq:
        adapted.append({"checkpoints": {
            h: {
                "chain_axis_error_deg": ck["chain_axis_error_deg"],
                "chain_position_error_m": ck["chain_position_error_m"],
            }
            for h, ck in sq["checkpoints"].items()
        }})
    horizon: dict[str, dict[str, Any]] = {}
    pass_at_h: dict[int, bool] = {}
    for h in CHECKPOINTS:
        e_n, e_p = [], []
        for sq in per_seq:
            ck = sq["checkpoints"].get(str(h))
            if ck is None:
                continue
            e_n.append(ck["chain_axis_error_deg"])
            e_p.append(ck["chain_position_error_m"])
        e_n_a = np.asarray(e_n, dtype=np.float64)
        e_p_a = np.asarray(e_p, dtype=np.float64)
        if e_n_a.size == 0:
            row = {"axis": _axis_stats(np.array([np.nan])), "position": _ep_stats(np.array([np.nan])), "pass": False}
        else:
            row = {
                "axis": _axis_stats(e_n_a),
                "position": {**_ep_stats(e_p_a), "E_p": float(np.nanmean(e_p_a))},
                "pass": _science_pass(e_n_a, e_p_a),
            }
        horizon[str(h)] = row
        pass_at_h[h] = row["pass"]
    return horizon, pass_at_h


def _replication_ok(per_seq: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    _, pass_at_h = _chain_horizon_table(per_seq)
    rep_pass = all(pass_at_h.get(h, False) for h in REPLICATION_CHECKPOINTS)
    h16 = [sq["checkpoints"].get("16", {}) for sq in per_seq]
    h16_med = float(np.median([c.get("chain_axis_error_deg", np.nan) for c in h16]))
    med_delta = abs(h16_med - SEQ0_H16_AXIS_MEDIAN_REF)
    med_ok = med_delta <= REPLICATION_H16_TOLERANCE_DEG
    ok = rep_pass and med_ok
    return ok, {
        "pass_at_h_replication": {str(k): pass_at_h.get(k, False) for k in REPLICATION_CHECKPOINTS},
        "h16_axis_median": h16_med,
        "h16_axis_median_delta_vs_seq0": med_delta,
        "ok": ok,
    }


def _failure_sequence_rate(per_seq: list[dict[str, Any]]) -> float:
    if not per_seq:
        return 0.0
    return float(np.mean([1.0 if sq["has_failure"] else 0.0 for sq in per_seq]))


def _step_hazard_dataset(per_seq: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    u, y, t_idx, u_bar, r_h = [], [], [], [], []
    for sq in per_seq:
        for st in sq["steps"]:
            lab = st["evaluator"].get("unsafe_within_4")
            if lab is None:
                continue
            u.append(st["online"]["cumulative_surprise"])
            u_bar.append(st["online"]["mean_surprise"])
            r_h.append(st["online"]["recent_hazard"])
            y.append(bool(lab))
            t_idx.append(float(st["t"]))
    return {
        "U": np.asarray(u, dtype=np.float64),
        "U_bar": np.asarray(u_bar, dtype=np.float64),
        "R": np.asarray(r_h, dtype=np.float64),
        "t": np.asarray(t_idx, dtype=np.float64),
        "y": np.asarray(y, dtype=bool),
    }


def _observability(ds: dict[str, np.ndarray]) -> dict[str, float]:
    y_all = ds["y"]
    mask = np.isfinite(y_all.astype(np.float64))
    y = y_all[mask].astype(bool)
    if y.size == 0 or len(np.unique(y)) < 2:
        return {
            "auc_cum_surprise": float("nan"),
            "auc_time": float("nan"),
            "auc_gain_over_time": float("nan"),
            "auprc_cum_surprise": float("nan"),
            "auc_mean_surprise": float("nan"),
            "auc_recent_hazard": float("nan"),
            "n_samples": int(y.size),
            "positive_rate": float(y.mean()) if y.size else float("nan"),
        }
    u = ds["U"][mask]
    t = ds["t"][mask]
    auc_u = _auroc(u, y)
    auc_t = _auroc(t, y)
    return {
        "auc_cum_surprise": auc_u,
        "auc_time": auc_t,
        "auc_gain_over_time": float(auc_u - auc_t) if np.isfinite(auc_u) and np.isfinite(auc_t) else float("nan"),
        "auprc_cum_surprise": _auprc(u, y),
        "auc_mean_surprise": _auroc(ds["U_bar"][mask], y),
        "auc_recent_hazard": _auroc(ds["R"][mask], y),
        "n_samples": int(y.size),
        "positive_rate": float(y.mean()),
    }


def _sequence_trigger_eval(per_seq: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    useful = false_alarm = triggered = failed = 0
    lead_steps: list[int] = []
    for sq in per_seq:
        steps = sq["steps"]
        t_fail = sq["first_failure_t"]
        trig_t = next((st["t"] for st in steps if st["online"]["cumulative_surprise"] >= tau), None)
        if trig_t is not None:
            triggered += 1
            for st in steps:
                if st["t"] == trig_t:
                    st["online"]["trigger_shadow"] = True
        if t_fail < 0:
            if trig_t is not None:
                false_alarm += 1
            continue
        failed += 1
        if trig_t is None:
            continue
        lead = t_fail - trig_t
        if 0 < lead <= FAILURE_HORIZON:
            useful += 1
            lead_steps.append(lead)
        elif lead > FAILURE_HORIZON:
            false_alarm += 1
        else:
            false_alarm += 1
    precision = float(useful / triggered) if triggered else float("nan")
    recall = float(useful / failed) if failed else float("nan")
    return {
        "tau": float(tau),
        "precision": precision,
        "useful_recall": recall,
        "median_lead_steps": float(np.median(lead_steps)) if lead_steps else float("nan"),
        "trigger_rate": float(triggered / max(len(per_seq) * len(per_seq[0]["steps"]) if per_seq else 1, 1)),
        "mean_triggers_per_sequence": float(triggered / max(len(per_seq), 1)),
        "n_triggered": triggered,
        "n_useful": useful,
        "n_false_alarm": false_alarm,
        "n_failed_sequences": failed,
    }


def _calibrate_trigger_tau(per_seq: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    candidates = sorted({
        st["online"]["cumulative_surprise"]
        for sq in per_seq for st in sq["steps"]
    })
    best: dict[str, Any] | None = None
    best_tau: float | None = None
    for tau in candidates:
        ev = _sequence_trigger_eval(per_seq, tau)
        prec = ev["precision"]
        if not np.isfinite(prec) or prec < TRIGGER_PRECISION_TARGET:
            continue
        rec = ev["useful_recall"]
        if best is None or rec > best["useful_recall"] + 1e-12 or (
            abs(rec - best["useful_recall"]) <= 1e-12 and tau > best_tau
        ):
            best = ev
            best_tau = float(tau)
    return best_tau, best or {}


def _run_chain_with_resets(
    clouds: list[np.ndarray],
    gt: dict[str, Any],
    seq_i: int,
    *,
    p0_hat: np.ndarray,
    n0_hat: np.ndarray,
    d_o: float,
    reset_times: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, ChainState]:
    state = ChainState(
        p=np.asarray(p0_hat, dtype=np.float64).reshape(3),
        n=np.asarray(n0_hat, dtype=np.float64).reshape(3) / max(np.linalg.norm(n0_hat), 1e-12),
        t_accum=np.eye(4),
    )
    ckpt: dict[int, ChainState] = {0: state}
    for t in range(1, len(clouds)):
        if t in reset_times:
            p_gt, n_gt = reset_times[t]
            state = ChainState(p=p_gt.copy(), n=n_gt.copy(), t_accum=state.t_accum.copy())
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], d_o, ICP_CFG)
        if res.valid:
            state = propagate_state(state, res.R, res.t)
        if t in CHECKPOINT_SET:
            ckpt[t] = ChainState(p=state.p.copy(), n=state.n.copy(), t_accum=state.t_accum.copy())
    return ckpt


def _h_star_from_ckpt(ckpt: dict[int, ChainState], gt: dict[str, Any], seq_i: int) -> int:
    pass_at_h: dict[int, bool] = {}
    for h in CHECKPOINTS:
        st = ckpt.get(h)
        if st is None:
            pass_at_h[h] = False
            continue
        e_n = e_axis_deg(st.n, gt["n"][seq_i, h])
        e_p = float(np.linalg.norm(st.p - gt["p"][seq_i, h]))
        pass_at_h[h] = _science_pass(np.array([e_n]), np.array([e_p]))
    return _largest_h_star(pass_at_h)


def _oracle_ceiling(
    per_seq: list[dict[str, Any]],
    obs: dict[str, Any],
    gt: dict[str, Any],
    *,
    s0: Any,
    lut: AxialVisibilityReference,
    n_const: np.ndarray,
    d_o: float,
    rng: np.random.Generator,
    tau: float | None,
) -> dict[str, Any]:
    h_stars: dict[str, list[int]] = {"noreset": [], "adaptive_reset": []}
    for k in PERIODIC_RESET_K:
        h_stars[f"periodic_{k}"] = []

    for sq in per_seq:
        si = sq["sequence_id"]
        clouds = _build_clouds_seq(obs, gt, s0, si, rng=rng, use_gt_mask=False)
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

        base_ckpt = _run_chain_with_resets(clouds, gt, si, p0_hat=p0_hat, n0_hat=n0_hat, d_o=d_o, reset_times={})
        h_stars["noreset"].append(_h_star_from_ckpt(base_ckpt, gt, si))

        if tau is not None:
            trig = next((st["t"] for st in sq["steps"] if st["online"]["cumulative_surprise"] >= tau), None)
            if trig is not None:
                resets = {
                    trig: (gt["p"][si, trig], gt["n"][si, trig]),
                }
                ad_ckpt = _run_chain_with_resets(clouds, gt, si, p0_hat=p0_hat, n0_hat=n0_hat, d_o=d_o, reset_times=resets)
                h_stars["adaptive_reset"].append(_h_star_from_ckpt(ad_ckpt, gt, si))

        for k in PERIODIC_RESET_K:
            resets = {t: (gt["p"][si, t], gt["n"][si, t]) for t in range(k, len(clouds), k)}
            p_ckpt = _run_chain_with_resets(clouds, gt, si, p0_hat=p0_hat, n0_hat=n0_hat, d_o=d_o, reset_times=resets)
            h_stars[f"periodic_{k}"].append(_h_star_from_ckpt(p_ckpt, gt, si))

    def _med(key: str) -> float:
        arr = h_stars.get(key, [])
        return float(np.median(arr)) if arr else float("nan")

    return {
        "noreset_H_star_median": _med("noreset"),
        "adaptive_reset_H_star_median": _med("adaptive_reset"),
        "periodic_8_H_star_median": _med("periodic_8"),
        "periodic_16_H_star_median": _med("periodic_16"),
        "periodic_32_H_star_median": _med("periodic_32"),
    }


def _pattern(
    *,
    replication_ok: bool,
    failure_rate: float,
    observability: dict[str, float],
    trigger_cal_ok: bool,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    if not replication_ok:
        return "belief_sequence_replication_failure", tags
    if failure_rate < MIN_FAILURE_SEQUENCE_RATE:
        return "belief_failure_support_insufficient", tags
    auc_u = observability.get("auc_cum_surprise", float("nan"))
    gain = observability.get("auc_gain_over_time", float("nan"))
    if not (np.isfinite(auc_u) and np.isfinite(gain) and auc_u >= PRIMARY_AUC_MIN and gain >= AUC_GAIN_OVER_TIME_MIN):
        return "relative_belief_insufficient", tags
    if not trigger_cal_ok:
        tags.append("trigger_calibration_failed")
    return "relative_belief_observable", tags


def _eval_split(
    split: Literal["calibration", "formal"],
    *,
    obs: dict[str, Any],
    gt: dict[str, Any],
    s0: Any,
    lut: AxialVisibilityReference,
    n_const: np.ndarray,
    d_o: float,
    rng: np.random.Generator,
    cfg: RTWXO0BEL0Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seq: list[dict[str, Any]] = []
    for si in range(cfg.n_sequences):
        if si % 20 == 0:
            print(f"[rtwx-o0bel0] eval {split} seq {si}/{cfg.n_sequences}", flush=True)
        per_seq.append(_eval_sequence_bel0(obs, gt, si, s0=s0, lut=lut, n_const=n_const, d_o=d_o, rng=rng))
    per_step = [st for sq in per_seq for st in sq["steps"]]
    return per_seq, per_step


def run_rtwx_o0bel0(output: str | Path, config: RTWXO0BEL0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0BEL0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    natural_root = _resolve_apr(cfg.natural_cache)
    p0_root = _resolve_apr(cfg.p0_cache)
    r5_root = _resolve_apr(cfg.r5_run)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    _assert_frozen_hashes()

    split_data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for split in SPLITS:
        split_data[split] = _load_or_collect_split(root, split, cfg)

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
    print("[rtwx-o0bel0] train S0", flush=True)
    s0 = _train_unet(
        np.concatenate([rh_tr, ro_tr]), np.concatenate([mh_tr, mo_tr]),
        np.concatenate([rh_va, ro_va]), np.concatenate([mh_va, mo_va]),
        seed=S0_TRAIN_SEED, epochs=cfg.epochs_unet,
    )

    rng = np.random.default_rng(cfg.seed)
    split_results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        obs, gt = split_data[split]
        per_seq, per_step = _eval_split(
            split, obs=obs, gt=gt, s0=s0, lut=lut, n_const=n_const, d_o=d_o, rng=rng, cfg=cfg,
        )
        horizon, pass_at_h = _chain_horizon_table(per_seq)
        rep_ok, rep_info = _replication_ok(per_seq)
        fail_rate = _failure_sequence_rate(per_seq)
        hazard_ds = _step_hazard_dataset(per_seq)
        obs_metrics = _observability(hazard_ds) if split == "calibration" else {}
        split_key = "calibration" if split == "calibration" else "formal"
        out_dir = root / split_key
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "per_sequence.json", per_seq)
        _write_json(out_dir / "per_step.json", per_step)
        split_results[split] = {
            "per_seq": per_seq,
            "per_step": per_step,
            "horizon": horizon,
            "pass_at_h": pass_at_h,
            "H_star": _largest_h_star(pass_at_h),
            "replication": rep_info,
            "failure_sequence_rate": fail_rate,
            "observability": obs_metrics,
        }

    cal = split_results["calibration"]
    formal = split_results["formal"]
    replication_ok = bool(cal["replication"]["ok"] and formal["replication"]["ok"])
    failure_rate = float(formal["failure_sequence_rate"])
    observability = cal["observability"]

    tau_star: float | None = None
    trigger_cal: dict[str, Any] = {}
    trigger_formal: dict[str, Any] = {}
    trigger_cal_ok = False
    l3_ok = (
        replication_ok
        and failure_rate >= MIN_FAILURE_SEQUENCE_RATE
        and np.isfinite(observability.get("auc_cum_surprise", np.nan))
        and observability.get("auc_cum_surprise", 0.0) >= PRIMARY_AUC_MIN
        and observability.get("auc_gain_over_time", 0.0) >= AUC_GAIN_OVER_TIME_MIN
    )
    if l3_ok:
        tau_star, trigger_cal = _calibrate_trigger_tau(cal["per_seq"])
        trigger_cal_ok = tau_star is not None
        if tau_star is not None:
            trigger_formal = _sequence_trigger_eval(formal["per_seq"], tau_star)

    pattern, tags = _pattern(
        replication_ok=replication_ok,
        failure_rate=failure_rate,
        observability=observability,
        trigger_cal_ok=trigger_cal_ok,
    )

    oracle_ceiling = _oracle_ceiling(
        formal["per_seq"],
        split_data["formal"][0],
        split_data["formal"][1],
        s0=s0,
        lut=lut,
        n_const=n_const,
        d_o=d_o,
        rng=np.random.default_rng(cfg.formal_seed + 7),
        tau=tau_star,
    )

    summary = {
        "pattern": pattern,
        "tags": tags,
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "b2_config_hash": R5_B2_CONFIG_HASH,
        "replication": {
            "ok": replication_ok,
            "calibration": cal["replication"],
            "formal": formal["replication"],
        },
        "horizon": {
            "H_star_calibration": int(cal["H_star"]),
            "H_star_formal": int(formal["H_star"]),
            "failure_sequence_rate_calibration": cal["failure_sequence_rate"],
            "failure_sequence_rate_formal": failure_rate,
            "horizon_table_formal": formal["horizon"],
            "pass_at_h_formal": {str(k): v for k, v in formal["pass_at_h"].items()},
        },
        "observability": observability,
        "trigger": {
            "calibration": trigger_cal,
            "formal": trigger_formal,
            "tau_U": tau_star,
        },
        "oracle_ceiling": oracle_ceiling,
        "unlocks_sparse_reanchor_prereg": pattern == "relative_belief_observable" and trigger_cal_ok,
        "unlocks_o0e1": False,
    }

    _write_json(root / "header.json", {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cal_seed": cfg.cal_seed,
        "formal_seed": cfg.formal_seed,
    })
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0bel0] pattern={pattern}", flush=True)
    return summary

"""RTWX-O0MEMB0: reference memory breadth probe (retrieval vs persistent surface)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .geometry.historical_reference import (
    RETRIEVAL_AGES,
    apply_retrieval,
    retrieve_reference,
)
from .geometry.periodic_reanchor import Keyframe, copy_state
from .geometry.persistent_surface_memory import (
    PersistentSurfaceMemory,
    invert_hom,
    register_surface_to_cloud,
    transform_cloud,
)
from .geometry.relative_rigid_registration import REL_ICP_CONFIG_HASH, estimate_relative_rigid
from .geometry.relative_state_chain import propagate_state
from .rtwx_o0bel0 import CHECKPOINTS, FORMAL_SEED as BEL0_FORMAL_SEED, N_SEQ as BEL0_N, T_TRANS as BEL0_T
from .rtwx_o0bel0 import _chain_horizon_table, _largest_h_star
from .rtwx_o0e0 import e_axis_deg
from .rtwx_o0g2 import EPOCHS as UNET_EPOCHS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_o0rab0 import (
    _load_assets,
    _train_s0,
    init_anchor_state,
    load_bel0_formal_cache,
)
from .rtwx_o0rel0 import ICP_CFG
from .rtwx_o0seq0 import RTWXO0SEQ0Config, _build_clouds_seq, collect_sequences_numpy
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0MEMB0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0memb0.memory_breadth.v1"
CORRECTION_T = frozenset({32, 64})
KINDS = ("none", "retrieval", "surface")
COMPLEXITY_RANK = {"retrieval": 0, "surface": 1}
CHECKPOINT_SET = frozenset(CHECKPOINTS)
Kind = Literal["none", "retrieval", "surface"]


@dataclass(frozen=True)
class RTWXO0MEMB0Config:
    output: str = "runs/rtwx_o0memb0"
    bel0_cache: str = "runs/rtwx_o0bel0"
    natural_cache: str = "runs/rtwx_o0e0"
    p0_cache: str = "runs/rtwx_o0e0p0"
    r5_run: str = "runs/rtwx_o0e0r5"
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = 47605
    n_sequences: int = BEL0_N
    n_transitions: int = BEL0_T
    epochs_unet: int = UNET_EPOCHS
    smoke: bool = False


def _lock(cfg: RTWXO0MEMB0Config) -> RTWXO0MEMB0Config:
    if cfg.smoke:
        return replace(cfg, n_sequences=4, n_transitions=16, epochs_unet=4)
    return cfg


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0MEMB0 must not write there")


def select_winner_mem(cells: list[dict[str, Any]], b0: dict[str, Any]) -> dict[str, Any] | None:
    h0 = int(b0["H_star"])
    cands = [c for c in cells if c["kind"] != "none" and int(c["H_star"]) > h0]
    if not cands:
        return None

    def _key(c: dict[str, Any]) -> tuple:
        p90 = c["h64_p90"]
        med = c["h64_median"]
        return (
            -int(c["H_star"]),
            p90 if np.isfinite(p90) else 1.0e9,
            med if np.isfinite(med) else 1.0e9,
            int(c.get("n_applied", 0)),
            COMPLEXITY_RANK.get(c["kind"], 9),
        )

    return min(cands, key=_key)


def run_sequence_memory(
    clouds: list[np.ndarray],
    gt: dict[str, Any],
    seq_i: int,
    *,
    kind: Kind,
    n_const: np.ndarray,
    lut: Any,
    d_o: float,
) -> dict[str, Any]:
    state = init_anchor_state(clouds[0], n_const, lut)
    s0 = copy_state(state)
    history: dict[int, Keyframe] = {
        0: Keyframe(t=0, cloud=np.asarray(clouds[0]).copy(), state=copy_state(state)),
    }
    voxel_size = float(ICP_CFG.voxel_frac * d_o)
    surface = PersistentSurfaceMemory(voxel_size=voxel_size)
    if kind == "surface":
        surface.update_from_cloud(clouds[0])

    steps: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, float]] = {}
    gains: list[float] = []
    ages_sel: list[int] = []
    d_m_hist: list[float] = []

    for t in range(1, len(clouds)):
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], d_o, ICP_CFG)
        if res.valid:
            state = propagate_state(state, res.R, res.t)
        state_chain = copy_state(state)
        history[t] = Keyframe(t=t, cloud=np.asarray(clouds[t]).copy(), state=copy_state(state_chain))
        if kind == "surface":
            cloud_m = transform_cloud(clouds[t], invert_hom(state_chain.t_accum))
            surface.update_from_cloud(cloud_m)
            d_m_hist.append(surface.contamination_median())

        attempted = applied = False
        source_k = None
        q_star = None
        age = None
        d_m = surface.contamination_median() if kind == "surface" else None
        state_before = copy_state(state_chain)
        if kind != "none" and t in CORRECTION_T:
            attempted = True
            if kind == "retrieval":
                pick = retrieve_reference(t, clouds[t], history, d_o, icp_cfg=ICP_CFG)
                if pick is not None:
                    state = apply_retrieval(pick, history)
                    applied = True
                    source_k = pick.k
                    q_star = pick.q
                    age = pick.age
                    ages_sel.append(pick.age)
            elif kind == "surface":
                new, meta = register_surface_to_cloud(surface, clouds[t], s0, d_o, icp_cfg=ICP_CFG)
                if new is not None:
                    state = new
                    applied = True
                    q_star = None
        else:
            state = state_chain

        e_n = float(e_axis_deg(state.n, gt["n"][seq_i, t]))
        e_p = float(np.linalg.norm(state.p - gt["p"][seq_i, t]))
        e_n_b = float(e_axis_deg(state_before.n, gt["n"][seq_i, t]))
        g = (e_n_b - e_n) if attempted else None
        if g is not None:
            gains.append(g)
        steps.append({
            "t": int(t),
            "correction_attempted": attempted,
            "correction_applied": applied,
            "source_k": source_k,
            "selected_age": age,
            "q_star": q_star,
            "D_M": d_m,
            "evaluator": {
                "axis_error_deg": e_n,
                "position_error_m": e_p,
                "reset_gain_axis_deg": g,
            },
        })
        if t in CHECKPOINT_SET:
            checkpoints[str(t)] = {
                "chain_axis_error_deg": e_n,
                "chain_position_error_m": e_p,
            }

    g_all = np.asarray(gains, dtype=np.float64)
    age_hist = {str(a): float(np.mean(np.asarray(ages_sel) == a)) if ages_sel else float("nan") for a in RETRIEVAL_AGES}
    return {
        "sequence_id": int(seq_i),
        "kind": kind,
        "checkpoints": checkpoints,
        "steps": steps,
        "reset_gain": {
            "median_Gn": float(np.median(g_all)) if g_all.size else float("nan"),
            "P_Gn_gt0": float(np.mean(g_all > 0)) if g_all.size else float("nan"),
            "n": int(g_all.size),
        },
        "selected_age_hist": age_hist,
        "D_M_final": float(d_m_hist[-1]) if d_m_hist else float("nan"),
        "D_M_start": float(d_m_hist[0]) if d_m_hist else float("nan"),
    }


def _cell_row(per_seq: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    horizon, pass_at_h = _chain_horizon_table(per_seq)
    h_star = _largest_h_star(pass_at_h)
    h64 = horizon.get("64", {})
    axis64 = h64.get("axis", {})
    pos64 = h64.get("position", {})
    all_g = [
        st["evaluator"]["reset_gain_axis_deg"]
        for sq in per_seq for st in sq["steps"]
        if st["evaluator"]["reset_gain_axis_deg"] is not None
    ]
    g = np.asarray(all_g, dtype=np.float64)
    applied = sum(1 for sq in per_seq for st in sq["steps"] if st["correction_applied"])
    attempted = sum(1 for sq in per_seq for st in sq["steps"] if st["correction_attempted"])
    ages = [st["selected_age"] for sq in per_seq for st in sq["steps"] if st.get("selected_age") is not None]
    age_hist = {str(a): float(np.mean(np.asarray(ages) == a)) if ages else float("nan") for a in RETRIEVAL_AGES}
    d_final = np.asarray([sq["D_M_final"] for sq in per_seq], dtype=np.float64)
    return {
        "kind": kind,
        "H_star": int(h_star),
        "h64_median": float(axis64.get("median_e_axis_deg", float("nan"))),
        "h64_p90": float(axis64.get("p90_e_axis_deg", float("nan"))),
        "h64_pos_median_m": float(pos64.get("median_ep_m", float("nan"))),
        "pass_at_h": {str(k): v for k, v in pass_at_h.items()},
        "median_reset_gain": float(np.nanmedian(g)) if g.size else float("nan"),
        "improve_rate": float(np.mean(g > 0)) if g.size else float("nan"),
        "n_attempted": attempted,
        "n_applied": applied,
        "selected_age_hist": age_hist,
        "D_M_final_median": float(np.nanmedian(d_final)) if d_final.size else float("nan"),
        "horizon": horizon,
    }


def run_rtwx_o0memb0(output: str | Path, config: RTWXO0MEMB0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0MEMB0Config(output=str(output)))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)

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
        print("[rtwx-o0memb0] smoke numpy collect", flush=True)
        obs, gt = collect_sequences_numpy(sub)
    else:
        obs, gt = load_bel0_formal_cache(_resolve_apr(cfg.bel0_cache))
        cfg = replace(
            cfg,
            n_sequences=min(cfg.n_sequences, int(obs["rgb_h"].shape[0])),
            n_transitions=min(cfg.n_transitions, int(obs["rgb_h"].shape[1]) - 1),
        )

    d_o, lut, _b2, n_const, _repo = _load_assets(cfg)
    s0 = _train_s0(cfg)
    rng = np.random.default_rng(cfg.seed)

    table: list[dict[str, Any]] = []
    for kind in KINDS:
        print(f"[rtwx-o0memb0] eval {kind}", flush=True)
        per_seq: list[dict[str, Any]] = []
        for si in range(cfg.n_sequences):
            clouds = _build_clouds_seq(obs, gt, s0, si, rng=rng, use_gt_mask=False)
            per_seq.append(run_sequence_memory(
                clouds, gt, si, kind=kind, n_const=n_const, lut=lut, d_o=d_o,
            ))
        row = _cell_row(per_seq, kind)
        table.append({k: v for k, v in row.items() if k != "horizon"})
        _write_json(root / f"per_sequence_{kind}.json", per_seq)

    b0 = next(r for r in table if r["kind"] == "none")
    winner = select_winner_mem(table, b0)
    summary = {
        "pattern": "memory_breadth_selection_complete",
        "scientific_result": False,
        "purpose": "architecture_selection",
        "winner": ({"kind": winner["kind"]} if winner else None),
        "winner_id": (winner["kind"] if winner else None),
        "H_star_B0": int(b0["H_star"]),
        "table": table,
        "rel_icp_config_hash": REL_ICP_CONFIG_HASH,
        "bel0_formal_seed": BEL0_FORMAL_SEED,
        "correction_times": sorted(CORRECTION_T),
        "retrieval_ages": list(RETRIEVAL_AGES),
        "next_frontier": (
            [] if winner else ["stop_world_perception_local_dfs", "reopen_wam_multirate"]
        ),
    }
    _write_json(root / "header.json", {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "scientific_result": False,
        "purpose": "architecture_selection",
    })
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-o0memb0] winner={summary.get('winner_id')}", flush=True)
    return summary

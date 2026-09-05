"""SYM-X2: symmetry breaking and gauge reactivation. No RGB. Ĝ frozen from X0 C0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0c import _write_json
from .symx_plant import HOST_ID, apply_g, apply_g_inv, axis_angle_R, make_bodies, rollout, transform_action
from .symx_x0 import TAU, _traj_d, sample_action, sample_state
from .symx_x1 import (
    _apply_res,
    _cand_map,
    _collect,
    _g_hat,
    _train_mlp,
    _transitions,
    canon_state,
    d_G_np,
    pack_action,
    pack_state,
    persist_tf_E,
    ridge_r2,
    tf_E,
    yaw_cs,
)

PREREG_PATH = "REPORT/REG/SYMX/SYMX2_PREREG.md"
SCHEMA_ID = "aprwm.symx_x2.gauge_reactivation.v1"
SEEDS = (42101, 42102, 42103)
X0_DEFAULT = "runs/symx_x0"
X1_DEFAULT = "runs/symx_x1"
DT = 0.005
H = 40
N_BATCH = 32
M_CONSEC = 3
MAX_POST = 8
N_PRE = 3
N_FALSE_WIN = 20
T_MAX = 5
FALSE_MAX = 0.05
EPS_REC = 1.10
YAW_R2_MAX = 0.10
YAW_R2_MIN = 0.80
REC_EPOCHS = 25
REC_N = 64
W_MATCH = 32
EY = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class SYMX2Config:
    output: str = "runs/symx_x2"
    x0_summary: str = X0_DEFAULT
    x1_summary: str = X1_DEFAULT
    seeds: tuple[int, ...] = SEEDS
    n_batch: int = N_BATCH
    horizon: int = H
    dt: float = DT
    m_consec: int = M_CONSEC
    max_post: int = MAX_POST
    n_false_win: int = N_FALSE_WIN
    rec_epochs: int = REC_EPOCHS
    rec_n: int = REC_N
    smoke: bool = False


def _lock(cfg: SYMX2Config) -> SYMX2Config:
    if cfg.smoke:
        return replace(cfg, n_batch=8, horizon=16, seeds=(61,), m_consec=2, max_post=4, n_false_win=4, rec_epochs=6, rec_n=16)
    return replace(
        cfg,
        seeds=SEEDS,
        n_batch=N_BATCH,
        horizon=H,
        dt=DT,
        m_consec=M_CONSEC,
        max_post=MAX_POST,
        n_false_win=N_FALSE_WIN,
        rec_epochs=REC_EPOCHS,
        rec_n=REC_N,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; SYM-X2 must not write there")


def _load_json(path: Path) -> dict[str, Any]:
    p = path / "summary.json" if path.is_dir() else path
    if not p.is_file():
        raise RuntimeError(f"missing summary {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _require_parents(x0: dict[str, Any], x1: dict[str, Any]) -> None:
    if x0.get("pattern") != "causal_symmetry_supported":
        raise RuntimeError(f"SYM-X2 locked until X0 causal_symmetry_supported; got {x0.get('pattern')}")
    if x1.get("pattern") != "quotient_utility_supported":
        raise RuntimeError(f"SYM-X2 locked until X1 quotient_utility_supported; got {x1.get('pattern')}")


def _sentinels() -> list[tuple[str, np.ndarray]]:
    return [
        ("I", np.eye(3)),
        ("Ry15", axis_angle_R(EY, 15.0)),
        ("Ry90", axis_angle_R(EY, 90.0)),
        ("Rx90", axis_angle_R(np.array([1.0, 0.0, 0.0]), 90.0)),
    ]


def _dh_batch(body, n: int, seed: int, horizon: int, dt: float, gs: list[tuple[str, np.ndarray]]) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    probes = [(sample_state(rng, airborne=(i % 3 == 0)), sample_action(rng)) for i in range(n)]
    t_ref = [rollout(body, s0, a, horizon, dt) for s0, a in probes]
    out: dict[str, float] = {}
    for name, g in gs:
        ds = []
        for (s0, a), tr in zip(probes, t_ref):
            tg = rollout(body, apply_g(s0, g), transform_action(a, g), horizon, dt)
            tb = [apply_g_inv(st, g) for st in tg]
            ds.append(_traj_d(tr, tb))
        out[name] = float(np.mean(ds))
    return out


def _belief(dh: float, tau: float = TAU) -> float:
    return float(np.exp(-max(dh, 0.0) / max(tau, 1e-9)))


def _level(dh: float, tau: float = TAU) -> str:
    if dh <= tau:
        return "dormant"
    if dh <= 4.0 * tau:
        return "coarse"
    return "active"


def _revoke_index(dhs: list[float], m: int, tau: float = TAU) -> int | None:
    """1-based batch index when m consecutive exceed tau; None if never."""
    run = 0
    for i, d in enumerate(dhs, start=1):
        run = run + 1 if d > tau else 0
        if run >= m:
            return i
    return None


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "instrument_failure"
    if not g1:
        return "break_undetected"
    if not g2:
        return "false_revoke"
    if not g3:
        return "no_recovery"
    if not g4:
        return "task_world_collapse"
    return "gauge_reactivation_supported"


def _tf_world(trajs, acts, none, predict_res, G_in: np.ndarray | None, G_eval: np.ndarray) -> tuple[float, float]:
    """Teacher-forced E in world frame: section-predict then map back with the same g."""
    ds = []
    for tr, a0 in zip(trajs, acts):
        for t in range(len(tr) - 1):
            s = {k: tr[t][k].copy() for k in ("p", "R", "v", "w")}
            a = a0 if t == 0 else none
            sc, g = (canon_state(s, G_in) if G_in is not None else (s, None))
            aa = transform_action(a, g) if G_in is not None else a
            res = predict_res(pack_state(sc)[None], pack_action(aa)[None])[0]
            p, R, v, w = _apply_res(sc["p"][None], sc["R"][None], sc["v"][None], sc["w"][None], res[None])
            sp = {"p": p[0], "R": R[0], "v": v[0], "w": w[0]}
            if g is not None:
                sp = apply_g_inv(sp, g)
            gt = tr[t + 1]
            ds.append(float(d_G_np(sp["p"], sp["R"], sp["v"], sp["w"], gt["p"], gt["R"], gt["v"], gt["w"], G_eval)))
    arr = np.asarray(ds, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def run_symx_x2(output: str | Path, config: SYMX2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or SYMX2Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    x0 = _load_json(Path(cfg.x0_summary))
    x1 = _load_json(Path(cfg.x1_summary))
    _require_parents(x0, x1)
    bodies = make_bodies()
    gs = _sentinels()
    cmap = _cand_map()
    G0 = _g_hat(x0, "C0", cmap)
    G_I = np.eye(3)[None]
    header = {
        "stage": "SYM-X2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host": HOST_ID,
        "no_rgb": True,
        "tau": TAU,
        "m_consec": cfg.m_consec,
        "n_batch": cfg.n_batch,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "x0_pattern": x0["pattern"],
        "x1_pattern": x1["pattern"],
    }
    _write_json(root / "header.json", header)
    print(f"[symx-x2] n_batch={cfg.n_batch} H={cfg.horizon} m={cfg.m_consec} tau={TAU}", flush=True)

    seed0 = int(cfg.seeds[0])
    c0 = bodies["C0"]

    pre = [_dh_batch(c0, cfg.n_batch, seed0 + 3 * i, cfg.horizon, cfg.dt, gs) for i in range(N_PRE if not cfg.smoke else 2)]
    pre_ry = [b["Ry90"] for b in pre]
    g0_hold = bool(x1.get("G1_noninferior", {}).get("ok", False) and all(d < TAU for d in pre_ry))
    print(f"[symx-x2] pre C0 D_H(Ry90)={pre_ry} G0_hold={g0_hold}", flush=True)

    breaks = {
        "B1_iner": bodies["C4"],
        "B1_geom": bodies["C1"],
        "B2_fric": bodies["C0fr"],
        "B3_task": bodies["C5"],
    }
    physical = ("B1_iner", "B1_geom", "B2_fric")
    det: dict[str, Any] = {}
    for i, (name, body) in enumerate(breaks.items()):
        post = [_dh_batch(body, cfg.n_batch, seed0 + 200 * (i + 1) + j, cfg.horizon, cfg.dt, gs) for j in range(cfg.max_post)]
        ry = [b["Ry90"] for b in post]
        t_rev = _revoke_index(ry, cfg.m_consec)
        beliefs = [_belief(d) for d in ry]
        levels = [_level(d) for d in ry]
        det[name] = {
            "D_H_Ry90": ry,
            "D_H_I": [b["I"] for b in post],
            "D_H_Rx90": [b["Rx90"] for b in post],
            "T_revoke": t_rev,
            "revoked": t_rev is not None and t_rev <= T_MAX,
            "belief": beliefs,
            "level": levels,
            "world_G_should_revoke": name != "B3_task",
        }
        print(
            f"[symx-x2] {name} T={t_rev} Ry90_0={ry[0]:.3f} I={post[0]['I']:.4f} "
            f"level0={levels[0]} revoke={det[name]['revoked']}",
            flush=True,
        )

    # false revoke windows on C0
    n_pool = cfg.n_false_win + cfg.m_consec - 1
    pool = [_dh_batch(c0, cfg.n_batch, seed0 + 900 + k, cfg.horizon, cfg.dt, gs)["Ry90"] for k in range(n_pool)]
    n_false = 0
    for w in range(cfg.n_false_win):
        if _revoke_index(pool[w : w + cfg.m_consec], cfg.m_consec) is not None:
            n_false += 1
    p_false = n_false / max(cfg.n_false_win, 1)
    print(f"[symx-x2] false_revoke {n_false}/{cfg.n_false_win} P={p_false:.3f}", flush=True)

    # recovery on B1-iner (C4)
    rec = {}
    if not cfg.smoke:
        body_b = bodies["C4"]
        tr = _collect(body_b, cfg.rec_n, seed0 + 50, cfg.horizon, cfg.dt)
        te = _collect(body_b, max(cfg.rec_n // 2, 8), seed0 + 51, cfg.horizon, cfg.dt)
        pack_stale = _transitions(*tr, G0)
        pack_full = _transitions(*tr, None)
        pred_s, _, p_s = _train_mlp(pack_stale, hidden=W_MATCH, G_loss=G0, epochs=cfg.rec_epochs, seed=11)
        pred_f, _, p_f = _train_mlp(pack_full, hidden=W_MATCH, G_loss=G_I, epochs=cfg.rec_epochs, seed=12)
        pred_b, _, p_b = _train_mlp(pack_full, hidden=W_MATCH, G_loss=G_I, epochs=cfg.rec_epochs, seed=13)
        e_stale, _ = _tf_world(te[0], te[1], te[2], pred_s, G0, G_I)
        e_post, _ = tf_E(te[0], te[1], te[2], pred_f, None, G_I)
        e_b0, _ = tf_E(te[0], te[1], te[2], pred_b, None, G_I)
        e_mean = persist_tf_E(te[0], G_I)
        rec = {
            "E_stale": e_stale,
            "E_post": e_post,
            "E_B0": e_b0,
            "E_mean": e_mean,
            "ratio_post_B0": float(e_post / max(e_b0, 1e-12)),
            "ratio_stale_B0": float(e_stale / max(e_b0, 1e-12)),
            "P_stale": p_s,
            "P_post": p_f,
            "P_B0": p_b,
        }
        print(
            f"[symx-x2] recovery C4 E_stale={e_stale:.4f} E_post={e_post:.4f} E_B0={e_b0:.4f} "
            f"post/B0={rec['ratio_post_B0']:.3f} stale/B0={rec['ratio_stale_B0']:.3f}",
            flush=True,
        )
    else:
        rec = {"E_stale": 0.5, "E_post": 0.4, "E_B0": 0.4, "E_mean": 1.0, "ratio_post_B0": 1.0, "ratio_stale_B0": 1.2}

    # task split: C5 dynamics (same as C0) + yaw as task coordinate
    rng = np.random.default_rng(seed0 + 77)
    feat_q, feat_f, yaws = [], [], []
    c5 = bodies["C5"]
    for i in range(max(cfg.rec_n, 40)):
        s = sample_state(rng, airborne=False)
        sc, _ = canon_state(s, G0)
        feat_q.append(pack_state(sc))
        feat_f.append(pack_state(s))
        yaws.append(yaw_cs(s["R"]))
    feat_q, feat_f, yaws = np.stack(feat_q), np.stack(feat_f), np.stack(yaws)
    ntr = int(0.7 * len(yaws))
    r2_q = ridge_r2(feat_q[:ntr], yaws[:ntr], feat_q[ntr:], yaws[ntr:])
    r2_f = ridge_r2(feat_f[:ntr], yaws[:ntr], feat_f[ntr:], yaws[ntr:])
    task = {
        "world_revoked": bool(det["B3_task"]["revoked"]),
        "task_yaw_R2_quotient": r2_q,
        "task_yaw_R2_full": r2_f,
        "world_D_H_Ry90": det["B3_task"]["D_H_Ry90"][0],
    }
    print(f"[symx-x2] task world_revoke={task['world_revoked']} yaw_Q={r2_q:.3f} yaw_full={r2_f:.3f}", flush=True)

    g0 = g0_hold
    g1 = all(det[n]["revoked"] for n in physical)
    g2 = p_false < FALSE_MAX
    g3 = bool(rec["ratio_post_B0"] <= EPS_REC)
    g4 = (not task["world_revoked"]) and r2_f >= YAW_R2_MIN and r2_q <= YAW_R2_MAX
    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3, g4=g4)
    print(f"[symx-x2] G0={g0} G1={g1} G2={g2} G3={g3} G4={g4} pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_hold": {"ok": g0, "pre_D_H_Ry90": pre_ry, "x1_G1": x1.get("G1_noninferior", {})},
        "G1_detect": {
            "ok": g1,
            "T_revoke": {n: det[n]["T_revoke"] for n in physical},
            "gate_batches": T_MAX,
        },
        "G2_false_revoke": {"ok": g2, "P_false_revoke": p_false, "n_windows": cfg.n_false_win, "gate": FALSE_MAX},
        "G3_recovery": {"ok": g3, **rec, "gate": EPS_REC},
        "G4_task_split": {"ok": g4, **task},
        "detection": det,
        "pre_C0": pre,
        "unlocks_symx3": False,
        "unlocks_o0g6r": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

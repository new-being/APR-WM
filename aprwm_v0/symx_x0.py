"""SYM-X0: causal symmetry discovery on oracle rigid-body rollouts. No RGB in D_H."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .symx_plant import (
    HOST_ID,
    apply_g,
    apply_g_inv,
    axis_angle_R,
    d_state,
    make_bodies,
    rollout,
    transform_action,
)
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/SYMX/SYMX0_PREREG.md"
SCHEMA_ID = "aprwm.symx_x0.causal_symmetry.v1"
SEEDS = (40101, 40102, 40103)
CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5")
THETAS = tuple(range(0, 360, 15))
AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
DT = 0.005
H = 40
N_PROBE = 96
TAU = 0.04
DH_I_MAX = 0.01
EXCITE_MIN = 0.05
P_EXCITE_MIN = 0.50
FQ_MAX = 0.01
RECALL_MIN = 0.90
GSTAR_DEG = 8.0


@dataclass(frozen=True)
class SYMX0Config:
    output: str = "runs/symx_x0"
    seeds: tuple[int, ...] = SEEDS
    n_probe: int = N_PROBE
    horizon: int = H
    dt: float = DT
    tau: float = TAU
    smoke: bool = False


def _lock(cfg: SYMX0Config) -> SYMX0Config:
    if cfg.smoke:
        return replace(cfg, n_probe=12, horizon=16, seeds=(41,))
    return replace(cfg, seeds=SEEDS, n_probe=N_PROBE, horizon=H, dt=DT, tau=TAU)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; SYM-X0 must not write there")


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    R = np.asarray(R, dtype=np.float64)
    ang = _geodesic_deg(np.eye(3), R)
    if ang < 1e-8:
        return np.array([0.0, 1.0, 0.0]), 0.0
    if ang > 179.0:
        vals, vecs = np.linalg.eig(R)
        i = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, i])
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            return np.array([0.0, 1.0, 0.0]), ang
        return axis / n, ang
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(ax))
    if n < 1e-8:
        return np.array([0.0, 1.0, 0.0]), ang
    return ax / n, ang


def _near_axis(axis: np.ndarray, target: np.ndarray, deg: float = 12.0) -> bool:
    c = float(np.clip(np.abs(np.dot(axis, target)), 0.0, 1.0))
    return float(np.degrees(np.arccos(c))) <= deg


def in_Gstar(R: np.ndarray, cond: str) -> bool:
    if cond == "C3":
        return True
    if _geodesic_deg(R, np.eye(3)) <= GSTAR_DEG:
        return True
    axis, ang = _axis_angle(R)
    rx180 = abs(ang - 180.0) <= GSTAR_DEG and _near_axis(axis, AXES["x"])
    ry180 = abs(ang - 180.0) <= GSTAR_DEG and _near_axis(axis, AXES["y"])
    rz180 = abs(ang - 180.0) <= GSTAR_DEG and _near_axis(axis, AXES["z"])
    ry_any = _near_axis(axis, AXES["y"])
    if cond in ("C0", "C5"):
        return bool(ry_any or rx180 or rz180)
    if cond == "C4":
        return bool(ry180 or rx180 or rz180)
    if cond == "C2":
        return bool(rx180 or ry180 or rz180)
    if cond == "C1":
        return False
    return False


def _candidates() -> list[tuple[str, np.ndarray]]:
    out = [("I", np.eye(3))]
    for ax_name, axis in AXES.items():
        for th in THETAS:
            if th == 0:
                continue
            out.append((f"R{ax_name}{th}", axis_angle_R(axis, th)))
    return out


def sample_state(rng: np.random.Generator, *, airborne: bool) -> dict[str, np.ndarray]:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    z0 = 0.14 if airborne else 0.07
    p = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), z0 + rng.uniform(0.0, 0.08)])
    v = rng.normal(size=3) * (0.15 if airborne else 0.02)
    w = rng.normal(size=3) * 1.2
    return {"p": p, "R": R, "v": v, "w": w}


def sample_action(rng: np.random.Generator) -> dict[str, Any]:
    k = int(rng.integers(0, 6))
    if k == 0:
        return {"kind": "none"}
    if k == 1:
        j = rng.normal(size=3)
        j[2] = 0.0
        j = 0.08 * j / (np.linalg.norm(j[:2]) + 1e-9)
        return {"kind": "cm_world", "j_W": j}
    if k == 2:
        return {"kind": "body_torque", "tau_B": rng.normal(size=3) * 0.012, "x_O": None, "j_B": None}
    x_O = rng.normal(size=3)
    x_O = 0.03 * x_O / (np.linalg.norm(x_O) + 1e-9)
    j_B = rng.normal(size=3) * 0.06
    return {"kind": "body_point", "x_O": x_O, "j_B": j_B, "tau_B": None}


def _traj_d(t1: list, t2: list) -> float:
    return float(np.mean([d_state(a, b) for a, b in zip(t1, t2)]))


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Higher score → predict G*. Use -D_H."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return 1.0 if y.all() else 0.0
    pos, neg = score[y], score[~y]
    # P(pos > neg) + 0.5 P(eq)
    gt = np.subtract.outer(pos, neg)
    return float(np.mean((gt > 0) + 0.5 * (gt == 0)))


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "instrument_failure"
    if not g3:
        return "shape_symmetry_confound"
    if not g4:
        return "appearance_confound"
    if g1 and g2:
        return "causal_symmetry_supported"
    return "symmetry_discovery_insufficient"


def run_symx_x0(output: str | Path, config: SYMX0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or SYMX0Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    bodies = make_bodies()
    cands = _candidates()
    header = {
        "stage": "SYM-X0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host": HOST_ID,
        "no_rgb_in_d": True,
        "n_candidates": len(cands),
        "seeds": list(cfg.seeds),
        "tau": cfg.tau,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)
    print(f"[symx-x0] K={len(cands)} n_probe={cfg.n_probe} H={cfg.horizon}", flush=True)

    per_cond: dict[str, Any] = {}
    acc_neg = 0
    n_neg = 0
    rec_pos = 0
    n_pos = 0
    c0_ry, c5_ry = set(), set()
    c4_bad = 0
    c4_yaw = 0
    g0_i = True
    excite_ok = True

    for cond in CONDITIONS:
        body = bodies[cond]
        probes: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
        for seed in cfg.seeds:
            rng = np.random.default_rng(int(seed) + 17 * (int(cond[-1]) if cond[-1].isdigit() else 0))
            for i in range(cfg.n_probe):
                probes.append((sample_state(rng, airborne=(i % 3 == 0)), sample_action(rng)))
        t_ref = [rollout(body, s0, a, cfg.horizon, cfg.dt) for s0, a in probes]
        n_exc = int(sum(d_state(tr[-1], s0) > EXCITE_MIN for (s0, _), tr in zip(probes, t_ref)))
        p_exc = n_exc / max(len(probes), 1)
        if p_exc < P_EXCITE_MIN:
            excite_ok = False
        rows = []
        for name, g in cands:
            ds = []
            for (s0, a), tr in zip(probes, t_ref):
                t_g = rollout(body, apply_g(s0, g), transform_action(a, g), cfg.horizon, cfg.dt)
                t_back = [apply_g_inv(st, g) for st in t_g]
                ds.append(_traj_d(tr, t_back))
            dh = float(np.mean(ds))
            acc = bool(dh <= cfg.tau)
            gst = in_Gstar(g, cond)
            rows.append({"g": name, "D_H": dh, "accept": acc, "in_Gstar": gst})
            if gst:
                rec_pos += int(acc)
                n_pos += 1
            else:
                acc_neg += int(acc)
                n_neg += 1
            if cond == "C4" and name.startswith("Ry") and name[2:].isdigit() and int(name[2:]) not in {0, 180}:
                c4_yaw += 1
                c4_bad += int(acc)
            if cond == "C0" and name.startswith("Ry") and acc:
                c0_ry.add(name)
            if cond == "C5" and name.startswith("Ry") and acc:
                c5_ry.add(name)
        dh_map = {r["g"]: r["D_H"] for r in rows}
        y = np.array([r["in_Gstar"] for r in rows], dtype=bool)
        sc = -np.array([r["D_H"] for r in rows])
        p_acc_star = float(np.mean([r["accept"] for r in rows if r["in_Gstar"]])) if any(r["in_Gstar"] for r in rows) else float("nan")
        n_false = [r for r in rows if (not r["in_Gstar"]) and r["accept"]]
        p_fq = float(len(n_false) / max(sum(not r["in_Gstar"] for r in rows), 1))
        med_I = float(dh_map["I"])
        if med_I >= DH_I_MAX:
            g0_i = False
        per_cond[cond] = {
            "median_D_H_I": med_I,
            "P_excite": p_exc,
            "P_accept_Gstar": p_acc_star,
            "P_false_quotient": p_fq,
            "AUROC": _auroc(sc, y),
            "D_H_Ry90": dh_map.get("Ry90", float("nan")),
            "D_H_Rx90": dh_map.get("Rx90", float("nan")),
            "D_H_Rz90": dh_map.get("Rz90", float("nan")),
            "n_accept": int(sum(r["accept"] for r in rows)),
            "n_Gstar": int(y.sum()),
            "accepted": [r["g"] for r in rows if r["accept"]],
        }
        print(
            f"[symx-x0] {cond} D_H(I)={med_I:.4f} AUROC={per_cond[cond]['AUROC']:.3f} "
            f"fq={p_fq:.4f} rec={p_acc_star:.3f} nG={int(y.sum())} nacc={int(sum(r['accept'] for r in rows))} "
            f"Ry90={dh_map.get('Ry90', float('nan')):.4f} Rx90={dh_map.get('Rx90', float('nan')):.4f} "
            f"Rx180={dh_map.get('Rx180', float('nan')):.4f}",
            flush=True,
        )

    p_fq = acc_neg / max(n_neg, 1)
    recall = rec_pos / max(n_pos, 1)
    g0 = bool(g0_i and excite_ok)
    g1 = bool(p_fq < FQ_MAX)
    g2 = bool(recall >= RECALL_MIN)
    g3 = bool(c4_bad == 0)
    g4 = bool(c0_ry == c5_ry)
    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3, g4=g4)
    print(f"[symx-x0] G0={g0} G1_fq={p_fq:.4f} G2_rec={recall:.3f} G3={g3} G4={g4} pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_instrument": {"ok": g0, "D_H_I_ok": g0_i, "excite_ok": excite_ok},
        "G1_false_quotient": {"ok": g1, "P_false_quotient": p_fq, "n_neg": n_neg, "gate": FQ_MAX},
        "G2_recall": {"ok": g2, "recall": recall, "n_pos": n_pos, "gate": RECALL_MIN},
        "G3_C4": {"ok": g3, "n_bad_yaw_accept": c4_bad, "n_yaw_tested": c4_yaw},
        "G4_C5": {"ok": g4, "C0_Ry_accept": sorted(c0_ry), "C5_Ry_accept": sorted(c5_ry)},
        "per_condition": per_cond,
        "unlocks_symx1_prereg": pattern == "causal_symmetry_supported",
        "unlocks_symx2": False,
        "unlocks_symx3": False,
        "unlocks_o0g6r": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

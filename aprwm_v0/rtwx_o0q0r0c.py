"""RTWX-O0Q0R0C: native demo action-trace + excitation qualification. No IK. No yaw science."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose
from .rtwx_o0q0 import (
    D_H_pair,
    G0_MAX,
    H,
    REGIMES,
    TAU,
    _apply_yaw,
    _hold_action,
    _physx,
    _read_yq,
    _restore,
    _robot_q,
    _set_inertia_ratio,
)
from .rtwx_o0q0r0 import (
    GRIP_CLOSED,
    P0_V,
    P1_V0,
    PROBE_YAW,
    _aniso_drag,
    _cup_rigid_audit,
    _hdf5_for_seed,
    _pack_snap,
    _play_one,
    _p0_ic,
)
from .rtwx_o0q0r0b import (
    D_GRASP,
    D_NEAR,
    DELTA_T,
    K_PHYS,
    N_P3,
    ROOT_DIST,
    _hold_phys,
)
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0C_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0r0c.native_event_trace.v1"
SEEDS = (0, 1, 2)
N_P2_MIN = 3
N_P3_MIN = 2
G_FRAC = 0.80
H_P0 = 6  # frozen R0B ballistic contract; do not resweep
H_P1 = H
I_RATIO_P0 = 4.0  # instrument control only; not cup physics
P0_W_C = np.array([2.0, 0.3, 1.5])  # off-principal-axis
PATTERNS = (
    "identity_regression_failure",
    "native_demo_action_replay_failure",
    "precontact_event_repro_failure",
    "freeflight_excitation_failure",
    "contact_excitation_failure",
    "counterfactual_instrument_qualified",
)


@dataclass(frozen=True)
class RTWXO0Q0R0CConfig:
    output: str = "runs/rtwx_o0q0r0c"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0R0CConfig) -> RTWXO0Q0R0CConfig:
    if cfg.smoke:
        return replace(cfg, seeds=(64,))
    return replace(cfg, seeds=SEEDS)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0R0C must not write there")


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool, g5: bool) -> str:
    if not g0:
        return "identity_regression_failure"
    if not g1:
        return "native_demo_action_replay_failure"
    if not g2:
        return "precontact_event_repro_failure"
    if not g3:
        return "freeflight_excitation_failure"
    if not (g4 and g5):
        return "contact_excitation_failure"
    return "counterfactual_instrument_qualified"


def _t_p2(evs: list[dict[str, Any]]) -> int | None:
    for i, s in enumerate(evs):
        if float(s["dist"]) <= D_NEAR:
            return i
    return None


def _t_p3(evs: list[dict[str, Any]]) -> int | None:
    streak = 0
    start: int | None = None
    for i, s in enumerate(evs):
        if float(s["grip"]) <= GRIP_CLOSED and float(s["dist"]) < D_GRASP:
            if streak == 0:
                start = i
            streak += 1
            if streak >= N_P3:
                return start
        else:
            streak = 0
            start = None
    return None


def _pick_t0(snaps: list[dict[str, Any]], t_p2: int | None) -> int | None:
    if t_p2 is None or t_p2 < 1:
        return None
    t0: int | None = None
    for i in range(int(t_p2)):
        if float(snaps[i]["dist"]) > ROOT_DIST:
            t0 = i
    return t0


def _zero_cnt(env: Any) -> None:
    if hasattr(env, "take_action_cnt"):
        env.take_action_cnt = 0


def _demo_A(repo: Path, seed: int, dim: int) -> tuple[Path | None, list[dict[str, Any]], str]:
    path = _hdf5_for_seed(repo, seed)
    if path is None:
        return None, [], "missing_hdf5"
    from .task_x0 import _load_hdf5_qpos

    pack = _load_hdf5_qpos(path)
    if pack is None:
        return path, [], "load_fail"
    qa = np.asarray(pack["qa"], dtype=np.float64)
    q = np.asarray(pack["q"], dtype=np.float64)
    src = "qa"
    if qa.ndim != 2 or qa.shape[1] != int(dim):
        qa = q
        src = "q"
    if qa.ndim != 2 or qa.shape[1] != int(dim):
        return path, [], "dim_mismatch"
    A = [{"kind": "qpos", "u": qa[t].copy(), "action_type": "qpos"} for t in range(int(qa.shape[0]))]
    return path, A, src


def _roll_p0(env: Any, snap: dict[str, Any], n: int, *, yaw: float | None, ratio: float | None) -> tuple[list, list]:
    _restore(env, snap)
    _zero_cnt(env)
    c = _physx(env.cup)
    I0 = None
    if ratio is not None and c is not None and hasattr(c, "inertia"):
        I0 = np.asarray(c.inertia, dtype=np.float64).copy()
        _set_inertia_ratio(env.cup, float(ratio))
    try:
        if yaw is not None and abs(float(yaw)) > 1e-9:
            _apply_yaw(env, float(yaw))
        q = np.asarray(snap["q"], dtype=np.float64)
        ys, ev = [], []
        for _ in range(int(n)):
            _hold_phys(env, q, K_PHYS)
            ys.append(_read_yq(env))
            ev.append(_pack_snap(env))
        return ys, ev
    finally:
        if I0 is not None and c is not None:
            c.inertia = I0


def _roll_A(
    env: Any,
    root: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    yaw: float | None,
    probe: str | None,
) -> tuple[list, list]:
    _restore(env, root)
    _zero_cnt(env)
    try:
        if yaw is not None and abs(float(yaw)) > 1e-9:
            _apply_yaw(env, float(yaw))
        if probe == "friction":
            c = _physx(env.cup)
            if c is not None:
                c.linear_velocity = P1_V0
        ys, ev = [], []
        for a in actions:
            if probe in ("friction", "contact_fric"):
                _aniso_drag(env)
            _play_one(env, a)
            ys.append(_read_yq(env))
            ev.append(_pack_snap(env))
        return ys, ev
    finally:
        _set_force_zero(env)


def _set_force_zero(env: Any) -> None:
    c = _physx(env.cup)
    if c is None:
        return
    z = np.zeros(3)
    if hasattr(c, "set_force_and_torque"):
        c.set_force_and_torque(z, z)
    elif hasattr(c, "add_force_torque"):
        c.add_force_torque(z, z)


def _p0_pack(env: Any, z_tab: float, quat0: np.ndarray) -> dict[str, Any] | None:
    if not _p0_ic(env, z_tab, quat0):
        return None
    c = _physx(env.cup)
    if c is not None:
        c.linear_velocity = P0_V
        c.angular_velocity = P0_W_C
    return _pack_snap(env)


def _forward_demo(env: Any, A: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _zero_cnt(env)
    snaps = [_pack_snap(env)]
    for a in A:
        _play_one(env, a)
        snaps.append(_pack_snap(env))
    return snaps


def _frac_ok(xs: list[float], *, need: int) -> bool:
    vals = [float(x) for x in xs if np.isfinite(x)]
    if len(vals) < int(need):
        return False
    return (sum(v > TAU for v in vals) / len(vals)) >= G_FRAC


def _collect(cfg: RTWXO0Q0R0CConfig, seed: int, args: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 20000
    try:
        audit = _cup_rigid_audit(env.cup)
        p_tab, quat0 = _cup_pose(env)
        z_tab = float(p_tab[2])
        table = _pack_snap(env)
        p0 = _p0_pack(env, z_tab, quat0)
        _restore(env, table)
        q0, _ = _robot_q(env)
        path, A, channel = _demo_A(repo, seed, int(q0.size))
        snaps: list[dict[str, Any]] = []
        if A:
            snaps = _forward_demo(env, A)
        t2 = _t_p2(snaps) if snaps else None
        t3 = _t_p3(snaps) if snaps else None
        t0 = _pick_t0(snaps, t2)
        branch = snaps[t0] if t0 is not None else None
        suffix = A[t0:] if t0 is not None else []
        min_d = min((float(s["dist"]) for s in snaps), default=None)
        print(
            f"[rtwx-o0q0r0c] seed={seed} channel={channel} |A|={len(A)} min_d={min_d} tP2={t2} tP3={t3} t0={t0}",
            flush=True,
        )
        return {
            "table": table,
            "p0": p0,
            "branch": branch,
            "suffix": suffix,
            "A_n": len(A),
            "channel": channel,
            "hdf5": str(path) if path is not None else None,
            "t_p2": t2,
            "t_p3": t3,
            "t0": t0,
            "min_dist": min_d,
            "z_tab": z_tab,
            "audit": audit,
            "H_P0": H_P0,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _numpy_run(_cfg: RTWXO0Q0R0CConfig) -> dict[str, Any]:
    return {
        "H_P0": H_P0,
        "I_ratio_p0": I_RATIO_P0,
        "ident_p0": [0.0],
        "p0_contact": [False],
        "ident_p1": [0.0],
        "ident_A": [0.0],
        "n_demo": {"P2": 1, "P3": 1},
        "n_repro": {"P2": 1, "P3": 1},
        "dt_evt": {"P2": [0], "P3": [0]},
        "excite": {r: [0.2] for r in REGIMES},
        "min_dist": [0.03],
        "t0": [2],
        "A_n": [40],
        "channel": ["qa"],
        "audit": {"note": "numpy smoke", "I_x_eq_I_z": True},
    }


def _measure(cfg: RTWXO0Q0R0CConfig, seed: int, args: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    env = setup(repo, TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.step_lim = 20000
    out: dict[str, Any] = {
        "ident_p0": [],
        "p0_contact": [],
        "ident_p1": [],
        "ident_A": [],
        "excite": {r: [] for r in REGIMES},
        "n_repro": {"P2": 0, "P3": 0},
        "dt_evt": {"P2": [], "P3": []},
        "min_dist": blob.get("min_dist"),
        "H_P0": H_P0,
    }
    try:
        if blob.get("p0") is not None:
            y1, e1 = _roll_p0(env, blob["p0"], H_P0, yaw=None, ratio=None)
            y2, e2 = _roll_p0(env, blob["p0"], H_P0, yaw=None, ratio=None)
            out["ident_p0"].append(D_H_pair(y1, y2))
            hit = any(float(s["J"]) > 1e-5 or float(s["z"]) <= float(blob["z_tab"]) + 0.01 for s in e1 + e2)
            out["p0_contact"].append(bool(hit))
            yp, _ = _roll_p0(env, blob["p0"], H_P0, yaw=None, ratio=I_RATIO_P0)
            ya, _ = _roll_p0(env, blob["p0"], H_P0, yaw=PROBE_YAW, ratio=I_RATIO_P0)
            out["excite"]["P0"].append(D_H_pair(yp, ya))
            print(
                f"[rtwx-o0q0r0c] seed={seed} P0 H={H_P0} D_H(I)={out['ident_p0'][-1]:.4f} contact={hit} D_H(Ry90,I=4)={out['excite']['P0'][-1]:.4f}",
                flush=True,
            )
        table = blob.get("table")
        if table is not None:
            _restore(env, table)
            hold = [{"kind": "qpos", "u": _hold_action(env).copy(), "action_type": "qpos"} for _ in range(H_P1)]
            y1, _ = _roll_A(env, table, hold, yaw=None, probe=None)
            y2, _ = _roll_A(env, table, hold, yaw=None, probe=None)
            dh = D_H_pair(y1, y2)
            out["ident_p1"].append(dh)
            yp, _ = _roll_A(env, table, hold, yaw=None, probe="friction")
            ya, _ = _roll_A(env, table, hold, yaw=PROBE_YAW, probe="friction")
            out["excite"]["P1"].append(D_H_pair(yp, ya))
            print(f"[rtwx-o0q0r0c] seed={seed} P1 D_H(I)={dh:.4f} D_H(Ry90,fric)={out['excite']['P1'][-1]:.4f}", flush=True)
        branch, suffix = blob.get("branch"), blob.get("suffix") or []
        if branch is not None and suffix:
            y1, e1 = _roll_A(env, branch, suffix, yaw=None, probe=None)
            y2, e2 = _roll_A(env, branch, suffix, yaw=None, probe=None)
            out["ident_A"].append(D_H_pair(y1, y2))
            md = min(float(s["dist"]) for s in e1) if e1 else None
            if md is not None:
                out["min_dist_suffix"] = md
            t2a, t2b = _t_p2(e1), _t_p2(e2)
            t3a, t3b = _t_p3(e1), _t_p3(e2)
            if t2a is not None and t2b is not None:
                out["dt_evt"]["P2"].append(abs(t2a - t2b))
                out["n_repro"]["P2"] = 1
            if t3a is not None and t3b is not None:
                out["dt_evt"]["P3"].append(abs(t3a - t3b))
                out["n_repro"]["P3"] = 1
            yp, _ = _roll_A(env, branch, suffix, yaw=None, probe="contact_fric")
            ya, _ = _roll_A(env, branch, suffix, yaw=PROBE_YAW, probe="contact_fric")
            dhc = D_H_pair(yp, ya)
            if out["n_repro"]["P2"]:
                out["excite"]["P2"].append(dhc)
            if out["n_repro"]["P3"]:
                out["excite"]["P3"].append(dhc)
            print(
                f"[rtwx-o0q0r0c] seed={seed} suffix n={len(suffix)} D_H(I)={out['ident_A'][-1]:.4f} "
                f"min_d={md} tP2={t2a}/{t2b} tP3={t3a}/{t3b} D_H(Ry90,fric)={dhc:.4f}",
                flush=True,
            )
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    need_p2 = 1 if smoke else N_P2_MIN
    need_p3 = 1 if smoke else N_P3_MIN
    ident_p0 = [float(x) for x in raw["ident_p0"] if np.isfinite(x)]
    ident_p1 = [float(x) for x in raw["ident_p1"] if np.isfinite(x)]
    ident_A = [float(x) for x in raw["ident_A"] if np.isfinite(x)]
    p0_hit = any(raw.get("p0_contact") or [])
    g0_p0 = bool(ident_p0) and max(ident_p0) < G0_MAX and not p0_hit
    g0_p1 = bool(ident_p1) and max(ident_p1) < G0_MAX
    if smoke and ident_p0 and max(ident_p0) < G0_MAX:
        g0_p0 = True
    g0 = bool(g0_p0 and g0_p1)
    n_demo = {k: int(raw["n_demo"][k]) for k in ("P2", "P3")}
    n_repro = {k: int(raw["n_repro"][k]) for k in ("P2", "P3")}
    g1 = n_demo["P2"] >= need_p2 and n_demo["P3"] >= need_p3
    g2_ident = bool(ident_A) and max(ident_A) < G0_MAX
    g2_t = True
    for k, need in (("P2", need_p2), ("P3", need_p3)):
        dts = [int(x) for x in raw["dt_evt"][k]]
        if n_repro[k] < need:
            g2_t = False
        elif any(d > DELTA_T for d in dts):
            g2_t = False
    g2 = bool(g2_ident and g2_t)
    g3 = _frac_ok(list(raw["excite"]["P0"]), need=need_p2)
    g4 = _frac_ok(list(raw["excite"]["P1"]), need=need_p2)
    g5 = _frac_ok(list(raw["excite"]["P2"]), need=need_p2) and _frac_ok(list(raw["excite"]["P3"]), need=need_p3)
    return {
        "G0": {
            "ok": g0,
            "P0": {"ok": g0_p0, "D_H_I": ident_p0, "H_P0": raw.get("H_P0"), "contact": raw.get("p0_contact")},
            "P1": {"ok": g0_p1, "D_H_I": ident_p1},
            "gate": G0_MAX,
        },
        "G1": {"ok": g1, "n_demo": n_demo, "need_p2": need_p2, "need_p3": need_p3, "min_dist": raw.get("min_dist"), "channel": raw.get("channel")},
        "G2": {"ok": g2, "ident_A": ident_A, "n_repro": n_repro, "dt": raw["dt_evt"], "delta_t": DELTA_T},
        "G3": {"ok": g3, "I_ratio": I_RATIO_P0, "Ry90": list(raw["excite"]["P0"])},
        "G4": {"ok": g4, "Ry90": list(raw["excite"]["P1"])},
        "G5": {"ok": g5, "Ry90": {"P2": list(raw["excite"]["P2"]), "P3": list(raw["excite"]["P3"])}},
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5},
    }


def run_rtwx_o0q0r0c(output: str | Path, config: RTWXO0Q0R0CConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0R0CConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0R0C",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "no_ik": True,
        "no_restore_contact": True,
        "no_teleport_into_contact": True,
        "no_hdf5_object_pose": True,
        "no_yaw_science": True,
        "tau": TAU,
        "G0_max": G0_MAX,
        "delta_t": DELTA_T,
        "H_P0": H_P0,
        "I_ratio_p0": I_RATIO_P0,
        "P0_W": P0_W_C.tolist(),
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
        "demo_trace": "hdf5_action_qpos_as_take_action_target",
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0r0c] backend={cfg.backend} smoke={cfg.smoke}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _numpy_run(cfg)
    else:
        import os
        import sys

        from .task_x0 import _install_mplib_fallback, _load_task_args

        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_raster_shader()
        _install_mplib_fallback(repo)
        args = _load_task_args(repo, TASK, "demo_clean")
        args["task_name"] = TASK
        args["render_freq"] = 0
        chunks = []
        demos = []
        for s in cfg.seeds:
            blob = _collect(cfg, int(s), args)
            chunks.append(_measure(cfg, int(s), args, blob))
            chunks[-1]["audit"] = blob.get("audit")
            demos.append(
                {
                    "seed": int(s),
                    "channel": blob.get("channel"),
                    "A_n": blob.get("A_n"),
                    "t_p2": blob.get("t_p2"),
                    "t_p3": blob.get("t_p3"),
                    "t0": blob.get("t0"),
                    "min_dist": blob.get("min_dist"),
                    "hdf5": blob.get("hdf5"),
                }
            )
        raw = {
            "H_P0": H_P0,
            "I_ratio_p0": I_RATIO_P0,
            "ident_p0": [x for c in chunks for x in c["ident_p0"]],
            "p0_contact": [x for c in chunks for x in c["p0_contact"]],
            "ident_p1": [x for c in chunks for x in c["ident_p1"]],
            "ident_A": [x for c in chunks for x in c["ident_A"]],
            "n_demo": {
                "P2": sum(1 for d in demos if d.get("t_p2") is not None),
                "P3": sum(1 for d in demos if d.get("t_p3") is not None),
            },
            "n_repro": {k: sum(int(c["n_repro"][k]) for c in chunks) for k in ("P2", "P3")},
            "dt_evt": {k: [x for c in chunks for x in c["dt_evt"][k]] for k in ("P2", "P3")},
            "excite": {r: [x for c in chunks for x in c["excite"][r]] for r in REGIMES},
            "audit": chunks[0].get("audit") if chunks else {},
            "min_dist": [d.get("min_dist") for d in demos],
            "t0": [d.get("t0") for d in demos],
            "A_n": [d.get("A_n") for d in demos],
            "channel": [d.get("channel") for d in demos],
            "demos": demos,
        }

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(g0=g["g0"], g1=g["g1"], g2=g["g2"], g3=g["g3"], g4=g["g4"], g5=g["g5"])
    qual = pattern == "counterfactual_instrument_qualified"
    print(f"[rtwx-o0q0r0c] G0={g['g0']} G1={g['g1']} G2={g['g2']} G3={g['g3']} G4={g['g4']} G5={g['g5']} pattern={pattern}", flush=True)
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G1", "G2", "G3", "G4", "G5")},
        "demos": raw.get("demos"),
        "rigid_audit": raw.get("audit"),
        "unlocks_o0q0r1_prereg": bool(qual),
        "unlocks_o0q1_prereg": False,
        "unlocks_o1": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

"""RTWX-O0Q0R0E: acquire official CuRobo expert dense drive trace. No new planner. No yaw."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK
from .rtwx_o0q0 import D_H_pair, G0_MAX, TAU, _read_yq, _restore
from .rtwx_o0q0r0 import _pack_snap
from .rtwx_o0q0r0c import H_P0, I_RATIO_P0, P0_W_C, _t_p2, _t_p3
from .rtwx_o0q0r0d import EPS_EE, EPS_Q, _Tap, _apply_drive, _ee_p, _meas_q, _pick_t0
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0E_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0q0r0e.native_dense_trace.v1"
SEEDS = (0, 1, 2)
MIN_STEPS = 400
MIN_ARM = 1
PATTERNS = (
    "dense_trace_unavailable",
    "dense_trace_semantics_failure",
    "native_trace_replay_failure",
    "native_dense_trace_qualified",
)


@dataclass(frozen=True)
class RTWXO0Q0R0EConfig:
    output: str = "runs/rtwx_o0q0r0e"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    seed_attempts: int = 32
    smoke: bool = False


def _lock(cfg: RTWXO0Q0R0EConfig) -> RTWXO0Q0R0EConfig:
    if cfg.smoke:
        return replace(cfg, seeds=(66,))
    return replace(cfg, seeds=SEEDS)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0Q0R0E must not write there")


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool) -> str:
    if not g0:
        return "dense_trace_unavailable"
    if not g1:
        return "dense_trace_semantics_failure"
    if not (g2 and g3):
        return "native_trace_replay_failure"
    return "native_dense_trace_qualified"


def curobo_available() -> dict[str, Any]:
    try:
        import curobo  # noqa: F401

        return {"ok": True, "file": getattr(curobo, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def traj_pkl_dir(repo: Path) -> Path:
    return Path(repo) / "data" / "demo_clean" / TASK / "aloha_agilex" / "_traj_data"


def find_traj_pkl(repo: Path, ep: int) -> Path | None:
    p = traj_pkl_dir(repo) / f"episode{ep}.pkl"
    if p.is_file():
        return p
    hits = sorted(Path(repo).joinpath("data").rglob(f"**/place_empty_cup/**/_traj_data/episode{ep}.pkl"))
    return hits[0] if hits else None


def audit_joint_path(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    segs: list[int] = []
    statuses: list[str] = []
    for side in ("left_joint_path", "right_joint_path"):
        for item in data.get(side) or []:
            if not isinstance(item, dict) or "position" not in item:
                continue
            pos = np.asarray(item["position"])
            if pos.ndim >= 1:
                segs.append(int(pos.shape[0]))
            statuses.append(str(item.get("status", "")))
    total = int(sum(segs))
    med = float(np.median(segs)) if segs else 0.0
    dense = bool(segs) and med > 2.0 and total >= MIN_STEPS
    return {
        "path": str(path),
        "n_seg": len(segs),
        "n_steps": total,
        "median_seg": med,
        "dense": dense,
        "all_success": bool(statuses) and all(s == "Success" for s in statuses),
        "kind": "take_dense_action_position" if dense else ("sparse_or_missing" if segs else "missing_position"),
    }


def _capture(cfg: RTWXO0Q0R0EConfig, seed: int, args: dict[str, Any], *, traj: Path | None) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    env_args = dict(args)
    if traj is not None:
        data = pickle.loads(Path(traj).read_bytes())
        env_args["need_plan"] = False
        env_args["left_joint_path"] = data.get("left_joint_path", [])
        env_args["right_joint_path"] = data.get("right_joint_path", [])
    env = setup(Path(cfg.robotwin_repo), TASK, seed, cfg.seed_attempts, env_args)
    if traj is not None:
        env.set_path_lst(env_args)
    env.save_data = False
    env.save_freq = None
    env.step_lim = 20000
    tap = _Tap()
    tap.install(env)
    err = None
    ok = False
    try:
        try:
            env.play_once()
            ok = bool(getattr(env, "plan_success", True))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ok = False
        t2 = _t_p2(tap.snaps) if tap.snaps else None
        t3 = _t_p3(tap.snaps) if tap.snaps else None
        print(
            f"[rtwx-o0q0r0e] seed={seed} src={'pkl' if traj else 'curobo'} ok={ok} err={err} "
            f"steps={tap.counts['scene_step']} arm={tap.counts['set_arm_joints']} tP2={t2} tP3={t3}",
            flush=True,
        )
        return {
            "ok": ok,
            "error": err,
            "counts": dict(tap.counts),
            "A": tap.A,
            "q_meas": tap.q_meas,
            "ee": tap.ee,
            "snaps": tap.snaps,
            "t_p2": t2,
            "t_p3": t3,
            "t0": _pick_t0(tap.snaps, t2),
            "src": "pkl" if traj else "curobo",
        }
    finally:
        tap.uninstall(env)
        try:
            env.close()
        except Exception:
            pass


def _replay_trace(cfg: RTWXO0Q0R0EConfig, seed: int, args: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup

    A = blob.get("A") or []
    out: dict[str, Any] = {"e_q": None, "e_ee": None, "t_p2": None, "t_p3": None, "ident": None, "n": 0}
    if not A:
        return out
    env = setup(Path(cfg.robotwin_repo), TASK, seed, cfg.seed_attempts, args)
    env.check_success = lambda *a, **k: False
    env.save_data = False
    env.save_freq = None
    env.step_lim = 20000
    try:
        q_rep, ee_rep, snaps = [], [], []
        for a in A:
            _apply_drive(env, a)
            env.scene.step()
            q_rep.append(_meas_q(env))
            ee_rep.append(_ee_p(env))
            snaps.append(_pack_snap(env))
        n = min(len(q_rep), len(blob.get("q_meas") or []))
        if n:
            out["e_q"] = float(max(float(np.max(np.abs(q_rep[i] - blob["q_meas"][i]))) for i in range(n)))
            out["e_ee"] = float(max(float(np.linalg.norm(ee_rep[i] - blob["ee"][i])) for i in range(n)))
            out["n"] = n
        out["t_p2"] = _t_p2(snaps)
        out["t_p3"] = _t_p3(snaps)
        t0 = blob.get("t0")
        if t0 is not None and blob.get("snaps") and int(t0) < len(A):
            root = blob["snaps"][int(t0)]
            suf = A[int(t0) :]
            y1: list = []
            y2: list = []
            for k in range(2):
                _restore(env, root)
                ys = []
                for a in suf:
                    _apply_drive(env, a)
                    env.scene.step()
                    ys.append(_read_yq(env))
                if k == 0:
                    y1 = ys
                else:
                    y2 = ys
            if y1 and y2:
                out["ident"] = D_H_pair(y1, y2)
        print(
            f"[rtwx-o0q0r0e] seed={seed} replay n={out['n']} e_q={out['e_q']} e_ee={out['e_ee']} "
            f"tP2={out['t_p2']} tP3={out['t_p3']} D_H(I)={out['ident']}",
            flush=True,
        )
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


def _numpy_run(_cfg: RTWXO0Q0R0EConfig) -> dict[str, Any]:
    return {
        "curobo": {"ok": True},
        "path_b": [],
        "source": "curobo",
        "g0": True,
        "g1": True,
        "e_q": [0.0],
        "e_ee": [0.0],
        "n_p2": 1,
        "n_p3": 1,
        "ident": [0.0],
        "n_ok": 1,
        "n_arm": [10],
        "n_step": [800],
    }


def _agg(raw: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    need = 1 if smoke else 3
    g0 = bool(raw.get("g0"))
    g1 = bool(raw.get("g1")) and int(raw.get("n_ok", 0)) >= need
    eq = [float(x) for x in (raw.get("e_q") or []) if x is not None and np.isfinite(x)]
    ee = [float(x) for x in (raw.get("e_ee") or []) if x is not None and np.isfinite(x)]
    g2_robot = bool(eq) and max(eq) < EPS_Q and bool(ee) and max(ee) < EPS_EE
    g2_evt = int(raw.get("n_p2", 0)) >= need and int(raw.get("n_p3", 0)) >= need
    g2 = g2_robot and g2_evt
    ident = [float(x) for x in (raw.get("ident") or []) if x is not None and np.isfinite(x)]
    g3 = bool(ident) and max(ident) < G0_MAX
    return {
        "G0": {"ok": g0, "source": raw.get("source"), "curobo": raw.get("curobo"), "path_b": raw.get("path_b")},
        "G1": {"ok": g1, "n_ok": raw.get("n_ok"), "n_step": raw.get("n_step"), "n_arm": raw.get("n_arm")},
        "G2": {"ok": g2, "e_q": eq, "e_ee": ee, "n_p2": raw.get("n_p2"), "n_p3": raw.get("n_p3")},
        "G3": {"ok": g3, "D_H_I": ident, "gate": G0_MAX},
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3},
    }


def run_rtwx_o0q0r0e(output: str | Path, config: RTWXO0Q0R0EConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Q0R0EConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0Q0R0E",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_rgb": True,
        "no_new_planner": True,
        "no_hdf5_as_action": True,
        "no_yaw_science": True,
        "no_p0_p1_reopen": True,
        "tau": TAU,
        "H_P0_frozen": H_P0,
        "I_ratio_p0_frozen": I_RATIO_P0,
        "P0_W_frozen": P0_W_C.tolist(),
        "min_steps": MIN_STEPS,
        "seeds": list(cfg.seeds),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
        "does_not_change_o0_target": True,
    }
    _write_json(root / "header.json", header)
    print(f"[rtwx-o0q0r0e] backend={cfg.backend} smoke={cfg.smoke}", flush=True)

    if cfg.smoke or cfg.backend == "numpy":
        raw = _numpy_run(cfg)
        demos: list[dict[str, Any]] = []
    else:
        cu = curobo_available()
        repo = Path(cfg.robotwin_repo)
        pkls = [find_traj_pkl(repo, i) for i in range(len(cfg.seeds))]
        audits = [audit_joint_path(p) if p is not None else None for p in pkls]
        path_a = bool(cu.get("ok"))
        path_b = bool(pkls) and all(p is not None for p in pkls) and all(a is not None and a.get("dense") for a in audits)
        print(f"[rtwx-o0q0r0e] curobo={cu} path_a={path_a} path_b={path_b} pkls={[str(p) if p else None for p in pkls]}", flush=True)
        if not path_a and not path_b:
            raw = {
                "curobo": cu,
                "path_b": audits,
                "source": None,
                "g0": False,
                "g1": False,
                "e_q": [],
                "e_ee": [],
                "n_p2": 0,
                "n_p3": 0,
                "ident": [],
                "n_ok": 0,
                "n_arm": [],
                "n_step": [],
            }
            demos = [{"seed": int(s), "src": None, "ok": False, "reason": "no_curobo_and_no_traj_pkl"} for s in cfg.seeds]
        else:
            import os
            import sys

            from .task_x0 import _load_task_args

            os.chdir(repo)
            sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_raster_shader()
            args = _load_task_args(repo, TASK, "demo_clean")
            args["task_name"] = TASK
            args["render_freq"] = 0
            args["collect_data"] = False
            blobs, reps = [], []
            for i, s in enumerate(cfg.seeds):
                blob = _capture(cfg, int(s), args, traj=pkls[i] if path_b and not path_a else None)
                blobs.append(blob)
                reps.append(_replay_trace(cfg, int(s), args, blob))
                if blob.get("A"):
                    np.savez_compressed(root / f"A_star_seed{s}.npz", q=np.stack([a["q"] for a in blob["A"]], axis=0))
            n_arm = [int(b["counts"]["set_arm_joints"]) for b in blobs]
            n_step = [int(b["counts"]["scene_step"]) for b in blobs]
            n_ok = sum(1 for b in blobs if b.get("ok"))
            g0 = all(
                int(b["counts"]["take_dense_action"]) > 0
                and int(b["counts"]["set_arm_joints"]) >= MIN_ARM
                and int(b["counts"]["take_action"]) == 0
                for b in blobs
            )
            g1 = n_ok >= 3 and all(
                ns > MIN_STEPS and na >= MIN_ARM and b.get("t_p2") is not None for b, ns, na in zip(blobs, n_step, n_arm)
            )
            raw = {
                "curobo": cu,
                "path_b": audits,
                "source": "curobo" if path_a else "traj_pkl",
                "g0": g0,
                "g1": g1,
                "e_q": [r.get("e_q") for r in reps],
                "e_ee": [r.get("e_ee") for r in reps],
                "n_p2": sum(1 for r in reps if r.get("t_p2") is not None),
                "n_p3": sum(1 for r in reps if r.get("t_p3") is not None),
                "ident": [r.get("ident") for r in reps],
                "n_ok": n_ok,
                "n_arm": n_arm,
                "n_step": n_step,
            }
            demos = [
                {
                    "seed": int(s),
                    "src": blobs[i]["src"],
                    "ok": blobs[i]["ok"],
                    "error": blobs[i]["error"],
                    "counts": blobs[i]["counts"],
                    "t_p2": reps[i]["t_p2"],
                    "t_p3": reps[i]["t_p3"],
                    "e_q": reps[i]["e_q"],
                    "ident": reps[i]["ident"],
                }
                for i, s in enumerate(cfg.seeds)
            ]

    agg = _agg(raw, smoke=bool(cfg.smoke))
    g = agg["gates"]
    pattern = _pattern(g0=g["g0"], g1=g["g1"], g2=g["g2"], g3=g["g3"])
    qual = pattern == "native_dense_trace_qualified"
    print(f"[rtwx-o0q0r0e] G0={g['g0']} G1={g['g1']} G2={g['g2']} G3={g['g3']} pattern={pattern}", flush=True)
    summary = {
        "header": header,
        "pattern": pattern,
        **{k: agg[k] for k in ("G0", "G1", "G2", "G3")},
        "demos": demos,
        "unlocks_o0q0r1_prereg": bool(qual),
        "unlocks_o0q1_prereg": False,
        "unlocks_o1": False,
        "o0_target_remains_full_TR": True,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary

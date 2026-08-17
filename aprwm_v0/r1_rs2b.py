"""R1-RS2B: contact-mediated consequence transport (diagnosis only)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .r1_mj import stage_status
from .r1_rs1 import _sample_door_points
from .r1_rs1_backend import DoorModeABackend
from .r1_rs1a import R1RS1AConfig
from .r1_rs1a3 import _spearman
from .r1_rs1a4 import A0, FREQ_HZ, HORIZON_MACRO, MACRO_STEPS, QUERY_COUNT, _compute_episode
from .r1_rs1b import H32, _require_rs1a5_unlock
from .r1_rs1b1 import _terminal_rmse
from .r1_rs2 import REGIME_ALPHA, DoorContactC0Backend, R1RS2C0Config
from .r1_rs2_formal import FORMAL_SEEDS
from .train import seed_everything
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1/R1_RS2B_PREREG.md"
REGIMES = ("C0", "C1-L", "C1-H", "C2")
SCRIPT = "fast_pull"
SCALE = np.asarray([1.0, 2.0], dtype=np.float64)
SMOKE_JOBS = ((11101, "C0"), (11101, "C2"))
PROTOCOL_Q = 0.12
PROTOCOL_V = 0.0


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R1RS2BConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    regimes: tuple[str, ...] = REGIMES
    script: str = SCRIPT
    h32: int = H32
    query_count: int = QUERY_COUNT


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _require_rs2a(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("RS2B locked until RS2A has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") or not payload.get("scientific_result"):
        raise RuntimeError("RS2B locked: RS2A scientific matrix has not been run")
    return payload


def _drag_fn(alpha: float):
    def force(_q: float, velocity: float) -> float:
        return float(alpha) * abs(float(velocity)) * float(velocity)

    return force


def _mode_a_consequence(intake: R1RS1AConfig, *, seed: int, regime: str) -> dict[str, float]:
    alpha = float(REGIME_ALPHA[regime])
    if regime != "C2":
        episode = _compute_episode(
            intake,
            seed=seed,
            alpha=alpha,
            amp_scale=1.0,
            freq_hz=FREQ_HZ,
            a0=A0,
        )
        return {
            "C_mode_a": float(episode["C"]),
            "L_mode_a": float(episode["L_no_rev"]),
            "L_oracle_mode_a": float(episode["L_adeq"]),
        }
    backend = DoorModeABackend(
        regime="C2-latch",
        friction=intake.friction,
        damping=intake.damping,
        seed=seed,
        alpha=0.0,
    )
    try:
        generator = torch.Generator().manual_seed(
            seed + 41_000 + int(hashlib.md5(b"rs2-c2").hexdigest()[:8], 16) % 10_000
        )
        query = _sample_door_points(QUERY_COUNT, "intervention", generator)
        horizon = HORIZON_MACRO * MACRO_STEPS
        cs, ls, lo = [], [], []
        for row in query:
            out = backend.forecast_pair(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                horizon=horizon,
                margin=intake.joint_limit_margin,
            )
            cs.append(float(out["rmse_nominal"]) - float(out["rmse_oracle"]))
            ls.append(float(out["rmse_nominal"]))
            lo.append(float(out["rmse_oracle"]))
        return {
            "C_mode_a": float(np.mean(cs)) if cs else math.nan,
            "L_mode_a": float(np.mean(ls)) if ls else math.nan,
            "L_oracle_mode_a": float(np.mean(lo)) if lo else math.nan,
        }
    finally:
        backend.close()


def _queries(seed: int, regime: str, count: int) -> np.ndarray:
    alpha = float(REGIME_ALPHA[regime])
    if regime == "C2":
        mix = int(hashlib.md5(b"rs2-c2").hexdigest()[:8], 16) % 10_000
    else:
        mix = int(abs(alpha) * 10_000) % 10_000
    generator = torch.Generator().manual_seed(seed + 41_000 + mix)
    return _sample_door_points(count, "intervention", generator).detach().cpu().numpy()


def _record_actions(c0: R1RS2C0Config, script: str) -> np.ndarray:
    backend = DoorContactC0Backend(c0, seed=0, regime="C0")
    try:
        backend.prepare()
        result = backend.rollout(script, scale=1.0)
        return np.asarray(result["actions"], dtype=np.float64)
    finally:
        backend.close()


def _contact_consequence(
    c0: R1RS2C0Config,
    *,
    seed: int,
    regime: str,
    actions: np.ndarray,
    config: R1RS2BConfig,
    intake: R1RS1AConfig,
) -> dict[str, Any]:
    alpha = float(REGIME_ALPHA[regime])
    queries = _queries(seed, regime, config.query_count)
    protocol = np.asarray([[PROTOCOL_Q, PROTOCOL_V, 0.0]], dtype=np.float64)
    all_ics = np.concatenate([queries, protocol], axis=0)
    truth_backend = DoorContactC0Backend(c0, seed=int(seed), regime=regime)
    nom_backend = DoorContactC0Backend(c0, seed=int(seed), regime="C0")
    oracle = _drag_fn(alpha)
    query_rows: list[dict[str, Any]] = []
    try:
        truth_backend.prepare()
        nom_backend.prepare()
        n_steps = int(config.h32)
        tape = np.asarray(actions, dtype=np.float64)
        if len(tape) > n_steps:
            tape = tape[-n_steps:]
        for index, state in enumerate(all_ics):
            q0, v0 = float(state[0]), float(state[1])
            truth = truth_backend.replay_actions(
                tape, q0=q0, v0=v0, include_hidden=True, revision_fn=None, n_steps=n_steps
            )
            nom = nom_backend.replay_actions(
                tape, q0=q0, v0=v0, include_hidden=False, revision_fn=None, n_steps=n_steps
            )
            ora = nom_backend.replay_actions(
                tape, q0=q0, v0=v0, include_hidden=False, revision_fn=oracle, n_steps=n_steps
            )
            rmse_n = _terminal_rmse(truth, nom, SCALE)
            rmse_o = _terminal_rmse(truth, ora, SCALE)
            query_rows.append(
                {
                    "query_index": index,
                    "protocol": bool(index >= len(queries)),
                    "finite": bool(truth["finite"] and nom["finite"] and ora["finite"]),
                    "L": rmse_n,
                    "L_oracle": rmse_o,
                    "C": (
                        float(rmse_n - rmse_o)
                        if math.isfinite(rmse_n) and math.isfinite(rmse_o)
                        else math.nan
                    ),
                }
            )
    finally:
        truth_backend.close()
        nom_backend.close()

    scientific = [row for row in query_rows if not row["protocol"]]
    protocol_rows = [row for row in query_rows if row["protocol"]]

    def mean_key(rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else math.nan

    return {
        "C_contact": mean_key(scientific, "C"),
        "L_contact": mean_key(scientific, "L"),
        "L_oracle_contact": mean_key(scientific, "L_oracle"),
        "L_protocol": mean_key(protocol_rows, "L"),
        "C_protocol": mean_key(protocol_rows, "C"),
        "n_finite_queries": int(sum(row["finite"] for row in scientific)),
        "joint_limit_margin": intake.joint_limit_margin,
        "queries": query_rows,
    }


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(np.median(finite))


def _aggregate(rows: list[dict[str, Any]], c_tol: float) -> dict[str, Any]:
    c0 = [row for row in rows if row["regime"] == "C0"]
    c1 = [row for row in rows if row["regime"] in {"C1-L", "C1-H"}]
    c2 = [row for row in rows if row["regime"] == "C2"]
    h0 = {
        "n": len(c0),
        "n_false_mode_a": int(sum(bool(row["consequential_mode_a"]) for row in c0)),
        "n_false_contact": int(sum(bool(row["consequential_contact"]) for row in c0)),
        "pass": bool(
            c0
            and all(not row["consequential_mode_a"] for row in c0)
            and all(not row["consequential_contact"] for row in c0)
        ),
    }
    agree = 0
    for row in c1:
        if bool(row["consequential_mode_a"]) == bool(row["consequential_contact"]):
            agree += 1
    rho = _spearman(
        np.asarray([row["C_mode_a"] for row in c1], dtype=np.float64),
        np.asarray([row["C_contact"] for row in c1], dtype=np.float64),
    )
    h1 = {
        "n": len(c1),
        "agreement": float(agree / len(c1)) if c1 else math.nan,
        "spearman": rho,
        "pass_a": bool(c1 and agree / len(c1) >= 0.8),
        "pass_b": bool(math.isfinite(rho) and rho > 0.0),
    }
    l_c2 = _median([row["L_contact"] for row in c2])
    l_c0 = _median([row["L_contact"] for row in c0])
    h2 = {
        "n": len(c2),
        "all_mode_a_below_tol": bool(
            c2 and all(not row["consequential_mode_a"] for row in c2)
        ),
        "median_L_contact_c2": l_c2,
        "median_L_contact_c0": l_c0,
        "median_C_contact_c2": _median([row["C_contact"] for row in c2]),
        "C_tol": c_tol,
        "pass_a": bool(c2 and all(not row["consequential_mode_a"] for row in c2)),
        "pass_b": bool(math.isfinite(l_c2) and l_c2 >= c_tol),
        "pass_c": bool(math.isfinite(l_c2) and math.isfinite(l_c0) and l_c2 > l_c0),
    }
    h2["pass"] = bool(h2["pass_a"] and h2["pass_b"] and h2["pass_c"])
    go = bool(h0["pass"] and h2["pass"])
    return {
        "n_episodes": len(rows),
        "C_tol": c_tol,
        "h0_c0_sanity": h0,
        "h1_c1_delta_c": h1,
        "h2_c2_latch_harm": h2,
        "rs2b_go": go,
        "rewrites_rs2_go": False,
        "rewrites_rs2a_go": False,
        "rewrites_rs1c_go": False,
        "unlocks_c_tol_retune": False,
        "reopens_rs2a": False,
    }


def run_r1_rs2b(
    output: str | Path,
    *,
    rs2a_summary: str | Path,
    rs1a5_summary: str | Path,
    config: R1RS2BConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS2BConfig()
    _require_rs2a(Path(rs2a_summary))
    policy = _require_rs1a5_unlock(Path(rs1a5_summary))
    c_tol = float(policy["aggregate"]["gates"]["C_tol"])
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    intake = R1RS1AConfig()
    c0_cfg = R1RS2C0Config()
    actions = _record_actions(c0_cfg, cfg.script)
    jobs = list(SMOKE_JOBS) if smoke else [(seed, regime) for seed in cfg.seeds for regime in cfg.regimes]
    scientific = not smoke
    rows: list[dict[str, Any]] = []
    for index, (seed, regime) in enumerate(jobs, start=1):
        print(f"[RS2B {index}/{len(jobs)}] seed={seed} {regime}", flush=True)
        seed_everything(int(seed))
        mode_a = _mode_a_consequence(intake, seed=int(seed), regime=str(regime))
        contact = _contact_consequence(
            c0_cfg,
            seed=int(seed),
            regime=str(regime),
            actions=actions,
            config=cfg,
            intake=intake,
        )
        queries = contact.pop("queries")
        row = {
            "seed": int(seed),
            "regime": str(regime),
            "script": cfg.script,
            **mode_a,
            "C_contact": contact["C_contact"],
            "L_contact": contact["L_contact"],
            "L_oracle_contact": contact["L_oracle_contact"],
            "L_protocol": contact["L_protocol"],
            "C_protocol": contact["C_protocol"],
            "n_finite_queries": contact["n_finite_queries"],
            "C_tol": c_tol,
        }
        row["consequential_mode_a"] = bool(row["C_mode_a"] >= c_tol)
        row["consequential_contact"] = bool(row["C_contact"] >= c_tol)
        row["query_rows"] = queries
        rows.append(row)

    aggregate = (
        _aggregate(rows, c_tol)
        if scientific
        else {
            "rs2b_go": False,
            "n_episodes": len(rows),
            "note": "plumbing smoke only; H0–H2 not evaluated",
        }
    )
    status = stage_status()
    status["R1-RS2A"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS2B"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs2b_go")),
        "smoke_only": smoke,
        "diagnosis_only": True,
    }
    slim = [{k: v for k, v in row.items() if k != "query_rows"} for row in rows]
    summary = {
        "stage": "R1-RS2B",
        "scientific_result": scientific,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "frozen": {
            "C_tol": c_tol,
            "script": cfg.script,
            "detectability": "closed_in_rs2a",
            "decision_policy": "untouched",
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "episodes": _jsonable(slim),
        "aggregate": _jsonable(aggregate),
        "rs2b_go": bool(aggregate.get("rs2b_go")),
        "rs2_go_unchanged": True,
        "rs2a_go_unchanged": True,
        "rs1c_go_unchanged": True,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", _jsonable(summary))
    _write_json(root / "stage_status.json", _jsonable(status))
    csv_keys = (
        "seed",
        "regime",
        "C_mode_a",
        "C_contact",
        "L_mode_a",
        "L_contact",
        "L_protocol",
        "consequential_mode_a",
        "consequential_contact",
        "C_tol",
    )
    _write_csv(
        root / "episodes.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    return summary

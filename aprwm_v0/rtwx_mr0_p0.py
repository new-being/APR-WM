"""RTWX-MR0-P0: action-chunk capability qualification (instrument, not science)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .control.action_chunk_buffer import ActionChunkBuffer, ActionChunkContractError
from .control.action_chunk_spec import ActionChunkSpec
from .control.chunk_policy_adapter import ChunkAdapterError, ChunkPolicyAdapter
from .rtwx_mr0 import MR0_TASKS, paired_episode_seeds
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0MR0P0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_mr0_p0.action_chunk_qualification.v1"
SEED = 49601
MIN_HA = 8
ACTION_DIM = 14
SEMANTICS = "joint_position"
POLICY_DT_ENV_TICKS = 1
G3_N_EP = 32
G3_MIN_POOLED_SUCCESS = 0.40
G3_MIN_PER_TASK_SUCCESS = 0.20
G3_MAX_SAFETY = 0.20
CANDIDATE_RELATIVE = (
    "runs/rtwx_mr0_p0/frozen_chunk_policy.pt",
    "runs/frozen_chunk_policy.pt",
)


@dataclass(frozen=True)
class RTWXMR0P0Config:
    output: str = "runs/rtwx_mr0_p0"
    p0_summary: str = ""
    robotwin_repo: str = "/root/RoboTwin"
    seed: int = SEED
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-MR0-P0 must not write there")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_spec() -> ActionChunkSpec:
    return ActionChunkSpec(
        horizon=MIN_HA,
        action_dim=ACTION_DIM,
        semantics=SEMANTICS,
        policy_dt_env_ticks=POLICY_DT_ENV_TICKS,
        deterministic=True,
    )


def discover_chunk_checkpoint() -> dict[str, Any]:
    checked: list[str] = []
    for rel in CANDIDATE_RELATIVE:
        p = _resolve_apr(rel)
        checked.append(str(p))
        if p.is_file():
            return {
                "found": True,
                "path": str(p),
                "checkpoint_sha256": _file_sha256(p),
                "checked": checked,
            }
    return {"found": False, "path": None, "checkpoint_sha256": None, "checked": checked, "reason": "no_frozen_chunk_policy"}


def load_ac0_infer(path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]] | None:
    import torch

    from .policy.chunk_normalizer import ChunkNormalizer
    from .policy.explicit_chunk_mlp import ExplicitChunkPolicy
    from .policy.task_embedding import STATE_DIM

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("schema") != "aprwm.ac0.chunk_mlp.v1":
        return None
    model = ExplicitChunkPolicy(
        int(ckpt["state_dim"]),
        int(ckpt["process_dim"]),
        use_process=bool(ckpt["use_process"]),
        n_tasks=3,
        task_emb=16,
        action_dim=int(ckpt["action_dim"]),
        horizon=int(ckpt["horizon"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    pdim = int(ckpt["process_dim"])

    class _Infer:
        fills_by_repeat = False
        fills_by_rollout = False
        ac0_interface = True
        horizon = int(ckpt["horizon"])
        action_dim = int(ckpt["action_dim"])

        def __call__(self, planner_input: Any) -> np.ndarray:
            from .evaluation.ac0_rollout_eval import predict_chunk

            if isinstance(planner_input, dict) and "frame" in planner_input:
                return predict_chunk(
                    model,
                    norm,
                    planner_input["frame"],
                    planner_input["task"],
                    device,
                    bool(ckpt["use_process"]),
                )
            st = np.asarray(planner_input.get("state", np.zeros(STATE_DIM)), dtype=np.float64).reshape(1, STATE_DIM)
            pr = np.asarray(planner_input.get("process", np.zeros(pdim)), dtype=np.float64).reshape(1, pdim)
            tid = int(planner_input.get("task_id", 0))
            s = torch.from_numpy(norm.n_state(st).astype(np.float32)).to(device)
            p = torch.from_numpy(norm.n_proc(pr).astype(np.float32)).to(device)
            t = torch.tensor([tid], dtype=torch.long, device=device)
            with torch.no_grad():
                y = model(s, t, p)
            return np.asarray(norm.denorm_act(y.cpu().numpy())[0], dtype=np.float64)

    return _Infer(), ckpt


def g0_interface(disc: dict[str, Any], spec: ActionChunkSpec) -> dict[str, Any]:
    if not disc.get("found"):
        return {
            "pass": False,
            "H_a": 0,
            "d_a": 0,
            "reason": disc.get("reason", "no_frozen_chunk_policy"),
            "spec": spec.as_dict(),
            "checkpoint_sha256": None,
        }
    loaded = load_ac0_infer(Path(disc["path"]))
    if loaded is None:
        return {
            "pass": False,
            "H_a": 0,
            "d_a": 0,
            "reason": "policy_loader_not_wired",
            "path": disc["path"],
            "checkpoint_sha256": disc.get("checkpoint_sha256"),
            "spec": spec.as_dict(),
        }
    infer, ckpt = loaded
    dummy = infer({"state": np.zeros(int(ckpt["state_dim"])), "task_id": 0, "process": np.zeros(int(ckpt["process_dim"]))})
    ha, da = int(dummy.shape[0]), int(dummy.shape[1])
    ok = ha >= MIN_HA and da == spec.action_dim
    return {
        "pass": bool(ok),
        "H_a": ha,
        "d_a": da,
        "reason": None if ok else "shape_mismatch",
        "spec": spec.as_dict(),
        "checkpoint_sha256": disc.get("checkpoint_sha256"),
        "policy_config_hash": ckpt.get("config_hash"),
        "action_semantics": spec.semantics,
        "policy_dt": spec.policy_dt_env_ticks,
        "infer": infer if ok else None,
        "ckpt": ckpt if ok else None,
    }


def g1_genuine_chunk(adapter: ChunkPolicyAdapter, rng: np.random.Generator) -> dict[str, Any]:
    shapes = []
    n0 = adapter.n_forward
    for _ in range(4):
        x = {"robot": rng.normal(size=ACTION_DIM), "world": rng.normal(size=7), "goal": rng.normal(size=3)}
        chunk = adapter.infer_chunk(x)
        shapes.append(list(chunk.shape))
        if chunk.shape != (adapter.spec.horizon, adapter.spec.action_dim):
            return {"pass": False, "genuine_chunk": False, "reason": "bad_shape", "shapes": shapes}
    if adapter.n_forward != n0 + 4:
        return {"pass": False, "genuine_chunk": False, "reason": "forward_count_mismatch"}
    return {"pass": True, "genuine_chunk": True, "shapes": shapes, "n_forward": adapter.n_forward}


def g2_native_execution(adapter: ChunkPolicyAdapter, stride: int = 4, n_ticks: int = 8) -> dict[str, Any]:
    buf = ActionChunkBuffer(stride=stride)
    trace = []
    control_ticks = 0
    try:
        for t in range(n_ticks):
            if buf.needs_replan(t):
                chunk = adapter.infer_chunk({"t": t})
                buf.load(chunk, start_tick=t)
                planner_called = True
            else:
                planner_called = False
            act = buf.consume(t)
            control_ticks += 1
            trace.append(
                {
                    "tick": t,
                    "planner_called": planner_called,
                    "chunk_id": int(buf.n_replans - 1),
                    "chunk_index": t - int(buf.chunk_start),
                    "executed_action_l2": float(np.linalg.norm(act)),
                    "low_level_control_ticks": 1,
                }
            )
    except (ActionChunkContractError, ChunkAdapterError) as e:
        return {"pass": False, "reason": str(e), "trace": trace}
    expected_plans = 1 + (n_ticks - 1) // stride
    ok = buf.n_replans == expected_plans and control_ticks == n_ticks
    idxs = [row["chunk_index"] for row in trace]
    expected_idx = [t % stride for t in range(n_ticks)]
    ok = bool(ok and idxs == expected_idx)
    called = [row["tick"] for row in trace if row["planner_called"]]
    expected_called = list(range(0, n_ticks, stride))
    ok = bool(ok and called == expected_called)
    return {
        "pass": ok,
        "n_replans": buf.n_replans,
        "control_ticks": control_ticks,
        "trace": trace,
        "reason": None if ok else "execution_trace_mismatch",
    }


def g3_run(infer: Any, ckpt: dict[str, Any], cfg: RTWXMR0P0Config) -> dict[str, Any]:
    import os
    import sys

    import torch

    from .evaluation.ac0_rollout_eval import rollout_episode, wrap_topp_large_jump
    from .policy.chunk_normalizer import ChunkNormalizer
    from .policy.explicit_chunk_mlp import ExplicitChunkPolicy
    from .policy.task_embedding import TASK_NAMES
    from .task_x0 import _install_mplib_fallback, _load_task_args, _setup_env

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExplicitChunkPolicy(
        int(ckpt["state_dim"]),
        int(ckpt["process_dim"]),
        use_process=bool(ckpt["use_process"]),
        n_tasks=len(TASK_NAMES),
        task_emb=16,
        action_dim=int(ckpt["action_dim"]),
        horizon=int(ckpt["horizon"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    n_ep = 2 if cfg.smoke else G3_N_EP
    seeds = paired_episode_seeds(n_ep)
    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_mplib_fallback(repo)
    rows = []
    for task in MR0_TASKS:
        args = _load_task_args(repo, task, "demo_clean")
        args["need_plan"] = False
        args["render_freq"] = 0
        for seed in seeds:
            env, used, err = _setup_env(repo, task, int(seed), 8, args)
            if env is None:
                rows.append({"task": task, "success": False, "safety_violation": True, "seed": seed, "err": str(err)})
                continue
            try:
                wrap_topp_large_jump(env)
                lim = int(getattr(env, "step_lim", None) or {"place_empty_cup": 500, "put_object_cabinet": 700, "stamp_seal": 400}[task])
                rec = rollout_episode(
                    env,
                    task,
                    model=model,
                    normalizer=norm,
                    device=device,
                    use_process=bool(ckpt["use_process"]),
                    max_ticks=min(lim, 80 if cfg.smoke else lim),
                )
                rec["seed"] = int(used)
                rows.append(rec)
            finally:
                try:
                    env.close_env()
                except Exception:
                    pass
    pooled = float(np.mean([1.0 if r["success"] else 0.0 for r in rows])) if rows else 0.0
    safety = float(np.mean([1.0 if r["safety_violation"] else 0.0 for r in rows])) if rows else 1.0
    per = {}
    for task in MR0_TASKS:
        sub = [r for r in rows if r["task"] == task]
        per[task] = float(np.mean([1.0 if r["success"] else 0.0 for r in sub])) if sub else 0.0
    ok = pooled >= G3_MIN_POOLED_SUCCESS and min(per.values()) >= G3_MIN_PER_TASK_SUCCESS and safety <= G3_MAX_SAFETY
    return {
        "pass": bool(ok),
        "skipped": False,
        "pooled": pooled,
        "per_task": per,
        "safety": safety,
        "n_ep": n_ep,
        "min_pooled": G3_MIN_POOLED_SUCCESS,
        "min_per_task": G3_MIN_PER_TASK_SUCCESS,
        "max_safety": G3_MAX_SAFETY,
        "rows": rows,
    }


def g3_competence_gate() -> dict[str, Any]:
    return {
        "pass": False,
        "skipped": True,
        "reason": "no_qualified_policy_to_evaluate",
        "n_ep": G3_N_EP,
        "min_pooled": G3_MIN_POOLED_SUCCESS,
        "min_per_task": G3_MIN_PER_TASK_SUCCESS,
        "max_safety": G3_MAX_SAFETY,
        "tasks": list(MR0_TASKS),
        "seeds": paired_episode_seeds(G3_N_EP),
    }


def decide_p0_pattern(*, g0: dict[str, Any], g1: dict[str, Any] | None, g2: dict[str, Any] | None, g3: dict[str, Any]) -> str:
    if not g0.get("pass"):
        return "action_chunk_contract_failure"
    if g1 is None or not g1.get("pass"):
        return "action_chunk_contract_failure"
    if g2 is None or not g2.get("pass"):
        return "action_chunk_execution_failure"
    if not g3.get("pass"):
        return "action_chunk_policy_incompetent"
    return "action_chunk_qualified"


def run_rtwx_mr0_p0(output: str | Path, config: RTWXMR0P0Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXMR0P0Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    spec = frozen_spec()
    print("[rtwx-mr0-p0] discover checkpoint", flush=True)
    disc = discover_chunk_checkpoint()
    g0 = g0_interface(disc, spec)
    infer = g0.pop("infer", None)
    ckpt = g0.pop("ckpt", None)
    print(f"[rtwx-mr0-p0] G0 pass={g0['pass']} reason={g0.get('reason')}", flush=True)
    g1 = None
    g2 = None
    g3 = g3_competence_gate()
    if g0.get("pass") and infer is not None:
        g1 = g1_genuine_chunk(ChunkPolicyAdapter(infer, spec), np.random.default_rng(cfg.seed))
        g2 = g2_native_execution(ChunkPolicyAdapter(infer, spec), stride=4, n_ticks=8)
        if g1.get("pass") and g2.get("pass") and ckpt is not None:
            print("[rtwx-mr0-p0] G3 closed-loop competence", flush=True)
            g3 = g3_run(infer, ckpt, cfg)
    pattern = decide_p0_pattern(g0=g0, g1=g1, g2=g2, g3=g3)
    note = "P0 qualification on frozen AC0 chunk MLP." if g0.get("pass") else (
        "No frozen H_a>=8 chunk policy was loadable. AC0 provisioning is a separate cell. TASK-X1 locked."
    )
    summary = {
        "pattern": pattern,
        "scientific_result": False,
        "unlocks_mr0_a": pattern == "action_chunk_qualified",
        "g0": g0,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "discovery": disc,
        "tasks": list(MR0_TASKS),
        "note": note,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-mr0-p0] pattern={pattern}", flush=True)
    return summary

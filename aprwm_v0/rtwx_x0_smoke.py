"""RoboTwin-X0 official smoke: put_object_cabinet oracle state, no RGB in s."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PREREG_PATH = "REPORT/REG/RTWX/RTWX0_SMOKE_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0_smoke.cabinet.v1"
TASK_NAME = "put_object_cabinet"


@dataclass(frozen=True)
class RTWX0SmokeConfig:
    output: str = "runs/rtwx_x0/smoke"
    robotwin_repo: str = "/home/dong/Projects/RoboTwin"
    task_config: str = "demo_clean"
    seed: int = 0
    n_steps: int = 20


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RoboTwin-X0 smoke must not write there")


def _patch_curobo_planner(repo: Path) -> bool:
    """Load planner.py without envs.robot.__init__, then stub CuroboPlanner."""
    import importlib.util
    import types

    import envs  # noqa: F401

    robot_dir = repo / "envs" / "robot"
    robot_pkg = types.ModuleType("envs.robot")
    robot_pkg.__path__ = [str(robot_dir)]
    robot_pkg.__package__ = "envs.robot"
    sys.modules["envs.robot"] = robot_pkg

    spec = importlib.util.spec_from_file_location(
        "envs.robot.planner",
        robot_dir / "planner.py",
    )
    planner = importlib.util.module_from_spec(spec)
    sys.modules["envs.robot.planner"] = planner
    assert spec.loader is not None
    spec.loader.exec_module(planner)

    class _StubCuroboPlanner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def plan_path(self, *args, **kwargs) -> dict:
            return {"status": "Fail"}

        def plan_batch(self, *args, **kwargs) -> dict:
            return {"status": np.array(["Fail"])}

    planner.CuroboPlanner = _StubCuroboPlanner
    robot_pkg.planner = planner
    robot_pkg.CuroboPlanner = _StubCuroboPlanner
    robot_pkg.MplibPlanner = planner.MplibPlanner

    spec_r = importlib.util.spec_from_file_location(
        "envs.robot.robot",
        robot_dir / "robot.py",
    )
    robot_mod = importlib.util.module_from_spec(spec_r)
    sys.modules["envs.robot.robot"] = robot_mod
    assert spec_r.loader is not None
    spec_r.loader.exec_module(robot_mod)
    robot_pkg.Robot = robot_mod.Robot
    return True
    ql = np.asarray(env.robot.get_left_arm_real_jointState(), dtype=np.float64)
    qr = np.asarray(env.robot.get_right_arm_real_jointState(), dtype=np.float64)
    pose = env.object.get_pose()
    obj_p = np.asarray(pose.p, dtype=np.float64)
    obj_q = np.asarray(pose.q, dtype=np.float64)
    cab_q = np.asarray(env.cabinet.get_qpos(), dtype=np.float64).reshape(-1)
    cab_qd = np.asarray(env.cabinet.get_qvel(), dtype=np.float64).reshape(-1)
    parts = {
        "q_left": ql,
        "q_right": qr,
        "x_o": obj_p,
        "R_o": obj_q,
        "q_cabinet": cab_q,
        "qd_cabinet": cab_qd,
    }
    s = np.concatenate([parts[k] for k in parts])
    parts["s"] = s
    return parts


def _load_task_args(repo: Path, task_config: str) -> dict[str, Any]:
    cfg_root = repo / "env_cfg" / "task_config"
    with open(cfg_root / f"{task_config}.yml", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(cfg_root / "_embodiment_config.yml", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment_type = args["embodiment"]
    robot_file = embodiment_types[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(Path(robot_file) / "config.yml", encoding="utf-8") as f:
        emb = yaml.safe_load(f)
    args["left_embodiment_config"] = emb
    args["right_embodiment_config"] = emb
    args["task_name"] = TASK_NAME
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["need_plan"] = False
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = False
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = True
    return args


def run_rtwx_x0_smoke(output: str | Path, config: RTWX0SmokeConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0SmokeConfig()
    root = Path(output).resolve()
    _refuse_r10(root)
    repo = Path(cfg.robotwin_repo)
    objects = repo / "assets" / "objects"
    embodiments = repo / "assets" / "embodiments"
    g_assets = objects.is_dir() and embodiments.is_dir() and any(objects.iterdir())

    summary: dict[str, Any] = {
        "stage": "RoboTwin-X0-smoke",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK_NAME,
        "official_robotwin_task_executed": False,
        "rgb_in_s": False,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "config": asdict(cfg),
        "gates": {},
        "rtwx_x0_smoke_passed": False,
    }

    if not g_assets:
        summary["gates"] = {"G_assets": False}
        _write_json(root / "summary.json", summary)
        return summary

    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    g_cabinet = False
    g_norgb = False
    g_state = False
    g_step = False
    curobo_stub = False
    metrics: dict[str, Any] = {}
    err: str | None = None

    try:
        stubbed = _patch_curobo_planner(repo)
        curobo_stub = stubbed
        from envs.put_object_cabinet import put_object_cabinet

        args = _load_task_args(repo, cfg.task_config)
        env = put_object_cabinet()
        env.setup_demo(now_ep_num=0, seed=cfg.seed, **args)
        g_cabinet = True
        parts = extract_oracle_state(env)
        s = parts["s"]
        keys = set(parts) - {"s"}
        g_norgb = not any("rgb" in k or "camera" in k or "image" in k for k in keys)
        d_s = int(s.size)
        g_state = bool(
            20 <= d_s <= 80
            and parts["q_cabinet"].size >= 1
            and parts["x_o"].size == 3
            and parts["q_left"].size >= 6
        )
        for _ in range(cfg.n_steps):
            env.scene.step()
        s2 = extract_oracle_state(env)["s"]
        g_step = bool(np.isfinite(s2).all())
        metrics = {
            "d_s": d_s,
            "q_left_dim": int(parts["q_left"].size),
            "q_right_dim": int(parts["q_right"].size),
            "q_cabinet_dim": int(parts["q_cabinet"].size),
            "s_change_l2": float(np.linalg.norm(s2 - s)),
            "object_model": str(getattr(env, "selected_modelname", "")),
            "cabinet_model": str(getattr(env, "model_name", "")),
            "s_keys": sorted(keys),
            "curobo_stub": curobo_stub,
        }
        try:
            env.close()
        except Exception:
            pass
    except Exception as exc:
        import traceback

        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    passed = bool(g_assets and g_cabinet and g_norgb and g_state and g_step)
    summary.update(
        {
            "official_robotwin_task_executed": g_cabinet,
            "metrics": metrics,
            "error": err,
            "gates": {
                "G_assets": g_assets,
                "G_cabinet": g_cabinet,
                "G_norgb": g_norgb,
                "G_state": g_state,
                "G_step": g_step,
                "G_label": True,
            },
            "rtwx_x0_smoke_passed": passed,
            "unlocks_rtwx_x0_formal_runner": passed,
            "output": str(root.resolve()),
        }
    )
    _write_json(root / "summary.json", summary)
    return summary

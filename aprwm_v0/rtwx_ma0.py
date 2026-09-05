"""RTWX-MA0: official RoboTwin→ACT→XPolicyLab positive-control probe."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation.ac3_native import demo_dir, raw_hdf5_candidates
from .evaluation.ac3_provenance import alignment_errors, load_converted_q14, load_raw_joint_vector
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0MA0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ma0.mature_stack.v1"
TASK = "place_empty_cup"
TASK_CONFIG = "demo_clean"
ENV_CFG_TYPE = "aloha_agilex"
ACTION_TYPE = "joint"
CUP_N = 16
CUP_G0 = 0.20
ACT_NUM_EPOCHS = 6000
ACT_CHUNK_SIZE = 50
ACT_SEED = 0
XPOLICYLAB_PIN = "c37109c500be67d0dea6b36bf7337bbd26e763cd"
XPOLICYLAB_ZIP_COMMIT = "a9ccf8dbc34ad047534a14b7ee0503bddf0e54b5"


@dataclass(frozen=True)
class RTWXMA0Config:
    output: str = "runs/rtwx_ma0"
    robotwin_repo: str = "/root/RoboTwin"
    conda_env: str = "Robotwin"
    smoke: bool = False
    p0_only: bool = False
    skip_train: bool = False
    gpu_id: str = "0"


def xpolicylab_root(repo: Path) -> Path:
    return Path(repo) / "XPolicyLab"


def xpolicylab_ready(repo: Path) -> bool:
    root = xpolicylab_root(repo)
    return (root / "setup_policy_server.py").is_file() and (root / "policy" / "ACT" / "train.sh").is_file()


def git_rev(path: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def _h5_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tobytes") and getattr(value, "shape", ()) == ():
        item = value.item() if hasattr(value, "item") else value
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _hdf5_meta(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as f:
        freq = None
        if "additional_info" in f and "frequency" in f["additional_info"]:
            freq = int(np.asarray(f["additional_info/frequency"]).reshape(-1)[0])
        cams = list(f["vision"].keys()) if "vision" in f else []
        return {
            "source_format": str(f.attrs.get("source_format", "")),
            "source_path": str(f.attrs.get("source_path", "")),
            "data_format_version": _h5_str(f["data_format_version"][()] if "data_format_version" in f else ""),
            "frequency": freq,
            "cameras": cams,
        }


def inferred_alignment_name(inferred: str) -> str:
    if inferred == "next_index":
        return "next_state_as_action"
    if inferred == "same_index":
        return "same_index"
    return inferred


def audit_converted_task(repo: Path, task: str = TASK) -> dict[str, Any]:
    ddir = demo_dir(repo, task)
    data_dir = ddir / "data"
    eps = sorted(data_dir.glob("episode_*.hdf5"))
    meta_path = ddir / "conversion_meta.json"
    conversion_meta = None
    if meta_path.is_file():
        conversion_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = []
    t_raws: list[int | None] = []
    t_convs: list[int] = []
    for path in eps:
        i = int(path.stem.split("_")[-1])
        st, act = load_converted_q14(path)
        hdf = _hdf5_meta(path)
        q4 = alignment_errors(st, act) if st is not None and act is not None else {}
        raw = None
        raw_path = None
        for cand in raw_hdf5_candidates(repo, task, i):
            raw = load_raw_joint_vector(cand)
            if raw is not None:
                raw_path = str(cand)
                break
        t_c = int(len(act) if act is not None else 0)
        t_r = int(len(raw)) if raw is not None else None
        t_convs.append(t_c)
        t_raws.append(t_r)
        da = int(act.shape[1]) if act is not None and act.ndim == 2 else None
        rows.append(
            {
                "episode": i,
                "path": str(path),
                "T_converted": t_c,
                "T_raw": t_r,
                "d_a": da,
                "raw_hdf5": raw_path,
                "alignment": q4,
                "frequency": hdf["frequency"],
                "source_path": hdf["source_path"],
                "source_format": hdf["source_format"],
                "cameras": hdf["cameras"],
            }
        )

    e_next = [r["alignment"].get("e_next") for r in rows if "e_next" in r.get("alignment", {})]
    e_same = [r["alignment"].get("e_same") for r in rows if "e_same" in r.get("alignment", {})]
    inferred = rows[0]["alignment"].get("inferred") if rows else None
    all_next = all(r["alignment"].get("inferred") == "next_index" for r in rows) if rows else False
    da_set = {r["d_a"] for r in rows}
    freq_set = {r["frequency"] for r in rows}
    t_eq = None
    if any(t is not None for t in t_raws):
        t_eq = all(
            tr is not None and tc == tr - 1 for tr, tc in zip(t_raws, t_convs)
        )

    source = "existing_xpolicylab_converted_hdf5"
    converter_rerun = False
    converter_rerun_blocked = True
    reason = (
        "Legacy raw RoboTwin HDF5 (data/<task>/demo_clean/data/episode*.hdf5) is absent; "
        "scripts/process_data_xpolicylab.py only converts that layout. "
        "HF/XPolicyLab archives are already converted. P0 audits the files on disk."
    )
    meta_align = None
    if conversion_meta is not None:
        meta_align = conversion_meta.get("action_alignment")

    return {
        "task": task,
        "task_config": TASK_CONFIG,
        "env_cfg_type": ENV_CFG_TYPE,
        "source": source,
        "raw_RoboTwin_episode_present": any(t is not None for t in t_raws),
        "converter_rerun": converter_rerun,
        "converter_rerun_blocked": converter_rerun_blocked,
        "converter_rerun_reason": reason,
        "n_episodes": len(eps),
        "conversion_meta_path": str(meta_path) if meta_path.is_file() else None,
        "action_alignment_metadata": meta_align,
        "action_alignment_inferred": inferred_alignment_name(str(inferred)) if inferred else None,
        "action_alignment_inferred_raw_label": inferred,
        "action_alignment_all_episodes_next_state_as_action": all_next,
        "e_next_mean": float(np.mean(e_next)) if e_next else None,
        "e_same_mean": float(np.mean(e_same)) if e_same else None,
        "frequency": next(iter(freq_set)) if len(freq_set) == 1 else sorted(freq_set, key=lambda x: (x is None, x)),
        "frequency_unique": list(freq_set),
        "d_a": 14 if da_set == {14} else (next(iter(da_set)) if len(da_set) == 1 else sorted(da_set, key=lambda x: (x is None, x))),
        "d_a_unique": list(da_set),
        "d_a_ok": da_set == {14},
        "T_converted_min": min(t_convs) if t_convs else None,
        "T_converted_max": max(t_convs) if t_convs else None,
        "T_converted_median": float(np.median(t_convs)) if t_convs else None,
        "T_raw": None if all(t is None for t in t_raws) else t_raws,
        "T_converted_equals_T_raw_minus_1": t_eq,
        "official_default_alignment": "next_state_as_action",
        "official_same_index_flag": "--same-index-action",
        "official_act_chunk_size": ACT_CHUNK_SIZE,
        "official_act_num_epochs": ACT_NUM_EPOCHS,
        "official_deploy": "get_action() then take_action each chunk element (eval_policy_xpolicylab.py)",
        "episodes_head": rows[:3],
        "episodes_tail": rows[-1:] if rows else [],
    }


def official_commands(repo: Path, conda_env: str, gpu_id: str) -> dict[str, Any]:
    act = xpolicylab_root(repo) / "policy" / "ACT"
    ckpt_setting = f"{TASK_CONFIG}-{TASK}-{ENV_CFG_TYPE}-{ACTION_TYPE}"
    ckpt_dir_name = f"{ckpt_setting}-{ACT_SEED}"
    return {
        "process_data": [
            "bash",
            "process_data.sh",
            TASK_CONFIG,
            TASK,
            ENV_CFG_TYPE,
            ACTION_TYPE,
        ],
        "train": [
            "bash",
            "train.sh",
            TASK_CONFIG,
            TASK,
            ENV_CFG_TYPE,
            ACTION_TYPE,
            str(ACT_SEED),
            str(gpu_id),
        ],
        "eval": [
            "bash",
            "eval.sh",
            TASK_CONFIG,
            TASK,
            ckpt_dir_name,
            ENV_CFG_TYPE,
            ACTION_TYPE,
            str(ACT_SEED),
            str(gpu_id),
            str(gpu_id),
            conda_env,
            conda_env,
        ],
        "cwd": str(act),
        "data_symlink": {
            "from": str(demo_dir(repo, TASK)),
            "to": str(xpolicylab_root(repo) / "data" / TASK_CONFIG / TASK / ENV_CFG_TYPE),
        },
        "frozen": {
            "num_epochs": ACT_NUM_EPOCHS,
            "chunk_size": ACT_CHUNK_SIZE,
            "do_not_change_train_budget": True,
            "do_not_write_custom_rollout": True,
        },
        "eval_test_num": CUP_N,
        "note": "eval.sh uses XPolicyLab client; pass --test_num 16 via ROBOTWIN_EVAL_ARGS_FILE if client forwards argv.",
    }


def cup_gate(successes: int, n: int, smoke: bool) -> bool:
    if n <= 0:
        return False
    thr = 1.0 if smoke else CUP_G0
    return successes / n >= thr


def run_rtwx_ma0(output: str, config: RTWXMA0Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXMA0Config(output=output)
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(cfg.robotwin_repo)
    xpl = xpolicylab_root(repo)
    xpl_ok = xpolicylab_ready(repo)

    audit = audit_converted_task(repo, TASK)
    audit["robotwin_commit"] = git_rev(repo)
    audit["xpolicylab_pin_in_robotwin"] = XPOLICYLAB_PIN
    zip_stamp = xpl / "ZIP_COMMIT.txt"
    audit["xpolicylab_zip_commit"] = XPOLICYLAB_ZIP_COMMIT if zip_stamp.is_file() else None
    audit["xpolicylab_commit_checked_out"] = git_rev(xpl) if xpl_ok else None
    audit["xpolicylab_ready"] = xpl_ok
    _write_json(root / "P0" / "MA0_conversion_audit.json", audit)

    cmds = official_commands(repo, cfg.conda_env, cfg.gpu_id)
    _write_json(root / "P1" / "official_commands.json", cmds)

    pattern = None
    p1_ran = False
    p1_log: dict[str, Any] = {"ran": False}
    if cfg.p0_only or cfg.smoke:
        pattern = "ma0_p0_complete"
    elif not xpl_ok:
        pattern = "xpolicylab_checkout_missing"
        p1_log = {
            "ran": False,
            "blocked": True,
            "reason": (
                "XPolicyLab submodule is not checked out (missing setup_policy_server.py / policy/ACT/train.sh). "
                f"RoboTwin pin {XPOLICYLAB_PIN}. git submodule update --init XPolicyLab failed: Needed a single revision. "
                "Do not substitute a homemade ACT trainer."
            ),
        }
    elif cfg.skip_train:
        pattern = "ma0_p1_skipped"
    else:
        p1_ran = True
        p1_log = _run_official_p1(repo, cfg, cmds, root)

    if p1_ran and p1_log.get("eval"):
        ev = p1_log["eval"]
        suc = int(ev.get("successes") or 0)
        n = int(ev.get("n") or CUP_N)
        if cup_gate(suc, n, cfg.smoke):
            pattern = "ma0_cup_pass"
        else:
            pattern = "official_act_stack_unqualified"

    summary = {
        "schema_id": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "phase": "P0" if not p1_ran else "P1",
        "task": TASK,
        "pattern": pattern,
        "xpolicylab_ready": xpl_ok,
        "converter_rerun": False,
        "d_a_ok": audit["d_a_ok"],
        "action_alignment": audit["action_alignment_inferred"],
        "frequency": audit["frequency"],
        "n_episodes": audit["n_episodes"],
        "cup_gate": {"N": CUP_N, "G0": CUP_G0, "need": ">=4/16"},
        "unlocks_ma0_c0": pattern == "ma0_cup_pass",
        "unlocks_ma1": False,
        "p1": p1_log,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH, "config": asdict(cfg)})
    _write_json(root / "summary.json", summary)
    return summary


def _run_official_p1(repo: Path, cfg: RTWXMA0Config, cmds: dict[str, Any], root: Path) -> dict[str, Any]:
    """Invoke official scripts only. Never a custom Ka=1 loop."""
    act = Path(cmds["cwd"])
    link = cmds["data_symlink"]
    dest = Path(link["to"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(link["from"])
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_dir():
            pass
        else:
            raise RuntimeError(f"unexpected file at ACT data path: {dest}")
    else:
        dest.symlink_to(src, target_is_directory=True)

    env = os.environ.copy()
    env.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(xpolicylab_root(repo)), env.get("PYTHONPATH", "")]
    )
    logs = root / "P1" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def _bash(name: str, argv: list[str], timeout: int) -> dict[str, Any]:
        log = logs / f"{name}.log"
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                argv,
                cwd=str(act),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        return {"argv": argv, "cwd": str(act), "returncode": proc.returncode, "log": str(log)}

    out: dict[str, Any] = {"ran": True, "blocked": False}
    out["process_data"] = _bash("process_data", cmds["process_data"], timeout=3600)
    if out["process_data"]["returncode"] != 0:
        out["failed_at"] = "process_data"
        return out
    out["train"] = _bash("train", cmds["train"], timeout=86400)
    if out["train"]["returncode"] != 0:
        out["failed_at"] = "train"
        return out
    eval_env = env.copy()
    args_file = root / "P1" / "robotwin_eval_args.txt"
    args_file.write_text("--test_num\n16\n", encoding="utf-8")
    eval_env["ROBOTWIN_EVAL_ARGS_FILE"] = str(args_file)
    out["eval_invoke"] = _bash("eval", cmds["eval"], timeout=14400)
    out["eval"] = {"n": CUP_N, "successes": None, "note": "parse official eval log; see P1/logs/eval.log"}
    return out

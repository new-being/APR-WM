"""RTWX-AC1-D1: expert-only state-support metric audit. No policy training. No AUROC selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluation.state_coverage_metrics import ood_occupancy, tau_95
from .evaluation.state_support_d1 import (
    EuclideanKNN,
    GlobalMahalanobis,
    LocalDiagKNN,
    PhaseConditionedKNN,
    phase_from_process,
)
from .policy.chunk_normalizer import ChunkNormalizer
from .policy.process_features import ProcessEncoder
from .policy.task_embedding import pack_state
from .policy.chunk_dataset import split_episode_indices
from .rtwx_ac0 import load_cached_episode
from .rtwx_mr0 import MR0_TASKS
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0AC1D1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_ac1_d1.support_metric.v1"
CONFIRM_SEED = 52100
MAX_FALSE_OOD = 0.10
BRANCH_TIE = ("B0", "B1", "B2", "B3")


@dataclass(frozen=True)
class RTWXAC1D1Config:
    output: str = "runs/rtwx_ac1_d1"
    ac0_run: str = "runs/rtwx_ac0"
    smoke: bool = False


def _load_task_eps(ac0: Path, task: str) -> list[dict[str, Any]]:
    enc = ProcessEncoder()
    out = []
    for i in range(50):
        cpath = ac0 / "cache" / task / f"ep_{i:03d}.npz"
        if not cpath.is_file():
            continue
        ep = load_cached_episode(cpath, task)
        st = np.stack([pack_state(fr) for fr in ep["frames"]]).astype(np.float64)
        pr = np.stack([enc.encode(task, fr) for fr in ep["frames"]]).astype(np.float64)
        ph = np.array([phase_from_process(z) for z in pr], dtype=np.int64)
        out.append({"i": i, "state": st, "phase": ph})
    return out


def _gather(eps: list[dict[str, Any]], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    by = {int(e["i"]): e for e in eps}
    chunks_s, chunks_p = [], []
    for i in ids:
        e = by[int(i)]
        chunks_s.append(e["state"])
        chunks_p.append(e["phase"])
    return np.concatenate(chunks_s, 0), np.concatenate(chunks_p, 0)


def _fit(branch: str, tr_s: np.ndarray, tr_p: np.ndarray, mu, sig):
    if branch == "B0":
        return EuclideanKNN(tr_s, mu, sig)
    if branch == "B1":
        return GlobalMahalanobis(tr_s, mu, sig)
    if branch == "B2":
        return PhaseConditionedKNN(tr_s, tr_p, mu, sig)
    if branch == "B3":
        return LocalDiagKNN(tr_s, mu, sig)
    raise ValueError(branch)


def _dist(model, branch: str, st: np.ndarray, ph: np.ndarray) -> np.ndarray:
    if branch == "B2":
        return model.distances(st, ph)
    return model.distances(st)


def _eval_split(branch: str, tr_s, tr_p, va_s, va_p, te_s, te_p, mu, sig) -> dict[str, float]:
    m = _fit(branch, tr_s, tr_p, mu, sig)
    d_va = _dist(m, branch, va_s, va_p)
    tau = tau_95(d_va)
    d_te = _dist(m, branch, te_s, te_p)
    return {"tau": float(tau), "r_ood": ood_occupancy(d_te, tau), "n_test": int(d_te.size), "n_val": int(d_va.size)}


def _select(per_branch: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    qualified = []
    for br in BRANCH_TIE:
        rs = [per_branch[br][t]["r_ood"] for t in MR0_TASKS]
        if all(r <= MAX_FALSE_OOD for r in rs):
            qualified.append((br, max(rs), float(np.mean(rs))))
    if not qualified:
        return {"winner": None, "reason": "no_branch_all_tasks_le_0.10"}
    qualified.sort(key=lambda x: (x[1], x[2], BRANCH_TIE.index(x[0])))
    w = qualified[0][0]
    return {"winner": w, "reason": "min_max_then_mean_then_simpler", "qualified": [q[0] for q in qualified]}


def run_rtwx_ac1_d1(output: str | Path, config: RTWXAC1D1Config | None = None) -> dict[str, Any]:
    cfg = config or RTWXAC1D1Config(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    ac0 = _resolve_apr(cfg.ac0_run)
    ckpt = torch.load(str(ac0 / "B0" / "best.pt"), map_location="cpu", weights_only=False)
    norm = ChunkNormalizer.from_dict(ckpt["normalizer"])
    mu, sig = np.asarray(norm.mu_s), np.asarray(norm.sig_s)
    man = json.loads((ac0 / "split_manifest.json").read_text())

    eps_by = {t: _load_task_eps(ac0, t) for t in MR0_TASKS}
    if cfg.smoke:
        for t in MR0_TASKS:
            eps_by[t] = eps_by[t][:6]

    selection: dict[str, dict[str, dict[str, float]]] = {b: {} for b in BRANCH_TIE}
    for task in MR0_TASKS:
        sp = man[task]
        if cfg.smoke:
            n = len(eps_by[task])
            ids = [int(e["i"]) for e in eps_by[task]]
            sp = {"train": ids[: max(n - 2, 1)], "val": ids[max(n - 2, 1) : max(n - 1, 1)], "test": ids[max(n - 1, 1) :]}
        packs = {k: _gather(eps_by[task], sp[k]) for k in ("train", "val", "test")}
        for br in BRANCH_TIE:
            selection[br][task] = _eval_split(br, *packs["train"], *packs["val"], *packs["test"], mu, sig)
            print(f"[rtwx-ac1-d1] sel {br} {task} r_ood={selection[br][task]['r_ood']:.4f}", flush=True)

    sel = _select(selection)
    winner = sel["winner"]

    confirm: dict[str, dict[str, float]] = {}
    confirm_ok = False
    if winner is not None and not cfg.smoke:
        rng = np.random.default_rng(CONFIRM_SEED)
        confirm = {t: {} for t in MR0_TASKS}
        ok_all = True
        for task in MR0_TASKS:
            n = len(eps_by[task])
            sp = split_episode_indices(n, rng, 40, 5, 5) if n == 50 else split_episode_indices(n, rng, n - 2, 1, 1)
            packs = {k: _gather(eps_by[task], sp[k]) for k in ("train", "val", "test")}
            rec = _eval_split(winner, *packs["train"], *packs["val"], *packs["test"], mu, sig)
            confirm[task] = rec
            ok_all = ok_all and rec["r_ood"] <= MAX_FALSE_OOD
            print(f"[rtwx-ac1-d1] confirm {winner} {task} r_ood={rec['r_ood']:.4f}", flush=True)
        confirm_ok = bool(ok_all)

    if winner is None:
        pattern = "support_geometry_unqualified"
    elif not confirm_ok:
        pattern = "support_metric_unstable" if not cfg.smoke else "support_metric_selected_smoke"
    else:
        pattern = "support_metric_qualified"

    summary = {
        "pattern": pattern,
        "scientific_result": True,
        "winner": winner,
        "selection_rule": sel,
        "selection": selection,
        "confirm_seed": CONFIRM_SEED,
        "confirm": confirm,
        "confirm_ok": confirm_ok,
        "max_false_ood": MAX_FALSE_OOD,
        "unlocks_ac1_r0": False,
        "unlocks_d0_rescore": pattern == "support_metric_qualified",
        "policy_auroc_used_for_selection": False,
        "smoke": cfg.smoke,
    }
    _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary})
    print(f"[rtwx-ac1-d1] winner={winner} pattern={pattern}", flush=True)
    return summary

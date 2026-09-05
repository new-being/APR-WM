"""TASK-X1 Stage A instrument gates. No L1-better-than-B0 gate."""

from __future__ import annotations

import numpy as np
import torch

from ..evaluation.ac0_offline_metrics import chunk_diversity, per_horizon_l1
from ..policy.diffusion_chunk_policy import DiffusionChunkPolicy, policy_seed
from ..policy.task_embedding import TASK_TO_ID


def percentile_range(x: np.ndarray) -> dict[str, list[float]]:
    # x: [N, H, D]
    flat = x.reshape(-1, x.shape[-1])
    return {
        "p01": [float(v) for v in np.percentile(flat, 1, axis=0)],
        "p50": [float(v) for v in np.percentile(flat, 50, axis=0)],
        "p99": [float(v) for v in np.percentile(flat, 99, axis=0)],
    }


@torch.no_grad()
def denoise_val_chunks(policy: DiffusionChunkPolicy, loader, normalizer, device: str) -> tuple[np.ndarray, np.ndarray]:
    preds, tgts = [], []
    n = 0
    for batch in loader:
        s = torch.from_numpy(normalizer.n_state(batch["state"].numpy()).astype(np.float32))
        a = batch["chunk"].numpy()
        tid = batch["task_id"].numpy()
        out = []
        for i in range(s.shape[0]):
            seed = policy_seed(int(tid[i]), 7, n + i)
            yn = policy.predict_chunk(s[i].numpy(), int(tid[i]), seed)
            out.append(normalizer.denorm_act(yn[None])[0])
        preds.append(np.stack(out, 0))
        tgts.append(a)
        n += int(s.shape[0])
        if n >= 256:
            break
    return np.concatenate(preds, 0), np.concatenate(tgts, 0)


def stage_a_gates(pred: np.ndarray, tgt: np.ndarray, policy: DiffusionChunkPolicy, dummy_state: np.ndarray) -> dict:
    g0 = pred.shape[-2:] == (8, 14)
    g1 = bool(np.isfinite(pred).all())
    lo = tgt.min(axis=(0, 1)) - 0.5
    hi = tgt.max(axis=(0, 1)) + 0.5
    g2 = bool((pred >= lo - 1e-6).all() and (pred <= hi + 1e-6).all())
    a = policy.predict_chunk(dummy_state, 0, 12345)
    b = policy.predict_chunk(dummy_state, 0, 12345)
    g3 = bool(np.max(np.abs(a - b)) < 1e-6)
    c = policy.predict_chunk(dummy_state, 0, 99999)
    d_pred = chunk_diversity(pred)
    d_demo = chunk_diversity(tgt)
    g4 = bool(d_pred >= 0.25 * max(d_demo, 1e-6) and not policy.fills_by_repeat)
    l1_h = per_horizon_l1(pred, tgt)
    return {
        "G0_shape": g0,
        "G1_finite": g1,
        "G2_bounds": g2,
        "G3_deterministic": g3,
        "G4_temporal_chunk": g4,
        "pass": bool(g0 and g1 and g2 and g3 and g4),
        "max_abs_seed_delta": float(np.max(np.abs(a - b))),
        "diff_seed_l1": float(np.mean(np.abs(a - c))),
        "l1": float(np.abs(pred - tgt).mean()),
        "l1_h": [float(x) for x in l1_h],
        "e0": float(l1_h[0]),
        "e7": float(l1_h[-1]),
        "d_pred": d_pred,
        "d_demo": d_demo,
        "pred_range": percentile_range(pred),
        "demo_range": percentile_range(tgt),
    }

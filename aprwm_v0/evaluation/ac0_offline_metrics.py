"""AC0 offline chunk metrics."""

from __future__ import annotations

import numpy as np
import torch


def per_horizon_l1(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    err = np.abs(pred - tgt).mean(axis=-1)
    return err.mean(axis=0)


def chunk_diversity(chunks: np.ndarray) -> float:
    d = np.linalg.norm(chunks[:, 1:] - chunks[:, :-1], axis=-1)
    return float(d.mean())


@torch.no_grad()
def eval_offline(model, loader, normalizer, device: str, use_process: bool) -> dict:
    model.eval()
    preds, tgts, n_h = [], [], None
    for batch in loader:
        s = normalizer.n_state(batch["state"].numpy())
        p = normalizer.n_proc(batch["process"].numpy())
        a = batch["chunk"].numpy()
        st = torch.from_numpy(s.astype(np.float32)).to(device)
        pr = torch.from_numpy(p.astype(np.float32)).to(device)
        tid = batch["task_id"].to(device)
        y = model(st, tid, pr if use_process else pr)
        y_np = normalizer.denorm_act(y.cpu().numpy())
        preds.append(y_np)
        tgts.append(a)
    pred = np.concatenate(preds, 0)
    tgt = np.concatenate(tgts, 0)
    l1_h = per_horizon_l1(pred, tgt)
    l1 = float(np.abs(pred - tgt).mean())
    return {
        "l1": l1,
        "l1_h": [float(x) for x in l1_h],
        "e0": float(l1_h[0]),
        "e7": float(l1_h[-1]),
        "d_pred": chunk_diversity(pred),
        "d_demo": chunk_diversity(tgt),
        "n": int(pred.shape[0]),
    }

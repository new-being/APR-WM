"""AC2 Stage A: shape, finite, bounds, determinism. No L1 winner."""

from __future__ import annotations

import numpy as np
import torch


def stage_a_gates(pred: np.ndarray, tgt: np.ndarray, model, dummy) -> dict:
    g0 = pred.shape[-1] == 14 and pred.ndim == 2
    g1 = bool(np.isfinite(pred).all())
    lo = tgt.min(axis=0) - 0.5
    hi = tgt.max(axis=0) + 0.5
    g2 = bool((pred >= lo - 1e-6).all() and (pred <= hi + 1e-6).all())
    model.eval()
    with torch.no_grad():
        a = model(*dummy).detach().cpu().numpy()
        b = model(*dummy).detach().cpu().numpy()
    g3 = bool(np.max(np.abs(a - b)) < 1e-6)
    return {
        "G0_shape": bool(g0),
        "G1_finite": g1,
        "G2_bounds": g2,
        "G3_deterministic": g3,
        "pass": bool(g0 and g1 and g2 and g3),
        "l1": float(np.abs(pred - tgt).mean()),
        "max_abs_seed_delta": float(np.max(np.abs(a - b))),
    }


@torch.no_grad()
def eval_l1(model, loader, norm, device: str, kind: str) -> dict:
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        st = torch.from_numpy(norm.n_state(batch["states"].numpy()).astype(np.float32)).to(device)
        tid = batch["task_id"].to(device)
        pa = None
        if kind == "B2":
            pa = torch.from_numpy(norm.n_act(batch["past_actions"].numpy()).astype(np.float32)).to(device)
        y = model(st, tid, pa)
        preds.append(norm.denorm_act(y.detach().cpu().numpy()))
        tgts.append(batch["target_action"].numpy())
    pred = np.concatenate(preds, 0)
    tgt = np.concatenate(tgts, 0)
    return {"l1": float(np.abs(pred - tgt).mean()), "pred": pred, "tgt": tgt}

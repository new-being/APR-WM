"""AC1-D0 occupancy / first exit / AUROC. Shadow metrics only."""

from __future__ import annotations

import numpy as np

PERSIST_TICKS = 3


def tau_95(distances: np.ndarray) -> float:
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return float("inf")
    return float(np.quantile(d, 0.95))


def ood_occupancy(distances: np.ndarray, tau: float) -> float:
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    return float(np.mean(d > tau))


def first_persistent_exit(distances: np.ndarray, tau: float, persist: int = PERSIST_TICKS) -> int | None:
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    run = 0
    for t, v in enumerate(d):
        if v > tau:
            run += 1
            if run >= persist:
                return int(t - persist + 1)
        else:
            run = 0
    return None


def auroc_higher_is_policy(policy_d: np.ndarray, expert_d: np.ndarray) -> float:
    p = np.asarray(policy_d, dtype=np.float64).reshape(-1)
    e = np.asarray(expert_d, dtype=np.float64).reshape(-1)
    if p.size == 0 or e.size == 0:
        return float("nan")
    # P(policy score > expert score) + 0.5 P(tie); Wilcoxon-Mann-Whitney
    # vectorized
    p_s = np.sort(p)
    e_s = np.sort(e)
    i = 0
    gt = 0.0
    eq = 0.0
    for ev in e_s:
        while i < p_s.size and p_s[i] < ev:
            i += 1
        j = i
        while j < p_s.size and p_s[j] == ev:
            j += 1
        gt += p_s.size - j
        eq += j - i
    n = float(p.size * e.size)
    return float((gt + 0.5 * eq) / n)

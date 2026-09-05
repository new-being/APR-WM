"""Non-oracle relative-registration support scores (HYB0)."""

from __future__ import annotations

import numpy as np

SUPPORT_THRESH_FRAC = 0.05


def _overlap_frac(
    src: np.ndarray,
    dst: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    max_dist: float,
) -> float:
    from scipy.spatial import cKDTree

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        return 0.0
    warped = (r @ src.T).T + t.reshape(1, 3)
    d, _ = cKDTree(dst).query(warped, k=1)
    return float((d <= max_dist).mean())


def symmetric_support_score(
    cloud0: np.ndarray,
    cloud1: np.ndarray,
    delta_r: np.ndarray,
    delta_t: np.ndarray,
    object_diameter: float,
    *,
    thresh_frac: float = SUPPORT_THRESH_FRAC,
) -> float:
    """q_rel = min(rho_01, rho_10) with estimated delta T."""
    r = np.asarray(delta_r, dtype=np.float64).reshape(3, 3)
    t = np.asarray(delta_t, dtype=np.float64).reshape(3)
    max_d = float(thresh_frac * object_diameter)
    r_inv = r.T
    t_inv = -r_inv @ t
    rho01 = _overlap_frac(cloud0, cloud1, r, t, max_d)
    rho10 = _overlap_frac(cloud1, cloud0, r_inv, t_inv, max_d)
    return float(min(rho01, rho10))


def calibrate_threshold_youden(
    scores: np.ndarray,
    safe_labels: np.ndarray,
) -> float:
    """Youden J; tie-break to largest tau (conservative)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(safe_labels, dtype=bool)
    if s.size == 0 or not np.any(y) or np.all(y):
        return 0.5
    candidates = np.unique(s)
    best_tau = float(candidates[0])
    best_j = -1.0
    n_pos = float(y.sum())
    n_neg = float((~y).sum())
    for tau in candidates:
        pred = s >= tau
        tpr = float((pred & y).sum()) / n_pos
        fpr = float((pred & ~y).sum()) / n_neg
        j = tpr - fpr
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and float(tau) > best_tau):
            best_j = j
            best_tau = float(tau)
    return best_tau


def relative_safe_label(
    e_axis_deg: float,
    e_pos_m: float,
    *,
    axis_thresh: float = 15.0,
    pos_thresh_m: float = 0.05,
) -> bool:
    return bool(e_axis_deg <= axis_thresh and e_pos_m <= pos_thresh_m)

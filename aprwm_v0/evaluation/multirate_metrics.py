"""MR0 non-inferiority and pattern helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

SUCCESS_NONINFERIORITY_PP = 0.05
PER_TASK_MAX_DROP_PP = 0.10
SAFETY_MAX_INCREASE_PP = 0.02
MIN_ASYMMETRY_RATIO = 2.0
JOINT_LOW_RATE_MIN = 4


def success_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if r["outcome"]["success"] else 0.0 for r in rows]))


def safety_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if r["outcome"]["safety_violation"] else 0.0 for r in rows]))


def noninferior(
    rows_k: list[dict[str, Any]],
    rows_b0: list[dict[str, Any]],
    *,
    pooled_drop: float = SUCCESS_NONINFERIORITY_PP,
    per_task_drop: float = PER_TASK_MAX_DROP_PP,
    safety_inc: float = SAFETY_MAX_INCREASE_PP,
) -> bool:
    s_k = success_rate(rows_k)
    s0 = success_rate(rows_b0)
    if not (np.isfinite(s_k) and np.isfinite(s0)):
        return False
    if s_k - s0 < -pooled_drop:
        return False
    tasks = sorted({r["task"] for r in rows_b0})
    for task in tasks:
        a = [r for r in rows_k if r["task"] == task]
        b = [r for r in rows_b0 if r["task"] == task]
        if success_rate(a) - success_rate(b) < -per_task_drop:
            return False
    if safety_rate(rows_k) - safety_rate(rows_b0) > safety_inc:
        return False
    return True


def max_safe_stride(
    sweep: dict[int, list[dict[str, Any]]],
    rows_b0: list[dict[str, Any]],
) -> int:
    star = 1
    for k in sorted(sweep):
        if k == 1:
            star = 1
            continue
        if noninferior(sweep[k], rows_b0):
            star = int(k)
        else:
            break
    return star


def decide_pattern(ks_star: int, ka_star: int) -> str:
    if ks_star >= JOINT_LOW_RATE_MIN and ka_star >= JOINT_LOW_RATE_MIN:
        return "joint_low_rate_supported"
    if ka_star >= MIN_ASYMMETRY_RATIO * max(ks_star, 1):
        return "multirate_asymmetry_supported"
    return "multirate_asymmetry_not_supported"

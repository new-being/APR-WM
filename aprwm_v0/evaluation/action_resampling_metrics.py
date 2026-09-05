"""MR0-A pattern helpers (action-only; no Ks*)."""

from __future__ import annotations

from typing import Any

from .multirate_metrics import max_safe_stride

ACTION_STRIDES = (1, 2, 4, 8)


def decide_action_reuse(ka_star: int) -> str:
    if ka_star >= 4:
        return "action_temporal_reuse_supported"
    if ka_star == 2:
        return "action_temporal_reuse_limited"
    return "action_temporal_reuse_not_supported"


def ka_star_from_sweep(
    sweep: dict[int, list[dict[str, Any]]],
    rows_a1: list[dict[str, Any]],
) -> int:
    return max_safe_stride(sweep, rows_a1)

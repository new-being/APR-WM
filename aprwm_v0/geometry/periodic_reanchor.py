"""Periodic re-anchor policies (RAB0). One mechanism per policy; no adaptive trigger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from .relative_state_chain import ChainState, propagate_state
from .relative_validity import symmetric_support_score

AbsFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class ReanchorConfig:
    kind: str
    period: int = 0


@dataclass
class Keyframe:
    t: int
    cloud: np.ndarray
    state: ChainState


@dataclass
class ReanchorResult:
    state: ChainState
    attempted: bool
    valid: bool
    applied: bool
    source_frame: int | None
    q_support: float | None
    icp_rmse: float | None
    n_corr: int | None


def copy_state(state: ChainState) -> ChainState:
    return ChainState(p=state.p.copy(), n=state.n.copy(), t_accum=state.t_accum.copy())


def no_op(state: ChainState, *, attempted: bool = False, valid: bool = True) -> ReanchorResult:
    return ReanchorResult(
        state=copy_state(state),
        attempted=attempted,
        valid=valid,
        applied=False,
        source_frame=None,
        q_support=None,
        icp_rmse=None,
        n_corr=None,
    )


def is_scheduled(t: int, period: int) -> bool:
    if period <= 0 or t <= 0:
        return False
    return t % int(period) == 0


def _applied(
    state: ChainState,
    *,
    source_frame: int,
    q_support: float | None,
    icp_rmse: float | None,
    n_corr: int | None,
) -> ReanchorResult:
    return ReanchorResult(
        state=copy_state(state),
        attempted=True,
        valid=True,
        applied=True,
        source_frame=int(source_frame),
        q_support=q_support,
        icp_rmse=icp_rmse,
        n_corr=n_corr,
    )


class PeriodicReanchorPolicy(Protocol):
    cfg: ReanchorConfig

    def initialize(self, t0: int, cloud0: np.ndarray, state0: ChainState) -> None: ...

    def maybe_reanchor(
        self,
        t: int,
        current_state: ChainState,
        current_cloud: np.ndarray,
        object_diameter: float,
        *,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> ReanchorResult: ...


class NoResetPolicy:
    def __init__(self) -> None:
        self.cfg = ReanchorConfig(kind="none", period=0)

    def initialize(self, t0: int, cloud0: np.ndarray, state0: ChainState) -> None:
        return None

    def maybe_reanchor(
        self,
        t: int,
        current_state: ChainState,
        current_cloud: np.ndarray,
        object_diameter: float,
        *,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> ReanchorResult:
        return no_op(current_state, attempted=False)


class AbsoluteResetPolicy:
    def __init__(self, period: int, abs_fn: AbsFn) -> None:
        self.cfg = ReanchorConfig(kind="absolute", period=int(period))
        self._abs_fn = abs_fn

    def initialize(self, t0: int, cloud0: np.ndarray, state0: ChainState) -> None:
        return None

    def maybe_reanchor(
        self,
        t: int,
        current_state: ChainState,
        current_cloud: np.ndarray,
        object_diameter: float,
        *,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> ReanchorResult:
        if not is_scheduled(t, self.cfg.period):
            return no_op(current_state)
        p, n = self._abs_fn(current_cloud)
        n = np.asarray(n, dtype=np.float64).reshape(3)
        n = n / max(np.linalg.norm(n), 1e-12)
        new = ChainState(
            p=np.asarray(p, dtype=np.float64).reshape(3),
            n=n,
            t_accum=current_state.t_accum.copy(),
        )
        return _applied(new, source_frame=t, q_support=None, icp_rmse=None, n_corr=None)


class InitialAnchorPolicy:
    def __init__(self, period: int) -> None:
        self.cfg = ReanchorConfig(kind="initial", period=int(period))
        self.initial: Keyframe | None = None

    def initialize(self, t0: int, cloud0: np.ndarray, state0: ChainState) -> None:
        self.initial = Keyframe(t=int(t0), cloud=np.asarray(cloud0).copy(), state=copy_state(state0))

    def maybe_reanchor(
        self,
        t: int,
        current_state: ChainState,
        current_cloud: np.ndarray,
        object_diameter: float,
        *,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> ReanchorResult:
        if not is_scheduled(t, self.cfg.period):
            return no_op(current_state)
        if self.initial is None:
            return no_op(current_state, attempted=True, valid=False)
        res = estimate_relative_rigid(self.initial.cloud, current_cloud, object_diameter, icp_cfg)
        if not res.valid:
            return no_op(current_state, attempted=True, valid=False)
        new = propagate_state(self.initial.state, res.R, res.t)
        q = symmetric_support_score(self.initial.cloud, current_cloud, res.R, res.t, object_diameter)
        return _applied(
            new,
            source_frame=self.initial.t,
            q_support=float(q),
            icp_rmse=float(res.rmse),
            n_corr=int(res.n_corr),
        )


class RollingKeyframePolicy:
    def __init__(self, period: int) -> None:
        self.cfg = ReanchorConfig(kind="rolling", period=int(period))
        self.keyframe: Keyframe | None = None

    def initialize(self, t0: int, cloud0: np.ndarray, state0: ChainState) -> None:
        self.keyframe = Keyframe(t=int(t0), cloud=np.asarray(cloud0).copy(), state=copy_state(state0))

    def maybe_reanchor(
        self,
        t: int,
        current_state: ChainState,
        current_cloud: np.ndarray,
        object_diameter: float,
        *,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> ReanchorResult:
        if not is_scheduled(t, self.cfg.period):
            return no_op(current_state)
        if self.keyframe is None:
            return no_op(current_state, attempted=True, valid=False)
        src = self.keyframe
        res = estimate_relative_rigid(src.cloud, current_cloud, object_diameter, icp_cfg)
        if not res.valid:
            return no_op(current_state, attempted=True, valid=False)
        new = propagate_state(src.state, res.R, res.t)
        q = symmetric_support_score(src.cloud, current_cloud, res.R, res.t, object_diameter)
        self.keyframe = Keyframe(t=int(t), cloud=np.asarray(current_cloud).copy(), state=copy_state(new))
        return _applied(
            new,
            source_frame=src.t,
            q_support=float(q),
            icp_rmse=float(res.rmse),
            n_corr=int(res.n_corr),
        )


def make_policy(kind: str, period: int, *, abs_fn: AbsFn | None = None) -> PeriodicReanchorPolicy:
    if kind == "none":
        return NoResetPolicy()
    if kind == "absolute":
        if abs_fn is None:
            raise ValueError("absolute policy requires abs_fn")
        return AbsoluteResetPolicy(period, abs_fn)
    if kind == "initial":
        return InitialAnchorPolicy(period)
    if kind == "rolling":
        return RollingKeyframePolicy(period)
    raise ValueError(f"unknown reanchor kind {kind}")

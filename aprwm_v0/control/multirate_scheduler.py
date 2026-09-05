"""World-state clock: compute-skipping stale hold (no dynamics extrapolation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..geometry.relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from ..geometry.relative_state_chain import ChainState, propagate_state
from ..rtwx_o0rel0 import ICP_CFG


@dataclass
class WorldStateClock:
    stride: int
    cached: ChainState | None = None
    last_cloud: np.ndarray | None = None
    n_updates: int = 0

    def should_update(self, t: int) -> bool:
        s = max(int(self.stride), 1)
        return t == 0 or t % s == 0

    def update_stride(
        self,
        t: int,
        cloud: np.ndarray,
        *,
        init_state: ChainState | None = None,
        object_diameter: float,
        icp_cfg: RelativeICPConfig | None = None,
    ) -> tuple[ChainState, bool]:
        """Compute-skipping: at update ticks register last_cloud ↔ cloud (not adjacent chain)."""
        if t == 0:
            if init_state is None:
                raise ValueError("t=0 requires init_state")
            self.cached = ChainState(
                p=init_state.p.copy(), n=init_state.n.copy(), t_accum=init_state.t_accum.copy(),
            )
            self.last_cloud = np.asarray(cloud).copy()
            self.n_updates += 1
            return self.cached, True
        if not self.should_update(t):
            assert self.cached is not None
            return self.cached, False
        cfg = icp_cfg or ICP_CFG
        assert self.cached is not None and self.last_cloud is not None
        res = estimate_relative_rigid(self.last_cloud, cloud, object_diameter, cfg)
        if res.valid:
            self.cached = propagate_state(self.cached, res.R, res.t)
        self.last_cloud = np.asarray(cloud).copy()
        self.n_updates += 1
        return self.cached, True

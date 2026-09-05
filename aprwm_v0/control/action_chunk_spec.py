"""Frozen action-chunk contract for MR0-P0 / MR0-A."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


ALLOWED_SEMANTICS = (
    "joint_position",
    "joint_delta",
    "ee_pose",
)


@dataclass(frozen=True)
class ActionChunkSpec:
    horizon: int
    action_dim: int
    semantics: str
    policy_dt_env_ticks: int
    deterministic: bool

    def __post_init__(self) -> None:
        if int(self.horizon) < 8:
            raise ValueError(f"H_a={self.horizon} < 8")
        if int(self.action_dim) < 1:
            raise ValueError("action_dim < 1")
        if self.semantics not in ALLOWED_SEMANTICS:
            raise ValueError(f"unknown semantics={self.semantics}")
        if int(self.policy_dt_env_ticks) != 1:
            raise ValueError("policy_dt must equal one env policy tick; interpolation is forbidden")

    def config_hash(self) -> str:
        payload = (
            f"mr0p0.spec.v1.H={self.horizon}.d={self.action_dim}."
            f"sem={self.semantics}.dt={self.policy_dt_env_ticks}.det={int(self.deterministic)}"
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d

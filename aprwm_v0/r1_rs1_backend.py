"""Door Mode-A backend for R1-RS1 (frozen R0.6 transfer)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .mujoco_force import (
    NominalPassiveParams,
    get_bias_force,
    get_constraint_force,
    get_mass_matrix,
    nominal_passive_force,
)
from .mujoco_physics import import_mujoco
from .r1_rs import (
    _find_hinge_dof,
    _joint_id_for_dof,
    _mujoco_handles,
    _set_joint_passive,
)

REGIME_ALPHA = {
    "C0": 0.0,
    "C1-L": -0.12,
    "C1-H": -0.24,
    "CNEG": +0.12,
    "C2-latch": 0.0,
}


class DoorModeABackend:
    """Mode-A Door hinge backend with optional hidden drag and latch."""

    def __init__(
        self,
        *,
        regime: str,
        friction: float,
        damping: float,
        seed: int,
        alpha: float | None = None,
    ):
        self.regime = regime
        self.friction = float(friction)
        self.damping = float(damping)
        self.alpha = float(REGIME_ALPHA[regime] if alpha is None else alpha)
        self.use_latch = regime == "C2-latch"
        import_mujoco(register=True)
        import robosuite as suite

        self.mujoco = import_mujoco(register=True)
        self.env = suite.make(
            "Door",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
            use_latch=self.use_latch,
            seed=seed,
        )
        self.env.reset()
        self.model, self.data = _mujoco_handles(self.env)
        self.model.opt.viscosity = 0.0
        self.model.opt.density = 0.0
        self.hinge_dof = _find_hinge_dof(self.model)
        self.joint_id = _joint_id_for_dof(self.model, self.hinge_dof)
        _set_joint_passive(
            self.model, self.joint_id, friction=self.friction, damping=self.damping
        )
        self.hinge_qpos_adr = int(self.model.jnt_qposadr[self.joint_id])
        self.lo, self.hi = [float(x) for x in self.model.jnt_range[self.joint_id]]
        self.nominal = NominalPassiveParams(damping=self.damping, frictionloss=0.0)
        self.robot_qpos0 = np.array(self.data.qpos, dtype=np.float64, copy=True)
        self.robot_qpos_idx = [
            i for i in range(self.model.nq) if i != self.hinge_qpos_adr
        ]
        self.robot_dof_idx = [i for i in range(self.model.nv) if i != self.hinge_dof]
        if self.hi > self.lo:
            self.data.qpos[self.hinge_qpos_adr] = 0.5 * (self.lo + self.hi)
            self.data.qvel[self.hinge_dof] = 0.0
        self._freeze_robot()
        self.mujoco.mj_forward(self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        mass = float(
            get_mass_matrix(self.model, self.data)[self.hinge_dof, self.hinge_dof]
        )
        self.mass_proxy = max(mass, 1.0e-6)

    def close(self) -> None:
        self.env.close()

    def _freeze_robot(self) -> None:
        self.data.qpos[self.robot_qpos_idx] = self.robot_qpos0[self.robot_qpos_idx]
        self.data.qvel[self.robot_dof_idx] = 0.0

    def _hidden(self, velocity: float) -> float:
        return self.alpha * abs(velocity) * velocity

    def _validity(self, q: float, margin: float) -> str:
        dist = min(q - self.lo, self.hi - q) if self.hi > self.lo else 1.0
        if dist <= margin:
            return "boundary"
        if self.use_latch and q < self.lo + 2.0 * margin:
            return "unsupported"
        return "modeled"

    def set_state(self, q: float, velocity: float) -> None:
        self.data.qpos[self.hinge_qpos_adr] = float(q)
        self.data.qvel[self.hinge_dof] = float(velocity)
        self._freeze_robot()
        if hasattr(self.data, "ctrl"):
            self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _learner_residual(self, torque_command: float) -> float:
        """Residual using only commanded torque as known control (no hidden)."""
        mass = get_mass_matrix(self.model, self.data)
        qacc = np.asarray(self.data.qacc, dtype=np.float64)
        bias = float(get_bias_force(self.data)[self.hinge_dof])
        constraint = float(get_constraint_force(self.data)[self.hinge_dof])
        passive = float(
            nominal_passive_force(
                np.asarray(self.data.qvel, dtype=np.float64),
                self.nominal,
                dof_index=self.hinge_dof,
            )[self.hinge_dof]
        )
        return float(
            (mass @ qacc)[self.hinge_dof]
            + bias
            - float(torque_command)
            - constraint
            - passive
        )

    def force_sample(
        self, q: float, velocity: float, torque: float
    ) -> tuple[float, float, float, float, str]:
        """Return residual, density_tangent, acceleration, hidden, support.

        Truth steps with ``torque + hidden``, but learner residual subtracts only
        the commanded torque.  Reading full ``qfrc_applied`` as known control
        would cancel the hidden intervention (label leakage).
        """
        self.set_state(q, velocity)
        hidden = self._hidden(velocity)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.hinge_dof] = float(torque) + hidden
        self.mujoco.mj_step(self.model, self.data)
        residual = self._learner_residual(torque)
        mass = get_mass_matrix(self.model, self.data)
        qacc = float(self.data.qacc[self.hinge_dof])
        density_tangent = float(mass[self.hinge_dof, self.hinge_dof] * qacc) / (
            self.mass_proxy
        )
        support = self._validity(float(self.data.qpos[self.hinge_qpos_adr]), 0.05)
        return residual, density_tangent, qacc, hidden, support

    def transition_with_validity(
        self,
        q: float,
        velocity: float,
        torque: float,
        *,
        horizon: int,
        residual_force: float = 0.0,
        margin: float = 0.05,
        include_hidden: bool = True,
    ) -> tuple[float, float, bool, str]:
        self.set_state(q, velocity)
        valid = True
        support = "modeled"
        for _ in range(max(1, horizon)):
            v = float(self.data.qvel[self.hinge_dof])
            hidden = self._hidden(v) if include_hidden else 0.0
            self._freeze_robot()
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.hinge_dof] = (
                float(torque) + hidden + float(residual_force)
            )
            self.mujoco.mj_step(self.model, self.data)
            q_now = float(self.data.qpos[self.hinge_qpos_adr])
            support = self._validity(q_now, margin)
            valid = valid and support == "modeled"
            if not np.isfinite(q_now) or not np.isfinite(
                float(self.data.qvel[self.hinge_dof])
            ):
                return q_now, float(self.data.qvel[self.hinge_dof]), False, "unsupported"
        return (
            float(self.data.qpos[self.hinge_qpos_adr]),
            float(self.data.qvel[self.hinge_dof]),
            bool(valid),
            support,
        )

    def instantaneous_vector_field(
        self,
        q: float,
        velocity: float,
        torque: float,
        *,
        residual_force: float = 0.0,
        include_hidden: bool = False,
    ) -> np.ndarray:
        """Return ``[qdot, vdot]`` without advancing simulation time.

        This is the RS1B dynamics diagnostic interface. The learner path never
        reads the truth passive force and only receives an explicitly supplied
        revision force.
        """
        self.set_state(q, velocity)
        hidden = self._hidden(velocity) if include_hidden else 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.hinge_dof] = (
            float(torque) + hidden + float(residual_force)
        )
        self.mujoco.mj_forward(self.model, self.data)
        return np.asarray(
            [float(self.data.qvel[self.hinge_dof]), float(self.data.qacc[self.hinge_dof])],
            dtype=np.float64,
        )

    def rollout_trace(
        self,
        q: float,
        velocity: float,
        torque: float,
        *,
        horizon: int,
        residual_fn: Callable[[float, float], float] | None = None,
        margin: float = 0.05,
        include_hidden: bool = False,
    ) -> dict[str, Any]:
        """Roll out a state-dependent vector field and retain its support trace."""
        self.set_state(q, velocity)
        qs: list[float] = []
        velocities: list[float] = []
        support_codes: list[int] = []
        finite = True
        label = {"modeled": 0, "boundary": 1, "unsupported": 2}
        for _ in range(max(1, horizon)):
            q_now = float(self.data.qpos[self.hinge_qpos_adr])
            v_now = float(self.data.qvel[self.hinge_dof])
            revision = 0.0 if residual_fn is None else float(residual_fn(q_now, v_now))
            hidden = self._hidden(v_now) if include_hidden else 0.0
            self._freeze_robot()
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.hinge_dof] = float(torque) + hidden + revision
            self.mujoco.mj_step(self.model, self.data)
            q_next = float(self.data.qpos[self.hinge_qpos_adr])
            v_next = float(self.data.qvel[self.hinge_dof])
            support = self._validity(q_next, margin)
            qs.append(q_next)
            velocities.append(v_next)
            support_codes.append(label[support])
            if not np.isfinite(q_next) or not np.isfinite(v_next):
                finite = False
                break
        return {
            "q": np.asarray(qs, dtype=np.float64),
            "qvel": np.asarray(velocities, dtype=np.float64),
            "support_code": np.asarray(support_codes, dtype=np.int8),
            "finite": bool(finite),
            "stable": bool(finite and all(code == 0 for code in support_codes)),
        }

    def forecast_pair(
        self,
        q: float,
        velocity: float,
        torque: float,
        *,
        horizon: int,
        margin: float = 0.05,
    ) -> dict[str, float | bool]:
        """Integrate truth (with α) vs nominal / oracle-drag models from one IC."""

        def _run(force_mode: str) -> tuple[float, float, bool]:
            self.set_state(q, velocity)
            valid = True
            for _ in range(max(1, horizon)):
                v = float(self.data.qvel[self.hinge_dof])
                if force_mode == "truth":
                    extra = self._hidden(v)
                elif force_mode == "oracle":
                    extra = self._hidden(v)  # same structural force, known to model
                else:  # nominal
                    extra = 0.0
                self._freeze_robot()
                self.data.qfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.hinge_dof] = float(torque) + extra
                self.mujoco.mj_step(self.model, self.data)
                q_now = float(self.data.qpos[self.hinge_qpos_adr])
                support = self._validity(q_now, margin)
                valid = valid and support == "modeled"
                if not np.isfinite(q_now) or not np.isfinite(
                    float(self.data.qvel[self.hinge_dof])
                ):
                    return q_now, float(self.data.qvel[self.hinge_dof]), False
            return (
                float(self.data.qpos[self.hinge_qpos_adr]),
                float(self.data.qvel[self.hinge_dof]),
                bool(valid),
            )

        qt, vt, valid_t = _run("truth")
        qn, vn, valid_n = _run("nominal")
        qo, vo, valid_o = _run("oracle")
        scale = np.asarray([1.0, 2.0], dtype=np.float64)
        err_n = np.asarray([(qn - qt) / scale[0], (vn - vt) / scale[1]])
        err_o = np.asarray([(qo - qt) / scale[0], (vo - vt) / scale[1]])
        return {
            "rmse_nominal": float(np.sqrt(np.mean(err_n * err_n))),
            "rmse_oracle": float(np.sqrt(np.mean(err_o * err_o))),
            "valid_truth": bool(valid_t),
            "valid_nominal": bool(valid_n),
            "valid_oracle": bool(valid_o),
        }

    def rollout_probe(
        self, torques: np.ndarray, margin: float = 0.05
    ) -> dict[str, np.ndarray]:
        if self.hi > self.lo:
            self.data.qpos[self.hinge_qpos_adr] = 0.5 * (self.lo + self.hi)
        self.data.qvel[:] = 0.0
        self._freeze_robot()
        self.mujoco.mj_forward(self.model, self.data)
        records: dict[str, list[Any]] = {
            "q": [],
            "qvel": [],
            "qacc": [],
            "tau_command": [],
            "tau_hidden": [],
            "residual": [],
            "validity": [],
            "support_code": [],
        }
        code = {"modeled": 1.0, "boundary": 0.5, "unsupported": 0.0}
        label = {"modeled": 0, "boundary": 1, "unsupported": 2}
        for tau in torques:
            self._freeze_robot()
            v = float(self.data.qvel[self.hinge_dof])
            hidden = self._hidden(v)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.hinge_dof] = float(tau) + hidden
            self.mujoco.mj_step(self.model, self.data)
            residual = self._learner_residual(tau)
            q = float(self.data.qpos[self.hinge_qpos_adr])
            support = self._validity(q, margin)
            records["q"].append(q)
            records["qvel"].append(float(self.data.qvel[self.hinge_dof]))
            records["qacc"].append(float(self.data.qacc[self.hinge_dof]))
            records["tau_command"].append(float(tau))
            records["tau_hidden"].append(hidden)
            records["residual"].append(residual)
            records["validity"].append(code[support])
            records["support_code"].append(label[support])
        return {key: np.asarray(val, dtype=np.float64) for key, val in records.items()}

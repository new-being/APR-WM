"""RoboTwin-X0 P0 host: 1-DoF prismatic drawer ``rtwx_drawer1.v1``.

Oracle state only. Not an official RoboTwin task. No RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HOST_PLANT_ID = "rtwx_drawer1.v1"
DT = 1.0 / 250.0
MACRO = 4  # 0.016 s learner step


@dataclass(frozen=True)
class DrawerPhi:
    mass: float
    damping: float
    friction: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.mass, self.damping, self.friction], dtype=np.float64)


class DrawerBackend:
    def __init__(self, phi: DrawerPhi) -> None:
        try:
            import sapien
        except ModuleNotFoundError as exc:
            raise RuntimeError("SAPIEN required (RoboTwin env)") from exc
        self.sapien = sapien
        self.phi = phi
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(DT)
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_name("rail")
        root.add_box_collision(half_size=(0.25, 0.04, 0.04), density=800.0)
        slider = builder.create_link_builder(root)
        slider.set_name("drawer")
        slider.set_joint_name("slide")
        density = max(float(phi.mass) / (0.20 * 0.16 * 0.08 * 8.0), 50.0)
        slider.add_box_collision(
            pose=sapien.Pose((0.10, 0.0, 0.0)),
            half_size=(0.10, 0.08, 0.04),
            density=density,
        )
        slider.set_joint_properties(
            "prismatic",
            ((0.0, 0.25),),
            sapien.Pose((0.0, 0.0, 0.0)),
            sapien.Pose((0.0, 0.0, 0.0)),
            damping=0.0,
            friction=0.0,
        )
        self.actor = builder.build(fix_root_link=True)
        self.actor.set_sleep_threshold(0.0)
        for joint in self.actor.get_active_joints():
            joint.set_friction(float(phi.friction))
        for link in self.actor.get_links():
            link.set_linear_damping(0.0)
            link.set_angular_damping(0.0)

    def reset(self, q: float, qd: float) -> None:
        self.actor.set_qpos((float(q),))
        self.actor.set_qvel((float(qd),))
        self.actor.set_qacc((0.0,))
        self.actor.set_qf((0.0,))

    def step_force(self, force: float) -> tuple[float, float]:
        for _ in range(MACRO):
            qd = float(self.actor.get_qvel()[0])
            passive = self.actor.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            applied = float(force) - float(self.phi.damping) * qd
            self.actor.set_qf(passive + applied)
            self.scene.step()
        return float(self.actor.get_qpos()[0]), float(self.actor.get_qvel()[0])

    def rollout(self, q0: float, qd0: float, forces: np.ndarray) -> dict[str, np.ndarray]:
        self.reset(q0, qd0)
        n = int(forces.shape[0])
        q = np.zeros(n, dtype=np.float64)
        qd = np.zeros(n, dtype=np.float64)
        for i, f in enumerate(forces):
            q[i], qd[i] = float(self.actor.get_qpos()[0]), float(self.actor.get_qvel()[0])
            self.step_force(float(f))
        return {"q": q, "qd": qd, "u": np.asarray(forces, dtype=np.float64)}

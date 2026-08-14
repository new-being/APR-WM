from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import WorldConfig
from .graph import FRICTION, MASS, MATERIAL, POS, RADIUS, VEL, interaction_features, scatter_edge_messages


@dataclass
class PhysicsOutput:
    acceleration: torch.Tensor
    base_acceleration: torch.Tensor
    edge_acceleration: torch.Tensor
    contact_mask: torch.Tensor
    complex_mask: torch.Tensor


class SyntheticWorld:
    """Small deterministic 2-D disk world used to isolate the V0 hypothesis.

    The explicit expert knows external force, drag, and linear spring-damper
    contact. The ground-truth simulator additionally contains nonlinear normal
    response and velocity-dependent tangential sticking for material=1 pairs.
    """

    def __init__(self, config: WorldConfig):
        self.config = config

    def physics_acceleration(self, state: torch.Tensor, action: torch.Tensor) -> PhysicsOutput:
        cfg = self.config
        if action.shape != (*state.shape[:2], 2):
            raise ValueError("action must have shape [B, N, 2]")
        features, receiver, _, contact, material_pair = interaction_features(state)
        rel_pos = features[..., 0:2]
        rel_vel = features[..., 2:4]
        distance = features[..., 4:5]
        penetration = features[..., 5:6]
        normal = rel_pos / distance
        relative_normal_speed = (rel_vel * normal).sum(dim=-1, keepdim=True)
        normal_magnitude = (
            cfg.spring_k * penetration - cfg.damping * relative_normal_speed
        ).clamp_min(0.0) * contact.unsqueeze(-1)
        edge_force = normal_magnitude * normal
        receiver_mass = state[:, receiver, MASS].unsqueeze(-1)
        edge_acceleration = edge_force / receiver_mass
        pair_acceleration = scatter_edge_messages(edge_acceleration, receiver, state.shape[1])
        mass = state[..., MASS].unsqueeze(-1)
        friction = state[..., FRICTION].unsqueeze(-1)
        base_acceleration = action / mass - cfg.ground_drag * (0.25 + friction) * state[..., VEL]
        return PhysicsOutput(
            acceleration=base_acceleration + pair_acceleration,
            base_acceleration=base_acceleration,
            edge_acceleration=edge_acceleration,
            contact_mask=contact,
            complex_mask=contact * material_pair,
        )

    def true_acceleration(self, state: torch.Tensor, action: torch.Tensor) -> PhysicsOutput:
        cfg = self.config
        physics = self.physics_acceleration(state, action)
        features, receiver, _, contact, material_pair = interaction_features(state)
        rel_pos = features[..., 0:2]
        rel_vel = features[..., 2:4]
        distance = features[..., 4:5]
        penetration = features[..., 5:6]
        friction_mean = features[..., 9:10]
        normal = rel_pos / distance
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        normal_speed = (rel_vel * normal).sum(dim=-1, keepdim=True)
        tangent_speed = (rel_vel * tangent).sum(dim=-1, keepdim=True)
        complex_contact = (contact * material_pair).unsqueeze(-1)

        nonlinear_normal = (
            cfg.nonlinear_k
            * penetration.square()
            * (1.0 + 0.35 * torch.tanh(-normal_speed / cfg.tangent_speed))
            * normal
        )
        base_normal_force = (
            cfg.spring_k * penetration - cfg.damping * normal_speed
        ).clamp_min(0.0)
        tangential_sticking = (
            -cfg.nonlinear_friction
            * friction_mean
            * base_normal_force
            * torch.tanh(tangent_speed / cfg.tangent_speed)
            * tangent
        )
        extra_force = complex_contact * (nonlinear_normal + tangential_sticking)
        receiver_mass = state[:, receiver, MASS].unsqueeze(-1)
        extra_edge_acceleration = extra_force / receiver_mass
        extra_acceleration = scatter_edge_messages(extra_edge_acceleration, receiver, state.shape[1])
        return PhysicsOutput(
            acceleration=physics.acceleration + extra_acceleration,
            base_acceleration=physics.base_acceleration,
            edge_acceleration=physics.edge_acceleration + extra_edge_acceleration,
            contact_mask=physics.contact_mask,
            complex_mask=physics.complex_mask,
        )

    def integrate(self, state: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
        next_state = state.clone()
        next_velocity = state[..., VEL] + self.config.dt * acceleration
        next_state[..., VEL] = next_velocity
        next_state[..., POS] = state[..., POS] + self.config.dt * next_velocity
        return next_state

    def step(self, state: torch.Tensor, action: torch.Tensor, *, ground_truth: bool = True) -> torch.Tensor:
        output = self.true_acceleration(state, action) if ground_truth else self.physics_acceleration(state, action)
        return self.integrate(state, output.acceleration)

    def sample_initial_state(
        self,
        batch_size: int,
        n_objects: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        ood: bool = False,
    ) -> torch.Tensor:
        cfg = self.config

        def uniform(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
            return low + (high - low) * torch.rand(shape, generator=generator, device=device)

        radius = uniform((batch_size, n_objects), cfg.radius_min, cfg.radius_max)
        position = uniform(
            (batch_size, n_objects, 2), -cfg.position_extent, cfg.position_extent
        )
        # Make useful interaction data common without forcing every pair to touch.
        for obj in range(1, n_objects, 2):
            angle = uniform((batch_size,), 0.0, 2.0 * torch.pi)
            direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
            overlap = uniform((batch_size,), 0.005, 0.055)
            paired_position = position[:, obj - 1] + direction * (
                radius[:, obj - 1] + radius[:, obj] - overlap
            ).unsqueeze(-1)
            use_contact = torch.rand((batch_size,), generator=generator, device=device) < cfg.contact_fraction
            position[:, obj] = torch.where(use_contact.unsqueeze(-1), paired_position, position[:, obj])

        velocity = uniform(
            (batch_size, n_objects, 2), -cfg.velocity_scale, cfg.velocity_scale
        )
        if ood:
            mass = uniform((batch_size, n_objects), cfg.ood_mass_min, cfg.ood_mass_max)
            friction = uniform(
                (batch_size, n_objects), cfg.ood_friction_min, cfg.ood_friction_max
            )
            complex_probability = cfg.ood_complex_probability
        else:
            mass = uniform((batch_size, n_objects), cfg.train_mass_min, cfg.train_mass_max)
            friction = uniform(
                (batch_size, n_objects), cfg.train_friction_min, cfg.train_friction_max
            )
            complex_probability = cfg.train_complex_probability
        material = (
            torch.rand((batch_size, n_objects), generator=generator, device=device)
            < complex_probability
        ).to(position.dtype)
        return torch.cat(
            (
                position,
                velocity,
                radius.unsqueeze(-1),
                mass.unsqueeze(-1),
                friction.unsqueeze(-1),
                material.unsqueeze(-1),
            ),
            dim=-1,
        )

    def sample_actions(
        self,
        batch_size: int,
        n_objects: int,
        horizon: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        ood: bool = False,
    ) -> torch.Tensor:
        scale = self.config.ood_action_scale if ood else self.config.train_action_scale
        smoothing = self.config.action_smoothing
        actions: list[torch.Tensor] = []
        current = torch.zeros((batch_size, n_objects, 2), device=device)
        for _ in range(horizon):
            noise = torch.randn(
                (batch_size, n_objects, 2), generator=generator, device=device
            ) * scale
            current = smoothing * current + (1.0 - smoothing) * noise
            actions.append(current)
        return torch.stack(actions, dim=1)

    def sample_trajectories(
        self,
        batch_size: int,
        n_objects: int,
        horizon: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        ood: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial = self.sample_initial_state(
            batch_size, n_objects, generator=generator, device=device, ood=ood
        )
        actions = self.sample_actions(
            batch_size,
            n_objects,
            horizon,
            generator=generator,
            device=device,
            ood=ood,
        )
        states = [initial]
        current = initial
        for step in range(horizon):
            current = self.step(current, actions[:, step], ground_truth=True)
            states.append(current)
        return torch.stack(states, dim=1), actions

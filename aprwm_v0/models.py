from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig, WorldConfig
from .graph import ACTION_DIM, EDGE_DIM, NODE_DIM, POS, VEL, interaction_features, scatter_edge_messages
from .simulator import SyntheticWorld


MODEL_NAMES = ("physics", "neural", "residual", "adaptive")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError("MLP depth must be at least 2")
        dimensions = [input_dim, *([hidden_dim] * (depth - 1)), output_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                layers.extend((nn.SiLU(), nn.LayerNorm(out_dim)))
        self.network = nn.Sequential(*layers)
        self.dimensions = dimensions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def approximate_flops(self) -> int:
        return sum(2 * left * right for left, right in zip(self.dimensions[:-1], self.dimensions[1:]))


def integrate_state(state: torch.Tensor, acceleration: torch.Tensor, dt: float) -> torch.Tensor:
    next_velocity = state[..., VEL] + dt * acceleration
    next_position = state[..., POS] + dt * next_velocity
    return torch.cat((next_position, next_velocity, state[..., 4:]), dim=-1)


def make_edge_inputs(
    state: torch.Tensor, action: torch.Tensor, physics_edge: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_features, receiver, sender, contact, material_pair = interaction_features(state)
    pieces = (
        state[:, receiver],
        state[:, sender],
        edge_features,
        action[:, receiver],
        action[:, sender],
    )
    inputs = torch.cat((*pieces, *((physics_edge,) if physics_edge is not None else ())), dim=-1)
    return inputs, receiver, sender, contact, material_pair


class DynamicsModel(nn.Module):
    model_name: str
    is_adaptive: bool = False

    def __init__(self, world_config: WorldConfig, model_config: ModelConfig):
        super().__init__()
        self.world_config = world_config
        self.model_config = model_config
        self.world = SyntheticWorld(world_config)

    def compute_profile(self) -> dict[str, int]:
        return {"router_flops_per_edge": 0, "expert_flops_per_edge": 0}


class PhysicsModel(DynamicsModel):
    model_name = "physics"

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, *, hard_routing: bool = False
    ) -> dict[str, torch.Tensor]:
        del hard_routing
        physics = self.world.physics_acceleration(state, action)
        zeros = torch.zeros_like(physics.contact_mask)
        return {
            "next_state": integrate_state(state, physics.acceleration, self.world_config.dt),
            "acceleration": physics.acceleration,
            "gate_probability": zeros,
            "gate_used": zeros,
            "residual_messages": torch.zeros_like(physics.edge_acceleration),
            "residual_applied": torch.zeros_like(physics.edge_acceleration),
            "contact_mask": physics.contact_mask,
            "router_logits": zeros,
        }


class NeuralGNN(DynamicsModel):
    model_name = "neural"

    def __init__(self, world_config: WorldConfig, model_config: ModelConfig):
        super().__init__(world_config, model_config)
        cfg = model_config
        node_input_dim = NODE_DIM + ACTION_DIM
        edge_input_dim = 2 * NODE_DIM + EDGE_DIM + 2 * ACTION_DIM
        self.node_expert = MLP(node_input_dim, cfg.hidden_dim, 2, cfg.depth)
        self.edge_expert = MLP(edge_input_dim, cfg.hidden_dim, 2, cfg.depth)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, *, hard_routing: bool = False
    ) -> dict[str, torch.Tensor]:
        del hard_routing
        edge_inputs, receiver, _, contact, _ = make_edge_inputs(state, action)
        edge_messages = self.edge_expert(edge_inputs) * contact.unsqueeze(-1)
        node_acceleration = self.node_expert(torch.cat((state, action), dim=-1))
        acceleration = node_acceleration + scatter_edge_messages(
            edge_messages, receiver, state.shape[1]
        )
        return {
            "next_state": integrate_state(state, acceleration, self.world_config.dt),
            "acceleration": acceleration,
            "gate_probability": contact,
            "gate_used": contact,
            "residual_messages": edge_messages,
            "residual_applied": edge_messages,
            "contact_mask": contact,
            "router_logits": torch.zeros_like(contact),
        }

    def compute_profile(self) -> dict[str, int]:
        return {
            "router_flops_per_edge": 0,
            "expert_flops_per_edge": self.edge_expert.approximate_flops(),
            "node_flops": self.node_expert.approximate_flops(),
        }


class ResidualHybrid(DynamicsModel):
    model_name = "residual"

    def __init__(self, world_config: WorldConfig, model_config: ModelConfig):
        super().__init__(world_config, model_config)
        cfg = model_config
        edge_input_dim = 2 * NODE_DIM + EDGE_DIM + 2 * ACTION_DIM + 2
        self.residual_expert = MLP(edge_input_dim, cfg.hidden_dim, 2, cfg.depth)

    def residual_and_gate(
        self, edge_inputs: torch.Tensor, contact: torch.Tensor, hard_routing: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del hard_routing
        residual = self.residual_expert(edge_inputs) * contact.unsqueeze(-1)
        return residual, contact, contact, torch.zeros_like(contact)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, *, hard_routing: bool = False
    ) -> dict[str, torch.Tensor]:
        physics = self.world.physics_acceleration(state, action)
        edge_inputs, receiver, _, contact, _ = make_edge_inputs(
            state, action, physics.edge_acceleration
        )
        residual, gate_probability, gate_used, router_logits = self.residual_and_gate(
            edge_inputs, contact, hard_routing
        )
        residual_applied = residual * gate_used.unsqueeze(-1)
        acceleration = physics.acceleration + scatter_edge_messages(
            residual_applied, receiver, state.shape[1]
        )
        return {
            "next_state": integrate_state(state, acceleration, self.world_config.dt),
            "acceleration": acceleration,
            "gate_probability": gate_probability,
            "gate_used": gate_used,
            "residual_messages": residual,
            "residual_applied": residual_applied,
            "contact_mask": contact,
            "router_logits": router_logits,
        }

    def compute_profile(self) -> dict[str, int]:
        return {
            "router_flops_per_edge": 0,
            "expert_flops_per_edge": self.residual_expert.approximate_flops(),
        }


class AdaptiveHybrid(ResidualHybrid):
    model_name = "adaptive"
    is_adaptive = True

    def __init__(self, world_config: WorldConfig, model_config: ModelConfig):
        super().__init__(world_config, model_config)
        self.force_open = False
        edge_input_dim = 2 * NODE_DIM + EDGE_DIM + 2 * ACTION_DIM + 2
        self.router = MLP(edge_input_dim, model_config.router_hidden_dim, 1, 2)
        final_linear = next(
            layer for layer in reversed(self.router.network) if isinstance(layer, nn.Linear)
        )
        # The expert is forced open by the trainer during warm-up. Keep router
        # logits at the high-gradient midpoint while it starts distillation.
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def budget_route(
        self, gate_probability: torch.Tensor, contact: torch.Tensor
    ) -> torch.Tensor:
        """Select the highest-scoring contact edges under a per-world budget."""

        contact_mask = contact.bool()
        contact_count = contact_mask.sum(dim=-1)
        budget_count = torch.ceil(
            contact_count * self.model_config.residual_budget_fraction
        ).long()
        scores = gate_probability.masked_fill(~contact_mask, float("-inf"))
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1)
        return (rank < budget_count.unsqueeze(-1)) & contact_mask

    def residual_and_gate(
        self, edge_inputs: torch.Tensor, contact: torch.Tensor, hard_routing: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(edge_inputs).squeeze(-1) / self.model_config.gate_temperature
        gate_probability = torch.sigmoid(logits) * contact
        if self.training:
            residual = self.residual_expert(edge_inputs) * contact.unsqueeze(-1)
            if self.force_open:
                gate_used = contact
            elif self.model_config.training_routing == "soft":
                gate_used = gate_probability
            else:
                hard_gate = self.budget_route(gate_probability, contact).to(edge_inputs.dtype)
                # Binary forward pass with a sigmoid surrogate gradient. This
                # prevents the residual from cancelling an arbitrarily tiny
                # soft gate by growing its own output.
                gate_used = hard_gate + gate_probability - gate_probability.detach()
            return residual, gate_probability, gate_used, logits
        if not hard_routing:
            residual = self.residual_expert(edge_inputs) * contact.unsqueeze(-1)
            return residual, gate_probability, gate_probability, logits

        route = self.budget_route(gate_probability, contact)
        flat_inputs = edge_inputs.reshape(-1, edge_inputs.shape[-1])
        flat_route = route.reshape(-1)
        flat_residual = edge_inputs.new_zeros((flat_inputs.shape[0], 2))
        if bool(flat_route.any()):
            selected_residual = self.residual_expert(flat_inputs[flat_route])
            flat_residual[flat_route] = selected_residual.to(flat_residual.dtype)
        residual = flat_residual.reshape(*edge_inputs.shape[:-1], 2)
        return residual, gate_probability, route.to(edge_inputs.dtype), logits

    def compute_profile(self) -> dict[str, int]:
        profile = super().compute_profile()
        profile["router_flops_per_edge"] = self.router.approximate_flops()
        return profile


def build_model(
    name: str, world_config: WorldConfig, model_config: ModelConfig
) -> DynamicsModel:
    models: dict[str, type[DynamicsModel]] = {
        "physics": PhysicsModel,
        "neural": NeuralGNN,
        "residual": ResidualHybrid,
        "adaptive": AdaptiveHybrid,
    }
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    return models[name](world_config, model_config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

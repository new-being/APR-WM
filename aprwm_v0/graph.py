from __future__ import annotations

from functools import lru_cache

import torch


NODE_DIM = 8
ACTION_DIM = 2
EDGE_DIM = 11

POS = slice(0, 2)
VEL = slice(2, 4)
RADIUS = 4
MASS = 5
FRICTION = 6
MATERIAL = 7


@lru_cache(maxsize=32)
def _cpu_edge_indices(n_objects: int) -> tuple[torch.Tensor, torch.Tensor]:
    receivers: list[int] = []
    senders: list[int] = []
    for receiver in range(n_objects):
        for sender in range(n_objects):
            if receiver != sender:
                receivers.append(receiver)
                senders.append(sender)
    return torch.tensor(receivers, dtype=torch.long), torch.tensor(senders, dtype=torch.long)


def edge_indices(n_objects: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    receiver, sender = _cpu_edge_indices(n_objects)
    return receiver.to(device=device), sender.to(device=device)


def interaction_features(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense directed-edge features and useful graph metadata.

    Edge direction is sender -> receiver. Relative position points from sender
    to receiver, so a positive normal force acts directly on the receiver.
    """

    if state.ndim != 3 or state.shape[-1] != NODE_DIM:
        raise ValueError(f"state must have shape [B, N, {NODE_DIM}], got {tuple(state.shape)}")
    n_objects = state.shape[1]
    receiver, sender = edge_indices(n_objects, state.device)
    recv = state[:, receiver]
    send = state[:, sender]
    rel_pos = recv[..., POS] - send[..., POS]
    rel_vel = recv[..., VEL] - send[..., VEL]
    distance = torch.linalg.vector_norm(rel_pos, dim=-1, keepdim=True).clamp_min(1.0e-6)
    radius_sum = (recv[..., RADIUS] + send[..., RADIUS]).unsqueeze(-1)
    penetration = (radius_sum - distance).clamp_min(0.0)
    contact = (penetration > 0).to(state.dtype)
    inv_mass = torch.stack(
        (recv[..., MASS].reciprocal(), send[..., MASS].reciprocal()), dim=-1
    )
    friction_mean = (0.5 * (recv[..., FRICTION] + send[..., FRICTION])).unsqueeze(-1)
    material_pair = torch.maximum(recv[..., MATERIAL], send[..., MATERIAL]).unsqueeze(-1)
    features = torch.cat(
        (
            rel_pos,
            rel_vel,
            distance,
            penetration,
            contact,
            inv_mass,
            friction_mean,
            material_pair,
        ),
        dim=-1,
    )
    return features, receiver, sender, contact.squeeze(-1), material_pair.squeeze(-1)


def scatter_edge_messages(
    messages: torch.Tensor, receiver: torch.Tensor, n_objects: int
) -> torch.Tensor:
    batch = messages.shape[0]
    output = messages.new_zeros((batch, n_objects, messages.shape[-1]))
    output.index_add_(1, receiver, messages)
    return output


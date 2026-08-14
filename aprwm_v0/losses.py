from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import TrainConfig


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0)
    if values.ndim > mask.ndim or values.shape[-1] != mask.shape[-1]:
        denominator = denominator * values.shape[-1]
    return (values * mask).sum() / denominator


def dynamics_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    train_config: TrainConfig,
    *,
    step: int,
    is_adaptive: bool,
    use_residual_regularizer: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = output["next_state"]
    position_loss = (prediction[..., 0:2] - target[..., 0:2]).square().mean()
    velocity_loss = (prediction[..., 2:4] - target[..., 2:4]).square().mean()
    prediction_loss = position_loss + velocity_loss

    contact = output["contact_mask"]
    residual_norm = torch.linalg.vector_norm(output["residual_applied"], dim=-1)
    residual_cost = masked_mean(residual_norm, contact)
    gate_cost = masked_mean(output["gate_probability"], contact)
    warmup = max(1, train_config.regularizer_warmup_steps)
    residual_regularizer_scale = min(1.0, step / warmup)
    gate_regularizer_scale = max(0.0, min(1.0, (step - warmup) / warmup))
    total = prediction_loss
    if use_residual_regularizer:
        total = (
            total
            + residual_regularizer_scale * train_config.lambda_residual * residual_cost
        )
    if is_adaptive:
        total = total + gate_regularizer_scale * train_config.lambda_gate * gate_cost
        residual_need = torch.linalg.vector_norm(
            output["residual_messages"].detach(), dim=-1
        )
        router_target = torch.sigmoid(
            (residual_need - train_config.residual_need_threshold)
            / train_config.residual_need_temperature
        )
        router_bce = F.binary_cross_entropy_with_logits(
            output["router_logits"], router_target, reduction="none"
        )
        router_loss = masked_mean(router_bce, contact)
        router_start = 0.5 * warmup
        router_scale = max(0.0, min(1.0, (step - router_start) / max(router_start, 1.0)))
        total = total + router_scale * train_config.lambda_router * router_loss
    else:
        router_target = torch.zeros_like(contact)
        router_loss = prediction_loss.new_zeros(())
        router_scale = 0.0

    metrics = {
        "loss": float(total.detach()),
        "prediction": float(prediction_loss.detach()),
        "position": float(position_loss.detach()),
        "velocity": float(velocity_loss.detach()),
        "residual_cost": float(residual_cost.detach()),
        "gate_cost": float(gate_cost.detach()),
        "router_loss": float(router_loss.detach()),
        "router_target_fraction": float(masked_mean(router_target, contact).detach()),
        "residual_regularizer_scale": residual_regularizer_scale,
        "gate_regularizer_scale": gate_regularizer_scale,
        "router_scale": router_scale,
    }
    return total, metrics

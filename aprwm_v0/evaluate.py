from __future__ import annotations

import contextlib
import math

import torch

from .config import ExperimentConfig
from .models import DynamicsModel
from .simulator import SyntheticWorld


def autocast_context(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.float().flatten()
    labels = labels.bool().flatten()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    if float(scores.max() - scores.min()) < 1.0e-8:
        return 0.5
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_positive = torch.cumsum(sorted_labels.float(), dim=0)
    false_positive = torch.cumsum((~sorted_labels).float(), dim=0)
    true_positive = torch.cat((true_positive.new_zeros(1), true_positive / positives))
    false_positive = torch.cat((false_positive.new_zeros(1), false_positive / negatives))
    return float(torch.trapezoid(true_positive, false_positive))


def _rollout(
    model: DynamicsModel,
    initial: torch.Tensor,
    actions: torch.Tensor,
    *,
    hard_routing: bool,
    amp: bool,
) -> torch.Tensor:
    current = initial
    predictions = [current]
    for step in range(actions.shape[1]):
        with autocast_context(initial.device, amp):
            output = model(current, actions[:, step], hard_routing=hard_routing)
        current = output["next_state"].float()
        predictions.append(current)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def evaluate_split(
    model: DynamicsModel,
    world: SyntheticWorld,
    config: ExperimentConfig,
    device: torch.device,
    *,
    ood: bool,
) -> dict[str, float | int | None]:
    model.eval()
    eval_cfg = config.eval
    n_objects = eval_cfg.ood_n_objects if ood else eval_cfg.id_n_objects
    split_seed = config.seed + (20_000 if ood else 10_000)
    generator = torch.Generator(device=device)
    generator.manual_seed(split_seed)

    pos_sq = 0.0
    vel_sq = 0.0
    state_count = 0
    rollout_sq = 0.0
    rollout_count = 0
    hard_rollout_sq = 0.0
    hard_rollout_count = 0
    stable_sequences = 0
    hard_stable_sequences = 0
    sequence_count = 0
    contact_count = 0.0
    graph_edge_count = 0.0
    gate_sum = 0.0
    active_sum = 0.0
    residual_sum = 0.0
    complex_gate_sum = 0.0
    complex_count = 0.0
    simple_gate_sum = 0.0
    simple_count = 0.0
    auc_scores: list[torch.Tensor] = []
    auc_labels: list[torch.Tensor] = []

    for _ in range(eval_cfg.batches):
        states, actions = world.sample_trajectories(
            eval_cfg.batch_size,
            n_objects,
            eval_cfg.horizon,
            generator=generator,
            device=device,
            ood=ood,
        )
        batch, horizon, nodes, _ = actions.shape
        flat_state = states[:, :-1].reshape(batch * horizon, nodes, -1)
        flat_action = actions.reshape(batch * horizon, nodes, -1)
        flat_target = states[:, 1:].reshape(batch * horizon, nodes, -1)
        with autocast_context(device, config.train.amp):
            output = model(flat_state, flat_action, hard_routing=False)
            hard_output = (
                model(flat_state, flat_action, hard_routing=True)
                if model.is_adaptive
                else output
            )
        prediction = output["next_state"].float()
        pos_sq += float((prediction[..., 0:2] - flat_target[..., 0:2]).square().sum())
        vel_sq += float((prediction[..., 2:4] - flat_target[..., 2:4]).square().sum())
        state_count += prediction.shape[0] * prediction.shape[1] * 2

        truth = world.true_acceleration(flat_state, flat_action)
        contact = truth.contact_mask.bool()
        complex_mask = truth.complex_mask.bool() & contact
        simple_mask = (~complex_mask) & contact
        gates = output["gate_probability"].float()
        active = hard_output["gate_used"].bool()
        residual_norm = torch.linalg.vector_norm(
            hard_output["residual_applied"].float(), dim=-1
        )
        contact_count += float(contact.sum())
        graph_edge_count += float(contact.numel())
        gate_sum += float(gates[contact].sum())
        active_sum += float(active[contact].sum())
        residual_sum += float(residual_norm[contact].sum())
        complex_count += float(complex_mask.sum())
        simple_count += float(simple_mask.sum())
        complex_gate_sum += float(gates[complex_mask].sum())
        simple_gate_sum += float(gates[simple_mask].sum())
        if bool(contact.any()):
            auc_scores.append(gates[contact].detach().cpu())
            auc_labels.append(complex_mask[contact].detach().cpu())

        predicted_rollout = _rollout(
            model,
            states[:, 0],
            actions,
            hard_routing=False,
            amp=config.train.amp,
        )
        rollout_error = predicted_rollout[:, 1:, :, 0:4] - states[:, 1:, :, 0:4]
        rollout_sq += float(rollout_error.square().sum())
        rollout_count += rollout_error.numel()
        finite = torch.isfinite(predicted_rollout).all(dim=(1, 2, 3))
        speed = torch.linalg.vector_norm(predicted_rollout[..., 2:4], dim=-1)
        stable = finite & (speed.amax(dim=(1, 2)) < eval_cfg.stability_speed)
        stable_sequences += int(stable.sum())

        if model.is_adaptive:
            hard_rollout = _rollout(
                model,
                states[:, 0],
                actions,
                hard_routing=True,
                amp=config.train.amp,
            )
        else:
            hard_rollout = predicted_rollout
        hard_error = hard_rollout[:, 1:, :, 0:4] - states[:, 1:, :, 0:4]
        hard_rollout_sq += float(hard_error.square().sum())
        hard_rollout_count += hard_error.numel()
        hard_finite = torch.isfinite(hard_rollout).all(dim=(1, 2, 3))
        hard_speed = torch.linalg.vector_norm(hard_rollout[..., 2:4], dim=-1)
        hard_stable = hard_finite & (hard_speed.amax(dim=(1, 2)) < eval_cfg.stability_speed)
        hard_stable_sequences += int(hard_stable.sum())
        sequence_count += batch

    contact_denominator = max(contact_count, 1.0)
    all_scores = torch.cat(auc_scores) if auc_scores else torch.empty(0)
    all_labels = torch.cat(auc_labels) if auc_labels else torch.empty(0, dtype=torch.bool)
    profile = model.compute_profile()
    expert_fraction = active_sum / contact_denominator
    graph_expert_fraction = active_sum / max(graph_edge_count, 1.0)
    router_flops = profile.get("router_flops_per_edge", 0)
    expert_flops = profile.get("expert_flops_per_edge", 0)
    if model.is_adaptive:
        routed_flops = router_flops + graph_expert_fraction * expert_flops
    else:
        routed_flops = expert_flops
    always_on_flops = router_flops + expert_flops
    compute_fraction = routed_flops / always_on_flops if always_on_flops else 0.0
    return {
        "n_objects": n_objects,
        "one_step_position_rmse": math.sqrt(pos_sq / max(state_count, 1)),
        "one_step_velocity_rmse": math.sqrt(vel_sq / max(state_count, 1)),
        "rollout_state_rmse": math.sqrt(rollout_sq / max(rollout_count, 1)),
        "hard_rollout_state_rmse": math.sqrt(
            hard_rollout_sq / max(hard_rollout_count, 1)
        ),
        "rollout_stable_fraction": stable_sequences / max(sequence_count, 1),
        "hard_rollout_stable_fraction": hard_stable_sequences / max(sequence_count, 1),
        "mean_contact_gate": gate_sum / contact_denominator,
        "mean_complex_gate": complex_gate_sum / max(complex_count, 1.0),
        "mean_simple_gate": simple_gate_sum / max(simple_count, 1.0),
        "routing_auroc": binary_auroc(all_scores, all_labels) if all_scores.numel() else None,
        "active_residual_edge_fraction": expert_fraction,
        "active_graph_edge_fraction": graph_expert_fraction,
        "mean_applied_residual": residual_sum / contact_denominator,
        "learned_compute_fraction": compute_fraction,
        "contact_edge_samples": int(contact_count),
    }


@torch.no_grad()
def evaluate_model(
    model: DynamicsModel,
    world: SyntheticWorld,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "model": model.model_name,
        "id": evaluate_split(model, world, config, device, ood=False),
        "ood": evaluate_split(model, world, config, device, ood=True),
        "compute_profile": model.compute_profile(),
    }

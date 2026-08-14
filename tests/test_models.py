import pytest
import torch

from aprwm_v0.config import ModelConfig, TrainConfig, WorldConfig
from aprwm_v0.losses import dynamics_loss
from aprwm_v0.models import MODEL_NAMES, build_model
from aprwm_v0.simulator import SyntheticWorld


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_shapes_and_finite(name):
    world_config = WorldConfig()
    model_config = ModelConfig(hidden_dim=16, router_hidden_dim=8, depth=2)
    world = SyntheticWorld(world_config)
    generator = torch.Generator().manual_seed(2)
    state = world.sample_initial_state(3, 4, generator=generator, device=torch.device("cpu"))
    action = torch.randn(3, 4, 2, generator=generator)
    model = build_model(name, world_config, model_config)
    output = model(state, action)
    assert output["next_state"].shape == state.shape
    assert output["gate_probability"].shape == (3, 12)
    assert torch.isfinite(output["next_state"]).all()


def test_adaptive_hard_routing_is_finite():
    world_config = WorldConfig()
    model_config = ModelConfig(hidden_dim=16, router_hidden_dim=8, depth=2)
    world = SyntheticWorld(world_config)
    generator = torch.Generator().manual_seed(9)
    state = world.sample_initial_state(3, 4, generator=generator, device=torch.device("cpu"))
    action = torch.randn(3, 4, 2, generator=generator)
    model = build_model("adaptive", world_config, model_config)
    model.eval()
    seen_batch_sizes = []
    handle = model.residual_expert.register_forward_pre_hook(
        lambda _module, inputs: seen_batch_sizes.append(inputs[0].shape[0])
    )
    output = model(state, action, hard_routing=True)
    handle.remove()
    assert output["next_state"].shape == state.shape
    assert set(output["gate_used"].unique().tolist()) <= {0.0, 1.0}
    contact_count = output["contact_mask"].sum(dim=-1)
    expected = torch.ceil(contact_count * model_config.residual_budget_fraction)
    torch.testing.assert_close(output["gate_used"].sum(dim=-1), expected)
    assert seen_batch_sizes == [int(expected.sum())]
    assert torch.isfinite(output["next_state"]).all()


@pytest.mark.parametrize("name", ("neural", "residual", "adaptive"))
def test_trainable_models_backpropagate(name):
    world_config = WorldConfig()
    model_config = ModelConfig(hidden_dim=16, router_hidden_dim=8, depth=2)
    train_config = TrainConfig(regularizer_warmup_steps=1)
    world = SyntheticWorld(world_config)
    generator = torch.Generator().manual_seed(4)
    states, actions = world.sample_trajectories(
        2, 4, 1, generator=generator, device=torch.device("cpu")
    )
    model = build_model(name, world_config, model_config)
    output = model(states[:, 0], actions[:, 0])
    loss, _ = dynamics_loss(
        output,
        states[:, 1],
        train_config,
        step=1,
        is_adaptive=model.is_adaptive,
        use_residual_regularizer=name in ("residual", "adaptive"),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if name == "adaptive":
        assert any(parameter.grad is not None for parameter in model.router.parameters())

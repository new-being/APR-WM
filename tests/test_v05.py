import torch

from aprwm_v0.config import ModelConfig, WorldConfig
from aprwm_v0.models import build_model
from aprwm_v0.simulator import SyntheticWorld
from aprwm_v0.v05 import _global_budget_route, routed_step


def test_global_budget_route_selects_exact_fraction_of_contacts():
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.7]])
    contact = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    route = _global_budget_route(scores, contact, 0.5)
    assert int(route.sum()) == 2
    assert not bool((route & ~contact.bool()).any())


def test_zero_budget_routes_nothing():
    scores = torch.ones(2, 3)
    contact = torch.ones(2, 3)
    assert int(_global_budget_route(scores, contact, 0.0).sum()) == 0


def test_routed_step_executes_only_selected_edges():
    world_config = WorldConfig()
    model_config = ModelConfig(hidden_dim=16, router_hidden_dim=8, depth=2)
    model = build_model("adaptive", world_config, model_config)
    model.eval()
    world = SyntheticWorld(world_config)
    generator = torch.Generator().manual_seed(12)
    state = world.sample_initial_state(8, 4, generator=generator, device=torch.device("cpu"))
    action = torch.randn(8, 4, 2, generator=generator)
    route_generator = torch.Generator().manual_seed(13)
    seen = []
    handle = model.residual_expert.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].shape[0])
    )
    next_state, active, contacts, graph_edges = routed_step(
        model,
        state,
        action,
        "learned",
        0.5,
        random_generator=route_generator,
    )
    handle.remove()
    assert next_state.shape == state.shape
    assert active == round(0.5 * contacts)
    assert seen == ([active] if active else [])
    assert graph_edges == 8 * 4 * 3


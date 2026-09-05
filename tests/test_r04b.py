import torch

from aprwm_v0.r04b import V6R04BConfig, _history_design, _memoryless_design


def test_r04b_seeds_are_independent_of_previous_bridge_evaluations():
    old = {13, 23, 33, 1001, 1011, 1021, 1031, 1041, 2001, 2011, 2021, 2031, 2041}
    assert set(V6R04BConfig().seeds).isdisjoint(old)


def test_history_design_adds_only_explicit_lagged_actions():
    state = torch.tensor([[[0.5, 0.2, -0.3]]])
    history = torch.tensor([[[0.4, -0.2, 0.1]]])
    memoryless = _memoryless_design(state)
    temporal = _history_design(state, history)
    assert memoryless.shape[-1] == 12
    assert temporal.shape[-1] == 15
    torch.testing.assert_close(temporal[..., :12], memoryless)
    torch.testing.assert_close(temporal[..., 12:], history / 0.8)

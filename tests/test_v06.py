import torch

from aprwm_v0.v06 import Expert, RegimeData, Router, _make_execution


def test_density_and_necessity_are_independent():
    data = RegimeData(torch.device("cpu"), 5)
    _, active, needed, residual = data.sample(50_000, 0.8, 0.1)
    assert abs(float(active.float().mean()) - 0.8) < 0.015
    conditional_necessity = float(needed.sum() / active.sum())
    assert abs(conditional_necessity - 0.1) < 0.015
    assert torch.all(residual[~needed] == 0)


def test_same_feature_regime_not_identity_controls_need():
    data = RegimeData(torch.device("cpu"), 8)
    features, active, needed, _ = data.sample(20_000, 1.0, 0.2)
    assert bool(active.all())
    assert bool(needed.any()) and bool((~needed).any())
    assert features.shape == (20_000, 12)


def test_hard_sparse_executes_fewer_rows_than_contact_baseline():
    data = RegimeData(torch.device("cpu"), 10)
    features, active, _, _ = data.sample(2_000, 0.8, 0.1)
    expert = Expert("1x")
    router = Router("linear")
    with torch.no_grad():
        router.network.weight.zero_()
        router.network.bias.fill_(-1.0)
    contact_output = _make_execution(
        expert, router, features, active, "contact_packed", 0.1
    )()
    sparse_output = _make_execution(
        expert, router, features, active, "hard_sparse", 0.1
    )()
    assert contact_output.shape[0] == int(active.sum())
    assert sparse_output.shape[0] == round(0.1 * int(active.sum()))


def test_expert_scale_labels_track_actual_flops():
    base = Expert("1x").flops_per_edge
    for name, expected in (("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)):
        ratio = Expert(name).flops_per_edge / base
        assert abs(ratio - expected) / expected < 0.03

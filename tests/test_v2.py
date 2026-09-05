import torch

from aprwm_v0.v2 import JointBelief, V2Config, force, residual_decision, sample_action


def test_nonlinear_models_are_nearly_equivalent_passively_but_separate_under_probe():
    config = V2Config()
    theta = torch.tensor(((35.0, 1.2), (35.0, 1.2)))
    adequate = torch.tensor((0.0, 0.0))
    bad = torch.tensor((1.0, 1.0))
    passive = torch.tensor(((0.02, 0.0), (0.02, 0.0)))
    probe = torch.tensor(((0.075, 0.0), (0.075, 0.0)))
    passive_gap = (
        force(passive, theta, bad, config.nonlinear_alpha)
        - force(passive, theta, adequate, config.nonlinear_alpha)
    ).abs().mean()
    probe_gap = (
        force(probe, theta, bad, config.nonlinear_alpha)
        - force(probe, theta, adequate, config.nonlinear_alpha)
    ).abs().mean()
    assert probe_gap > 40 * passive_gap


def test_joint_ig_prefers_amplitude_when_model_class_is_uncertain():
    config = V2Config(episodes=64)
    device = torch.device("cpu")
    belief = JointBelief(config, device)
    weights = belief.prior(64)
    generator = torch.Generator().manual_seed(3)
    candidates = torch.stack(
        (
            sample_action("passive", 64, generator, device),
            sample_action("velocity_probe", 64, generator, device),
            sample_action("amplitude_probe", 64, generator, device),
        ),
        dim=1,
    )
    joint = belief.expected_information_gain(weights, candidates, target="joint")
    assert float((joint.argmax(dim=-1) == 2).float().mean()) > 0.9


def test_belief_update_normalizes_particle_weights():
    config = V2Config(episodes=8)
    belief = JointBelief(config, torch.device("cpu"))
    weights = belief.prior(8)
    features = torch.tensor(((0.02, 0.0),)).expand(8, -1)
    observations = torch.ones(8)
    updated = belief.update(weights, features, observations)
    torch.testing.assert_close(updated.exp().sum(dim=-1), torch.ones(8))


def test_force_shapes_for_single_and_multiple_queries():
    theta = torch.ones((5, 2))
    model = torch.zeros(5)
    assert force(torch.ones((5, 2)), theta, model, 220.0).shape == (5,)
    assert force(torch.ones((5, 7, 2)), theta, model, 220.0).shape == (5, 7)


def test_compute_decision_requires_positive_net_utility_and_model_confidence():
    probability = torch.tensor((0.5, 0.95, 0.995))
    correction = torch.ones(3)
    route = residual_decision(probability, correction, 0.1, 0.99)
    assert route.tolist() == [False, False, True]


def test_exact_model_conditioning_has_finite_entropy():
    config = V2Config(episodes=4)
    belief = JointBelief(config, torch.device("cpu"))
    log_weights = belief.prior(4)
    true_model = torch.tensor((0.0, 1.0, 0.0, 1.0))
    incompatible = belief.model_bad.unsqueeze(0) != true_model.unsqueeze(1)
    log_weights = log_weights.masked_fill(incompatible, float("-inf"))
    log_weights -= torch.logsumexp(log_weights, dim=-1, keepdim=True)
    assert torch.isfinite(belief.summary(log_weights)["entropy"]).all()
    assert torch.isfinite(belief.model_entropy(log_weights)).all()

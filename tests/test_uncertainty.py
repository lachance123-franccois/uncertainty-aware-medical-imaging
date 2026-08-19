import numpy as np
import pytest
import torch

from umi.losses import (
    EvidentialLoss,
    evidence_from_logits,
    evidential_mse,
    kl_dirichlet_uniform,
)
from umi.uncertainty import decompose_uncertainty, dirichlet_uncertainty, entropy


def test_entropy_extremes():
    uniform = torch.full((4, 5), 0.2)
    onehot = torch.eye(5)[[0, 1, 2, 3]]
    assert entropy(uniform).allclose(torch.ones(4), atol=1e-5)  # normalised -> 1
    assert entropy(onehot).abs().max() < 1e-5


def test_identical_samples_have_zero_epistemic():
    p = torch.softmax(torch.randn(1, 32, 3), dim=-1).repeat(10, 1, 1)
    out = decompose_uncertainty(p)
    assert out["epistemic"].abs().max() < 1e-6
    assert torch.allclose(out["total"], out["aleatoric"], atol=1e-6)


def test_disagreeing_samples_have_positive_epistemic():
    a = torch.tensor([[[0.99, 0.01]]]).repeat(5, 8, 1)
    b = torch.tensor([[[0.01, 0.99]]]).repeat(5, 8, 1)
    samples = torch.cat([a, b], dim=0)
    out = decompose_uncertainty(samples)
    assert out["epistemic"].min() > 0.9
    assert out["aleatoric"].max() < 0.1


def test_total_equals_aleatoric_plus_epistemic():
    p = torch.softmax(torch.randn(12, 64, 4), dim=-1)
    out = decompose_uncertainty(p)
    assert torch.allclose(out["total"], out["aleatoric"] + out["epistemic"], atol=1e-5)


def test_decompose_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decompose_uncertainty(torch.rand(8, 3))


def test_vacuity_is_one_without_evidence():
    """alpha = 1 (zero evidence) is the 'I have never seen anything like this' state."""
    alpha = torch.ones(4, 5)
    out = dirichlet_uncertainty(alpha)
    assert out["vacuity"].allclose(torch.ones(4), atol=1e-6)
    assert out["total"].allclose(torch.ones(4), atol=1e-5)


def test_vacuity_decreases_with_evidence():
    low = dirichlet_uncertainty(torch.tensor([[1.5, 1.1]]))["vacuity"]
    high = dirichlet_uncertainty(torch.tensor([[50.0, 1.1]]))["vacuity"]
    assert high < low


def test_dirichlet_matches_sampling_estimate():
    torch.manual_seed(0)
    alpha = torch.tensor([[4.0, 2.0, 1.5]])
    samples = torch.distributions.Dirichlet(alpha).sample((20000,))  # (S, 1, K)
    mc = decompose_uncertainty(samples)
    closed = dirichlet_uncertainty(alpha)
    assert closed["total"].item() == pytest.approx(mc["total"].item(), abs=0.01)
    assert closed["aleatoric"].item() == pytest.approx(mc["aleatoric"].item(), abs=0.01)


def test_kl_of_uniform_dirichlet_is_zero():
    assert kl_dirichlet_uniform(torch.ones(3, 6)).abs().max() < 1e-5


def test_kl_is_positive_for_peaked_dirichlet():
    assert (kl_dirichlet_uniform(torch.tensor([[20.0, 1.0, 1.0]])) > 0).all()


def test_evidential_mse_rewards_correct_evidence():
    y = torch.tensor([[1.0, 0.0]])
    good = evidential_mse(torch.tensor([[20.0, 1.0]]), y)
    bad = evidential_mse(torch.tensor([[1.0, 20.0]]), y)
    unsure = evidential_mse(torch.tensor([[1.0, 1.0]]), y)
    assert good < unsure < bad


def test_evidence_activation_is_non_negative():
    for act in ("softplus", "exp", "relu"):
        e = evidence_from_logits(torch.randn(100, 4) * 5, act)
        assert (e >= 0).all()


def test_kl_annealing_schedule():
    loss = EvidentialLoss(n_classes=3, annealing_epochs=10, lambda_max=1.0)
    loss.set_epoch(0)
    assert loss.kl_weight == 0.0
    loss.set_epoch(5)
    assert loss.kl_weight == pytest.approx(0.5)
    loss.set_epoch(50)
    assert loss.kl_weight == 1.0


def test_evidential_loss_backward():
    loss_fn = EvidentialLoss(n_classes=3, annealing_epochs=1)
    loss_fn.set_epoch(1)
    logits = torch.randn(16, 3, requires_grad=True)
    loss = loss_fn(logits, torch.randint(0, 3, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_evidential_training_increases_true_class_evidence():
    torch.manual_seed(0)
    x = torch.randn(256, 8)
    w = torch.randn(8, 2)
    y = (x @ w).argmax(1)
    model = torch.nn.Linear(8, 2)
    loss_fn = EvidentialLoss(2, annealing_epochs=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(60):
        loss_fn.set_epoch(epoch)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    alpha = evidence_from_logits(model(x)) + 1.0
    acc = (alpha.argmax(1) == y).float().mean()
    assert acc > 0.85
    assert dirichlet_uncertainty(alpha)["vacuity"].mean() < 0.95


def test_uncertainty_is_higher_on_random_noise_than_on_learned_data():
    torch.manual_seed(0)
    x = torch.randn(256, 8)
    y = (x @ torch.randn(8, 2)).argmax(1)
    model = torch.nn.Linear(8, 2)
    loss_fn = EvidentialLoss(2, annealing_epochs=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(80):
        loss_fn.set_epoch(epoch)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    v_in = dirichlet_uncertainty(evidence_from_logits(model(x)) + 1.0)["vacuity"].mean()
    v_out = dirichlet_uncertainty(
        evidence_from_logits(model(torch.zeros(256, 8))) + 1.0
    )["vacuity"].mean()
    assert v_out > v_in  # zero input carries no evidence at all


def test_numpy_interop():
    p = torch.softmax(torch.randn(5, 10, 3), -1)
    out = decompose_uncertainty(p)
    assert np.isfinite(out["total"].numpy()).all()

import numpy as np
import pytest
import torch

from umi.calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)

rng = np.random.default_rng(0)


def make_calibrated(n=20000, k=2):
    conf = rng.uniform(1.0 / k, 1.0, size=n)
    correct = (rng.uniform(size=n) < conf).astype(float)
    probs = np.zeros((n, k))
    probs[:, 0] = conf
    probs[:, 1:] = ((1 - conf) / (k - 1))[:, None]
    labels = np.where(correct == 1, 0, 1)
    return probs, labels, conf, correct


def test_perfectly_calibrated_has_near_zero_ece():
    _, _, conf, correct = make_calibrated()
    assert expected_calibration_error(conf, correct, n_bins=15) < 0.02


def test_overconfident_model_has_large_ece():
    n = 5000
    conf = np.full(n, 0.99)
    correct = (rng.uniform(size=n) < 0.60).astype(float)  # only 60% right
    ece = expected_calibration_error(conf, correct)
    assert ece == pytest.approx(0.39, abs=0.03)


def test_ece_is_bounded_and_nonnegative():
    for _ in range(10):
        conf = rng.uniform(0, 1, 500)
        correct = rng.integers(0, 2, 500).astype(float)
        assert 0.0 <= expected_calibration_error(conf, correct) <= 1.0


def test_mce_at_least_ece():
    conf = rng.uniform(0.5, 1.0, 2000)
    correct = (rng.uniform(size=2000) < 0.7).astype(float)
    assert maximum_calibration_error(conf, correct) >= expected_calibration_error(conf, correct)


def test_reliability_curve_bins_sum_to_n():
    conf = rng.uniform(0, 1, 1000)
    correct = rng.integers(0, 2, 1000).astype(float)
    for adaptive in (False, True):
        curve = reliability_curve(conf, correct, n_bins=10, adaptive=adaptive)
        assert curve.bin_count.sum() == 1000


def test_adaptive_bins_are_balanced():
    """Equal-mass binning must not leave nearly-empty bins on skewed confidence."""
    conf = np.clip(rng.beta(8, 1, 5000), 0, 1)  # piled up near 1.0
    correct = (rng.uniform(size=5000) < conf).astype(float)
    curve = reliability_curve(conf, correct, n_bins=10, adaptive=True)
    counts = curve.bin_count[curve.bin_count > 0]
    assert counts.min() > 0.5 * counts.mean()


def test_brier_and_nll_are_minimised_by_truth():
    labels = rng.integers(0, 3, 1000)
    onehot = np.eye(3)[labels]
    noisy = 0.6 * onehot + 0.4 / 3
    assert brier_score(onehot, labels) < brier_score(noisy, labels)
    assert negative_log_likelihood(onehot, labels) < negative_log_likelihood(noisy, labels)


def test_temperature_recovers_known_scaling():
    torch.manual_seed(0)
    n, k = 4000, 4
    labels = torch.randint(0, k, (n,))
    logits = torch.randn(n, k)
    logits[torch.arange(n), labels] += 2.0

    t_ref = fit_temperature(logits, labels)
    t_scaled = fit_temperature(logits * 3.0, labels)
    # Scaling the logits by c must scale the optimal temperature by c.
    assert t_scaled == pytest.approx(3.0 * t_ref, rel=0.05)


def test_temperature_reduces_ece_of_overconfident_model():
    torch.manual_seed(0)
    n, k = 4000, 3
    labels = torch.randint(0, k, (n,))
    logits = torch.randn(n, k)
    logits[torch.arange(n), labels] += 1.0
    logits = logits * 5.0  # deliberately over-confident

    before = calibration_report(torch.softmax(logits, -1).numpy(), labels.numpy())["ece"]
    t = fit_temperature(logits, labels)
    after = calibration_report(torch.softmax(logits / t, -1).numpy(), labels.numpy())["ece"]
    assert after < before


def test_calibration_report_keys():
    probs, labels, _, _ = make_calibrated(2000)
    report = calibration_report(probs, labels)
    for key in ("accuracy", "ece", "ece_adaptive", "mce", "brier", "nll", "overconfidence"):
        assert key in report and np.isfinite(report[key])

import numpy as np
import pytest
import torch

from umi.corruptions import CORRUPTIONS, apply_corruption
from umi.data import get_dataloaders, get_datasets, make_shifted_testset
from umi.maps import occlusion_uncertainty_map, pixelwise_uncertainty
from umi.methods import build_predictor
from umi.metrics import (
    auroc,
    misclassification_auroc,
    ood_detection_auroc,
    risk_coverage_curve,
)
from umi.models import build_model, enable_mc_dropout
from umi.train import TrainConfig, train_single


@pytest.fixture(scope="module")
def tiny_setup():
    datasets, meta = get_datasets("synthetic", n_train_synthetic=400, seed=0)
    loaders = get_dataloaders(datasets, batch_size=64)
    cfg = TrainConfig(
        dataset="synthetic", epochs=2, n_classes=meta.n_classes,
        in_channels=meta.in_channels, width=8, p_drop=0.3,
    )
    model, _ = train_single(loaders, cfg, "baseline", device="cpu", verbose=False)
    return datasets, loaders, meta, cfg, model



def test_enable_mc_dropout_activates_only_dropout():
    model = build_model("smallcnn", 1, 2, p_drop=0.2)
    n = enable_mc_dropout(model)
    assert n > 0
    dropouts = [m for m in model.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))]
    assert all(m.training for m in dropouts)
    norms = [m for m in model.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert all(not m.training for m in norms)


def test_mc_dropout_requires_dropout():
    with pytest.raises(RuntimeError):
        enable_mc_dropout(build_model("smallcnn", 1, 2, p_drop=0.0))


def test_mc_dropout_produces_varying_predictions():
    torch.manual_seed(0)
    model = build_model("smallcnn", 1, 2, p_drop=0.5)
    pred = build_predictor("mc_dropout", model, n_samples=15, device="cpu")
    x = torch.randn(4, 1, 28, 28)
    samples = pred._forward_samples(x)
    assert samples.shape == (15, 4, 2)
    assert samples.std(0).max() > 1e-4  # the passes really do differ


def test_baseline_has_exactly_zero_epistemic(tiny_setup):
    _, loaders, _, _, model = tiny_setup
    out = build_predictor("baseline", model, device="cpu").predict(loaders["test"])
    assert np.allclose(out.epistemic, 0.0, atol=1e-6)
    assert np.allclose(out.total, out.aleatoric, atol=1e-6)


def test_all_predictors_agree_on_shapes(tiny_setup):
    datasets, loaders, meta, cfg, model = tiny_setup
    models = [model, build_model("smallcnn", 1, 2, width=8, p_drop=0.3)]
    n = len(datasets["test"])
    for name, m in [
        ("baseline", model),
        ("mc_dropout", model),
        ("evidential", model),
        ("deep_ensemble", models),
    ]:
        out = build_predictor(name, m, device="cpu", n_samples=5).predict(loaders["test"])
        assert out.probs.shape == (n, meta.n_classes)
        assert np.allclose(out.probs.sum(1), 1.0, atol=1e-5)
        assert out.total.shape == (n,)
        assert (out.total >= -1e-6).all() and (out.total <= 1 + 1e-6).all()
        assert (out.epistemic >= -1e-6).all()


def test_evidential_exposes_vacuity(tiny_setup):
    _, loaders, _, _, model = tiny_setup
    out = build_predictor("evidential", model, device="cpu").predict(loaders["test"])
    assert out.vacuity is not None
    assert ((out.vacuity > 0) & (out.vacuity <= 1 + 1e-6)).all()


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        build_predictor("bayes_by_backprop", build_model())


def test_ensemble_needs_two_members():
    with pytest.raises(ValueError):
        build_predictor("deep_ensemble", [build_model()])



def test_auroc_matches_known_values():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert auroc(scores, np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert auroc(scores, np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert auroc(np.ones(4), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_perfect_uncertainty_gives_auroc_one():
    correct = np.array([1, 1, 1, 0, 0, 0])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert misclassification_auroc(uncertainty, correct) == pytest.approx(1.0)


def test_risk_coverage_is_lower_with_useful_uncertainty():
    rng = np.random.default_rng(0)
    correct = (rng.uniform(size=1000) > 0.2).astype(int)
    informative = np.where(correct == 1, rng.uniform(0, 0.4, 1000), rng.uniform(0.6, 1, 1000))
    useless = rng.uniform(size=1000)
    assert (risk_coverage_curve(informative, correct).aurc
            < risk_coverage_curve(useless, correct).aurc)


def test_risk_at_full_coverage_equals_error_rate():
    correct = np.array([1, 0, 1, 1, 0])
    rc = risk_coverage_curve(np.random.rand(5), correct)
    assert rc.risk_at_coverage[1.0] == pytest.approx(0.4)


def test_excess_aurc_is_non_negative():
    rng = np.random.default_rng(1)
    correct = rng.integers(0, 2, 500)
    assert risk_coverage_curve(rng.uniform(size=500), correct).excess_aurc >= -1e-9


def test_ood_auroc_separates_shifted_data():
    rng = np.random.default_rng(0)
    assert ood_detection_auroc(rng.normal(0.2, 0.05, 500), rng.normal(0.8, 0.05, 500)) > 0.95


# --------------------------------------------------------------------------- #
def test_corruptions_stay_in_range_and_change_the_image():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.2, 0.8, (1, 28, 28)).astype(np.float32)
    for name in CORRUPTIONS:
        out = apply_corruption(img, name, severity=5, rng=rng)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert np.abs(out - img).mean() > 1e-4, name


def test_shifted_testset_keeps_labels(tiny_setup):
    datasets, *_ = tiny_setup
    shifted = make_shifted_testset(datasets["test"], "gaussian_noise", 3)
    assert np.array_equal(shifted.labels, datasets["test"].labels)
    assert len(shifted) == len(datasets["test"])


def test_unknown_corruption_raises(tiny_setup):
    datasets, *_ = tiny_setup
    with pytest.raises(ValueError):
        make_shifted_testset(datasets["test"], "cosmic_rays", 3)



def test_occlusion_map_shape_and_finiteness(tiny_setup):
    datasets, _, _, _, model = tiny_setup
    predictor = build_predictor("mc_dropout", model, device="cpu", n_samples=3)
    image, _ = datasets["test"][0]
    m = occlusion_uncertainty_map(predictor, image, patch=8, stride=6)
    assert m["delta"].shape == (28, 28)
    assert np.isfinite(m["delta"]).all()


def test_pixelwise_uncertainty_for_segmentation():
    probs = torch.softmax(torch.randn(6, 2, 3, 16, 16), dim=2)
    maps = pixelwise_uncertainty(probs)
    assert maps["total"].shape == (2, 16, 16)
    assert np.allclose(maps["total"], maps["aleatoric"] + maps["epistemic"], atol=1e-5)


def test_training_is_reproducible_given_a_seed():
    datasets, meta = get_datasets("synthetic", n_train_synthetic=200, seed=3)
    loaders = get_dataloaders(datasets, 64)
    cfg = TrainConfig(epochs=1, n_classes=2, in_channels=1, width=8)
    a, _ = train_single(loaders, cfg, seed=7, verbose=False)
    b, _ = train_single(loaders, cfg, seed=7, verbose=False)
    x = torch.randn(4, 1, 28, 28)
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


def test_ensemble_members_differ():
    datasets, _ = get_datasets("synthetic", n_train_synthetic=200, seed=3)
    loaders = get_dataloaders(datasets, 64)
    cfg = TrainConfig(epochs=1, n_classes=2, in_channels=1, width=8)
    a, _ = train_single(loaders, cfg, seed=1, verbose=False)
    b, _ = train_single(loaders, cfg, seed=2, verbose=False)
    x = torch.randn(8, 1, 28, 28)
    a.eval(), b.eval()
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x), atol=1e-4)

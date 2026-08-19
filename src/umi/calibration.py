r"""Calibration: is the reported confidence *true*?

A model is **calibrated** if, among all the cases it calls with 80% confidence,
it is right about 80% of the time. Accuracy alone cannot detect miscalibration:
a model can be 95% accurate and still claim 99.9% confidence on every case,
which is exactly the behaviour that makes a clinical decision-support tool
dangerous -- the clinician has no way to know which 5% to double-check.

Metrics implemented
-------------------
* **ECE** (Expected Calibration Error), Guo et al. 2017 -- confidence binned
  into ``M`` equal-width bins, then

  .. math:: \mathrm{ECE} = \sum_m \frac{|B_m|}{n}\,
            \bigl|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\bigr|

* **Adaptive ECE** -- equal-*mass* bins. Preferred when confidences pile up near
  1.0 (the usual case for over-confident deep nets), where equal-width bins
  leave most bins nearly empty and understate the error.
* **MCE** -- the worst bin. The number a safety reviewer cares about.
* **Brier score**, **NLL** -- proper scoring rules; they penalise miscalibration
  and inaccuracy jointly and cannot be gamed by a constant-confidence trick.
* **Temperature scaling** -- a one-parameter post-hoc fix fitted on the
  validation split. Included as the "cheap competitor": if a single scalar
  divided into the logits matches a 5-model ensemble on ECE, that is a result
  worth reporting, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


@dataclass
class ReliabilityCurve:

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_count: np.ndarray

    @property
    def nonempty(self) -> np.ndarray:
        return self.bin_count > 0

    @property
    def gaps(self) -> np.ndarray:
        return self.bin_accuracy - self.bin_confidence


def _as_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def reliability_curve(
    confidences, correct, n_bins: int = 15, adaptive: bool = False
) -> ReliabilityCurve:
    conf = _as_np(confidences).astype(np.float64).ravel()
    corr = _as_np(correct).astype(np.float64).ravel()

    if adaptive:  
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(conf, quantiles))
        edges[0], edges[-1] = min(edges[0], 0.0), max(edges[-1], 1.0)
    else:  
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.digitize(conf, edges[1:-1], right=False)
    n_actual = len(edges) - 1
    acc = np.zeros(n_actual)
    avg_conf = np.zeros(n_actual)
    count = np.zeros(n_actual)

    for b in range(n_actual):
        mask = idx == b
        count[b] = mask.sum()
        if count[b] > 0:
            acc[b] = corr[mask].mean()
            avg_conf[b] = conf[mask].mean()

    return ReliabilityCurve(
        bin_edges=edges,
        bin_centers=0.5 * (edges[:-1] + edges[1:]),
        bin_accuracy=acc,
        bin_confidence=avg_conf,
        bin_count=count,
    )


def expected_calibration_error(
    confidences, correct, n_bins: int = 15, adaptive: bool = False, norm: str = "l1"
) -> float:
    curve = reliability_curve(confidences, correct, n_bins, adaptive)
    w = curve.bin_count / max(curve.bin_count.sum(), 1)
    gap = np.abs(curve.gaps)
    if norm == "l2":
        return float(np.sqrt(np.sum(w * gap**2)))
    return float(np.sum(w * gap))


def maximum_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    curve = reliability_curve(confidences, correct, n_bins)
    gaps = np.abs(curve.gaps)[curve.nonempty]
    return float(gaps.max()) if gaps.size else 0.0


def static_calibration_error(probs, labels, n_bins: int = 15) -> float:

    probs, labels = _as_np(probs), _as_np(labels)
    k = probs.shape[1]
    return float(
        np.mean(
            [
                expected_calibration_error(probs[:, c], (labels == c).astype(float), n_bins)
                for c in range(k)
            ]
        )
    )


def brier_score(probs, labels) -> float:
    probs, labels = _as_np(probs), _as_np(labels)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs, labels) -> float:
    probs, labels = _as_np(probs), _as_np(labels)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def overconfidence(confidences, correct) -> float:
    return float(_as_np(confidences).mean() - _as_np(correct).mean())


def calibration_report(probs, labels, n_bins: int = 15) -> dict[str, float]:
    probs, labels = _as_np(probs), _as_np(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct, n_bins),
        "ece_adaptive": expected_calibration_error(conf, correct, n_bins, adaptive=True),
        "mce": maximum_calibration_error(conf, correct, n_bins),
        "classwise_ece": static_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "mean_confidence": float(conf.mean()),
        "overconfidence": overconfidence(conf, correct),
    }

@torch.no_grad()
def collect_logits(
    model: torch.nn.Module, loader, device="cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    model = model.to(device).eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100, lr: float = 0.1
) -> float:
    logits, labels = logits.detach().float(), labels.detach().long()

    with torch.no_grad():
        grid = torch.linspace(-3.0, 3.0, 121)  # T in [~0.05, ~20]
        losses = torch.stack(
            [F.cross_entropy(logits / lt.exp(), labels) for lt in grid]
        )
        init = grid[int(losses.argmin())]

    log_t = init.clone().reshape(1).requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(log_t.exp().item())
    return float(np.clip(t, 1e-2, 100.0))

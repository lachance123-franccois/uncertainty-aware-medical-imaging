r"""Uncertainty measures and their decomposition.

Every method in this repo eventually produces either

* a set of ``S`` probability vectors per image (MC Dropout, Deep Ensembles), or
* a Dirichlet ``alpha`` per image (Evidential),

and both are reduced here to the *same* three numbers so the comparison is
apples-to-apples:

======================  ======================================================
Quantity                Meaning
======================  ======================================================
**Total** uncertainty   :math:`H[\bar p]`, entropy of the mean prediction.
**Aleatoric**           :math:`\mathbb{E}_\theta H[p_\theta]`, irreducible
                        noise in the data (an ambiguous lesion stays ambiguous
                        no matter how much data you collect).
**Epistemic**           Mutual information :math:`I[y;\theta] =
                        H[\bar p] - \mathbb{E}_\theta H[p_\theta]`, the model's
                        ignorance -- disagreement between plausible models. It
                        *is* reducible: more data of that kind removes it.
======================  ======================================================

The distinction matters clinically. High *aleatoric* uncertainty means "this
image cannot be called with confidence, get another modality". High *epistemic*
uncertainty means "this case is unlike anything I was trained on, do not trust
me at all" -- the out-of-distribution alarm.

All entropies are in nats and, where noted, normalised by ``log K`` so that
values are comparable across datasets with different numbers of classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

EPS = 1e-12


@dataclass
class UQOutput:

    probs: np.ndarray  # (N, K) mean predictive distribution
    labels: np.ndarray | None = None  # (N,)
    total: np.ndarray = field(default_factory=lambda: np.array([]))  # (N,)
    aleatoric: np.ndarray = field(default_factory=lambda: np.array([]))
    epistemic: np.ndarray = field(default_factory=lambda: np.array([]))
    vacuity: np.ndarray | None = None  # evidential only
    samples: np.ndarray | None = None  # (S, N, K) if available
    method: str = "unknown"
    inference_seconds: float = float("nan")

    @property
    def preds(self) -> np.ndarray:
        return self.probs.argmax(axis=1)

    @property
    def confidence(self) -> np.ndarray:
        return self.probs.max(axis=1)

    @property
    def correct(self) -> np.ndarray:
        if self.labels is None:
            raise ValueError("labels are required to compute correctness")
        return (self.preds == self.labels).astype(np.int64)

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    def uncertainty(self, kind: str = "total") -> np.ndarray:
        mapping = {
            "total": self.total,
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
            "entropy": self.total,
            "confidence": 1.0 - self.confidence,
        }
        if kind == "vacuity":
            if self.vacuity is None:
                raise ValueError("vacuity is only defined for the evidential method")
            return self.vacuity
        if kind not in mapping:
            raise ValueError(f"Unknown uncertainty kind '{kind}'")
        return mapping[kind]

def entropy(probs: torch.Tensor, dim: int = -1, normalize: bool = True) -> torch.Tensor:
    h = -(probs.clamp_min(EPS) * probs.clamp_min(EPS).log()).sum(dim=dim)
    if normalize:
        k = probs.shape[dim]
        h = h / float(np.log(k))
    return h


def decompose_uncertainty(
    probs_samples: torch.Tensor, normalize: bool = True
) -> dict[str, torch.Tensor]:
    if probs_samples.ndim != 3:
        raise ValueError(f"expected (S, N, K), got {tuple(probs_samples.shape)}")
    mean_probs = probs_samples.mean(dim=0)  # (N, K)
    total = entropy(mean_probs, normalize=normalize)  # H[E p]
    aleatoric = entropy(probs_samples, normalize=normalize).mean(dim=0)  # E H[p]
    epistemic = (total - aleatoric).clamp_min(0.0)  # mutual information
    return {
        "mean_probs": mean_probs,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        # unbiased=False keeps this defined for single-sample (deterministic) methods
        "variance": probs_samples.var(dim=0, unbiased=False).sum(dim=-1),
    }


def dirichlet_uncertainty(alpha: torch.Tensor, normalize: bool = True) -> dict[str, torch.Tensor]:
    r"""Uncertainty measures of a Dirichlet ``(N, K)``.

    * ``vacuity`` :math:`= K/S` -- total lack of evidence, the evidential
      analogue of epistemic uncertainty and the cheapest OOD score available
      (one forward pass, no sampling).
    * ``aleatoric`` :math:`= \mathbb{E}_{p\sim Dir(\alpha)} H[p]`, available in
      closed form via digamma functions.
    * ``epistemic`` :math:`= H[\bar p] - \mathbb{E} H[p]`, the same mutual
      information as for the sampling methods, so the numbers are comparable.
    """
    k = alpha.shape[-1]
    s = alpha.sum(dim=-1, keepdim=True)
    mean_probs = alpha / s

    total = entropy(mean_probs, normalize=normalize)
    # E[H(p)] = -sum_k (a_k/S) * (psi(a_k + 1) - psi(S + 1))
    expected_h = -(
        (alpha / s) * (torch.digamma(alpha + 1.0) - torch.digamma(s + 1.0))
    ).sum(dim=-1)
    if normalize:
        expected_h = expected_h / float(np.log(k))
    epistemic = (total - expected_h).clamp_min(0.0)
    vacuity = (k / s).squeeze(-1)

    return {
        "mean_probs": mean_probs,
        "total": total,
        "aleatoric": expected_h,
        "epistemic": epistemic,
        "vacuity": vacuity,
        "evidence": (alpha - 1.0).sum(dim=-1),
    }


def to_numpy(d: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in d.items()}

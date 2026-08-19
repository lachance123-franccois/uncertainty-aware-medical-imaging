r"""Losses.

Cross-entropy for the baseline / MC Dropout / ensemble members, and the
Evidential Deep Learning objectives of Sensoy et al. (NeurIPS 2018),
*Evidential Deep Learning to Quantify Classification Uncertainty*.

Evidential formulation
----------------------
The network outputs non-negative **evidence** :math:`e_k = \mathrm{softplus}(z_k)`
which parameterises a Dirichlet distribution over the class probabilities:

.. math::
    \alpha_k = e_k + 1, \qquad S = \sum_k \alpha_k, \qquad
    \hat p_k = \alpha_k / S, \qquad u = K / S

Instead of a point estimate of the class probabilities, the model predicts a
*distribution over* them. ``u`` is the **vacuity**: when the model has seen no
evidence at all (:math:`e = 0`), :math:`S = K` and :math:`u = 1` -- "I don't
know" -- which softmax literally cannot express, since its outputs always sum
to one no matter how unfamiliar the input.

Two data-fit terms are implemented:

* ``mse`` (Eq. 5 in the paper): Bayes risk of the sum-of-squares loss, which
  decomposes into a fit term and a variance term. Empirically the most stable.
* ``digamma`` (Eq. 4): Bayes risk of the cross-entropy loss.

Both are combined with a KL term that pulls the evidence of the *wrong* classes
back to the uniform Dirichlet, i.e. it actively penalises misleading evidence.
The KL weight is annealed from 0 so the model first learns to classify and only
then learns to say "I don't know" -- annealing too fast collapses all evidence
to zero and the model becomes uniformly, uselessly uncertain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def evidence_from_logits(logits: torch.Tensor, activation: str = "softplus") -> torch.Tensor:
    if activation == "softplus":
        return F.softplus(logits)
    if activation == "exp":
        return torch.exp(torch.clamp(logits, -10, 10))
    if activation == "relu":
        return F.relu(logits)

import math

def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    s_alpha = alpha.sum(dim=-1, keepdim=True)
    log_b = (
        torch.lgamma(s_alpha.squeeze(-1))
        - math.lgamma(k)                        # scalaire Python, pas de tenseur
        - torch.lgamma(alpha).sum(dim=-1)
    )
    dg = torch.digamma(alpha) - torch.digamma(s_alpha)
    return log_b + ((alpha - 1.0) * dg).sum(dim=-1)

def evidential_mse(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    r"""Bayes risk of the Brier score under Dir(alpha): ``err + var``."""
    s = alpha.sum(dim=-1, keepdim=True)
    p = alpha / s
    err = (y_onehot - p).pow(2).sum(dim=-1)
    var = (p * (1 - p) / (s + 1)).sum(dim=-1)
    return err + var


def evidential_digamma(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    r"""Bayes risk of cross-entropy under Dir(alpha)."""
    s = alpha.sum(dim=-1, keepdim=True)
    return (y_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=-1)


class EvidentialLoss(torch.nn.Module):
 
    def __init__(
        self,
        n_classes: int,
        kind: str = "mse",
        annealing_epochs: int = 10,
        lambda_max: float = 1.0,
        evidence_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.kind = kind
        self.annealing_epochs = max(annealing_epochs, 1)
        self.lambda_max = lambda_max
        self.evidence_activation = evidence_activation
        self.register_buffer("_epoch", torch.tensor(0.0))

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(float(epoch))

    @property
    def kl_weight(self) -> float:
        return float(
            self.lambda_max * min(1.0, float(self._epoch.item()) / self.annealing_epochs)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        evidence = evidence_from_logits(logits, self.evidence_activation)
        alpha = evidence + 1.0
        y = F.one_hot(target, self.n_classes).float()

        fit = evidential_mse(alpha, y) if self.kind == "mse" else evidential_digamma(alpha, y)

        alpha_tilde = y + (1.0 - y) * alpha
        kl = kl_dirichlet_uniform(alpha_tilde)

        return (fit + self.kl_weight * kl).mean()


def build_loss(
    method: str,
    n_classes: int,
    label_smoothing: float = 0.0,
    evidential_kind: str = "mse",
    annealing_epochs: int = 10,
    lambda_max: float = 1.0,
) -> torch.nn.Module:
    if method == "evidential":
        return EvidentialLoss(
            n_classes,
            kind=evidential_kind,
            annealing_epochs=annealing_epochs,
            lambda_max=lambda_max,
        )
    return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import evidence_from_logits
from .models import enable_mc_dropout
from .uncertainty import UQOutput, decompose_uncertainty, dirichlet_uncertainty


class Predictor(ABC):
    name: str = "base"

    def __init__(self, device: torch.device | str = "cpu", temperature: float = 1.0) -> None:
        self.device = torch.device(device)
        self.temperature = float(temperature)

    @abstractmethod
    def _forward_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability samples of shape ``(S, B, K)`` for a batch."""

    def _extras(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}

    @torch.no_grad()
    def predict(self, loader: DataLoader, keep_samples: bool = False) -> UQOutput:
        all_samples: list[torch.Tensor] = []
        all_labels: list[np.ndarray] = []
        extras: dict[str, list[torch.Tensor]] = {}

        start = time.perf_counter()
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            samples = self._forward_samples(x).cpu()  # (S, B, K)
            all_samples.append(samples)
            all_labels.append(np.asarray(y))
            for k, v in self._extras(x).items():
                extras.setdefault(k, []).append(v.cpu())
        elapsed = time.perf_counter() - start

        samples = torch.cat(all_samples, dim=1)  # (S, N, K)
        labels = np.concatenate(all_labels)
        stats = decompose_uncertainty(samples)
        merged = {k: torch.cat(v, dim=0) for k, v in extras.items()}

        return UQOutput(
            probs=stats["mean_probs"].numpy(),
            labels=labels,
            total=stats["total"].numpy(),
            aleatoric=stats["aleatoric"].numpy(),
            epistemic=stats["epistemic"].numpy(),
            vacuity=merged["vacuity"].numpy() if "vacuity" in merged else None,
            samples=samples.numpy() if keep_samples else None,
            method=self.name,
            inference_seconds=elapsed,
        )

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> UQOutput:
        """Same as :meth:`predict` for a single in-memory batch (used by ``maps.py``)."""
        x = x.to(self.device)
        samples = self._forward_samples(x).cpu()
        stats = decompose_uncertainty(samples)
        extras = {k: v.cpu() for k, v in self._extras(x).items()}
        return UQOutput(
            probs=stats["mean_probs"].numpy(),
            total=stats["total"].numpy(),
            aleatoric=stats["aleatoric"].numpy(),
            epistemic=stats["epistemic"].numpy(),
            vacuity=extras["vacuity"].numpy() if "vacuity" in extras else None,
            method=self.name,
        )

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)


class DeterministicPredictor(Predictor):
    name = "baseline"

    def __init__(self, model: torch.nn.Module, device="cpu", temperature: float = 1.0):
        super().__init__(device, temperature)
        self.model = model.to(self.device).eval()

    def _forward_samples(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self._softmax(self.model(x)).unsqueeze(0)


class MCDropoutPredictor(Predictor):
    name = "mc_dropout"

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 30,
        device="cpu",
        temperature: float = 1.0,
        seed: int | None = 0,
    ):
        super().__init__(device, temperature)
        self.model = model.to(self.device)
        self.n_samples = n_samples
        self.seed = seed
        enable_mc_dropout(self.model)

    def _forward_samples(self, x: torch.Tensor) -> torch.Tensor:
        if self.seed is not None:
            torch.manual_seed(self.seed)  
        enable_mc_dropout(self.model)
        return torch.stack([self._softmax(self.model(x)) for _ in range(self.n_samples)])


class DeepEnsemblePredictor(Predictor):
    name = "deep_ensemble"

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        device="cpu",
        temperature: float = 1.0,
    ):
        super().__init__(device, temperature)
        if len(models) < 2:
            raise ValueError("A deep ensemble needs at least 2 members")
        self.models = [m.to(self.device).eval() for m in models]

    def _forward_samples(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.models:
            m.eval()
        return torch.stack([self._softmax(m(x)) for m in self.models])


class EvidentialPredictor(Predictor):

    name = "evidential"

    def __init__(
        self,
        model: torch.nn.Module,
        device="cpu",
        evidence_activation: str = "softplus",
        temperature: float = 1.0,
    ):
        super().__init__(device, temperature)
        self.model = model.to(self.device).eval()
        self.evidence_activation = evidence_activation

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()  
        logits = self.model(x)
        return evidence_from_logits(logits, self.evidence_activation) + 1.0

    def _forward_samples(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(x)
        return (alpha / alpha.sum(-1, keepdim=True)).unsqueeze(0)

    def _extras(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"vacuity": dirichlet_uncertainty(self._alpha(x))["vacuity"]}

    @torch.no_grad()
    def predict(self, loader: DataLoader, keep_samples: bool = False) -> UQOutput:
        probs, tot, ale, epi, vac, labels = [], [], [], [], [], []
        start = time.perf_counter()
        for x, y in loader:
            u = dirichlet_uncertainty(self._alpha(x.to(self.device)))
            probs.append(u["mean_probs"].cpu())
            tot.append(u["total"].cpu())
            ale.append(u["aleatoric"].cpu())
            epi.append(u["epistemic"].cpu())
            vac.append(u["vacuity"].cpu())
            labels.append(np.asarray(y))
        elapsed = time.perf_counter() - start
        return UQOutput(
            probs=torch.cat(probs).numpy(),
            labels=np.concatenate(labels),
            total=torch.cat(tot).numpy(),
            aleatoric=torch.cat(ale).numpy(),
            epistemic=torch.cat(epi).numpy(),
            vacuity=torch.cat(vac).numpy(),
            method=self.name,
            inference_seconds=elapsed,
        )

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> UQOutput:
        u = dirichlet_uncertainty(self._alpha(x.to(self.device)))
        return UQOutput(
            probs=u["mean_probs"].cpu().numpy(),
            total=u["total"].cpu().numpy(),
            aleatoric=u["aleatoric"].cpu().numpy(),
            epistemic=u["epistemic"].cpu().numpy(),
            vacuity=u["vacuity"].cpu().numpy(),
            method=self.name,
        )


def build_predictor(
    method: str,
    models: torch.nn.Module | Iterable[torch.nn.Module],
    device="cpu",
    n_samples: int = 30,
    temperature: float = 1.0,
    evidence_activation: str = "softplus",
) -> Predictor:
    if method == "deep_ensemble":
        return DeepEnsemblePredictor(list(models), device, temperature)

    model = models[0] if isinstance(models, (list, tuple)) else models
    if method == "baseline":
        return DeterministicPredictor(model, device, temperature)
    if method == "mc_dropout":
        return MCDropoutPredictor(model, n_samples, device, temperature)
    if method == "evidential":
        return EvidentialPredictor(model, device, evidence_activation)
    raise ValueError(
        f"Unknown method '{method}'. Choose from "
        "baseline | mc_dropout | deep_ensemble | evidential"
    )
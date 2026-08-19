r"""Spatial uncertainty maps: *where* in the image is the model unsure?

For a segmentation network the answer is free -- you get one predictive
distribution per pixel. For a **classification** network there is a single
distribution for the whole image, so a spatial map has to be built by
intervention. This module uses **occlusion sensitivity applied to the
uncertainty**, not to the class score:

1. Slide a patch of size ``patch`` with stride ``stride`` over the image.
2. Replace the patch content with a neutral value (image mean or blur).
3. Re-run the *full UQ predictor* and record the uncertainty of the occluded
   image.
4. ``delta_map = u(occluded) - u(original)``.

Reading the maps
* **Red / high delta** -- hiding this region makes the model unsure, so the
  region was carrying the evidence for the decision. This is the "what is the
  model looking at" map.
* **Blue / negative delta** -- hiding the region makes the model *more*
  confident: it contained contradictory or distracting signal (an artefact, a
  marker, an overlapping structure).

Cost warning: one map is ``(H/stride) x (W/stride)`` forward passes *times* the
cost of the method (x S for MC Dropout, x M for the ensemble). Evidential maps
are the cheapest by an order of magnitude -- another concrete argument in the
cost/benefit comparison.
"""

from __future__ import annotations

import numpy as np
import torch

from .methods import Predictor


def _occlude(img: np.ndarray, y: int, x: int, size: int, fill: float) -> np.ndarray:
    out = img.copy()
    out[:, y : y + size, x : x + size] = fill
    return out


def occlusion_uncertainty_map(
    predictor: Predictor,
    image: np.ndarray | torch.Tensor,
    patch: int = 8,
    stride: int = 2,
    fill: str = "mean",
    uncertainty_kind: str = "total",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:

    img = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if img.ndim == 4:
        img = img[0]
    c, h, w = img.shape
    fill_value = float(img.mean()) if fill == "mean" else float(fill) if fill != "zero" else 0.0

    base = predictor.predict_batch(torch.from_numpy(img[None]).float())
    base_u = float(base.uncertainty(uncertainty_kind)[0])
    base_conf = float(base.confidence[0])
    pred_class = int(base.preds[0])

    ys = list(range(0, max(h - patch + 1, 1), stride))
    xs = list(range(0, max(w - patch + 1, 1), stride))
    variants = np.stack([_occlude(img, y, x, patch, fill_value) for y in ys for x in xs])

    u_vals, conf_vals = [], []
    for i in range(0, len(variants), batch_size):
        batch = torch.from_numpy(variants[i : i + batch_size]).float()
        out = predictor.predict_batch(batch)
        u_vals.append(out.uncertainty(uncertainty_kind))
        conf_vals.append(out.probs[:, pred_class])
    u_vals = np.concatenate(u_vals).reshape(len(ys), len(xs))
    conf_vals = np.concatenate(conf_vals).reshape(len(ys), len(xs))

    return {
        "uncertainty": _to_pixel_grid(u_vals, ys, xs, patch, h, w),
        "delta": _to_pixel_grid(u_vals - base_u, ys, xs, patch, h, w),
        "confidence_drop": _to_pixel_grid(base_conf - conf_vals, ys, xs, patch, h, w),
        "base_uncertainty": np.float32(base_u),
        "base_confidence": np.float32(base_conf),
        "pred": np.int64(pred_class),
    }


def _to_pixel_grid(
    values: np.ndarray, ys: list[int], xs: list[int], patch: int, h: int, w: int
) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            acc[y : y + patch, x : x + patch] += values[i, j]
            cnt[y : y + patch, x : x + patch] += 1
    cnt[cnt == 0] = 1
    return (acc / cnt).astype(np.float32)


def normalize_map(m: np.ndarray, symmetric: bool = False) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if symmetric:
        scale = np.abs(m).max() + 1e-8
        return m / scale
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo + 1e-8)


def pixelwise_uncertainty(probs_samples: torch.Tensor) -> dict[str, np.ndarray]:
    from .uncertainty import decompose_uncertainty  # local import avoids a cycle

    if probs_samples.ndim != 5:
        raise ValueError(f"expected (S, B, K, H, W), got {tuple(probs_samples.shape)}")
    s, b, k, h, w = probs_samples.shape
    flat = probs_samples.permute(0, 1, 3, 4, 2).reshape(s, b * h * w, k)
    stats = decompose_uncertainty(flat)
    return {
        key: stats[key].reshape(b, h, w).cpu().numpy()
        for key in ("total", "aleatoric", "epistemic")
    }

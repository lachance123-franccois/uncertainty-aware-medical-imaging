from __future__ import annotations

import numpy as np

_SEVERITY_SCALE = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def _s(severity: int) -> float:
    return _SEVERITY_SCALE[int(np.clip(severity, 1, 5))]


def gaussian_noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img + rng.normal(0, 0.05 + 0.25 * _s(severity), img.shape), 0, 1)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(int(3 * sigma), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:

    sigma = 0.4 + 2.0 * _s(severity)
    k = _gaussian_kernel1d(sigma)
    pad = len(k) // 2
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        p = np.pad(img[c], ((0, 0), (pad, pad)), mode="reflect")
        tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, p)
        p = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
        out[c] = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, p)
    return np.clip(out, 0, 1)


def contrast(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factor = 1.0 - 0.85 * _s(severity)
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 1)


def brightness(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img + 0.5 * _s(severity), 0, 1)


def poisson_noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    photons = 300.0 * (1.0 - 0.95 * _s(severity)) + 5.0
    return np.clip(rng.poisson(img * photons) / photons, 0, 1)


def occlusion(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    c, h, w = img.shape
    size = max(int(min(h, w) * (0.15 + 0.35 * _s(severity))), 2)
    y = rng.integers(0, max(h - size, 1))
    x = rng.integers(0, max(w - size, 1))
    out = img.copy()
    out[:, y : y + size, x : x + size] = 0.0
    return out


def pixelate(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:

    c, h, w = img.shape
    factor = 1.0 - 0.75 * _s(severity)
    nh, nw = max(int(h * factor), 2), max(int(w * factor), 2)
    yi = (np.arange(nh) * h / nh).astype(int)
    xi = (np.arange(nw) * w / nw).astype(int)
    small = img[:, yi][:, :, xi]
    yb = (np.arange(h) * nh / h).astype(int)
    xb = (np.arange(w) * nw / w).astype(int)
    return np.clip(small[:, yb][:, :, xb], 0, 1)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "poisson_noise": poisson_noise,
    "blur": blur,
    "contrast": contrast,
    "brightness": brightness,
    "occlusion": occlusion,
    "pixelate": pixelate,
}


def apply_corruption(
    img: np.ndarray,
    name: str,
    severity: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    return CORRUPTIONS[name](img.astype(np.float32), severity, rng).astype(np.float32)

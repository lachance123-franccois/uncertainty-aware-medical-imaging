from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .corruptions import CORRUPTIONS, apply_corruption
from .utils import log

MEDMNIST_FLAGS = {
    "pneumoniamnist": ("PneumoniaMNIST", 2, 1),
    "breastmnist": ("BreastMNIST", 2, 1),
    "dermamnist": ("DermaMNIST", 7, 3),
    "octmnist": ("OCTMNIST", 4, 1),
    "bloodmnist": ("BloodMNIST", 8, 3),
}


@dataclass
class DataMeta:
    name: str
    n_classes: int
    in_channels: int
    image_size: int
    class_names: list[str]


class ArrayDataset(Dataset):
 
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        corruption: str | None = None,
        severity: int = 3,
        seed: int = 0,
    ) -> None:
        assert images.ndim == 4, f"expected (N,C,H,W), got {images.shape}"
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64).reshape(-1)
        self.augment = augment
        self.corruption = corruption
        self.severity = severity
        self._rng = np.random.default_rng(seed)

    def reset_rng(self, seed: int) -> None:

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx].copy()
        if self.corruption is not None:
            img = apply_corruption(img, self.corruption, self.severity, self._rng)
        if self.augment:
            if self._rng.random() < 0.5:  # horizontal flip
                img = img[:, :, ::-1].copy()
            # small random shift (+/- 2 px), zero padded
            dx, dy = self._rng.integers(-2, 3, size=2)
            img = np.roll(img, (int(dy), int(dx)), axis=(1, 2))
        return torch.from_numpy(img), int(self.labels[idx])


def make_synthetic_lesions(
    n: int,
    size: int = 28,
    seed: int = 0,
    ambiguous_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = rng.integers(0, 2, size=n)
    yy, xx = np.mgrid[0:size, 0:size]

    for i in range(n):
        # Anatomy-like smooth background: sum of a few broad Gaussians.
        bg = np.zeros((size, size), dtype=np.float32)
        for _ in range(3):
            cy, cx = rng.uniform(0, size, 2)
            s = rng.uniform(size / 3, size / 1.5)
            bg += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s**2))
        bg = 0.35 + 0.25 * bg / (bg.max() + 1e-8)

        if labels[i] == 1:
            ambiguous = rng.random() < ambiguous_fraction
            amp = rng.uniform(0.05, 0.12) if ambiguous else rng.uniform(0.35, 0.6)
            r = rng.uniform(1.5, 2.2) if ambiguous else rng.uniform(2.0, 3.5)
            cy, cx = rng.uniform(0.25 * size, 0.75 * size, 2)
            bg += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))

        bg += rng.normal(0, 0.05, size=(size, size))
        imgs[i, 0] = np.clip(bg, 0.0, 1.0)

    return imgs, labels.astype(np.int64)


def _load_medmnist(name: str, root: str, size: int) -> tuple[dict[str, tuple], DataMeta]:
    
    try:
        import medmnist  # noqa: PLC0415
        from medmnist import INFO  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "The 'medmnist' package is required for real data. "
            "Install it with `pip install medmnist`, or run with "
            "`--dataset synthetic` to use the built-in fallback."
        ) from exc

    info = INFO[name]
    cls = getattr(medmnist, info["python_class"])
    Path(root).mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "val", "test"):
        kwargs = dict(split=split, download=True, root=root)
        if size != 28:
            kwargs["size"] = size
        ds = cls(**kwargs)
        imgs = ds.imgs.astype(np.float32) / 255.0
        if imgs.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
            imgs = imgs[:, None, :, :]
        else:  # (N, H, W, C) -> (N, C, H, W)
            imgs = imgs.transpose(0, 3, 1, 2)
        splits[split] = (imgs, ds.labels.astype(np.int64).reshape(-1))

    meta = DataMeta(
        name=name,
        n_classes=len(info["label"]),
        in_channels=splits["train"][0].shape[1],
        image_size=splits["train"][0].shape[-1],
        class_names=[info["label"][str(i)] for i in range(len(info["label"]))],
    )
    return splits, meta


def _load_synthetic(size: int, n_train: int, seed: int) -> tuple[dict[str, tuple], DataMeta]:
    tr = make_synthetic_lesions(n_train, size, seed=seed)
    va = make_synthetic_lesions(max(n_train // 5, 200), size, seed=seed + 1000)
    te = make_synthetic_lesions(max(n_train // 3, 400), size, seed=seed + 2000)
    meta = DataMeta("synthetic", 2, 1, size, ["normal", "lesion"])
    return {"train": tr, "val": va, "test": te}, meta


def get_datasets(
    dataset: str = "pneumoniamnist",
    root: str = "./data",
    image_size: int = 28,
    n_train_synthetic: int = 4000,
    augment: bool = True,
    seed: int = 0,
) -> tuple[dict[str, ArrayDataset], DataMeta]:
    if dataset == "synthetic":
        splits, meta = _load_synthetic(image_size, n_train_synthetic, seed)
    elif dataset in MEDMNIST_FLAGS:
        splits, meta = _load_medmnist(dataset, root, image_size)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from "
            f"{['synthetic', *MEDMNIST_FLAGS]}"
        )

    out = {
        "train": ArrayDataset(*splits["train"], augment=augment, seed=seed),
        "val": ArrayDataset(*splits["val"], augment=False, seed=seed),
        "test": ArrayDataset(*splits["test"], augment=False, seed=seed),
    }
    log.info(
        "Dataset %s | train=%d val=%d test=%d | %d classes | %dx%d x%d ch",
        meta.name, len(out["train"]), len(out["val"]), len(out["test"]),
        meta.n_classes, meta.image_size, meta.image_size, meta.in_channels,
    )
    return out, meta


def get_dataloaders(
    datasets: dict[str, ArrayDataset],
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }


def make_shifted_testset(
    test_ds: ArrayDataset, corruption: str, severity: int = 3
) -> ArrayDataset:

    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{corruption}'. Options: {list(CORRUPTIONS)}")
    return ArrayDataset(
        test_ds.images,
        test_ds.labels,
        augment=False,
        corruption=corruption,
        severity=severity,
        seed=1234,
    )

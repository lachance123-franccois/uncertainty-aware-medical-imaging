from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import EvidentialLoss, build_loss
from .models import build_model
from .utils import count_parameters, ensure_dir, log, save_json, set_seed


@dataclass
class TrainConfig:

    dataset: str = "pneumoniamnist"
    data_root: str = "./data"
    image_size: int = 28
    arch: str = "smallcnn"
    width: int = 32
    p_drop: float = 0.3
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 8
    seed: int = 0
    augment: bool = True
    num_workers: int = 0
    # evidential
    evidential_kind: str = "mse"
    annealing_epochs: int = 10
    lambda_max: float = 1.0
    # ensemble
    n_members: int = 5
    # bookkeeping
    n_classes: int = 2
    in_channels: int = 1
    class_names: list[str] = field(default_factory=list)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )

@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).detach().item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_single(
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    method: str = "baseline",
    seed: int | None = None,
    device: torch.device | str = "cpu",
    out_dir: str | Path | None = None,
    tag: str = "model",
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict]:
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    device = torch.device(device)

    train_ds = getattr(loaders["train"], "dataset", None)
    if hasattr(train_ds, "reset_rng"):
        train_ds.reset_rng(seed)

    model = build_model(
        cfg.arch, cfg.in_channels, cfg.n_classes, p_drop=cfg.p_drop, width=cfg.width
    ).to(device)
    criterion = build_loss(
        method,
        cfg.n_classes,
        label_smoothing=cfg.label_smoothing,
        evidential_kind=cfg.evidential_kind,
        annealing_epochs=cfg.annealing_epochs,
        lambda_max=cfg.lambda_max,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else None
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "kl_weight": []
    }
    best_val, best_state, patience = float("inf"), None, 0

    if verbose:
        log.info(
            "Training %s [%s, seed=%d, %s params] for %d epochs on %s",
            tag, method, seed, f"{count_parameters(model):,}", cfg.epochs, device,
        )

    for epoch in range(cfg.epochs):
        if isinstance(criterion, EvidentialLoss):
            criterion.set_epoch(epoch)

        model.train()
        running, correct, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.detach().item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)

        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc = running / max(n, 1), correct / max(n, 1)
        val_loss, val_acc = evaluate_epoch(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["kl_weight"].append(
            criterion.kl_weight if isinstance(criterion, EvidentialLoss) else 0.0
        )

        if verbose:
            log.info(
                "  epoch %2d/%d | train %.4f (%.3f) | val %.4f (%.3f)",
                epoch + 1, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
            )

        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                if verbose:
                    log.info("  early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(cfg), "method": method,
             "seed": seed, "history": history},
            Path(out_dir) / f"{tag}.pt",
        )
        save_json(history, Path(out_dir) / f"{tag}_history.json")

    return model, history


def train_ensemble(
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[list[torch.nn.Module], list[dict]]:
    models, histories = [], []
    for i in range(cfg.n_members):
        model, hist = train_single(
            loaders, cfg, method="baseline", seed=cfg.seed + 1000 * (i + 1),
            device=device, out_dir=out_dir, tag=f"member_{i}", verbose=verbose,
        )
        models.append(model)
        histories.append(hist)
        if verbose:
            log.info("  member %d/%d done (best val acc %.3f)",
                     i + 1, cfg.n_members, max(hist["val_acc"]))
    return models, histories


def load_checkpoint(path: str | Path, device="cpu") -> tuple[torch.nn.Module, TrainConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.arch, cfg.in_channels, cfg.n_classes, cfg.p_drop, cfg.width)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def load_models(
    ckpt_dir: str | Path, device="cpu", pattern: str = "*.pt"
) -> tuple[list[torch.nn.Module], TrainConfig]:
    paths = sorted(Path(ckpt_dir).glob(pattern))
    models, cfg = [], None
    for p in paths:
        model, cfg = load_checkpoint(p, device)
        models.append(model)
    return models, cfg


def summarize_history(histories: list[dict]) -> dict[str, float]:
    return {
        "final_val_acc_mean": float(np.mean([h["val_acc"][-1] for h in histories])),
        "final_val_acc_std": float(np.std([h["val_acc"][-1] for h in histories])),
        "epochs_run_mean": float(np.mean([len(h["val_acc"]) for h in histories])),
    }

from __future__ import annotations

if __package__ in (None, ""):  # executed as a plain script, not as a module
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "umi"

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import collect_logits, fit_temperature
from .data import get_dataloaders, get_datasets
from .evaluate import run_evaluation, summarize_console
from .maps import occlusion_uncertainty_map
from .methods import build_predictor
from .train import TrainConfig, load_models, train_ensemble, train_single
from .utils import ensure_dir, get_device, log, save_json, set_seed
from .viz import plot_map_grid, plot_training_curves, plot_uncertainty_overlay

METHODS = ["baseline", "mc_dropout", "deep_ensemble", "evidential"]


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        default="pneumoniamnist",
        help="pneumoniamnist | breastmnist | dermamnist | octmnist | bloodmnist | synthetic",
    )
    p.add_argument("--data-root", default="./data")
    p.add_argument("--image-size", type=int, default=28)
    p.add_argument("--arch", default="smallcnn", choices=["smallcnn", "resnet18"])
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--p-drop", type=float, default=0.3,
                   help="dropout rate; must be > 0 for MC Dropout")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-members", type=int, default=5, help="deep ensemble size")
    p.add_argument("--n-samples", type=int, default=30, help="MC Dropout forward passes")
    p.add_argument("--evidential-kind", default="mse", choices=["mse", "digamma"])
    p.add_argument("--annealing-epochs", type=int, default=10)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/exp1", help="output directory")
    p.add_argument("--n-train-synthetic", type=int, default=4000)


def cfg_from_args(args, meta) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset, data_root=args.data_root, image_size=args.image_size,
        arch=args.arch, width=args.width, p_drop=args.p_drop, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, augment=not args.no_augment, num_workers=args.num_workers,
        evidential_kind=args.evidential_kind, annealing_epochs=args.annealing_epochs,
        lambda_max=args.lambda_max, n_members=args.n_members,
        n_classes=meta.n_classes, in_channels=meta.in_channels,
        class_names=list(meta.class_names),
    )


def prepare_data(args):
    datasets, meta = get_datasets(
        dataset=args.dataset, root=args.data_root, image_size=args.image_size,
        n_train_synthetic=args.n_train_synthetic, augment=not args.no_augment, seed=args.seed,
    )
    loaders = get_dataloaders(datasets, args.batch_size, args.num_workers)
    return datasets, loaders, meta


# --------------------------------------------------------------------------- #
def cmd_train(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    datasets, loaders, meta = prepare_data(args)
    cfg = cfg_from_args(args, meta)
    ckpt_dir = ensure_dir(Path(args.out) / "checkpoints" / args.method)

    if args.method == "deep_ensemble":
        _, histories = train_ensemble(loaders, cfg, device, ckpt_dir)
    else:
        method = "evidential" if args.method == "evidential" else "baseline"
        _, hist = train_single(loaders, cfg, method=method, device=device,
                               out_dir=ckpt_dir, tag="model")
        histories = [hist]

    save_json(dataclasses.asdict(cfg), Path(args.out) / "config.json")
    plot_training_curves({args.method: histories},
                         Path(args.out) / "figures" / f"training_{args.method}.png")
    log.info("Checkpoints saved to %s", ckpt_dir)


def build_all_predictors(args, ckpt_root: Path, loaders, device):
    """Instantiate every predictor from checkpoints on disk."""
    predictors = {}
    ckpt_root = Path(ckpt_root)

    single_dir = ckpt_root / "baseline"
    if single_dir.exists():
        models, _ = load_models(single_dir, device)
        predictors["baseline"] = build_predictor("baseline", models[0], device)
        predictors["mc_dropout"] = build_predictor(
            "mc_dropout", models[0], device, n_samples=args.n_samples
        )
        # Post-hoc temperature scaling, fitted on the validation split only.
        logits, labels = collect_logits(models[0], loaders["val"], device)
        t = fit_temperature(logits, labels)
        log.info("Fitted temperature T = %.3f on the validation split", t)
        predictors["baseline_temp"] = build_predictor(
            "baseline", models[0], device, temperature=t
        )

    ens_dir = ckpt_root / "deep_ensemble"
    if ens_dir.exists():
        members, _ = load_models(ens_dir, device, "member_*.pt")
        if len(members) >= 2:
            predictors["deep_ensemble"] = build_predictor("deep_ensemble", members, device)

    ev_dir = ckpt_root / "evidential"
    if ev_dir.exists():
        models, _ = load_models(ev_dir, device)
        predictors["evidential"] = build_predictor("evidential", models[0], device)

    if not predictors:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}; run `train` first.")
    return predictors


def cmd_evaluate(args) -> None:
    device = get_device(args.device)
    datasets, loaders, meta = prepare_data(args)
    ckpt_root = Path(args.checkpoints or Path(args.out) / "checkpoints")
    predictors = build_all_predictors(args, ckpt_root, loaders, device)

    res = run_evaluation(
        predictors, loaders, datasets["test"], out_dir=args.out, dataset=args.dataset,
        uncertainty_kind=args.uncertainty, n_bins=args.n_bins, n_samples=args.n_samples,
        n_members=args.n_members, corruption=args.corruption,
        severities=tuple(args.severities), do_sweep=not args.no_sweep,
    )
    print("\n" + summarize_console(res["results"]["metrics"]) + "\n")


def cmd_maps(args) -> None:
    device = get_device(args.device)
    datasets, loaders, meta = prepare_data(args)
    ckpt_root = Path(args.checkpoints or Path(args.out) / "checkpoints")
    predictors = build_all_predictors(args, ckpt_root, loaders, device)
    if args.method not in predictors:
        raise ValueError(f"No checkpoint for '{args.method}'. Available: {list(predictors)}")
    predictor = predictors[args.method]

    test = datasets["test"]
    out = predictor.predict(DataLoader(test, batch_size=args.batch_size))
    unc = out.uncertainty(args.uncertainty)
    # Show the extremes: the cases the model finds hardest and easiest.
    order = np.argsort(-unc)
    picks = list(order[: args.n_images // 2]) + list(order[-(args.n_images // 2) :])

    fig_dir = ensure_dir(Path(args.out) / "figures" / f"maps_{args.method}")
    items = []
    for idx in picks:
        image, label = test[int(idx)]
        m = occlusion_uncertainty_map(
            predictor, image, patch=args.patch, stride=args.stride,
            uncertainty_kind=args.uncertainty,
        )
        title = (
            f"true={meta.class_names[int(label)][:12]} "
            f"pred={meta.class_names[int(out.preds[idx])][:12]} "
            f"u={unc[idx]:.3f}"
        )
        plot_uncertainty_overlay(
            image.numpy(), m["delta"], title=f"{args.method} | {title}",
            path=Path(fig_dir) / f"map_{int(idx)}.png",
        )
        items.append({"image": image.numpy(), "map": m["delta"], "title": title})

    plot_map_grid(items, Path(args.out) / "figures" / f"map_grid_{args.method}.png")
    log.info("Uncertainty maps saved to %s", fig_dir)


def cmd_run_all(args) -> None:
    """Train every method, then evaluate them all, then draw the maps."""
    set_seed(args.seed)
    device = get_device(args.device)
    datasets, loaders, meta = prepare_data(args)
    cfg = cfg_from_args(args, meta)
    ckpt_root = ensure_dir(Path(args.out) / "checkpoints")
    histories = {}

    log.info("=== 1/3  Baseline (also used, unchanged, by MC Dropout) ===")
    _, hist = train_single(loaders, cfg, "baseline", device=device,
                           out_dir=ckpt_root / "baseline", tag="model")
    histories["baseline"] = [hist]

    log.info("=== 2/3  Deep Ensemble (%d members) ===", cfg.n_members)
    _, ens_hist = train_ensemble(loaders, cfg, device, ckpt_root / "deep_ensemble")
    histories["deep_ensemble"] = ens_hist

    log.info("=== 3/3  Evidential ===")
    _, ev_hist = train_single(loaders, cfg, "evidential", device=device,
                              out_dir=ckpt_root / "evidential", tag="model")
    histories["evidential"] = [ev_hist]

    save_json(dataclasses.asdict(cfg), Path(args.out) / "config.json")
    plot_training_curves(histories, Path(args.out) / "figures" / "training_curves.png")

    log.info("=== Evaluation ===")
    cmd_evaluate(args)
    if not args.no_maps:
        log.info("=== Uncertainty maps ===")
        for method in ("mc_dropout", "deep_ensemble", "evidential"):
            args.method = method
            try:
                cmd_maps(args)
            except Exception as exc:  # maps are a nice-to-have, never fatal
                log.warning("map generation failed for %s: %s", method, exc)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umi",
        description="Uncertainty quantification for medical image classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train one method")
    add_common_args(p_train)
    p_train.add_argument("--method", default="baseline", choices=METHODS)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate every trained method")
    add_common_args(p_eval)
    p_eval.add_argument("--checkpoints", default=None)
    p_eval.add_argument("--uncertainty", default="total",
                        choices=["total", "aleatoric", "epistemic", "confidence"])
    p_eval.add_argument("--n-bins", type=int, default=15)
    p_eval.add_argument("--corruption", default="gaussian_noise")
    p_eval.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    p_eval.add_argument("--no-sweep", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_maps = sub.add_parser("maps", help="occlusion-based spatial uncertainty maps")
    add_common_args(p_maps)
    p_maps.add_argument("--checkpoints", default=None)
    p_maps.add_argument("--method", default="mc_dropout", choices=METHODS)
    p_maps.add_argument("--uncertainty", default="total",
                        choices=["total", "aleatoric", "epistemic", "vacuity"])
    p_maps.add_argument("--n-images", type=int, default=8)
    p_maps.add_argument("--patch", type=int, default=8)
    p_maps.add_argument("--stride", type=int, default=2)
    p_maps.set_defaults(func=cmd_maps)

    p_all = sub.add_parser("run-all", help="train + evaluate + maps, end to end")
    add_common_args(p_all)
    p_all.add_argument("--checkpoints", default=None)
    p_all.add_argument("--uncertainty", default="total",
                       choices=["total", "aleatoric", "epistemic", "confidence"])
    p_all.add_argument("--n-bins", type=int, default=15)
    p_all.add_argument("--corruption", default="gaussian_noise")
    p_all.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    p_all.add_argument("--no-sweep", action="store_true")
    p_all.add_argument("--no-maps", action="store_true")
    p_all.add_argument("--n-images", type=int, default=6)
    p_all.add_argument("--patch", type=int, default=8)
    p_all.add_argument("--stride", type=int, default=4)
    p_all.set_defaults(func=cmd_run_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    args.func(args)


if __name__ == "__main__":
    main()
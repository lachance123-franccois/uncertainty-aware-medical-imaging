from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: works in CI and over SSH
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import reliability_curve  # noqa: E402
from .metrics import risk_coverage_curve  # noqa: E402
from .uncertainty import UQOutput  # noqa: E402
from .utils import ensure_dir  # noqa: E402

METHOD_COLORS = {
    "baseline": "#8c8c8c",
    "baseline_temp": "#5b8fa8",
    "mc_dropout": "#e07b39",
    "deep_ensemble": "#3b7dd8",
    "evidential": "#57a773",
}
METHOD_LABELS = {
    "baseline": "Baseline (softmax)",
    "baseline_temp": "Baseline + temp. scaling",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "evidential": "Evidential DL",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def _color(m: str) -> str:
    return METHOD_COLORS.get(m, None)


def _save(fig, path: str | Path | None):
    if path is not None:
        ensure_dir(Path(path).parent)
        fig.savefig(path)
        plt.close(fig)
    return fig


#
def plot_reliability_diagram(
    probs, labels, n_bins: int = 15, title: str = "", path=None, adaptive: bool = False
):
    _style()
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    curve = reliability_curve(conf, correct, n_bins, adaptive)
    ece = float(np.sum(curve.bin_count / curve.bin_count.sum() * np.abs(curve.gaps)))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(4.2, 5.0), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    widths = np.diff(curve.bin_edges)
    ax.bar(curve.bin_centers, curve.bin_accuracy, width=widths * 0.95,
           color="#3b7dd8", edgecolor="white", label="Accuracy", zorder=3)
    ax.bar(curve.bin_centers, curve.gaps * -1, bottom=curve.bin_accuracy, width=widths * 0.95,
           color="#d94f4f", alpha=0.45, edgecolor="#a33", hatch="//",
           label="Gap (|acc - conf|)", zorder=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration", zorder=4)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.set_title(f"{title}\nECE = {ece:.4f} | acc = {correct.mean():.3f}", fontsize=9)

    ax2.bar(curve.bin_centers, curve.bin_count, width=widths * 0.95, color="#999")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("# samples")
    ax2.set_yscale("symlog")
    return _save(fig, path)


def plot_reliability_comparison(
    outputs: dict[str, UQOutput], n_bins: int = 15, path=None, min_bin_frac: float = 0.01
):
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, out in outputs.items():
        curve = reliability_curve(out.confidence, out.correct, n_bins)
        n = curve.bin_count.sum()
        mask = curve.bin_count >= max(min_bin_frac * n, 1)
        ax.plot(curve.bin_confidence[mask], curve.bin_accuracy[mask], "-",
                lw=1.6, color=_color(name), label=_label(name))
        ax.scatter(curve.bin_confidence[mask], curve.bin_accuracy[mask],
                   s=8 + 120 * curve.bin_count[mask] / max(n, 1),
                   color=_color(name), alpha=0.85, zorder=3)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram - all methods\n(marker area = samples in bin)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, path)


def plot_risk_coverage(outputs: dict[str, UQOutput], kind: str = "total", path=None):
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name, out in outputs.items():
        rc = risk_coverage_curve(out.uncertainty(kind), out.correct)
        ax.plot(rc.coverage, rc.risk, lw=1.8, color=_color(name),
                label=f"{_label(name)} (AURC={rc.aurc:.3f})")
    ax.set_xlabel("Coverage (fraction of cases kept)")
    ax.set_ylabel("Risk (error rate on kept cases)")
    ax.set_title(f"Risk-coverage - {kind} uncertainty")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_uncertainty_separation(outputs: dict[str, UQOutput], kind: str = "total", path=None):

    _style()
    n = len(outputs)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, out) in zip(axes, outputs.items(), strict=False):
        u, c = out.uncertainty(kind), out.correct.astype(bool)
        bins = np.linspace(0, max(u.max(), 1e-6), 25)
        ax.hist(u[c], bins=bins, alpha=0.65, color="#3b7dd8", label="correct", density=True)
        ax.hist(u[~c], bins=bins, alpha=0.65, color="#d94f4f", label="error", density=True)
        ax.set_title(_label(name), fontsize=9)
        ax.set_xlabel(f"{kind} uncertainty")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    fig.suptitle("Is the model uncertain where it is wrong?", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_uncertainty_overlay(
    image: np.ndarray,
    umap: np.ndarray,
    title: str = "",
    path=None,
    alpha: float = 0.55,
    symmetric: bool = True,
):

    _style()
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0) if img.shape[0] in (1, 3) else img
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    vmax = float(np.abs(umap).max() + 1e-8)
    vmin = -vmax if symmetric else float(umap.min())

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(umap, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title("Uncertainty map")
    axes[2].imshow(img, cmap="gray")
    im = axes[2].imshow(umap, cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=alpha)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="uncertain (red) / confident (blue)")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_map_grid(items: list[dict], path=None, cols: int = 4):
    _style()
    cols = min(cols, max(len(items), 1))
    if len(items) % cols and cols > 2 and len(items) % (cols - 1) == 0:
        cols -= 1  # avoid a ragged last row when a smaller grid divides evenly
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items, strict=False):
        img = np.asarray(item["image"])
        img = img[0] if img.ndim == 3 and img.shape[0] == 1 else img
        m = item["map"]
        vmax = float(np.abs(m).max() + 1e-8)
        ax.imshow(img, cmap="gray")
        ax.imshow(m, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.5)
        ax.set_title(item.get("title", ""), fontsize=7)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_bars(table: dict[str, dict[str, float]], metrics: list[str], path=None):
    _style()
    methods = list(table)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 3.1))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics, strict=False):
        vals = [table[m].get(metric, np.nan) for m in methods]
        ax.bar(range(len(methods)), vals, color=[_color(m) for m in methods])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods], rotation=35, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=9)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_corruption_sweep(sweep: dict[str, dict[int, dict[str, float]]], path=None):

    _style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for name, per_sev in sweep.items():
        sev = sorted(per_sev)
        ax1.plot(sev, [per_sev[s]["accuracy"] for s in sev], "o-",
                 color=_color(name), label=_label(name))
        ax2.plot(sev, [per_sev[s]["mean_uncertainty"] for s in sev], "o-", color=_color(name))
        ax3.plot(sev, [per_sev[s]["ece"] for s in sev], "o-", color=_color(name))
    ax1.set_title("Accuracy vs. severity")
    ax2.set_title("Mean uncertainty vs. severity")
    ax3.set_title("ECE vs. severity")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Corruption severity (0 = clean)")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curves(histories: dict[str, list[dict]], path=None):
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for name, hists in histories.items():
        for i, h in enumerate(hists):
            ax1.plot(h["val_loss"], color=_color(name), alpha=0.8 if i == 0 else 0.35,
                     label=_label(name) if i == 0 else None)
            ax2.plot(h["val_acc"], color=_color(name), alpha=0.8 if i == 0 else 0.35)
    ax1.set_title("Validation loss")
    ax2.set_title("Validation accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_cost_vs_quality(table: dict[str, dict[str, float]], path=None):
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name, row in table.items():
        x = row.get("relative_cost", np.nan)
        y = row.get("ece", np.nan)
        ax.scatter(x, y, s=90, color=_color(name), zorder=3)
        ax.annotate(_label(name), (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Relative compute cost (train + inference, log scale)")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Calibration vs. computational cost")
    return _save(fig, path)

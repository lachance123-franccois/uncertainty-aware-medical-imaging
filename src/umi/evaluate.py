from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import calibration_report
from .data import ArrayDataset, make_shifted_testset
from .methods import Predictor
from .metrics import error_uncertainty_report, ood_detection_auroc
from .uncertainty import UQOutput
from .utils import ensure_dir, log, save_json
from .viz import (
    plot_corruption_sweep,
    plot_cost_vs_quality,
    plot_metric_bars,
    plot_reliability_comparison,
    plot_reliability_diagram,
    plot_risk_coverage,
    plot_uncertainty_separation,
)

HEADLINE_METRICS = ["accuracy", "ece", "misclassification_auroc", "aurc", "nll"]

RELATIVE_COST = {  # forward passes per prediction, relative to one model
    "baseline": 1.0,
    "baseline_temp": 1.0,
    "evidential": 1.0,
}


def method_cost(name: str, n_samples: int, n_members: int) -> dict[str, float]:
    train_cost = {"baseline": 1.0, "baseline_temp": 1.0, "mc_dropout": 1.0,
                  "evidential": 1.0, "deep_ensemble": float(n_members)}
    infer_cost = {"baseline": 1.0, "baseline_temp": 1.0, "evidential": 1.0,
                  "mc_dropout": float(n_samples), "deep_ensemble": float(n_members)}
    t = train_cost.get(name, 1.0)
    i = infer_cost.get(name, 1.0)
    return {"train_cost": t, "inference_cost": i, "relative_cost": t * i}


def evaluate_predictor(
    predictor: Predictor,
    loader: DataLoader,
    uncertainty_kind: str = "total",
    n_bins: int = 15,
) -> tuple[UQOutput, dict[str, float]]:
    out = predictor.predict(loader)
    metrics = calibration_report(out.probs, out.labels, n_bins=n_bins)
    metrics.update(error_uncertainty_report(out.uncertainty(uncertainty_kind), out.correct))
    metrics["inference_seconds"] = out.inference_seconds
    metrics["mean_total_uncertainty"] = float(out.total.mean())
    metrics["mean_aleatoric"] = float(out.aleatoric.mean())
    metrics["mean_epistemic"] = float(out.epistemic.mean())
    if out.vacuity is not None:
        metrics["mean_vacuity"] = float(out.vacuity.mean())

    channels = ["total", "aleatoric", "epistemic"] + (
        ["vacuity"] if out.vacuity is not None else []
    )
    for ch in channels:
        u = out.uncertainty(ch)
        if np.allclose(u, u[0]):  # degenerate (e.g. epistemic == 0 for the baseline)
            metrics[f"auroc_{ch}"] = float("nan")
        else:
            metrics[f"auroc_{ch}"] = error_uncertainty_report(u, out.correct)[
                "misclassification_auroc"
            ]
    return out, metrics


def corruption_sweep(
    predictors: dict[str, Predictor],
    test_ds: ArrayDataset,
    corruption: str = "gaussian_noise",
    severities: tuple[int, ...] = (1, 2, 3, 4, 5),
    batch_size: int = 256,
    uncertainty_kind: str = "total",
    clean_outputs: dict[str, UQOutput] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    sweep: dict[str, dict[int, dict[str, float]]] = {name: {} for name in predictors}

    for name, pred in predictors.items():
        clean = (
            clean_outputs[name]
            if clean_outputs is not None
            else pred.predict(DataLoader(test_ds, batch_size=batch_size))
        )
        sweep[name][0] = {
            "accuracy": clean.accuracy,
            "mean_uncertainty": float(clean.uncertainty(uncertainty_kind).mean()),
            "ece": calibration_report(clean.probs, clean.labels)["ece"],
            "ood_auroc": float("nan"),
        }
        for sev in severities:
            shifted = make_shifted_testset(test_ds, corruption, sev)
            out = pred.predict(DataLoader(shifted, batch_size=batch_size))
            sweep[name][sev] = {
                "accuracy": out.accuracy,
                "mean_uncertainty": float(out.uncertainty(uncertainty_kind).mean()),
                "ece": calibration_report(out.probs, out.labels)["ece"],
                "ood_auroc": ood_detection_auroc(
                    clean.uncertainty(uncertainty_kind), out.uncertainty(uncertainty_kind)
                ),
            }
        log.info(
            "%-14s severity sweep done (acc %.3f -> %.3f, unc %.3f -> %.3f)",
            name,
            sweep[name][0]["accuracy"], sweep[name][max(severities)]["accuracy"],
            sweep[name][0]["mean_uncertainty"], sweep[name][max(severities)]["mean_uncertainty"],
        )
    return sweep


def _fmt(v) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def markdown_table(table: dict[str, dict[str, float]], columns: list[str]) -> str:
    header = "| Method | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    rows = [
        "| " + name + " | " + " | ".join(_fmt(row.get(c, float("nan"))) for c in columns) + " |"
        for name, row in table.items()
    ]
    return "\n".join([header, sep, *rows])


def _best(table: dict[str, dict[str, float]], metric: str, lower_is_better: bool) -> str:
    vals = {k: v.get(metric, np.nan) for k, v in table.items()}
    vals = {k: v for k, v in vals.items() if np.isfinite(v)}
    if not vals:
        return "n/a"
    return min(vals, key=vals.get) if lower_is_better else max(vals, key=vals.get)


def write_report(
    table: dict[str, dict[str, float]],
    sweep: dict | None,
    out_dir: str | Path,
    dataset: str,
    uncertainty_kind: str = "total",
) -> Path:
    out_dir = ensure_dir(out_dir)
    best_ece = _best(table, "ece", True)
    best_auroc = _best(table, "misclassification_auroc", False)
    best_acc = _best(table, "accuracy", False)
    best_aurc = _best(table, "aurc", True)

    lines = [
        f"# Uncertainty quantification report - `{dataset}`",
        "",
        f"Uncertainty channel used for error detection: **{uncertainty_kind}**.",
        "",
        "## 1. Calibration",
        "",
        markdown_table(
            table,
            ["accuracy", "ece", "ece_adaptive", "mce", "classwise_ece", "brier", "nll",
             "mean_confidence", "overconfidence"],
        ),
        "",
        "*ECE* = Expected Calibration Error (15 equal-width bins); *overconfidence* =",
        "mean confidence minus accuracy, so a positive value means the model claims more",
        "certainty than it earns.",
        "",
        "## 2. Does uncertainty predict the errors?",
        "",
        markdown_table(
            table,
            ["misclassification_auroc", "point_biserial_r", "aurc", "excess_aurc",
             "risk_at_80_coverage", "mean_uncertainty_error", "mean_uncertainty_correct",
             "p_value"],
        ),
        "",
        "`risk_at_80_coverage` is the error rate once the 20% most uncertain cases are",
        "deferred to a human reader - the number that matters for deployment.",
        "",
        "## 3. Cost",
        "",
        markdown_table(table, ["train_cost", "inference_cost", "relative_cost",
                               "inference_seconds"]),
        "",
        "## 4. Uncertainty channels (error-detection AUROC per channel)",
        "",
        markdown_table(table, ["auroc_total", "auroc_aleatoric", "auroc_epistemic",
                               "auroc_vacuity"]),
        "",
    ]

    if sweep:
        lines += ["## 5. Distribution shift", "", "| Method | " + " | ".join(
            f"sev {s}" for s in sorted(next(iter(sweep.values())))) + " |"]
        sevs = sorted(next(iter(sweep.values())))
        lines.append("|" + "---|" * (len(sevs) + 1))
        for name, per_sev in sweep.items():
            acc = " | ".join(
                f"{per_sev[s]['accuracy']:.3f} / {per_sev[s]['mean_uncertainty']:.3f}"
                for s in sevs
            )
            lines.append(f"| {name} | {acc} |")
        lines += ["", "Cells are `accuracy / mean uncertainty`. The desirable pattern is",
                  "accuracy falling **and** uncertainty rising.", ""]

    lines += [
        "## Conclusions (auto-generated)",
        "",
        f"- Best calibrated (lowest ECE): **{best_ece}**",
        f"- Best error detection (highest misclassification AUROC): **{best_auroc}**",
        f"- Best selective prediction (lowest AURC): **{best_aurc}**",
        f"- Best raw accuracy: **{best_acc}**",
        "",
        "Read these together with the cost table: if the cheapest method is within noise",
        "of the most expensive one, the cheap method wins in practice.",
        "",
    ]

    path = Path(out_dir) / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", path)
    return path


def run_evaluation(
    predictors: dict[str, Predictor],
    loaders: dict[str, DataLoader],
    test_ds: ArrayDataset,
    out_dir: str | Path,
    dataset: str = "unknown",
    uncertainty_kind: str = "total",
    n_bins: int = 15,
    n_samples: int = 30,
    n_members: int = 5,
    corruption: str = "gaussian_noise",
    severities: tuple[int, ...] = (1, 3, 5),
    do_sweep: bool = True,
) -> dict:
    out_dir = ensure_dir(out_dir)
    fig_dir = ensure_dir(Path(out_dir) / "figures")

    outputs: dict[str, UQOutput] = {}
    table: dict[str, dict[str, float]] = {}

    for name, pred in predictors.items():
        log.info("Evaluating %s ...", name)
        out, metrics = evaluate_predictor(pred, loaders["test"], uncertainty_kind, n_bins)
        metrics.update(method_cost(name, n_samples, n_members))
        outputs[name] = out
        table[name] = metrics
        plot_reliability_diagram(
            out.probs, out.labels, n_bins, title=f"{name} - {dataset}",
            path=Path(fig_dir) / f"reliability_{name}.png",
        )

    plot_reliability_comparison(outputs, n_bins, Path(fig_dir) / "reliability_all.png")
    plot_risk_coverage(outputs, uncertainty_kind, Path(fig_dir) / "risk_coverage.png")
    plot_uncertainty_separation(outputs, uncertainty_kind, Path(fig_dir) / "uncertainty_split.png")
    plot_metric_bars(table, HEADLINE_METRICS, Path(fig_dir) / "headline_metrics.png")
    plot_cost_vs_quality(table, Path(fig_dir) / "cost_vs_calibration.png")

    sweep = None
    if do_sweep:
        sweep = corruption_sweep(
            predictors, test_ds, corruption, severities,
            uncertainty_kind=uncertainty_kind, clean_outputs=outputs,
        )
        plot_corruption_sweep(sweep, Path(fig_dir) / f"shift_{corruption}.png")

    results = {
        "dataset": dataset,
        "uncertainty_kind": uncertainty_kind,
        "metrics": table,
        "shift": {"corruption": corruption, "sweep": sweep} if sweep else None,
    }
    save_json(results, Path(out_dir) / "results.json")
    np.savez_compressed(
        Path(out_dir) / "predictions.npz",
        **{f"{n}_probs": o.probs for n, o in outputs.items()},
        **{f"{n}_unc": o.uncertainty(uncertainty_kind) for n, o in outputs.items()},
        labels=next(iter(outputs.values())).labels,
    )
    write_report(table, sweep, out_dir, dataset, uncertainty_kind)
    return {"results": results, "outputs": outputs}


@torch.no_grad()
def summarize_console(table: dict[str, dict[str, float]]) -> str:
    cols = HEADLINE_METRICS
    width = max(len(k) for k in table) + 2
    header = "method".ljust(width) + "".join(c[:14].rjust(16) for c in cols)
    lines = [header, "-" * len(header)]
    for name, row in table.items():
        lines.append(
            name.ljust(width) + "".join(_fmt(row.get(c, float("nan"))).rjust(16) for c in cols)
        )
    return "\n".join(lines)

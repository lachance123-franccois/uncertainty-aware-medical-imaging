r"""Does the uncertainty point at the errors?

Calibration answers "is the confidence *number* right on average?". This module
answers the operational question a clinician actually asks: **if I only trust
the model when it is confident, do I get a safer system?**

* ``misclassification_auroc`` -- how well uncertainty *ranks* errors above
  correct predictions. 0.5 = the uncertainty carries no information about
  correctness; 1.0 = every error is flagged before any correct case.
* ``mann_whitney_test`` / ``point_biserial`` -- the statistical version of the
  same claim: misclassified samples have significantly higher uncertainty.
* ``risk_coverage_curve`` / ``aurc`` -- **selective prediction**. Sort by
  uncertainty, defer the most uncertain fraction to a human, and plot the error
  rate on the remainder. This directly quantifies the deployment story: "at 80%
  coverage the model's error rate drops from 8% to 2%, and the 20% deferred
  cases go to a radiologist."
* ``ood_detection_auroc`` -- separating in-distribution from shifted data using
  uncertainty alone.

Only NumPy is required; the p-values use a normal approximation, which is
accurate for the sample sizes involved here (n >> 20).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    pos = np.asarray(positives).astype(bool).ravel()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, positives: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    pos = np.asarray(positives).astype(bool).ravel()
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = pos[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def misclassification_auroc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    return auroc(np.asarray(uncertainty), ~np.asarray(correct).astype(bool))


def mann_whitney_test(uncertainty: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    c = np.asarray(correct).astype(bool).ravel()
    err, ok = u[~c], u[c]
    n1, n2 = len(err), len(ok)
    if n1 == 0 or n2 == 0:
        return {"u_statistic": float("nan"), "z": float("nan"), "p_value": float("nan")}

    ranks = _rank(u)
    u_stat = ranks[~c].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mu) / sigma if sigma > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))  # P(Z >= z), one-sided
    return {
        "u_statistic": float(u_stat),
        "z": float(z),
        "p_value": float(p),
        "mean_uncertainty_error": float(err.mean()),
        "mean_uncertainty_correct": float(ok.mean()),
        "effect_size_rank_biserial": float(2 * u_stat / (n1 * n2) - 1),
    }


def point_biserial(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = (~np.asarray(correct).astype(bool)).astype(np.float64)
    if u.std() == 0 or e.std() == 0:
        return float("nan")
    return float(np.corrcoef(u, e)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(np.asarray(a, dtype=np.float64)), _rank(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class RiskCoverage:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float
    excess_aurc: float
    risk_at_coverage: dict[float, float]


def risk_coverage_curve(uncertainty: np.ndarray, correct: np.ndarray) -> RiskCoverage:

    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    err = (~np.asarray(correct).astype(bool)).astype(np.float64)
    n = len(u)
    order = np.argsort(u, kind="mergesort")  # most confident first
    err_sorted = err[order]

    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    risk = cum_err / k
    coverage = k / n
    aurc_val = float(risk.mean())

    # Oracle: all correct predictions first, then all errors.
    n_err = int(err.sum())
    oracle = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    oracle_risk = np.cumsum(oracle) / k
    excess = aurc_val - float(oracle_risk.mean())

    at = {}
    for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        idx = max(int(round(c * n)) - 1, 0)
        at[c] = float(risk[idx])

    return RiskCoverage(coverage, risk, aurc_val, excess, at)


def ood_detection_auroc(unc_in: np.ndarray, unc_out: np.ndarray) -> float:
    scores = np.concatenate([np.asarray(unc_in).ravel(), np.asarray(unc_out).ravel()])
    labels = np.concatenate([np.zeros(len(unc_in)), np.ones(len(unc_out))])
    return auroc(scores, labels.astype(bool))


def error_uncertainty_report(
    uncertainty: np.ndarray, correct: np.ndarray
) -> dict[str, float]:
    rc = risk_coverage_curve(uncertainty, correct)
    test = mann_whitney_test(uncertainty, correct)
    return {
        "misclassification_auroc": misclassification_auroc(uncertainty, correct),
        "misclassification_ap": average_precision(
            uncertainty, ~np.asarray(correct).astype(bool)
        ),
        "point_biserial_r": point_biserial(uncertainty, correct),
        "aurc": rc.aurc,
        "excess_aurc": rc.excess_aurc,
        "risk_at_80_coverage": rc.risk_at_coverage[0.8],
        "risk_at_100_coverage": rc.risk_at_coverage[1.0],
        **test,
    }

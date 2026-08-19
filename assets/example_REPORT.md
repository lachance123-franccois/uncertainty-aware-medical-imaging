# Uncertainty quantification report - `pneumoniamnist`

Uncertainty channel used for error detection: **total**.

## 1. Calibration

| Method | accuracy | ece | ece_adaptive | mce | classwise_ece | brier | nll | mean_confidence | overconfidence |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8478 | 0.0948 | 0.0916 | 0.2511 | 0.1330 | 0.2453 | 0.4670 | 0.9301 | 0.0824 |
| mc_dropout | 0.8333 | 0.0785 | 0.0780 | 0.2367 | 0.1453 | 0.2497 | 0.4396 | 0.9112 | 0.0778 |
| baseline_temp | 0.8478 | 0.0997 | 0.0955 | 0.3840 | 0.1352 | 0.2480 | 0.4868 | 0.9347 | 0.0869 |
| deep_ensemble | 0.8718 | 0.0647 | 0.0619 | 0.2527 | 0.1094 | 0.2009 | 0.3762 | 0.9318 | 0.0600 |
| evidential | 0.8397 | 0.0583 | 0.0735 | 0.4286 | 0.1295 | 0.2292 | 0.3740 | 0.8980 | 0.0583 |

*ECE* = Expected Calibration Error (15 equal-width bins); *overconfidence* =
mean confidence minus accuracy, so a positive value means the model claims more
certainty than it earns.

## 2. Does uncertainty predict the errors?

| Method | misclassification_auroc | point_biserial_r | aurc | excess_aurc | risk_at_80_coverage | mean_uncertainty_error | mean_uncertainty_correct | p_value |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.7811 | 0.3627 | 0.0561 | 0.0438 | 0.0902 | 0.5103 | 0.1967 | 0.0000 |
| mc_dropout | 0.7939 | 0.3968 | 0.0587 | 0.0438 | 0.1042 | 0.5978 | 0.2425 | 0.0000 |
| baseline_temp | 0.7811 | 0.3595 | 0.0561 | 0.0438 | 0.0902 | 0.4878 | 0.1817 | 0.0000 |
| deep_ensemble | 0.8111 | 0.3958 | 0.0397 | 0.0310 | 0.0641 | 0.5614 | 0.1876 | 0.0000 |
| evidential | 0.8080 | 0.4469 | 0.0548 | 0.0411 | 0.0822 | 0.6451 | 0.3410 | 0.0000 |

`risk_at_80_coverage` is the error rate once the 20% most uncertain cases are
deferred to a human reader - the number that matters for deployment.

## 3. Cost

| Method | train_cost | inference_cost | relative_cost | inference_seconds |
|---|---|---|---|---|
| baseline | 1.0000 | 1.0000 | 1.0000 | 1.4935 |
| mc_dropout | 1.0000 | 30.0000 | 30.0000 | 34.5273 |
| baseline_temp | 1.0000 | 1.0000 | 1.0000 | 1.0854 |
| deep_ensemble | 5.0000 | 5.0000 | 25.0000 | 5.6708 |
| evidential | 1.0000 | 1.0000 | 1.0000 | 1.3687 |

## 4. Uncertainty channels (error-detection AUROC per channel)

| Method | auroc_total | auroc_aleatoric | auroc_epistemic | auroc_vacuity |
|---|---|---|---|---|
| baseline | 0.7811 | 0.7811 | n/a | n/a |
| mc_dropout | 0.7939 | 0.7926 | 0.8006 | n/a |
| baseline_temp | 0.7811 | 0.7811 | n/a | n/a |
| deep_ensemble | 0.8111 | 0.8124 | 0.7915 | n/a |
| evidential | 0.8080 | 0.8090 | 0.8081 | 0.8081 |

## 5. Distribution shift

| Method | sev 0 | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |
|---|---|---|---|---|---|---|
| baseline | 0.848 / 0.244 | 0.732 / 0.498 | 0.625 / 0.611 | 0.583 / 0.719 | 0.546 / 0.819 | 0.510 / 0.879 |
| mc_dropout | 0.833 / 0.302 | 0.776 / 0.629 | 0.700 / 0.780 | 0.676 / 0.863 | 0.647 / 0.901 | 0.630 / 0.911 |
| baseline_temp | 0.848 / 0.228 | 0.732 / 0.473 | 0.625 / 0.586 | 0.583 / 0.698 | 0.546 / 0.803 | 0.510 / 0.867 |
| deep_ensemble | 0.872 / 0.236 | 0.837 / 0.508 | 0.726 / 0.699 | 0.628 / 0.792 | 0.583 / 0.851 | 0.540 / 0.893 |
| evidential | 0.840 / 0.390 | 0.796 / 0.550 | 0.684 / 0.640 | 0.561 / 0.665 | 0.470 / 0.648 | 0.431 / 0.630 |

Cells are `accuracy / mean uncertainty`. The desirable pattern is
accuracy falling **and** uncertainty rising.

## Conclusions (auto-generated)

- Best calibrated (lowest ECE): **evidential**
- Best error detection (highest misclassification AUROC): **deep_ensemble**
- Best selective prediction (lowest AURC): **deep_ensemble**
- Best raw accuracy: **deep_ensemble**

Read these together with the cost table: if the cheapest method is within noise
of the most expensive one, the cheap method wins in practice.

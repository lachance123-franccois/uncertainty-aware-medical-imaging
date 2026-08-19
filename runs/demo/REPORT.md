# Uncertainty quantification report - `synthetic`

Uncertainty channel used for error detection: **total**.

## 1. Calibration

| Method | accuracy | ece | ece_adaptive | mce | classwise_ece | brier | nll | mean_confidence | overconfidence |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.8760 | 0.0242 | 0.0306 | 0.3081 | 0.0343 | 0.1862 | 0.2942 | 0.8690 | -0.0070 |
| mc_dropout | 0.8750 | 0.0281 | 0.0280 | 0.1234 | 0.0340 | 0.1866 | 0.2955 | 0.8638 | -0.0112 |
| baseline_temp | 0.8760 | 0.0258 | 0.0275 | 0.2866 | 0.0299 | 0.1874 | 0.2953 | 0.9008 | 0.0248 |
| deep_ensemble | 0.8700 | 0.0284 | 0.0537 | 0.4453 | 0.0291 | 0.1971 | 0.3082 | 0.8611 | -0.0089 |
| evidential | 0.8690 | 0.0356 | 0.0691 | 0.2167 | 0.0362 | 0.2127 | 0.3548 | 0.8390 | -0.0300 |

*ECE* = Expected Calibration Error (15 equal-width bins); *overconfidence* =
mean confidence minus accuracy, so a positive value means the model claims more
certainty than it earns.

## 2. Does uncertainty predict the errors?

| Method | misclassification_auroc | point_biserial_r | aurc | excess_aurc | risk_at_80_coverage | mean_uncertainty_error | mean_uncertainty_correct | p_value |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.8136 | 0.3416 | 0.0356 | 0.0275 | 0.0737 | 0.7767 | 0.4074 | 0.0000 |
| mc_dropout | 0.8141 | 0.3479 | 0.0360 | 0.0278 | 0.0725 | 0.7907 | 0.4216 | 0.0000 |
| baseline_temp | 0.8136 | 0.3625 | 0.0356 | 0.0275 | 0.0737 | 0.6792 | 0.3363 | 0.0000 |
| deep_ensemble | 0.8261 | 0.3286 | 0.0356 | 0.0267 | 0.0712 | 0.7875 | 0.4300 | 0.0000 |
| evidential | 0.7031 | 0.2991 | 0.0561 | 0.0471 | 0.1138 | 0.7330 | 0.5998 | 0.0000 |

`risk_at_80_coverage` is the error rate once the 20% most uncertain cases are
deferred to a human reader - the number that matters for deployment.

## 3. Cost

| Method | train_cost | inference_cost | relative_cost | inference_seconds |
|---|---|---|---|---|
| baseline | 1.0000 | 1.0000 | 1.0000 | 2.4998 |
| mc_dropout | 1.0000 | 20.0000 | 20.0000 | 44.2968 |
| baseline_temp | 1.0000 | 1.0000 | 1.0000 | 2.3719 |
| deep_ensemble | 3.0000 | 3.0000 | 9.0000 | 6.0329 |
| evidential | 1.0000 | 1.0000 | 1.0000 | 1.8254 |

## 4. Uncertainty channels (error-detection AUROC per channel)

| Method | auroc_total | auroc_aleatoric | auroc_epistemic | auroc_vacuity |
|---|---|---|---|---|
| baseline | 0.8136 | 0.8136 | n/a | n/a |
| mc_dropout | 0.8141 | 0.8084 | 0.7736 | n/a |
| baseline_temp | 0.8136 | 0.8136 | n/a | n/a |
| deep_ensemble | 0.8261 | 0.8070 | 0.7189 | n/a |
| evidential | 0.7031 | 0.7031 | 0.7031 | 0.7032 |

## 5. Distribution shift

| Method | sev 0 | sev 1 | sev 3 | sev 5 |
|---|---|---|---|---|
| baseline | 0.876 / 0.453 | 0.868 / 0.490 | 0.833 / 0.590 | 0.729 / 0.791 |
| mc_dropout | 0.875 / 0.468 | 0.866 / 0.519 | 0.829 / 0.689 | 0.725 / 0.844 |
| baseline_temp | 0.876 / 0.379 | 0.868 / 0.425 | 0.833 / 0.516 | 0.729 / 0.710 |
| deep_ensemble | 0.870 / 0.476 | 0.867 / 0.474 | 0.859 / 0.684 | 0.675 / 0.983 |
| evidential | 0.869 / 0.617 | 0.863 / 0.621 | 0.823 / 0.643 | 0.782 / 0.685 |

Cells are `accuracy / mean uncertainty`. The desirable pattern is
accuracy falling **and** uncertainty rising.

## Conclusions (auto-generated)

- Best calibrated (lowest ECE): **baseline**
- Best error detection (highest misclassification AUROC): **deep_ensemble**
- Best selective prediction (lowest AURC): **baseline**
- Best raw accuracy: **baseline**

Read these together with the cost table: if the cheapest method is within noise
of the most expensive one, the cheap method wins in practice.

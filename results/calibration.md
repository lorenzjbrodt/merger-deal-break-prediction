# Calibration

Platt scaling (a logistic fit on the logit of the score) is fitted on the training-period out-of-fold predictions only (validation blocks 2007-2018) and then applied to the test predictions. Target: adverse break, core sample.

| Model | Validation predictions (positives) | Mean predicted, raw | Mean predicted, calibrated | Observed test rate | Brier raw | Brier calibrated | ECE (5 bins) raw | ECE calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3x. LightGBM + extended deal, buyer and target features | 2161 (95) | 0.047 | 0.047 | 0.037 | 0.0357 | 0.0355 | 0.012 | 0.011 |
| 2. Logistic, base tabular | 2161 (95) | 0.047 | 0.042 | 0.037 | 0.0360 | 0.0355 | 0.010 | 0.007 |

See `figures/calibration.png` for the reliability plot (quintile bins with 95% intervals).


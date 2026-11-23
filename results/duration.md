# Time to closing

Target: calendar days from announcement to closing, completed core-sample deals. Models predict log days; the prediction in days is the exponential (a median forecast). Training period 2001-2018: 3,148 completed deals, median 107 days, 10th to 90th percentile 43 to 234. Intervals: bootstrap over deals for the time split, over announcement years for the pooled set.

## Time split: test period 2019-2025, all core deals: 1,026 completed deals

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 72.6 [67.1, 78.4] | 43.0 | -0.014 [-0.032, -0.003] | 38% |
| 1. Median by structure (tender offer / merger) | 63.5 [58.2, 69.1] | 34.0 | 0.260 [0.207, 0.318] | 46% |
| 2. Ridge regression, base terms | 60.7 [55.8, 66.0] | 29.5 | 0.287 [0.188, 0.365] | 51% |
| 3. LightGBM, base terms | 58.4 [53.6, 63.4] | 30.8 | 0.391 [0.322, 0.463] | 49% |
| 3x. LightGBM + extended features | 53.3 [48.7, 57.9] | 26.3 | 0.514 [0.442, 0.587] | 54% |
| 3m. 3x + market context | 53.6 [49.0, 58.2] | 27.2 | 0.515 [0.445, 0.585] | 53% |
| 4. 3m + LLM-extracted terms | 52.7 [48.1, 57.4] | 27.1 | 0.540 [0.474, 0.600] | 52% |
| Final model (chosen by CV inside the training years) | 52.7 [48.1, 57.4] | 27.1 | 0.540 [0.474, 0.600] | 52% |
| R. Log-normal survival model, base terms, broken deals censored | 61.4 [56.5, 66.5] | 30.1 | 0.311 [0.250, 0.371] | 50% |

Paired differences (negative MAE = better):

- LightGBM minus ridge, base terms: MAE -2.3 days [-3.7, -0.9]; R2 on log days +0.104 [+0.038, +0.193].
- Extended features (3x minus 3): MAE -5.1 days [-6.5, -3.8]; R2 on log days +0.124 [+0.092, +0.157].
- Market context (3m minus 3x): MAE +0.3 days [-0.0, +0.6]; R2 on log days +0.001 [-0.005, +0.007].
- LLM-extracted terms (4 minus 3m): MAE -0.9 days [-1.8, +0.1]; R2 on log days +0.025 [+0.007, +0.044].
- Final model minus median by structure: MAE -10.8 days [-13.1, -8.5]; R2 on log days +0.281 [+0.219, +0.350].
- Survival model minus ridge (effect of censoring): MAE +0.6 days [-0.1, +1.4]; R2 on log days +0.024 [-0.009, +0.083].

## Time split: test period 2019-2025, deals with announcement text: 923 completed deals

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 70.3 [64.8, 76.3] | 42.0 | -0.006 [-0.021, -0.000] | 39% |
| 1. Median by structure (tender offer / merger) | 60.8 [55.5, 66.4] | 33.0 | 0.289 [0.228, 0.352] | 47% |
| 2. Ridge regression, base terms | 57.9 [52.8, 63.3] | 27.3 | 0.326 [0.268, 0.380] | 53% |
| 3. LightGBM, base terms | 56.3 [51.5, 61.4] | 29.1 | 0.387 [0.329, 0.445] | 51% |
| 3x. LightGBM + extended features | 52.1 [47.4, 57.0] | 26.2 | 0.486 [0.420, 0.548] | 54% |
| 3m. 3x + market context | 52.3 [47.6, 57.2] | 26.1 | 0.482 [0.418, 0.542] | 54% |
| 4. 3m + LLM-extracted terms | 50.8 [46.2, 55.6] | 26.2 | 0.525 [0.459, 0.586] | 54% |
| 5. 3m + LLM terms + text embeddings | 50.7 [46.1, 55.7] | 26.0 | 0.523 [0.459, 0.583] | 54% |
| 6. 3m + LLM terms + stated expected closing | 45.0 [40.7, 49.4] | 22.2 | 0.602 [0.524, 0.668] | 59% |

Paired differences (negative MAE = better):

- **LLM-extracted terms** (4 minus 3m): MAE -1.6 days [-2.6, -0.6]; R2 on log days +0.043 [+0.026, +0.062].
- **Text embeddings** (5 minus 4): MAE -0.0 days [-0.4, +0.4]; R2 on log days -0.002 [-0.009, +0.005].
- All text (5 minus 3m): MAE -1.6 days [-2.5, -0.6]; R2 on log days +0.041 [+0.023, +0.061].
- Stated expected closing as a feature (6 minus 4): MAE -5.8 days [-7.3, -4.2]; R2 on log days +0.077 [+0.052, +0.104].

## Pooled rolling origin, 2013-2025, all core deals: 2,057 completed deals

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 71.2 [65.8, 76.2] | 47.0 | -0.005 [-0.018, 0.000] | 35% |
| 1. Median by structure (tender offer / merger) | 61.0 [55.6, 66.2] | 33.0 | 0.281 [0.243, 0.325] | 46% |
| 2. Ridge regression, base terms | 57.4 [52.5, 62.2] | 31.0 | 0.354 [0.294, 0.400] | 49% |
| 3. LightGBM, base terms | 54.6 [50.0, 59.2] | 31.1 | 0.430 [0.390, 0.476] | 49% |
| 3x. LightGBM + extended features | 49.4 [45.2, 53.5] | 27.2 | 0.544 [0.506, 0.582] | 54% |
| 3m. 3x + market context | 49.8 [45.8, 54.0] | 27.2 | 0.539 [0.501, 0.577] | 53% |
| 4. 3m + LLM-extracted terms | 48.3 [44.2, 52.4] | 26.1 | 0.564 [0.526, 0.603] | 54% |
| Final model (chosen by CV inside the training years) | 48.3 [44.2, 52.4] | 26.1 | 0.564 [0.526, 0.603] | 54% |
| R. Log-normal survival model, base terms, broken deals censored | 58.0 [53.3, 62.3] | 31.7 | 0.356 [0.320, 0.392] | 49% |

Paired differences (negative MAE = better):

- LightGBM minus ridge, base terms: MAE -2.8 days [-3.8, -1.8]; R2 on log days +0.077 [+0.046, +0.118].
- Extended features (3x minus 3): MAE -5.2 days [-6.1, -4.4]; R2 on log days +0.114 [+0.093, +0.132].
- Market context (3m minus 3x): MAE +0.5 days [+0.1, +0.9]; R2 on log days -0.005 [-0.012, +0.001].
- LLM-extracted terms (4 minus 3m): MAE -1.6 days [-2.4, -0.7]; R2 on log days +0.025 [+0.010, +0.039].
- Final model minus median by structure: MAE -12.8 days [-15.1, -10.3]; R2 on log days +0.283 [+0.241, +0.321].
- Survival model minus ridge (effect of censoring): MAE +0.6 days [-0.3, +1.5]; R2 on log days +0.002 [-0.022, +0.037].

## Pooled rolling origin, 2013-2025, deals with announcement text: 1,861 completed deals

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 69.8 [64.4, 75.0] | 46.0 | -0.002 [-0.014, 0.001] | 37% |
| 1. Median by structure (tender offer / merger) | 59.1 [53.5, 64.4] | 32.0 | 0.309 [0.269, 0.354] | 47% |
| 2. Ridge regression, base terms | 56.3 [51.3, 60.9] | 28.9 | 0.356 [0.321, 0.396] | 51% |
| 3. LightGBM, base terms | 53.9 [49.3, 58.3] | 30.3 | 0.407 [0.363, 0.459] | 50% |
| 3x. LightGBM + extended features | 48.6 [44.2, 52.9] | 25.7 | 0.521 [0.476, 0.569] | 55% |
| 3m. 3x + market context | 48.8 [44.5, 53.2] | 26.2 | 0.518 [0.473, 0.566] | 55% |
| 4. 3m + LLM-extracted terms | 47.2 [43.1, 51.3] | 24.8 | 0.551 [0.504, 0.600] | 56% |
| 5. 3m + LLM terms + text embeddings | 47.1 [42.9, 51.3] | 24.6 | 0.550 [0.504, 0.601] | 55% |
| 6. 3m + LLM terms + stated expected closing | 41.8 [38.2, 45.0] | 21.3 | 0.616 [0.565, 0.666] | 60% |

Paired differences (negative MAE = better):

- **LLM-extracted terms** (4 minus 3m): MAE -1.6 days [-2.3, -0.9]; R2 on log days +0.033 [+0.023, +0.044].
- **Text embeddings** (5 minus 4): MAE -0.1 days [-0.3, +0.3]; R2 on log days -0.000 [-0.004, +0.003].
- All text (5 minus 3m): MAE -1.7 days [-2.3, -1.1]; R2 on log days +0.033 [+0.020, +0.044].
- Stated expected closing as a feature (6 minus 4): MAE -5.4 days [-6.6, -4.2]; R2 on log days +0.065 [+0.048, +0.083].

- 80% interval from the quantile models, test period: covers 73% of completed deals (10% closed faster, 17% slower); median width 111 days.
- 80% interval from the quantile models, pooled 2013-2025: covers 73% of completed deals (12% closed faster, 15% slower); median width 110 days.

## Does the model beat management guidance? (test period 2019-2025, 807 completed deals with stated guidance, 87% of deals with text)

Guidance = the expected closing stated in the announcement, converted to days (midpoint of the stated window; see `results/closing_guidance_validation.md`).

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 65.0 [59.3, 70.7] | 38.0 | -0.001 [-0.010, -0.000] | 41% |
| Benchmark: the parties' stated expected closing | 46.4 [43.0, 50.2] | 30.5 | 0.507 [0.437, 0.571] | 49% |
| 3m. 3x + market context | 48.3 [43.8, 53.0] | 25.4 | 0.516 [0.461, 0.568] | 55% |
| 4. 3m + LLM-extracted terms | 46.9 [42.5, 51.4] | 25.0 | 0.565 [0.514, 0.611] | 55% |
| 6. 3m + LLM terms + stated expected closing | 39.8 [36.0, 43.7] | 21.6 | 0.676 [0.629, 0.719] | 61% |

- Model without any text (3m) minus guidance: MAE +1.9 days [-2.0, +5.7]; R2 on log days +0.009 [-0.064, +0.085].
- Model with LLM terms (4) minus guidance: MAE +0.4 days [-3.5, +4.1]; R2 on log days +0.058 [-0.012, +0.131].
- Model that also sees the guidance (6) minus guidance: MAE -6.6 days [-9.5, -3.9]; R2 on log days +0.169 [+0.116, +0.228].

Deals closed a median of -14 days relative to guidance; 34% closed later than the stated midpoint.

## Does the model beat management guidance? (pooled 2013-2025, 1,585 completed deals with stated guidance, 85% of deals with text)

Guidance = the expected closing stated in the announcement, converted to days (midpoint of the stated window; see `results/closing_guidance_validation.md`).

| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |
|---|---|---:|---|---:|
| 0. Median of the training period | 65.3 [59.7, 71.0] | 44.0 | -0.001 [-0.009, 0.001] | 38% |
| Benchmark: the parties' stated expected closing | 45.4 [42.0, 48.8] | 30.5 | 0.507 [0.449, 0.559] | 50% |
| 3m. 3x + market context | 45.3 [41.5, 49.6] | 25.2 | 0.579 [0.541, 0.616] | 56% |
| 4. 3m + LLM-extracted terms | 43.6 [39.9, 47.6] | 23.5 | 0.619 [0.583, 0.656] | 57% |
| 6. 3m + LLM terms + stated expected closing | 36.8 [33.6, 40.1] | 19.8 | 0.713 [0.681, 0.745] | 63% |

- Model without any text (3m) minus guidance: MAE -0.1 days [-3.6, +3.5]; R2 on log days +0.072 [+0.013, +0.133].
- Model with LLM terms (4) minus guidance: MAE -1.9 days [-5.2, +1.6]; R2 on log days +0.112 [+0.058, +0.166].
- Model that also sees the guidance (6) minus guidance: MAE -8.7 days [-11.1, -6.1]; R2 on log days +0.206 [+0.164, +0.248].

Deals closed a median of -14 days relative to guidance; 35% closed later than the stated midpoint.

## Segments (test period, final model against the median by structure)

| Segment | Deals | Median days | MAE, median by structure | MAE, final model | Within 30 days, final model |
|---|---:|---:|---:|---:|---:|
| Structure: merger | 855 | 128 | 73.1 | 60.3 | 45% |
| Structure: tender offer | 171 | 45 | 15.4 | 14.4 | 89% |
| Size: small (bottom third by value) | 342 | 107 | 50.7 | 40.5 | 56% |
| Size: large (top third) | 342 | 132 | 87.0 | 71.6 | 44% |
| Size: mid | 342 | 98 | 52.9 | 45.9 | 57% |
| Cross-border: domestic | 840 | 112 | 63.8 | 53.3 | 52% |
| Cross-border: cross-border | 186 | 115 | 62.2 | 49.7 | 54% |
| Regulated industry: other industries | 809 | 96 | 55.9 | 47.0 | 58% |
| Regulated industry: banks | 157 | 169 | 77.5 | 60.6 | 34% |
| Regulated industry: insurance | 19 | 234 | 113.1 | 87.3 | 21% |
| Regulated industry: utilities | 23 | 133 | 139.7 | 111.7 | 35% |
| Regulated industry: telecom and media | 18 | 183 | 133.4 | 125.1 | 28% |

## Product model: the forecast shown in the demo and used in the expected-return step

Where the announcement states an expected closing, the product uses model 6 (which also reads it); otherwise the final model. The switch is decided per origin by the CV inside its training years (model 6 against model 4 on the deals with text). CV preferred model 6 in 14 of 14 origins.

| Evaluation | Completed deals | With stated closing | MAE days, final model | MAE days, product model | Difference [95% CI] | R2 on log days: difference [95% CI] |
|---|---:|---:|---:|---:|---|---|
| Test period 2019-2025 | 1026 | 807 | 52.7 | 47.1 | -5.6 [-6.9, -4.2] | +0.074 [+0.055, +0.095] |
| Pooled rolling origin 2013-2025 | 2057 | 1585 | 48.3 | 43.1 | -5.1 [-6.1, -4.1] | +0.062 [+0.050, +0.075] |

## Calibrated 80% interval (split conformal)

The quantile models are refitted on the training years except the last three; those three years are the calibration set. The interval is widened (or narrowed) on both sides by the conformal quantile of the calibration errors. Nothing from the evaluated years is used.

| Evaluation | Interval | Coverage of completed deals | Closed faster | Closed slower | Median width, days |
|---|---|---:|---:|---:|---:|
| Test period 2019-2025 | quantile models, uncalibrated | 73% | 10% | 17% | 111 |
| Test period 2019-2025 | final model, conformal | 80% | 6% | 14% | 133 |
| Test period 2019-2025 | product model, conformal (shown in the demo) | 80% | 7% | 13% | 114 |
| Pooled rolling origin 2013-2025 | quantile models, uncalibrated | 73% | 12% | 15% | 110 |
| Pooled rolling origin 2013-2025 | final model, conformal | 81% | 7% | 12% | 131 |
| Pooled rolling origin 2013-2025 | product model, conformal (shown in the demo) | 81% | 8% | 11% | 114 |

## Notes

- Final model: the feature set with the lowest CV error inside the training years. Time split: 4; rolling origins: 4 x13.
- The survival model treats broken deals as censored at their cancellation date instead of dropping them. Dropping them is the main specification because the quantity an arbitrageur needs is time to closing given that the deal closes; the survival row shows the effect of that choice.
- A model for year Y is trained on completed deals announced by Y-1, some of which closed during Y. Deals still open at the training cut-off are longer than average, so a live forecast would have slightly fewer long deals to learn from.


# Loss if the deal breaks

Core sample, CRSP daily prices. Return of the target from the close on t1 (the first trading day after the announcement) to the close one and five trading days after the cancellation date, total return including dividends; "net of market" divides by the market's return over the same days (CRSP value-weighted market from the Fama-French file). All considerations: the long target leg is the same in cash and stock deals; no acquirer hedge is modelled.

## What happened

| Outcome | Period | Deals | Mean, 1 day after | Median | 10th pct | 90th pct | Mean, 5 days after | Mean net of market, 1 day | Mean net of market, 5 days | Median days held |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adverse break | 2001-2018 | 146 | -27.6% | -24.1% | -72.6% | +11.2% | -28.8% | -29.2% | -30.7% | 194 |
| Adverse break | 2019-2025 | 38 | -29.1% | -32.5% | -73.0% | +5.9% | -31.9% | -33.2% | -36.0% | 326 |
| Adverse break | all years | 184 | -27.9% | -25.9% | -73.0% | +11.5% | -29.4% | -30.0% | -31.8% | 204 |
| Topped by a higher bid | 2001-2018 | 71 | +19.8% | +14.8% | -0.5% | +39.3% | +20.1% | +17.6% | +17.6% | 70 |
| Topped by a higher bid | 2019-2025 | 17 | +17.6% | +17.3% | +2.5% | +34.9% | +17.0% | +16.8% | +15.6% | 65 |
| Topped by a higher bid | all years | 88 | +19.4% | +14.9% | +0.4% | +37.8% | +19.5% | +17.5% | +17.2% | 70 |

For adverse breaks the fall-back benchmark (the price returns to the pre-announcement close) implies a mean loss of -10.7%; the realised mean is -27.9% raw and -30.0% net of market. All-cash deals: 88 adverse breaks, mean -30.9%; stock and mixed: 96, mean -25.2%. Breaks take months to happen, so the raw return also contains whatever the market and the company did in between; that is why raw and market-adjusted differ.

## Adverse breaks: predicting the loss (raw return to one day after the cancellation; 184 deals)

Leave-one-year-out (every year predicted from all other years):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.288 [0.259, 0.318] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.301 [0.262, 0.341] | +0.012 [-0.018, +0.044] | -0.31 [-0.69, -0.03] | 0.14 |
| One-variable regression on the fall-back | 0.286 [0.255, 0.317] | -0.003 [-0.011, +0.005] | -0.01 [-0.09, +0.05] | -0.06 |
| Ridge regression, all day-1 features | 0.288 [0.258, 0.318] | -0.001 [-0.008, +0.007] | -0.00 [-0.07, +0.05] | -0.02 |
| Shallow boosted trees, all day-1 features | 0.281 [0.253, 0.310] | -0.007 [-0.020, +0.005] | +0.04 [-0.05, +0.12] | 0.16 |

Time split (trained on 146 breaks announced 2001-2018, tested on 38 announced 2019-2025):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.310 [0.228, 0.413] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.301 [0.227, 0.394] | -0.008 [-0.069, +0.055] | +0.12 [-0.46, +0.37] | 0.47 |
| One-variable regression on the fall-back | 0.309 [0.227, 0.412] | -0.001 [-0.002, -0.000] | +0.01 [+0.00, +0.01] | 0.47 |
| Ridge regression, all day-1 features | 0.291 [0.205, 0.396] | -0.019 [-0.034, -0.004] | +0.05 [-0.02, +0.16] | 0.25 |
| Shallow boosted trees, all day-1 features | 0.279 [0.198, 0.378] | -0.031 [-0.054, -0.010] | +0.13 [+0.01, +0.28] | 0.40 |

One-variable fit on all 184 deals: realised = -0.249 +0.28 x fall-back. A slope of 1 with intercept 0 would mean prices return exactly to the unaffected price.

## Adverse breaks: predicting the loss (return net of market; 183 deals)

Leave-one-year-out (every year predicted from all other years):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.257 [0.229, 0.286] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.295 [0.258, 0.335] | +0.038 [+0.006, +0.070] | -0.47 [-0.91, -0.14] | 0.13 |
| One-variable regression on the fall-back | 0.256 [0.227, 0.285] | -0.002 [-0.008, +0.005] | -0.01 [-0.08, +0.05] | -0.05 |
| Ridge regression, all day-1 features | 0.260 [0.230, 0.291] | +0.003 [-0.009, +0.015] | -0.04 [-0.15, +0.06] | 0.03 |
| Shallow boosted trees, all day-1 features | 0.249 [0.221, 0.279] | -0.008 [-0.021, +0.005] | +0.04 [-0.06, +0.13] | 0.17 |

Time split (trained on 146 breaks announced 2001-2018, tested on 37 announced 2019-2025):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.283 [0.201, 0.379] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.310 [0.234, 0.393] | +0.027 [-0.042, +0.093] | -0.02 [-0.74, +0.31] | 0.43 |
| One-variable regression on the fall-back | 0.282 [0.200, 0.378] | -0.001 [-0.002, -0.000] | +0.00 [+0.00, +0.01] | 0.43 |
| Ridge regression, all day-1 features | 0.268 [0.187, 0.363] | -0.015 [-0.031, -0.001] | +0.07 [-0.01, +0.17] | 0.30 |
| Shallow boosted trees, all day-1 features | 0.250 [0.172, 0.342] | -0.033 [-0.063, -0.004] | +0.16 [+0.01, +0.31] | 0.44 |

One-variable fit on all 183 deals: realised = -0.275 +0.24 x fall-back. A slope of 1 with intercept 0 would mean prices return exactly to the unaffected price.

Rolling origin, 2013-2025 (81 adverse breaks): each year's forecast comes from breaks announced and cancelled before that year, and the forecast type is chosen by leave-one-year-out inside those. Chosen: training mean x8, shallow boosted trees, all day-1 features x5. MAE 0.277 against 0.285 for the pooled mean of the same deals; correlation with the realised return 0.35.

## Deals topped by a higher bid: predicting the gain (raw return to one day after the cancellation; 88 deals)

Leave-one-year-out (every year predicted from all other years):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.163 [0.121, 0.223] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.366 [0.302, 0.447] | +0.203 [+0.157, +0.249] | -1.85 [-4.47, -1.16] | -0.28 |
| One-variable regression on the fall-back | 0.175 [0.132, 0.234] | +0.012 [-0.005, +0.028] | -0.03 [-0.46, +0.10] | 0.03 |
| Ridge regression, all day-1 features | 0.164 [0.119, 0.229] | +0.002 [-0.016, +0.027] | -0.06 [-0.79, +0.15] | 0.04 |
| Shallow boosted trees, all day-1 features | 0.154 [0.113, 0.213] | -0.009 [-0.027, +0.012] | +0.08 [-0.09, +0.28] | 0.24 |

Time split (trained on 71 breaks announced 2001-2018, tested on 17 announced 2019-2025):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.111 [0.079, 0.140] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.349 [0.270, 0.428] | +0.238 [+0.147, +0.329] | -8.04 [-16.23, -3.85] | 0.24 |
| One-variable regression on the fall-back | 0.152 [0.103, 0.204] | +0.042 [-0.001, +0.088] | -1.08 [-2.65, -0.09] | -0.24 |
| Ridge regression, all day-1 features | 0.161 [0.090, 0.272] | +0.050 [-0.014, +0.149] | -3.05 [-8.30, +0.14] | -0.28 |
| Shallow boosted trees, all day-1 features | 0.147 [0.102, 0.190] | +0.036 [-0.012, +0.081] | -0.86 [-1.89, -0.07] | 0.05 |

One-variable fit on all 88 deals: realised = +0.104 -0.56 x fall-back. A slope of 1 with intercept 0 would mean prices return exactly to the unaffected price.

## Deals topped by a higher bid: predicting the gain (return net of market; 88 deals)

Leave-one-year-out (every year predicted from all other years):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.164 [0.119, 0.223] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.346 [0.282, 0.431] | +0.182 [+0.141, +0.224] | -1.70 [-4.01, -1.09] | -0.28 |
| One-variable regression on the fall-back | 0.180 [0.136, 0.239] | +0.016 [-0.001, +0.034] | -0.03 [-0.47, +0.10] | 0.05 |
| Ridge regression, all day-1 features | 0.179 [0.126, 0.250] | +0.015 [-0.012, +0.061] | -0.29 [-2.49, +0.14] | 0.04 |
| Shallow boosted trees, all day-1 features | 0.153 [0.110, 0.210] | -0.011 [-0.031, +0.010] | +0.11 [-0.06, +0.31] | 0.30 |

Time split (trained on 71 breaks announced 2001-2018, tested on 17 announced 2019-2025):

| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |
|---|---|---|---|---:|
| Training mean | 0.123 [0.073, 0.197] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] | n/a |
| Falls back to the unaffected price (P0 / close(t1) - 1) | 0.341 [0.239, 0.467] | +0.218 [+0.136, +0.302] | -4.02 [-16.73, -2.32] | -0.02 |
| One-variable regression on the fall-back | 0.152 [0.093, 0.226] | +0.029 [-0.011, +0.073] | -0.27 [-2.68, +0.15] | 0.02 |
| Ridge regression, all day-1 features | 0.140 [0.079, 0.217] | +0.017 [-0.021, +0.068] | -0.28 [-3.26, +0.13] | 0.01 |
| Shallow boosted trees, all day-1 features | 0.137 [0.086, 0.194] | +0.015 [-0.028, +0.055] | +0.05 [-1.71, +0.39] | 0.36 |

One-variable fit on all 88 deals: realised = +0.085 -0.56 x fall-back. A slope of 1 with intercept 0 would mean prices return exactly to the unaffected price.

Rolling origin, 2013-2025 (33 deals topped by a higher bid): each year's forecast comes from breaks announced and cancelled before that year, and the forecast type is chosen by leave-one-year-out inside those. Chosen: shallow boosted trees, all day-1 features x13. MAE 0.144 against 0.116 for the pooled mean of the same deals; correlation with the realised return 0.30.

## Reading

- The numbers above are returns of an unhedged long position entered one day after the announcement. They are the loss leg of an arbitrageur's payoff, not a prediction of where the price settles in the long run.
- Samples are small and the losses are widely dispersed, so the intervals are wide. A forecast whose MAE interval against the training mean includes zero has not been shown to help.


# Market benchmark: does the model add information beyond the market?

Model: lgbm_ext at the announcement date. Market: the day-1 spread and `p_mkt` = (offer - close(t1)) / (offer - P0) from CRSP daily prices, where t1 is the first trading day after the announcement and P0 the close on the last trading day before it. Core sample, all-cash deals, target adverse break. `p_mkt` ignores time value and assumes a broken deal falls back to exactly P0; `p_mkt_carry` discounts the offer at the 3-month T-bill rate over the expected remaining time.

The market sees one more trading day than the model. The market logistic, the stack and everything else on the market side are fitted on earlier years only.

## Test period 2019-2025: 623 priced all-cash deals, 22 adverse breaks

Fitted on 1257 training-period deals (37 adverse breaks) with out-of-fold model scores. The earlier version on monthly closes had 552 priced test deals and 16 adverse breaks (`results/monthly/market_benchmark.md`). Intervals resample deals.

| Score | ROC AUC [95% CI] | Within-year AUC [95% CI] | Adverse breaks in riskiest 20% |
|---|---|---|---:|
| Market: p_mkt | 0.643 [0.523, 0.755] | 0.683 [0.559, 0.796] | 0.27 |
| Market: p_mkt with carry | 0.676 [0.566, 0.778] | 0.704 [0.592, 0.807] | 0.36 |
| Market: raw day-1 spread | 0.696 [0.575, 0.808] | 0.742 [0.627, 0.841] | 0.45 |
| Market: logistic on p_mkt and spread (fitted on earlier years) | 0.644 [0.523, 0.758] | 0.682 [0.556, 0.796] | 0.27 |
| Model (announcement date) | 0.632 [0.519, 0.734] | 0.648 [0.523, 0.768] | 0.27 |
| Model + market (stack fitted on earlier years) | 0.664 [0.560, 0.762] | 0.688 [0.571, 0.797] | 0.41 |

- Model added to the market (stack minus market logistic): **+0.020 [-0.081, +0.124]** AUC; positive in 66% of resamples. Within-year AUC: +0.006 [-0.102, +0.130].
- Market added to the model (stack minus model): **+0.032 [-0.041, +0.126]** AUC; positive in 76% of resamples. Within-year AUC: +0.040 [-0.037, +0.150].
- Model minus p_mkt: **-0.011 [-0.166, +0.146]** AUC; positive in 45% of resamples. Within-year AUC: -0.035 [-0.208, +0.136].
- Stack minus p_mkt: **+0.021 [-0.080, +0.124]** AUC; positive in 67% of resamples. Within-year AUC: +0.005 [-0.103, +0.128].

## Pooled out-of-time 2013-2025: 1205 priced all-cash deals, 38 adverse breaks

Each deal is scored by a model fitted on earlier years only (13 origins, `results/pooled_oot.md`); the market logistic and the stack are refitted for every test year on the years before it. Intervals resample announcement years.

| Score | ROC AUC [95% CI] | Within-year AUC [95% CI] | Adverse breaks in riskiest 20% |
|---|---|---|---:|
| Market: p_mkt | 0.721 [0.642, 0.800] | 0.744 [0.646, 0.839] | 0.47 |
| Market: p_mkt with carry | 0.734 [0.660, 0.808] | 0.757 [0.664, 0.849] | 0.50 |
| Market: raw day-1 spread | 0.746 [0.679, 0.824] | 0.786 [0.679, 0.878] | 0.47 |
| Market: logistic on p_mkt and spread (fitted on earlier years) | 0.642 [0.508, 0.772] | 0.746 [0.649, 0.842] | 0.47 |
| Model (announcement date) | 0.599 [0.476, 0.702] | 0.615 [0.449, 0.777] | 0.32 |
| Model + market (stack fitted on earlier years) | 0.614 [0.517, 0.736] | 0.688 [0.562, 0.804] | 0.32 |

- Model added to the market (stack minus market logistic): **-0.028 [-0.194, +0.107]** AUC (resampling deals instead of years: [-0.136, +0.072]); positive in 36% of resamples. Within-year AUC: -0.058 [-0.200, +0.070].
- Market added to the model (stack minus model): **+0.015 [-0.037, +0.098]** AUC (resampling deals instead of years: [-0.051, +0.082]); positive in 72% of resamples. Within-year AUC: +0.074 [+0.003, +0.159].
- Model minus p_mkt: **-0.121 [-0.282, +0.005]** AUC (resampling deals instead of years: [-0.242, -0.005]); positive in 3% of resamples. Within-year AUC: -0.129 [-0.335, +0.053].
- Stack minus p_mkt: **-0.107 [-0.235, +0.012]** AUC (resampling deals instead of years: [-0.207, -0.016]); positive in 5% of resamples. Within-year AUC: -0.056 [-0.197, +0.071].

## Robustness

| Set | Deals | Adverse breaks | p_mkt | Model | Stack | Stack minus market logistic [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| Test 2019-2025, all considerations (stock and mixed spreads from an implied exchange ratio) | 902 | 31 | 0.613 | 0.598 | 0.627 | +0.012 [-0.062, +0.088] |
| Pooled 2013-2025, all considerations (stock and mixed spreads from an implied exchange ratio) | 1811 | 61 | 0.697 | 0.590 | 0.633 | -0.018 [-0.107, +0.056] |
| Test 2019-2025, all-cash, offer confirmed by or taken from the press release | 528 | 19 | 0.651 | 0.616 | 0.653 | +0.000 [-0.123, +0.127] |
| Pooled 2013-2025, all-cash, offer confirmed by or taken from the press release | 1013 | 33 | 0.730 | 0.598 | 0.614 | -0.037 [-0.180, +0.104] |

## Different decision time: day 1 after announcement

**Not comparable with the rows above as a like-for-like model step.** The models above decide on the announcement date. This step adds what the market shows one trading day later (day-1 spread, `p_mkt`, `p_mkt` with carry, days to t1) to the final model's features; deals without a usable price keep missing values. Hyper-parameters are those of the final model.

| Evaluation set | Deals | Adverse breaks | Final model, announcement date | Final model + day-1 market features | Difference [95% CI] | Day-1 model minus p_mkt [95% CI] |
|---|---:|---:|---:|---:|---|---|
| Test 2019-2025, all core deals | 1083 | 40 | 0.633 | 0.636 [0.552, 0.717] | +0.002 [-0.056, +0.072] | n/a |
| Test 2019-2025, priced all-cash deals | 623 | 22 | 0.632 | 0.646 [0.549, 0.739] | +0.015 [-0.076, +0.117] | +0.004 [-0.093, +0.111] |
| Pooled out-of-time 2013-2025, all core deals | 2176 | 84 | 0.609 | 0.655 [0.596, 0.717] | +0.046 [+0.006, +0.082] | n/a |
| Pooled out-of-time 2013-2025, priced all-cash deals | 1205 | 38 | 0.599 | 0.661 [0.544, 0.752] | +0.062 [+0.010, +0.118] | -0.060 [-0.192, +0.036] |

Intervals resample deals for the test period and announcement years for the pooled set.

Reading the tables: an increment whose interval includes zero means the data cannot tell whether one source adds to the other.


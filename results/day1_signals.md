# Day-1 market signals beyond the target spread

Question: does anything observable at the close of the first trading day after the announcement predict adverse breaks better than the target spread alone?

**Specification, fixed before any out-of-time result was looked at.** Two models, both on the same inputs: the market-implied probability, the day-1 spread, the announcement-date model score, and ten day-1 signals (acquirer return to t1, net of market; acquirer return x stock share; abnormal target volume, announcement day; abnormal target volume, t1; target closed above the offer; pre-announcement run-up (30 days); target idiosyncratic volatility; spread minus recent median spread; deal value / acquirer market cap; product overlap x deal size). Model 1 is a logistic regression (L2, C = 0.1, standardised inputs, missing-value indicators); model 2 is a LightGBM with 3 leaves and 200 trees. No hyper-parameter was tuned. Both are refitted for every test year on earlier years only. **Specifications tried: these two.** They were evaluated once; nothing was changed after seeing the results below.

Sample: priced all-cash core deals, as in `results/market_benchmark.md`. Benchmark: the market-implied probability `p_mkt` by itself. Intervals resample announcement years (13 for the pooled set, 7 for the test period, so the test-period intervals are rough).

Share of priced all-cash deals with each signal: acquirer return to t1, net of market 38%, acquirer return x stock share 38%, abnormal target volume, announcement day 97%, abnormal target volume, t1 97%, target closed above the offer 100%, pre-announcement run-up (30 days) 97%, target idiosyncratic volatility 96%, spread minus recent median spread 100%, deal value / acquirer market cap 43%, product overlap x deal size 32%. For all-cash deals the acquirer-return-times-stock-share term is zero by construction; it matters only in the robustness set with stock and mixed deals.

## All-cash deals (main)

### Pooled out-of-time 2013-2025: 1205 deals, 38 adverse breaks

| Score | ROC AUC [95% CI by year] | Adverse breaks in riskiest 20% [95% CI] |
|---|---|---|
| Market-only benchmark: p_mkt | 0.721 [0.642, 0.800] | 0.47 [0.31, 0.61] |
| Market logistic (p_mkt and spread) | 0.642 [0.508, 0.772] | 0.47 [0.29, 0.63] |
| Day-1 model 1: logistic regression | 0.596 [0.492, 0.719] | 0.34 [0.17, 0.51] |
| Day-1 model 2: LightGBM | 0.660 [0.570, 0.752] | 0.39 [0.26, 0.53] |
| Probability passed to the expected-return step | 0.631 [0.532, 0.734] | 0.32 [0.21, 0.50] |

| Difference | ROC AUC [95% CI by year] | [95% CI by deal] | Riskiest-20% capture [95% CI by year] |
|---|---|---|---|
| Logistic day-1 model minus p_mkt | -0.125 [-0.246, -0.010] | [-0.220, -0.037] | -0.13 [-0.31, +0.07] |
| LightGBM day-1 model minus p_mkt | -0.061 [-0.143, +0.011] | [-0.133, +0.019] | -0.08 [-0.26, +0.12] |
| Logistic day-1 model minus market logistic | -0.046 [-0.196, +0.069] | [-0.146, +0.051] | -0.13 [-0.35, +0.08] |
| LightGBM day-1 model minus market logistic | +0.018 [-0.101, +0.115] | [-0.066, +0.115] | -0.08 [-0.28, +0.14] |
| Probability passed on minus p_mkt | -0.089 [-0.168, -0.009] | [-0.152, -0.030] | -0.16 [-0.23, +0.00] |

### Test period 2019-2025: 623 deals, 22 adverse breaks

| Score | ROC AUC [95% CI by year] | Adverse breaks in riskiest 20% [95% CI] |
|---|---|---|
| Market-only benchmark: p_mkt | 0.643 [0.552, 0.726] | 0.27 [0.13, 0.47] |
| Market logistic (p_mkt and spread) | 0.644 [0.555, 0.727] | 0.27 [0.13, 0.45] |
| Day-1 model 1: logistic regression | 0.613 [0.461, 0.718] | 0.32 [0.11, 0.54] |
| Day-1 model 2: LightGBM | 0.625 [0.433, 0.714] | 0.36 [0.22, 0.49] |
| Probability passed to the expected-return step | 0.625 [0.433, 0.714] | 0.36 [0.22, 0.49] |

| Difference | ROC AUC [95% CI by year] | [95% CI by deal] | Riskiest-20% capture [95% CI by year] |
|---|---|---|---|
| Logistic day-1 model minus p_mkt | -0.030 [-0.174, +0.043] | [-0.160, +0.109] | +0.05 [-0.12, +0.12] |
| LightGBM day-1 model minus p_mkt | -0.017 [-0.165, +0.042] | [-0.138, +0.101] | +0.09 [-0.04, +0.20] |
| Logistic day-1 model minus market logistic | -0.031 [-0.176, +0.043] | [-0.161, +0.108] | +0.05 [-0.12, +0.12] |
| LightGBM day-1 model minus market logistic | -0.018 [-0.165, +0.040] | [-0.139, +0.101] | +0.09 [-0.04, +0.20] |
| Probability passed on minus p_mkt | -0.017 [-0.165, +0.042] | [-0.138, +0.101] | +0.09 [-0.04, +0.20] |

## All considerations (robustness; spreads of stock and mixed deals from an implied exchange ratio)

### Pooled out-of-time 2013-2025: 1811 deals, 61 adverse breaks

| Score | ROC AUC [95% CI by year] | Adverse breaks in riskiest 20% [95% CI] |
|---|---|---|
| Market-only benchmark: p_mkt | 0.697 [0.616, 0.775] | 0.49 [0.31, 0.63] |
| Market logistic (p_mkt and spread) | 0.652 [0.541, 0.749] | 0.44 [0.31, 0.57] |
| Day-1 model 1: logistic regression | 0.628 [0.535, 0.724] | 0.39 [0.25, 0.50] |
| Day-1 model 2: LightGBM | 0.665 [0.585, 0.743] | 0.38 [0.25, 0.52] |
| Probability passed to the expected-return step | 0.658 [0.570, 0.745] | 0.41 [0.25, 0.55] |

| Difference | ROC AUC [95% CI by year] | [95% CI by deal] | Riskiest-20% capture [95% CI by year] |
|---|---|---|---|
| Logistic day-1 model minus p_mkt | -0.069 [-0.151, +0.012] | [-0.133, -0.004] | -0.10 [-0.23, +0.02] |
| LightGBM day-1 model minus p_mkt | -0.032 [-0.099, +0.030] | [-0.094, +0.031] | -0.11 [-0.27, +0.05] |
| Logistic day-1 model minus market logistic | -0.024 [-0.119, +0.055] | [-0.089, +0.042] | -0.05 [-0.18, +0.02] |
| LightGBM day-1 model minus market logistic | +0.013 [-0.073, +0.087] | [-0.052, +0.079] | -0.07 [-0.22, +0.06] |
| Probability passed on minus p_mkt | -0.038 [-0.107, +0.031] | [-0.091, +0.011] | -0.08 [-0.26, +0.08] |

### Test period 2019-2025: 902 deals, 31 adverse breaks

| Score | ROC AUC [95% CI by year] | Adverse breaks in riskiest 20% [95% CI] |
|---|---|---|
| Market-only benchmark: p_mkt | 0.613 [0.516, 0.718] | 0.29 [0.14, 0.45] |
| Market logistic (p_mkt and spread) | 0.615 [0.517, 0.720] | 0.29 [0.14, 0.45] |
| Day-1 model 1: logistic regression | 0.557 [0.463, 0.662] | 0.23 [0.08, 0.39] |
| Day-1 model 2: LightGBM | 0.576 [0.470, 0.678] | 0.29 [0.15, 0.43] |
| Probability passed to the expected-return step | 0.576 [0.470, 0.678] | 0.29 [0.15, 0.43] |

| Difference | ROC AUC [95% CI by year] | [95% CI by deal] | Riskiest-20% capture [95% CI by year] |
|---|---|---|---|
| Logistic day-1 model minus p_mkt | -0.057 [-0.156, +0.010] | [-0.154, +0.058] | -0.06 [-0.15, +0.00] |
| LightGBM day-1 model minus p_mkt | -0.037 [-0.106, -0.001] | [-0.125, +0.056] | +0.00 [-0.09, +0.09] |
| Logistic day-1 model minus market logistic | -0.058 [-0.156, +0.009] | [-0.156, +0.056] | -0.06 [-0.15, +0.00] |
| LightGBM day-1 model minus market logistic | -0.039 [-0.107, -0.003] | [-0.127, +0.054] | +0.00 [-0.09, +0.09] |
| Probability passed on minus p_mkt | -0.037 [-0.106, -0.001] | [-0.125, +0.056] | +0.00 [-0.09, +0.09] |

## What the training-period cross-validation preferred (all-cash deals)

For every origin the two day-1 models and `p_mkt` are compared on time-ordered validation blocks inside the training years. The day-1 model is passed to the expected-return step for that year only if its mean validation AUC is higher than that of `p_mkt`; otherwise the market-implied probability (through the market logistic) is used.

| Origin | Training deals | Adverse breaks | Folds | CV AUC p_mkt | CV AUC logistic | CV AUC LightGBM | Passed to the expected-return step |
|---|---:|---:|---:|---:|---:|---:|---|
| year 2013 | 1065 | 33 | 2 | 0.453 | 0.671 | 0.754 | day-1 model (LightGBM) |
| year 2014 | 1280 | 45 | 3 | 0.581 | 0.777 | 0.789 | day-1 model (LightGBM) |
| year 2015 | 1311 | 41 | 3 | 0.573 | 0.730 | 0.761 | day-1 model (LightGBM) |
| year 2016 | 1349 | 39 | 3 | 0.528 | 0.673 | 0.715 | day-1 model (LightGBM) |
| year 2017 | 1380 | 38 | 3 | 0.637 | 0.772 | 0.787 | day-1 model (LightGBM) |
| year 2018 | 1355 | 42 | 3 | 0.633 | 0.727 | 0.741 | day-1 model (LightGBM) |
| year 2019 | 1257 | 37 | 3 | 0.634 | 0.704 | 0.709 | day-1 model (LightGBM) |
| year 2020 | 1131 | 23 | 2 | 0.808 | 0.711 | 0.666 | market-implied probability |
| year 2021 | 1116 | 29 | 2 | 0.749 | 0.583 | 0.614 | market-implied probability |
| year 2022 | 1165 | 29 | 2 | 0.756 | 0.522 | 0.544 | market-implied probability |
| year 2023 | 1138 | 32 | 2 | 0.758 | 0.590 | 0.589 | market-implied probability |
| year 2024 | 1123 | 33 | 2 | 0.686 | 0.600 | 0.609 | market-implied probability |
| year 2025 | 1106 | 35 | 2 | 0.642 | 0.612 | 0.636 | market-implied probability |
| test period (2001-2018) | 1257 | 37 | 3 | 0.634 | 0.704 | 0.709 | day-1 model (LightGBM) |

**Used in the expected-return step:** the day-1 model in 7 of 13 pooled origins and the day-1 model for the test period.

## Logistic model fitted on 2001-2018 (standardised coefficients, all-cash deals)

| Input | Coefficient |
|---|---:|
| market-implied probability | +0.49 |
| announcement-date model score (log-odds) | +0.45 |
| target closed above the offer | +0.27 |
| abnormal target volume, announcement day | -0.22 |
| deal value / acquirer market cap is missing | +0.18 |
| pre-announcement run-up (30 days) | -0.10 |
| deal value / acquirer market cap | +0.10 |
| product overlap x deal size | +0.09 |
| acquirer return to t1, net of market | -0.07 |
| target idiosyncratic volatility is missing | -0.04 |
| abnormal target volume, t1 | -0.04 |
| acquirer return to t1, net of market is missing | +0.03 |
| acquirer return x stock share is missing | +0.03 |
| target idiosyncratic volatility | +0.03 |
| product overlap x deal size is missing | +0.02 |
| abnormal target volume, announcement day is missing | -0.02 |
| abnormal target volume, t1 is missing | -0.02 |
| day-1 spread | -0.02 |
| pre-announcement run-up (30 days) is missing | -0.01 |
| spread minus recent median spread | +0.00 |
| acquirer return x stock share | +0.00 |
| spread minus recent median spread is missing | +0.00 |

Positive = more adverse-break risk. Coefficients are shrunk (C = 0.1) and describe the training period only.


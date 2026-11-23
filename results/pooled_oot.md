# Pooled out-of-time evaluation, 2013-2025

Every deal announced from 2013 to 2025 is scored by models fitted only on deals announced in earlier years (13 origins, one test year each, no overlap). Hyper-parameters, the one rule and the choice of feature set are all re-selected inside each origin's training years. Target: adverse break. Sample: core (reverse mergers excluded).

Why: the 2019-2025 test period has 40 adverse breaks. Pooling 13 out-of-time years gives more, at the price of mixing 13 different fitted models. `ROC AUC` pools raw scores across years (what a manager using one cut-off over time would experience); `within-year AUC` only compares deals announced in the same year and is immune to drift in score levels between origins.

Intervals: bootstrap resampling announcement years (13 clusters, 2,000 draws). The deal-level interval is shown for the paired differences as a comparison; it ignores that deals of one year share a model and a market, and is narrower for that reason.

## All core deals: 2,176 deals, 84 adverse breaks (2013-2025)

| Model | ROC AUC [95% CI by year] | Within-year AUC | PR AUC | Breaks in riskiest 10% | Breaks in riskiest 20% |
|---|---|---|---|---|---|
| 0. Base rate (training-period break rate) | 0.455 [0.376, 0.531] | 0.500 [0.500, 0.500] | 0.036 [0.027, 0.047] | 0.09 [0.04, 0.12] | 0.15 [0.09, 0.25] |
| 1. One rule | 0.479 [0.399, 0.564] | 0.527 [0.465, 0.616] | 0.039 [0.030, 0.049] | 0.09 [0.05, 0.15] | 0.19 [0.12, 0.27] |
| 2. Logistic, base terms | 0.590 [0.525, 0.651] | 0.609 [0.524, 0.707] | 0.052 [0.035, 0.074] | 0.15 [0.07, 0.23] | 0.29 [0.20, 0.38] |
| 3. LightGBM, base terms | 0.579 [0.519, 0.641] | 0.596 [0.521, 0.694] | 0.052 [0.036, 0.078] | 0.15 [0.09, 0.26] | 0.30 [0.20, 0.41] |
| 3x. LightGBM + extended features | 0.604 [0.536, 0.682] | 0.621 [0.528, 0.743] | 0.056 [0.038, 0.091] | 0.15 [0.09, 0.26] | 0.33 [0.21, 0.48] |
| 3m. 3x + market context | 0.576 [0.513, 0.655] | 0.620 [0.512, 0.735] | 0.050 [0.036, 0.079] | 0.14 [0.07, 0.24] | 0.29 [0.16, 0.43] |
| Final-model procedure (3, 3x or 3m, chosen by CV inside each origin) | 0.609 [0.538, 0.686] | 0.620 [0.527, 0.737] | 0.055 [0.038, 0.083] | 0.17 [0.09, 0.25] | 0.31 [0.21, 0.45] |
| 5f. 3m + text groups (has_text, keywords, embeddings, LLM terms) | 0.599 [0.540, 0.663] | 0.634 [0.544, 0.724] | 0.056 [0.040, 0.084] | 0.17 [0.08, 0.26] | 0.25 [0.16, 0.34] |

Paired differences:

| Comparison | ROC AUC [95% CI by year] | ROC AUC [95% CI by deal] | Within-year AUC [by year] | Breaks in riskiest 20% [by year] |
|---|---|---|---|---|
| 3 minus 2 | -0.011 [-0.035, +0.012] | -0.011 [-0.057, +0.035] | -0.013 [-0.049, +0.022] | +0.01 [-0.06, +0.09] |
| 3x minus 3 | +0.025 [-0.025, +0.076] | +0.025 [-0.028, +0.079] | +0.025 [-0.021, +0.084] | +0.04 [-0.07, +0.13] |
| 3m minus 3x | -0.028 [-0.054, -0.002] | -0.028 [-0.054, -0.003] | -0.001 [-0.032, +0.023] | -0.05 [-0.11, +0.04] |
| Final-model procedure minus 2 | +0.018 [-0.050, +0.088] | +0.018 [-0.042, +0.076] | +0.011 [-0.064, +0.097] | +0.02 [-0.10, +0.15] |
| Final-model procedure minus 3 | +0.030 [-0.026, +0.086] | +0.030 [-0.024, +0.084] | +0.024 [-0.021, +0.083] | +0.01 [-0.08, +0.11] |
| 5f minus 3m | +0.022 [-0.014, +0.059] | +0.022 [-0.015, +0.058] | +0.014 [-0.030, +0.056] | -0.04 [-0.17, +0.08] |

## Deals with announcement text: 1,968 deals, 74 adverse breaks (2013-2025)

| Model | ROC AUC [95% CI by year] | Within-year AUC | PR AUC | Breaks in riskiest 10% | Breaks in riskiest 20% |
|---|---|---|---|---|---|
| 0. Base rate (training-period break rate) | 0.430 [0.359, 0.501] | 0.500 [0.500, 0.500] | 0.033 [0.024, 0.044] | 0.08 [0.03, 0.12] | 0.13 [0.06, 0.24] |
| 1. One rule | 0.488 [0.426, 0.564] | 0.530 [0.462, 0.627] | 0.039 [0.029, 0.050] | 0.08 [0.02, 0.14] | 0.19 [0.09, 0.27] |
| 2. Logistic, base terms | 0.608 [0.550, 0.668] | 0.617 [0.534, 0.708] | 0.053 [0.037, 0.081] | 0.15 [0.09, 0.22] | 0.27 [0.17, 0.39] |
| 3. LightGBM, base terms | 0.559 [0.496, 0.621] | 0.574 [0.483, 0.682] | 0.050 [0.033, 0.083] | 0.18 [0.12, 0.28] | 0.27 [0.18, 0.39] |
| 3x. LightGBM + extended features | 0.596 [0.521, 0.677] | 0.603 [0.509, 0.736] | 0.051 [0.036, 0.077] | 0.18 [0.07, 0.30] | 0.30 [0.18, 0.42] |
| 3m. 3x + market context | 0.589 [0.518, 0.668] | 0.595 [0.503, 0.721] | 0.048 [0.036, 0.074] | 0.12 [0.04, 0.26] | 0.27 [0.15, 0.42] |
| 4. 3m + keywords + TF-IDF | 0.566 [0.502, 0.639] | 0.613 [0.502, 0.727] | 0.046 [0.034, 0.073] | 0.11 [0.04, 0.20] | 0.23 [0.12, 0.36] |
| 5. 3m + keywords + embeddings | 0.568 [0.512, 0.634] | 0.604 [0.512, 0.716] | 0.049 [0.037, 0.088] | 0.14 [0.04, 0.26] | 0.19 [0.10, 0.32] |
| 5L. 3m + keywords + LLM terms | 0.572 [0.503, 0.650] | 0.615 [0.510, 0.732] | 0.049 [0.034, 0.081] | 0.14 [0.06, 0.25] | 0.27 [0.14, 0.45] |
| 5A. 3m + all text groups | 0.588 [0.530, 0.649] | 0.614 [0.512, 0.719] | 0.048 [0.035, 0.077] | 0.12 [0.05, 0.22] | 0.27 [0.16, 0.39] |
| 6a. Text only: TF-IDF logistic | 0.611 [0.548, 0.682] | 0.639 [0.559, 0.729] | 0.054 [0.039, 0.078] | 0.15 [0.11, 0.22] | 0.35 [0.25, 0.45] |

Paired differences:

| Comparison | ROC AUC [95% CI by year] | ROC AUC [95% CI by deal] | Within-year AUC [by year] | Breaks in riskiest 20% [by year] |
|---|---|---|---|---|
| 3x minus 3 | +0.037 [-0.021, +0.106] | +0.037 [-0.023, +0.093] | +0.029 [-0.014, +0.100] | +0.03 [-0.05, +0.11] |
| 4 minus 3m | -0.023 [-0.081, +0.039] | -0.023 [-0.073, +0.023] | +0.018 [-0.038, +0.066] | -0.04 [-0.15, +0.07] |
| 5 minus 3m | -0.021 [-0.077, +0.032] | -0.021 [-0.070, +0.022] | +0.009 [-0.048, +0.049] | -0.08 [-0.20, +0.05] |
| 5L minus 3m | -0.017 [-0.053, +0.016] | -0.017 [-0.048, +0.011] | +0.019 [-0.019, +0.047] | +0.00 [-0.06, +0.10] |
| 5A minus 3m | -0.001 [-0.074, +0.065] | -0.001 [-0.060, +0.058] | +0.019 [-0.034, +0.070] | +0.00 [-0.11, +0.11] |
| 5A minus 3 | +0.029 [-0.009, +0.065] | +0.029 [-0.029, +0.085] | +0.040 [-0.023, +0.106] | +0.00 [-0.10, +0.10] |

## By test year (final-model procedure)

| Test year | Deals | Adverse breaks | Chosen feature set | ROC AUC | Mean score | Break rate |
|---|---:|---:|---|---:|---:|---:|
| 2013 | 148 | 5 | 3x | 0.547 | 0.021 | 0.034 |
| 2014 | 168 | 6 | 3x | 0.769 | 0.033 | 0.036 |
| 2015 | 207 | 12 | 3x | 0.610 | 0.034 | 0.058 |
| 2016 | 195 | 3 | 3m | 0.766 | 0.025 | 0.015 |
| 2017 | 188 | 9 | 3x | 0.554 | 0.035 | 0.048 |
| 2018 | 187 | 9 | 3x | 0.550 | 0.034 | 0.048 |
| 2019 | 161 | 1 | 3x | 0.500 | 0.049 | 0.006 |
| 2020 | 103 | 9 | 3x | 0.605 | 0.038 | 0.087 |
| 2021 | 193 | 7 | 3x | 0.474 | 0.037 | 0.036 |
| 2022 | 148 | 9 | 3m | 0.835 | 0.042 | 0.061 |
| 2023 | 148 | 4 | 3x | 0.917 | 0.045 | 0.027 |
| 2024 | 145 | 5 | 3x | 0.676 | 0.044 | 0.034 |
| 2025 | 185 | 5 | 3x | 0.409 | 0.048 | 0.027 |

## Embargo check

A model for year Y trains on deals announced by Y-1, a few of which were still pending on 1 January of Y. Training on deals announced by Y-2 instead (same specification) gives a pooled AUC of 0.612 [0.544, 0.686], a change of +0.003 [-0.014, +0.019]. The convention does not drive the result.


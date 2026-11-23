# Market benchmark on monthly closes (fallback without CRSP)

The headline version uses CRSP daily prices: `results/market_benchmark.md`. This file is what the repository produces from the Capital IQ monthly closes alone. It uses the export's final offer, so later price increases inflate some spreads (see `results/crsp_prices.md`).

Model: lgbm_ext (announcement-date features only; trained on 2001-2018). Market: `p_mkt` = (offer - close(t1)) / (offer - P0), the share of the premium the market has not yet paid, where t1 is the first monthly grid date after the announcement and P0 the price one day before it. `p_mkt` ignores time value and assumes a broken deal falls back to exactly P0; `p_mkt_carry` discounts the offer at the 3-month T-bill rate over the expected remaining time to closing.

**Caveat that favours the market:** t1 is up to a month after the announcement (median about two weeks), so the market price reflects news the announcement-date model cannot see. A model that only matches the market here would still be useful on the announcement day; a model that adds to it despite the head start is stronger evidence.

### All-cash deals, core sample (main), target `label_adverse`

Test deals with usable prices: 552 (16 positives). Stack fitted on 1200 training-period deals (33 positives) with out-of-fold model scores.

| Score | ROC AUC [95% CI] | Positives in riskiest 20% |
|---|---|---:|
| Market: p_mkt | 0.619 [0.458, 0.771] | 0.25 |
| Market: p_mkt with carry | 0.646 [0.487, 0.790] | 0.31 |
| Market: raw spread | 0.659 [0.474, 0.815] | 0.56 |
| Market: logistic on p_mkt + spread (fitted on train) | 0.614 [0.437, 0.763] | 0.25 |
| Model (announcement date) | 0.599 [0.457, 0.731] | 0.25 |
| Model + market (stack fitted on train) | 0.641 [0.511, 0.766] | 0.25 |

- Incremental gain of the model over the market (stack minus market-only): **+0.027 AUC** [95% CI -0.096, +0.153]; positive in 67% of resamples.
- Incremental gain of the market over the model (stack minus model-only): +0.042 AUC [-0.061, +0.177].
- Model minus p_mkt: -0.019 AUC [-0.232, +0.182].
- Stack coefficients (logit of model score, p_mkt, spread): 0.69, 1.92, 0.09.

### All-cash deals, core sample (main), target `label`

Test deals with usable prices: 552 (21 positives). Stack fitted on 1200 training-period deals (63 positives) with out-of-fold model scores.

| Score | ROC AUC [95% CI] | Positives in riskiest 20% |
|---|---|---:|
| Market: p_mkt | 0.603 [0.462, 0.738] | 0.24 |
| Market: p_mkt with carry | 0.631 [0.503, 0.756] | 0.33 |
| Market: raw spread | 0.659 [0.504, 0.792] | 0.52 |
| Market: logistic on p_mkt + spread (fitted on train) | 0.599 [0.453, 0.735] | 0.24 |
| Model (announcement date) | 0.614 [0.496, 0.726] | 0.24 |
| Model + market (stack fitted on train) | 0.637 [0.527, 0.746] | 0.29 |

- Incremental gain of the model over the market (stack minus market-only): **+0.038 AUC** [95% CI -0.071, +0.144]; positive in 75% of resamples.
- Incremental gain of the market over the model (stack minus model-only): +0.023 AUC [-0.069, +0.125].
- Model minus p_mkt: +0.011 AUC [-0.158, +0.175].
- Stack coefficients (logit of model score, p_mkt, spread): 0.62, 1.55, 0.18.

### All considerations, core sample (robustness), target `label_adverse`

Test deals with usable prices: 849 (28 positives). Stack fitted on 1863 training-period deals (65 positives) with out-of-fold model scores.

| Score | ROC AUC [95% CI] | Positives in riskiest 20% |
|---|---|---:|
| Market: p_mkt | 0.592 [0.462, 0.713] | 0.36 |
| Market: p_mkt with carry | 0.612 [0.487, 0.724] | 0.32 |
| Market: raw spread | 0.613 [0.483, 0.739] | 0.39 |
| Market: logistic on p_mkt + spread (fitted on train) | 0.596 [0.474, 0.715] | 0.32 |
| Model (announcement date) | 0.610 [0.509, 0.710] | 0.29 |
| Model + market (stack fitted on train) | 0.626 [0.517, 0.728] | 0.36 |

- Incremental gain of the model over the market (stack minus market-only): **+0.030 AUC** [95% CI -0.016, +0.075]; positive in 89% of resamples.
- Incremental gain of the market over the model (stack minus model-only): +0.016 AUC [-0.093, +0.130].
- Model minus p_mkt: +0.018 AUC [-0.125, +0.165].
- Stack coefficients (logit of model score, p_mkt, spread): 0.32, 2.03, 0.51.

### All considerations, core sample (robustness), target `label`

Test deals with usable prices: 849 (40 positives). Stack fitted on 1863 training-period deals (110 positives) with out-of-fold model scores.

| Score | ROC AUC [95% CI] | Positives in riskiest 20% |
|---|---|---:|
| Market: p_mkt | 0.545 [0.439, 0.645] | 0.30 |
| Market: p_mkt with carry | 0.566 [0.461, 0.663] | 0.28 |
| Market: raw spread | 0.580 [0.473, 0.689] | 0.35 |
| Market: logistic on p_mkt + spread (fitted on train) | 0.548 [0.438, 0.653] | 0.30 |
| Model (announcement date) | 0.619 [0.540, 0.695] | 0.25 |
| Model + market (stack fitted on train) | 0.602 [0.521, 0.687] | 0.33 |

- Incremental gain of the model over the market (stack minus market-only): **+0.054 AUC** [95% CI +0.001, +0.108]; positive in 98% of resamples.
- Incremental gain of the market over the model (stack minus model-only): -0.017 AUC [-0.096, +0.064].
- Model minus p_mkt: +0.074 AUC [-0.044, +0.199].
- Stack coefficients (logit of model score, p_mkt, spread): 0.41, 1.53, 0.23.

Reading the table: with this few breaks among priced test deals the intervals are wide. A positive incremental gain whose interval includes zero means the data cannot tell whether the model adds to the market.


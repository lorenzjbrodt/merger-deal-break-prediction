# Rolling-origin evaluation

Train on announcements through year T, test on T+1 to T+2, for T = 2012 to 2023. Core sample, adverse-break target, full sample (text features missing where a deal has no text). Hyper-parameters are those selected in the main run (training years 2001-2018) and are held fixed across origins, so origins before 2018 carry a mild look-ahead in the hyper-parameters only; all model fitting uses data up to T.

| Train through | Test years | Test deals | Adverse breaks | AUC 2 | AUC 3 | AUC 3m | AUC 5f | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 3m | Top-20% capture 5f |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 2013-2014 | 316 | 11 | 0.694 | 0.633 | 0.706 | 0.694 | 0.45 | 0.45 | 0.55 | 0.36 |
| 2013 | 2014-2015 | 375 | 18 | 0.694 | 0.644 | 0.649 | 0.631 | 0.33 | 0.22 | 0.17 | 0.22 |
| 2014 | 2015-2016 | 402 | 15 | 0.603 | 0.596 | 0.634 | 0.656 | 0.20 | 0.27 | 0.27 | 0.33 |
| 2015 | 2016-2017 | 383 | 12 | 0.485 | 0.547 | 0.590 | 0.601 | 0.08 | 0.33 | 0.25 | 0.42 |
| 2016 | 2017-2018 | 375 | 18 | 0.583 | 0.558 | 0.595 | 0.615 | 0.28 | 0.28 | 0.28 | 0.33 |
| 2017 | 2018-2019 | 348 | 10 | 0.609 | 0.554 | 0.568 | 0.599 | 0.30 | 0.30 | 0.20 | 0.20 |
| 2018 | 2019-2020 | 264 | 10 | 0.494 | 0.527 | 0.589 | 0.630 | 0.10 | 0.20 | 0.20 | 0.30 |
| 2019 | 2020-2021 | 296 | 16 | 0.632 | 0.615 | 0.576 | 0.610 | 0.31 | 0.31 | 0.19 | 0.25 |
| 2020 | 2021-2022 | 341 | 16 | 0.677 | 0.654 | 0.664 | 0.675 | 0.38 | 0.38 | 0.44 | 0.31 |
| 2021 | 2022-2023 | 296 | 13 | 0.749 | 0.757 | 0.833 | 0.792 | 0.54 | 0.54 | 0.69 | 0.46 |
| 2022 | 2023-2024 | 293 | 9 | 0.709 | 0.736 | 0.780 | 0.799 | 0.44 | 0.67 | 0.44 | 0.56 |
| 2023 | 2024-2025 | 330 | 10 | 0.550 | 0.521 | 0.527 | 0.546 | 0.10 | 0.30 | 0.20 | 0.30 |
| **Mean** | | | | **0.623** | **0.612** | **0.643** | **0.654** | **0.29** | **0.35** | **0.32** | **0.34** |
| Std. dev. across origins | | | | 0.081 | 0.074 | 0.087 | 0.073 | 0.14 | 0.13 | 0.16 | 0.10 |

Origins overlap (each test year appears in two windows), so the rows are not independent. 3 beats the logistic benchmark in 4 of 12 origins; 3m beats the logistic benchmark in 7 of 12 origins; 5f beats the logistic benchmark in 6 of 12 origins.
Model keys: 2 = Logistic, base tabular; 3 = LightGBM, base tabular (as in the base run); 3m = LightGBM + extended features + market context; 5f = 3m + has_text + keywords + embeddings + LLM terms.


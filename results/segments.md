# Segment results on the test set

Core sample, adverse-break target. AUC and capture are computed inside each segment; segments with fewer than 5 adverse breaks show n/a. Segment numbers rest on few breaks and are noisy. Deal-size terciles use cut-offs from the training years.

## By consideration

| Segment | Deals | Adverse breaks | Rate | AUC 2 | AUC 3 | AUC 3x | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 3x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed/other | 187 | 5 | 2.7% | 0.391 | 0.518 | 0.621 | 0.00 | 0.20 | 0.20 |
| cash | 686 | 24 | 3.5% | 0.626 | 0.621 | 0.649 | 0.38 | 0.29 | 0.33 |
| stock | 210 | 11 | 5.2% | 0.646 | 0.683 | 0.578 | 0.36 | 0.45 | 0.27 |

## By deal size tercile

| Segment | Deals | Adverse breaks | Rate | AUC 2 | AUC 3 | AUC 3x | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 3x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid (<= $1,554m) | 297 | 9 | 3.0% | 0.719 | 0.720 | 0.703 | 0.56 | 0.44 | 0.33 |
| large (> $1,554m) | 534 | 21 | 3.9% | 0.620 | 0.643 | 0.639 | 0.19 | 0.38 | 0.33 |
| small (<= $350m) | 252 | 10 | 4.0% | 0.469 | 0.488 | 0.557 | 0.10 | 0.30 | 0.20 |

## By period

| Segment | Deals | Adverse breaks | Rate | AUC 2 | AUC 3 | AUC 3x | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 3x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-2020 | 264 | 10 | 3.8% | 0.494 | 0.527 | 0.579 | 0.10 | 0.20 | 0.20 |
| 2021-2022 | 341 | 16 | 4.7% | 0.689 | 0.694 | 0.693 | 0.44 | 0.38 | 0.44 |
| 2023-2025 | 478 | 14 | 2.9% | 0.614 | 0.651 | 0.622 | 0.21 | 0.36 | 0.29 |

## By buyer type

| Segment | Deals | Adverse breaks | Rate | AUC 2 | AUC 3 | AUC 3x | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 3x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| financial buyer | 261 | 6 | 2.3% | 0.783 | 0.814 | 0.820 | 0.50 | 0.67 | 0.67 |
| strategic / other | 822 | 34 | 4.1% | 0.574 | 0.597 | 0.626 | 0.24 | 0.32 | 0.29 |

Model keys: 2 = Logistic, base tabular; 3 = LightGBM, base tabular (as in the base run); 3x = LightGBM + extended deal, buyer and target features.


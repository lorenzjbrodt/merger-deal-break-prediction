# Segment results on the test set

AUC and capture are computed inside each segment; segments with fewer than 5 breaks show n/a. Segment-level numbers rest on few breaks and are noisy. Deal-size terciles use cut-offs from the training years.

## Full sample, by consideration

| Segment | Deals | Breaks | Break rate | AUC 2 | AUC 3 | AUC 5f | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 5f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed/other | 224 | 21 | 9.4% | 0.656 | 0.700 | 0.587 | 0.43 | 0.38 | 0.33 |
| cash | 686 | 30 | 4.4% | 0.515 | 0.551 | 0.591 | 0.23 | 0.20 | 0.20 |
| stock | 448 | 58 | 12.9% | 0.554 | 0.497 | 0.478 | 0.24 | 0.16 | 0.19 |

## Full sample, by deal size tercile (total transaction value)

| Segment | Deals | Breaks | Break rate | AUC 2 | AUC 3 | AUC 5f | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 5f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid (<= $1,555m) | 410 | 35 | 8.5% | 0.572 | 0.669 | 0.570 | 0.34 | 0.31 | 0.26 |
| large (> $1,555m) | 588 | 41 | 7.0% | 0.559 | 0.589 | 0.587 | 0.24 | 0.17 | 0.20 |
| small (<= $350m) | 360 | 33 | 9.2% | 0.540 | 0.609 | 0.558 | 0.27 | 0.21 | 0.27 |

## Full sample, by period

| Segment | Deals | Breaks | Break rate | AUC 2 | AUC 3 | AUC 5f | Top-20% capture 2 | Top-20% capture 3 | Top-20% capture 5f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-2020 | 313 | 17 | 5.4% | 0.406 | 0.440 | 0.498 | 0.18 | 0.00 | 0.24 |
| 2021-2022 | 482 | 44 | 9.1% | 0.561 | 0.619 | 0.559 | 0.30 | 0.27 | 0.25 |
| 2023-2025 | 563 | 48 | 8.5% | 0.624 | 0.694 | 0.627 | 0.35 | 0.44 | 0.29 |

## Text subset, by consideration

| Segment | Deals | Breaks | Break rate | AUC 3 | AUC 4 | AUC 5 | Top-20% capture 3 | Top-20% capture 4 | Top-20% capture 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed/other | 181 | 21 | 11.6% | 0.674 | 0.690 | 0.653 | 0.57 | 0.33 | 0.33 |
| cash | 630 | 26 | 4.1% | 0.543 | 0.611 | 0.572 | 0.27 | 0.35 | 0.19 |
| stock | 264 | 52 | 19.7% | 0.694 | 0.694 | 0.632 | 0.35 | 0.29 | 0.29 |

## Text subset, by deal size tercile (total transaction value)

| Segment | Deals | Breaks | Break rate | AUC 3 | AUC 4 | AUC 5 | Top-20% capture 3 | Top-20% capture 4 | Top-20% capture 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid (<= $1,555m) | 316 | 30 | 9.5% | 0.772 | 0.805 | 0.687 | 0.57 | 0.50 | 0.33 |
| large (> $1,555m) | 520 | 40 | 7.7% | 0.590 | 0.679 | 0.617 | 0.28 | 0.42 | 0.25 |
| small (<= $350m) | 239 | 29 | 12.1% | 0.758 | 0.638 | 0.692 | 0.59 | 0.31 | 0.41 |

## Text subset, by period

| Segment | Deals | Breaks | Break rate | AUC 3 | AUC 4 | AUC 5 | Top-20% capture 3 | Top-20% capture 4 | Top-20% capture 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-2020 | 237 | 14 | 5.9% | 0.509 | 0.656 | 0.562 | 0.07 | 0.36 | 0.21 |
| 2021-2022 | 356 | 40 | 11.2% | 0.702 | 0.693 | 0.663 | 0.47 | 0.42 | 0.30 |
| 2023-2025 | 482 | 45 | 9.3% | 0.766 | 0.756 | 0.705 | 0.62 | 0.40 | 0.40 |

Model keys: 2 = Logistic, tabular; 3 = LightGBM, tabular; 5f = LightGBM, tabular + has_text + keywords + embeddings; 4 = LightGBM, tabular + keywords + TF-IDF; 5 = LightGBM, tabular + keywords + embeddings


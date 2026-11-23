# Antitrust overlap: product similarity and industry shares

Feature group added to the break model and the time-to-closing model. Sources: Hoberg-Phillips TNIC-3 product similarity of acquirer and target (public data, latest year before the announcement, data end in 2023) and Compustat sales shares by 4-digit SIC industry. The acquirer must be a Compustat firm; for private, sponsor and most foreign buyers the pair features are missing, which the tree models treat as its own category.

## Coverage (core sample)

- Acquirer matched to Compustat: 2,500 of 4,468 deals.
- TNIC pair status known (both firms in the TNIC universe): 2,074; of these 1,440 are TNIC-3 peers.
- Same 4-digit SIC known: 2,499; same industry in 1,212.

## What the raw data say (core sample, all years; descriptive)

| Group | Deals | Adverse break rate | Median days to closing (completed deals) |
|---|---:|---:|---:|
| Acquirer not in Compustat (private, sponsor, foreign) | 1968 | 5.0% | 98 |
| Both in TNIC, not peers | 634 | 3.5% | 93 |
| TNIC peers, similarity below the peer median | 719 | 4.2% | 106 |
| TNIC peers, similarity above the peer median | 721 | 5.4% | 150 |
| Same 4-digit SIC | 1212 | 4.5% | 132 |
| Different 4-digit SIC | 1287 | 3.6% | 105 |

## Break model (adverse break, core sample)

| Evaluation | Deals | Adverse breaks | Final model | + overlap group | Difference [95% CI] | + SIC and sales measures only | Difference [95% CI] |
|---|---:|---:|---:|---:|---|---:|---|
| Test period 2019-2025 | 1083 | 40 | 0.633 | 0.628 | -0.005 [-0.028, +0.014] | 0.633 | -0.000 [-0.022, +0.019] |
| Test period, acquirer in Compustat | 597 | 22 | 0.617 | 0.613 | -0.005 [-0.041, +0.027] | n/a | n/a |
| Pooled out-of-time 2013-2025 | 2176 | 84 | 0.609 | 0.615 | +0.006 [-0.010, +0.019] | 0.613 | +0.004 [-0.017, +0.019] |

CV AUC inside the training years (2001-2018): final model 0.682, with the overlap group 0.686, with SIC and sales measures only 0.692. Intervals: bootstrap over deals (test period) and over announcement years (pooled).

## Time-to-closing model (completed deals)

| Evaluation | Completed deals | MAE days, final model | MAE days, + overlap group | Difference [95% CI] | R2 on log days: difference [95% CI] |
|---|---:|---:|---:|---|---|
| Test period 2019-2025 | 1026 | 52.7 | 52.6 | -0.0 [-0.3, +0.3] | -0.001 [-0.005, +0.004] |
| Pooled out-of-time 2013-2025 | 2057 | 48.3 | 48.1 | -0.1 [-0.4, +0.1] | +0.001 [-0.001, +0.004] |

Same hyper-parameters as the final time-to-closing model of each origin; only the feature group is added.


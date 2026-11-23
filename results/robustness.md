# Robustness checks

Model: 3x. LightGBM + extended deal, buyer and target features, adverse-break target, test period 2019-2025.

| Variant | Test deals | Adverse breaks | ROC AUC | PR AUC | Breaks in riskiest 10% | Breaks in riskiest 20% |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1083 | 40 | 0.633 | 0.068 | 0.15 | 0.35 |
| exclude_resolved_within_10_days (evaluation only) | 1079 | 38 | 0.615 | 0.051 | 0.11 | 0.32 |
| exclude_revised_offers (train and test) | 1009 | 28 | 0.590 | 0.040 | 0.11 | 0.32 |
| exclude_revised_offers (evaluation only) | 1009 | 28 | 0.634 | 0.059 | 0.14 | 0.39 |
| no missingness signal (A2, median-imputed 3m) | 1083 | 40 | 0.620 | 0.059 | 0.17 | 0.33 |
| with leakage suspects (A1) | 1083 | 40 | 0.629 | 0.061 | 0.17 | 0.33 |
| sample as specified, reverse mergers included | 1358 | 88 | 0.747 | 0.170 | 0.35 | 0.60 |
| reverse mergers only (model trained on the specified sample) | 275 | 48 | 0.747 | 0.291 | 0.15 | 0.31 |

- *Revised offers*: deals whose Capital IQ notes mention a raised, lowered or amended offer after announcement. For these the exported price, premium and value can be final terms. Excluding them removes that possible look-ahead.
- *Resolved within 10 days*: deals that closed or were cancelled almost immediately, where the text window could overlap the outcome.
- *No missingness signal*: Capital IQ fills fewer fields for cancelled deals, so missing values are themselves predictive. A2 removes that signal.
- *Leakage suspects*: Deal Responses and Deal Attitude can be updated after announcement; A1 shows what they would add.


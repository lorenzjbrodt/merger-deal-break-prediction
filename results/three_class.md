# Three-class view: closes, topped by a higher bid, adverse break

A multiclass LightGBM with the features and hyper-parameters of the final tabular model, trained on 2001-2018 (core sample).

| Class | Train deals | Test deals | One-vs-rest AUC on test [95% CI] | Mean binary adverse-break score |
|---|---:|---:|---|---:|
| Closes | 3148 | 1026 | 0.610 [0.540, 0.678] | 0.046 |
| Topped by a higher bid | 79 | 17 | 0.615 [0.506, 0.708] | 0.062 |
| Adverse break | 158 | 40 | 0.610 [0.532, 0.701] | 0.062 |

Mean predicted class probabilities by true class (rows: true class; columns: P(closes), P(topped), P(adverse)):

| True class | P(closes) | P(topped) | P(adverse) |
|---|---:|---:|---:|
| Closes | 0.940 | 0.017 | 0.043 |
| Topped | 0.918 | 0.018 | 0.064 |
| Adverse break | 0.926 | 0.017 | 0.057 |

Topped deals are good outcomes for an arbitrageur (the price usually rises), which is why the adverse-break target excludes them. The number of topped deals in the test period is small, so its AUC interval is wide.


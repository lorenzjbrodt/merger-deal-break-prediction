# Ablation by feature group

Text subset, core sample, adverse-break target (test: 975 deals, 35 adverse breaks). The reference is LightGBM on all groups (model 5A) with its CV-selected hyper-parameters (num_leaves 6, 200 trees), kept fixed for every row. "Without" drops one group from the full model; "only" trains on that group alone. Brackets: bootstrap 95% intervals.

| Feature group | AUC of the group alone | AUC without the group | Loss in AUC when dropped (all minus without) |
|---|---|---|---|
| Deal terms and structure | 0.603 [0.511, 0.683] | 0.567 [0.476, 0.654] | +0.025 [-0.005, +0.055] |
| Premium and price | 0.613 [0.508, 0.707] | 0.605 [0.514, 0.687] | -0.014 [-0.032, +0.004] |
| Conditions | 0.497 [0.399, 0.595] | 0.598 [0.504, 0.684] | -0.007 [-0.019, +0.006] |
| Buyer and target financials | 0.559 [0.471, 0.643] | 0.603 [0.505, 0.690] | -0.012 [-0.039, +0.016] |
| Market context | 0.469 [0.371, 0.560] | 0.608 [0.516, 0.693] | -0.017 [-0.035, +0.001] |
| Keyword flags | 0.549 [0.444, 0.645] | 0.613 [0.521, 0.696] | -0.021 [-0.037, -0.006] |
| TF-IDF text | 0.598 [0.503, 0.687] | 0.606 [0.516, 0.689] | -0.014 [-0.051, +0.017] |
| Frozen embeddings | 0.616 [0.521, 0.707] | 0.578 [0.483, 0.668] | +0.013 [-0.011, +0.034] |
| LLM-extracted terms | 0.596 [0.499, 0.688] | 0.592 [0.499, 0.679] | -0.001 [-0.020, +0.018] |

All groups together: AUC 0.591 [0.499, 0.676].
The market-spread group is dated t1 (up to a month after announcement) and is evaluated separately on the price-covered subset in `results/market_benchmark.md`.


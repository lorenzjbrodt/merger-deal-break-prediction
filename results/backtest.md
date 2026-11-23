# Backtest with daily marks (headline impact)

Core sample, CRSP daily prices. Entry: target close on t1, the first trading day after the announcement. Completed deals are held to the last trading day and earn CRSP total returns including dividends and the delisting return (the cash paid at closing). Broken deals are sold at the close of the first trading day after the cancellation date; a variant sells five trading days after it. Equal weights. Annual return = total P&L divided by total capital-years, so `$ per $100 million per year` assumes the capital is continuously redeployed into deals of the same kind.

**Limits.** No transaction costs, no financing costs, no borrow, no position or capacity limits; every deal can be bought at the day-1 close in equal size. A few deals without announcement text keep the export's final offer in the spread (`results/crsp_prices.md`); this affects the spread-based rules, not the returns. Model: lgbm_ext for the test period, the same procedure refitted each year for the pooled set. "Riskiest 10%" ranks within the test period, and within each announcement year in the pooled set. Intervals: bootstrap over deals (test period) and over announcement years (pooled).

## All-cash deals (main)

### Test period 2019-2025

623 deals: 595 closed, 22 adverse breaks, 6 topped by a higher bid. Hold-all annual return +6.8% [95% CI +3.8%, +9.7%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 623 | 22 | 6 | +2.62% | 92 | +6.8% | +6.76 | reference |
| Skip riskiest 10% by model | 561 | 20 | 5 | +2.67% | 90 | +7.0% | +7.04 | +0.28 [-0.56, +1.41] |
| Skip riskiest 20% by model | 498 | 16 | 4 | +2.42% | 86 | +6.8% | +6.78 | +0.01 [-1.92, +1.88] |
| Model-versus-market rule (threshold from earlier years) | 335 | 10 | 2 | +4.39% | 89 | +11.1% | +11.10 | +4.34 [+1.64, +7.21] |
| Reference: same number of deals, highest market-implied risk first (no model) | 335 | 16 | 2 | +4.66% | 122 | +9.3% | +9.33 | +2.57 [+0.61, +4.56] |
| Reference: skip the widest 10% of spreads | 561 | 16 | 5 | +1.18% | 89 | +3.4% | +3.38 | -3.38 [-5.30, -1.55] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+1.78 points [95% CI -0.32, +4.41]**; the two hold 61% of the same deals. Mean day-1 spread of the deals the rule holds: 5.9%, of all deals: 3.7%.

Per-deal returns by outcome:

| Outcome | Deals | Mean return per deal | Median return | Worst | Best | Median holding days | Mean model score | Mean p_mkt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed | 595 | +4.0% | +1.8% | -44.6% | +97.9% | 91 | 0.039 | 0.16 |
| Topped by a higher bid | 6 | +16.2% | +15.7% | +3.5% | +33.8% | 68 | 0.053 | 0.10 |
| Adverse break | 22 | -37.9% | -44.9% | -83.4% | +61.7% | 396 | 0.046 | 0.30 |

### Pooled out-of-time 2013-2025

1205 deals: 1150 closed, 38 adverse breaks, 17 topped by a higher bid. Hold-all annual return +6.4% [95% CI +4.5%, +8.2%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 1205 | 38 | 17 | +2.27% | 88 | +6.4% | +6.35 | reference |
| Skip riskiest 10% by model | 1085 | 32 | 16 | +2.39% | 85 | +7.0% | +7.02 | +0.67 [-0.10, +1.37] |
| Skip riskiest 20% by model | 965 | 27 | 14 | +2.13% | 83 | +6.5% | +6.49 | +0.13 [-0.79, +1.10] |
| Model-versus-market rule (threshold from earlier years) | 676 | 24 | 9 | +3.13% | 86 | +8.7% | +8.71 | +2.36 [+0.31, +4.64] |
| Reference: same number of deals, highest market-implied risk first (no model) | 676 | 33 | 6 | +3.55% | 110 | +7.9% | +7.88 | +1.53 [+0.42, +2.87] |
| Reference: skip the widest 10% of spreads | 1085 | 24 | 16 | +1.15% | 84 | +3.5% | +3.54 | -2.81 [-4.24, -1.55] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+0.83 points [95% CI -0.99, +2.88]**; the two hold 65% of the same deals. Mean day-1 spread of the deals the rule holds: 4.0%, of all deals: 2.7%.

Per-deal returns by outcome:

| Outcome | Deals | Mean return per deal | Median return | Worst | Best | Median holding days | Mean model score | Mean p_mkt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed | 1150 | +3.2% | +1.4% | -44.6% | +97.9% | 88 | 0.030 | 0.13 |
| Topped by a higher bid | 17 | +14.6% | +14.2% | +3.5% | +33.8% | 58 | 0.027 | 0.10 |
| Adverse break | 38 | -31.2% | -28.9% | -83.4% | +61.7% | 360 | 0.036 | 0.34 |

## All-cash deals with a day-1 spread between -5% and +25%

### Test period 2019-2025

580 deals: 557 closed, 18 adverse breaks, 5 topped by a higher bid. Hold-all annual return +4.5% [95% CI +1.9%, +6.9%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 580 | 18 | 5 | +1.70% | 92 | +4.5% | +4.52 | reference |
| Skip riskiest 10% by model | 522 | 16 | 4 | +1.69% | 90 | +4.6% | +4.60 | +0.08 [-0.79, +1.26] |
| Skip riskiest 20% by model | 464 | 14 | 4 | +1.50% | 86 | +4.2% | +4.23 | -0.29 [-1.55, +1.28] |
| Model-versus-market rule (threshold from earlier years) | 308 | 6 | 2 | +3.13% | 86 | +8.3% | +8.29 | +3.77 [+1.38, +6.21] |
| Reference: same number of deals, highest market-implied risk first (no model) | 308 | 12 | 2 | +3.17% | 122 | +6.6% | +6.56 | +2.04 [+0.35, +3.97] |
| Reference: skip the widest 10% of spreads | 522 | 13 | 4 | +0.98% | 89 | +2.9% | +2.94 | -1.58 [-3.12, -0.02] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+1.73 points [95% CI -0.50, +3.96]**; the two hold 59% of the same deals. Mean day-1 spread of the deals the rule holds: 3.6%, of all deals: 2.7%.

### Pooled out-of-time 2013-2025

1144 deals: 1097 closed, 34 adverse breaks, 13 topped by a higher bid. Hold-all annual return +4.7% [95% CI +3.3%, +5.9%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 1144 | 34 | 13 | +1.66% | 88 | +4.7% | +4.73 | reference |
| Skip riskiest 10% by model | 1029 | 29 | 12 | +1.74% | 86 | +5.1% | +5.14 | +0.41 [-0.13, +1.00] |
| Skip riskiest 20% by model | 916 | 25 | 11 | +1.65% | 83 | +5.1% | +5.09 | +0.36 [-0.65, +1.32] |
| Model-versus-market rule (threshold from earlier years) | 643 | 20 | 8 | +2.19% | 84 | +6.3% | +6.31 | +1.58 [-0.43, +3.89] |
| Reference: same number of deals, highest market-implied risk first (no model) | 643 | 28 | 6 | +2.51% | 109 | +5.8% | +5.77 | +1.03 [+0.17, +1.97] |
| Reference: skip the widest 10% of spreads | 1029 | 21 | 11 | +1.11% | 83 | +3.6% | +3.56 | -1.17 [-2.16, -0.22] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+0.54 points [95% CI -1.28, +2.68]**; the two hold 63% of the same deals. Mean day-1 spread of the deals the rule holds: 2.5%, of all deals: 2.1%.

## All-cash deals, broken deals sold five trading days after the cancellation date

### Test period 2019-2025

623 deals: 595 closed, 22 adverse breaks, 6 topped by a higher bid. Hold-all annual return +6.5% [95% CI +3.5%, +9.5%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 623 | 22 | 6 | +2.53% | 92 | +6.5% | +6.52 | reference |
| Skip riskiest 10% by model | 561 | 20 | 5 | +2.55% | 90 | +6.7% | +6.71 | +0.20 [-0.62, +1.31] |
| Skip riskiest 20% by model | 498 | 16 | 4 | +2.29% | 87 | +6.4% | +6.40 | -0.11 [-2.05, +1.81] |
| Model-versus-market rule (threshold from earlier years) | 335 | 10 | 2 | +4.26% | 89 | +10.8% | +10.76 | +4.25 [+1.53, +7.17] |
| Reference: same number of deals, highest market-implied risk first (no model) | 335 | 16 | 2 | +4.48% | 122 | +8.9% | +8.94 | +2.43 [+0.45, +4.41] |
| Reference: skip the widest 10% of spreads | 561 | 16 | 5 | +1.11% | 89 | +3.2% | +3.19 | -3.33 [-5.25, -1.47] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+1.82 points [95% CI -0.34, +4.50]**; the two hold 61% of the same deals. Mean day-1 spread of the deals the rule holds: 5.9%, of all deals: 3.7%.

### Pooled out-of-time 2013-2025

1205 deals: 1150 closed, 38 adverse breaks, 17 topped by a higher bid. Hold-all annual return +6.2% [95% CI +4.4%, +8.0%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 1205 | 38 | 17 | +2.23% | 88 | +6.2% | +6.22 | reference |
| Skip riskiest 10% by model | 1085 | 32 | 16 | +2.34% | 86 | +6.9% | +6.86 | +0.64 [-0.12, +1.31] |
| Skip riskiest 20% by model | 965 | 27 | 14 | +2.09% | 83 | +6.3% | +6.34 | +0.12 [-0.79, +1.08] |
| Model-versus-market rule (threshold from earlier years) | 676 | 24 | 9 | +3.08% | 86 | +8.6% | +8.57 | +2.34 [+0.26, +4.70] |
| Reference: same number of deals, highest market-implied risk first (no model) | 676 | 33 | 6 | +3.45% | 110 | +7.6% | +7.64 | +1.42 [+0.27, +2.76] |
| Reference: skip the widest 10% of spreads | 1085 | 24 | 16 | +1.12% | 84 | +3.4% | +3.44 | -2.78 [-4.25, -1.48] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+0.93 points [95% CI -0.95, +3.08]**; the two hold 65% of the same deals. Mean day-1 spread of the deals the rule holds: 4.0%, of all deals: 2.7%.

## All considerations, including stock and mixed deals, unhedged (robustness)

### Test period 2019-2025

902 deals: 857 closed, 31 adverse breaks, 14 topped by a higher bid. Hold-all annual return +9.3% [95% CI +6.3%, +12.3%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 902 | 31 | 14 | +3.85% | 109 | +9.3% | +9.29 | reference |
| Skip riskiest 10% by model | 812 | 28 | 13 | +3.80% | 106 | +9.3% | +9.28 | -0.01 [-1.51, +1.45] |
| Skip riskiest 20% by model | 722 | 22 | 10 | +3.64% | 103 | +9.4% | +9.37 | +0.08 [-1.83, +2.15] |
| Model-versus-market rule (threshold from earlier years) | 405 | 13 | 3 | +6.18% | 111 | +14.2% | +14.21 | +4.92 [+1.71, +8.03] |
| Reference: same number of deals, highest market-implied risk first (no model) | 405 | 17 | 4 | +7.11% | 153 | +12.9% | +12.86 | +3.56 [+1.03, +6.17] |
| Reference: skip the widest 10% of spreads | 812 | 24 | 13 | +2.50% | 104 | +6.5% | +6.50 | -2.79 [-4.34, -1.28] |

Model-versus-market rule minus the model-free reference with the same number of deals: **+1.35 points [95% CI -1.38, +3.93]**; the two hold 60% of the same deals. Mean day-1 spread of the deals the rule holds: 7.6%, of all deals: 4.3%.

### Pooled out-of-time 2013-2025

1811 deals: 1722 closed, 61 adverse breaks, 28 topped by a higher bid. Hold-all annual return +8.9% [95% CI +5.9%, +11.5%].

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 1811 | 61 | 28 | +3.56% | 109 | +8.9% | +8.86 | reference |
| Skip riskiest 10% by model | 1631 | 52 | 26 | +3.75% | 107 | +9.6% | +9.60 | +0.74 [-0.31, +1.81] |
| Skip riskiest 20% by model | 1449 | 45 | 22 | +3.57% | 103 | +9.5% | +9.52 | +0.67 [-0.42, +1.82] |
| Model-versus-market rule (threshold from earlier years) | 988 | 39 | 12 | +4.63% | 107 | +11.4% | +11.36 | +2.50 [+0.72, +4.43] |
| Reference: same number of deals, highest market-implied risk first (no model) | 988 | 47 | 11 | +5.79% | 144 | +11.4% | +11.37 | +2.52 [+1.76, +3.35] |
| Reference: skip the widest 10% of spreads | 1631 | 43 | 25 | +2.30% | 104 | +6.2% | +6.25 | -2.61 [-3.46, -1.79] |

Model-versus-market rule minus the model-free reference with the same number of deals: **-0.02 points [95% CI -1.47, +1.50]**; the two hold 64% of the same deals. Mean day-1 spread of the deals the rule holds: 5.4%, of all deals: 3.6%.

## The model-versus-market rule

The rule skips a deal when the calibrated model probability exceeds the market logistic by more than a threshold (in log-odds). The threshold is the training-period quantile that maximised the training-period annual return, searched over skipping 0% to 50% of deals in steps of 5 points.

- Test period: the training years chose to skip the top 40% of deals by that gap (training-period gain +1.77 points a year).
- Pooled set: the chosen share varies by origin year: 50%, 45%, 40%, 50%, 35%, 50%, 40%, 45%, 50%, 50%, 50%, 50%, 50%.

A threshold picked as the best of eleven candidates on a few dozen training breaks is itself noisy; read the row as a test of the procedure, not of one cut-off.

**The rule is mostly a bet on wide spreads.** Deals where the model sees less risk than the market are, almost by construction, deals with wide spreads, and wide spreads earned more than narrow ones in this sample (see the reference row that skips the widest 10%). The fair test of the model is therefore the line under each table that compares the rule with a model-free rule holding the same number of deals, chosen by the market-implied risk alone.


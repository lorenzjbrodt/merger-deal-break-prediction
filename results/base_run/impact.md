# Dollar impact: stylized simulation

**This is not a backtest.** The project has no price data. The numbers below translate the model's ranking of test-period deals into dollars under the assumptions listed here. Every figure is conditional on those assumptions.

## Assumptions

- **Assumption 1 (portfolio).** The fund holds every deal announced in the test period (2019-2025: 1,358 deals, 109 breaks, break rate 8.0%), equal-weighted.
- **Assumption 2 (payoffs).** A deal that closes earns a gross spread *s*; a deal that breaks loses *L*. Both are identical across deals. Real spreads are wider for deals the market already sees as risky, so a constant *s* overstates the benefit of skipping flagged deals. See the break-even spread below.
- **Assumption 3 (capital).** $100 million of deal exposure is put on per year and turned once per year. No leverage, financing cost, transaction cost, fees or taxes. Deals are pooled over the test period, so the figures are per-year averages, not a year-by-year path.
- **Assumption 4 (actions).** "Skip" removes flagged deals and redeploys their capital equally into the remaining deals. "Halve" gives flagged deals half weight, again fully invested.
- **Assumption 5 (flagging).** Flagged deals are the riskiest 10% or 20% of test-period deals by model score. This ranks deals inside the test period. A check with a cut-off fixed in advance is shown at the end.
- **Sensitivity grid.** *s* from 2% to 5%, *L* from 15% to 30%. Base case for the summary rows: *s* = 3%, *L* = 20%.
- Scores come from models trained on 2001-2018 only.

## Hold every deal

**Hold all deals: P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.64 | +0.23 | -0.17 | -0.57 |
| 3% | +1.56 | +1.15 | +0.75 | +0.35 |
| 4% | +2.47 | +2.07 | +1.67 | +1.27 |
| 5% | +3.39 | +2.99 | +2.59 | +2.19 |

At the test-period break rate of 8.0% and *L* = 20%, holding everything breaks even at a spread of 1.75%.

## Change versus holding every deal, using model 3 (LightGBM, tabular)

**Skip the riskiest 10% (136 deals, 13 of 109 breaks = 12%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.03 | +0.04 | +0.05 | +0.05 |
| 3% | +0.03 | +0.04 | +0.05 | +0.06 |
| 4% | +0.03 | +0.04 | +0.05 | +0.06 |
| 5% | +0.03 | +0.04 | +0.05 | +0.06 |

**Skip the riskiest 20% (272 deals, 33 of 109 breaks = 30%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.17 | +0.23 | +0.28 | +0.33 |
| 3% | +0.19 | +0.24 | +0.29 | +0.34 |
| 4% | +0.20 | +0.25 | +0.30 | +0.35 |
| 5% | +0.21 | +0.26 | +0.31 | +0.36 |

**Halve the riskiest 10% (136 deals, 13 of 109 breaks = 12%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.01 | +0.02 | +0.02 | +0.03 |
| 3% | +0.01 | +0.02 | +0.02 | +0.03 |
| 4% | +0.02 | +0.02 | +0.02 | +0.03 |
| 5% | +0.02 | +0.02 | +0.02 | +0.03 |

**Halve the riskiest 20% (272 deals, 33 of 109 breaks = 30%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.08 | +0.10 | +0.12 | +0.15 |
| 3% | +0.08 | +0.11 | +0.13 | +0.15 |
| 4% | +0.09 | +0.11 | +0.13 | +0.16 |
| 5% | +0.09 | +0.11 | +0.14 | +0.16 |

## Base case (*s* = 3%, *L* = 20%): change versus holding every deal, by model

Hold-all P&L in the base case: $1.15 million per $100 million per year. Brackets: bootstrap 95% interval over test deals (2000 resamples; deals re-ranked in each resample).

| Model | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---|---|---|---|
| 1. One rule | -0.13 [-0.07, +0.16] | -0.21 [-0.09, +0.26] | -0.06 [-0.03, +0.08] | -0.09 [-0.04, +0.12] |
| 2. Logistic, tabular | +0.17 [+0.02, +0.33] | +0.22 [+0.02, +0.40] | +0.08 [+0.01, +0.15] | +0.10 [+0.01, +0.18] |
| 3. LightGBM, tabular | +0.04 [-0.07, +0.17] | +0.24 [+0.05, +0.44] | +0.02 [-0.03, +0.08] | +0.11 [+0.02, +0.19] |
| 5f. LightGBM, tabular + text (full sample) | +0.06 [-0.06, +0.20] | +0.09 [-0.09, +0.27] | +0.03 [-0.03, +0.10] | +0.04 [-0.04, +0.12] |

A random flag (no model) changes expected P&L by $0 in this setup, because capital is redeployed into deals with the same expected payoff.

## Text subset: tabular versus tabular + text

Deals with an announcement text only (1,075 test deals, 99 breaks). Base case, change versus holding every deal in this subset. The last row is the paired difference between the two models.

| Model | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---|---|---|---|
| 3. LightGBM, tabular | +0.43 [+0.23, +0.62] | +0.70 [+0.45, +0.96] | +0.20 [+0.11, +0.29] | +0.31 [+0.20, +0.43] |
| 4. + keywords + TF-IDF | +0.26 [+0.10, +0.47] | +0.49 [+0.27, +0.78] | +0.12 [+0.05, +0.22] | +0.22 [+0.12, +0.35] |
| 5. + keywords + embeddings | +0.19 [+0.03, +0.40] | +0.33 [+0.11, +0.59] | +0.09 [+0.02, +0.19] | +0.15 [+0.05, +0.26] |
| 5 minus 3 (paired) | -0.24 [-0.40, -0.02] | -0.37 [-0.64, -0.13] | -0.11 [-0.19, -0.01] | -0.17 [-0.29, -0.06] |

## Checks on the assumptions

| Flagged group (model 3) | Deals | Break rate | Break-even spread on flagged deals |
|---|---:|---:|---:|
| riskiest 10% | 136 | 9.6% (rest: 7.9%) | 3.4% |
| riskiest 20% | 272 | 12.1% (rest: 7.0%) | 4.3% |

Reading: with *L* = 20% and the remaining deals earning *s* = 3%, skipping the flagged deals only pays if their actual market spread is below the break-even value. If the market already prices these deals at a wider spread than that, the fund is being paid for the risk and skipping destroys value. This is the main reason the numbers above are an upper-end view.

| Variant (model 3, base case) | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---:|---:|---:|---:|
| Redeploy skipped capital (main) | +0.04 | +0.24 | +0.02 | +0.11 |
| Skipped capital idle at 0% | -0.08 | -0.04 | -0.04 | -0.02 |
| Cut-off fixed in advance (training-period out-of-fold score quantile) | +0.08 (12% flagged) | +0.24 (24% flagged) | +0.04 (12% flagged) | +0.10 (24% flagged) |

$ million per $100 million per year, change versus holding every deal.


# Dollar impact: stylized simulation

**Secondary view.** The headline impact is the price-based backtest in `results/backtest.md`, which covers the deals with usable prices. This grid covers every test-period deal instead, at the cost of assumed payoffs: it translates the model's ranking into dollars under the assumptions listed here, and every figure is conditional on them.

## Assumptions

- **Assumption 1 (portfolio).** The fund holds every deal announced in the test period (2019-2025, core sample without reverse mergers: 1,083 deals, 40 adverse breaks, rate 3.7%), equal-weighted. A deal topped by a higher bid is treated like a completed deal (it earns *s*); in reality it usually earns more.
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
| 2% | +1.37 | +1.19 | +1.00 | +0.82 |
| 3% | +2.34 | +2.15 | +1.97 | +1.78 |
| 4% | +3.30 | +3.11 | +2.93 | +2.74 |
| 5% | +4.26 | +4.08 | +3.89 | +3.71 |

At the test-period break rate of 3.7% and *L* = 20%, holding everything breaks even at a spread of 0.77%.

## Change versus holding every deal, using the final model (lgbm_ext)

**Skip the riskiest 10% (108 deals, 6 of 40 adverse breaks = 15%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.04 | +0.05 | +0.06 | +0.07 |
| 3% | +0.04 | +0.05 | +0.06 | +0.07 |
| 4% | +0.04 | +0.05 | +0.06 | +0.07 |
| 5% | +0.04 | +0.05 | +0.06 | +0.07 |

**Skip the riskiest 20% (217 deals, 14 of 40 adverse breaks = 35%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.12 | +0.15 | +0.19 | +0.22 |
| 3% | +0.12 | +0.16 | +0.19 | +0.23 |
| 4% | +0.13 | +0.17 | +0.20 | +0.23 |
| 5% | +0.14 | +0.17 | +0.21 | +0.24 |

**Halve the riskiest 10% (108 deals, 6 of 40 adverse breaks = 15%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.02 | +0.02 | +0.03 | +0.03 |
| 3% | +0.02 | +0.02 | +0.03 | +0.03 |
| 4% | +0.02 | +0.02 | +0.03 | +0.03 |
| 5% | +0.02 | +0.02 | +0.03 | +0.03 |

**Halve the riskiest 20% (217 deals, 14 of 40 adverse breaks = 35%): change in P&L** ($ million per $100 million per year)

| Spread s \ Loss L | 15% | 20% | 25% | 30% |
|---|---:|---:|---:|---:|
| 2% | +0.05 | +0.07 | +0.08 | +0.10 |
| 3% | +0.06 | +0.07 | +0.09 | +0.10 |
| 4% | +0.06 | +0.07 | +0.09 | +0.10 |
| 5% | +0.06 | +0.08 | +0.09 | +0.11 |

## Base case (*s* = 3%, *L* = 20%): change versus holding every deal, by model

Hold-all P&L in the base case: $2.15 million per $100 million per year. Brackets: bootstrap 95% interval over test deals (2000 resamples; deals re-ranked in each resample).

| Model | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---|---|---|---|
| 1. One rule | -0.07 [-0.07, +0.11] | +0.03 [-0.10, +0.17] | -0.03 [-0.03, +0.05] | +0.01 [-0.04, +0.08] |
| 2. Logistic, base tabular | +0.05 [-0.04, +0.18] | +0.08 [-0.06, +0.24] | +0.02 [-0.02, +0.08] | +0.04 [-0.03, +0.11] |
| 3. LightGBM, base tabular | +0.09 [-0.02, +0.21] | +0.21 [+0.04, +0.37] | +0.04 [-0.01, +0.10] | +0.09 [+0.02, +0.17] |
| Final model (lgbm_ext) | +0.05 [-0.04, +0.16] | +0.16 [+0.00, +0.32] | +0.02 [-0.02, +0.08] | +0.07 [+0.00, +0.14] |

A random flag (no model) changes expected P&L by $0 in this setup, because capital is redeployed into deals with the same expected payoff.

## Text subset: extended tabular versus tabular + text

Deals with an announcement text only (975 test deals, 35 breaks). Base case, change versus holding every deal in this subset. The last row is the paired difference between the two models.

| Model | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---|---|---|---|
| 3m. Extended tabular + market context | -0.04 [-0.10, +0.05] | -0.06 [-0.15, +0.11] | -0.02 [-0.05, +0.02] | -0.03 [-0.07, +0.05] |
| 4. + keywords + TF-IDF | +0.07 [-0.04, +0.19] | +0.03 [-0.08, +0.21] | +0.03 [-0.02, +0.09] | +0.01 [-0.04, +0.09] |
| 5. + keywords + embeddings | +0.04 [-0.06, +0.14] | +0.00 [-0.12, +0.18] | +0.02 [-0.03, +0.07] | +0.00 [-0.05, +0.08] |
| 5L. + keywords + LLM terms | +0.04 [-0.05, +0.16] | +0.00 [-0.11, +0.17] | +0.02 [-0.03, +0.07] | +0.00 [-0.05, +0.08] |
| 5A. + all text groups | +0.04 [-0.05, +0.16] | +0.09 [-0.06, +0.25] | +0.02 [-0.02, +0.08] | +0.04 [-0.03, +0.11] |
| 5A minus 3m (paired) | +0.08 [+0.00, +0.18] | +0.15 [-0.00, +0.27] | +0.04 [-0.00, +0.09] | +0.07 [-0.00, +0.12] |

## Checks on the assumptions

| Flagged group (final model) | Deals | Break rate | Break-even spread on flagged deals |
|---|---:|---:|---:|
| riskiest 10% | 108 | 5.6% (rest: 3.5%) | 3.5% |
| riskiest 20% | 217 | 6.5% (rest: 3.0%) | 3.8% |

Reading: with *L* = 20% and the remaining deals earning *s* = 3%, skipping the flagged deals only pays if their actual market spread is below the break-even value. If the market already prices these deals at a wider spread than that, the fund is being paid for the risk and skipping destroys value. This is the main reason the numbers above are an upper-end view.

| Variant (final model, base case) | Skip riskiest 10% | Skip riskiest 20% | Halve riskiest 10% | Halve riskiest 20% |
|---|---:|---:|---:|---:|
| Redeploy skipped capital (main) | +0.05 | +0.16 | +0.02 | +0.07 |
| Skipped capital idle at 0% | -0.17 | -0.30 | -0.09 | -0.15 |
| Cut-off fixed in advance (training-period out-of-fold score quantile) | +0.05 (12% flagged) | +0.15 (24% flagged) | +0.02 (12% flagged) | +0.06 (24% flagged) |

$ million per $100 million per year, change versus holding every deal.


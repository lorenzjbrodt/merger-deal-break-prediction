# Backtest on Capital IQ monthly closes (fallback without CRSP)

The headline version uses CRSP daily prices: `results/backtest.md`.

Test period 2019-2025, core sample. Entry: target close at t1, the first monthly grid date (the 20th) after the announcement. Exit: the offer price on the closing date for completed all-cash deals; the first grid close after the cancellation date for broken deals. Equal weights. Annual return = total P&L divided by total capital-years (each deal ties up one unit of capital for its holding period), so `$ per $100 million per year` assumes the capital is continuously redeployed into deals of the same kind.

**Limits.** Monthly marks only: entry is up to a month after the announcement (part of the spread is already gone and some early news is in the price), and broken deals are exited on the next grid date, not at the first print after the news. No transaction costs, dividends, financing cost, borrow or position limits. Deals without a usable price are missing (see `results/price_coverage.md`), and that loss is larger for broken deals. Model: lgbm_ext; scores from the model trained on 2001-2018.

## All-cash deals (main)

580 test deals with usable prices: 559 closed, 16 adverse breaks, 5 topped by a higher bid.

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 580 | 16 | 5 | +3.06% | 83 | +8.7% | +8.73 | reference |
| Skip riskiest 10% by model | 522 | 15 | 4 | +3.26% | 78 | +9.6% | +9.60 | +0.87 [-0.25, +2.03] |
| Skip riskiest 20% by model | 464 | 12 | 4 | +2.92% | 72 | +9.2% | +9.19 | +0.46 [-1.45, +2.52] |
| Hold only if model probability < market-implied | 377 | 12 | 4 | +6.13% | 100 | +14.6% | +14.59 | +5.86 [+3.81, +8.00] |
| Hold only if model risk percentile < market risk percentile | 286 | 8 | 2 | +6.96% | 88 | +18.2% | +18.16 | +9.43 [+6.21, +13.11] |
| Reference: skip widest-spread 10% (market's own ranking) | 522 | 12 | 4 | +0.37% | 78 | +1.1% | +1.14 | -7.59 [-10.23, -5.11] |

Change vs hold-all is in $ million per $100 million per year (equivalently percentage points of annual return).

By outcome (all usable deals):

| Outcome | Deals | Mean return per deal | Median return | Worst | Median holding days | Mean model score | Mean p_mkt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Closed | 559 | +4.4% | +1.7% | -44.6% | 79 | 0.040 | 0.19 |
| Topped by a higher bid | 5 | +9.2% | +10.0% | -4.6% | 61 | 0.050 | 0.19 |
| Adverse break | 16 | -44.9% | -48.3% | -79.8% | 396 | 0.044 | 0.31 |

## All considerations, including stock and mixed (robustness)

911 test deals with usable prices: 869 closed, 28 adverse breaks, 14 topped by a higher bid.

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 911 | 28 | 14 | +3.26% | 92 | +8.9% | +8.85 | reference |
| Skip riskiest 10% by model | 820 | 25 | 11 | +3.23% | 92 | +8.8% | +8.80 | -0.05 [-1.89, +1.51] |
| Skip riskiest 20% by model | 729 | 21 | 10 | +2.71% | 91 | +7.8% | +7.82 | -1.03 [-3.14, +1.37] |
| Hold only if model probability < market-implied | 614 | 21 | 7 | +5.45% | 114 | +12.9% | +12.89 | +4.03 [+2.32, +5.86] |
| Hold only if model risk percentile < market risk percentile | 458 | 12 | 5 | +5.77% | 101 | +14.7% | +14.68 | +5.83 [+2.86, +8.94] |
| Reference: skip widest-spread 10% (market's own ranking) | 820 | 21 | 11 | +1.33% | 92 | +3.7% | +3.72 | -5.13 [-7.20, -3.13] |

Change vs hold-all is in $ million per $100 million per year (equivalently percentage points of annual return).

By outcome (all usable deals):

| Outcome | Deals | Mean return per deal | Median return | Worst | Median holding days | Mean model score | Mean p_mkt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Closed | 869 | +4.2% | +1.8% | -72.9% | 92 | 0.045 | 0.27 |
| Topped by a higher bid | 14 | +14.1% | +13.9% | -4.6% | 61 | 0.061 | 0.23 |
| Adverse break | 28 | -32.2% | -29.8% | -83.5% | 350 | 0.054 | 0.41 |

## All-cash deals with an entry spread between -5% and +25% (robustness)

533 test deals with usable prices: 516 closed, 13 adverse breaks, 4 topped by a higher bid.

| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Hold all deals | 533 | 13 | 4 | +2.09% | 79 | +6.2% | +6.19 | reference |
| Skip riskiest 10% by model | 480 | 12 | 3 | +2.02% | 76 | +6.2% | +6.18 | -0.01 [-0.76, +1.12] |
| Skip riskiest 20% by model | 426 | 10 | 3 | +1.86% | 72 | +6.1% | +6.06 | -0.13 [-1.39, +1.51] |
| Hold only if model probability < market-implied | 350 | 11 | 3 | +3.39% | 100 | +8.3% | +8.27 | +2.08 [+1.08, +3.38] |
| Hold only if model risk percentile < market risk percentile | 269 | 10 | 1 | +3.04% | 88 | +7.8% | +7.77 | +1.58 [-0.20, +3.67] |
| Reference: skip widest-spread 10% (market's own ranking) | 480 | 8 | 4 | +1.12% | 75 | +3.6% | +3.63 | -2.56 [-4.16, -0.85] |

Change vs hold-all is in $ million per $100 million per year (equivalently percentage points of annual return).

By outcome (all usable deals):

| Outcome | Deals | Mean return per deal | Median return | Worst | Median holding days | Mean model score | Mean p_mkt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Closed | 516 | +3.2% | +1.7% | -5.0% | 78 | 0.040 | 0.17 |
| Topped by a higher bid | 4 | +6.5% | +7.9% | -4.6% | 61 | 0.056 | 0.12 |
| Adverse break | 13 | -42.5% | -44.5% | -72.7% | 365 | 0.042 | 0.30 |

This panel was added after the first backtest showed that a few deals with very wide entry spreads dominate the averages. A spread above 25% on a cash deal is either a genuinely distressed situation or a mismatch between the exported offer and the price series (revised offers, contingent consideration); a negative spread below -5% means the market expected a higher bid. Neither can be told apart from the export, so both panels are shown.

**How to read the model-versus-market rows.** `p_mkt` is not calibrated: it averages about 0.19 for deals that closed, several times the actual break rate, because part of every spread is time value and compensation for risk. The rule "model probability below market-implied" therefore mostly selects deals with wide spreads, and its return comes from holding those, not from the model. The percentile version compares ranks instead of levels and is the fairer test.

For stock and mixed deals the exit for completed deals is the last available close before delisting and the spread uses the exported offer value per share, which moves with the acquirer's share price; no hedge of the acquirer leg is modelled. Treat that panel as a rough check only.


# Expected annualized return per deal, and the ranking test

All-cash core deals with CRSP prices. On day 1 each deal gets: a probability of an adverse break, of being topped and of closing; an expected time to closing; an expected loss if it breaks; and from these an expected return and an expected annualized return. Every component comes from models fitted on earlier years only (`results/pooled_oot.md`, `results/duration.md`, `results/loss_given_break.md`).

**Break probability used:** as decided in `results/day1_signals.md` by cross-validation inside each origin's training years: the day-1 model in 7 of 13 pooled origins and the market-implied probability (through the market logistic) in the others; the day-1 model for the test period. Rankings with the announcement-date model and with the model-plus-market stack are shown next to it.

The test: sort the deals of each announcement year into thirds by the ranking, buy at the day-1 close, hold to the outcome, and compare realised returns. Annual return = total P&L over total capital-years. Intervals: bootstrap over announcement years (pooled) and over deals (test period). **Limits:** no transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1 close.

## Pooled rolling origin, 2013-2025: 1205 deals, 38 adverse breaks, 17 topped

| Ranking | Top third: annual return | Middle | Bottom | Top minus bottom [95% CI] | Top third minus all deals [95% CI] | Adverse breaks top / middle / bottom |
|---|---:|---:|---:|---|---|---|
| Expected annualized return (break probability chosen by the day-1 step) | +12.3% | +3.1% | +2.2% | +10.0 [+6.5, +13.9] | +5.9 [+3.9, +8.0] | 22 / 12 / 4 |
| Expected annualized return (break probability from the announcement-date model) | +11.8% | +3.6% | +1.2% | +10.6 [+6.4, +16.4] | +5.4 [+3.5, +7.8] | 24 / 10 / 4 |
| Expected annualized return (break probability from model + market stack) | +12.0% | +3.2% | +2.3% | +9.7 [+6.7, +13.6] | +5.6 [+3.9, +7.6] | 23 / 11 / 4 |
| Benchmark: raw day-1 spread | +9.8% | +4.1% | +1.4% | +8.4 [+5.0, +12.7] | +3.4 [+1.9, +5.3] | 29 / 6 / 3 |
| Benchmark: spread annualized with the median duration by structure | +10.6% | +2.7% | +2.1% | +8.4 [+4.7, +12.7] | +4.2 [+2.8, +5.9] | 26 / 9 / 3 |
| Benchmark: lowest market-implied break probability | +1.1% | +4.9% | +9.3% | -8.2 [-12.8, -3.8] | -5.3 [-8.2, -2.6] | 4 / 9 / 25 |
| Attribution: break probability replaced by its average | +10.6% | +4.2% | +1.1% | +9.4 [+5.7, +13.9] | +4.2 [+2.5, +6.4] | 27 / 8 / 3 |
| Attribution: time to closing replaced by its average | +10.0% | +4.2% | +1.8% | +8.1 [+4.5, +12.4] | +3.6 [+1.7, +5.5] | 28 / 7 / 3 |
| Attribution: loss if it breaks replaced by its average | +12.9% | +2.4% | +2.6% | +10.3 [+6.7, +14.0] | +6.5 [+4.5, +8.5] | 21 / 13 / 4 |

Differences are in points of annual return, equal to $ million per $100 million per year. All ranked deals together returned +6.4% a year.

**Does the model ranking beat the free one?** Top third by expected annualized return minus top third by raw spread: **+2.5 points [95% CI +0.9, +4.1]** (the two top thirds share 72% of their deals); minus top third by annualized spread: +1.7 [+0.2, +3.2] (77% shared).

Detail for the model ranking:

| Third | Deals | Mean expected annualized return | Realised annual return | $ million per $100 million per year | Mean return per deal | Adverse break rate | Topped | Median holding days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top | 397 | +19.5% | +12.3% | +12.26 | +5.09% | 5.5% | 4 | 90 |
| middle | 401 | +4.0% | +3.1% | +3.07 | +1.10% | 3.0% | 2 | 92 |
| bottom | 407 | -7.1% | +2.2% | +2.25 | +0.67% | 1.0% | 11 | 84 |

Attribution: rank correlation of the model ranking with the ranking obtained when one component is replaced by its unconditional average (1 = that component does not move the ranking): break probability 0.87, time to closing 0.92, loss if it breaks 1.00. Correlation with the raw spread ranking: 0.82; with the annualized spread: 0.86.

Calendar-time portfolio, 2013-01-04 to 2026-11-12, daily CRSP total returns. Equal weights across open deals with a 10% position cap (each deal gets 1 / max(open deals, 10) of the capital; the rest is cash earning nothing), rebalanced daily. "Top half" holds a deal only if its score at entry is at or above the median of the deals announced in the previous twelve months.

| Portfolio | Deals | Average open positions | Annual return | Volatility | Sharpe ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| All open deals | 1205 | 30.9 | +7.2% | 7.5% | 0.72 | -20.2% |
| Top half by expected annualized return | 620 | 17.0 | +9.6% | 9.7% | 0.80 | -26.7% |
| Top half by raw day-1 spread (no model) | 620 | 20.7 | +8.7% | 9.0% | 0.77 | -22.8% |

Top half by expected annualized return minus all deals: annual return +2.38 points [95% CI +0.03, +4.84], Sharpe ratio +0.08 [-0.16, +0.30]. Minus the top half by raw spread: annual return +0.87 points [-0.88, +2.60], Sharpe ratio +0.03 [-0.14, +0.22] (block bootstrap of daily returns, 21-day blocks; Sharpe ratio over the 1-month T-bill rate).

## Test period, 2019-2025: 623 deals, 22 adverse breaks, 6 topped

| Ranking | Top third: annual return | Middle | Bottom | Top minus bottom [95% CI] | Top third minus all deals [95% CI] | Adverse breaks top / middle / bottom |
|---|---:|---:|---:|---|---|---|
| Expected annualized return (break probability chosen by the day-1 step) | +15.1% | +2.0% | +2.5% | +12.6 [+5.3, +20.3] | +8.4 [+4.1, +13.2] | 10 / 7 / 5 |
| Expected annualized return (break probability from the announcement-date model) | +13.9% | +1.6% | +2.0% | +11.9 [+4.4, +19.4] | +7.1 [+3.5, +11.1] | 12 / 8 / 2 |
| Expected annualized return (break probability from model + market stack) | +14.6% | +1.3% | +2.5% | +12.1 [+4.7, +18.9] | +7.9 [+3.8, +12.0] | 11 / 9 / 2 |
| Benchmark: raw day-1 spread | +11.1% | +3.0% | +1.7% | +9.4 [+1.8, +16.8] | +4.3 [+1.4, +7.3] | 15 / 5 / 2 |
| Benchmark: spread annualized with the median duration by structure | +12.0% | +1.6% | +2.6% | +9.4 [+1.8, +16.5] | +5.2 [+2.2, +8.6] | 13 / 7 / 2 |
| Benchmark: lowest market-implied break probability | +1.0% | +3.8% | +10.7% | -9.6 [-17.5, -0.6] | -5.7 [-11.6, +1.1] | 3 / 7 / 12 |
| Attribution: break probability replaced by its average | +12.1% | +3.8% | +0.9% | +11.2 [+3.3, +18.5] | +5.4 [+1.9, +9.1] | 14 / 6 / 2 |
| Attribution: time to closing replaced by its average | +12.0% | +2.8% | +2.4% | +9.6 [+2.1, +16.7] | +5.2 [+1.8, +8.6] | 12 / 6 / 4 |
| Attribution: loss if it breaks replaced by its average | +15.3% | +2.0% | +2.5% | +12.8 [+5.5, +20.4] | +8.6 [+4.2, +13.2] | 10 / 7 / 5 |

Differences are in points of annual return, equal to $ million per $100 million per year. All ranked deals together returned +6.8% a year.

**Does the model ranking beat the free one?** Top third by expected annualized return minus top third by raw spread: **+4.1 points [95% CI +1.6, +7.1]** (the two top thirds share 63% of their deals); minus top third by annualized spread: +3.1 [+0.5, +6.1] (70% shared).

Detail for the model ranking:

| Third | Deals | Mean expected annualized return | Realised annual return | $ million per $100 million per year | Mean return per deal | Adverse break rate | Topped | Median holding days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top | 206 | +25.3% | +15.1% | +15.13 | +6.24% | 4.9% | 2 | 82 |
| middle | 207 | +5.5% | +2.0% | +1.97 | +0.78% | 3.4% | 1 | 94 |
| bottom | 210 | -8.3% | +2.5% | +2.52 | +0.90% | 2.4% | 3 | 94 |

Attribution: rank correlation of the model ranking with the ranking obtained when one component is replaced by its unconditional average (1 = that component does not move the ranking): break probability 0.86, time to closing 0.91, loss if it breaks 1.00. Correlation with the raw spread ranking: 0.75; with the annualized spread: 0.81.

Calendar-time portfolio, 2019-01-09 to 2026-11-12, daily CRSP total returns. Equal weights across open deals with a 10% position cap (each deal gets 1 / max(open deals, 10) of the capital; the rest is cash earning nothing), rebalanced daily. "Top half" holds a deal only if its score at entry is at or above the median of the deals announced in the previous twelve months.

| Portfolio | Deals | Average open positions | Annual return | Volatility | Sharpe ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| All open deals | 623 | 30.7 | +6.6% | 8.6% | 0.44 | -20.5% |
| Top half by expected annualized return | 330 | 16.3 | +8.6% | 11.4% | 0.51 | -25.6% |
| Top half by raw day-1 spread (no model) | 336 | 21.0 | +8.5% | 9.8% | 0.57 | -23.6% |

Top half by expected annualized return minus all deals: annual return +1.99 points [95% CI -1.73, +6.10], Sharpe ratio +0.07 [-0.24, +0.36]. Minus the top half by raw spread: annual return +0.11 points [-2.92, +3.17], Sharpe ratio -0.07 [-0.37, +0.15] (block bootstrap of daily returns, 21-day blocks; Sharpe ratio over the 1-month T-bill rate).

## Old versus new: effect of the product time-to-closing model

One pre-specified change, run once: where the announcement states an expected closing, the time to closing now comes from the model that also reads it (`results/duration.md`); nothing else changed. "Old" is the run with the final time-to-closing model for every deal (`results/expected_return_v1.json`).

| | Pooled 2013-2025: old | new | Test 2019-2025: old | new |
|---|---|---|---|---|
| Top third by expected annualized return: annual return | +11.7% | +12.3% | +14.4% | +15.1% |
| Middle third | +3.4% | +3.1% | +2.1% | +2.0% |
| Bottom third | +2.3% | +2.2% | +2.8% | +2.5% |
| Top minus bottom, points [95% CI] | +9.3 [+5.5, +13.5] | +10.0 [+6.5, +13.9] | +11.6 [+4.1, +19.0] | +12.6 [+5.3, +20.3] |
| Adverse breaks in top / middle / bottom | 25 / 8 / 5 | 22 / 12 / 4 | 11 / 7 / 4 | 10 / 7 / 5 |
| Model top third minus raw-spread top third | +1.9 [+0.4, +3.3] | +2.5 [+0.9, +4.1] | +3.3 [+1.1, +6.1] | +4.1 [+1.6, +7.1] |
| Model top third minus annualized-spread top third | +1.1 [-0.4, +2.5] | +1.7 [+0.2, +3.2] | +2.4 [+0.1, +4.9] | +3.1 [+0.5, +6.1] |
| Calendar-time, top half by the model: annual return / Sharpe | +9.3% / 0.80 | +9.6% / 0.80 | +9.7% / 0.67 | +8.6% / 0.51 |
| Calendar-time, top half minus all deals, points | +2.11 [-0.13, +4.54] | +2.38 [+0.03, +4.84] | +3.15 [-0.03, +6.39] | +1.99 [-1.73, +6.10] |
| Calendar-time, top half minus top half by raw spread, points | +0.60 [-0.93, +2.04] | +0.87 [-0.88, +2.60] | +1.27 [-0.57, +3.21] | +0.11 [-2.92, +3.17] |
| Rank correlation with the ranking that uses an average time to closing | 0.94 | 0.92 | 0.92 | 0.91 |

The spread-only and market-only benchmark rows do not depend on the time-to-closing model, apart from bootstrap noise in their intervals.

## Reading

- A ranking adds value only if the top third beats the bottom third (or all deals) by more than the interval. Compare the model rows with the spread rows: the spread is free.
- The attribution rows are not separate strategies; they show which of the three numbers moves the ranking and what is lost without it.
- Returns are before all costs. Equal weighting and daily rebalancing are not achievable at scale in small deals.


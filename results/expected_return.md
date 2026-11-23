# Expected annualized return per deal, and the ranking test

All-cash core deals with CRSP prices. On day 1 each deal gets: a probability of an adverse break, of being topped and of closing; an expected time to closing; an expected loss if it breaks; and from these an expected return and an expected annualized return. Every component comes from models fitted on earlier years only (`results/pooled_oot.md`, `results/duration.md`, `results/loss_given_break.md`).

**Break probability used:** as decided in `results/day1_signals.md` by cross-validation inside each origin's training years: the day-1 model in 7 of 13 pooled origins and the market-implied probability (through the market logistic) in the others; the day-1 model for the test period. Rankings with the announcement-date model and with the model-plus-market stack are shown next to it.

The test: sort the deals of each announcement year into thirds by the ranking, buy at the day-1 close, hold to the outcome, and compare realised returns. Annual return = total P&L over total capital-years. Intervals: bootstrap over announcement years (pooled) and over deals (test period). **Limits:** no transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1 close.

## Pooled rolling origin, 2013-2025: 1205 deals, 38 adverse breaks, 17 topped

| Ranking | Top third: annual return | Middle | Bottom | Top minus bottom [95% CI] | Top third minus all deals [95% CI] | Adverse breaks top / middle / bottom |
|---|---:|---:|---:|---|---|---|
| Expected annualized return (break probability chosen by the day-1 step) | +11.7% | +3.4% | +2.3% | +9.3 [+5.5, +13.5] | +5.3 [+3.1, +7.4] | 25 / 8 / 5 |
| Expected annualized return (break probability from the announcement-date model) | +10.9% | +4.4% | +1.1% | +9.8 [+5.5, +15.4] | +4.6 [+2.7, +6.7] | 26 / 8 / 4 |
| Expected annualized return (break probability from model + market stack) | +10.9% | +4.0% | +2.2% | +8.7 [+5.5, +12.7] | +4.6 [+2.8, +6.6] | 26 / 8 / 4 |
| Benchmark: raw day-1 spread | +9.8% | +4.1% | +1.4% | +8.4 [+5.0, +12.7] | +3.4 [+1.9, +5.3] | 29 / 6 / 3 |
| Benchmark: spread annualized with the median duration by structure | +10.6% | +2.7% | +2.1% | +8.4 [+4.7, +12.7] | +4.2 [+2.8, +5.9] | 26 / 9 / 3 |
| Benchmark: lowest market-implied break probability | +1.1% | +4.9% | +9.3% | -8.2 [-12.8, -3.8] | -5.3 [-8.2, -2.6] | 4 / 9 / 25 |
| Attribution: break probability replaced by its average | +10.2% | +4.4% | +1.2% | +9.0 [+5.4, +13.5] | +3.9 [+2.2, +5.9] | 28 / 7 / 3 |
| Attribution: time to closing replaced by its average | +10.0% | +4.2% | +1.8% | +8.1 [+4.5, +12.4] | +3.6 [+1.7, +5.5] | 28 / 7 / 3 |
| Attribution: loss if it breaks replaced by its average | +11.7% | +3.3% | +2.5% | +9.2 [+5.2, +13.5] | +5.4 [+3.1, +7.5] | 25 / 9 / 4 |

Differences are in points of annual return, equal to $ million per $100 million per year. All ranked deals together returned +6.4% a year.

**Does the model ranking beat the free one?** Top third by expected annualized return minus top third by raw spread: **+1.9 points [95% CI +0.4, +3.3]** (the two top thirds share 73% of their deals); minus top third by annualized spread: +1.1 [-0.4, +2.5] (79% shared).

Detail for the model ranking:

| Third | Deals | Mean expected annualized return | Realised annual return | $ million per $100 million per year | Mean return per deal | Adverse break rate | Topped | Median holding days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top | 397 | +19.3% | +11.7% | +11.66 | +4.96% | 6.3% | 4 | 97 |
| middle | 401 | +3.9% | +3.4% | +3.38 | +1.23% | 2.0% | 2 | 92 |
| bottom | 407 | -6.6% | +2.3% | +2.34 | +0.66% | 1.2% | 11 | 81 |

Attribution: rank correlation of the model ranking with the ranking obtained when one component is replaced by its unconditional average (1 = that component does not move the ranking): break probability 0.88, time to closing 0.94, loss if it breaks 1.00. Correlation with the raw spread ranking: 0.84; with the annualized spread: 0.88.

Calendar-time portfolio, 2013-01-04 to 2026-11-12, daily CRSP total returns. Equal weights across open deals with a 10% position cap (each deal gets 1 / max(open deals, 10) of the capital; the rest is cash earning nothing), rebalanced daily. "Top half" holds a deal only if its score at entry is at or above the median of the deals announced in the previous twelve months.

| Portfolio | Deals | Average open positions | Annual return | Volatility | Sharpe ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| All open deals | 1205 | 30.9 | +7.2% | 7.5% | 0.72 | -20.2% |
| Top half by expected annualized return | 623 | 17.6 | +9.3% | 9.4% | 0.80 | -25.6% |
| Top half by raw day-1 spread (no model) | 620 | 20.7 | +8.7% | 9.0% | 0.77 | -22.8% |

Top half by expected annualized return minus all deals: annual return +2.11 points [95% CI -0.13, +4.54], Sharpe ratio +0.08 [-0.17, +0.32]. Minus the top half by raw spread: annual return +0.60 points [-0.93, +2.04], Sharpe ratio +0.03 [-0.12, +0.20] (block bootstrap of daily returns, 21-day blocks; Sharpe ratio over the 1-month T-bill rate).

## Test period, 2019-2025: 623 deals, 22 adverse breaks, 6 topped

| Ranking | Top third: annual return | Middle | Bottom | Top minus bottom [95% CI] | Top third minus all deals [95% CI] | Adverse breaks top / middle / bottom |
|---|---:|---:|---:|---|---|---|
| Expected annualized return (break probability chosen by the day-1 step) | +14.4% | +2.1% | +2.8% | +11.6 [+4.1, +19.0] | +7.6 [+3.4, +12.2] | 11 / 7 / 4 |
| Expected annualized return (break probability from the announcement-date model) | +13.1% | +2.0% | +1.9% | +11.1 [+3.7, +18.5] | +6.3 [+2.9, +10.0] | 13 / 7 / 2 |
| Expected annualized return (break probability from model + market stack) | +13.4% | +2.1% | +2.4% | +10.9 [+3.6, +17.8] | +6.6 [+2.8, +10.5] | 13 / 7 / 2 |
| Benchmark: raw day-1 spread | +11.1% | +3.0% | +1.7% | +9.4 [+1.8, +16.8] | +4.3 [+1.4, +7.3] | 15 / 5 / 2 |
| Benchmark: spread annualized with the median duration by structure | +12.0% | +1.6% | +2.6% | +9.4 [+1.8, +16.5] | +5.2 [+2.2, +8.6] | 13 / 7 / 2 |
| Benchmark: lowest market-implied break probability | +1.0% | +3.8% | +10.7% | -9.6 [-17.5, -0.6] | -5.7 [-11.6, +1.1] | 3 / 7 / 12 |
| Attribution: break probability replaced by its average | +11.9% | +3.4% | +1.5% | +10.4 [+2.4, +18.1] | +5.2 [+1.8, +8.8] | 15 / 5 / 2 |
| Attribution: time to closing replaced by its average | +12.0% | +2.8% | +2.4% | +9.6 [+2.1, +16.7] | +5.2 [+1.8, +8.6] | 12 / 6 / 4 |
| Attribution: loss if it breaks replaced by its average | +14.4% | +2.1% | +2.8% | +11.6 [+4.2, +19.0] | +7.6 [+3.4, +12.3] | 11 / 7 / 4 |

Differences are in points of annual return, equal to $ million per $100 million per year. All ranked deals together returned +6.8% a year.

**Does the model ranking beat the free one?** Top third by expected annualized return minus top third by raw spread: **+3.3 points [95% CI +1.1, +6.1]** (the two top thirds share 67% of their deals); minus top third by annualized spread: +2.4 [+0.1, +4.9] (74% shared).

Detail for the model ranking:

| Third | Deals | Mean expected annualized return | Realised annual return | $ million per $100 million per year | Mean return per deal | Adverse break rate | Topped | Median holding days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top | 206 | +24.8% | +14.4% | +14.39 | +6.11% | 5.3% | 2 | 84 |
| middle | 207 | +5.3% | +2.1% | +2.11 | +0.84% | 3.4% | 1 | 95 |
| bottom | 210 | -7.7% | +2.8% | +2.80 | +0.96% | 1.9% | 3 | 92 |

Attribution: rank correlation of the model ranking with the ranking obtained when one component is replaced by its unconditional average (1 = that component does not move the ranking): break probability 0.86, time to closing 0.92, loss if it breaks 1.00. Correlation with the raw spread ranking: 0.77; with the annualized spread: 0.83.

Calendar-time portfolio, 2019-01-09 to 2026-11-12, daily CRSP total returns. Equal weights across open deals with a 10% position cap (each deal gets 1 / max(open deals, 10) of the capital; the rest is cash earning nothing), rebalanced daily. "Top half" holds a deal only if its score at entry is at or above the median of the deals announced in the previous twelve months.

| Portfolio | Deals | Average open positions | Annual return | Volatility | Sharpe ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| All open deals | 623 | 30.7 | +6.6% | 8.6% | 0.44 | -20.5% |
| Top half by expected annualized return | 330 | 16.8 | +9.7% | 10.2% | 0.67 | -23.3% |
| Top half by raw day-1 spread (no model) | 336 | 21.0 | +8.5% | 9.8% | 0.57 | -23.6% |

Top half by expected annualized return minus all deals: annual return +3.15 points [95% CI -0.03, +6.39], Sharpe ratio +0.24 [-0.07, +0.54]. Minus the top half by raw spread: annual return +1.27 points [-0.57, +3.21], Sharpe ratio +0.10 [-0.10, +0.26] (block bootstrap of daily returns, 21-day blocks; Sharpe ratio over the 1-month T-bill rate).

## Reading

- A ranking adds value only if the top third beats the bottom third (or all deals) by more than the interval. Compare the model rows with the spread rows: the spread is free.
- The attribution rows are not separate strategies; they show which of the three numbers moves the ranking and what is lost without it.
- Returns are before all costs. Equal weighting and daily rebalancing are not achievable at scale in small deals.


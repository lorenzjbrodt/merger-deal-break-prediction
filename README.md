# Underwriting merger deals at announcement

MIT 15.071 Project 1 (Predictive AI), Lorenz J. Brodt. [Project page](https://lorenzjbrodt.github.io/merger-deal-break-prediction/)

**Result.** Of the three numbers a merger-arbitrage manager underwrites, only time to closing is predicted well at
announcement (off by 47 days on average, against 64 for the median of its deal type); break risk is judged better by the day-1
spread than by any model here, and the loss when a deal breaks is large (28% on average) and close to unpredictable.
**Recommendation.** Use the model for timing and ranking, the spread for break risk and a 31% loss for sizing: turn
each spread into an expected annualized return with the time-to-closing model and rank deals by it (top third 12.26% a
year, 1.67 points a year more than the top third by annualized spread and 2.48 more than by raw spread, before
costs), read break risk off the day-1 spread, and size positions for a 31% loss, the mean loss of the 38 adverse
breaks among the ranked all-cash deals (31.21%; 27.94% over all 184 adverse breaks), not on the break model.

## Problem

A merger-arbitrage fund buys the target after a deal is announced: a small gain if it closes, a large loss if it
breaks. Per deal the manager needs the probability of an adverse break (cancelled for a reason other than a higher bid),
the time to closing, and the loss if it breaks.

## Data (licensed data is not in the repo: [DATA.md](DATA.md))

- **Capital IQ**: 4,744 signed US public deals, 2001-2025; 4,468 without reverse mergers, of which 198 adverse breaks.
- **SEC EDGAR**: press releases; an LLM reads each press release into 29 contract terms and the stated closing date.
- **CRSP daily prices** (WRDS) for 93% of targets; Compustat; Hoberg-Phillips product similarity; FRED.

## Method

Pipeline: deal data (4,468 deals), then an LLM that reads each press release into 29 contract terms and the stated
closing date, then three models (timing, break risk, loss), then an expected annualized return that ranks deals within
each year. Test design: the timing model trains on 2001-2018 and is tested on 1,026 completed deals announced 2019-2025;
the break probability in the ranking comes from the day-1 model in 7 of 13 origins and otherwise from the market-implied
probability, picked by cross-validation (the day-1 model for the test period), and the break probability and the ranking
are refitted every year on earlier years and tested on 1,205 priced all-cash deals announced 2013-2025; the loss
forecast for each year comes from all other years, tested on 184 adverse breaks 2001-2025. Deals are bought one trading
day after announcement and held to the outcome. Brackets: bootstrap 95% intervals.

## Headline results

| | Test 2019-2025 | Pooled 2013-2025 |
|---|---|---|
| Time to closing: mean error, days (median of the deal type: 64 / 61) | 47.1 [43.0, 51.5] | 43.1 [39.6, 46.5] |
| Deals inside the 80% interval for time to closing | 80% | 81% |
| Loss if it breaks: mean return of adverse breaks | -29% | -25% |
| Break probability, ROC AUC: model / day-1 spread | 0.633 / 0.643 | 0.609 / 0.721 |
| Ranking by expected annualized return: top minus bottom third, points a year | +12.6 [+5.3, +20.3] | +10.0 [+6.5, +13.9] |
| Top third by the model minus top third by annualized spread, points a year ($m per $100m) | +3.14 [+0.47, +6.09] | +1.67 [+0.20, +3.25] |
| Top third by the model minus top third by raw spread, points a year ($m per $100m) | +4.05 [+1.55, +7.08] | +2.48 [+0.95, +4.14] |

- **For the manager**: a closing forecast 6 days better than the parties' own guidance; a ranking whose top third
  earned 12.26% a year, $1.67m a year per $100m more than the top third by annualized spread (the day-1 spread divided
  by the deal-type median time to closing) and 6.35% for all deals; a demo with the three numbers per deal.
- **Break prediction, nothing helped** (pooled AUC change): extended features +0.025 [-0.025, +0.076]; text and LLM
  terms +0.022 [-0.014, +0.059]; product overlap +0.006 [-0.010, +0.019]; day-1 market signals -0.061
  [-0.143, +0.011] against the day-1 spread; model added to the day-1 spread -0.028 [-0.194, +0.107].
- **Loss if it breaks**: no forecast beats the historical average.

Everything else: [results/](results/README.md), [DECISIONS.md](DECISIONS.md), [SLIDE_NOTES.md](SLIDE_NOTES.md).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh synthetic                 # smoke test on synthetic rows
./run_all.sh                           # real run: Capital IQ exports, `claude` CLI, WRDS (see DATA.md)
.venv/bin/streamlit run app/demo.py    # demo; falls back to synthetic values
```

## Limitations

- **Few events**: 40 adverse breaks in the test period, 84 pooled (22 and 38 with prices).
- **Before costs**: no transaction or financing costs, equal weights, unlimited capacity.
- **Mostly a risk premium**: the top third holds 22 of the 38 priced breaks; in a daily portfolio the model beats the
  raw spread by 0.87 points a year [-0.88, +2.60], which is not statistically confirmed.
- **Licensed vendor data** with known flaws (final terms stored, sparse fields for cancelled deals).

# Underwriting merger deals at announcement

MIT 15.071 Project 1 (Predictive AI), Lorenz J. Brodt. [Project page](https://lorenzjbrodt.github.io/merger-deal-break-prediction/)

**Result.** AI improves deal timing and ranking, the market price remains the better break signal, and breaks cost far
more than assumed (deck page 3). The closing date error is 47.10 days against 63.52 for the deal-type median; the
day-1 spread ranks break risk with an AUC of 0.72 against 0.60 for the model on the same deals; adverse breaks lose
27.94% on average (historical average of 184 breaks), 2.62x the 10.67% fallback assumption.
**Recommendation.** Use the model for timing and ranking, the spread for break risk, and a 31% loss for sizing (page
10). Ranked by expected annualized return, the top third earned 12.26% a year, $1.67m per $100m a year more than the top
third by annualized spread [0.20 to 3.25] and 2.48 points more than by raw spread [0.95 to 4.14], before costs. Size
for 31.21%, the mean loss of the 38 adverse breaks among ranked deals, not for the 27.94% historical average of all 184.

## Problem

A merger arbitrage fund buys the target after a deal is announced. At the median it earns 1.39% when the deal closes
(95.44% of deals) and loses 28.91% in an adverse break (3.15%), about 21 times as much (1,205 priced all-cash deals,
2013 to 2025). Three estimates decide a position: time to closing, break risk (an adverse break is a cancellation for a
reason other than a higher bid), and the loss if a deal breaks.

## Data (licensed data is not in the repo: [DATA.md](DATA.md))

- **Capital IQ**: 4,744 signed US public deals, 2001 to 2025; 4,468 without reverse mergers, with 198 adverse breaks.
- **SEC EDGAR**: press releases for 88% of deals; Claude Haiku reads each into 29 contract terms plus the parties'
  stated closing date (count in [results/llm_extraction_validation.md](results/llm_extraction_validation.md)).
- **CRSP daily prices** (WRDS) for 93% of targets; Compustat; Hoberg-Phillips product similarity; FRED.

## Method (deck page 5)

| | Time to closing | Break risk | Loss if a deal breaks |
|---|---|---|---|
| Predicts | Days from announcement to closing | Adverse break, yes or no | Return from day-1 close to break, in % |
| Method | Boosted trees on deal, firm, and LLM data | Boosted trees, adding day-1 market data | Historical average; no model beat it reliably |
| Learns from | Deals announced 2001 to 2018 | All earlier years, refitted each year | All other years, one year left out at a time |
| Tested on | 1,026 completed deals, 2019 to 2025 | 1,205 priced all-cash deals, 2013 to 2025 | 184 adverse breaks, 2001 to 2025 |
| Benchmark | Deal-type median and parties' stated date | Day-1 spread | Fallback assumption |

The three estimates give each deal's expected annualized return, (P(close) x spread + P(topped) x topping gain -
P(break) x loss) / years to resolution, which ranks the deals within each year. In the ranking, break risk comes from
the day-1 model in 7 of 13 years (picked by cross-validation on prior years), else the spread. Deals are bought at the
day-1 close and held to the outcome. Brackets are bootstrap 95% intervals.

## Headline results

| | Test 2019 to 2025 | Pooled 2013 to 2025 |
|---|---|---|
| Closing date error, days (deal-type median 63.52 / 61.02) | 47.10 [42.95 to 51.49] | 43.13 |
| Deals inside the 80% interval for time to closing | 80% | 81% |
| Break risk ranking, AUC on the same priced deals: model / day-1 spread | 0.632 / 0.643 | 0.599 / 0.721 |
| Mean loss of adverse breaks among ranked deals | 37.89% (22 breaks) | 31.21% (38 breaks) |
| Top minus bottom third by expected annualized return, points a year | +12.62 [+5.31 to +20.33] | +10.01 [+6.51 to +13.95] |
| Top third by the model minus top third by annualized spread ($m per $100m a year) | +3.14 [+0.47 to +6.09] | +1.67 [+0.20 to +3.25] |
| Top third by the model minus top third by raw spread ($m per $100m a year) | +4.05 [+1.55 to +7.08] | +2.48 [+0.95 to +4.14] |
| Daily portfolio, top half by the model minus top half by raw spread, points a year | +0.11 [-2.92 to +3.17] | +0.87 [-0.88 to +2.60] |

- **Timing**: with the parties' stated closing date, the model beats that date by 6.61 days [3.86 to 9.51]; deals
  closed a median of 14.50 days before it (page 6).
- **Break risk, nothing closes the gap** (pooled AUC change, page 7): buyer and target financials +0.025 [-0.025 to
  +0.076]; press release text and LLM-read terms +0.022 [-0.014 to +0.059]; product overlap +0.006 [-0.010 to +0.019];
  market conditions -0.028 [-0.054 to -0.002]; ten day-1 market signals added to the spread -0.125 [-0.246 to -0.010].
- **Loss**: no loss forecast reliably beat the historical average (page 15).

**How to read the results.** The ranking edge is significant deal by deal (top third vs. annualized spread +1.67 [0.20
to 3.25], vs. raw spread +2.48 [0.95 to 4.14], 2019 to 2025 +3.14 [0.47 to 6.09]); the daily-portfolio test (top half,
overlapping positions, equal weights) is positive but not significant, +0.87 [-0.88 to +2.60], so it neither confirms
the edge nor shows it is zero. Timing contributes most: top minus bottom third falls from +10.01 to +8.15 points a year
without the timing estimate, and to +9.45 without the break-risk estimate. Sizing (31.21% loss) is in neither test,
which use equal weights; it lowers risk per position, not the average return.

Everything else: [results/](results/README.md), [DECISIONS.md](DECISIONS.md), [SLIDE_NOTES.md](SLIDE_NOTES.md) (deck
text and the source of every number).

## Reproduce

Data and access ([DATA.md](DATA.md) has the screens and one-off setup): the S&P Capital IQ transaction screen export
(licensed) goes to `data/raw/ciq_transactions_v2.xls`, with monthly prices in `data/raw/ciq_prices_{1,2,3}.xls`; CRSP and
Compustat come through a WRDS account (a `wrds-pgdata` line in `~/.pgpass`, pulls cached in `data/wrds/`); FRED and
Hoberg-Phillips files go to `data/external/`; SEC EDGAR press releases are downloaded by the pipeline (set
`SEC_USER_AGENT`), and the LLM steps need a logged-in `claude` CLI. All of these paths are git-ignored. `./run_all.sh`
rebuilds `results/` from them; `./run_all.sh synthetic` runs the same pipeline on synthetic rows without any license.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh synthetic                 # smoke test on synthetic rows
./run_all.sh                           # real run: rebuilds results/ from the licensed data above
.venv/bin/streamlit run app/demo.py    # demo; falls back to synthetic values
```

## Limitations

- **Few events**: 38 adverse breaks among ranked deals (22 in the test period); losses vary widely (10th percentile of
  all 184 at -73.01%).
- **Before costs**: no transaction or financing costs, equal weights, and unlimited capacity.
- **Pay for break risk**: the model's top third holds 22 of the 38 priced adverse breaks, against 26 for the top third
  by annualized spread, so part of the return is pay for bearing break risk. The backtest used the model's break risk
  in 7 of 13 years.
- **Licensed vendor data** with known flaws (final terms stored, sparse fields for canceled deals).

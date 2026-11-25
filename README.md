# Underwriting merger deals at announcement

MIT 15.071 Project 1 (Predictive AI), Lorenz J. Brodt. [Project page](https://lorenzjbrodt.github.io/merger-deal-break-prediction/)

**Result.** AI improves deal timing and ranking but does not beat the market price on break risk, and breaks cost far
more than assumed (deck page 3). The closing date error is 47.10 days against 63.52 for the median by deal type. On
break risk the model does not beat the day-1 spread: its AUC is 0.60 against 0.72 on the same deals, a gap of -0.12 AUC
[-0.28 to +0.005] that is not significant at 95%. LLM-read contract terms cut the timing error and do not measurably
improve break risk. Adverse breaks lose 27.94% on average (184 breaks), 2.62x the 10.67% fallback assumption.

**Recommendation.** Use the model for timing and ranking, the spread for break risk, and a 31% loss estimate for sizing
(page 10). Ranked by expected annualized return, the top third earned 12.26% a year, $1.67m per $100m a year more than
the top third by annualized spread [0.20 to 3.25], before costs. Positions are sized for 31.21%, the mean loss of the 38
adverse breaks among ranked deals.

## Problem and data (licensed data is not in the repo: [DATA.md](DATA.md))

A merger arbitrage fund buys the target after a deal is announced. At the median it earns 1.39% when the deal closes
(95.44% of deals) and loses 28.91% in an adverse break (3.15%), about 21 times as much (1,205 priced all-cash deals,
2013 to 2025). An adverse break is a cancellation for a reason other than a higher bid.

- **Capital IQ**: 4,744 signed US public deals, 2001 to 2025; 4,468 without reverse mergers, with 198 adverse breaks.
- **SEC EDGAR**: press releases for 88% of deals; Claude Haiku reads each into 29 contract terms plus the parties'
  stated closing date ([validation](results/llm_extraction_validation.md)).
- **CRSP daily prices** (WRDS) for 93% of targets; Compustat; Hoberg-Phillips product similarity; FRED.

## Method (deck page 5)

| | Time to closing | Break risk | Loss if a deal breaks |
|---|---|---|---|
| Predicts | Days from announcement to closing | Adverse break, yes or no | Return from day-1 close to break, in % |
| Method | Boosted trees on deal, firm, and LLM data | Boosted trees on deal terms and financials | Historical average; no model beat it reliably |
| Learns from | Deals announced 2001 to 2018 | All earlier years, refitted each year | All other years, one year left out at a time |
| Tested on | 1,026 completed deals, 2019 to 2025 | 1,205 priced all-cash deals, 2013 to 2025 | 184 adverse breaks, 2001 to 2025 |
| Benchmark | Median by deal type and parties' stated date | Day-1 spread | Fallback assumption |

The three estimates give each deal's expected annualized return, (P(close) x spread + P(topped) x topping gain -
P(break) x loss) / years to resolution, which ranks the deals each year on earlier years only. In the ranking, break
risk comes from a day-1 price model in 7 of 13 years (picked by cross-validation on prior years), else the spread. Deals
are bought at the day-1 close and held to the outcome. Brackets are bootstrap 95% intervals.

**How to read the results.** The ranking edge is significant deal by deal: top third +1.67 [0.20 to 3.25] vs. annualized
spread, +2.48 [0.95 to 4.14] vs. raw spread, +3.14 [0.47 to 6.09] in 2019 to 2025. The daily-portfolio test (top half,
overlapping positions, equal weights) gives +0.87 [-0.88 to +2.60]; the interval includes zero, so it neither confirms
nor rules out the edge. Timing adds most: top minus bottom third falls from +10.01 to +8.15 points without it, to +9.45
without break risk. Sizing (31.21% loss) is outside both equal-weight tests; its job is to limit losses.
Full results table and the source of every number: [results/README.md](results/README.md),
[SLIDE_NOTES.md](SLIDE_NOTES.md) (deck text), [DECISIONS.md](DECISIONS.md).

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
  all 184 at -73.01%). **Before costs**: no transaction or financing costs, equal weights, and unlimited capacity.
- **Pay for break risk**: the model's top third holds 22 of the 38 priced adverse breaks, against 26 for the top third
  by annualized spread, so part of the return is pay for bearing break risk. A day-1 price model gave break risk in 7 of
  13 backtest years. **Licensed vendor data** with known flaws (final terms stored, sparse fields for canceled deals).

# Predicting M&A deal breaks at announcement

MIT 15.071 Project 1 (Predictive AI), solo project by Lorenz J. Brodt. **Base run.** An extended run (more
features, outcome taxonomy, market data, LLM-extracted terms, price backtest) follows in later commits.

## Problem

A merger-arbitrage fund buys the target after a deal is announced and earns the spread if the deal closes. It loses
several times the spread if the deal breaks. The portfolio manager needs, on the announcement day, a break probability
to decide whether to take a position and how large. Label: 1 = deal cancelled after a definitive agreement, 0 = closed.

## Data

- **Capital IQ transaction screen** (licensed, not in this repo; rebuild instructions in [DATA.md](DATA.md)): US public
  targets, deal value above $100m, majority stake, definitive agreement signed, announced 2001-2025, SPACs removed.
  **4,744 deals, 346 breaks (7.3%).**
- **SEC EDGAR** (public): the target's 8-K, 425, DEFA14A, SC TO-C, SC 14D9, SC TO-T and PREM14A filings from one day
  before to five business days after the announcement. Two text fields per deal: the press release and the merger
  agreement. 88% of deals have a press release, 78% an agreement ([coverage by year](results/text_coverage.md)).

## Method

Train on announcements 2001-2018, test on 2019-2025. Model selection by expanding-window CV inside the training years.
Every encoder, scaler, TF-IDF vocabulary, SVD and PCA is fitted in-fold. Features are restricted to what is known on the
announcement date; status, outcome dates and spreads are never used, and two fields that Capital IQ can update later
(Deal Responses, Deal Attitude) appear only in a labelled ablation. Text enters as keyword flags, TF-IDF components and
frozen sentence embeddings (`bge-small-en-v1.5`). Confidence intervals: 2,000 bootstrap resamples of test deals.

## Headline results (test 2019-2025)

| Model | ROC AUC, full sample (1,358 deals, 109 breaks) | ROC AUC, text subset (1,075 deals, 99 breaks) | Breaks in riskiest 20%, text subset |
|---|---|---|---|
| 0. Base rate | 0.500 | 0.500 | 0.20 |
| 1. One rule (financial condition attached) | 0.554 [0.505, 0.600] | 0.592 [0.542, 0.638] | 0.26 |
| 2. Logistic regression, tabular | 0.559 [0.500, 0.618] | 0.669 [0.613, 0.723] | 0.37 |
| 3. LightGBM, tabular | 0.622 [0.574, 0.670] | 0.698 [0.642, 0.754] | 0.46 |
| 4. + keywords + TF-IDF text | | 0.712 [0.659, 0.760] | 0.38 |
| 5. + keywords + frozen embeddings | 0.575 [0.522, 0.625] (with has_text flag) | 0.661 [0.605, 0.714] | 0.32 |
| 6. Text only (TF-IDF logistic) | | 0.672 [0.626, 0.717] | 0.28 |

- **Tabular data carries a modest signal.** LightGBM beats the logistic benchmark on the full sample by +0.062 AUC
  [+0.011, +0.119]. An AUC of 0.62 is far from a trading signal on its own.
- **Text adds nothing reliable.** Embeddings minus tabular: -0.037 AUC [-0.086, +0.010], and they capture fewer breaks in
  the riskiest 20% (-0.14 [-0.24, -0.05]). TF-IDF: +0.014 AUC [-0.026, +0.060] with worse PR AUC. Announcement press
  releases are written by the deal parties and read alike whether or not the deal later breaks; the terms that matter are
  already in the tabular fields. Text alone still reaches 0.67 AUC, so it is redundant, not empty.
- **Dollar view** (stylized simulation, no price data, all assumptions in [results/impact.md](results/impact.md)): with a 3%
  spread and a 20% loss on a break, holding every test-period deal earns $1.15m per $100m per year. Skipping the riskiest
  20% by the tabular model adds $0.24m on the full sample, and $0.70m [0.45, 0.96] on the text subset. The break-even
  check in that file shows the gain disappears if flagged deals trade about 1.3 points wider than the rest.

Full tables: [benchmark](results/benchmark_table.md), [segments](results/segments.md), [impact](results/impact.md),
[metrics.json](results/metrics.json). Figures are in [figures/](figures).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
./run_all.sh synthetic     # smoke test on 200 synthetic rows, no Capital IQ needed (results are meaningless)
./run_all.sh               # real run: needs data/raw/ciq_transactions_v2.xls (see DATA.md); EDGAR download takes ~100 min
```

`data/sample/synthetic_ciq_export.csv` is **synthetic**. Every number in `results/` and `figures/` comes from the real export.

## Limitations

- **Licensed data.** The deal table cannot be shared; results can be rebuilt only with Capital IQ access.
- **No price data in the base run.** The dollar impact is a stylized simulation with a constant spread. The market
  already charges wider spreads for risky deals, so the benefit of skipping them is overstated.
- **Label noise and field timing.** Capital IQ fills fewer fields for cancelled deals (missingness predicts the label)
  and may store revised terms. "Cancelled" also covers deals lost to a higher bid, which is good news for an arbitrageur.
- **Sample composition shifts.** 275 test-period deals (52 of the 109 test breaks) are reverse mergers, mostly
  de-SPACs, a deal type almost absent before 2019 and not what a merger-arbitrage fund trades. The SPAC filter in the
  screen misses them. The extended run reports results with and without them.
- **Few breaks.** 109 test breaks: intervals are wide, segment results are noisy, and small differences between models
  are not distinguishable from zero.

Decisions and their reasons: [DECISIONS.md](DECISIONS.md). Slide outline: [SLIDE_NOTES.md](SLIDE_NOTES.md). Code: MIT licence; data: not included.

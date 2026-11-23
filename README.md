# Predicting M&A deal breaks at announcement

MIT 15.071 Project 1 (Predictive AI), solo project by Lorenz J. Brodt. Project page:
<https://lorenzjbrodt.github.io/merger-deal-break-prediction/>

## Problem

A merger-arbitrage fund buys the target after a deal is announced. It earns a small spread if the deal closes and loses
many times that if it breaks. On the announcement day the portfolio manager wants a break probability to decide whether
to take the position and how large. Primary target: **adverse break** = a signed deal cancelled for a reason other than a
superior competing bid (a higher bid is good news for an arbitrageur). Secondary target: any break.

## Data (none of the licensed data is in this repo; see [DATA.md](DATA.md))

- **Capital IQ** deal screen: US public targets, value above $100m, majority stake, definitive agreement, announced
  2001-2025. 4,744 deals as specified; the **core sample of 4,468** drops 276 reverse mergers (mostly de-SPACs). 346 breaks:
  100 superior competing bids, **246 adverse breaks** ([taxonomy](results/outcome_taxonomy.md)).
- **SEC EDGAR**: the target's filings in the week after announcement (press release for 88% of deals, agreement for 78%),
  plus 29 validated **LLM-extracted terms** ([validation](results/llm_extraction_validation.md)).
- **CRSP daily prices via WRDS** for 93% of core targets ([linking](results/crsp_link.md), [checks](results/crsp_prices.md)):
  day-1 spread, market-implied break probability, daily backtest. Compustat daily prices fill gaps; the Capital IQ
  monthly closes remain as the fallback without WRDS ([results/monthly](results/monthly)). **FRED** market series.

## Method

Train on 2001-2018, test on 2019-2025; hyper-parameters by expanding-window CV inside the training years; every transform
fitted in-fold; only information known at announcement enters a model. Ladder: base rate, one rule, logistic regression,
LightGBM on deal terms, then extended features, market context, text, LLM terms, a fine-tuned encoder. Second evaluation
set: **pooled out-of-time 2013-2025**, every year scored by models refitted on earlier years only. Intervals: bootstrap
over deals (test period) and over announcement years (pooled).

## Headline results (adverse breaks, core sample)

| | Test 2019-2025 | Pooled out-of-time 2013-2025 |
|---|---|---|
| Deals / adverse breaks, all core deals | 1,083 / 40 | 2,176 / 84 |
| Logistic regression, base deal terms | 0.606 [0.523, 0.694] | 0.590 [0.525, 0.651] |
| LightGBM, base deal terms | 0.628 [0.538, 0.713] | 0.579 [0.519, 0.641] |
| **Final model** (LightGBM + extended features; chosen by CV) | **0.633 [0.551, 0.717]** | **0.609 [0.538, 0.686]** |
| + text (keywords, embeddings, LLM terms) | 0.646 [0.559, 0.730] | 0.599 [0.540, 0.663] |
| Priced all-cash deals / adverse breaks | 623 / 22 | 1,205 / 38 |
| Market-implied probability, day 1 | 0.643 [0.523, 0.755] | 0.721 [0.642, 0.800] |
| Final model on the same deals | 0.632 [0.519, 0.734] | 0.599 [0.476, 0.702] |
| Model added to the market (stack minus market) | +0.020 [-0.081, +0.124] | -0.028 [-0.194, +0.107] |

ROC AUC with 95% intervals. Tables: [ladder](results/benchmark_table.md), [pooled ladder](results/pooled_oot.md),
[market benchmark](results/market_benchmark.md).

- **There is a signal, and it is weak.** Deal terms rank adverse breaks better than chance in both sets.
- **Nothing added to the base terms is distinguishable from zero, also with twice the breaks.** Pooled: extended features
  +0.025 [-0.025, +0.076], text +0.022 [-0.014, +0.059], market context -0.028 [-0.054, -0.002].
- **The market ranks breaks better than the model.** With daily prices the market's day-1 spread reaches 0.72 in the
  pooled set against 0.60 for the model (difference -0.12 [-0.28, +0.01]), and the model adds nothing measurable to it.
  Seeing the day-1 spread helps the model (+0.046 [+0.006, +0.082] pooled); that is a different decision time.
- **In dollars** ([backtest](results/backtest.md), all-cash deals, entry at the day-1 close, daily marks, before costs):
  holding every deal returned 6.8% a year in the test period (6.4% pooled). Skipping the model's riskiest 10% changes
  that by +0.3 points [-0.6, +1.4] (pooled +0.7 [-0.1, +1.4]). A model-versus-market rule with a threshold fitted on
  earlier years adds +4.3 [+1.6, +7.2] (pooled +2.4 [+0.3, +4.6]), but it mostly buys wide spreads: against a model-free
  rule holding the same number of deals the gain is +1.8 [-0.3, +4.4] (pooled +0.8 [-1.0, +2.9]).
- **Data lesson.** The export stores the final offer. For deals that were later bumped this produced day-1 spreads of 30%
  and more; the announcement-day offer is now read from the press release (247 cash deals).

More: [segments](results/segments.md), [SHAP](results/shap.md), [calibration](results/calibration.md),
[ablation](results/ablation.md), [robustness](results/robustness.md), [three-class view](results/three_class.md),
[case studies](results/case_studies.md), [label audit](results/label_audit.md), [stylized impact](results/impact.md),
[LLM zero-shot (exploratory)](results/llm_zero_shot.md), [base run](results/base_run).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
./run_all.sh synthetic                  # smoke test on 200 synthetic rows; results are meaningless
./run_all.sh                            # real run: Capital IQ exports, a logged-in `claude` CLI, optionally WRDS (DATA.md)
.venv/bin/streamlit run app/demo.py     # local demo; falls back to the synthetic sample
```

`data/sample/synthetic_ciq_export.csv` is **synthetic**. Every number in `results/` and `figures/` comes from the real data.

## Limitations

- **Few events.** 40 adverse breaks in the test period (22 priced all-cash), 84 pooled (38 priced). Intervals are wide;
  "no measurable gain" is not "no gain". Pooled results mix 13 fitted models and resample only 13 years.
- **Licensed data and vendor fields.** Results need Capital IQ and WRDS access. The vendor fills fewer fields for
  cancelled deals, stores final terms, and classifies buyers as of today. Label audit against EDGAR: 3 of 76 statuses
  contradicted. 15% of breaks have no stated reason.
- **Backtest.** No transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1
  close. Deals without announcement text keep the export's final offer in the spread.
- **The model is blind to antitrust overlap,** the cause of the largest losses (see the case studies).
- **The LLM zero-shot score (AUC 0.82) is not evidence:** the same LLM named 92% of the anonymised targets.

Decisions: [DECISIONS.md](DECISIONS.md). Slides: [SLIDE_NOTES.md](SLIDE_NOTES.md). Code: MIT licence. Data: not included.

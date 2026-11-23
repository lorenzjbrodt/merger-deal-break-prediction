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
  2001-2025. 4,744 deals as specified; the **core sample of 4,468** drops 276 reverse mergers (mostly de-SPACs, almost all
  after 2019). 346 breaks: 100 superior competing bids, **246 adverse breaks** (LLM reading of deal notes, rule checks,
  50 cases re-read: [taxonomy](results/outcome_taxonomy.md)).
- **SEC EDGAR**: the target's filings in the week after announcement; press release for 88% of deals, merger agreement for 78%.
- **Capital IQ monthly closes** of the target (spread, market-implied break probability, backtest), **FRED** market series.
- **LLM-extracted terms**: 29 validated fields read from the filings by Claude Haiku ([validation](results/llm_extraction_validation.md)).

## Method

Train on 2001-2018, test on 2019-2025; hyper-parameters by expanding-window CV inside the training years; every transform
fitted in-fold; only information known at announcement enters a model. Ladder: base rate, one rule, logistic regression,
LightGBM on deal terms, then extended buyer/target features, market context, TF-IDF, frozen embeddings, LLM-extracted terms,
a fine-tuned encoder, and the market's own spread. Intervals: 2,000 bootstrap resamples of test deals.

## Headline results (adverse breaks, core sample, test 2019-2025: 1,083 deals, 40 adverse breaks)

| Model | ROC AUC [95% CI] | Adverse breaks in riskiest 20% |
|---|---|---|
| Base rate | 0.500 | 0.20 |
| One rule (any stock in the consideration) | 0.528 [0.449, 0.606] | 0.23 |
| Logistic regression, base deal terms | 0.606 [0.523, 0.694] | 0.28 |
| LightGBM, base deal terms | 0.628 [0.538, 0.713] | 0.40 |
| **LightGBM + extended buyer, target and structure features (final model)** | **0.633 [0.551, 0.717]** | **0.35 [0.20, 0.50]** |
| + market context | 0.626 [0.545, 0.709] | 0.33 |
| + text (keywords, embeddings, LLM-extracted terms) | 0.646 [0.559, 0.730] | 0.33 |
| Market-implied probability (552 priced cash deals, 16 breaks) | 0.619 [0.458, 0.771] | 0.25 |

- **There is a signal, and it is weak.** Deal terms rank adverse breaks better than chance (AUC 0.63, lower bound 0.55).
  The same holds across 12 rolling origins (mean AUC 0.61 to 0.65, [table](results/rolling_origin.md)).
- **Nothing added to the base deal terms is distinguishable from zero.** Extended features +0.006 AUC [-0.059, +0.066];
  all text groups +0.023 [-0.039, +0.080] on deals with filings; LLM-extracted terms +0.000 [-0.041, +0.042]; the
  fine-tuned encoder is at chance (0.517). [Ablation by group](results/ablation.md).
- **The model does not demonstrably beat the market.** Adding the model to the market-implied probability changes AUC by
  +0.027 [-0.096, +0.153] ([market benchmark](results/market_benchmark.md)).
- **In dollars** ([backtest](results/backtest.md), all-cash deals, monthly marks, before costs): holding all 580 priced test
  deals returned 8.7% a year on deployed capital. Skipping the model's riskiest 10% changes that by +0.9 points
  [-0.3, +2.0], the riskiest 20% by +0.5 [-1.5, +2.5]. Only 16 priced adverse breaks; they lost 45% on average.
- **What the base run got wrong.** Its AUC of 0.62 to 0.70 for "any break" was helped by reverse mergers (52 of 109 test
  breaks) and by counting topped deals as breaks. On that same definition the extended features do help
  (0.691 [0.641, 0.742] versus 0.622), mostly by separating deal types an arbitrageur would not trade.

More: [benchmark tables](results/benchmark_table.md), [segments](results/segments.md), [SHAP reading](results/shap.md),
[calibration](results/calibration.md), [robustness](results/robustness.md), [three-class view](results/three_class.md),
[six case studies](results/case_studies.md), [label audit](results/label_audit.md), [price coverage](results/price_coverage.md),
[stylized impact grid](results/impact.md), [exploratory LLM zero-shot score](results/llm_zero_shot.md), [base run](results/base_run).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
./run_all.sh synthetic                  # smoke test on 200 synthetic rows; results are meaningless
./run_all.sh                            # real run; needs the Capital IQ exports (DATA.md) and a logged-in `claude` CLI
.venv/bin/streamlit run app/demo.py     # local demo; falls back to the synthetic sample
```

`data/sample/synthetic_ciq_export.csv` is **synthetic**. Every number in `results/` and `figures/` comes from the real exports.

## Limitations

- **Few events.** 40 adverse breaks in the test period, 16 with usable cash-deal prices. Every interval is wide; "no
  measurable gain" is not "no gain".
- **Licensed data and vendor fields.** Results need Capital IQ access. The vendor fills fewer fields for cancelled deals,
  may store revised terms, and classifies buyers as of today. Label audit against EDGAR: 3 of 76 checkable statuses
  contradicted. Break reasons come from vendor notes; 15% of breaks have no stated reason.
- **Prices are monthly.** Entry is up to a month after announcement, exits are on the next monthly mark, no costs,
  dividends or financing. A few deals with very wide spreads (March 2020) dominate average returns.
- **The model is blind to antitrust overlap,** the cause of the largest losses (see the case studies).
- **The LLM zero-shot score (AUC 0.82) is not evidence about the text:** the same LLM named 92% of the anonymised targets.

Decisions: [DECISIONS.md](DECISIONS.md). Slides: [SLIDE_NOTES.md](SLIDE_NOTES.md). Code: MIT licence. Data: not included.

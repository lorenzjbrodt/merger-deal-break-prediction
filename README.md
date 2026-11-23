# Underwriting merger deals at announcement: break risk, time to closing, loss if it breaks

MIT 15.071 Project 1 (Predictive AI), solo project by Lorenz J. Brodt. Project page:
<https://lorenzjbrodt.github.io/merger-deal-break-prediction/>

## Problem

A merger-arbitrage fund buys the target after a deal is announced. It earns a small spread if the deal closes and loses
many times that if it breaks. The portfolio manager underwrites three numbers per deal: the **probability of an adverse
break** (cancelled for a reason other than a superior competing bid), the **time to closing**, and the **loss if it
breaks**. This project predicts all three from what is known at announcement, combines them into an expected annualized
return, and tests whether ranking deals by it makes money out of sample.

## Data (none of the licensed data is in this repo; see [DATA.md](DATA.md))

- **Capital IQ** deal screen: 4,744 signed US public deals 2001-2025; core sample 4,468 without reverse mergers; 346
  breaks, of which 246 adverse ([taxonomy](results/outcome_taxonomy.md)).
- **SEC EDGAR** announcement filings (press release for 88% of deals), read by an LLM into 29 validated contract terms and
  the parties' stated expected closing date ([validation](results/llm_extraction_validation.md), [closing guidance](results/closing_guidance_validation.md)).
- **CRSP daily prices via WRDS** for 93% of core targets ([linking](results/crsp_link.md), [checks](results/crsp_prices.md));
  Compustat; Hoberg-Phillips product similarity; FRED. Fallback without WRDS: Capital IQ monthly closes ([results/monthly](results/monthly)).

## Method

Train on 2001-2018, test on 2019-2025, everything fitted in-fold. Second evaluation set: **pooled out-of-time
2013-2025**, each year scored by models refitted on earlier years only (13 origins). Intervals: bootstrap over deals
(test) and over announcement years (pooled). Deals are bought at the close one trading day after the announcement.

## Headline results

| The three numbers | Test 2019-2025 | Pooled out-of-time 2013-2025 |
|---|---|---|
| **Break probability**: final model, ROC AUC (1,083 / 2,176 deals; 40 / 84 adverse breaks) | 0.633 [0.551, 0.717] | 0.609 [0.538, 0.686] |
| The market's day-1 implied probability, priced all-cash deals (22 / 38 adverse breaks) | 0.643 [0.523, 0.755] | 0.721 [0.642, 0.800] |
| **Time to closing**: final model, mean absolute error in days (median by structure: 64 / 61) | 52.7 [48.1, 57.4] | 48.3 [44.2, 52.4] |
| The parties' own stated expected closing, same deals (model: 46.9 / 43.6) | 46.4 | 45.4 |
| **Loss if it breaks**: mean return of adverse breaks from the day-1 close (38 / 81 deals) | -29% | -25% |
| **Ranking by expected annualized return**: top third minus bottom third, points a year | +11.6 [+4.1, +19.0] | +9.3 [+5.5, +13.5] |
| The same test for the raw spread (no model) | +9.4 [+1.8, +16.8] | +8.4 [+5.0, +12.7] |
| Top third by the model minus top third by raw spread | +3.3 [+1.1, +6.1] | +1.9 [+0.4, +3.3] |

- **Break probability: weak, and the market does it better.** Nothing added to base deal terms helps measurably
  (pooled: extended features +0.025 [-0.025, +0.076], text +0.022 [-0.014, +0.059], product overlap +0.006
  [-0.010, +0.019]). Ten further day-1 signals in two pre-specified models did not beat the market-implied probability
  (pooled -0.125 [-0.246, -0.010] and -0.061 [-0.143, +0.011]). [Ladder](results/benchmark_table.md),
  [pooled](results/pooled_oot.md), [market](results/market_benchmark.md), [day-1 signals](results/day1_signals.md),
  [overlap](results/overlap.md).
- **Time to closing: predictable.** R2 on log days 0.54. LLM-extracted terms cut the error by 1.6 days [-2.6, -0.6];
  text embeddings add nothing. The model matches management guidance, and a model that also reads the guidance beats
  it by 6.6 days [-9.5, -3.9]. [Details](results/duration.md).
- **Loss if it breaks: large and barely predictable.** Adverse breaks lost 28% on average over all years (184 deals), far more than the 11% a fall
  back to the pre-announcement price implies. No forecast beats the average leave-one-year-out. [Details](results/loss_given_break.md).
- **In dollars** (all-cash deals, before costs): the top third by expected annualized return earned 11.7% a year pooled
  against 2.3% for the bottom third; holding everything earned 6.4%. Most of that is the spread itself. In a daily
  calendar-time portfolio the top half by the model beat the top half by raw spread by 0.6 points a year
  [-0.9, +2.0], Sharpe ratio 0.80 against 0.77. [Ranking test](results/expected_return.md), [backtest](results/backtest.md).

More: [segments](results/segments.md), [SHAP](results/shap.md), [calibration](results/calibration.md),
[robustness](results/robustness.md), [case studies](results/case_studies.md), [label audit](results/label_audit.md),
[LLM zero-shot (exploratory)](results/llm_zero_shot.md), [base run](results/base_run).

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
./run_all.sh synthetic                  # smoke test on 200 synthetic rows; results are meaningless
./run_all.sh                            # real run: Capital IQ exports, a logged-in `claude` CLI, optionally WRDS (DATA.md)
.venv/bin/streamlit run app/demo.py     # local demo with the three numbers per deal; falls back to the synthetic sample
```

`data/sample/synthetic_ciq_export.csv` is **synthetic**. Every number in `results/` and `figures/` comes from the real data.

## Limitations

- **Few events.** 40 adverse breaks in the test period (22 priced all-cash), 84 pooled (38 priced). Pooled results mix
  13 fitted models and resample only 13 years. "No measurable gain" is not "no gain".
- **The ranking result is mostly a risk premium.** The top third holds 25 of the 38 pooled adverse breaks: wide spreads
  pay because they are risky. The model's edge over the free spread ranking is 1 to 3 points in the tercile test and
  not distinguishable from zero in the calendar-time portfolio.
- **Backtest.** No transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1
  close. Deals without announcement text keep the vendor's final offer in the spread.
- **Licensed data and vendor fields.** Results need Capital IQ and WRDS. The vendor fills fewer fields for cancelled
  deals and stores final terms. Label audit against EDGAR: 3 of 76 statuses contradicted.
- **The LLM zero-shot score (AUC 0.82) is not evidence:** the same LLM named 92% of the anonymised targets.

Decisions: [DECISIONS.md](DECISIONS.md). Slides: [SLIDE_NOTES.md](SLIDE_NOTES.md). Code: MIT licence. Data: not included.

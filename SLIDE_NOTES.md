# Slide notes

Audience: the portfolio manager of a merger-arbitrage fund. Five slides, two minutes. Brackets are bootstrap 95%
intervals. "Test" = deals announced 2019-2025, models trained on 2001-2018. "Pooled" = deals announced 2013-2025, each
year scored by models refitted on earlier years only. Returns are before costs.

## Storyline in one sentence

An arbitrageur underwrites three numbers per deal; of the three, only time to closing is well predicted from the
announcement, the market's own spread beats the break model, and ranking deals by expected annualized return earns
about ten points a year more in the top third than in the bottom third, most of which the raw spread delivers for free.

Result and recommendation in plain words (also the first lines of the README): only time to closing is predicted well;
use it to annualize spreads and rank deals, read break risk off the spread, do not size on the break model.

## Five slides

**1. The decision: three numbers per deal.**
- Buy the target one day after announcement: a few percent if the deal closes, a large loss if it breaks.
- Expected annualized return = [P(close) x spread + P(topped) x (spread + bump) + P(adverse break) x loss] / expected time.
- Data: 4,744 signed US public deals since 2001 (Capital IQ), announcement filings (EDGAR), daily prices (CRSP).
- Cleaned first: 100 of 346 breaks were higher bids, 276 "deals" were de-SPACs, and the vendor's offer is the final
  offer, not the one announced.
- Figure: `figures/break_rate_by_year.png`.

**2. Number one, break probability: weak, and the market does it better.**
- Final model: AUC 0.633 [0.551, 0.717] test, 0.609 [0.538, 0.686] pooled (2,176 deals, 84 adverse breaks).
- Market-implied probability from the day-1 spread, priced all-cash deals: 0.643 test, 0.721 [0.642, 0.800] pooled.
  Model minus market, pooled: -0.12 [-0.28, +0.01].
- Nothing added helps: extended features, text, LLM terms, product overlap, ten day-1 market signals (two
  pre-specified models, evaluated once: -0.125 and -0.061 against the market).
- Figure: `figures/model_vs_market.png`.

**3. Number two, time to closing: predictable.**
- Median deal: 107 days. Model without the parties' guidance: mean absolute error 52.7 days [48.1, 57.4] test, against
  63.5 for the median by structure; it ties with the guidance itself (46.9 against 46.4 days on the same deals).
- Product forecast (also reads the stated closing where there is one, chosen by CV in every origin): 47.1 days
  [43.0, 51.5] test, 43.1 [39.6, 46.5] pooled; R2 on log days 0.62; 6.6 days better than guidance alone [-9.5, -3.9].
- LLM-extracted terms help a little (-1.6 days [-2.6, -0.6]); text embeddings add nothing.
- The 80% interval, conformally calibrated on training years only, contains 80% of test deals and 81% pooled
  (73% before calibration); median width 114 days.
- Figure: `figures/duration.png`.

**4. Number three, loss if it breaks: large, and barely predictable.**
- 184 adverse breaks with prices: mean -28%, median -26%, worst tenth below -73%. Topped deals: +19%.
- A fall back to the pre-announcement price would imply only -11%.
- No forecast beats the historical average leave-one-year-out; in the time split a small boosted model lowers the
  mean absolute error from 0.31 to 0.28 (-0.031 [-0.054, -0.010]) on 38 test breaks.
- Figure: `figures/loss_given_break.png`.

**5. The dollar result.**
- Pooled, 1,205 all-cash deals: top third by expected annualized return 12.3% a year, middle 3.1%, bottom 2.2%
  ($12.3m, $3.1m and $2.2m per $100m); top minus bottom +10.0 points [+6.5, +13.9]. Test period: +12.6 [+5.3, +20.3].
- The raw spread alone gives +8.4 [+5.0, +12.7]. Model top third minus spread top third: +2.5 [+0.9, +4.1] pooled,
  +4.1 [+1.6, +7.1] test.
- Daily calendar-time portfolio, 10% position cap, pooled: all deals 7.2% a year, Sharpe 0.72; top half by the model
  9.6%, Sharpe 0.80; top half by raw spread 8.7%, Sharpe 0.77. Model minus spread: +0.9 points [-0.9, +2.6].
  Test period alone: 8.6% against 8.5%, +0.1 [-2.9, +3.2].
- The top third holds 22 of the 38 adverse breaks: wide spreads pay because they are risky.
- Recommendation: use the duration model to annualize spreads, read break risk off the spread, do not size on the break
  model. Next dollar: break-reason data the market also lacks on day 1.
- Figure: `figures/expected_return.png`.

## Backup slides

- Break model ladder, four target and sample combinations, day-1 step: `results/benchmark_table.md`.
- Pooled out-of-time ladder, by-year table, embargo check: `results/pooled_oot.md`.
- Market benchmark and robustness panels: `results/market_benchmark.md`, `figures/model_vs_market.png`.
- Day-1 signals: specification, results, what CV preferred, coefficients: `results/day1_signals.md`.
- Antitrust overlap (Hoberg-Phillips similarity, SIC shares): `results/overlap.md`.
- Time to closing: full ladder, guidance benchmark, product model, conformal interval (80% and 81% coverage), segments,
  survival robustness: `results/duration.md`; guidance extraction: `results/closing_guidance_validation.md`.
- Loss if it breaks: realised returns, raw and net of market, all forecasts: `results/loss_given_break.md`.
- Expected return: all rankings, attribution, calendar-time portfolios, test period, and the old-versus-new table for the
  product time-to-closing model: `results/expected_return.md`. Index of all result files: `results/README.md`.
- Numbers for charts (monthly portfolio values, returns per third with intervals, counts, segments): `results/deck_numbers.json`.
- Backtest of skip rules and the model-versus-market rule: `results/backtest.md`, `figures/backtest.png`.
- Linking to CRSP and price checks, which offer is known on day 1: `results/crsp_link.md`, `results/crsp_prices.md`.
- Monthly-price fallback (runs without WRDS): `results/monthly/`.
- Base run versus later runs: `results/base_run/benchmark_table.md`.
- Calibration, SHAP, segments, three-class view, robustness, ablation: `results/calibration.md`, `results/shap.md`,
  `results/segments.md`, `results/three_class.md`, `results/robustness.md`, `results/ablation.md`.
- Six case studies: `results/case_studies.md`. Label audit: `results/label_audit.md`.
- LLM extraction validation: `results/llm_extraction_validation.md`. Zero-shot score and why it cannot be trusted:
  `results/llm_zero_shot.md`.
- Demo with the three numbers per deal: `streamlit run app/demo.py`. The committed screenshots show the product layout
  on the synthetic fallback (made-up deals and values): `figures/demo_synthetic_*.png`.

## Two-minute pitch outline

- The trade: small gain if the deal closes, large loss if it breaks. A PM underwrites three numbers: break
  probability, time to closing, loss if it breaks.
- The data: 4,744 signed deals, SEC filings from the first week, CRSP daily prices; cleaned for higher bids, de-SPACs
  and the vendor's final-offer field.
- Break probability: AUC 0.61 to 0.63 from deal terms; nothing else helps; the market's day-1 spread reaches 0.72.
- Time to closing: the predictable one. Error of 47 days against 64 for a structure median; 7 days better than
  management guidance; the 80% interval holds 80% of deals.
- Loss if it breaks: 28% on average, not the 11% a fall back to the old price implies, and close to unpredictable.
- Put together: the top third by expected annualized return earned 12.3% a year, the bottom third 2.2%.
- Honest part: the raw spread delivers most of that for free; the model adds 2.5 to 4 points in the tercile test and
  0.9 points [-0.9, +2.6] in a daily portfolio. The top third also holds most of the breaks.
- So what: annualize spreads with the duration model, trust the spread for break risk, treat the rest as a screen.
- What I would not trust: anything resting on 22 to 38 priced adverse breaks, results before costs, and the LLM
  zero-shot score.

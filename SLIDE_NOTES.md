# Slide notes

- Deck: “AI underwriting for merger arbitrage”, 16 pages: 1 cover, 2 agenda, 3 executive summary, 4 Problem, 5 Data and
  methodology, 6 Time to closing, 7 Break risk, 8 Loss if a deal breaks, 9 Impact, 10 Recommendation, 11 Appendix
  divider, 12 to 16 = A1 to A5. The deck file is not in this repository.
- Audience: the portfolio manager of a merger arbitrage fund.
- “Test” = deals announced 2019 to 2025, models trained on 2001 to 2018. “Pooled” = deals announced 2013 to 2025, each
  year scored by models refitted on earlier years only.
- Intervals are 95% bootstrap intervals, written “[a to b]” as on the deck. Returns are before costs.
- Slide text below is verbatim. “Numbers” bullets name the result file each figure comes from; “computed” marks
  arithmetic on figures shown on the same page.

## Video path (2:00)

- A pitch without a live demo, pages in the order 1, 4, 5, 3, 9, 10.
- Timings: page 1 0:05; page 4 0:30; page 5 0:20; page 3 0:30 (shown after page 5); page 9 0:15; page 10 0:20. Total
  2:00. The repo link is shown at the end.
- Not shown in the video: pages 2, 6, 7, 8, and the appendix (11 to 16).

## Page 1. Cover

- Kicker: 15.071 THE AI EDGE | PROJECT 1: PREDICTIVE AI
- Headline: AI underwriting for merger arbitrage
- Subline: When a deal closes, whether it breaks, at what cost
- Byline: Prepared for the portfolio manager of a merger arbitrage fund / Lorenz Brodt | November 2026 / Code and
  results: github.com/lorenzjbrodt/merger-deal-break-prediction
- Speaker notes:
  - Video: 0:05
  - Where AI helps a merger arbitrage fund underwrite a deal

## Page 2. Agenda

- Headline: Agenda
- Items (marker, title, subtitle, page):
  - I Executive summary: Results and recommendation on one page (3)
  - 1 Problem: What a merger arbitrage position earns and loses (4)
  - 2 Data and methodology: How deal data, an LLM, and machine learning produce the three estimates (5)
  - 3 Time to closing: How long a deal takes to close (6)
  - 4 Break risk: How likely a deal is to break (7)
  - 5 Loss if a deal breaks: What the fund loses when a deal breaks (8)
  - 6 Impact: What better estimates are worth in fund returns (9)
  - 7 Recommendation: How a fund should use the results (10)
  - II Appendix: Data, model detail, and robustness (11)
- Speaker notes:
  - Video: not shown

## Page 3. Executive summary

- Kicker: EXECUTIVE SUMMARY (marker I)
- Headline: AI improves deal timing and ranking but does not beat the market price on break risk, and breaks cost far
  more than assumed
- Exhibit “Closing date error, in days”: Model 47.10, Median by deal type 63.52; label “-16.42 days”.
- Exhibit “Break risk ranking, AUC(1)”: Model 0.60, Day-1 spread 0.72; label “-0.12 AUC”.
- Exhibit “Loss if a deal breaks, in %(2)”: Historical average 27.94, Fallback assumption 10.67; label “2.62x”.
- Exhibit “Top-third return, in % a year(3)”: Model 12.26, Annualized spread 10.58; label “+1.67 points”.
- Box: Recommendation: Use the model for timing and ranking, the spread for break risk, and a 31% loss estimate for
  sizing
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) AUC = chance a broken deal ranks riskier than a closed one; bars from 0.50; (2) 184 breaks; no model
  beat it; sizing uses 31.21%; (3) Unrounded gap; 95% interval 0.20 to 3.25; daily edge unconfirmed.
- Speaker notes:
  - Video: 0:30, shown after page 5
  - Timing and ranking improve; the model does not beat the spread on break risk; losses are far larger than assumed
- Numbers:
  - 47.10 and 63.52: `results/duration.json` (`product.split.scores.product.mae`, `split_full.scores.median_structure.mae`);
    -16.42 computed.
  - 0.60, 0.72 and -0.12: `results/market_benchmark.json` (`benchmark.pooled_cash`: model 0.599, `p_mkt` 0.721, model
    minus `p_mkt` -0.121), 1,205 priced all-cash deals, 2013 to 2025.
  - 27.94, 10.67 and 184: `results/deck_numbers.json` (`loss_if_break`), `results/loss_given_break.json` (`realised`);
    2.62 computed (27.94 / 10.67). 31.21: `results/common_sample_outcomes.json` (`pooled_2013_2025`, mean return of the
    38 adverse breaks).
  - 12.26, 10.58, +1.67 and 0.20 to 3.25: `results/expected_return.json` (`views.pooled.rankings.ear` and
    `rankings.spread_ann`, top third; `model_top_minus_benchmark_top.spread_ann`).

## Page 4. Problem

- Kicker: PROBLEM (marker 1)
- Headline: A merger arbitrage position earns little when a deal closes and loses about 21 times as much when it breaks,
  so three estimates decide its value
- Box “Stakeholder decision”: After a takeover announcement, the portfolio manager decides whether and how much to
  invest; the spread is earned only if the deal closes
- Box “Expected annualized return of a position”: P(close) x spread + P(topped) x topping gain - P(break) x loss, over
  “Years to resolution”
- Exhibit “Median return from the day-1 close to the outcome, in %(1)”:
  - Deal closes, 95.44% of deals: +1.39
  - Adverse break, 3.15% of deals: -28.91
  - Topped by a higher bid, 1.41% of deals: +14.20
  - Label: about 21x the gain of a closed deal
- Table “Three estimates and the benchmark for each” (Estimate, Benchmark):
  - Time to closing: Median by deal type and the parties’ stated closing date
  - Break risk: Day-1 spread (market-implied)
  - Loss if a deal breaks: Fallback assumption: price returns to its pre-announcement level
- Source(s): S&P Capital IQ, CRSP via WRDS; own analysis.
- Note(s): (1) Medians of 1,205 priced all-cash deals, 2013 to 2025; 28.91 / 1.39 = 20.80; spread = offer price / day-1
  close - 1.
- Speaker notes:
  - Video: 0:30
  - Close earns 1.39%, break costs 28.91%; three estimates decide the value
- Numbers: all from `results/common_sample_outcomes.md` and `.json` (`pooled_2013_2025`: 1,205 deals, 1,150 closed, 38
  adverse breaks, 17 topped; outcome shares and median returns); 20.80 computed from the two medians shown. Formula:
  `src/expected_return.py`, where the topping gain is spread + bump and the loss enters as a negative return that is
  added. Spread definition: `results/crsp_prices.md`.

## Page 5. Data and methodology

- Kicker: DATA AND METHODOLOGY (marker 2)
- Headline: Deal data, an LLM that reads each press release, and machine learning produce three estimates, each tested
  only on deals the model has not seen
- Card “Data”: 4,468 US public takeovers from S&P Capital IQ, 2001 to 2025, plus daily prices (CRSP), financials
  (Compustat), and market data (FRED)
- Card “LLM”: Claude Haiku reads each SEC EDGAR press release and extracts 29 contract terms plus the parties’ stated
  closing date
- Table “How each estimate is made and tested” (Time to closing / Break risk / Loss if a deal breaks):
  - Predicts: Days from announcement to closing / Adverse break, yes or no / Return from day-1 close to break, in %
  - Method: Boosted trees on deal, firm, and LLM data / Boosted trees on deal terms and financials / Historical
    average; no model beat it reliably
  - Learns from: Deals announced 2001 to 2018 / All earlier years, refitted each year / All other years, one year left
    out at a time
  - Tested on: 1,026 completed deals, 2019 to 2025 / 1,205 priced all-cash deals, 2013 to 2025 / 184 adverse breaks,
    2001 to 2025
  - Benchmark: Median by deal type and parties’ stated date / Day-1 spread / Fallback assumption
- Strip: Output: The three estimates give each deal’s expected annualized return, which ranks the deals each year on
  earlier years only
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): US public targets above $100m with a signed agreement; models tuned by cross-validation on training years;
  corrections in A1, alternative models in A2 and A4.
- Speaker notes:
  - Video: 0:20
  - LLM reads press releases, models make three estimates, each tested on unseen deals
- Numbers:
  - 4,468: `results/deck_numbers.json` (`core_sample.total`); filter steps in `results/sample_filters.json`
    (`filter_log`).
  - 29 contract terms: the 26 fields kept in `results/llm_extraction_validation.md`, with buyer type coded as four
    indicators and structure as one (the count is written by `src/llm_extract.py validate`); the stated closing date is
    read separately by `src/llm_closing.py`.
  - 1,026: `results/duration.json` (`product.split.n`). 1,205: `results/expected_return.json` (`views.pooled.n`).
    184: `results/loss_given_break.json` (`realised`, adverse breaks, all years).

## Page 6. Time to closing

- Kicker: TIME TO CLOSING (marker 3)
- Headline: The model beats the median by deal type, and with the parties’ stated date it beats that date too, so
  spreads can be annualized more precisely
- Exhibit “Closing date error, in days(1)” (column chart; Benchmark, Model step, Final model): Median of all deals
  72.62; Median by deal type 63.52; Model with deal terms 58.45; Plus financials and LLM terms 52.68; Plus stated date
  47.10; label “-16.42 days”.
- Exhibit “Closing date error on deals with a stated date, in days(2)” (column chart): Parties’ stated date 46.43;
  Model without stated date 46.86; Model with stated date 39.82; bracket “-6.61 days [-9.51 to -3.86]”.
- Strip: Practical implication: Annualize spreads with the model forecast; the stated date is a median of 14.50 days
  too late, which understates the annualized spread
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) 1,026 completed deals, 2019 to 2025; deal type = merger or tender offer; whiskers are 95% intervals;
  (2) Only 807 of the 1,026 deals state a closing date, so this comparison uses those 807.
- Speaker notes:
  - Video: not shown
  - Each input lowers the error; with the stated date the model beats that date
- Numbers:
  - Left chart: `results/duration.json` (`split_full.scores`: `median`, `median_structure`, `lgbm_base`, `final`;
    `product.split.scores.product`); -16.42 computed (63.52 minus 47.10).
  - Right chart: `results/duration.json` (`guidance_split.scores`: `guidance`, `lgbm_llm`, `lgbm_guide`; `n` = 807);
    -6.61 [-9.51 to -3.86] from `guidance_split.pairs["lgbm_guide minus guidance"]`.
  - 14.50: `results/duration.json` (`guidance_split.bias.median_actual_minus_guidance`).

## Page 7. Break risk

- Kicker: BREAK RISK (marker 4)
- Headline: The model does not beat the market price at predicting breaks, and more data does not reliably help, so
  break risk should come from the spread
- Exhibit “Break risk ranking, AUC \[95% interval\](1)”: Model 0.60 [0.48 to 0.70]; Day-1 spread 0.72 [0.64 to 0.80];
  label “-0.12 AUC [-0.28 to +0.00]”.
- Table with interval plot “Change in AUC from adding each input(2)” (Added information, Change [95% interval]; axis
  at 0):
  - Buyer and target financials: +0.03 [-0.02 to +0.08]
  - Press release text and LLM-read terms: +0.02 [-0.01 to +0.06]
  - Product overlap of buyer and target: +0.01 [-0.01 to +0.02]
  - Market conditions: -0.03 [-0.05 to -0.00]
- Box: Excluded evidence: An LLM risk score reached an AUC of 0.82, but the LLM identified 92% of anonymized targets,
  so it may know the outcome (look-ahead bias)
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) 1,205 priced deals; AUC = chance a broken deal ranks riskier than a closed one; bars from 0.50; gap not
  significant; (2) 2,176 deals; each row vs. the model without it; dot = change, line = interval.
- Speaker notes:
  - Video: not shown
  - The model does not beat the spread on break risk, and more data does not reliably help
- Numbers:
  - Left chart: `results/market_benchmark.json` (`benchmark.pooled_cash`: model 0.599 [0.476 to 0.702], `p_mkt` 0.721
    [0.642 to 0.800], model minus `p_mkt` -0.121 [-0.282 to +0.005]; the upper bound is +0.0046 unrounded).
  - Table: `results/pooled_oot.json` (`full.pairs`): financials = `lgbm_ext minus lgbm_base`, +0.025 [-0.025 to +0.076]
    (lower bound -0.0248 unrounded); market conditions = `lgbm_extm minus lgbm_ext`, -0.028 [-0.054 to -0.002]; text
    and LLM terms = `lgbm_full_text minus lgbm_extm`, +0.022 [-0.014 to +0.059]. Product overlap:
    `results/overlap.json` (`break_pooled.pairs["overlap minus final"]`), +0.006 [-0.010 to +0.019].
  - 0.82 and 92%: `results/llm_zero_shot.md` (0.822 on test years 2019 to 2025; 92% identified).

## Page 8. Loss if a deal breaks

- Kicker: LOSS IF A DEAL BREAKS (marker 5)
- Headline: Adverse breaks lose 2.62 times the fallback assumption, so positions should be sized for a much larger
  loss
- Exhibit “Average loss in an adverse break, in %(1)”: Historical average of 184 adverse breaks 27.94; Fallback
  assumption, price returns to pre-announcement level 10.67; label “2.62x”.
- Exhibit “Break probability at which a position breaks even, in %(2)” (line chart over “Day-1 spread, in %”, 1 to 6):
  Fallback assumption loss of 10.67%: 8.57, 15.79, 21.95, 27.27, 31.91, 35.99; Historical average loss of 27.94%: 3.46,
  6.68, 9.70, 12.52, 15.18, 17.68; marker “At a 3% spread”: 21.95% and 9.70%.
- Strip: Practical implication: No loss forecast reliably beat an average, so size positions for 31.21%, the mean of the
  38 adverse breaks among ranked deals
- Source(s): S&P Capital IQ, CRSP via WRDS; own analysis.
- Note(s): (1) 184 adverse breaks, 2001 to 2025; (2) Break-even = spread / (spread + loss); 3.00 / (3.00 + 27.94) =
  9.70%.
- Speaker notes:
  - Video: not shown
  - Breaks cost 2.62x the fallback assumption, so a position tolerates far less break risk
- Numbers:
  - 27.94, 10.67 and 184: `results/deck_numbers.json` (`loss_if_break`: `mean_realised_pct`, `mean_fallback_pct`,
    `adverse_breaks`), `results/loss_given_break.json` (`realised`); 2.62 computed.
  - Line chart: computed as spread / (spread + loss) for spreads of 1% to 6%.
  - 31.21 and 38: `results/common_sample_outcomes.json` (`pooled_2013_2025`); no loss forecast beat the average:
    `results/loss_given_break.json` (`adverse.ret.loyo`, detail on A4).

## Page 9. Impact

- Kicker: IMPACT (marker 6)
- Headline: Buying the top third by the model’s expected annualized return earns $1.67m more per $100m a year than the
  top third by annualized spread
- Exhibit “Annual return by third of deals, in %(1)” (By the model, By annualized spread): Top third 12.26 and 10.58;
  Middle third 3.07 and 2.75; Bottom third 2.25 and 2.13; label “+1.67 points”.
- Table “Top third by the model, in $m per $100m a year(2)” (Compared with, Difference [95% interval]):
  - All deals: +5.90 [3.92 to 7.97]
  - Top third by annualized spread: +1.67 [0.20 to 3.25]
  - Same comparison, 2019 to 2025: +3.14 [0.47 to 6.09]
  - Top third by raw spread: +2.48 [0.95 to 4.14]
- Table “Adverse breaks that land in the top third” (Ranked by, Adverse breaks): The model 22 of 38; Annualized spread
  26 of 38.
- Box: Caveat: As a daily portfolio, the edge over the raw spread is +0.87 points a year (95% interval -0.88 to +2.60),
  so it is not confirmed; returns are before costs
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) 1,205 priced all-cash deals, 2013 to 2025, ranked on prior years; return = profit / capital-years;
  break risk from a day-1 price model in 7 of 13 years, else the spread; unrounded gap; (2) Bootstrap.
- Speaker notes:
  - Video: 0:15
  - Top third 12.26% a year, $1.67m per $100m more than the annualized spread
- Numbers: all from `results/expected_return.json`:
  - Thirds: `views.pooled.rankings.ear.terciles` and `rankings.spread_ann.terciles` (`annual_return`); the annualized
    spread is the day-1 spread divided by the deal-type median time to closing.
  - All deals: `rankings.ear.top_minus_all`; annualized spread: `model_top_minus_benchmark_top.spread_ann`; raw spread:
    `model_top_minus_benchmark_top.spread`; 2019 to 2025: `views.test.model_top_minus_benchmark_top.spread_ann`.
  - 22 and 26 of 38: `rankings.ear.terciles.top.adverse`, `rankings.spread_ann.terciles.top.adverse`, `views.pooled.adverse`.
  - +0.87 [-0.88 to +2.60]: `views.pooled.calendar.model_minus_spread`.
  - 7 of 13 years: `results/expected_return.md` (“the day-1 model in 7 of 13 pooled origins”) and
    `results/day1_signals.json` (`used_in_M`).

## Page 10. Recommendation

- Kicker: RECOMMENDATION (marker 7)
- Headline: Use the model for timing and ranking, the spread for break risk, and a 31% loss estimate for sizing
- Four steps:
  - 1 TIMING: 47.10 days, average error, against 63.52 for the median by deal type; Annualize the spread with the
    forecast
  - 2 BREAK RISK: 0.72, AUC of the spread, against 0.60 for the model; Read break risk from the spread
  - 3 SIZING: 31.21%, mean loss of 38 adverse breaks among ranked deals; Size each position for a 31% loss
  - 4 RANKING: +$1.67m, per $100m a year; 95% interval 0.20 to 3.25; Rank by expected annualized return
- Box “Limitations”: 38 adverse breaks among ranked deals; losses vary widely (10th percentile of all 184 at -73.01%);
  before costs; daily-portfolio edge not confirmed; a day-1 price model gave break risk in 7 of 13 backtest years
- Box “Next step”: Run the ranking alongside the current process, with an agent underwriting each new deal (Project 2),
  and invest once live results confirm the edge
- Source(s): Own analysis; code and results at github.com/lorenzjbrodt/merger-deal-break-prediction.
- Note(s): Evidence on pages 6 to 9.
- Speaker notes:
  - Video: 0:20; order 1, 4, 5, 3, 9, 10 = 2:00; repo link on screen at the end, no live demo
  - Timing and ranking from the model, break risk from the spread, sizing for a 31% loss
- Numbers: as on pages 3, 6, 7, 8, and 9; -73.01: `results/loss_given_break.json` (`realised["adverse_break | all
  years"].p10`).

## Page 11. Appendix divider

- A1 Data sources, corrections, and cancellation reasons (12); A2 Break model in the test period (13); A3 Model drivers
  and stability over time (14); A4 Time to closing and loss in detail (15); A5 Daily portfolio and attribution (16).

## Page 12. A1

- Kicker: DATA SOURCES, CORRECTIONS, AND CANCELLATION REASONS (marker A1)
- Headline: The data combine six sources, and higher bids are separated from true breaks because they are gains for the
  fund
- Table “Data sources” (Source, Content):
  - S&P Capital IQ: Deal terms, buyer and target financials, and outcomes
  - SEC EDGAR: Press release for 88% of deals; an LLM extracts 29 contract terms and the stated closing date
  - CRSP (WRDS): Daily share prices for 93% of targets
  - Compustat, Hoberg-Phillips, FRED: Fundamentals, product overlap, and market conditions
- Stat tiles “Corrections before modeling(2)”: 100 higher competing bids labeled as gains; 247 offers reset to the
  original offer
- Exhibit “Cancellations by reason(1)” (legend: Gain for the fund, Adverse break): Higher competing bid 100; then “246
  adverse cancellations, 198 of them in the core sample”: Mutual termination or other 77, Regulatory or antitrust block
  57, No reason stated 52, Shareholder vote failed 31, Buyer walked or adverse change 21, Financing failure 8.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) LLM-classified from vendor deal notes, checked on 50 deals; 184 of the 198 breaks have prices for the
  loss analysis; (2) The vendor stores the final offer, so 247 offers were reset from the press release.
- Numbers: 88%: `results/text_coverage.md` (all deals); 93%: `results/crsp_link.md` (4,140 of 4,468 core deals linked);
  29: see page 5; 247: `results/crsp_prices.md`; reasons and the check on 50 deals: `results/outcome_taxonomy.md`;
  246 and 198: `results/deck_numbers.json` (`core_sample`); 184: `results/deck_numbers.json` (`loss_if_break`).

## Page 13. A2

- Kicker: BREAK MODEL IN THE TEST PERIOD (marker A2)
- Headline: In the 2019 to 2025 test, the break model ranks breaks better than chance and matches the day-1 spread on
  priced deals
- Exhibit “AUC [95% interval], test 2019 to 2025(1)”: One rule 0.53 [0.45 to 0.61]; Logistic model 0.61 [0.52 to
  0.69]; Break model 0.63 [0.55 to 0.72].
- Table “AUC by segment, test 2019 to 2025(2)” (Deals, Adverse breaks, AUC): Cash 686, 24, 0.65; Stock 210, 11, 0.58;
  Mixed 187, 5, 0.62; Small 252, 10, 0.56; Mid 297, 9, 0.70; Large 534, 21, 0.64.
- Table “Break model and day-1 spread on priced test deals(3)” (Ranking by, AUC [95% interval]): Break model 0.63 [0.52
  to 0.73]; Day-1 spread 0.64 [0.52 to 0.76].
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) 1,083 deals, 40 adverse breaks; one rule = best single-variable rule; all trained on 2001 to 2018; bars
  from 0.50; (2) Size cutoffs $350m and $1,554m; (3) 623 priced all-cash deals, 22 breaks.
- Numbers: left chart `results/metrics.json` (`runs.primary.full`: `one_rule`, `logit_base`, `lgbm_ext`) and
  `results/benchmark_table.md`; segments `results/segments.md` (column AUC 3x); priced test deals
  `results/market_benchmark.json` (`benchmark.test_cash`, intervals over deals).

## Page 14. A3

- Kicker: MODEL DRIVERS AND STABILITY OVER TIME (marker A3)
- Headline: The break model relies most on target size and termination fees, and its accuracy varies strongly from
  year to year
- Exhibit “Model drivers(1)” (Higher value, higher risk / Higher value, lower risk): Target revenue 0.13; Reverse
  termination fee, % of value 0.13; Price-to-book multiple 0.09; Cash share of consideration 0.09; Target price vs.
  52-week high 0.08; Target termination fee, % of value 0.08; Acquirer price vs. 52-week high 0.07; Target cash to assets
  0.07.
- Exhibit “AUC by year, range 0.41 to 0.92(2)”: 2013 0.55; 2014 0.77; 2015 0.61; 2016 0.77; 2017 0.55; 2018 0.55; 2019
  0.50; 2020 0.61; 2021 0.47; 2022 0.84; 2023 0.92; 2024 0.68; 2025 0.41.
- Exhibit “Adverse break rate by announcement year, in %(3)”: 2013 3.38; 2014 3.57; 2015 5.80; 2016 1.54; 2017 4.79;
  2018 4.81; 2019 0.62; 2020 8.74; 2021 3.63; 2022 6.08; 2023 2.70; 2024 3.45; 2025 2.70.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) Mean absolute SHAP value, deals 2019 to 2025; associations only; (2) Each year scored by a model fitted
  on earlier years; 1 to 12 adverse breaks a year; bars from 0.50; (3) 2,176 deals, 2013 to 2025.
- Numbers: drivers `results/shap.md` (`results/metrics.json`, `extras.shap.rows`); AUC by year and break rates
  (adverse breaks / deals, computed) `results/pooled_oot.json` (`by_year`).

## Page 15. A4

- Kicker: TIME TO CLOSING AND LOSS IN DETAIL (marker A4)
- Headline: The closing forecast has a lower error than the median by deal type in every segment, while no loss
  forecast reliably beats the historical average
- Exhibit “Closing date error by segment, in days(1)” (grouped column chart, Median by deal type and Model; label
  under each segment): Mergers, “855 deals; -19.18”, 73.14 and 53.96; Tender offers, “171 deals; -2.61”, 15.40 and
  12.79; Bank deals, “157 deals; -25.93”, 77.52 and 51.59; Cross-border, “186 deals; -19.14”, 62.18 and 43.04.
- Table “Contribution of the press release \[95% interval\](2)” (Read from the press release by an LLM, Change in days):
  Contract terms (approvals, structure, and votes) -1.55 [-2.55 to -0.60]; Parties’ stated closing date -5.80 [-7.33 to
  -4.23].
- Table “Target return from the day-1 close to the outcome, in %(3)” (Outcome, Deals, Mean, Median, 10th pct., 90th
  pct.): Adverse break 184, -27.94, -25.88, -73.01, +11.51; Topped by a higher bid 88, +19.37, +14.88, +0.37, +37.76.
- Table “Loss forecast error, in % points \[95% interval\](4)” (Forecast, Error, Vs. average): Historical average 28.84,
  Baseline; Fallback assumption 30.09, +1.25 [-1.78 to +4.40]; Regularized regression 28.77, -0.07 [-0.84 to +0.72];
  Boosted trees 28.09, -0.75 [-1.99 to +0.51].
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) 1,026 completed deals, 2019 to 2025; segments overlap; days saved below each; (2) Error change from
  adding each input, 923 deals with text; (3) 2001 to 2025, incl. dividends; (4) Mean absolute error.
- Numbers: segments `results/deck_numbers.json` (`time_to_closing_test_segments`; days saved computed); press release
  contribution `results/duration.json` (`split_text`, n = 923; `split_text.pairs`: `lgbm_llm minus lgbm_extm`,
  `lgbm_guide minus lgbm_llm`); returns `results/loss_given_break.json` (`realised`, all years); loss forecasts
  `results/loss_given_break.json` (`adverse.ret.loyo`: `mean`, `fallback`, `ridge`, `gbm`).

## Page 16. A5

- Kicker: DAILY PORTFOLIO AND ATTRIBUTION (marker A5)
- Headline: As a daily portfolio, ranking by the model earned more than ranking by raw spread at a similar Sharpe
  ratio, but the edge is not confirmed
- Exhibit “Value of $100 invested in January 2013(1)”, monthly to November 2026, axis labels 2013, 2015, 2017, 2019,
  2021, 2023, 2025: $352.40 Top half by the model; $315.42 Top half by raw spread; $260.20 All open deals.
- Table “Daily portfolio with a 10% position cap(2)” (Portfolio, Annual return, Sharpe, Drawdown): Top half by the model
  9.57%, 0.80, -26.68%; Top half by raw spread 8.70%, 0.77, -22.83%; All open deals 7.19%, 0.72, -20.20%.
- Table “Top minus bottom third by estimates used \[95% interval\](3)” (Ranking, Points a year): All three estimates
  +10.01 [6.51 to 13.95]; Without the break-risk estimate +9.45 [5.73 to 13.90]; Without the timing estimate +8.15
  [4.48 to 12.35].
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, and FRED; own
  analysis.
- Note(s): (1) Equal weights, rebalanced daily, before costs; (2) January 2013 to November 2026; drawdown = largest fall
  from a previous high; (3) Deals 2013 to 2025; a left-out estimate is held at its average.
- Speaker notes:
  - Q&A only: 2019 to 2025 top third 15.13% a year, +3.14 points vs. annualized spread; detail in the repository
- Numbers: portfolio values `results/deck_numbers.json` (`calendar_time_portfolios_pooled.month_end_value`, last month
  2026-11); table `results/expected_return.json` (`views.pooled.calendar`); attribution `results/expected_return.json`
  (`views.pooled.rankings`: `ear`, `ear_no_break`, `ear_no_duration`, `top_minus_bottom`); Q&A
  `results/expected_return.json` (`views.test.rankings.ear.terciles.top`, `views.test.model_top_minus_benchmark_top.spread_ann`).

## Other files

- Demo with the three estimates per deal: `streamlit run app/demo.py`. The committed screenshots show the layout on the
  synthetic fallback (made-up deals and values): `figures/demo_synthetic_*.png`.
- Index of all result files: `results/README.md`. Numbers for charts: `results/deck_numbers.json`.

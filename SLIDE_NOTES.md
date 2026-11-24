# Slide notes

- Deck: "Underwriting announced M&A deals: break risk, time to closing and loss if a deal breaks". 9 main slides, a
  divider (slide 10) and 5 appendix slides (A1 to A5, slides 11 to 15). The deck file is not in this repository.
- Audience: the portfolio manager of a merger arbitrage fund.
- "Test" = deals announced 2019 to 2025, models trained on 2001 to 2018.
- "Pooled" = deals announced 2013 to 2025, each year scored by models refitted on earlier years only.
- Brackets are 95% bootstrap intervals. Returns are before costs.
- Each number names the result file it comes from; "computed" marks arithmetic on numbers shown on the same slide.

## Storyline

- Every position rests on three estimates made on the day after the announcement: the probability of an adverse
  break, the time to closing and the loss if the deal breaks.
- A position earns about 1.4% in 95 of 100 deals and loses about 21 times that in the 3 that break.
- The model improves timing and ranking, while the spread stays the better break signal.
- Recommendation, word for word on the cover, slide 2 and the slide 9 headline: use the model for timing and ranking,
  the spread for break risk and a 31% loss for sizing.
- Basis for sizing: 31% is the mean loss of the 38 adverse breaks among the 1,205 ranked all-cash deals (31.21%,
  `results/common_sample_outcomes.md`); 27.94% is the mean over all 184 adverse breaks of any consideration (slide 7,
  `results/loss_given_break.json`).

## Video path

- Slide 1, 0:05: the three estimates and the recommendation.
- Slide 3, 0:20: what a position pays, and the three estimates with their benchmarks.
- Slide 6, 0:20: time to closing, 47.10 against 63.52 days.
- Slide 7, 0:10: loss if a deal breaks, 27.94% against 10.67%.
- Slide 8, 0:25: ranking, 12.26% a year in the top third, 2.48 points over raw spread.
- Slide 9, 0:10, plus 0:20 for the demo: recommendation, then `streamlit run app/demo.py`.
- Total 1:50, within the two-minute limit. Slides 2, 4 and 5 and the appendix are skipped in the video.
- The speaker notes give only the video time; slides 2, 4 and 5 say "Video: not shown".

## Slide 1. Title

- Kicker: 15.071 THE AI EDGE | PROJECT 1: PREDICTIVE AI.
- Headline: Underwriting announced M&A deals: break risk, time to closing and loss if a deal breaks.
- Subline: Recommendation: use the model for timing and ranking, the spread for break risk and a 31% loss for sizing.
- Speaker notes:
  - Video: 0:05.
  - Topic: a briefing for the portfolio manager of a merger arbitrage fund.
  - Point: every position rests on three estimates made on the day after the announcement.
  - Recommendation: use the model for timing and ranking, the spread for break risk and a 31% loss for sizing (same
    words on slides 2 and 9).
  - Transition: the summary first, then the evidence for each estimate.

## Slide 2. Summary

- Kicker: SUMMARY.
- Headline: The model improves timing and ranking, while the spread stays the better break signal.
- Exhibits: four bar pairs, "This analysis" (dark) against "Benchmark" (gray), with a verdict band under each (dark
  when the analysis is ahead, neutral light blue for the loss, which is measured, not modeled); "lower is better" on
  the timing chart; the loss chart is titled "Realized loss if a deal breaks (%)"; a recommendation box with the same
  words as the cover and the slide 9 headline ("Use the model for timing and ranking, the spread for break risk and a
  31% loss for sizing"); an agenda: Decision (3), Data (4), Break risk (5), Timing (6), Loss (7), Impact (8),
  Recommendation (9), Appendix (10).
- Numbers:
  - Break risk, AUC: 0.60 against 0.72 for the day-1 market spread; "Benchmark wins by 0.12". Source:
    `results/market_benchmark.md`, pooled, priced all-cash deals: model on the same deals 0.599, market 0.721, model
    minus market -0.121.
  - Timing error: 47.10 against 63.52 days for the deal-type median; "Model wins by 16.42 days". Source:
    `results/duration.json` (`product.split` mean absolute error 47.10; `split_full` median by structure 63.52);
    16.42 computed.
  - Realized loss if a deal breaks: 27.94% against 10.67% for a fall back to the pre-announcement price; "2.62x the
    benchmark". Source: `results/deck_numbers.json` (`loss_if_break`), `results/loss_given_break.json`; 2.62 computed
    (27.94 / 10.67).
  - Top-third return: 12.26% by the model against 9.78% by raw spread; "Model wins by 2.48 points". Source:
    `results/deck_numbers.json` (`ranking_test.pooled`), `results/expected_return.json`.
- Note(s): the 2.48 points have a 95% interval of 0.95 to 4.14 (`results/expected_return.json`,
  `model_top_minus_benchmark_top.spread`); as a daily portfolio the edge is +0.87 points a year [-0.88, +2.60]
  (`results/expected_return.json`, `calendar.model_minus_spread`); benchmarks from left to right are the day-1
  market spread, the deal-type median, the fall back to the pre-announcement price and the top third by raw spread;
  the realized loss is the mean of all 184 adverse breaks, the ranked all-cash deals lost 31.21% (slide 3,
  `results/common_sample_outcomes.md`); AUC bars start at 0.50; returns are before costs.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.
- Speaker notes:
  - Video: not shown.
  - Topic: the answer first, four comparisons of this analysis against the benchmark a manager uses today.
  - Orient: dark bars are this analysis, gray bars are the benchmark; the band under each chart gives the verdict,
    dark when the analysis is ahead, light blue for the loss, which is a measured value and no model.
  - Point: the market spread ranks break risk better than the model (0.72 against 0.60), so the spread stays the
    break signal.
  - Point: the model wins on timing by 16.42 days (lower is better), and the realized loss is 2.62x the usual
    assumption; sizing uses the 31.21% of the ranked all-cash deals.
  - Point: ranking by expected annualized return earned 2.48 points more than ranking by raw spread.
  - Confidence: the 2.48 points hold deal by deal; as a daily portfolio the edge is +0.87 points and not confirmed.
  - Transition: why these estimates decide the return of a position.

## Slide 3. Decision

- Kicker: DECISION (pointer: break history in appendix A3).
- Headline: A position earns about 1.4% in 95 of 100 deals and loses about 21 times that in the 3 that break.
- Exhibit 1: "Median return to the fund by outcome, in % of the day-1 price".
  - Deal closes: +1.39%, 95.44% of deals.
  - Adverse break: -28.91%, 3.15% of deals.
  - Topped by a higher bid: +14.20%, 1.41% of deals.
  - Label "about 21x": 28.91 / 1.39 = 20.80.
  - Source for all: `results/common_sample_outcomes.md` and `.json`, common sample of 1,205 priced all-cash deals
    announced 2013 to 2025 (1,150 closed, 38 adverse breaks, 17 topped); the ratio 20.80 is computed from the two
    medians shown.
- Exhibit 2: "Three estimates and their benchmarks".
  - Probability of an adverse break: day-1 market spread.
  - Time to closing: deal-type median and the parties' stated guidance.
  - Loss if the deal breaks: fall back to the pre-announcement price.
- Formula box: expected annualized return = (P(close) x spread + P(topped) x (spread + bump) + P(break) x loss) /
  years to resolution (`src/expected_return.py`).
- Note(s): the mean loss of the 38 breaks is 31.21% (`results/common_sample_outcomes.md`); slide 7 uses the 27.94%
  mean of 184 adverse breaks of any consideration announced 2001 to 2025 (`results/loss_given_break.json`).
- Source(s): S&P Capital IQ, CRSP; own analysis.
- Speaker notes:
  - Video: 0:20.
  - Topic: what a merger arbitrage position pays.
  - Orient: each bar is one outcome of an announced all-cash deal; the height is the return to the fund, the label
    below is how often it happens.
  - Point: in 95 of 100 deals the fund earns 1.39% at the median; in the 3 that break it loses 28.91% at the median,
    about 21 times as much.
  - Basis: all three bars are medians over the same 1,205 all-cash deals announced 2013 to 2025; the mean loss of
    these breaks is 31.21%.
  - Point: so three estimates decide a position, and the table shows the benchmark each one is tested against.
  - Transition: the data and the test design behind the three estimates.

## Slide 4. Data

- Kicker: DATA (detail: appendix A1).
- Headline: The sample holds 4,468 signed US public deals from 2001 to 2025, and every model is tested on later years
  than it was trained on.
- Exhibit 1: "Sample construction: number of deals". Source: `results/sample_filters.json` (`filter_log`) and
  `results/deck_numbers.json` (`core_sample`).
  - US public targets above $100m, closed or cancelled: 7,971.
  - Definitive agreement signed: 7,294.
  - Majority stake, announced 2001 to 2025: 5,156 (three filters in the log: percent sought of at least 50, target
    not among the buyers, announcement year).
  - Excluding SPACs: 4,744 (the log drops SPACs to 4,745 and one duplicate transaction to 4,744).
  - Excluding reverse mergers (core sample): 4,468.
- Exhibit 2: "Evaluation design by announcement year". Main split: training 2001 to 2018, test 2019 to 2025. Pooled
  set: first training window from 2001, 13 test years 2013 to 2025, refitted every year.
- Models box: gradient-boosted trees (LightGBM); an LLM reads each announcement filing into 29 contract terms.
- Note(s): models are tuned by cross-validation on training years and compared with simple rules, linear models, the
  market spread and the parties' own guidance; the target is the 198 adverse breaks in the core sample
  (`results/deck_numbers.json`, `core_sample.total`); Capital IQ and CRSP data stay out of the public repository,
  which runs on a synthetic sample.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP via WRDS; own analysis.
- Speaker notes:
  - Video: not shown.
  - Topic: which deals are in the sample and how the models are tested.
  - Orient, left: each bar is one filter; the dark bar is the core sample of 4,468 deals.
  - Orient, right: each row is a time axis from 2001 to 2025; gray years train the model, dark years test it.
  - Point: in the main split the model trains on 2001 to 2018 and is tested on 2019 to 2025; in the pooled set each
    year from 2013 to 2025 is scored by a model refitted on earlier years.
  - Point: models are boosted trees, and an LLM turns each announcement filing into 29 contract terms.
  - Transition: the first estimate, the probability of a break.

## Slide 5. Break risk

- Kicker: BREAK RISK (detail: appendix A2 and A3).
- Headline: On break risk, the model does not beat the day-1 spread, with an AUC of 0.60 against 0.72 on the same
  deals.
- Exhibit 1: "ROC AUC on the same deals". Break model 0.60 [0.48, 0.70]; day-1 market spread 0.72 [0.64, 0.80];
  difference -0.12 [-0.28, +0.00]. Source: `results/market_benchmark.md`, pooled, 1,205 priced all-cash deals with 38
  adverse breaks (0.599 [0.476, 0.702]; 0.721 [0.642, 0.800]; -0.121 [-0.282, +0.005]; the upper bound is +0.0046
  unrounded in `results/market_benchmark.json`).
- Exhibit 2: "Change in AUC from added information", all 2,176 core deals announced 2013 to 2025.
  - Buyer and target financials: +0.03 [-0.02, +0.08]. Source: `results/pooled_oot.md`, 3x minus 3,
    +0.025 [-0.025, +0.076]; the lower bound is -0.0248 unrounded in `results/pooled_oot.json`.
  - Announcement text and LLM-read terms: +0.02 [-0.01, +0.06]. Source: `results/pooled_oot.md`, 5f minus 3m,
    +0.022 [-0.014, +0.059].
  - Product overlap of buyer and target: +0.01 [-0.01, +0.02]. Source: `results/overlap.md`, pooled,
    +0.006 [-0.010, +0.019].
  - Market conditions: -0.03 [-0.05, -0.00]. Source: `results/pooled_oot.md`, 3m minus 3x, -0.028 [-0.054, -0.002].
  - Ten day-1 market signals, against the spread: -0.12 and -0.06. Source: `results/day1_signals.md`, the two
    pre-specified models minus the market-implied probability, -0.125 and -0.061; the first is -0.12498 unrounded in
    `results/day1_signals.json`.
- Note(s): 1,205 priced all-cash deals with 38 adverse breaks; whiskers and brackets are 95% bootstrap intervals; +0.00
  and -0.00 are bounds just above and just below zero; the spread is the break probability implied by the day-1
  close; the last row of exhibit 2 adds the signals to the market-implied probability in two pre-specified models.
- Excluded evidence box: an LLM risk score on anonymized press releases reached 0.82, but the LLM named 92% of the
  targets. Source: `results/llm_zero_shot.md` (0.822 on test years 2019 to 2025; 92% identified).
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.
- Speaker notes:
  - Video: not shown.
  - Topic: can a model rank break risk better than the market.
  - Orient: bars start at 0.50, which means no skill; dark is the model, gray is the probability implied by the day-1
    spread; whiskers are 95% intervals.
  - Point: 0.60 for the model against 0.72 for the spread on the same 1,205 deals; the gap of 0.12 has an interval
    from -0.28 to +0.00, so it ends just above zero.
  - Point: financials, text, product overlap and market signals did not close the gap.
  - Honest result: an LLM score looked better at 0.82, but the LLM recognized 92% of the deals, so it is excluded.
  - Transition: the second estimate is where the model helps.

## Slide 6. Timing

- Kicker: TIMING (detail: appendix A4).
- Headline: The closing forecast is off by 47.10 days on average, 16.42 days less than the deal-type median.
- Exhibit 1: "Mean error in days (lower is better)", 1,026 completed deals announced 2019 to 2025, models trained on
  2001 to 2018. Source: `results/duration.json` (`split_full.scores`, `product.split.scores`).
  - Benchmarks: median of all deals 72.62; median by deal type 63.52.
  - Model steps: deal terms 58.45; plus firm data and LLM terms 52.68.
  - Final model, plus stated closing date: 47.10.
  - Label: -16.42 days (computed, 63.52 minus 47.10).
- Exhibit 2: "Against the parties' own guidance, 807 deals", the deals that state an expected closing. Source:
  `results/duration.json` (`guidance_split`).
  - Parties' stated expected closing: 46.43 days.
  - Model without the stated closing: 46.86 days.
  - Model that also reads the stated closing: 39.82 days.
  - Improvement on guidance: 6.61 days [3.86, 9.51].
- Box for the portfolio manager: annualize each spread with the forecast; deals closed a median of 14.50 days earlier
  than the stated guidance (`results/duration.json`, `guidance_split.bias`).
- Note(s): 1,026 completed deals announced 2019 to 2025, models trained on 2001 to 2018; whiskers and brackets are 95%
  bootstrap intervals; 807 of these deals state an expected closing; guidance is the midpoint of the stated window.
- Source(s): S&P Capital IQ, SEC EDGAR; own analysis.
- Speaker notes:
  - Video: 0:20.
  - Topic: how long a deal takes to close, which sets the annualized return.
  - Orient: each bar is the average miss in days, so lower is better; gray bars are benchmarks, light blue bars are
    model steps, the dark bar is the final model.
  - Point: the final model misses by 47.10 days against 63.52 for the deal-type median, 16.42 days less.
  - Point: it also beats the parties' own guidance by 6.61 days, with an interval from 3.86 to 9.51.
  - Why it works: the stated closing date and the contract terms read by the LLM carry most of the gain.
  - Transition: the third estimate, the loss if the deal breaks.

## Slide 7. Loss

- Kicker: LOSS (detail: appendix A4).
- Headline: Adverse breaks lost 27.94% on average, 2.62x the 10.67% implied by a fall back to the pre-announcement
  price.
- Exhibit 1: "Average loss in an adverse break". Fall back to the pre-announcement price 10.67%; realized average
  27.94%; label 2.62x (computed). Source: `results/deck_numbers.json` (`loss_if_break`, 184 adverse breaks
  announced 2001 to 2025), `results/loss_given_break.json` (`realised`).
- Exhibit 2: "Break-even break probability by day-1 spread", spreads of 1% to 6%, computed as spread / (spread +
  loss), which ignores higher bids and the time value of money. At a 3.00% spread: 3.00 / (3.00 + 10.67) = 21.95% and
  3.00 / (3.00 + 27.94) = 9.70%.
- Loss forecasts box: three models did not beat the historical average, so sizing uses the mean loss of the ranked
  all-cash deals, 31.21% (`results/loss_given_break.md`, detail on A4; `results/common_sample_outcomes.md`).
- Source(s): S&P Capital IQ, CRSP; own analysis.
- Speaker notes:
  - Video: 0:10.
  - Topic: how much the fund loses when a deal breaks.
  - Orient, left: the gray bar is the usual assumption that the price falls back to its level before the
    announcement; the dark bar is what adverse breaks lost.
  - Point: 27.94% against 10.67%, a factor of 2.62.
  - Orient, right: the break probability at which a position just breaks even, for each spread; the dashed line marks
    a 3.00% spread.
  - Point: at a 3.00% spread the position needs a break probability below 9.70%, where the usual assumption suggests
    21.95%.
  - Transition: what the three estimates are worth together.

## Slide 8. Impact

- Kicker: IMPACT (detail: appendix A5).
- Headline: Ranked by expected annualized return, the top third earned 12.26% a year, 2.48 points more than the top
  third by raw spread.
- Exhibit 1: "Annual return by third", 1,205 all-cash deals announced 2013 to 2025, thirds within each announcement
  year, equal weights. Source: `results/deck_numbers.json` (`ranking_test.pooled`), `results/expected_return.json`.
  - Ranked by the model: top 12.26%, middle 3.07%, bottom 2.25%.
  - Ranked by raw spread: top 9.78%, middle 4.06%, bottom 1.42%.
  - Label: +2.48 points.
- Exhibit 2: "Top third by the model, in $m per $100m a year". Source: `results/expected_return.json`, pooled.
  - Against all deals: +5.90 [3.92, 7.97] (`rankings.ear.top_minus_all`).
  - Against the top third by raw spread: +2.48 [0.95, 4.14] (`model_top_minus_benchmark_top.spread`).
  - Against the top third by annualized spread: +1.67 [0.20, 3.25] (`model_top_minus_benchmark_top.spread_ann`).
  - Holding every deal returned 6.35% a year (`rankings.ear.all`).
- Confidence box: as a daily portfolio the edge over the spread ranking is +0.87 points a year [-0.88, +2.60]
  (`calendar.model_minus_spread`); the top third holds 22 of the 38 adverse breaks
  (`rankings.ear.terciles.top.adverse`).
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.
- Speaker notes:
  - Video: 0:25.
  - Topic: what the three estimates are worth when they are combined into an expected annualized return.
  - Orient: deals are sorted into thirds each year; dark bars rank by the model, gray bars rank by the raw spread; the
    height is the realized annual return.
  - Point: the top third by the model earned 12.26% a year, 2.48 points more than the top third by raw spread; on
    $100 million that is $2.48 million a year.
  - Confidence: deal by deal the interval is 0.95 to 4.14; as a daily portfolio the edge is +0.87 points and its
    interval includes zero.
  - Caveat: the top third also holds 22 of the 38 adverse breaks, so part of its return is a risk premium.
  - Transition: what the manager should do with this.

## Slide 9. Recommendation

- Kicker: RECOMMENDATION.
- Headline: Use the model for timing and ranking, the spread for break risk and a 31% loss for sizing.
- Exhibit: four steps, read left to right.
  - 1 Timing: 47.10 days against 63.52; annualize the spread with the forecast (`results/duration.json`).
  - 2 Break risk: AUC 0.72 against 0.60; read break risk from the spread (`results/market_benchmark.md`).
  - 3 Sizing: 31.21%, ranked all-cash deals; size for a 31% loss (`results/common_sample_outcomes.md`, mean return
    of the 38 adverse breaks).
  - 4 Ranking: +2.48 points over raw spread [0.95, 4.14]; rank by expected annualized return
    (`results/expected_return.json`).
- Limitations box: 38 priced adverse breaks; returns before costs; daily-portfolio edge (+0.87 points) not confirmed.
- Next step box: Project 2, an agent that writes this memo for each new deal.
- Note(s): steps 1 and 2 against their benchmark (slides 5 and 6); step 3 is the mean loss of the 38 adverse breaks
  among the ranked all-cash deals (slide 3), against 27.94% over all deals (slide 7, `results/loss_given_break.json`);
  step 4 is the difference between the top thirds, 95% interval 0.95 to 4.14 (slide 8); equal weights, no capacity
  limit.
- Source(s): Own analysis; code and results at github.com/lorenzjbrodt/merger-deal-break-prediction.
- Speaker notes:
  - Video: 0:10, plus 0:20 for the demo.
  - Topic: how a new deal is underwritten with these findings.
  - Orient: read left to right; the three light boxes are the estimates with their evidence, the dark box is the
    decision.
  - Point: take timing from the model (47.10 against 63.52 days), break risk from the spread (0.72 against 0.60) and
    size for a 31% loss, the mean of the ranked all-cash deals (27.94% over all deals).
  - Point: then rank deals by expected annualized return, which earned 2.48 points more than ranking by raw spread.
  - Biggest limitation: only 38 priced adverse breaks, and the edge is not confirmed as a daily portfolio.
  - Close: restate the recommendation; Project 2 turns this flow into a memo for each new deal.

## Slide 10. Appendix divider

- A1 Data sources, corrections and cancellation reasons (slide 11); A2 Break model in the test period (12); A3 Model
  drivers and stability over time (13); A4 Time to closing and loss in detail (14); A5 Daily portfolio and
  attribution (15). Of the appendix slides only A5 has a speaker note (Q&A only).

## A1. Data sources, corrections and cancellation reasons

- Kicker: APPENDIX: DATA SOURCES, CORRECTIONS AND CANCELLATION REASONS.
- Headline: Four data sources feed the models, and 100 of the 346 cancellations are higher bids that count as gains.
- Numbers: press release for 88% of deals (`results/text_coverage.md`, all deals); daily share prices for 93% of
  targets (`results/crsp_link.md`, 4,140 of 4,468 core deals linked); 29 contract terms (the 26 fields kept in
  `results/llm_extraction_validation.md`, with buyer type coded as four indicators and structure as one); 247 offers
  replaced by the announcement-day offer (`results/crsp_prices.md`); 346 cancellations by reason: higher competing bid
  100, mutual termination or other 77, regulatory or antitrust block 57, no reason stated 52, shareholder vote failed
  31, buyer walked or MAC dispute 21, financing failure 8 (`results/outcome_taxonomy.md`); of the 246 other
  cancellations, 198 are in the core sample (`results/deck_numbers.json`, `core_sample`); reasons checked on a sample
  of 50 (`results/outcome_taxonomy.md`).
- Note(s): reasons classified by an LLM from the vendor's deal notes and checked on a sample of 50.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP and Compustat via WRDS, Hoberg-Phillips data library, FRED; own
  analysis.

## A2. Break model in the test period

- Kicker: APPENDIX: BREAK MODEL IN THE TEST PERIOD.
- Headline: In the test period the break model reaches an AUC of 0.63, against 0.53 for a single rule and 0.64 for the
  day-1 spread on priced deals.
- Numbers: one rule 0.53 [0.45, 0.61], logistic model 0.61 [0.52, 0.69], break model 0.63 [0.55, 0.72] on 1,083 test
  deals with 40 adverse breaks (`results/benchmark_table.md`); AUC by segment, cash 0.65 (686 deals, 24 adverse
  breaks), stock 0.58 (210, 11), mixed 0.62 (187, 5), small 0.56 (252, 10), mid 0.70 (297, 9), large 0.64 (534, 21),
  size cut-offs $350m and $1,554m (`results/segments.md`, column AUC 3x); priced test deals, break model
  0.63 [0.52, 0.73] against the day-1 spread 0.64 [0.52, 0.76] on 623 deals with 22 adverse breaks
  (`results/market_benchmark.md`).
- Note(s): 1,083 deals with 40 adverse breaks, models trained on 2001 to 2018; whiskers and brackets are 95% bootstrap
  intervals; size thirds by deal value, with cut-offs at $350m and $1,554m; 623 priced all-cash deals with 22 adverse
  breaks; slide 5 pools 2013 to 2025 with yearly refits (0.60 against 0.72), so the spread leads by less in the test
  years.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.

## A3. Model drivers and stability over time

- Kicker: APPENDIX: MODEL DRIVERS AND STABILITY OVER TIME.
- Headline: The break model leans on financial strength and deal structure, and its AUC ranges from 0.41 to 0.92
  across years.
- Numbers: mean absolute SHAP of the top eight drivers, 0.13 target revenue, 0.13 reverse termination fee, 0.09
  price-to-book, 0.09 cash share, 0.08 target price against its 52-week high, 0.08 target termination fee, 0.07
  acquirer price against its 52-week high, 0.07 target cash to assets (`results/shap.md`); AUC by year 2013 to 2025,
  0.55, 0.77, 0.61, 0.77, 0.55, 0.55, 0.50, 0.61, 0.47, 0.84, 0.92, 0.68, 0.41, with 1 to 12 adverse breaks a year,
  pooled 0.61 [0.54, 0.69] (`results/pooled_oot.md`); adverse break rate by year, 3.38, 3.57, 5.80, 1.54, 4.79, 4.81,
  0.62, 8.74, 3.63, 6.08, 2.70, 3.45, 2.70 (computed from the deals and adverse breaks per year in
  `results/pooled_oot.md`, 2,176 core deals).
- Note(s): SHAP on deals announced 2019 to 2025, associations used by the model and not causal effects; each year is
  scored by a model fitted on earlier years, pooled AUC 0.61 [0.54, 0.69]; core sample, 2,176 deals announced 2013 to
  2025.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.

## A4. Time to closing and loss in detail

- Kicker: APPENDIX: TIME TO CLOSING AND LOSS IN DETAIL.
- Headline: The closing forecast improves on the deal-type median in every segment, while loss forecasts do not beat
  the historical average.
- Numbers: error by segment, deal-type median against model (product model), with the reduction computed: mergers 855
  deals, 73.14 against 53.96, 19.18; tender offers 171, 15.40 against 12.79, 2.61; bank deals 157, 77.52 against
  51.59, 25.93; cross-border 186, 62.18 against 43.04, 19.14 (`results/deck_numbers.json`,
  `time_to_closing_test_segments`); LLM-read contract terms -1.55 days [-2.55, -0.60], stated expected closing date
  -5.80 [-7.33, -4.23] (`results/duration.json`, `split_text.pairs`); calibrated 80% interval contains 80.31% of the
  1,026 test deals, median width 113.58 days (`results/duration.json`, `conformal`); target return to the outcome,
  adverse breaks (184) mean -27.94%, median -25.88%, 10th percentile -73.01%, 90th percentile +11.51%, topped deals
  (88) mean +19.37%, median +14.88%, 10th percentile +0.37%, 90th percentile +37.76%
  (`results/loss_given_break.json`, `realised`); loss forecasts, mean absolute error in points, historical average
  28.84, fall back 30.09 (+1.25 [-1.78, +4.40]), regularized regression 28.77 (-0.07 [-0.84, +0.72]), boosted trees
  28.09 (-0.75 [-1.99, +0.51]) (`results/loss_given_break.json`, `adverse.ret.loyo`).
- Note(s): 1,026 completed deals announced 2019 to 2025, the calibrated 80% interval contains 80.31% of them with a
  median width of 113.58 days; change in mean absolute error, 95% bootstrap interval; deals announced 2001 to 2025
  with prices, total return with dividends to the day after the outcome; mean absolute error in percentage points,
  95% intervals, each year forecast from all other years.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.

## A5. Daily portfolio and attribution

- Kicker: APPENDIX: DAILY PORTFOLIO AND ATTRIBUTION.
- Headline: Run as a daily portfolio, ranking by the model returned 9.57% a year against 8.70% by raw spread, an edge
  that is not statistically confirmed.
- Numbers: value of $100 invested in January 2013, at December 2025, $378.52 top half by the model, $332.70 top half
  by raw spread, $267.57 all open deals (`results/deck_numbers.json`, `calendar_time_portfolios_pooled`); daily
  portfolio with a 10% position cap, January 2013 to November 2026: return 9.57%, Sharpe 0.80, drawdown -26.68% (top
  half by the model), 8.70%, 0.77, -22.83% (top half by raw spread), 7.19%, 0.72, -20.20% (all open deals); model minus
  spread +0.87 points a year [-0.88, +2.60] (`results/expected_return.json`, `calendar`); attribution, top minus
  bottom third, all three estimates +10.01 [6.51, 13.95], without the break-risk estimate +9.45 [5.73, 13.90],
  without the timing estimate +8.15 [4.48, 12.35] (`results/expected_return.json`, `rankings`).
- Note(s): equal weights, rebalanced daily, before costs; the chart ends in December 2025, when few deals remain open;
  January 2013 to November 2026, drawdown is the largest fall from a previous high, the edge over the spread ranking
  is +0.87 points a year [-0.88, +2.60]; deals announced 2013 to 2025, 95% bootstrap intervals; a left-out estimate
  is held at its average, and leaving out timing costs the most.
- Source(s): S&P Capital IQ, SEC EDGAR, CRSP; own analysis.
- Speaker notes:
  - Q&A only: in the test period 2019 to 2025 the top third by the model earned 15.13% a year, 4.05 points
    [1.55, 7.08] more than the top third by raw spread (`results/expected_return.json`, `views.test`).

## Other files

- Demo with the three numbers per deal: `streamlit run app/demo.py`. The committed screenshots show the layout on
  the synthetic fallback (made-up deals and values): `figures/demo_synthetic_*.png`.
- Index of all result files: `results/README.md`. Numbers for charts: `results/deck_numbers.json`.

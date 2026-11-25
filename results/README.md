# Results: where to find what

All tables are aggregates from the real data. "Test" = deals announced 2019-2025 (models trained on 2001-2018);
"pooled" = deals announced 2013-2025, each year scored by models refitted on earlier years only.

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
- **Break risk, more data does not reliably help** (pooled AUC change, each row vs. the model without it, page 7): buyer
  and target financials +0.025 [-0.025 to +0.076]; press release text and LLM-read terms +0.022 [-0.014 to +0.059];
  product overlap +0.006 [-0.010 to +0.019]; market conditions -0.028 [-0.054 to -0.002]; ten day-1 market signals added
  to the spread -0.125 [-0.246 to -0.010]. The same LLM-read terms cut the closing date error by 1.55 days [-2.55 to
  -0.60].
- **Loss**: no loss forecast reliably beat the historical average (page 15).

## The three numbers and the dollar result

| File | Content |
|---|---|
| [expected_return.md](expected_return.md) | Expected annualized return per deal; tercile test; calendar-time portfolio; attribution; old versus new time-to-closing model |
| [duration.md](duration.md) | Time to closing: ladder, management-guidance benchmark, product model, calibrated 80% interval, segments |
| [closing_guidance_validation.md](closing_guidance_validation.md) | How the stated expected closing was read from press releases and checked |
| [loss_given_break.md](loss_given_break.md) | Loss if a deal breaks and gain if it is topped; forecasts |
| [benchmark_table.md](benchmark_table.md) | Break model ladder, test period, four target and sample combinations, day-1 step |
| [pooled_oot.md](pooled_oot.md) | Break model ladder, pooled out-of-time 2013-2025 |
| [market_benchmark.md](market_benchmark.md) | Break model against the market's day-1 spread |
| [day1_signals.md](day1_signals.md) | Ten day-1 market signals beyond the spread (pre-specified, evaluated once) |
| [overlap.md](overlap.md) | Antitrust overlap features (Hoberg-Phillips similarity, industry shares) |
| [backtest.md](backtest.md) | Skip rules and the model-versus-market rule, daily marks |
| [deck_numbers.json](deck_numbers.json) | Aggregates for the slide deck: monthly portfolio values, returns per third with intervals, sample counts, time-to-closing error by segment |
| [common_sample_outcomes.md](common_sample_outcomes.md), [.json](common_sample_outcomes.json) | Outcome shares, day-1 spread, realised returns and loss-to-spread ratios on one common sample (the 1,205 ranking-test deals), with two comparison samples |

## Data, labels and checks

| File | Content |
|---|---|
| [outcome_taxonomy.md](outcome_taxonomy.md) | Why deals broke; adverse breaks versus higher bids |
| [crsp_link.md](crsp_link.md), [crsp_prices.md](crsp_prices.md) | Linking deals to CRSP; price checks; which offer is known on day 1 |
| [text_coverage.md](text_coverage.md), [llm_extraction_validation.md](llm_extraction_validation.md) | EDGAR text coverage; validation of LLM-extracted terms |
| [label_audit.md](label_audit.md) | Capital IQ deal status checked against SEC filings |
| [price_coverage.md](price_coverage.md), [monthly/](monthly) | Capital IQ monthly closes: the fallback analysis without WRDS |

## Diagnostics of the break model

[ablation.md](ablation.md), [calibration.md](calibration.md), [shap.md](shap.md), [segments.md](segments.md),
[three_class.md](three_class.md), [robustness.md](robustness.md), [rolling_origin.md](rolling_origin.md),
[case_studies.md](case_studies.md), [impact.md](impact.md) (stylized dollar grid),
[llm_zero_shot.md](llm_zero_shot.md) (exploratory, not trustworthy), [base_run/](base_run) (first version of the project).

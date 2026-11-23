# Results: where to find what

All tables are aggregates from the real data. "Test" = deals announced 2019-2025 (models trained on 2001-2018);
"pooled" = deals announced 2013-2025, each year scored by models refitted on earlier years only.

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

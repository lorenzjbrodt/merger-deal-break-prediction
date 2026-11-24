# Outcomes, day-1 spread and loss on one common sample

Read-only export (`src/common_sample_outcomes.py`); no model was refitted and no other result file changed. Aggregates only. Returns are per deal, not annualized, unhedged and before costs.

**Why.** Figures quoted elsewhere come from different samples, for example break rates from the core sample, the spread from usable all-cash deals and the 28% loss from 184 adverse breaks of any consideration. Here outcome shares, spread and loss are computed on the same deals.

## Definitions

- **Sample filter.** Core sample (signed deals without reverse mergers), all-cash consideration, usable CRSP price (deal still open at the day-1 close, pre-announcement price within 15% of the price implied by the vendor's offer and premium, day-1 spread between -50% and +100%, an exit exists), and a defined market-implied break probability (offer above the pre-announcement close). Identical to the `priced` filter of src/expected_return.py; the three samples differ only in the announcement years.
- **Outcomes.** Closed = completed. Adverse break = cancelled for a reason other than a higher competing bid. Topped = cancelled because a higher competing bid won.
- **Day-1 spread.** Offer known on day 1 / close on t1 - 1, where t1 is the first trading day after the announcement date (results/crsp_prices.md). The offer is the per-share price in the announcement press release where it differs from the vendor's final offer.
- **Return.** Total return of the target share (dividends included) from the close on t1 to the exit, the field behind loss_if_break in results/deck_numbers.json. Closed deals: to the closing (CRSP delisting return; the offer paid on the closing date where CRSP has no delisting record). Adverse breaks and topped deals: to the close of the first trading day after the cancellation date. Not annualized, unhedged, before costs.
- **Loss to spread.** Loss = return of adverse breaks with the sign reversed; spread = day-1 spread of closed deals; both from the same sample. The ratio is computed from unrounded values; 'calculation' shows it with the two-decimal inputs.

## Common sample: priced all-cash deals announced 2013-2025 (the ranking-test sample)

1,205 deals.

| | Deals | Share of deals | Mean | Median |
|---|---:|---:|---:|---:|
| Outcome: closed | 1,150 | 95.44% | | |
| Outcome: adverse break | 38 | 3.15% | | |
| Outcome: topped by a higher bid | 17 | 1.41% | | |
| Day-1 spread, closed deals | 1,150 | | +2.51% | +1.20% |
| Return to closing, closed deals | 1,150 | | +3.19% | +1.39% |
| Return, adverse breaks | 38 | | -31.21% | -28.91% |
| Return, topped deals | 17 | | +14.61% | +14.20% |

| Loss to spread | Calculation (inputs in %) | Ratio |
|---|---|---:|
| mean loss / median spread | 31.21 / 1.20 = 26.01 (26.08 from unrounded inputs) | 26.08 |
| mean loss / mean spread | 31.21 / 2.51 = 12.43 (12.44 from unrounded inputs) | 12.44 |
| median loss / median spread | 28.91 / 1.20 = 24.09 (24.15 from unrounded inputs) | 24.15 |

## Comparison: priced all-cash deals announced 2019-2025

623 deals.

| | Deals | Share of deals | Mean | Median |
|---|---:|---:|---:|---:|
| Outcome: closed | 595 | 95.51% | | |
| Outcome: adverse break | 22 | 3.53% | | |
| Outcome: topped by a higher bid | 6 | 0.96% | | |
| Day-1 spread, closed deals | 595 | | +3.34% | +1.72% |
| Return to closing, closed deals | 595 | | +3.99% | +1.81% |
| Return, adverse breaks | 22 | | -37.89% | -44.87% |
| Return, topped deals | 6 | | +16.20% | +15.73% |

| Loss to spread | Calculation (inputs in %) | Ratio |
|---|---|---:|
| mean loss / median spread | 37.89 / 1.72 = 22.03 (21.96 from unrounded inputs) | 21.96 |
| mean loss / mean spread | 37.89 / 3.34 = 11.34 (11.33 from unrounded inputs) | 11.33 |
| median loss / median spread | 44.87 / 1.72 = 26.09 (26.02 from unrounded inputs) | 26.02 |

## Comparison: priced all-cash deals announced 2001-2025

2,461 deals.

| | Deals | Share of deals | Mean | Median |
|---|---:|---:|---:|---:|
| Outcome: closed | 2,329 | 94.64% | | |
| Outcome: adverse break | 81 | 3.29% | | |
| Outcome: topped by a higher bid | 51 | 2.07% | | |
| Day-1 spread, closed deals | 2,329 | | +2.43% | +1.41% |
| Return to closing, closed deals | 2,329 | | +3.25% | +1.65% |
| Return, adverse breaks | 81 | | -29.38% | -28.83% |
| Return, topped deals | 51 | | +15.41% | +14.20% |

| Loss to spread | Calculation (inputs in %) | Ratio |
|---|---|---:|
| mean loss / median spread | 29.38 / 1.41 = 20.84 (20.87 from unrounded inputs) | 20.87 |
| mean loss / mean spread | 29.38 / 2.43 = 12.09 (12.11 from unrounded inputs) | 12.11 |
| median loss / median spread | 28.83 / 1.41 = 20.45 (20.48 from unrounded inputs) | 20.48 |

## Side by side

| | 2013-2025 | 2019-2025 | 2001-2025 |
|---|---:|---:|---:|
| Deals | 1,205 | 623 | 2,461 |
| Closed, share of deals | 95.44% (1,150) | 95.51% (595) | 94.64% (2,329) |
| Adverse break, share of deals | 3.15% (38) | 3.53% (22) | 3.29% (81) |
| Topped by a higher bid, share of deals | 1.41% (17) | 0.96% (6) | 2.07% (51) |
| Day-1 spread, closed deals: mean / median | +2.51% / +1.20% | +3.34% / +1.72% | +2.43% / +1.41% |
| Return to closing, closed deals: mean / median | +3.19% / +1.39% | +3.99% / +1.81% | +3.25% / +1.65% |
| Return, adverse breaks: mean / median | -31.21% / -28.91% | -37.89% / -44.87% | -29.38% / -28.83% |
| Return, topped deals: mean / median | +14.61% / +14.20% | +16.20% / +15.73% | +15.41% / +14.20% |
| Mean loss / median spread | 26.08 | 21.96 | 20.87 |
| Mean loss / mean spread | 12.44 | 11.33 | 12.11 |
| Median loss / median spread | 24.15 | 26.02 | 20.48 |

## Checks

- pooled_2013_2025: same rows as the ranking test in expected_return.py; deals, adverse breaks and topped deals equal results/expected_return.json (1205, 38, 17); mean return per deal (+2.27%) and mean day-1 spread (2.69%) of all deals equal the deal-weighted means of the thirds published there.
- test_2019_2025: same rows as the ranking test in expected_return.py; deals, adverse breaks and topped deals equal results/expected_return.json (623, 22, 6); mean return per deal (+2.62%) and mean day-1 spread (3.65%) of all deals equal the deal-weighted means of the thirds published there.
- spread and return columns reproduce all 6 rows of the day-1 spread table in results/crsp_prices.md (usable all-cash deals, which also include deals without a market-implied probability: deals, median and mean spread, mean return).
- all_2001_2025: same filter, no published count to compare with; it contains the two samples above as subsets.

## Reading

- The adverse breaks of the 2019-2025 sample are a subset of those of the 2013-2025 sample, and both are subsets of the 2001-2025 sample; the three columns are not independent.
- Means of the break returns rest on 22 to 81 deals and move by several points between samples; the ratios move with them.
- The return of closed deals is not the same number as their day-1 spread: it is the realised total return and includes dividends and any later price increase, while the spread uses the offer known on day 1.
- The 28% average loss quoted elsewhere is from 184 adverse breaks of any consideration under a different price filter (`results/loss_given_break.md`); it is not the loss of this sample.


# Price data: coverage and checks

Source: Capital IQ transaction screen with the display columns reduced to IDs plus the target's `Day Close Price` on a monthly grid (the 20th of each month). Three exports, parsed by `src/prices.py`. Licensed data: only aggregates appear here.

## Grid

- 312 monthly dates from 2000-12-20 to 2026-11-20; overlapping dates between files: 0; missing months: 0.
- 88 grid dates fall on a weekend. When the months before and after are priced, the weekend date is priced in 99.6% of cases, so a non-trading 20th returns the prior close.
- 4,739 of 4,744 sample deals have at least one price.

## Is the close consistent with the export's offer and premium?

For 4,304 deals both the last monthly close before announcement and the implied pre-announcement price (offer per share / (1 + premium 1 month prior)) exist. Median ratio close / implied: 1.006; within 15%: 3,755 (87%).

The grid close can be up to a month away from the 1-month-prior reference, so part of the mismatch is ordinary price movement (for example a run-up before the announcement). Using the premium horizon closest in time to the grid date (1 day, 1 week or 1 month) raises the share within 15% to 4,029 (94%). The price-based analyses use this second rule; the count under the literal 1-month rule is shown for transparency.

Split adjustment: of the 549 deals outside 15% under the 1-month rule, only 34 sit within 8% of a clean split factor (2:1 11, 1:2 7, 10:1 4, 5:1 4, others 8), and 61 of the 549 are cancelled deals, whose targets kept trading. The closes therefore look like unadjusted historical prices for the large majority of deals, with a small number of adjusted or mis-matched series. All deals outside the tolerance are excluded from price-based analyses, which removes those series whatever the cause.

## Coverage by outcome class and period

| Period | Outcome | Deals | Any price | t1 exists (grid date between announcement and outcome) | Price valid | All-cash | Usable all-cash (backtest) | Usable any consideration |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test 2019-2025 | adverse_break | 88 | 100% | 94% | 34% | 27% | 16 | 28 |
| test 2019-2025 | closed | 1249 | 100% | 98% | 71% | 53% | 559 | 869 |
| test 2019-2025 | topped | 21 | 100% | 95% | 67% | 29% | 5 | 14 |
| train 2001-2018 | adverse_break | 158 | 100% | 96% | 73% | 47% | 53 | 107 |
| train 2001-2018 | closed | 3149 | 100% | 98% | 92% | 61% | 1747 | 2869 |
| train 2001-2018 | topped | 79 | 100% | 92% | 90% | 63% | 42 | 66 |

Completed deals with delisted targets are covered: 4,290 of 4,393 closed deals with prices stop being priced within two months of the closing date, and their pre-closing closes are present. Deals with no grid date between announcement and outcome (closed or cancelled within weeks) cannot be entered: 38 deals.

## Cross-check of the exit price for completed cash deals

For 2,306 completed all-cash deals, offer per share versus the last available close before delisting: median difference +0.31%, 5th to 95th percentile -0.7% to +4.0%; 6.2% differ by more than 5% (revised offers, contingent consideration or stale marks).

## Spread at t1 and market-implied break probability (usable all-cash deals)

| Period | Outcome | Deals | Median spread | Mean spread | Median p_mkt | Mean p_mkt | Median days announcement to t1 |
|---|---|---:|---:|---:|---:|---:|---:|
| test 2019-2025 | adverse_break | 16 | 7.1% | 8.2% | 0.22 | 0.31 | 17 |
| test 2019-2025 | closed | 559 | 1.7% | 4.4% | 0.08 | 0.19 | 13 |
| test 2019-2025 | topped | 5 | 4.8% | 8.0% | 0.11 | 0.19 | 17 |
| train 2001-2018 | adverse_break | 53 | 2.9% | 3.5% | 0.26 | 0.38 | 17 |
| train 2001-2018 | closed | 1747 | 1.1% | 2.4% | 0.06 | 0.17 | 16 |
| train 2001-2018 | topped | 42 | 1.9% | 4.7% | 0.09 | 0.26 | 16 |

t1 can be up to a month after the announcement, so the market price already reflects a month of news that the announcement-date model does not see. This favours the market benchmark.


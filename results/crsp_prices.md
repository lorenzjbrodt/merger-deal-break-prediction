# Daily prices from CRSP: coverage, checks, day-1 spread

Source: WRDS. CRSP daily stock file (`crsp.dsf_v2`, current format, prices and total returns including the delisting return) for linked targets and acquirers; Compustat daily security prices (`comp.secd`) as the second source. Licensed data: this file contains aggregates only. CRSP ends on 2025-12-31.

Entry is at the close of t1, the first trading day after the announcement date. P0 is the close on the last trading day before the announcement. A deal is usable when it is still open at t1, the price before announcement agrees within 15% with the price implied by the export's offer and one-day premium, the entry spread lies between -50% and +100%, and an exit exists.

## Coverage by period and outcome (core sample)

| Period | Outcome | Deals | Linked to CRSP | Close at t1 and deal still open | Price agrees with the export | All-cash | Usable all-cash | Usable, any consideration |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2001-2012 | adverse_break | 114 | 89% | 97% | 90% | 57 | 47 | 89 |
| 2001-2012 | closed | 2117 | 93% | 96% | 95% | 1312 | 1238 | 1864 |
| 2001-2012 | topped | 61 | 89% | 95% | 90% | 39 | 36 | 50 |
| 2013-2018 | adverse_break | 44 | 98% | 100% | 98% | 18 | 17 | 35 |
| 2013-2018 | closed | 1031 | 92% | 97% | 96% | 602 | 584 | 925 |
| 2013-2018 | topped | 18 | 94% | 94% | 89% | 11 | 11 | 14 |
| 2019-2025 | adverse_break | 40 | 88% | 95% | 100% | 24 | 22 | 35 |
| 2019-2025 | closed | 1026 | 93% | 98% | 97% | 658 | 632 | 937 |
| 2019-2025 | topped | 17 | 94% | 100% | 100% | 6 | 6 | 16 |

Usable all-cash deals: test period 2019-2025: 660 deals, **22 adverse breaks**, 6 topped; pooled 2013-2025: 1272 deals, **39 adverse breaks**, 17 topped; training period 2001-2018: 1933 deals, **64 adverse breaks**, 47 topped.

For comparison, the Capital IQ monthly closes gave 580 usable all-cash test deals with 16 adverse breaks. Daily data helps mainly because a deal no longer needs a monthly grid date between announcement and outcome, and because entry is one day after the announcement.

## Price source and exit rule (usable core deals)

Entry price source: CRSP 3818, Compustat 147.

Exit rule: delisting return 3482, close after cancellation 238, offer on the closing date 153, close on the closing date 91, delisted before the exit date 1.

Deals whose outcome date is after the end of CRSP: 83. Completed cash deals among them are paid the offer on the closing date; broken deals among them exit on Compustat daily prices or, failing that, on the Capital IQ monthly close.

## Does the CRSP price before announcement agree with the export?

| Link method | Deals with both prices | Median CRSP P0 / implied price | Within 2% | Within 15% |
|---|---:|---:|---:|---:|
| cik_ccm | 4003 | 1.000 | 100% | 100% |
| none (Compustat prices only) | 180 | 1.000 | 96% | 99% |
| name | 97 | 1.000 | 96% | 97% |
| ticker | 3 | 1.000 | 100% | 100% |

Overall 99% of linked deals agree within 2% and 100% within 15%. Of the deals outside 15%, 0 are broken deals. Deals outside the tolerance are excluded from all price-based analyses.

## Cross-check against the Capital IQ monthly closes

22,384 month-end-style observations (the 20th of each month, or the prior trading day) from 4,293 deals fall inside the CRSP window of the same deal. Capital IQ close / CRSP close - 1: median +0.0000; within 0.5%: 84.1%; within 2%: 90.9%; more than 10% apart: 3.5%. For 79% of deals every observation agrees within 2%. Large differences are split-adjusted Capital IQ histories and a few wrong links; the 15% rule above removes the affected deals.

## Cross-check of the amount paid at closing (completed all-cash deals)

For 2,294 completed all-cash deals with a CRSP delisting payment: payment / export offer - 1 has median +0.00%, 5th to 95th percentile +0.0% to +0.0%; 2.5% differ by more than 5% (revised offers and contingent payments). The backtest uses the CRSP payment, the spread uses the export's offer.

## Which offer price is known on day 1?

The export stores the final terms of a deal. If the price was raised later, the final offer divided by the day-1 close looks like a very wide spread that no investor could have earned as a spread. For cash deals the per-share price in the announcement press release is therefore used when it differs from the export:

| Offer used at t1 | Usable all-cash deals |
|---|---:|
| export, confirmed by the press release | 1928 |
| export, not checked (no text or no price found) | 418 |
| press release (differs from the export) | 247 |

In the 247 deals where the press release differs, the median press-release offer is 0.91 times the export's. Usable all-cash deals with an entry spread above 25%: 77 with the export's offer, 44 with the announcement-day offer. Deals without announcement text keep the export's offer and can still carry this error; the backtest therefore also shows a panel restricted to spreads between -5% and +25%.

## Day-1 spread and market-implied break probability (usable all-cash deals)

| Period | Outcome | Deals | Median spread | Mean spread | Median p_mkt | Mean p_mkt | Mean return to exit | Median holding days |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test 2019-2025 | adverse_break | 22 | 3.3% | 12.7% | 0.12 | 0.30 | -37.9% | 396 |
| test 2019-2025 | closed | 632 | 1.7% | 2.9% | 0.07 | 0.16 | +3.6% | 92 |
| test 2019-2025 | topped | 6 | -0.1% | 1.0% | 0.01 | 0.10 | +16.2% | 68 |
| train 2001-2018 | adverse_break | 64 | 2.7% | 3.6% | 0.25 | 0.37 | -26.5% | 170 |
| train 2001-2018 | closed | 1822 | 1.3% | 1.8% | 0.07 | 0.16 | +2.9% | 86 |
| train 2001-2018 | topped | 47 | 0.1% | 0.5% | 0.00 | 0.12 | +14.8% | 58 |

Median calendar days from announcement to t1: 1. With monthly data it was about two weeks, so the market now has a one-day head start over the announcement-date model, not a month.

## Stock and mixed deals (robustness only)

1372 usable stock and mixed deals have an acquirer price (73 adverse breaks). The exchange ratio is implied from the export's stock part of the offer and the acquirer's close before announcement, which assumes a fixed ratio (no collars, no fixed-value deals) and that the export's offer value was struck at that close. The backtest return for these deals is the unhedged target return.


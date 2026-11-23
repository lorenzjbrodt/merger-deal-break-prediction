# Linking deals to CRSP

Source: WRDS (CRSP daily stock file in the current format, `crsp.dsf_v2`; Compustat `comp.company`; CRSP/Compustat Merged link table `crsp.ccmxpf_lnkhist`). Licensed data: this file contains counts only.

**CRSP coverage ends on 2025-12-31.** Deals whose outcome date is later (104 deals, 13 of them broken) cannot be followed to the end in CRSP; for those the exit uses Compustat daily prices where available and otherwise the Capital IQ monthly closes (see `results/crsp_prices.md`).

## Target link, by period and outcome (core sample)

| Period | Outcome | Deals | Linked to a permno | via CIK and CCM | via ticker | via name | Not linked |
|---|---|---:|---:|---:|---:|---:|---:|
| 2001-2012 | adverse_break | 114 | 101 (89%) | 95 | 0 | 6 | 13 |
| 2001-2012 | closed | 2117 | 1969 (93%) | 1887 | 2 | 80 | 148 |
| 2001-2012 | topped | 61 | 54 (89%) | 52 | 0 | 2 | 7 |
| 2013-2018 | adverse_break | 44 | 43 (98%) | 43 | 0 | 0 | 1 |
| 2013-2018 | closed | 1031 | 950 (92%) | 939 | 2 | 9 | 81 |
| 2013-2018 | topped | 18 | 17 (94%) | 17 | 0 | 0 | 1 |
| 2019-2025 | adverse_break | 40 | 35 (88%) | 34 | 0 | 1 | 5 |
| 2019-2025 | closed | 1026 | 955 (93%) | 951 | 0 | 4 | 71 |
| 2019-2025 | topped | 17 | 16 (94%) | 16 | 0 | 0 | 1 |
| All | all | 4468 | 4140 (93%) | 4034 | 4 | 102 | 328 |

Reverse mergers (outside the core sample): 276 deals, 215 linked.

Targets without a link are mostly companies whose shares did not trade on NYSE, NYSE American or Nasdaq (CRSP does not cover OTC stocks) and deals for which the export has no CIK.

## Acquirer link (needed for stock and mixed deals)

| Period | Consideration | Deals | Buyer shown with a US ticker | Acquirer linked | Target and acquirer linked |
|---|---|---:|---:|---:|---:|
| 2001-2012 | all cash | 1408 | 374 | 531 | 506 |
| 2001-2012 | stock or mixed | 884 | 368 | 736 | 698 |
| 2013-2018 | all cash | 631 | 173 | 245 | 231 |
| 2013-2018 | stock or mixed | 462 | 265 | 391 | 358 |
| 2019-2025 | all cash | 688 | 202 | 239 | 229 |
| 2019-2025 | stock or mixed | 395 | 294 | 358 | 323 |

Acquirer link methods: none 2086, name 1219, ticker_date 1163, ticker_permno 276. Capital IQ shows a ticker only for buyers listed today, so buyers that were later acquired or delisted are found by name. The remaining unlinked buyers are private companies, financial sponsors or foreign-listed companies; a stock deal with such a buyer has no CRSP acquirer price.


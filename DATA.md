# Data

Two sources. Neither is redistributed in this repository.

## 1. Capital IQ transaction screen (licensed, not included)

The deal table comes from S&P Capital IQ. It is licensed data and is never committed
(`data/raw/`, `data/interim/`, `*.xls`, `*.xlsx` and `*.parquet` are git-ignored). Anyone with
Capital IQ access can rebuild the export with the screen below.

### Screen

Capital IQ > Screening > **Transaction Screening**, with these criteria:

| # | Criterion | Value |
|---|---|---|
| 1 | Total Transaction Value ($USDmm, Historical rate) | is greater than 100 |
| 2 | Transaction Types | Merger/Acquisition |
| 3 | Geographic Locations (Target/Issuer) | United States of America (Primary) |
| 4 | Transaction Status | Closed OR Cancelled |
| 5 | Target Market Cap 1-Day Prior ($USDmm, Historical rate) | is greater than 0 |

The export used here has 7,971 transactions.

### Display columns

Add these display columns, in this order (names exactly as they appear in the file header,
row 8 of the sheet `Screening`):

1. All Transactions Announced Date
2. Target/Issuer
3. Exchange:Ticker
4. Transaction Types
5. Transaction Status
6. Total Transaction Value ($USDmm, Historical rate)
7. Buyers/Investors
8. Sellers
9. CIQ Transaction ID
10. Geographic Locations [Target/Issuer]
11. Implied Equity Value ($USDmm, Historical rate)
12. Implied Enterprise Value ($USDmm, Historical rate)
13. Implied Enterprise Value/EBITDA (x)
14. Percent Sought (%)
15. Offer per Share ($USD, Historical rate)
16. Cash % of Consideration (%)
17. Stock % of Consideration (%)
18. Target Spread ($USD, Historical rate)
19. Target Spread (%)
20. Sell-Side Termination Fee (%)
21. Buy-Side Termination Fee (%)
22. Target Stock Premium - 1 Day Prior (%)
23. Target Stock Premium - 1 Week Prior (%)
24. Target Stock Premium - 1 Month Prior (%)
25. Target LTM Return % (%)
26. Target % of 52-Week High (%)
27. Target Market Cap 1-Day Prior ($USDmm, Historical rate)
28. M&A Announced Date
29. M&A Definitive Agreement Date
30. M&A Closed Date
31. M&A Cancelled Date
32. Special Committee
33. Deal Attitude
34. Deal Approach
35. Deal Responses
36. Deal Conditions
37. Consideration Offered
38. CIK [Target/Issuer]
39. Excel Company ID [Target/Issuer]
40. Security Tickers [Target/Issuer]
41. SIC Codes (Primary Code Only) [Target/Issuer]

### Export

Export **All Transactions** to Excel. The file is a legacy `.xls` workbook with three sheets
(`Screening`, `Aggregates`, `Screen Criteria`). Save it as:

```
data/raw/ciq_transactions.xls
```

`src/prepare.py` reads sheet `Screening` with the header on row 8 (`header=7`) and treats `-`
as missing.

### Sample filters (applied in `src/prepare.py`, in this order)

| Step | Deals | Breaks |
|---|---:|---:|
| Raw export | 7,971 | 1,175 |
| Status in {Closed, Cancelled} | 7,971 | 1,175 |
| M&A Definitive Agreement Date present | 7,294 | 572 |
| Percent Sought >= 50 | 6,974 | 560 |
| Target name not contained in Buyers/Investors | 6,947 | 560 |
| Announcement year 2001-2025 | 5,156 | 456 |
| Drop SPACs (primary SIC 6770, or target name matching Acquisition Corp / Acquisition Co / Acquisition Holdings / Merger Corp / Blank Check) | 4,745 | 346 |
| Drop one duplicated transaction ID | 4,744 | 346 |

Final sample: **4,744 deals, 346 breaks (7.3%)**. Label: 1 = Cancelled, 0 = Closed.

### Columns never used as features

Transaction Status, M&A Closed Date, M&A Cancelled Date, Target Spread ($ and %) and anything
derived from them (outcome information). Sell-side and buy-side termination fees are almost
empty and are dropped. Deal Responses and Deal Attitude can be updated after the announcement;
they are excluded from all main models and appear in one labelled ablation.

## 2. SEC EDGAR filings (public, downloaded by the code)

`src/edgar_download.py` uses the target CIK(s) from the export and the SEC submissions API
(`https://data.sec.gov/submissions/CIK##########.json`, including the older `files` pages) to
list each target's filings from **one calendar day before to five business days after** the
announcement date. Forms kept: 8-K, 425, DEFA14A, SC TO-C, SC 14D9, SC TO-T, PREM14A. For each
filing it downloads the complete submission text file and keeps the primary document and the
exhibits EX-99.* and EX-2.*.

- Requests are throttled below 5 per second. Everything is cached in `data/edgar_cache/`
  (git-ignored) and the download is resumable.
- The SEC asks automated clients to identify themselves. Set
  `SEC_USER_AGENT="Your Name your-contact-address"` in the environment before running.
- `src/build_text.py` turns the cache into two text fields per deal, `press_release` and
  `agreement`. Coverage by year is in `results/text_coverage.md`.

## 3. Synthetic sample (included)

`data/sample/synthetic_ciq_export.csv` has 200 **synthetic** rows with the exact 41-column
header of the export. Values are drawn from made-up distributions by `src/make_synthetic.py`;
no Capital IQ record is used and company names are placeholders. It exists so that
`./run_all.sh synthetic` runs the whole pipeline without a licence. Numbers produced from it
are meaningless and are written to `results_synthetic/` and `figures_synthetic/`, which are
git-ignored. All results in `results/` and `figures/` come from the real export.

# Data

Two sources. Neither is redistributed in this repository.

## 1. Capital IQ transaction screen (licensed, not included)

The deal table comes from S&P Capital IQ. It is licensed data and is never committed
(`data/raw/`, `data/interim/`, `data/external/`, `artifacts/`, `*.xls`, `*.xlsx` and `*.parquet` are git-ignored). Anyone with
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

### Extended export (used from the extended run on)

The same screen and rows with 40 more display columns (81 in total), saved as `data/raw/ciq_transactions_v2.xls`.
Add these after the 41 columns above, in this order:

42. Net Assumed Liabilities ($USDmm, Historical rate)
43. Implied Equity Value/LTM Net Income (x)
44. Implied Equity Value/Book Value (x)
45. Implied Enterprise Value/Revenues (x)
46. Total Cash ($USDmm, Historical rate)
47. Total Stock ($USDmm, Historical rate)
48. Total Earnout/Contingent Payments ($USDmm, Historical rate)
49. Sell-Side Termination Fee ($USDmm, Historical rate)
50. Buy-Side Termination Fee ($USDmm, Historical rate)
51. Acquirer LTM Return % (%)
52. Target % of 52-Week Low (%)
53. Acquirer % of 52-Week High (%)
54. Acquirer Market Cap 1-Day Prior ($USDmm, Historical rate)
55. Transaction Comments
56. Pre-Deal Situation
57. Deal Resolution
58. Advisors To Target
59. Advisors To Buyers
60. Transaction Primary Features
61. Transaction Secondary Features
62. Majority / Minority Stakes
63. Change of Control
64. Related Transactions
65. to 71. Target/Issuer LTM Financials (at Announcement) ($USDmm, Historical rate): Total Revenue, EBITDA, Net Income,
    Total Common Equity, Total Debt, Total Cash & ST Investments, Total Assets
72. to 77. Acquirer LTM Financials (at Announcement) ($USDmm, Historical rate): Total Revenue, EBITDA, Net Income,
    Total Debt, Total Cash & ST Investments, Total Assets
78. Primary Industry [Buyers/Investors]
79. Primary Sector [Buyers/Investors]
80. Buyer Type [Buyers/Investors]
81. Company Type [Buyers/Investors]

How the new columns are used:

- **Model features (known at announcement):** 42-54, 58-61, 65-81. Advisor fields are reduced to the number of financial
  and legal advisers per side and a top-adviser flag computed from training-period frequencies.
  Caveat: buyer `Company Type` and `Buyer Type` are Capital IQ's *current* classification, not the one at announcement.
- **Never model features (written or updated after the fact):** Transaction Comments, Pre-Deal Situation, Deal Resolution,
  Related Transactions. They are used only to classify why a cancelled deal broke and to flag revised offers.
- **Dropped:** Majority / Minority Stakes and Change of Control are constant in the sample. Four secondary-feature tags are
  applied when a deal completes (for example "Squeeze Out / Short Form Merger": 428 training deals, zero breaks) and are
  excluded as outcome-driven.

### Price exports

Same screen, display columns reduced to the identifiers (All Transactions Announced Date, Target/Issuer, Exchange:Ticker,
Transaction Status, CIQ Transaction ID, M&A Announced Date, M&A Closed Date, M&A Cancelled Date) plus the target's
**Day Close Price** on a monthly grid (the 20th of each month), `$USD, Historical rate`, 96 to 111 periods per export
(Capital IQ limits the number of periods per screen):

| File | Save as | Periods | Dates |
|---|---|---:|---|
| export 3 of 3 | `data/raw/ciq_prices_3.xls` | 111 | 12/20/2000 to 2/20/2010 (fixed dates) |
| export 2 of 3 | `data/raw/ciq_prices_2.xls` | 105 | 3/20/2010 to 11/20/2018 (fixed dates) |
| export 1 of 3 | `data/raw/ciq_prices_1.xls` | 96 | "Latest - 95 Month(s)" to "Latest" = 12/20/2018 to 11/20/2026 |

`src/prices.py` parses the dates from the column headers, checks that the three files form one grid of 312 months with
no gap or overlap, and writes a long table (deal, date, close). A non-trading 20th returns the prior close. `-` means no
price (not yet listed, or delisted). Checks and coverage: `results/price_coverage.md`.

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

## 3. FRED market series (public, cached locally)

`src/market_context.py` reads daily CSVs from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` saved as
`data/external/fred/<ID>.csv` (git-ignored): VIXCLS, DBAA, DAAA, DGS10, DTB3, NASDAQCOM, SP500, BAMLH0A0HYM2.
FRED answered only browser requests when this was built, so the files were saved through a browser.

Two series do not go back far enough on FRED and are replaced by proxies:

| Wanted | Problem on FRED | Used instead |
|---|---|---|
| S&P 500 trailing returns (`SP500`) | only the last 10 years (from November 2016) | NASDAQ Composite (`NASDAQCOM`) trailing 3- and 12-month returns |
| High-yield OAS (`BAMLH0A0HYM2`) | only the last 3 years (from November 2023) | Moody's Baa yield minus the 10-year Treasury yield (`DBAA` - `DGS10`) |

## 4. LLM outputs (derived, cached locally)

Term extraction, break-reason classification, the parties' stated expected closing and the exploratory zero-shot score are produced with headless Claude Code
(`claude -p`) and cached per document under `data/interim/` (git-ignored). Only aggregate validation tables are committed.

## 5. WRDS: CRSP and Compustat (licensed, not included)

Daily prices, the day-1 spread, the daily backtest, the loss-if-break model and the overlap features use CRSP and
Compustat through the WRDS PostgreSQL server. Nothing from WRDS is committed: the pull is cached in `data/wrds/`
(git-ignored) and only aggregate tables appear in `results/`. Without WRDS the repository still runs: the Capital IQ
monthly closes produce `results/monthly/` (documented fallback).

### One-off setup

1. A WRDS account with CRSP and Compustat access. Log in once on <https://wrds-www.wharton.upenn.edu/> from the network
   you will pull from: WRDS ties its two-factor (Duo) session to the IP address for up to 30 days, and a database
   connection from an address without a session is refused.
2. Store the password where the PostgreSQL client reads it (`~/.pgpass`, mode 600). This asks for it with hidden input
   and never puts it on the command line or in the repo; replace `YOUR_WRDS_USER`:

```bash
.venv/bin/python -c 'import getpass,os,pathlib; u="YOUR_WRDS_USER"; f=pathlib.Path.home()/".pgpass"; h="wrds-pgdata.wharton.upenn.edu:9737:wrds:"+u+":"; p=getpass.getpass("WRDS password (hidden): ").replace("\\","\\\\").replace(":","\\:"); keep=[l for l in (f.read_text().splitlines() if f.exists() else []) if not l.startswith(h)]; f.write_text("\n".join(keep+[h+p])+"\n"); os.chmod(f,0o600)'
```

   The code takes the user name from that line (or from `WRDS_USER`). Connection: host `wrds-pgdata.wharton.upenn.edu`,
   port 9737, database `wrds`, SSL required (`psycopg`).

### Rebuilding the pull

```bash
.venv/bin/python src/wrds_pull.py lookups   # reference tables, under a minute
.venv/bin/python src/crsp_link.py           # deals -> permno / gvkey, results/crsp_link.md
.venv/bin/python src/wrds_pull.py daily     # daily data for linked securities, about 5 minutes
.venv/bin/python src/crsp_prices.py         # per-deal price features, results/crsp_prices.md
```

| Table | Used for |
|---|---|
| `comp.company`, `comp.security` | CIK -> gvkey; company names; primary issue |
| `crsp.ccmxpf_lnkhist` | gvkey -> permno (link types LU, LC, LS; valid on the announcement date) |
| `crsp.stocknames_v2` | ticker and name history for the fallback links and for acquirers |
| `crsp.dsf_v2` | daily close, total return including dividends and the delisting return, adjustment factors, delisting flag (current CRSP format, to 31 December 2025) |
| `crsp.stkdelists` | delisting records (pulled for reference; the delisting return and payment are already in `dsf_v2`) |
| `comp.secd` | Compustat daily prices: second source after the end of CRSP and for targets CRSP does not cover |
| `comp.funda` | annual sales and historical SIC codes (industry shares, concentration) |
| `ff.factors_daily` | daily market return and T-bill rate (market adjustment, beta, Sharpe ratio) |

Windows: from 75 calendar days before the announcement to 30 after the outcome date (trimmed to trading days in the
code), plus one earlier year of target returns for volatility and beta. How deals are linked, how many, and the last
date covered: `results/crsp_link.md`. Price checks against the export and against the Capital IQ monthly closes:
`results/crsp_prices.md`.

## 6. Hoberg-Phillips TNIC-3 (public, downloaded once)

Product-similarity pairs of Compustat firms by year (Hoberg and Phillips 2010, 2016), used for the antitrust-overlap
feature group. Download <https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip> (151 MB) and unpack it to
`data/external/hoberg_phillips/extracted/tnic3_data.txt` (git-ignored). The file is keyed by gvkey pair and year and
ends in 2023; `src/overlap.py` uses the latest year before the announcement, at most three years back.

## 7. Synthetic sample (included)

`data/sample/synthetic_ciq_export.csv` has 200 **synthetic** rows with the exact 81-column header of the extended export.
Values are drawn from made-up distributions by `src/make_synthetic.py`; no Capital IQ record is used and company names are
placeholders. It exists so that `./run_all.sh synthetic` runs the core pipeline (and the demo) without a licence. Numbers
produced from it are meaningless and are written to `results_synthetic/` and `figures_synthetic/`, which are git-ignored.
All results in `results/` and `figures/` come from the real exports.

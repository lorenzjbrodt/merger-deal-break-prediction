# Decisions

Every judgment call made while building this project, in the order the pipeline runs.
"Brief" means the project instructions. Numbers are from the real Capital IQ export.

## 1. Data and sample

1. **Export file name.** The brief names `Transaction_Screening_Report*.xls`. The file in Downloads is
   `Transaction Screening Report.xls` (spaces). It was identified by content: sheet `Screen Criteria`
   contains "Target Market Cap 1-Day Prior". Copied to `data/raw/ciq_transactions.xls`.
2. **Sample size versus the sanity check.** The filters in the brief give 4,745 deals and 346 breaks;
   one transaction ID appears twice in the export, so the final sample is **4,744 deals, 346 breaks (7.3%)**.
   Expected: about 4,723 and 344. Difference +0.4% and +0.6%, inside the 5% tolerance. Not investigated further.
3. **Announcement date.** `M&A Announced Date`. It equals `All Transactions Announced Date` for every sample deal.
4. **"Target name not contained in Buyers/Investors".** Case-insensitive substring match after removing the
   `(Exchange:Ticker)` suffix from the target name. Removes 27 self-tenders.
5. **SPAC filter.** Exactly as specified (primary SIC 6770, or the target name matches Acquisition Corp /
   Acquisition Co / Acquisition Holdings / Merger Corp / Blank Check). See decision 34: this filter misses
   most de-SPAC deals of 2020-2024.
6. **Missing values.** `-` is read as missing. Footer rows without a transaction ID are dropped.

## 2. Leakage rules and tabular features

7. **Never used:** Transaction Status, Closed Date, Cancelled Date, Target Spread ($, %), and anything derived
   from them. Termination-fee percentage columns are about 96% empty in the base export and are dropped.
8. **Leakage suspects:** Deal Responses and Deal Attitude are excluded from every main model and appear in one
   labelled ablation (A1). "Rejection" has a 65% break rate, which is what an after-the-fact field looks like.
9. **Days between definitive agreement and announcement.** 6% of deals were signed after the announcement.
   That gap is not known on the announcement date, so the feature is the number of days signed *before*
   announcement (clipped to 0-60) plus a flag "not yet signed at announcement". The length of a later gap is not used.
10. **Cash % and stock %.** Capital IQ leaves the complement blank for all-cash and all-stock deals. The missing
    side is filled with 100 minus the other side when one of them is present.
11. **Outliers.** Fixed, label-free bounds (for example premiums clipped to -50%..200%, EV/EBITDA to 0..100).
    No bounds are estimated from the data.
12. **Missingness is informative, and that is a data artefact.** Capital IQ fills fewer fields for cancelled
    deals: the break rate is 19% when the premium is missing and 6% when it is present. LightGBM uses missing
    values natively, so the main models can exploit this. A portfolio manager would know the premium on the
    announcement day, so part of the measured skill may not transfer. Ablation A2 (median-imputed, no missing
    indicators) quantifies it.
13. **Announcement year as a feature.** Listed as usable in the brief, but a time split extrapolates it. Whether
    to include it is decided by cross-validation inside the training years (once, for the tabular LightGBM,
    then held fixed for every LightGBM variant so that text uplifts are like for like).
14. **Capital IQ fields may reflect final terms.** Deal value, offer price and premium can be updated when an
    offer is revised. This cannot be undone from the export. It is listed as a limitation and tested in the
    extension (decision 40).

## 3. SEC EDGAR text

15. **User-Agent.** The brief asks for a name-and-email User-Agent on every SEC request. A privacy guard in the
    tool that built this project blocks writing that address into a script, so `edgar_download.py` reads
    `SEC_USER_AGENT` from the environment and otherwise sends a generic client name. The download for these
    results ran with the generic name at 4.4 requests per second (limit in the brief: 5).
16. **One request per filing.** The complete submission text file is downloaded once per filing and split into
    documents locally, rather than fetching the index page and each exhibit. Fewer requests, same content.
    Only the primary document and EX-99.* / EX-2.* exhibits are cached (as gzipped JSON); binaries are skipped.
17. **Cap of 8 filings per deal**, taken round-robin over the form types, earliest first (two 8-Ks before the
    second filing of any other form). A few deals have dozens of near-duplicate 425 / DEFA14A filings.
18. **Window.** One calendar day before to five business days after the announcement. Business days are
    Monday to Friday; exchange holidays are not modelled, so the window can be one day short around holidays.
19. **Several CIKs.** All CIKs in the target field are queried and filings are de-duplicated by accession number.
20. **Coverage.** 4,482 of 4,744 deals have a target CIK; 4,480 were downloaded (two filings kept timing out).
    88% of deals have a usable announcement text and 78% a merger agreement (`results/text_coverage.md`).
21. **Press-release selection.** One document per deal, by tier: (0) an EX-99 exhibit that looks like a press
    release and announces the deal in its opening; (1) another EX-99 that announces the deal; (2) the body of a
    425 / DEFA14A / SC TO-C / SC 14D9 that announces the deal; (3) the 8-K body; (4+) fallbacks. Earliest
    filing wins within a tier. A candidate must mention deal terms at least twice. Legal agreements filed as
    EX-99 (voting agreements, or the merger agreement itself in older filings) are not press releases; a long
    one titled as a merger agreement is used as the agreement.
22. **Agreement.** The longest EX-2.* exhibit whose opening mentions "agreement", capped at 600,000 characters.
23. **Boilerplate.** HTML is stripped; press releases are cut at the first trailing section header (forward-looking
    statements, additional information, participants in the solicitation, "About ...", contacts); the 8-K cover
    page and signature block are removed.
24. **Text subset** = deals with a press-release text of at least 400 characters. Text models are trained and
    compared on this subset only; the full-sample text model uses a `has_text` flag and missing values elsewhere.
25. **Deals resolved inside the text window.** 33 deals closed or were cancelled within 10 days of announcement,
    so their in-window filings could mention the outcome. The brief's window is kept; a robustness check drops
    these deals from the evaluation.

## 4. Models and evaluation

26. **Split.** Train 2001-2018 (3,386 deals, 237 breaks), test 2019-2025 (1,358 deals, 109 breaks).
27. **Cross-validation.** Expanding window inside the training years: validate on 2007-09, 2010-12, 2013-15,
    2016-18, each time training on all earlier years. Selection criterion: mean fold ROC AUC.
28. **Small grids on purpose.** LightGBM: num_leaves in {3, 6, 12} x trees in {100, 200, 400}; learning rate 0.03,
    min_child_samples 25, column subsample 0.5, row subsample 0.8, L2 5. Logistic: C in six values. With about
    30-60 breaks per validation block a large search would fit noise.
29. **No class weights, five seeds.** LightGBM is trained on the natural class balance so that its output is a
    probability, and final predictions average five seeds.
30. **One-rule benchmark.** Chosen by CV among seven candidate rules; its score is the training break rate in each group.
31. **Text features.** TF-IDF uses letter-only tokens (no dates, years or prices, which would encode the period),
    unigrams and bigrams for the press release, unigrams for the agreement, reduced by truncated SVD to 30 + 10
    components. Embeddings: `BAAI/bge-small-en-v1.5`, frozen, 300-word chunks, first 10 chunks of the press release
    and 24 evenly spaced chunks of the agreement, mean-pooled, PCA to 20 + 10 components. Every vocabulary, SVD and
    PCA is fitted on the training fold. MPS is used for embedding inference only.
32. **Metrics.** "Breaks captured in the riskiest 10% / 20%" ranks test deals by score and splits ties pro rata.
    Confidence intervals: 2,000 bootstrap resamples of test deals; uplifts use the same resamples for both models.
33. **Test-set discipline.** The pipeline was debugged end to end on the synthetic sample before the real test
    set was scored. Bugs found afterwards were fixed without changing any modelling choice in response to test numbers.

## 5. Stylized dollar impact (no price data in the base run)

See the assumption list at the top of `results/impact.md`. The main choices: equal weights; one capital turn per
year; skipped capital is redeployed into the remaining deals; flagged deals are ranked inside the test period,
with a variant whose cut-off is fixed in advance from training-period out-of-fold scores; a break-even spread is
reported because real spreads are wider for risky deals.

## 6. Housekeeping

- LightGBM on macOS needs the OpenMP runtime: `brew install libomp`.
- The synthetic sample uses a break rate near 20% (real: 7%) so that 200 rows contain enough events to run.
- `gh` was switched to the `lorenzjbrodt` account as instructed and left there. No other account was logged out.
- Commits are made as the configured git user, with no co-author or tool trailers.

---

# Extended run

Decisions made for the scope extension (extended export, outcome taxonomy, market context, LLM term extraction,
fine-tuned encoder, price data, evaluation rigor, demo, project page).

## 7. Sample and targets

34. **Reverse mergers are excluded from the primary sample.** While validating break reasons it became clear that 276 sample
    deals carry Capital IQ's "Reverse Merger" tag (de-SPACs and shell mergers), 275 of them in the test period and 1 in the
    training period, and that they account for 52 of the 109 test-period breaks. The SPAC filter in the brief (SIC 6770 plus
    name patterns) misses names such as "... Capital Corp VII" or "... Acquisition II Corp." These are not merger-arbitrage
    situations, and a model trained on 2001-2018 has never seen one. The extended analysis therefore uses a **core sample**
    of 4,468 deals as primary and reports the sample as specified (4,744) next to it. The tag describes the deal structure
    and is known at announcement, so filtering on it is not leakage. The base run is unchanged and kept in `results/base_run/`.
35. **Two targets.** (a) any break, as in the base run; (b) **adverse break** = cancelled for a reason other than a superior
    competing bid. With 246 adverse breaks (at least 150 were required), (b) is the primary target for the impact analysis.
    A three-class view (closes, topped, adverse break) is reported as well.
36. **Extended export.** `data/raw/ciq_transactions_v2.xls` replaces the base export (same rows, 81 columns). The EDGAR cache
    stays valid. "Majority / Minority Stakes" and "Change of Control" are constant in the sample and dropped.

## 8. New tabular features (A)

37. **Outcome-driven tags are excluded.** Four "Transaction Secondary Features" tags are applied when a deal completes:
    "Squeeze Out / Short Form Merger" (428 training deals, zero breaks), "Platform" (42, zero), "Secondary LBO" and
    "Successor Issuer Merger". They would leak the outcome and are never used.
38. **Advisers.** Only financial and legal advisers are counted. Information agents, transfer agents and proxy solicitors are
    appointed later in the process, so their presence partly reflects how far a deal got. The top-adviser flag uses the ten
    most frequent advisers per side and role, counted on the training rows of each fold only.
39. **Same-sector flag.** The export has the buyer's sector but only a SIC code for the target. The target's sector comes from
    an approximate SIC-to-sector mapping (`GICS_FROM_SIC` in `features.py`); the flag is therefore approximate.
40. **Buyer type and company type are current classifications**, not as of announcement (a buyer that was public in 2005 and
    is private now shows as private). They are used, with this caveat documented.
41. **Termination fees** in USD and as a percentage of deal value are used where present (sell-side 89%, buy-side 38%).

## 9. Outcome taxonomy (B)

42. **Classification.** Every cancelled deal was classified from the Capital IQ notes by Claude Sonnet (headless `claude -p`,
    batches of 8 cases, fixed JSON schema, instructed to use only the notes). Keyword rules give an independent cross-check.
43. **Rival-deal rule.** The notes of a topped deal often say only "cancelled". If the export contains another buyer's
    majority deal for the same target, announced between the deal's announcement and 45 days after its cancellation, and the
    notes gave no specific adverse reason (`unknown` or `mutual_other`), the break is relabelled as a competing bid. This
    moved 20 deals. 70 of the 80 LLM-labelled competing bids have such a rival deal, which supports both the label and the rule.
44. **Validation.** 50 random cancelled deals were re-read by hand. All 50 labels were faithful to the notes; 3 were wrong
    against what actually happened (silent notes on topped deals), and the rival-deal rule fixes all 3.
45. **Revised offers.** A deal is flagged when its comments mention a raised, lowered or amended offer (regular expression on
    all deals, plus the LLM flag for cancelled deals): 413 deals. The flag is used only for a sensitivity analysis.
46. **None of these fields is a model feature.** Deal Resolution, Transaction Comments, Pre-Deal Situation and Related
    Transactions define labels and sensitivity flags only.

## 10. Market context (C)

47. **FRED access.** FRED rejected command-line requests, so the CSVs were saved through a browser into `data/external/fred/`.
48. **Proxies.** FRED carries the S&P 500 only from November 2016 and the high-yield OAS only from November 2023. Trailing
    equity returns use the NASDAQ Composite, and the credit-stress measure is the Moody's Baa yield minus the 10-year
    Treasury yield (in addition to Baa minus Aaa). Both proxies cover 1999 onward.
49. **As-of rule.** Every market feature uses the last observation strictly before the announcement date. The trailing break
    rate uses only deals whose outcome date lies before the announcement and is undefined until 20 such deals exist. The
    90-day deal count is truncated at the start of the sample (early 2001).

## 11. LLM term extraction (D)

50. **Method.** Headless `claude -p` with Claude Haiku, one document per call, no tools, a fixed JSON schema, instructions in
    the system prompt: use only the provided text, return null when a term is not stated, never use outside knowledge, do not
    predict the outcome. Input per deal: the press release (first 12,000 characters) and key passages of the merger agreement
    (24,000 characters of keyword windows: termination fees, outside date, financing, regulatory approvals, go-shop,
    conditions). 4,180 documents, cached per document, resumable. The CLI login had expired; it was renewed by the project
    owner before the run.
51. **Validation.** 60 random documents were re-extracted with the strongest model (Claude Fable). A field is kept when
    overall agreement is at least 85% and agreement on documents where either model found a value is at least 60%.
    Dropped: expected closing timing, antitrust covenant strength, acquirer country, and the catch-all "other" regulatory
    approval. Regulatory approvals are judged per approval type because exact agreement on the whole list is a harsh test.
    29 features remain (`results/llm_extraction_validation.md`).
52. **Order of work.** The extraction was started before steps A to C were finished because it is the longest job. It does
    not depend on them.
53. **Exploratory zero-shot score.** Claude Sonnet rates anonymised test-period press releases. It is labelled exploratory
    everywhere, never enters a model, and carries the caveat that the LLM may know how well-known deals ended.

## 12. Fine-tuned encoder (E)

54. `distilroberta-base`, first 512 tokens of the press release, trained on announcements up to 2015, early stopping on
    2016-2018 validation AUC (patience 2), inverse-frequency class weights, batch size 8 on MPS, with a watchdog (60 minutes,
    20 GB). Both runs finished in about 10 minutes at 5.5 GB. The score is reported as a text-only step and fused with the
    tabular model through a two-variable logistic stacker fitted on 2016-2018, where both scores are out of sample.

## 13. Price data (F)

55. **Grid.** 312 monthly dates, no gaps or overlaps across the three files; a non-trading 20th returns the prior close.
56. **Validity rule.** The brief compares the last monthly close before announcement with offer / (1 + 1-month premium) and
    excludes deviations above 15%. The grid close can be a month away from that reference, so ordinary price movement fails
    13% of deals. The analyses use the premium horizon closest in time to the grid date (1 day, 1 week or 1 month), which
    fails 6%; the count under the literal rule is reported in `results/price_coverage.md`.
57. **Split adjustment.** Deviations cluster at clean split factors for only 34 deals; the closes are unadjusted historical
    prices for the large majority. All deals outside the tolerance are excluded whatever the cause.
58. **Definitions.** t1, spread, p_mkt, the carry variant and the exit rules are in the header of `src/price_analysis.py`.
    Entry spreads outside (-50%, +100%) are treated as data errors.
59. **Backtest return.** Annual return = total P&L divided by total capital-years, so a deal that ties up capital for longer
    weighs more. Rankings ("riskiest 10%") are made inside the priced test deals.
60. **Two additions made after seeing the first backtest output** (disclosed because they were not planned): a panel
    restricted to entry spreads between -5% and +25%, because a few very wide spreads dominate the averages, and a rank-based
    version of "model below market", because p_mkt is not calibrated and the level comparison mostly selects wide spreads.
    The original panels and rules are reported unchanged next to them.
61. **Market benchmark.** The stack of model and market, and the market-only logistic, are fitted on training-period deals
    using the model's out-of-fold scores there, then applied to the test deals.

## 14. Evaluation (G)

62. **Final model.** Chosen by CV AUC on the training years among the full-sample candidates; the test set plays no part.
63. **Rolling origin.** Hyper-parameters are those selected in the main run and are held fixed across origins; origins
    before 2018 therefore carry a mild look-ahead in the hyper-parameters only.
64. **Calibration.** Platt scaling fitted on training-period out-of-fold predictions (validation blocks 2007-2018).
65. **Ablation.** Drop-one-group and group-alone, text subset, hyper-parameters of the all-groups model held fixed.
66. **SHAP.** LightGBM's exact TreeSHAP (`pred_contrib`), averaged over five seeds; no sampling approximation.
67. **Label audit.** Automatic check of 40 + 40 deals against EDGAR filing metadata (rules in `src/label_audit.py`).
68. **Case studies.** Selected by a fixed rule from the final model's scores (highest-scored adverse breaks, lowest-scored
    adverse breaks, highest-scored completed deals) and written from public filing text.
69. **Test-set discipline in the extension.** The extended script was debugged on the synthetic sample and then with one
    quick pass on the real data. No feature, hyper-parameter grid or selection rule was changed after test metrics were
    seen. The only analysis choices added after seeing results are the two backtest additions in decision 60.

## 15. Demo, page, housekeeping

70. **Demo.** "Pick a deal" uses the final model's stored scores and SHAP values. "Enter your own deal" uses a compact model
    on 14 typed terms plus keyword flags, because the full feature set cannot be typed in; the page says so.
71. **Screenshots.** The committed screenshots are taken from the synthetic fallback. Screenshots of the real app show one
    deal's Capital IQ-derived terms and scores, so they are saved under `artifacts/screenshots/` (git-ignored).
72. **Stylized impact grid** is kept as a secondary view; a topped deal is treated like a completed deal there.
73. `caffeinate -dims` kept the machine awake during the run and was stopped at the end.
74. **Leakage probe for the zero-shot score.** Because the exploratory LLM score ranked test deals far better than any
    model (AUC 0.82), the same anonymised texts were given to the same LLM with the question "which companies are these?".
    It named the target in 92% of deals. The zero-shot score is therefore reported as not interpretable: it cannot be
    separated from the LLM's memory of how the deals ended. It is used nowhere else.
75. **Backtest figure.** The figure shows the change in annual return against hold-all with 95% intervals for both the
    full panel and the plausible-spread panel, not the level of each strategy, so that the wide-spread deals of March 2020
    do not read as model skill.
76. **Demo terms.** The demo shows only extracted fields that passed validation.
77. **Final model.** CV on the training years selected the extended tabular model (3x) over the variants with market
    context or text; that choice is reported as made, although the full-sample text variant (5f) scored slightly higher
    on the test set.

## WRDS add-on (CRSP daily prices)

78. **Connection.** WRDS PostgreSQL through `psycopg` (host `wrds-pgdata.wharton.upenn.edu`, port 9737). The password
    lives only in `~/.pgpass`, typed by the account holder at a hidden prompt; the code reads the user name from that
    file or from `WRDS_USER`. WRDS refused the first connections with "PAM authentication failed" until the stored
    password was corrected; after a website login from the same network no Duo prompt was needed.
79. **Everything pulled once and cached.** Because the two-factor session is tied to the IP address, all WRDS data,
    including what later steps need (one extra year of target returns, Compustat annual sales, the daily market
    return), was pulled into `data/wrds/` in one sitting. Nothing after that needs the connection.
80. **CRSP format.** The current-format daily file `crsp.dsf_v2` is used because it runs to 31 December 2025; the
    legacy `crsp.dsf` stops at 31 December 2024. In this format the delisting return and the delisting payment sit on a
    final row of the daily file, so "hold to the last trading day including the delisting return" is the product of
    the daily returns through that row.
81. **Target link.** As specified: CIK -> gvkey (`comp.company`) -> permno through the CCM link valid on the
    announcement date (link types LU, LC, LS; primary links preferred). Fallbacks: ticker and date in the CRSP names
    history (the export shows a ticker for only 129 targets, so this adds 4 links), then, as an addition, an exact
    normalised name match against CRSP issuer names valid on the announcement date or Compustat names with a valid CCM
    link, unique matches only (258 links). Every link is checked against the export: the CRSP close before the
    announcement must be within 15% of offer / (1 + one-day premium).
82. **Acquirer link.** The export shows the buyer's current ticker, and none for buyers that were later delisted. A
    ticker match on the announcement date is accepted only if the CRSP issuer name resembles the buyer name (tickers
    are reused); otherwise any security that ever carried the ticker, resembles the name and traded on the date; then
    exact name matches through CRSP and Compustat (short and legal names). The name routes roughly doubled the number
    of linked acquirers of stock and mixed deals.
83. **Second price source.** The brief said to keep the Capital IQ monthly closes for deals beyond the end of CRSP.
    The same WRDS account also has Compustat daily prices to November 2026, which are daily and keyed by gvkey, so the
    order is: CRSP; Compustat daily where a CRSP series stops without a delisting record or a target has no permno;
    Capital IQ monthly close only as the last resort for the exit of a broken deal. Counts by source are in
    `results/crsp_prices.md`.
84. **Day-1 definitions.** As specified: P0 = close on the last trading day before the announcement date, t1 = first
    trading day after it, entry at close(t1), and the deal must still be open at t1.
85. **The export's offer is the final offer.** Capital IQ stores final terms. For deals whose price was raised later,
    final offer / close(t1) looked like a spread of 30% or more that nobody could have earned as a spread, and it made
    the market-implied probability wrong in the direction that hurts the market benchmark. For cash deals the per-share
    price stated in the announcement press release is therefore used at t1 when it differs from the export by more than
    1% and implies a premium between -30% and +200% over P0. Deals without announcement text keep the export's offer;
    a panel restricted to the offers confirmed by or taken from the press release is reported as robustness.
86. **Exits.** Completed deals: CRSP total return through the delisting record. Where there is no delisting record
    near the closing date (CRSP ends, or the security kept trading), cash deals are paid the offer on the closing date
    and stock or mixed deals exit at the close on that date. Broken deals: close of the first trading day after the
    cancellation date, and five trading days after it as the variant.
87. **Stock and mixed deals** (robustness only): exchange ratio = stock part of the export's offer per share divided
    by the acquirer's close before the announcement; everything that is not stock is treated as a fixed amount. This
    assumes a fixed ratio and that the export struck the offer value at that close. No acquirer hedge is modelled.
88. **Pooled out-of-time design.** One origin per test year from 2013 to 2025; a model for year Y is fitted on deals
    announced by Y-1. Hyper-parameters, the one rule, the use of the year and the choice among the three tabular
    feature sets are re-selected by CV inside each origin's training years, so nothing chosen on 2001-2018 as a whole
    leaks into the earlier test years. The fine-tuned encoder is left out (it was trained on data up to 2015).
89. **Pending deals in training.** A model for year Y trains on deals announced by Y-1 whose outcome may have become
    known during Y, the same convention as the main split. An embargo variant that trains on deals announced by Y-2 is
    reported to show that this does not drive the result.
90. **Intervals for pooled results** resample announcement years (13 clusters), as requested. With 13 clusters the
    percentile interval is itself rough, so the deal-level interval is printed next to it, and a within-year AUC is
    reported because scores from 13 different fitted models are pooled.
91. **Market side of the comparison.** The market logistic, the stack and the model-versus-market threshold are fitted
    on earlier years only, per origin in the pooled set. The threshold is the training-period quantile of the gap
    between model and market (in log-odds) that gave the best training-period annual return among eleven candidates
    (skip 0% to 50%); "skip nothing" is allowed to win.
92. **"Riskiest 10%"** is ranked within the test period for the 2019-2025 set (as before) and within each announcement
    year for the pooled set.
93. **Day-1 ladder step.** Final model plus day-1 spread, p_mkt, p_mkt with carry and days to t1, same
    hyper-parameters, reported in its own section of the benchmark table and never compared as a like-for-like step
    with the announcement-date models.
94. **Monthly analysis kept as the fallback.** `src/price_analysis.py` and `src/backtest.py` still run on the Capital
    IQ monthly closes and now write to `results/monthly/`; the headline files `results/market_benchmark.md` and
    `results/backtest.md` come from the daily data.
95. **Model-free reference for the rule.** The model-versus-market rule beat hold-all, but the deals it keeps have wider
    spreads than average, and wide spreads earned more in this sample. A reference rule was therefore added after seeing
    that result: hold the same number of deals, chosen by the market-implied risk alone. The model's contribution is the
    rule minus that reference, and its interval includes zero.
96. **Bootstrap for the market comparison** covers the pooled and the within-year ROC AUC only (rank-based, fast). The
    within-year interval is shown because scores refitted every year (market logistic, stack) are not on one scale across
    years, which lowers their pooled AUC; `p_mkt` and the raw spread do not have that problem.

## Underwriting extension (time to closing, loss if it breaks, expected return)

97. **Time-to-closing target and sample.** Log calendar days from announcement to closing, completed core-sample deals;
    broken deals are dropped in the main specification because the arbitrageur needs time to closing given that the
    deal closes. A log-normal survival model with broken deals censored at cancellation is the robustness row.
98. **Duration grid.** The first grid (3 to 12 leaves, up to 400 trees) put the CV optimum on its edge, so it was
    widened to 6 to 24 leaves and 200 to 800 trees before any test result was produced. Selection criterion: CV mean
    absolute error on log days.
99. **Final duration model** = the feature set with the lowest CV error inside the training years, among base,
    extended, extended plus market context, and those plus LLM terms (with a has-text flag, so it applies to every
    deal). CV chose the LLM-terms variant in the time split and in all 13 origins.
100. **Management guidance.** The LLM-extracted field `expected_closing_months` had failed validation in the extended
    run, so the stated expected closing was read again with a dedicated step: sentences found by pattern, converted to
    a date window by Claude Sonnet, checked against the strongest model on 60 deals (97% within 7 days). Guidance in
    days = midpoint of the stated window, or the stated bound when only one is given. It is a benchmark; one separately
    labelled ladder step also uses it as an input.
101. **Loss if it breaks.** Return from the day-1 close to the close one (and five) trading days after the
    cancellation date, all considerations, unhedged. Forecasts are restricted to a constant, the fall-back benchmark,
    a one-variable fit, ridge and a 3-leaf boosted model. For the expected-return step, a forecast for year Y uses
    only breaks announced and cancelled before Y, and the forecast type is chosen by leave-one-year-out inside them.
102. **Day-1 signals (step P).** The two specifications, their inputs and all hyper-parameters were written into
    `src/day1_signals.py` before the out-of-time evaluation was run, and the evaluation was run once. Before that run,
    the signal distributions (not outcomes) were inspected and five signals were clipped to fixed limits because of
    obvious data errors (an acquirer return of +1,146%). Nothing was changed after seeing the results. Specifications
    tried: two.
103. **Which break probability the expected-return step uses.** As instructed: the day-1 model for a test year only
    where the cross-validation inside that origin's training years prefers it over `p_mkt`, otherwise the
    market-implied probability (through the market logistic, because raw `p_mkt` is not calibrated). The CV preferred
    the day-1 model in 7 of 13 origins and for the test period, although out of time it did not beat `p_mkt`; the rule
    was followed as written.
104. **P(topped)** comes from a three-variable logistic regression (day-1 spread, `p_mkt`, deal size) fitted on earlier
    years; the bump if topped is the mean extra return of earlier topped deals over their day-1 spread. Too few topped
    deals exist to model more.
105. **Expected time to resolution** uses the mean of the predicted log-normal for closing (exp(mu + sigma^2 / 2), sigma
    from out-of-fold residuals) and the mean time to a break and to a topping bid in earlier years.
106. **Terciles** are formed within each announcement year, for the test period and the pooled set alike.
107. **"Spread annualized with the median duration"** uses the training-period median by structure (tender offer or
    merger). With one global median it would rank deals exactly like the raw spread.
108. **Calendar-time portfolio: 10% position cap.** The first version gave each open deal a weight of 1 / (open deals).
    In the last weeks of the sample only one broken deal was still open, at 100% weight, which produced a 50% drawdown
    that has nothing to do with the strategy. Each deal now gets 1 / max(open deals, 10) and the rest is cash. The cap
    was added after seeing that drawdown and before comparing the portfolios.
109. **Spread-only references.** After the model ranking and its top half were seen to beat hold-all, the same
    constructions were run with the raw spread (top third, top half), and the model is judged by the difference to
    those, because the spread is free.
110. **Antitrust overlap.** Hoberg-Phillips TNIC-3 was downloaded from the authors' site (data to 2023; latest year
    before the announcement, at most three years back). Acquirer gvkey comes from the CRSP link through CCM. Missing
    pair features (private, sponsor and foreign buyers) are left missing for the tree models.
111. **Smoke test side effect fixed.** The synthetic smoke test overwrote `artifacts/final_tabular_shap.pkl`; the file
    now carries the run tag, and the real one was regenerated (refit predictions identical to the stored ones).
112. **Demo.** The underwriting row is shown for test-period deals from the real artefacts only. Market-implied
    probability, spread and expected annualized return are shown for all-cash deals only.

## Final polish

113. **Product time-to-closing model.** Where the announcement states an expected closing, the product uses model 6
    (which also reads it), otherwise the final model. The switch is decided per origin by the CV already stored for
    that origin (model 6 against model 4 on the deals with text, same folds); CV preferred model 6 in all 14 origins.
    No model was refitted for the point forecast. The spread of the forecast (used for the mean of the log-normal) is
    the residual standard deviation of model 6 on the calibration years.
114. **Expected-return rerun.** The ranking test and the calendar-time portfolio were rerun once with that one change.
    Old and new numbers are printed side by side in `results/expected_return.md`; the old run is kept as
    `results/expected_return_v1.json`. The tercile results improved slightly, the test-period calendar-time result
    got worse; both are reported and nothing was adjusted afterwards.
115. **Conformal interval.** Conformalized quantile regression with a time-ordered split: the 10% and 90% quantile
    models are refitted on the training years except the last three, the last three are the calibration set, and the
    interval is [q10 - Q, q90 + Q] with Q the ceil((n + 1) x 0.8) / n quantile of max(q10 - y, y - q90). Done per
    origin, separately for the final model (all deals) and for model 6 (deals with a stated closing).
116. **Synthetic demo values.** The synthetic fallback now shows the underwriting row with made-up values drawn from
    fixed distributions (time to closing around 45 days for tender offers and 120 for mergers, spread around 3%, loss
    around -25%); the expected annualized return is computed from them and the synthetic break score. The page states
    that the numbers are meaningless. The committed screenshots are taken from this fallback.
117. **README** cut to about 500 words as requested; everything removed from it is reachable through
    `results/README.md`.

## Export for the slide deck (code frozen)

118. **Read-only export.** `src/deck_numbers.py` writes `results/deck_numbers.json` from the cached per-deal files. It
    fits nothing and changes no result file; every number is checked against the published result files and the
    script stops on a mismatch. New in it: 95% intervals for the annual return of each third (same resampling as the
    report: announcement years for the pooled set, deals for the test period, 2,000 draws, seed 0, thirds fixed), the
    monthly values of the three calendar-time portfolios, and the product model's error by segment. The script is not
    part of `run_all.sh`.
119. **Counts made explicit.** 246 adverse breaks and 100 topped deals are counts among all 4,744 signed deals. The core
    sample of 4,468 deals (reverse mergers excluded) has 198 adverse breaks (158 training, 40 test) and 96 topped
    deals (79 and 17). The README and the project page named 246 next to the core sample; both now state 198. No
    result changed: every model and table already used the core sample.

## Final verification pass

120. **Fresh-clone test.** The public repo was cloned into an empty directory, a new Python 3.13 environment was built
    from `requirements.txt`, and `./run_all.sh synthetic` and the demo were run as the README describes. Both worked
    without changes. One display defect was found and fixed in `app/demo.py`: a conditional expression used as a
    statement made Streamlit print its own help text under the similar-deals table; it is now an if/else statement.
    No result, number or figure changed. The site title in `docs/_config.yml` now matches the page heading.
121. **Demo renamed to the project framing.** Page title "Deal underwriting"; main title "Underwriting an announced
    deal: break risk, time to closing and loss if it breaks". The underwriting row (time to closing, loss if it
    breaks, expected annualized return) now comes first, directly under the deal heading and above the break-risk row;
    the deal name and the synthetic-sample notice stay above it so the numbers are never shown without them. The
    three screenshots in `figures/` were retaken using `app/screenshots.py --force-synthetic`. No number changed.

## Outcomes, spread and loss on one common sample (code frozen)

122. **Read-only export.** `src/common_sample_outcomes.py` writes `results/common_sample_outcomes.md` and `.json`:
    outcome shares, the day-1 spread of closed deals, realised returns by outcome and three loss-to-spread ratios, all
    on the 1,205 priced all-cash deals of the ranking test (1,150 closed, 38 adverse breaks, 17 topped). The sample is
    the `priced` filter of `expected_return.py` with nothing added; the script stops if the counts or the rows differ
    from the ranking test. The same filter gives the two comparison samples (2019-2025: 623 deals; 2001-2025: 2,461).
    Spread and return are the fields of `crsp_prices.py`; the script first reproduces the published spread table in
    `results/crsp_prices.md`. Nothing was refitted, no existing result file changed, and the script is not part of
    `run_all.sh`.
123. **Ratios.** Loss over spread uses the return of adverse breaks (sign reversed) over the day-1 spread of closed
    deals from the same sample. Each ratio is computed from unrounded values; the calculation is also shown with the
    two-decimal inputs, which can differ in the second decimal.

## Slide notes follow the final deck

124. **SLIDE_NOTES.md rewritten** to follow the final deck slide by slide (9 main slides, a divider, appendix A1 to
    A5): headline, numbers with their result file, and the speaker bullets for each main slide, the two-minute video
    path, and a short section per appendix slide. The deck itself is not in the repository. Every number was checked
    against `results/`. Three values on slide 5 had come from rounding a three-decimal value a second time (-0.13 for
    -0.12498, +0.01 for +0.0046, -0.03 for -0.0248); they were corrected in the deck to -0.12, +0.00 and -0.02, and the
    deck now notes that a signed 0.00 shows on which side of zero a bound lies. The notes follow the corrected deck and
    keep the unrounded values from the result files; no result changed. README and project page: the recommendation
    now names sizing for the realized 28% average loss, and the page's "a spread of a few percent" now reads "about
    1.4% at the median", as on the deck.
125. **Sizing at 31%** (deck v9). The recommendation, word for word on the cover, slide 2 and the slide 9 headline, is
    now "Use the model for timing and ranking, the spread for break risk and a 31% loss for sizing". 31% is the mean
    loss of the 38 adverse breaks among the 1,205 ranked all-cash deals (31.21%, `results/common_sample_outcomes.md`),
    the sample the ranking and the expected annualized return use; 27.94% stays as the mean over all 184 adverse breaks
    (slide 7). SLIDE_NOTES.md, the README and the project page follow; no result changed.

## Final deck (vF)

126. **Final deck (vF).** The deck has 10 main pages, a divider, and A1 to A5 (16 pages in all); SLIDE_NOTES.md quotes
    it verbatim. The final wording is "median by deal type" and "a 31% loss estimate for sizing". The notes on pages 3
    and 9 state that the +1.67 gap is computed before rounding (12.26 - 10.58 shows 1.68; the unrounded thirds are
    12.2559 and 10.5825, `results/expected_return.json`). Market conditions shows -0.03 [-0.05 to -0.00] from the pooled
    value -0.028 [-0.054 to -0.002] (`results/pooled_oot.json`). Entries 124 and 125 describe intermediate versions of the
    deck; no result changed at any step.

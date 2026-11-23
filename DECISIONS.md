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

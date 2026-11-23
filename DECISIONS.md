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

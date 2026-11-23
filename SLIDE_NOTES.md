# Slide notes

Audience: the portfolio manager of a merger-arbitrage fund. Five slides, two minutes. Brackets are bootstrap 95%
intervals. "Test" = deals announced 2019-2025, models trained on 2001-2018. "Pooled" = deals announced 2013-2025, each
year scored by models refitted on earlier years only.

## Storyline in one sentence

Announcement-day data ranks deal-break risk only modestly, richer data and text do not improve it measurably, and the
market's own day-1 spread ranks breaks better than the model; the useful output is a cleaner definition of the risk and
a screen, not a trading signal.

## Five slides

**1. The decision and the asymmetry.**
- Buy the target at announcement: earn a few percent if the deal closes, lose far more if it breaks.
- Daily CRSP prices, all-cash deals bought at the day-1 close: completed deals returned a median +1.8%; the 22 adverse
  breaks of the test period lost 38% on average.
- 4,744 signed US public deals since 2001, 346 breaks (7.3%).
- Figure: `figures/break_rate_by_year.png`.

**2. Define the risk before modelling it.**
- 100 of the 346 breaks were topped by a higher bid: good news for an arbitrageur, bad as a "break" label.
- 276 "deals" are reverse mergers (de-SPACs), 275 of them after 2019 and 52 of the 109 test-period breaks.
- Primary target: adverse break (246 cases). Primary sample: 4,468 deals without reverse mergers.
- The vendor stores the final offer: for deals that were later bumped, the day-1 spread looked like 30% or more until
  the announcement-day offer was read from the press release.
- Source: `results/outcome_taxonomy.md`, `results/crsp_prices.md`.

**3. What the model can and cannot do.**
- Final model (LightGBM on deal, buyer and target features): AUC 0.633 [0.551, 0.717] in the test period, 0.609
  [0.538, 0.686] pooled (2,176 deals, 84 adverse breaks). Logistic regression on base terms: 0.606 and 0.590.
- More data did not help, also with twice the breaks. Pooled: extended features +0.025 [-0.025, +0.076], text +0.022
  [-0.014, +0.059], market context -0.028 [-0.054, -0.002]. Fine-tuned encoder: chance.
- It ranks on financial weakness and structure. It is blind to antitrust overlap: Aerojet Rocketdyne / Lockheed scored
  in the safest 8%.
- Figures: `figures/roc_pr_ladder.png`, `figures/ablation_groups.png`.

**4. Against the market.**
- One trading day after announcement, the market-implied break probability from the spread: AUC 0.643 [0.523, 0.755]
  on 623 priced all-cash test deals (22 adverse breaks); 0.721 [0.642, 0.800] pooled (1,205 deals, 38 adverse breaks).
- The model on the same deals: 0.632 and 0.599. Model minus market, pooled: -0.12 [-0.28, +0.01].
- Adding the model to the market: +0.020 [-0.081, +0.124] test, -0.028 [-0.194, +0.107] pooled.
- Figure: `figures/model_vs_market.png`.

**5. In dollars, and what to do.**
- Backtest, all-cash deals, daily marks, before costs: hold everything = 6.8% a year on deployed capital in the test
  period ($6.8m per $100m), 6.4% pooled.
- Skip the model's riskiest 10%: +0.3 points [-0.6, +1.4] test, +0.7 [-0.1, +1.4] pooled. Riskiest 20%: about zero.
- Model-versus-market rule (threshold fitted on earlier years): +4.3 [+1.6, +7.2] test, +2.4 [+0.3, +4.6] pooled. Most
  of it is a tilt to wide spreads: against a model-free rule with the same number of deals, +1.8 [-0.3, +4.4] and
  +0.8 [-1.0, +2.9].
- Recommendation: read the spread first; use the model as a checklist trigger where it disagrees with the market, not
  for sizing; invest next in what neither sees early (competitive overlap, regulator stance).
- Figure: `figures/backtest.png`.

## Backup slides

- Benchmark ladder, all four target and sample combinations, and the day-1 step: `results/benchmark_table.md`.
- Pooled out-of-time ladder with intervals by year, by-year table, embargo check: `results/pooled_oot.md`.
- Market benchmark detail, robustness panels, within-year AUC: `results/market_benchmark.md`.
- Backtest detail: plausible-spread panel, exit five days after cancellation, stock and mixed deals: `results/backtest.md`.
- Linking deals to CRSP, coverage by period and outcome: `results/crsp_link.md`.
- Price checks against the export and the Capital IQ monthly closes; which offer is known on day 1: `results/crsp_prices.md`.
- Monthly-price fallback (runs without WRDS): `results/monthly/`.
- Base run versus extended run: `results/base_run/benchmark_table.md`.
- Rolling-origin (two-year windows): `results/rolling_origin.md`, `figures/rolling_origin.png`.
- Calibration: `results/calibration.md`, `figures/calibration.png`. SHAP: `results/shap.md`.
- Segments: `results/segments.md`. Three-class view: `results/three_class.md`. Robustness: `results/robustness.md`.
- Six case studies: `results/case_studies.md`.
- Stylized impact grid on all test deals: `results/impact.md`, `figures/impact_sensitivity.png`.
- LLM extraction validation: `results/llm_extraction_validation.md`. Label audit: `results/label_audit.md`.
- Exploratory LLM zero-shot score and why it cannot be trusted: `results/llm_zero_shot.md`.
- Demo: `streamlit run app/demo.py` (screenshots of the synthetic fallback in `figures/demo_synthetic_*.png`).

## Two-minute pitch outline

- The trade: small gain if the deal closes, large loss if it breaks; the PM needs a break probability on day one.
- The data: 4,744 signed deals, SEC filings from the first week, CRSP daily prices, 346 breaks.
- First finding: a third of "breaks" are not the risk the PM cares about (higher bids, de-SPACs). Redefine the target.
- Second finding: on the cleaned target the model reaches AUC 0.63 in the test period and 0.61 over 13 out-of-time years.
  Real, weak, and not improved by richer features, text or LLM-read contract terms.
- Third finding: the market's day-1 spread ranks breaks better than the model (0.72 against 0.60 pooled), and the model
  adds nothing measurable on top of it.
- Fourth finding: skipping the model's riskiest deals adds 0.3 to 0.7 points a year with intervals that include zero.
  A model-versus-market rule looks better, but most of its gain is simply holding wide spreads.
- So what: a clean risk taxonomy and a second opinion on the spread, not a sizing model. Next dollar goes to antitrust
  overlap data, where the largest losses come from.
- What I would not trust: anything resting on the 22 to 38 priced adverse breaks, the model-versus-market rule, and the
  LLM zero-shot score.

# Slide notes

Audience: the portfolio manager of a merger-arbitrage fund. Five slides, two minutes. All numbers are from the test
period 2019-2025 (models trained on 2001-2018) unless stated. Brackets are bootstrap 95% intervals.

## Storyline in one sentence

Announcement-day data ranks deal-break risk only modestly, richer data and text do not improve it measurably, and it
does not demonstrably beat the market's own spread; the useful output is a cleaner definition of the risk and a
screen, not a trading signal.

## Five slides

**1. The decision and the asymmetry.**
- Buy the target at announcement: earn a few percent if the deal closes, lose far more if it breaks.
- In the backtest, completed cash deals returned a median +1.7%; the 16 priced adverse breaks lost 45% on average.
- 4,744 signed US public deals since 2001, 346 breaks (7.3%).
- Figure: `figures/break_rate_by_year.png`.

**2. Define the risk before modelling it.**
- 100 of the 346 breaks were topped by a higher bid: good news for an arbitrageur, bad as a "break" label.
- 276 "deals" are reverse mergers (de-SPACs), 275 of them after 2019 and 52 of the 109 test-period breaks.
- Primary target: adverse break (246 cases). Primary sample: 4,468 deals without reverse mergers.
- Consequence: the first-pass AUC of 0.62 to 0.70 was partly about deal type, not deal risk.
- Source: `results/outcome_taxonomy.md`; figure: `figures/break_rate_by_year.png` (three break-rate lines).

**3. What the model can and cannot do.**
- Final model (LightGBM on deal, buyer and target features): AUC 0.633 [0.551, 0.717]; the riskiest 20% of deals hold
  35% [20%, 50%] of adverse breaks. Logistic regression on base terms: 0.606. One rule: 0.528.
- Stable across 12 rolling origins: mean AUC 0.61 to 0.65, standard deviation about 0.08.
- It ranks on financial weakness and structure (weak target and acquirer prices, all-stock deals, size relative to the
  buyer, no reverse termination fee). It is blind to antitrust overlap: Aerojet Rocketdyne / Lockheed scored in the
  safest 8%.
- Figures: `figures/roc_pr_ladder.png`, `figures/shap_summary.png`.

**4. More data did not help.**
- Extended buyer and target features: +0.006 AUC [-0.059, +0.066]. Market context: -0.007. Text (TF-IDF, embeddings,
  LLM-extracted terms): +0.023 [-0.039, +0.080]. LLM-extracted terms alone: +0.000. Fine-tuned encoder: 0.517, chance.
- Why: announcements are written by the deal parties and read alike; the terms that differ are already in the table.
- Figure: `figures/ablation_groups.png`.

**5. Against the market, and in dollars.**
- Market-implied break probability from the spread: AUC 0.619 [0.458, 0.771] on 552 priced cash deals. Model: 0.599.
  Both together: 0.641; the model's increment is +0.027 [-0.096, +0.153].
- Backtest, all-cash deals, monthly marks, before costs: hold everything = 8.7% a year on deployed capital
  ($8.7m per $100m). Skip the model's riskiest 10%: +0.9 points [-0.3, +2.0]. Riskiest 20%: +0.5 [-1.5, +2.5].
- Recommendation: use the score as a checklist trigger for extra diligence on the top decile, do not size positions on
  it, and invest next in what the model cannot see (competitive overlap, regulator stance).
- Figures: `figures/backtest.png`, `figures/model_vs_market.png`.

## Backup slides

- Benchmark ladder, all four target and sample combinations: `results/benchmark_table.md`.
- Base run versus extended run: `results/base_run/benchmark_table.md` next to `results/benchmark_table.md`.
- Rolling-origin table and chart: `results/rolling_origin.md`, `figures/rolling_origin.png`.
- Calibration before and after Platt scaling: `results/calibration.md`, `figures/calibration.png`.
- SHAP table with plain readings: `results/shap.md`.
- Segments (consideration, size, period, buyer type): `results/segments.md`.
- Three-class view (closes, topped, adverse): `results/three_class.md`.
- Robustness (revised offers, missingness, leakage suspects, reverse mergers): `results/robustness.md`.
- Six case studies (two flagged, two missed, two false alarms): `results/case_studies.md`.
- Price coverage and checks; why only 16 priced adverse breaks: `results/price_coverage.md`.
- Backtest detail, including the plausible-spread panel and the model-versus-market rules: `results/backtest.md`.
- Stylized impact grid on all test deals: `results/impact.md`, `figures/impact_sensitivity.png`.
- LLM extraction validation (60 documents, field-level agreement): `results/llm_extraction_validation.md`.
- Label audit against EDGAR (3 of 76 contradicted): `results/label_audit.md`.
- Exploratory LLM zero-shot score, AUC 0.82, and why it cannot be trusted (92% of anonymised targets named):
  `results/llm_zero_shot.md`.
- Text coverage by year: `results/text_coverage.md`. Lift chart: `figures/lift_chart.png`.
- Demo: `streamlit run app/demo.py` (screenshots of the synthetic fallback in `figures/demo_synthetic_*.png`).

## Two-minute pitch outline

- The trade: small gain if the deal closes, large loss if it breaks; the PM needs a break probability on day one.
- The data: 4,744 signed deals, SEC filings from the first week, monthly prices, 346 breaks.
- First finding: a third of "breaks" are not the risk the PM cares about (higher bids, de-SPACs). Redefine the target.
- Second finding: on the cleaned target the model reaches AUC 0.63. Real, weak, stable over time.
- Third finding: richer features, text and LLM-read contract terms add nothing measurable. The announcement does not
  contain the reason deals later break.
- Fourth finding: the market's spread is about as informative as the model; together they are slightly better, not
  provably so. In the backtest, skipping flagged deals adds 0.5 to 0.9 points a year with intervals that include zero.
- So what: a screen for the riskiest decile and a clean risk taxonomy, not a sizing model. Next dollar goes to
  antitrust overlap data, where the largest losses come from.
- What I would not trust: anything resting on the 16 priced adverse breaks, and the LLM zero-shot score.

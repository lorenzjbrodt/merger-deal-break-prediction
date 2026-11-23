# Slide notes (base run)

Audience: the portfolio manager of a merger-arbitrage fund. Five slides, two minutes. An extended version replaces this
file once the extended run is complete.

## Key messages

1. **The decision.** On announcement day the PM sizes a position that earns about 3% if the deal closes and loses about 20%
   if it breaks. 7.3% of 4,744 signed US public deals since 2001 broke (346 breaks). Figure: `figures/break_rate_by_year.png`.
2. **Deal terms carry a modest signal.** LightGBM on announcement-day tabular fields: test AUC 0.62 [0.57, 0.67] on all
   1,358 test deals (base rate 0.50, one rule 0.55, logistic 0.56). On deals with filings: 0.70, and the riskiest 20% hold
   46% of the breaks. Figure: `figures/roc_pr_ladder.png`, `figures/lift_chart.png`.
3. **Press-release text does not add to the terms.** Embeddings: -0.037 AUC [-0.086, +0.010] versus tabular; TF-IDF:
   +0.014 [-0.026, +0.060]. Text alone reaches 0.67, so it repeats what the terms already say.
4. **In dollars (stylized, no prices).** Hold everything: $1.15m per $100m per year at a 3% spread and 20% break loss.
   Skip the riskiest 20%: +$0.24m (all deals) to +$0.70m (deals with filings). Gone if flagged deals trade about 1.3 points
   wider. Figure: `figures/impact_sensitivity.png`.
5. **What I would not trust yet.** Half of the test-period breaks are de-SPAC reverse mergers; missing Capital IQ fields
   predict the label; "break" includes deals topped by a higher bid; no market prices. The extended run addresses each.

## Two-minute pitch outline

- The trade and the asymmetry: small gain, large loss.
- Data: 4,744 deals, 346 breaks, announcement-day terms plus the SEC filings of the first week.
- Result: terms rank deals moderately well; text adds nothing beyond terms.
- Use: a screen for the riskiest fifth of deals, not a stand-alone signal.
- Caveats and next step: price data, a cleaner definition of an adverse break, de-SPACs out.

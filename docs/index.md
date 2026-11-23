# Underwriting merger deals at announcement

MIT 15.071, Project 1 (Predictive AI). Solo project by Lorenz J. Brodt.
Code and full results: <https://github.com/lorenzjbrodt/merger-deal-break-prediction>

## Problem

A merger-arbitrage fund buys the target's shares after a deal is announced. If the deal closes, it earns a spread of a few
percent. If the deal breaks, it loses many times that. For every deal the portfolio manager underwrites three numbers:
the probability of an adverse break, the time to closing, and the loss if it breaks. This project asks how well each can
be predicted from what is known at announcement, and whether ranking deals by the expected annualized return they imply
makes money out of sample.

## Approach

- **Deals.** 4,744 signed US public M&A deals announced 2001-2025 from a Capital IQ screen (licensed; not redistributed).
- **Cleaning the question.** Of 346 breaks, 100 were lost to a higher competing bid, which is good news for an
  arbitrageur. The target is the **adverse break** (246 cases). 276 reverse mergers (mostly de-SPACs) are excluded. The
  vendor stores the final offer, so the announcement-day offer is read from the press release.
- **Inputs.** Deal terms, buyer and target financials, market conditions; the press release and merger agreement from SEC
  EDGAR, read by an LLM into 29 validated contract terms and the parties' stated expected closing; daily share prices
  from CRSP; product similarity of buyer and target (Hoberg-Phillips).
- **Evaluation.** Train on 2001-2018, test on 2019-2025. A second, larger set pools 2013-2025, with every year scored by
  models refitted on earlier years only. Bootstrap confidence intervals throughout. Positions are entered at the close
  one trading day after the announcement; returns are before costs.

![Deals and break rates by year](figures/break_rate_by_year.png)

## Results

### 1. Break probability: weak, and the market does it better

| Adverse breaks, ROC AUC [95% CI] | Test 2019-2025 | Pooled out-of-time 2013-2025 |
|---|---|---|
| Logistic regression, base deal terms | 0.606 [0.523, 0.694] | 0.590 [0.525, 0.651] |
| Final model (LightGBM, extended features) | 0.633 [0.551, 0.717] | 0.609 [0.538, 0.686] |
| + text (keywords, embeddings, LLM terms) | 0.646 [0.559, 0.730] | 0.599 [0.540, 0.663] |
| Market-implied probability, day 1 (priced all-cash deals) | 0.643 [0.523, 0.755] | 0.721 [0.642, 0.800] |
| Final model on the same deals | 0.632 [0.519, 0.734] | 0.599 [0.476, 0.702] |

Test: 1,083 deals, 40 adverse breaks (623 priced all-cash, 22 breaks). Pooled: 2,176 deals, 84 adverse breaks (1,205
priced, 38 breaks). Extended features, text, LLM-extracted terms, product overlap and ten further day-1 market signals
each change the AUC by an amount whose interval includes zero or is negative. The model keys on financial weakness and
deal structure and is blind to what blocks clean cash deals.

![Model versus market](figures/model_vs_market.png)

### 2. Time to closing: predictable

The median completed deal takes 107 days. The final model's mean absolute error is 52.7 days [48.1, 57.4] in the test
period against 63.5 for the median by structure (tender offer or merger), with an R2 of 0.54 on log days. LLM-extracted
contract terms cut the error by 1.6 days [-2.6, -0.6]; text embeddings add nothing. The model is as accurate as the
closing date the parties state themselves (46.9 against 46.4 days), and a model that also reads that guidance beats it by
6.6 days [-9.5, -3.9].

![Time to closing](figures/duration.png)

### 3. Loss if it breaks: large and barely predictable

From the day-1 close to the day after the cancellation, adverse breaks lost 28% on average (184 deals), with a tenth of
them losing more than 73%. A fall back to the pre-announcement price would imply only 11%. No forecast beats the
historical average when every year is predicted from the others.

![Loss if the deal breaks](figures/loss_given_break.png)

### 4. In dollars: ranking by expected annualized return

Combining the three numbers per deal and sorting each year's all-cash deals into thirds (pooled 2013-2025, 1,205
deals): the top third earned 11.7% a year, the middle 3.4%, the bottom 2.3%; top minus bottom +9.3 points
[+5.5, +13.5]. The raw spread alone gives +8.4 [+5.0, +12.7]. The model's top third beats the spread's top third by 1.9
points [+0.4, +3.3]. In a daily calendar-time portfolio the top half by the model returned 9.3% a year with a Sharpe
ratio of 0.80, the top half by raw spread 8.7% and 0.77, all deals 7.2% and 0.72; model minus spread +0.6 points
[-0.9, +2.0].

![Ranking test and calendar-time portfolio](figures/expected_return.png)

### 5. What to take from it

Annualize spreads with the duration model, read break risk off the spread, and do not size positions on the break
model. Most of the ranking profit is a risk premium: the top third holds 25 of the 38 adverse breaks.

## Limitations

- **Few events.** 40 adverse breaks in the test period (22 priced all-cash) and 84 in the pooled set (38 priced).
  Intervals are wide, and "no measurable gain" does not mean "no gain". Pooled intervals resample only 13 years.
- **Before costs.** No transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1
  close.
- **Licensed vendor data.** Results can be rebuilt only with Capital IQ and WRDS access. The vendor fills fewer fields
  for cancelled deals, stores final terms, and 15% of breaks have no stated reason. An audit of 80 deal statuses against
  SEC filings found 3 of 76 checkable ones contradicted.
- **LLM caveat.** An exploratory zero-shot LLM score ranked deals far better (AUC 0.82), but the same LLM could name 92%
  of the anonymised targets, so it may simply know how the deals ended. It is not used anywhere.

## Repository

<https://github.com/lorenzjbrodt/merger-deal-break-prediction>: code, result tables, figures, a synthetic sample to
run the pipeline without a licence, a decision log and a local Streamlit demo. No row-level Capital IQ, CRSP or
Compustat data is published.

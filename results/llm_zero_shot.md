# Exploratory: LLM zero-shot break-risk score from anonymised text

**Not part of the main models.** A general LLM (Claude Sonnet, no fitting on this sample) reads the anonymised press release of each test-period deal and returns a probability that the deal will not be completed for a reason other than a superior bid.

**Caveat.** Anonymisation replaces company names, tickers and years, but size, industry, price and wording can still identify well-known deals, and the LLM may know how they ended from its training data. Treat any skill as an upper bound on what the text itself contains. Deals announced in 2025 are shown separately because they are closest to (or beyond) the model's knowledge cut-off; the count of breaks there is small.

| Test deals (core sample, with text) | Deals | Adverse breaks | LLM zero-shot AUC [95% CI] | Final model AUC on the same deals |
|---|---:|---:|---|---|
| All test years 2019-2025 | 975 | 35 | 0.822 [0.757, 0.879] | 0.620 [0.522, 0.712] |
| 2019-2023 | 677 | 27 | 0.814 [0.731, 0.886] | 0.663 [0.552, 0.761] |
| 2024-2025 | 298 | 8 | 0.856 [0.767, 0.939] | 0.494 [0.289, 0.704] |
| 2025 only | 167 | 5 | 0.869 [0.747, 0.958] | 0.393 [0.123, 0.707] |

Mean stated probability: 9.5% (observed adverse-break rate 3.6%); mean for deals that broke 17.9%, for the rest 9.1%. Rank correlation with the final model's score: 0.40.

## Leakage probe: can the LLM name the companies?

The same anonymised texts were given to the same LLM with the question "which companies are TARGET and BUYER?". A deal counts as identified when the guessed target name matches the real one on its leading distinctive word.

| Group | Deals | Adverse breaks | LLM zero-shot AUC [95% CI] | Final model AUC |
|---|---:|---:|---|---|
| Target identified from the anonymised text | 896 | 30 | 0.817 [0.742, 0.888] | 0.593 [0.487, 0.697] |
| Target not identified | 79 | 5 | 0.843 [0.686, 0.974] | 0.770 [0.586, 0.935] |

Identified: 92% of deals overall; 86% of the deals that broke and 92% of the rest. If the LLM names broken deals more often than completed ones, part of its ranking skill can come from memory of the outcome, not from the text. Skill among deals it could not name is the cleaner estimate, and even that is not proof: not naming a company is not the same as not recognising the situation.


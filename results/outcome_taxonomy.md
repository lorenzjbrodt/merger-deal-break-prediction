# Outcome taxonomy for cancelled deals

Source: Capital IQ deal notes (Deal Resolution, Transaction Comments, Pre-Deal Situation, Related Transactions). These fields are written after the fact. They are used for labels and analysis only and never enter a feature matrix.

Cancelled deals in the sample: 346. Classified by an LLM reading of the notes, checked against keyword rules, and validated by re-reading a random sample (below).

| Reason | All | Train 2001-2018 | Test 2019-2025 | Share of breaks |
|---|---:|---:|---:|---:|
| Superior competing bid accepted | 100 | 79 | 21 | 29% |
| Regulatory or antitrust block | 57 | 40 | 17 | 16% |
| Financing failure | 8 | 6 | 2 | 2% |
| Shareholder vote failed / insufficient tenders | 31 | 22 | 9 | 9% |
| Buyer walked or MAC dispute | 21 | 15 | 6 | 6% |
| Mutual termination or other stated reason | 77 | 47 | 30 | 22% |
| Unknown (no reason stated) | 52 | 28 | 24 | 15% |

## Targets

| Target | Positives (all) | Train | Test | Rate |
|---|---:|---:|---:|---:|
| (a) Any break | 346 | 237 | 109 | 7.3% |
| (b) Adverse break (break excluding superior competing bids) | 246 | 158 | 88 | 5.2% |

Three-class view: closed 4398, adverse_break 246, topped 100.

## Rule check

Keyword rules applied to the Deal Resolution text and the last 1,200 characters of the comments agree with the LLM label on 37% of cancelled deals across the seven classes and on 82% for the binary question that defines the adverse-break target (competing bid or not). The rules are a coarse cross-check (first keyword match wins); the LLM label is the one used.

| LLM label \ rule label | competing_bid | regulatory | financing | shareholder_vote | buyer_walked_mac | mutual_other | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| competing_bid | 32 | 13 | 1 | 0 | 3 | 2 | 29 |
| regulatory | 0 | 34 | 2 | 0 | 4 | 2 | 15 |
| financing | 0 | 0 | 6 | 0 | 1 | 0 | 1 |
| shareholder_vote | 3 | 4 | 1 | 7 | 2 | 1 | 13 |
| buyer_walked_mac | 2 | 6 | 3 | 0 | 5 | 1 | 4 |
| mutual_other | 9 | 26 | 2 | 0 | 8 | 16 | 26 |
| unknown | 2 | 16 | 6 | 0 | 10 | 1 | 27 |

## Rival-deal rule

102 cancelled deals have another buyer's majority deal for the same target in the export, announced between the deal's announcement and 45 days after its cancellation. 20 of them had been labelled `unknown` or `mutual_other` from the notes and were relabelled as competing bids. All counts above are after this rule.


## Validation by re-reading

A random sample of 50 cancelled deals (seed 7) was re-read by hand: the Deal Resolution text and the end of the Transaction Comments, compared with the LLM label. In all 50 cases the LLM label was a faithful reading of the notes (it returned `unknown` whenever the notes gave no reason). Checked against what actually happened in the well-known deals of the sample, 3 of the 50 notes-based labels were wrong for the question that matters here (topped by a rival bid or not): three targets were lost to rival bidders, but their notes only say that the deal was cancelled or mutually terminated, so they were labelled `unknown` or `mutual_other`. This motivated the rival-deal rule above, which uses the export itself and corrects all three. After the rule, none of the 50 re-read cases is known to be on the wrong side of the adverse-break definition. Remaining weakness: `unknown` and `mutual_other` still hold deals whose true reason was regulatory or a buyer walking away (two such cases were recognised in the sample). That blurs the seven-class breakdown but not the adverse-break target, because all of those classes count as adverse.


## Deals whose exported terms may reflect a revised offer

413 of 4744 deals (8.7%) have notes that mention a revised, raised, lowered or amended offer after the announcement (106 cancelled, 307 closed). For these deals the exported offer price, premium and deal value can be the final terms, not the announcement terms. The sensitivity of the results to excluding them is reported in `results/robustness.md`.


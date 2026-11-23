# Label audit against EDGAR

40 Cancelled and 40 Closed deals with a target CIK were drawn at random (seed 11) and their Capital IQ status was checked against the target's SEC filing index: Form 15 / Form 25 and 8-K completion items for closed deals; 8-K Item 1.02 and continued periodic reporting for cancelled deals. The rules are in the header of `src/label_audit.py`. The check is automatic and uses filing metadata, not the text of the filings; 8-K item codes exist only from late 2004.

| Capital IQ status | Confirmed | Consistent | Unclear | No evidence | Contradicted |
|---|---:|---:|---:|---:|---:|
| Cancelled | 34 | 1 | 0 | 3 | 2 |
| Closed | 38 | 0 | 0 | 1 | 1 |

Reasons recorded:

- Cancelled, confirmed: 8-K Item 1.02 at the cancellation date (28)
- Cancelled, confirmed: target still files periodic reports 6+ months later (6)
- Cancelled, consistent: 8-K at the cancellation date (no item codes available) (1)
- Cancelled, contradicted: Form 15 / 25 within 150 days of the cancellation and no later periodic report (2)
- Cancelled, no evidence: no filings in the window (3)
- Closed, confirmed: Form 15 / 25 after closing (38)
- Closed, contradicted: target still files periodic reports 15+ months after the closing date (1)
- Closed, no evidence: no filings in the window (1)

## Estimated label noise

- Cancelled: 2 of 37 checkable deals contradicted = 5.4% (95% Wilson interval 1.5% to 17.7%); 3 of 40 could not be checked.
- Closed: 1 of 39 checkable deals contradicted = 2.6% (95% Wilson interval 0.5% to 13.2%); 1 of 40 could not be checked.

A contradiction means the filing record does not fit the status, not that the status is proven wrong: a target with public debt can keep filing after it is acquired, and a cancelled target can be acquired by someone else soon after. The upper ends of the intervals are the relevant figures for how much label noise the models may be facing. The reason for a break (the outcome taxonomy) is a separate source of noise; see `results/outcome_taxonomy.md`.


# Management guidance on closing: extraction and validation

Source: the announcement press release (EDGAR). Sentences that state an expected closing time are found by pattern (3,472 of 4,180 deals with text), then read by Claude (sonnet) into a date window. The reader sees the announcement date and the sentences only.

- Deals with usable guidance: 3,373 (81% of deals with text).
- Form: window 2830, by a date 538, other 5; fiscal-period wording in 127 deals.
- Guidance in days from announcement (midpoint of the window, or the stated bound): median 120, 10th to 90th percentile 58 to 228.

## Validation against the strongest model (strong, 60 deals, independent reading)

- Agreement on whether guidance is stated: 100%.
- Where both give a window (59 deals): identical within 7 days in 97%, within 31 days in 98%; median absolute difference 0 days.

The guidance is the benchmark for the time-to-closing models. One separately labelled ladder step also uses it as an input; no other model sees it.


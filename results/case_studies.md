# Six case studies from the test period

Selection is mechanical (see `src/case_studies.py`): among core-sample test deals with an announcement text, the two
adverse breaks with the highest model score ("flagged"), the two adverse breaks with the lowest score ("missed"), and the
two completed deals with the highest score ("false alarms"). Model = the final tabular model, trained on 2001-2018.
"Percentile" is the deal's rank among test-period scores (100 = riskiest). Deal facts are taken from the announcement
filed with the SEC; what happened afterwards is public record.

## Flagged breaks

**1. Diversified Healthcare Trust / Office Properties Income Trust (announced 11 April 2023). Percentile 98.**
An all-share merger of two REITs managed by the same external adviser. The announcement itself says the deal "ensures
immediate debt covenant compliance" for the target, and both sets of shareholders had to vote. What the model saw: a
target trading far below its 52-week high and below book value, an acquirer whose own shares had roughly halved, and no
cash in the consideration. Large shareholders of the target opposed the price, and the parties terminated by mutual
agreement on 1 September 2023 before the vote. The model had no way to know about shareholder opposition; it flagged
the deal because two weak share prices and an all-stock structure have preceded breaks before.

**2. Columbia Care (now The Cannabist Company) / Cresco Labs (announced 23 March 2022). Percentile 97.**
A cross-border, all-stock combination of two cannabis operators, with an expected closing eight months out and
divestitures needed in several states. What the model saw: a deal almost as large as the acquirer, both share prices
well below their highs and a falling acquirer. The required asset sales could not be completed and the parties
terminated on 30 July 2023, sixteen months after announcement. A correct flag for a reason the model captures only
indirectly (size relative to the buyer, weak equity), not through any regulatory variable.

## Missed breaks

**3. Aerojet Rocketdyne / Lockheed Martin (announced 20 December 2020). Percentile 8.**
A $5.0 billion all-cash purchase at a premium of about 33% by a large, profitable, investment-grade buyer for whom the
deal was small. Every tabular signal reads as safe: cash, a strong acquirer, low relative size. The risk was vertical
antitrust: the target was the last independent US supplier of missile propulsion. The FTC sued in January 2022 and
Lockheed terminated in February 2022. The extracted terms list the HSR and foreign approvals, but so do most deals;
nothing in the announcement distinguishes a filing formality from a real competition problem. This is the model's
main blind spot: it has no measure of market overlap or supplier position.

**4. LENSAR / Alcon (announced 24 March 2025). Percentile 8.**
A cash-plus-contingent-value-right offer by a much larger strategic buyer for a small medical-device maker whose shares
had tripled in the prior year. Again the tabular picture is benign. The announcement lists US antitrust, foreign
antitrust and foreign-investment approvals. The deal is recorded as cancelled in March 2026, a year after
announcement, with no reason classified. As with case 3, a horizontal overlap in a narrow product market is invisible
to the features.

## False alarms

**5. Strongbridge Biopharma / Xeris Pharmaceuticals (announced 24 May 2021). Percentile 100.**
A stock-plus-CVR acquisition of an Irish-domiciled rare-disease company under the Irish Takeover Rules, needing votes on
both sides. What the model saw: no cash, a target at about half its 52-week high, an acquirer down sharply over the year,
and no recorded termination fee (fee arrangements under the Irish rules differ from US practice, so the field is
empty). The deal closed on schedule on 5 October 2021. The score reflects a pattern, small all-stock deals between
two weak share prices, that breaks more often than average but still closes most of the time.

**6. Pacific Office Properties Trust / Shidler Equities (announced 25 January 2019). Percentile 100.**
A going-private purchase of a thinly traded REIT by its controlling holder, disclosed in an 8-K with no press release.
What the model saw: a target share price a small fraction of its 52-week high, several empty Capital IQ fields (no
termination fee, no price-to-book, no buyer legal adviser recorded) and a negative one-year return. It closed two
months later. A buyer who already controls the company rarely fails to close; the model has a flag for minority
buy-ins but too few examples to learn that it should override the price signals. Part of this score also comes from
missing fields, which is a data artefact and not a property of the deal.

## What the cases show

- The model ranks on **financial weakness and structure**: weak target and acquirer share prices, all-stock
  consideration, deal size relative to the buyer, missing protections. That finds breaks caused by fragile buyers
  and unconvinced shareholders (cases 1 and 2).
- It misses **antitrust breaks of clean, well-financed cash deals** (cases 3 and 4), which is where an arbitrageur's
  largest losses come from. Listing required approvals does not help, because nearly every deal lists them.
- Its false alarms are small, illiquid situations (cases 5 and 6) where the same weakness signals are present but the
  buyer is committed. Missing Capital IQ fields add to these scores.

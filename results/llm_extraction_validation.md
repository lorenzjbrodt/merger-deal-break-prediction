# LLM term extraction: validation against the strongest model

60 randomly sampled documents were extracted twice: by the small production model (run `haiku`) and by the strongest model (run `strong`). Agreement = identical value, both-null counting as agreement; numeric fields agree when within 10% or one unit. A field is kept as a model feature when overall agreement is at least 85% and agreement on documents where either model found a value is at least 60%.

| Field | Agreement | Both not stated | Agreement when either stated | Kept | Note |
|---|---:|---:|---:|---|---|
| buyer_type | 97% | 0% | 97% | yes |  |
| financing_condition | 100% | 77% | 100% | yes |  |
| committed_financing_mentioned | 98% | 60% | 96% | yes |  |
| regulatory_approvals | 60% | 10% | 56% | dropped | exact match of the whole list; mean Jaccard 0.82. Judged per approval below |
| reg_HSR | 98% | 27% | 98% | yes | approval listed: yes / no |
| reg_foreign_antitrust | 88% | 63% | 68% | yes | approval listed: yes / no |
| reg_CFIUS | 100% | 93% | 100% | yes | approval listed: yes / no |
| reg_FCC | 100% | 92% | 100% | yes | approval listed: yes / no |
| reg_banking | 98% | 82% | 91% | yes | approval listed: yes / no |
| reg_insurance | 100% | 95% | 100% | yes | approval listed: yes / no |
| reg_FERC_or_state_utility | 100% | 92% | 100% | yes | approval listed: yes / no |
| reg_other | 70% | 57% | 31% | dropped | approval listed: yes / no |
| target_termination_fee_usd_m | 97% | 43% | 94% | yes | numeric: equal within 10% or 1 unit |
| reverse_termination_fee_usd_m | 98% | 80% | 92% | yes | numeric: equal within 10% or 1 unit |
| go_shop | 88% | 55% | 74% | yes |  |
| go_shop_days | 98% | 90% | 83% | yes | numeric: equal within 10% or 1 unit |
| structure | 98% | 2% | 98% | yes |  |
| minimum_tender_condition | 100% | 83% | 100% | yes |  |
| target_shareholder_vote_required | 97% | 13% | 96% | yes |  |
| acquirer_shareholder_vote_required | 95% | 73% | 81% | yes |  |
| expected_closing_months | 82% | 18% | 78% | dropped | numeric: equal within 10% or 1 unit |
| outside_date_months | 97% | 73% | 88% | yes | numeric: equal within 10% or 1 unit |
| voting_support_agreements | 98% | 42% | 97% | yes |  |
| support_agreement_pct | 97% | 85% | 78% | yes | numeric: equal within 10% or 1 unit |
| collar | 100% | 90% | 100% | yes |  |
| cvr | 100% | 95% | 100% | yes |  |
| antitrust_covenant | 80% | 37% | 68% | dropped |  |
| specific_performance | 85% | 28% | 79% | yes |  |
| cross_border_acquirer | 90% | 5% | 89% | yes |  |
| acquirer_country | 62% | 5% | 60% | dropped |  |
| stated_premium_pct | 98% | 62% | 96% | yes | numeric: equal within 10% or 1 unit |

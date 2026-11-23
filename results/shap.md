# SHAP reading of the tabular model

Model: 3x. LightGBM + extended deal, buyer and target features, trained on 2001-2018 (core sample, adverse-break target). TreeSHAP values on the test deals, averaged over the five seeds. Mean |SHAP| is in log-odds units. Direction is the rank correlation between the feature value and its SHAP value on the test deals: positive means higher values push the predicted break risk up.

| Rank | Feature | Mean abs. SHAP | Share of total | Direction | Plain reading |
|---:|---|---:|---:|---|---|
| 1 | `log_tgt_revenue` | 0.131 | 7% | higher = riskier (rho +0.57) | More / higher target revenue goes with higher predicted break risk |
| 2 | `buy_fee_pct_value` | 0.129 | 7% | higher = safer (rho -0.64) | Less / lower buyer (reverse) termination fee as % of deal value goes with higher predicted break risk |
| 3 | `pb_multiple` | 0.088 | 5% | higher = safer (rho -0.63) | Less / lower price / book multiple goes with higher predicted break risk |
| 4 | `cash_share_of_value` | 0.085 | 5% | higher = safer (rho -0.83) | Less / lower cash consideration as a share of deal value goes with higher predicted break risk |
| 5 | `pct_52w_high` | 0.080 | 4% | higher = safer (rho -0.88) | Less / lower target price as % of its 52-week high goes with higher predicted break risk |
| 6 | `sell_fee_pct_value` | 0.075 | 4% | higher = riskier (rho +0.73) | More / higher target termination fee as % of deal value goes with higher predicted break risk |
| 7 | `acq_pct_52w_high` | 0.068 | 4% | higher = safer (rho -0.89) | Less / lower acquirer price as % of its 52-week high goes with higher predicted break risk |
| 8 | `tgt_cash_to_assets` | 0.066 | 4% | higher = safer (rho -0.78) | Less / lower tgt cash to assets goes with higher predicted break risk |
| 9 | `log_offer_per_share` | 0.058 | 3% | higher = safer (rho -0.80) | Less / lower offer price per share goes with higher predicted break risk |
| 10 | `ev_ebitda` | 0.055 | 3% | higher = safer (rho -0.66) | Less / lower EV / EBITDA multiple goes with higher predicted break risk |
| 11 | `rel_size_deal_to_acq` | 0.055 | 3% | higher = riskier (rho +0.81) | More / higher deal value relative to acquirer market cap goes with higher predicted break risk |
| 12 | `target_ltm_return` | 0.053 | 3% | higher = safer (rho -0.95) | Less / lower target's 12-month return before the deal goes with higher predicted break risk |
| 13 | `log_buy_fee_usd` | 0.051 | 3% | higher = safer (rho -0.71) | Less / lower buyer termination fee (USD) goes with higher predicted break risk |
| 14 | `feat_going_private` | 0.045 | 2% | higher = riskier (rho +0.69) | Deals with this flag (going-private transaction) get a higher predicted break risk |
| 15 | `acq_roa` | 0.043 | 2% | higher = safer (rho -0.57) | Less / lower acq roa goes with higher predicted break risk |

These are associations the model uses, not causal effects. Features with many missing values also carry a missingness signal (see `results/robustness.md`). Figure: `figures/shap_summary.png`.


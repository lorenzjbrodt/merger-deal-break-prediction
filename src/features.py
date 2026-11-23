"""Feature construction.

Everything here is either a stateless row-wise transform of fields known on the
announcement date, or an sklearn transformer that is fitted inside the training fold
by train_eval.py. No label, status or outcome date is read in this module.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that must never be used as features (outcome or outcome-derived / near-empty).
FORBIDDEN = [
    "status", "closed_date", "cancelled_date", "target_spread_usd", "target_spread_pct",
    "sell_side_term_fee_pct", "buy_side_term_fee_pct", "label",
]
# Fields Capital IQ can update after the announcement: excluded from main models, one ablation.
LEAKAGE_SUSPECTS = ["deal_responses", "deal_attitude"]

CONDITION_FLAGS = ["Legal", "Management", "Shareholder Approvals", "Financial", "Other"]
CONSIDERATION_FLAGS = {
    "cons_cash": "cash",
    "cons_common_equity": "common equity",
    "cons_combinations": "combinations",
    "cons_elections": "elections",
    "cons_rights_warrants_options": "rights / warrants / options",
    "cons_preferred": "preferred",
    "cons_debt": "debt",
    "cons_earnout_contingent": "contingent|earn",
}

NUMERIC = [
    "log_deal_value", "log_implied_equity_value", "log_implied_ev", "ev_ebitda", "log_offer_per_share",
    "cash_pct", "stock_pct", "pct_sought", "premium_1d", "premium_1w", "premium_1m",
    "target_ltm_return", "pct_52w_high", "log_target_mktcap", "value_to_mktcap",
    "buyer_count", "buyer_listed", "solicited", "n_conditions", "conditions_missing",
    "days_signed_before_announcement", "signed_after_announcement", "month_sin", "month_cos",
] + [f"cond_{c.lower().replace(' ', '_')}" for c in CONDITION_FLAGS] + list(CONSIDERATION_FLAGS)
YEAR = ["year"]
CATEGORICAL = ["consideration_type", "special_committee_cat", "sic2"]
SUSPECT_FEATURES = ["resp_approval", "resp_rejection", "resp_no_response", "resp_missing",
                    "attitude_hostile", "attitude_friendly_to_hostile"]


def _slog(x: pd.Series) -> pd.Series:
    return np.log1p(x.clip(lower=0))


def build_tabular(d: pd.DataFrame) -> pd.DataFrame:
    """Row-wise tabular features. Winsorising uses fixed, label-free bounds."""
    X = pd.DataFrame(index=d.index)
    X["log_deal_value"] = _slog(d["deal_value"])
    X["log_implied_equity_value"] = _slog(d["implied_equity_value"])
    X["log_implied_ev"] = _slog(d["implied_ev"])
    X["ev_ebitda"] = d["ev_ebitda"].clip(0, 100)
    X["log_offer_per_share"] = _slog(d["offer_per_share"]).clip(upper=np.log1p(1e4))
    # Capital IQ leaves the complementary share blank for all-cash / all-stock deals.
    cash, stock = d["cash_pct"].astype(float), d["stock_pct"].astype(float).clip(0, 100)
    cash = cash.where(cash.notna(), (100 - stock).where(stock.notna()))
    stock = stock.where(stock.notna(), (100 - cash).where(cash.notna()))
    X["cash_pct"], X["stock_pct"] = cash, stock
    X["pct_sought"] = d["pct_sought"].clip(50, 100)
    for c in ["premium_1d", "premium_1w", "premium_1m"]:
        X[c] = d[c].clip(-50, 200)
    X["target_ltm_return"] = d["target_ltm_return"].clip(-100, 300)
    X["pct_52w_high"] = d["pct_52w_high"].clip(0, 105)
    X["log_target_mktcap"] = _slog(d["target_mktcap"])
    X["value_to_mktcap"] = (d["deal_value"] / d["target_mktcap"].replace(0, np.nan)).clip(0, 10)

    buyers = d["buyers"].fillna("").astype(str)
    X["buyer_count"] = buyers.map(lambda s: len([b for b in s.split(";") if b.strip()])).clip(upper=6)
    X["buyer_listed"] = buyers.str.contains(r"\([A-Za-z]+:[A-Za-z0-9.\-]+\)").astype(int)
    X["solicited"] = (d["deal_approach"].fillna("") == "Solicited").astype(int)

    conds = d["deal_conditions"].fillna("").astype(str)
    parts = conds.map(lambda s: {p.strip() for p in s.split(";") if p.strip()})
    for c in CONDITION_FLAGS:
        X[f"cond_{c.lower().replace(' ', '_')}"] = parts.map(lambda p, c=c: int(c in p))
    X["n_conditions"] = parts.map(len)
    X["conditions_missing"] = d["deal_conditions"].isna().astype(int)

    cons = d["consideration"].fillna("").astype(str).str.lower()
    for name, pat in CONSIDERATION_FLAGS.items():
        X[name] = cons.str.contains(pat).astype(int)
    has_cash = cons.str.contains("cash")
    has_stock = cons.str.contains("common equity|preferred")
    has_mix = cons.str.contains("combinations|elections")
    ctype = np.where(has_mix | (has_cash & has_stock), "mixed",
                     np.where(has_cash, "cash", np.where(has_stock, "stock", "other")))
    X["consideration_type"] = ctype

    X["special_committee_cat"] = d["special_committee"].fillna("missing").astype(str)
    sic = d["sic"]
    X["sic2"] = (sic // 100).map(lambda v: f"{int(v):02d}" if pd.notna(v) else "missing")
    X["year"] = d["announced_date"].dt.year.astype(float)
    month = d["announced_date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Days between definitive agreement and announcement. An agreement signed AFTER the
    # announcement is not observable on the announcement date, so only its sign is kept
    # (the PM knows whether a deal is already signed) and the gap is clipped at zero.
    gap = (d["definitive_date"] - d["announced_date"]).dt.days
    X["days_signed_before_announcement"] = (-gap).clip(0, 60)
    X["signed_after_announcement"] = (gap > 0).astype(int)

    # Leakage suspects (ablation only).
    resp = d["deal_responses"].fillna("missing").astype(str)
    X["resp_approval"] = (resp == "Approval").astype(int)
    X["resp_rejection"] = (resp == "Rejection").astype(int)
    X["resp_no_response"] = (resp == "No Response").astype(int)
    X["resp_missing"] = (resp == "missing").astype(int)
    att = d["deal_attitude"].fillna("").astype(str)
    X["attitude_hostile"] = (att == "Hostile").astype(int)
    X["attitude_friendly_to_hostile"] = (att == "Friendly to Hostile").astype(int)
    return X


def cash_stock_segment(X: pd.DataFrame) -> pd.Series:
    return X["consideration_type"].map({"cash": "cash", "stock": "stock"}).fillna("mixed/other")


# ------------------------------------------------------------------ keyword features
KEYWORDS = {
    "kw_financing_condition": r"financing (condition|contingenc)|subject to (the )?(receipt of |obtaining )?financing|conditioned (up)?on (the )?(receipt of |obtaining )?financing",
    "kw_no_financing_condition": r"not (subject to|conditioned (up)?on|contingent (up)?on) (any |a )?financing|no financing (condition|contingenc)",
    "kw_committed_financing": r"(debt|equity|committed|fully committed) (financing )?commitment|committed financing|commitment letter",
    "kw_antitrust": r"antitrust|anti-trust|competition law",
    "kw_hsr": r"hart[\s\-]*scott[\s\-]*rodino|\bhsr\b",
    "kw_second_request": r"second request",
    "kw_cfius": r"\bcfius\b|committee on foreign investment",
    "kw_go_shop": r"go[\s\-]shop",
    "kw_reverse_termination_fee": r"reverse (termination|break[\s\-]?up) fee|parent termination fee|buyer termination fee",
    "kw_termination_fee": r"termination fee|break[\s\-]?up fee",
    "kw_regulatory_approval": r"regulatory (approval|clearance|consent)|approval of (the )?(federal reserve|fcc|ferc|occ|fdic)|(governmental|government) approvals?",
    "kw_shareholder_vote": r"(stockholder|shareholder)s?['’]? (approval|vote|meeting)|approval (of|by) (the |its )?(holders|stockholders|shareholders)|vote of (the )?(stockholders|shareholders)",
    "kw_tender_offer": r"tender offer|exchange offer",
    "kw_unanimous_board": r"unanimously (approved|recommend)",
    "kw_voting_agreement": r"voting agreements?|support agreements?|tender and support",
    "kw_material_adverse": r"material adverse (effect|change)",
    "kw_private_equity": r"private equity|financial sponsor|investment funds? (affiliated|managed|advised)",
    "kw_management_rollover": r"roll[\s\-]?over|management[\s\-]led|going[\s\-]private",
}
_KW_RE = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}
KEYWORD_FEATURES = list(KEYWORDS) + ["log_press_len", "log_agreement_len", "has_agreement"]


def keyword_features(text: pd.DataFrame) -> pd.DataFrame:
    """Binary keyword flags over press release + agreement, plus document lengths.
    Rows without any text get NaN flags (LightGBM treats them as missing)."""
    press = text["press_release"].fillna("")
    agr = text["agreement"].fillna("")
    both = press + "\n" + agr
    K = pd.DataFrame(index=text.index)
    has_any = (press.str.len() + agr.str.len()) > 0
    for k, rx in _KW_RE.items():
        K[k] = both.map(lambda s, rx=rx: float(bool(rx.search(s)))).where(has_any)
    K["log_press_len"] = np.log1p(press.str.len()).where(has_any)
    K["log_agreement_len"] = np.log1p(agr.str.len()).where(has_any)
    K["has_agreement"] = (agr.str.len() > 0).astype(float).where(has_any)
    return K


# ------------------------------------------------------------------ in-fold preprocessors
def logistic_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Median impute + missing indicators + scale; one-hot with rare levels pooled."""
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def tree_preprocessor(numeric: list[str], categorical: list[str], impute: bool = False) -> ColumnTransformer:
    """Pass numerics through (NaN kept for LightGBM) and one-hot the categoricals in-fold."""
    num = SimpleImputer(strategy="median") if impute else "passthrough"
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             verbose_feature_names_out=False)


# =====================================================================================
# Extended export (v2): buyer, target financials, structure flags, fees, advisors
# =====================================================================================
# Capital IQ "Transaction Secondary Features" kept as announcement-time structure flags.
STRUCTURE_FLAGS = {
    "feat_tender_offer": "Tender Offer", "feat_cash_merger": "Cash Merger", "feat_stock_merger": "Stock Merger",
    "feat_cross_border": "Cross-Border", "feat_going_private": "Going Private Transaction",
    "feat_lbo": "Leveraged Buy Out (LBO)", "feat_go_shop": "Go Shop Provision",
    "feat_minority_increasing": "Minority Shareholder Increasing Ownership Stake",
    "feat_collar": "Collar Consideration Structure", "feat_merger_of_equals": "Merger of Equals",
    "feat_management_participated": "Management Participated", "feat_management_buyout": "Management Buyout",
    "feat_equity_reinvestment": "Equity Reinvestment", "feat_earnout": "Earnout Payments",
    "feat_bankruptcy_sale": "Bankruptcy Sale", "feat_reverse_merger": "Reverse Merger",
    "feat_individual_investors": "Individual Investors Participated",
}
# Tags that are applied when (or because) a deal completes: never used as features.
# On the training years "Squeeze Out / Short Form Merger" has 428 deals and zero breaks,
# "Platform" 42 deals and zero breaks.
OUTCOME_DRIVEN_TAGS = ["Squeeze Out / Short Form Merger", "Platform", "Secondary LBO", "Successor Issuer Merger"]

GICS_FROM_SIC = [  # (low, high, sector) - approximate mapping of the target's primary SIC to the buyer sector names
    (1000, 1099, "Materials"), (1200, 1399, "Energy"), (1400, 1499, "Materials"), (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"),
    (2500, 2599, "Consumer Discretionary"), (2600, 2699, "Materials"), (2700, 2799, "Communication Services"),
    (2800, 2829, "Materials"), (2830, 2836, "Health Care"), (2840, 2844, "Consumer Staples"), (2845, 2899, "Materials"),
    (2900, 2999, "Energy"), (3000, 3099, "Materials"), (3100, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"),
    (3400, 3569, "Industrials"), (3570, 3579, "Information Technology"), (3580, 3629, "Industrials"),
    (3630, 3659, "Consumer Discretionary"), (3660, 3699, "Information Technology"), (3700, 3719, "Consumer Discretionary"),
    (3720, 3799, "Industrials"), (3800, 3839, "Information Technology"), (3840, 3859, "Health Care"),
    (3860, 3899, "Information Technology"), (3900, 3999, "Consumer Discretionary"), (4000, 4599, "Industrials"),
    (4600, 4699, "Energy"), (4700, 4799, "Industrials"), (4800, 4899, "Communication Services"), (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"), (5200, 5399, "Consumer Discretionary"), (5400, 5499, "Consumer Staples"),
    (5500, 5899, "Consumer Discretionary"), (5900, 5911, "Consumer Discretionary"), (5912, 5912, "Consumer Staples"),
    (5913, 5999, "Consumer Discretionary"), (6000, 6499, "Financials"), (6500, 6599, "Real Estate"),
    (6700, 6797, "Financials"), (6798, 6798, "Real Estate"), (6799, 6799, "Financials"),
    (7000, 7299, "Consumer Discretionary"), (7300, 7369, "Industrials"), (7370, 7379, "Information Technology"),
    (7380, 7399, "Industrials"), (7500, 7799, "Consumer Discretionary"), (7800, 7849, "Communication Services"),
    (7900, 7999, "Consumer Discretionary"), (8000, 8099, "Health Care"), (8100, 8299, "Consumer Discretionary"),
    (8300, 8399, "Health Care"), (8700, 8730, "Industrials"), (8731, 8734, "Health Care"), (8735, 8999, "Industrials"),
]


def sic_sector(sic) -> str:
    if pd.isna(sic):
        return "missing"
    s = int(sic)
    for lo, hi, name in GICS_FROM_SIC:
        if lo <= s <= hi:
            return name
    return "other"


def _ratio(a: pd.Series, b: pd.Series, lo: float, hi: float) -> pd.Series:
    return (a / b.where(b > 0)).clip(lo, hi)


def _split_set(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(lambda v: {p.strip() for p in v.split(";") if p.strip()})


def _advisors(s: pd.Series, role: str) -> pd.Series:
    """List of advisor names with the given role, e.g. role='Financial Advisor'."""
    tag = f"({role})"
    return s.fillna("").astype(str).map(lambda v: [a.strip()[: -len(tag)].strip() for a in v.split(";") if a.strip().endswith(tag)])


def build_extended(d: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features from the extended export. All fields are stated as of announcement,
    except that Capital IQ's buyer 'Company Type' / 'Buyer Type' are CURRENT classifications."""
    E = pd.DataFrame(index=d.index)
    # ---- structure flags and fees (deal terms)
    sec = _split_set(d["secondary_features"])
    for name, tag in STRUCTURE_FLAGS.items():
        E[name] = sec.map(lambda s, tag=tag: int(any(x.startswith(tag) for x in s)))
    E["log_sell_fee_usd"] = _slog(d["sell_side_term_fee_usd"])
    E["sell_fee_pct_value"] = (100 * _ratio(d["sell_side_term_fee_usd"], d["deal_value"], 0, 0.15))
    E["log_buy_fee_usd"] = _slog(d["buy_side_term_fee_usd"])
    E["buy_fee_pct_value"] = (100 * _ratio(d["buy_side_term_fee_usd"], d["deal_value"], 0, 0.25))
    E["has_buy_side_fee"] = d["buy_side_term_fee_usd"].notna().astype(int)
    E["cash_share_of_value"] = _ratio(d["total_cash_consideration"], d["deal_value"], 0, 1.5)
    E["net_assumed_liab_to_value"] = (d["net_assumed_liabilities"] / d["deal_value"].where(d["deal_value"] > 0)).clip(-1, 2)
    for side, col in (("target", "advisors_target"), ("buyer", "advisors_buyers")):
        E[f"n_fin_adv_{side}"] = _advisors(d[col], "Financial Advisor").map(len).clip(upper=6)
        E[f"n_legal_adv_{side}"] = _advisors(d[col], "Legal Advisor").map(len).clip(upper=6)
    # ---- premium / price
    E["pct_52w_low"] = d["pct_52w_low"].clip(100, 1000)
    # ---- target financials
    E["log_tgt_revenue"] = _slog(d["tgt_revenue"])
    E["log_tgt_assets"] = _slog(d["tgt_assets"])
    E["tgt_ebitda_margin"] = (d["tgt_ebitda"] / d["tgt_revenue"].where(d["tgt_revenue"] > 0)).clip(-2, 1)
    E["tgt_net_margin"] = (d["tgt_net_income"] / d["tgt_revenue"].where(d["tgt_revenue"] > 0)).clip(-3, 1)
    E["tgt_roa"] = (d["tgt_net_income"] / d["tgt_assets"].where(d["tgt_assets"] > 0)).clip(-1, 0.5)
    E["tgt_leverage"] = _ratio(d["tgt_debt"], d["tgt_assets"], 0, 2)
    E["tgt_net_debt_to_ebitda"] = ((d["tgt_debt"] - d["tgt_cash"]) / d["tgt_ebitda"].where(d["tgt_ebitda"] > 0)).clip(-10, 20)
    E["tgt_cash_to_assets"] = _ratio(d["tgt_cash"], d["tgt_assets"], 0, 1)
    E["tgt_equity_to_assets"] = (d["tgt_equity"] / d["tgt_assets"].where(d["tgt_assets"] > 0)).clip(-1, 1)
    E["tgt_loss_making"] = (d["tgt_net_income"] < 0).astype(float).where(d["tgt_net_income"].notna())
    E["ev_revenue"] = d["ev_revenue"].clip(0, 50)
    E["pe_multiple"] = d["pe_multiple"].clip(0, 200)
    E["pb_multiple"] = d["pb_multiple"].clip(0, 50)
    # ---- buyer
    bt = d["buyer_type"].fillna("").astype(str)
    ct = d["buyer_company_type"].fillna("").astype(str)
    E["buyer_financial"] = bt.str.contains("Financial Buyer").astype(float).where(d["buyer_type"].notna())
    E["buyer_public_company"] = ct.str.contains("Public Company").astype(float).where(d["buyer_company_type"].notna())
    E["buyer_private_company"] = ct.str.contains("Private Company").astype(float).where(d["buyer_company_type"].notna())
    E["buyer_investment_firm"] = ct.str.contains("Investment Firm|Investment Arm").astype(float).where(d["buyer_company_type"].notna())
    bsec = d["buyer_sector"].fillna("").astype(str).map(lambda v: v.split(";")[0].strip())
    bsec = bsec.map(lambda v: v[v.rfind("(") + 1: -1] if v.endswith(")") and "(" in v else v)   # "TPG Capital (Financials)" -> "Financials"
    E["buyer_sector_cat"] = bsec.where(bsec != "", "missing")
    tsec = d["sic"].map(sic_sector)
    E["target_sector_cat"] = tsec
    E["same_sector"] = (E["buyer_sector_cat"] == tsec).astype(float).where((E["buyer_sector_cat"] != "missing") & (tsec != "missing"))
    bind = d["buyer_industry"].fillna("missing").astype(str).map(lambda v: v.split(";")[0].strip())
    E["buyer_industry_cat"] = bind.map(lambda v: v[v.rfind("(") + 1: -1] if v.endswith(")") and "(" in v else v)
    E["log_acq_mktcap"] = _slog(d["acq_mktcap"])
    E["acq_has_mktcap"] = d["acq_mktcap"].notna().astype(int)
    E["rel_size_deal_to_acq"] = _ratio(d["deal_value"], d["acq_mktcap"], 0, 10)
    E["acq_ltm_return"] = d["acq_ltm_return"].clip(-100, 300)
    E["acq_pct_52w_high"] = d["acq_pct_52w_high"].clip(0, 105)
    E["acq_leverage"] = _ratio(d["acq_debt"], d["acq_assets"], 0, 2)
    E["log_acq_cash"] = _slog(d["acq_cash"])
    E["acq_cash_to_deal_value"] = _ratio(d["acq_cash"], d["deal_value"], 0, 20)
    E["log_acq_revenue"] = _slog(d["acq_revenue"])
    E["acq_roa"] = (d["acq_net_income"] / d["acq_assets"].where(d["acq_assets"] > 0)).clip(-1, 0.5)
    E["acq_net_debt_to_ebitda"] = ((d["acq_debt"] - d["acq_cash"]) / d["acq_ebitda"].where(d["acq_ebitda"] > 0)).clip(-10, 20)
    return E


class TopAdvisorFlags:
    """Top-advisor flags from advisor frequencies in the TRAINING rows only (fit in-fold)."""
    SPECS = [("advisors_target", "Financial Advisor", "top_fin_adv_target"), ("advisors_buyers", "Financial Advisor", "top_fin_adv_buyer"),
             ("advisors_target", "Legal Advisor", "top_legal_adv_target"), ("advisors_buyers", "Legal Advisor", "top_legal_adv_buyer")]
    NAMES = [s[2] for s in SPECS]

    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.top: dict[str, set] = {}

    def fit(self, d: pd.DataFrame):
        for col, role, name in self.SPECS:
            counts = _advisors(d[col], role).explode().dropna().value_counts()
            self.top[name] = set(counts.head(self.top_n).index)
        return self

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(d), len(self.SPECS)))
        for j, (col, role, name) in enumerate(self.SPECS):
            out[:, j] = _advisors(d[col], role).map(lambda L, t=self.top[name]: float(any(a in t for a in L))).to_numpy()
        return out


# ---- feature groups (numeric, categorical) used for the ladder and the group ablation
EXT_DEAL_TERMS = list(STRUCTURE_FLAGS) + ["log_sell_fee_usd", "sell_fee_pct_value", "log_buy_fee_usd", "buy_fee_pct_value",
                                           "has_buy_side_fee", "cash_share_of_value", "net_assumed_liab_to_value",
                                           "n_fin_adv_target", "n_legal_adv_target", "n_fin_adv_buyer", "n_legal_adv_buyer"]
EXT_FINANCIALS = ["log_tgt_revenue", "log_tgt_assets", "tgt_ebitda_margin", "tgt_net_margin", "tgt_roa", "tgt_leverage",
                  "tgt_net_debt_to_ebitda", "tgt_cash_to_assets", "tgt_equity_to_assets", "tgt_loss_making", "ev_revenue",
                  "pe_multiple", "pb_multiple", "buyer_financial", "buyer_public_company", "buyer_private_company",
                  "buyer_investment_firm", "same_sector", "log_acq_mktcap", "acq_has_mktcap", "rel_size_deal_to_acq",
                  "acq_ltm_return", "acq_pct_52w_high", "acq_leverage", "log_acq_cash", "acq_cash_to_deal_value",
                  "log_acq_revenue", "acq_roa", "acq_net_debt_to_ebitda"]
_PRICE = ["premium_1d", "premium_1w", "premium_1m", "target_ltm_return", "pct_52w_high", "log_target_mktcap"]
_COND = [f"cond_{c.lower().replace(' ', '_')}" for c in CONDITION_FLAGS] + ["n_conditions", "conditions_missing"]
_BASE_FIN = ["ev_ebitda", "buyer_listed"]
BASE_DEAL_TERMS = [c for c in NUMERIC if c not in _PRICE + _COND + _BASE_FIN]

GROUPS = {
    "deal_terms": (BASE_DEAL_TERMS + EXT_DEAL_TERMS, ["consideration_type", "special_committee_cat", "sic2"]),
    "premium_price": (_PRICE + ["pct_52w_low"], []),
    "conditions": (_COND, []),
    "financials": (_BASE_FIN + EXT_FINANCIALS, ["buyer_sector_cat", "buyer_industry_cat"]),
}
GROUP_LABELS = {
    "deal_terms": "Deal terms and structure", "premium_price": "Premium and price", "conditions": "Conditions",
    "financials": "Buyer and target financials", "market_context": "Market context", "advisors": "Top-advisor flags",
    "tfidf": "TF-IDF text", "emb": "Frozen embeddings", "kw": "Keyword flags", "llm_terms": "LLM-extracted terms",
    "ft_score": "Fine-tuned encoder score", "market_spread": "Market spread (dated t1)",
}

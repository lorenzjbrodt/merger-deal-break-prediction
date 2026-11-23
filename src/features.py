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

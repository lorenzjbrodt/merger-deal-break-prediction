"""Load the Capital IQ transaction screening export, apply the sample filters and
write the cleaned deal table to data/interim/deals.parquet.

Usage:
    python src/prepare.py                      # real export at data/raw/ciq_transactions_v2.xls
    python src/prepare.py --input data/sample/synthetic_ciq_export.csv --tag synthetic

Nothing written by this script may be committed: the deal table contains licensed
Capital IQ fields. Only aggregate counts go to results/.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Exact header names in the Capital IQ export -> short internal names.
COLS = {
    "All Transactions Announced Date": "all_announced_date",
    "Target/Issuer": "target",
    "Exchange:Ticker": "exchange_ticker",
    "Transaction Types": "transaction_types",
    "Transaction Status": "status",
    "Total Transaction Value ($USDmm, Historical rate)": "deal_value",
    "Buyers/Investors": "buyers",
    "Sellers": "sellers",
    "CIQ Transaction ID": "deal_id",
    "Geographic Locations [Target/Issuer]": "geo",
    "Implied Equity Value ($USDmm, Historical rate)": "implied_equity_value",
    "Implied Enterprise Value ($USDmm, Historical rate)": "implied_ev",
    "Implied Enterprise Value/EBITDA (x)": "ev_ebitda",
    "Percent Sought (%)": "pct_sought",
    "Offer per Share ($USD, Historical rate)": "offer_per_share",
    "Cash % of Consideration (%)": "cash_pct",
    "Stock % of Consideration (%)": "stock_pct",
    "Target Spread ($USD, Historical rate)": "target_spread_usd",
    "Target Spread (%)": "target_spread_pct",
    "Sell-Side Termination Fee (%)": "sell_side_term_fee_pct",
    "Buy-Side Termination Fee (%)": "buy_side_term_fee_pct",
    "Target Stock Premium - 1 Day Prior (%)": "premium_1d",
    "Target Stock Premium - 1 Week Prior (%)": "premium_1w",
    "Target Stock Premium - 1 Month Prior (%)": "premium_1m",
    "Target LTM Return % (%)": "target_ltm_return",
    "Target % of 52-Week High (%)": "pct_52w_high",
    "Target Market Cap 1-Day Prior ($USDmm, Historical rate)": "target_mktcap",
    "M&A Announced Date": "announced_date",
    "M&A Definitive Agreement Date": "definitive_date",
    "M&A Closed Date": "closed_date",
    "M&A Cancelled Date": "cancelled_date",
    "Special Committee": "special_committee",
    "Deal Attitude": "deal_attitude",
    "Deal Approach": "deal_approach",
    "Deal Responses": "deal_responses",
    "Deal Conditions": "deal_conditions",
    "Consideration Offered": "consideration",
    "CIK [Target/Issuer]": "cik",
    "Excel Company ID [Target/Issuer]": "company_id",
    "Security Tickers [Target/Issuer]": "security_tickers",
    "SIC Codes (Primary Code Only) [Target/Issuer]": "sic",
}
# Additional columns of the extended (v2) export. Optional: the base export lacks them.
EXTRA_COLS = {
    "Net Assumed Liabilities ($USDmm, Historical rate)": "net_assumed_liabilities",
    "Implied Equity Value/LTM Net Income (x)": "pe_multiple",
    "Implied Equity Value/Book Value (x)": "pb_multiple",
    "Implied Enterprise Value/Revenues (x)": "ev_revenue",
    "Total Cash ($USDmm, Historical rate)": "total_cash_consideration",
    "Total Stock ($USDmm, Historical rate)": "total_stock_consideration",
    "Total Earnout/Contingent Payments ($USDmm, Historical rate)": "earnout_consideration",
    "Sell-Side Termination Fee ($USDmm, Historical rate)": "sell_side_term_fee_usd",
    "Buy-Side Termination Fee ($USDmm, Historical rate)": "buy_side_term_fee_usd",
    "Acquirer LTM Return % (%)": "acq_ltm_return",
    "Target % of 52-Week Low (%)": "pct_52w_low",
    "Acquirer % of 52-Week High (%)": "acq_pct_52w_high",
    "Acquirer Market Cap 1-Day Prior ($USDmm, Historical rate)": "acq_mktcap",
    "Transaction Comments": "transaction_comments",
    "Pre-Deal Situation": "pre_deal_situation",
    "Deal Resolution": "deal_resolution",
    "Advisors To Target": "advisors_target",
    "Advisors To Buyers": "advisors_buyers",
    "Transaction Primary Features": "primary_features",
    "Transaction Secondary Features": "secondary_features",
    "Majority / Minority Stakes": "majority_minority",
    "Change of Control": "change_of_control",
    "Related Transactions": "related_transactions",
    "Target/Issuer LTM Financials - Total Revenue (at Announcement) ($USDmm, Historical rate)": "tgt_revenue",
    "Target/Issuer LTM Financials - EBITDA (at Announcement) ($USDmm, Historical rate)": "tgt_ebitda",
    "Target/Issuer LTM Financials - Net Income (at Announcement) ($USDmm, Historical rate)": "tgt_net_income",
    "Target/Issuer LTM Financials - Total Common Equity (at Announcement) ($USDmm, Historical rate)": "tgt_equity",
    "Target/Issuer LTM Financials - Total Debt (at Announcement) ($USDmm, Historical rate)": "tgt_debt",
    "Target/Issuer LTM Financials - Total Cash & ST Investments (at Announcement) ($USDmm, Historical rate)": "tgt_cash",
    "Target/Issuer LTM Financials - Total Assets (at Announcement) ($USDmm, Historical rate)": "tgt_assets",
    "Acquirer LTM Financials - Total Revenue (at Announcement) ($USDmm, Historical rate)": "acq_revenue",
    "Acquirer LTM Financials - EBITDA (at Announcement) ($USDmm, Historical rate)": "acq_ebitda",
    "Acquirer LTM Financials - Net Income (at Announcement) ($USDmm, Historical rate)": "acq_net_income",
    "Acquirer LTM Financials - Total Debt (at Announcement) ($USDmm, Historical rate)": "acq_debt",
    "Acquirer LTM Financials - Total Cash & ST Investments (at Announcement) ($USDmm, Historical rate)": "acq_cash",
    "Acquirer LTM Financials - Total Assets (at Announcement) ($USDmm, Historical rate)": "acq_assets",
    "Primary Industry [Buyers/Investors]": "buyer_industry",
    "Primary Sector [Buyers/Investors]": "buyer_sector",
    "Buyer Type [Buyers/Investors]": "buyer_type",
    "Company Type [Buyers/Investors]": "buyer_company_type",
}
EXTRA_TEXT = ["transaction_comments", "pre_deal_situation", "deal_resolution", "advisors_target", "advisors_buyers",
              "primary_features", "secondary_features", "majority_minority", "change_of_control", "related_transactions",
              "buyer_industry", "buyer_sector", "buyer_type", "buyer_company_type"]
EXTRA_NUM = [v for v in EXTRA_COLS.values() if v not in EXTRA_TEXT]
DATE_COLS = ["all_announced_date", "announced_date", "definitive_date", "closed_date", "cancelled_date"]
NUM_COLS = [
    "deal_value", "implied_equity_value", "implied_ev", "ev_ebitda", "pct_sought", "offer_per_share",
    "cash_pct", "stock_pct", "target_spread_usd", "target_spread_pct", "sell_side_term_fee_pct",
    "buy_side_term_fee_pct", "premium_1d", "premium_1w", "premium_1m", "target_ltm_return",
    "pct_52w_high", "target_mktcap", "sic",
]
SPAC_RE = re.compile(
    r"acquisition\s+corp|acquisition\s+co\b|acquisition\s+holdings|merger\s+corp|blank\s+check", re.I
)
TICKER_RE = re.compile(r"\(([A-Za-z]+:[A-Za-z0-9.\-]+)\)")


def load_export(path: Path) -> pd.DataFrame:
    """Read the raw export (.xls from Capital IQ, or .csv with the same header)."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, na_values=["-"], dtype=str, keep_default_na=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(path, sheet_name="Screening", header=7, engine="xlrd",
                               na_values=["-"], dtype=object)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Export is missing expected columns: {missing}")
    extra = {k: v for k, v in EXTRA_COLS.items() if k in df.columns}
    df = df[list(COLS) + list(extra)].rename(columns={**COLS, **extra})
    # Drop the footer rows ("*Denotes proprietary information." etc.): no transaction id.
    df = df[df["deal_id"].notna()].copy()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = NUM_COLS + [c for c in EXTRA_NUM if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c not in DATE_COLS and c not in num_cols:
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].replace({"-": pd.NA, "": pd.NA})
    return df.reset_index(drop=True)


def clean_target_name(name: str) -> str:
    """'Helen of Troy Limited (NasdaqGS:HELE)' -> 'Helen of Troy Limited'."""
    if not isinstance(name, str):
        return ""
    return TICKER_RE.sub("", name).strip()


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log: list[dict] = []

    def step(name: str, mask: pd.Series, d: pd.DataFrame) -> pd.DataFrame:
        out = d[mask.fillna(False).astype(bool)].copy()
        log.append({
            "step": name, "rows": int(len(out)),
            "breaks": int((out["status"] == "Cancelled").sum()),
        })
        return out

    log.append({"step": "raw export", "rows": int(len(df)),
                "breaks": int((df["status"] == "Cancelled").sum())})
    d = step("status in {Closed, Cancelled}", df["status"].isin(["Closed", "Cancelled"]), df)
    d = step("definitive agreement date present", d["definitive_date"].notna(), d)
    d = step("percent sought >= 50", d["pct_sought"] >= 50, d)
    tname = d["target"].map(clean_target_name).str.lower()
    buyers = d["buyers"].fillna("").str.lower()
    self_deal = pd.Series([bool(t) and t in b for t, b in zip(tname, buyers)], index=d.index)
    d = step("target not among buyers", ~self_deal, d)
    yr = d["announced_date"].dt.year
    d = step("announcement year 2001-2025", (yr >= 2001) & (yr <= 2025), d)
    spac = (d["sic"] == 6770) | d["target"].fillna("").str.contains(SPAC_RE)
    d = step("drop SPACs", ~spac, d)
    return d, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/raw/ciq_transactions_v2.xls"),
                    help="extended export (v2); the base export data/raw/ciq_transactions.xls also works")
    ap.add_argument("--tag", default="real", help="'real' or 'synthetic'; controls output paths")
    args = ap.parse_args()

    suffix = "" if args.tag == "real" else f"_{args.tag}"
    df = load_export(Path(args.input))
    d, log = apply_filters(df)

    d["label"] = (d["status"] == "Cancelled").astype(int)
    d["target_clean"] = d["target"].map(clean_target_name)
    d["year"] = d["announced_date"].dt.year.astype(int)
    d = d.sort_values(["announced_date", "deal_id"]).reset_index(drop=True)
    if d["deal_id"].duplicated().any():
        n = int(d["deal_id"].duplicated().sum())
        d = d.drop_duplicates("deal_id").reset_index(drop=True)
        log.append({"step": f"drop {n} duplicate transaction ids", "rows": int(len(d)),
                    "breaks": int(d["label"].sum())})

    out_dir = ROOT / "data/interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_dir / f"deals{suffix}.parquet", index=False)

    n, b = len(d), int(d["label"].sum())
    summary = {
        "input": Path(args.input).name, "tag": args.tag, "n_deals": n, "n_breaks": b,
        "break_rate": round(b / n, 4) if n else None, "filter_log": log,
        "expected": {"n_deals": 4723, "n_breaks": 344},
    }
    res = ROOT / ("results" if args.tag == "real" else f"results_{args.tag}")
    res.mkdir(exist_ok=True)
    (res / "sample_filters.json").write_text(json.dumps(summary, indent=2))

    print(f"{'step':45s} {'rows':>7s} {'breaks':>7s}")
    for r in log:
        print(f"{r['step']:45s} {r['rows']:7d} {r['breaks']:7d}")
    print(f"\nFinal sample: {n} deals, {b} breaks ({b / max(n, 1):.1%})")
    if args.tag == "real":
        dn, db = abs(n - 4723) / 4723, abs(b - 344) / 344
        flag = "OK" if max(dn, db) <= 0.05 else "OUTSIDE 5% TOLERANCE"
        print(f"Sanity check vs expected 4,723 / 344: deals {dn:+.1%}, breaks {db:+.1%} -> {flag}")


if __name__ == "__main__":
    main()

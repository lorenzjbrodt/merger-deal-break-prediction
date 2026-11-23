"""Load the Capital IQ transaction screening export, apply the sample filters and
write the cleaned deal table to data/interim/deals.parquet.

Usage:
    python src/prepare.py                      # real export at data/raw/ciq_transactions.xls
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
    df = df[list(COLS)].rename(columns=COLS)
    # Drop the footer rows ("*Denotes proprietary information." etc.): no transaction id.
    df = df[df["deal_id"].notna()].copy()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c not in DATE_COLS and c not in NUM_COLS:
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
    ap.add_argument("--input", default=str(ROOT / "data/raw/ciq_transactions.xls"))
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

"""Generate data/sample/synthetic_ciq_export.csv: 200 SYNTHETIC rows with the exact column
header of the (extended, 81-column) Capital IQ export, so the pipeline can be run end to end
without a licence.

Every value is drawn from simple made-up distributions with a fixed seed. No Capital IQ
record is used. Company names are placeholders ("Target 017 Inc."). Results computed on
this file are meaningless; it exists only as a smoke test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prepare import COLS, EXTRA_COLS

ROOT = Path(__file__).resolve().parents[1]
N = 200


def main() -> None:
    rng = np.random.default_rng(15071)
    rows = []
    for i in range(N):
        year = int(rng.integers(2001, 2026))
        ann = pd.Timestamp(year=year, month=int(rng.integers(1, 13)), day=int(rng.integers(1, 28)))
        stock_deal = rng.random() < 0.3
        solicited = rng.random() < 0.2
        listed_buyer = rng.random() < 0.6
        mktcap = float(np.exp(rng.normal(6.5, 1.3)))
        prem = float(rng.normal(30, 20))
        value = mktcap * (1 + prem / 100) * float(rng.uniform(0.9, 1.4))
        # synthetic break probability: higher than reality so a 200-row sample has enough events
        p = 0.12 + 0.12 * stock_deal + 0.08 * solicited + 0.06 * (not listed_buyer)
        broke = rng.random() < p
        days = int(rng.integers(40, 300))
        signed = ann - pd.Timedelta(days=int(rng.integers(0, 2)))
        cash_pct = "-" if stock_deal else 100.0
        stock_pct = 100.0 if stock_deal else "-"
        conds = [c for c in ["Legal", "Management", "Shareholder Approvals", "Financial", "Other"] if rng.random() < 0.6]
        buyer = f"Buyer {i:03d} Corp." + (f" (NYSE:B{i:03d})" if listed_buyer else "")
        target = f"Target {i:03d} Inc. (NasdaqGS:T{i:03d})"
        sic = int(rng.choice([2834, 3674, 4911, 6021, 7372, 1311, 5812, 3841]))
        row = {
            "All Transactions Announced Date": ann.strftime("%Y-%m-%d"),
            "Target/Issuer": target,
            "Exchange:Ticker": "-",
            "Transaction Types": "Merger/Acquisition",
            "Transaction Status": "Cancelled" if broke else "Closed",
            "Total Transaction Value ($USDmm, Historical rate)": round(value, 2),
            "Buyers/Investors": buyer,
            "Sellers": "-",
            "CIQ Transaction ID": f"SYNTH{i:06d}",
            "Geographic Locations [Target/Issuer]": "United States of America (Primary)",
            "Implied Equity Value ($USDmm, Historical rate)": round(value * 0.95, 2),
            "Implied Enterprise Value ($USDmm, Historical rate)": round(value * 1.1, 2) if rng.random() < 0.75 else "-",
            "Implied Enterprise Value/EBITDA (x)": round(float(rng.uniform(5, 25)), 2) if rng.random() < 0.6 else "-",
            "Percent Sought (%)": 100.0,
            "Offer per Share ($USD, Historical rate)": round(float(rng.uniform(5, 120)), 2),
            "Cash % of Consideration (%)": cash_pct,
            "Stock % of Consideration (%)": stock_pct,
            "Target Spread ($USD, Historical rate)": "-",
            "Target Spread (%)": "-",
            "Sell-Side Termination Fee (%)": "-",
            "Buy-Side Termination Fee (%)": "-",
            "Target Stock Premium - 1 Day Prior (%)": round(prem, 2) if rng.random() < 0.93 else "-",
            "Target Stock Premium - 1 Week Prior (%)": round(prem + float(rng.normal(2, 5)), 2),
            "Target Stock Premium - 1 Month Prior (%)": round(prem + float(rng.normal(5, 10)), 2),
            "Target LTM Return % (%)": round(float(rng.normal(10, 40)), 2),
            "Target % of 52-Week High (%)": round(float(rng.uniform(40, 100)), 2),
            "Target Market Cap 1-Day Prior ($USDmm, Historical rate)": round(mktcap, 2),
            "M&A Announced Date": ann.strftime("%Y-%m-%d"),
            "M&A Definitive Agreement Date": signed.strftime("%Y-%m-%d"),
            "M&A Closed Date": "-" if broke else (ann + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "M&A Cancelled Date": (ann + pd.Timedelta(days=days)).strftime("%Y-%m-%d") if broke else "-",
            "Special Committee": str(rng.choice(["Yes", "No", "-"], p=[0.2, 0.3, 0.5])),
            "Deal Attitude": "Friendly",
            "Deal Approach": "Solicited" if solicited else "Unsolicited",
            "Deal Responses": str(rng.choice(["Approval", "No Response", "-"], p=[0.9, 0.05, 0.05])),
            "Deal Conditions": "; ".join(conds) if conds else "-",
            "Consideration Offered": "Common Equity" if stock_deal else "Cash",
            "CIK [Target/Issuer]": "-",   # no real CIKs: the synthetic sample never touches EDGAR
            "Excel Company ID [Target/Issuer]": f"SYNTHCO{i:05d}",
            "Security Tickers [Target/Issuer]": "-",
            "SIC Codes (Primary Code Only) [Target/Issuer]": sic,
        }
        fin_buyer = rng.random() < 0.2
        rev = float(np.exp(rng.normal(6.0, 1.2)))
        sec = [t for t, pr in (("Cash Merger", 0.0 if stock_deal else 0.9), ("Stock Merger", 0.9 if stock_deal else 0.0),
                               ("Tender Offer", 0.2), ("Cross-Border", 0.15), ("Going Private Transaction", 0.5 if fin_buyer else 0.05),
                               ("Leveraged Buy Out (LBO)", 0.8 if fin_buyer else 0.0), ("Go Shop Provision", 0.1)) if rng.random() < pr]
        acq_cap = float(np.exp(rng.normal(8.5, 1.4)))
        row.update({
            "Net Assumed Liabilities ($USDmm, Historical rate)": round(value * float(rng.uniform(-0.1, 0.4)), 2),
            "Implied Equity Value/LTM Net Income (x)": round(float(rng.uniform(8, 40)), 2) if rng.random() < 0.6 else "-",
            "Implied Equity Value/Book Value (x)": round(float(rng.uniform(0.8, 6)), 2),
            "Implied Enterprise Value/Revenues (x)": round(float(rng.uniform(0.5, 8)), 2),
            "Total Cash ($USDmm, Historical rate)": "-" if stock_deal else round(value, 2),
            "Total Stock ($USDmm, Historical rate)": round(value, 2) if stock_deal else "-",
            "Total Earnout/Contingent Payments ($USDmm, Historical rate)": "-",
            "Sell-Side Termination Fee ($USDmm, Historical rate)": round(value * 0.03, 2) if rng.random() < 0.85 else "-",
            "Buy-Side Termination Fee ($USDmm, Historical rate)": round(value * 0.05, 2) if rng.random() < 0.35 else "-",
            "Acquirer LTM Return % (%)": round(float(rng.normal(12, 30)), 2) if listed_buyer else "-",
            "Target % of 52-Week Low (%)": round(float(rng.uniform(105, 250)), 2),
            "Acquirer % of 52-Week High (%)": round(float(rng.uniform(50, 100)), 2) if listed_buyer else "-",
            "Acquirer Market Cap 1-Day Prior ($USDmm, Historical rate)": round(acq_cap, 2) if listed_buyer else "-",
            "Transaction Comments": f"SYNTHETIC NOTE. {buyer} agreed to acquire the target.",
            "Pre-Deal Situation": "-",
            "Deal Resolution": "SYNTHETIC NOTE. The buyer cancelled the acquisition." if broke else "SYNTHETIC NOTE. The buyer completed the acquisition.",
            "Advisors To Target": f"Synthetic Bank {int(rng.integers(1, 6))} (Financial Advisor); Synthetic Law {int(rng.integers(1, 6))} LLP (Legal Advisor)",
            "Advisors To Buyers": f"Synthetic Bank {int(rng.integers(1, 6))} (Financial Advisor); Synthetic Law {int(rng.integers(1, 6))} LLP (Legal Advisor)",
            "Transaction Primary Features": "Acquisition of Equity Stake",
            "Transaction Secondary Features": "; ".join(sec) if sec else "-",
            "Majority / Minority Stakes": "Majority",
            "Change of Control": "Yes",
            "Related Transactions": "-",
            "Target/Issuer LTM Financials - Total Revenue (at Announcement) ($USDmm, Historical rate)": round(rev, 2),
            "Target/Issuer LTM Financials - EBITDA (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(-0.05, 0.35)), 2),
            "Target/Issuer LTM Financials - Net Income (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(-0.15, 0.2)), 2),
            "Target/Issuer LTM Financials - Total Common Equity (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(0.2, 1.5)), 2),
            "Target/Issuer LTM Financials - Total Debt (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(0, 1.2)), 2),
            "Target/Issuer LTM Financials - Total Cash & ST Investments (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(0.02, 0.5)), 2),
            "Target/Issuer LTM Financials - Total Assets (at Announcement) ($USDmm, Historical rate)": round(rev * float(rng.uniform(1.0, 3.0)), 2),
            "Acquirer LTM Financials - Total Revenue (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 0.6, 2) if listed_buyer else "-",
            "Acquirer LTM Financials - EBITDA (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 0.1, 2) if listed_buyer else "-",
            "Acquirer LTM Financials - Net Income (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 0.05, 2) if listed_buyer else "-",
            "Acquirer LTM Financials - Total Debt (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 0.3, 2) if listed_buyer else "-",
            "Acquirer LTM Financials - Total Cash & ST Investments (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 0.1, 2) if listed_buyer else "-",
            "Acquirer LTM Financials - Total Assets (at Announcement) ($USDmm, Historical rate)": round(acq_cap * 1.2, 2) if listed_buyer else "-",
            "Primary Industry [Buyers/Investors]": str(rng.choice(["Software", "Banks", "Pharmaceuticals", "Oil and Gas", "Asset Management"])),
            "Primary Sector [Buyers/Investors]": "Financials" if fin_buyer else str(rng.choice(["Information Technology", "Health Care", "Energy", "Industrials"])),
            "Buyer Type [Buyers/Investors]": "Financial Buyer" if fin_buyer else "Strategic Buyer",
            "Company Type [Buyers/Investors]": "Private Investment Firm" if fin_buyer else ("Public Company" if listed_buyer else "Private Company"),
        })
        rows.append(row)
    df = pd.DataFrame(rows)[list(COLS) + list(EXTRA_COLS)]
    # A few rows that the sample filters must remove, to exercise them.
    df.loc[0, "M&A Definitive Agreement Date"] = "-"                       # no definitive agreement
    df.loc[1, "Percent Sought (%)"] = 20.0                                 # minority stake
    df.loc[2, "Buyers/Investors"] = "Target 002 Inc. (NasdaqGS:T002)"      # self-tender
    df.loc[3, "Target/Issuer"] = "Synthetic Acquisition Corp. (NasdaqGS:SYNA)"  # SPAC by name
    df.loc[4, "SIC Codes (Primary Code Only) [Target/Issuer]"] = 6770      # SPAC by SIC
    out = ROOT / "data/sample/synthetic_ciq_export.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(df)} synthetic rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()

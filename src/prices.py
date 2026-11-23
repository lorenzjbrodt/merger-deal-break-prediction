"""Parse the three Capital IQ price exports into one long table (deal, date, close).

Each export has the same rows as the deal export (sheet "Screening", header on row 8, key
"CIQ Transaction ID") and one column per monthly grid date holding the TARGET's daily close:

    Day Close Price [12/20/2000] ($USD, Historical rate) [Target/Issuer]
    Day Close Price [Latest - 3 Month(s)] ($USD, Historical rate) [Target/Issuer]
    Day Close Price [Latest] ($USD, Historical rate) [Target/Issuer]

"Latest" is the last trading day before the export date (2026-11-22 -> 2026-11-20), and
"Latest - k Month(s)" is k calendar months earlier. "-" means no price (not listed yet, or
delisted). Output: data/interim/prices.parquet (licensed, never committed).

Usage: python src/prices.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
LATEST = pd.Timestamp("2026-11-20")     # last trading day before the export date
PRICE_RE = re.compile(r"^Day Close Price \[(.+?)\]")


def header_date(col: str) -> pd.Timestamp | None:
    m = PRICE_RE.match(str(col))
    if not m:
        return None
    tag = m.group(1).strip()
    if tag == "Latest":
        return LATEST
    k = re.match(r"Latest - (\d+) Month", tag)
    if k:
        return LATEST - pd.DateOffset(months=int(k.group(1)))
    return pd.to_datetime(tag, format="%m/%d/%Y")


def load_one(path: Path) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(path, sheet_name="Screening", header=7, engine="xlrd", na_values=["-"], dtype=object)
    df = df[df["CIQ Transaction ID"].notna()].copy()
    price_cols = {c: header_date(c) for c in df.columns if header_date(c) is not None}
    wide = df[["CIQ Transaction ID"] + list(price_cols)].rename(columns={"CIQ Transaction ID": "deal_id", **price_cols})
    wide = wide.drop_duplicates("deal_id").set_index("deal_id")
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide, sorted(price_cols.values())


def main() -> None:
    wides, grids, report = [], [], {"files": {}}
    for i in (1, 2, 3):
        w, dates = load_one(RAW / f"ciq_prices_{i}.xls")
        wides.append(w); grids.append(dates)
        report["files"][f"ciq_prices_{i}.xls"] = {
            "rows": int(len(w)), "price_columns": len(dates),
            "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
            "share_cells_with_price": float(w.notna().to_numpy().mean())}
    all_dates = sorted(d for g in grids for d in g)
    dup = sorted({str(d.date()) for d in all_dates if all_dates.count(d) > 1})
    full = pd.date_range(all_dates[0].replace(day=1), all_dates[-1], freq="MS") + pd.Timedelta(days=19)   # every 20th
    missing = sorted(str(d.date()) for d in set(full) - set(all_dates))
    extra = sorted(str(d.date()) for d in set(all_dates) - set(full))
    report.update({"grid_first": str(all_dates[0].date()), "grid_last": str(all_dates[-1].date()),
                   "n_grid_dates": len(set(all_dates)), "overlapping_dates": dup,
                   "missing_months": missing, "dates_not_on_the_20th": extra})

    wide = pd.concat(wides, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    wide = wide[sorted(wide.columns)]
    long = wide.stack().rename("close").reset_index()
    long.columns = ["deal_id", "date", "close"]
    long["date"] = pd.to_datetime(long["date"])
    long = long[long["close"] > 0].sort_values(["deal_id", "date"]).reset_index(drop=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    long.to_parquet(INTERIM / "prices.parquet", index=False)

    weekend_dates = [d for d in sorted(set(all_dates)) if d.day_of_week >= 5]
    report["grid_dates_on_weekend"] = len(weekend_dates)
    # Does a non-trading 20th carry a price (i.e. the prior close) or a blank?
    if weekend_dates:
        wk = wide[weekend_dates]
        prev_ok = pd.DataFrame({d: wide[[c for c in wide.columns if c < d][-1]].notna() & wide[[c for c in wide.columns if c > d][0]].notna()
                                for d in weekend_dates if any(c < d for c in wide.columns) and any(c > d for c in wide.columns)})
        filled = wk[prev_ok.columns].notna()
        report["weekend_20th_filled_share_when_neighbours_priced"] = float(filled.to_numpy()[prev_ok.to_numpy()].mean())
    report["deals_with_any_price"] = int(long["deal_id"].nunique())
    report["rows_long"] = int(len(long))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/price_grid_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

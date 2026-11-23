"""Market-context features as of the announcement date (public data + the deal sample).

FRED series (daily, CSV from https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>), cached
under data/external/fred/<ID>.csv:
    VIXCLS      CBOE VIX
    DBAA, DAAA  Moody's seasoned Baa / Aaa corporate bond yields
    DGS10       10-year Treasury constant-maturity yield
    DTB3        3-month Treasury bill (risk-free rate for the carry adjustment)
    NASDAQCOM   NASDAQ Composite (equity market proxy with full history)
    SP500       S&P 500 (FRED only carries the last 10 years)

Every market feature uses the last observation STRICTLY BEFORE the announcement date.
Deal-flow features use only deals announced before the date, and the trailing break rate
uses only deals whose outcome date lies before the announcement date.

FRED answers browser clients only (command-line clients get an HTTP/2 error), so the CSVs
are saved once into the cache folder; `download()` tries a plain request and otherwise
tells you which files to place there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRED = ROOT / "data/external/fred"
SERIES = ["VIXCLS", "DBAA", "DAAA", "DGS10", "DTB3", "NASDAQCOM", "SP500"]
MARKET_FEATURES = ["mkt_vix", "mkt_vix_chg_30d", "mkt_baa_aaa_spread", "mkt_baa_10y_spread", "mkt_ust10y",
                   "mkt_equity_ret_3m", "mkt_equity_ret_12m", "mkt_deals_90d", "mkt_break_rate_12m"]


def download() -> None:
    import requests
    FRED.mkdir(parents=True, exist_ok=True)
    for s in SERIES:
        path = FRED / f"{s}.csv"
        if path.exists():
            continue
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=1999-01-01", timeout=30)
            r.raise_for_status()
            path.write_text(r.text)
        except Exception as e:
            print(f"could not fetch {s} ({type(e).__name__}); download it in a browser and save it as {path}")


def load_series(name: str) -> pd.Series:
    df = pd.read_csv(FRED / f"{name}.csv")
    df.columns = ["date", "value"]
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]))
    return s.dropna().sort_index()


def asof(s: pd.Series, dates: pd.Series, lag_days: int = 0) -> np.ndarray:
    """Last observation strictly before (date - lag_days)."""
    key = (pd.to_datetime(dates) - pd.Timedelta(days=lag_days)).to_numpy()
    pos = s.index.searchsorted(key, side="left") - 1
    out = np.full(len(key), np.nan)
    ok = pos >= 0
    out[ok] = s.to_numpy()[pos[ok]]
    return out


def market_features(deals: pd.DataFrame) -> pd.DataFrame:
    """One row per deal, aligned with `deals`. `deals` needs announced_date, closed_date,
    cancelled_date and label (outcome fields are used only for PAST deals' resolved outcomes)."""
    ann = pd.to_datetime(deals["announced_date"])
    M = pd.DataFrame(index=deals.index)
    vix, baa, aaa, y10, eq = (load_series(k) for k in ["VIXCLS", "DBAA", "DAAA", "DGS10", "NASDAQCOM"])
    M["mkt_vix"] = asof(vix, ann)
    M["mkt_vix_chg_30d"] = M["mkt_vix"] - asof(vix, ann, 30)
    M["mkt_baa_aaa_spread"] = asof(baa, ann) - asof(aaa, ann)
    M["mkt_baa_10y_spread"] = asof(baa, ann) - asof(y10, ann)
    M["mkt_ust10y"] = asof(y10, ann)
    M["mkt_equity_ret_3m"] = asof(eq, ann) / asof(eq, ann, 91) - 1
    M["mkt_equity_ret_12m"] = asof(eq, ann) / asof(eq, ann, 365) - 1

    # deal flow: computed from the sample itself, using only information dated before the announcement
    a = ann.to_numpy()
    order = np.sort(a)
    M["mkt_deals_90d"] = np.searchsorted(order, a, side="left") - np.searchsorted(order, a - np.timedelta64(90, "D"), side="left")
    out_date = pd.to_datetime(deals["closed_date"].fillna(deals["cancelled_date"])).to_numpy()
    lab = deals["label"].to_numpy()
    o = np.argsort(out_date)
    od, cum_n, cum_b = out_date[o], np.arange(1, len(o) + 1), np.cumsum(lab[o])
    hi = np.searchsorted(od, a, side="left")                       # outcomes strictly before the announcement
    lo = np.searchsorted(od, a - np.timedelta64(365, "D"), side="left")
    n = hi - lo
    b = np.where(hi > 0, cum_b[np.maximum(hi - 1, 0)], 0) - np.where(lo > 0, cum_b[np.maximum(lo - 1, 0)], 0)
    M["mkt_break_rate_12m"] = np.where(n >= 20, b / np.maximum(n, 1), np.nan)   # undefined until 20 resolved deals exist
    return M[MARKET_FEATURES]


def risk_free(dates: pd.Series) -> np.ndarray:
    """3-month T-bill yield (decimal, annualised) known before each date."""
    return asof(load_series("DTB3"), dates) / 100.0


if __name__ == "__main__":
    download()
    d = pd.read_parquet(ROOT / "data/interim/deals.parquet")
    M = market_features(d)
    print(M.describe().T[["count", "mean", "min", "max"]].round(3))

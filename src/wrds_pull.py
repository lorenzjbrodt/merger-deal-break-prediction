"""Pull CRSP / Compustat data from WRDS (PostgreSQL) into data/wrds/ (licensed, git-ignored).

Connection: host wrds-pgdata.wharton.upenn.edu, port 9737, database wrds. The user name comes from
the WRDS_USER environment variable or from the WRDS line in ~/.pgpass; the password is read by
libpq from ~/.pgpass and never appears in this code. See DATA.md for the one-off setup.

  python src/wrds_pull.py lookups   reference tables (Compustat company/security/annual sales, CCM links, CRSP names, delistings,
                                    trading calendar, Fama-French daily market return)
  python src/wrds_pull.py daily     daily CRSP data for linked targets and acquirers, Compustat daily prices as a second source
                                    (needs data/interim/crsp_link.parquet from src/crsp_link.py)

Every stage caches its output; delete the file to pull again.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
PRE_DAYS, POST_DAYS = 75, 30           # calendar-day padding around [announcement, outcome]; trimmed to trading days later


def wrds_user() -> str:
    if os.environ.get("WRDS_USER"):
        return os.environ["WRDS_USER"]
    pg = Path.home() / ".pgpass"
    if pg.exists():
        for line in pg.read_text().splitlines():
            parts = line.split(":")
            if len(parts) >= 5 and parts[0] == HOST:
                return parts[3]
    raise SystemExit("No WRDS user: set WRDS_USER or add the WRDS line to ~/.pgpass (see DATA.md).")


def connect():
    import psycopg
    con = psycopg.connect(host=HOST, port=PORT, dbname=DB, user=wrds_user(), sslmode="require", connect_timeout=60)
    con.autocommit = True
    return con


def query(con, sql: str, params=None) -> pd.DataFrame:
    from decimal import Decimal
    with con.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in df.columns:                                   # numeric columns arrive as Decimal
        s = df[c].dropna()
        if len(s) and isinstance(s.iloc[0], Decimal):
            df[c] = df[c].astype(float)
    return df


def cached(name: str, fn):
    WRDS.mkdir(parents=True, exist_ok=True)
    path = WRDS / f"{name}.parquet"
    if path.exists():
        print(f"  {name}: cached ({len(pd.read_parquet(path)):,} rows)")
        return
    t = time.time()
    df = fn()
    df.to_parquet(path, index=False)
    print(f"  {name}: {len(df):,} rows in {time.time() - t:.0f}s")


def lookups() -> None:
    con = connect()
    cached("comp_company", lambda: query(con, "select gvkey, conm, conml, cik, priusa, prican, costat, dldte, dlrsn, sic, loc, fic from comp.company"))
    cached("comp_security", lambda: query(con, "select gvkey, iid, tic, cusip, exchg, excntry, tpci, secstat, dldtei from comp.security"))
    cached("ccm_link", lambda: query(con, "select gvkey, linkprim, liid, linktype, lpermno, lpermco, linkdt, linkenddt from crsp.ccmxpf_lnkhist"))
    cached("crsp_names", lambda: query(con, """select permno, permco, namedt, nameenddt, securitybegdt, securityenddt, cusip, hdrcusip, ticker, issuernm,
                                                     primaryexch, tradingstatusflg, shareclass, sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd
                                              from crsp.stocknames_v2"""))
    cached("crsp_delist", lambda: query(con, "select * from crsp.stkdelists"))
    cached("calendar_crsp", lambda: query(con, "select distinct dlycaldt as date from crsp.dsf_v2 where permno in (10107, 12490, 11850) and dlycaldt >= '2000-06-01' order by 1"))
    cached("ff_daily", lambda: query(con, "select date, mktrf, rf from ff.factors_daily where date >= '2000-01-01' order by 1"))
    cached("comp_funda", lambda: query(con, """select gvkey, datadate, fyear, sale, at, sich from comp.funda
                                              where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and fyear >= 1998"""))
    cached("calendar_comp", lambda: query(con, "select distinct datadate as date from comp.secd where gvkey in ('006066', '012141') and datadate >= '2000-06-01' order by 1"))
    con.close()


def _windows(link: pd.DataFrame, key: str) -> pd.DataFrame:
    """One padded date window per security: the union of the windows of all deals it appears in is pulled per deal (duplicates dropped later)."""
    w = link.dropna(subset=[key]).copy()
    w["d0"] = (w["announced_date"] - pd.Timedelta(days=PRE_DAYS)).dt.date
    w["d1"] = (w["outcome_date"] + pd.Timedelta(days=POST_DAYS)).dt.date
    return w[[key, "d0", "d1"]].drop_duplicates()


def _pull_windows(con, w: pd.DataFrame, key: str, sql_tpl: str, cast: str, batch: int = 400) -> pd.DataFrame:
    out = []
    for i in range(0, len(w), batch):
        b = w.iloc[i:i + batch]
        vals = ", ".join(f"(%s::{cast}, %s::date, %s::date)" for _ in range(len(b)))
        params = [x for r in b.itertuples(index=False) for x in (r[0] if cast != "int" else int(r[0]), r[1], r[2])]
        out.append(query(con, sql_tpl.format(values=vals), params))
        print(f"    {min(i + batch, len(w)):,}/{len(w):,} windows, {sum(len(o) for o in out):,} rows", flush=True)
    return pd.concat(out, ignore_index=True).drop_duplicates() if out else pd.DataFrame()


CRSP_DAILY = """select a.permno, a.dlycaldt as date, a.dlyprc, a.dlyprcflg, a.dlyclose, a.dlybid, a.dlyask, a.dlyret, a.dlyretx, a.dlyretmissflg,
                       a.dlydelflg, a.dlycumfacpr, a.dlycumfacshr, a.dlyfacprc, a.dlyorddivamt, a.dlynonorddivamt, a.dlydistretflg,
                       a.dlyvol, a.shrout, a.ticker, a.primaryexch, a.tradingstatusflg
                from crsp.dsf_v2 a join (values {values}) v(permno, d0, d1) on a.permno = v.permno and a.dlycaldt between v.d0 and v.d1"""
CRSP_PRE = """select a.permno, a.dlycaldt as date, a.dlyprc, a.dlyret, a.dlyvol
              from crsp.dsf_v2 a join (values {values}) v(permno, d0, d1) on a.permno = v.permno and a.dlycaldt between v.d0 and v.d1"""
COMP_DAILY = """select a.gvkey, a.iid, a.datadate as date, a.prccd, a.ajexdi, a.trfd, a.div, a.cshoc, a.curcdd, a.exchg, a.tpci, a.tic
                from comp.secd a join (values {values}) v(gvkey, d0, d1) on a.gvkey = v.gvkey and a.datadate between v.d0 and v.d1"""


def daily() -> None:
    link = pd.read_parquet(INTERIM / "crsp_link.parquet")
    link["outcome_date"] = link["outcome_date"].fillna(link["announced_date"] + pd.Timedelta(days=400))
    con = connect()
    cached("crsp_daily_target", lambda: _pull_windows(con, _windows(link, "permno"), "permno", CRSP_DAILY, "int"))
    cached("crsp_daily_acquirer", lambda: _pull_windows(con, _windows(link.rename(columns={"permno": "_t", "acq_permno": "permno"}), "permno"), "permno", CRSP_DAILY, "int"))
    pre = link.dropna(subset=["permno"]).copy()                 # one year before the window above: volatility, beta, run-up
    pre["d0"] = (pre["announced_date"] - pd.Timedelta(days=PRE_DAYS + 380)).dt.date
    pre["d1"] = (pre["announced_date"] - pd.Timedelta(days=PRE_DAYS + 1)).dt.date
    cached("crsp_daily_target_pre", lambda: _pull_windows(con, pre[["permno", "d0", "d1"]].drop_duplicates(), "permno", CRSP_PRE, "int"))
    cached("comp_daily_target", lambda: _pull_windows(con, _windows(link, "gvkey"), "gvkey", COMP_DAILY, "text"))
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lookups", "daily"])
    a = ap.parse_args()
    {"lookups": lookups, "daily": daily}[a.cmd]()

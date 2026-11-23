"""Daily price quantities per deal from CRSP (Compustat daily prices as the second source).

Builds (all git-ignored, licensed):
  data/interim/price_features_crsp.parquet   one row per deal: day-1 spread, market-implied probability, backtest legs
  data/interim/deal_daily_returns.parquet    daily total returns of the target from t1 to the exit, for the calendar-time portfolio
and the aggregate report results/crsp_prices.md.

Definitions (A = announcement date, O = outcome date)
  tm1     last trading day strictly before A;  P0 = target close on tm1 (the unaffected price)
  t1      first trading day strictly after A;  entry at close(t1). The deal must still be open: t1 < O.
  cash deals       spread = offer / close(t1) - 1
                   p_mkt  = (offer - close(t1)) / (offer - P0), clipped to [0, 1]
                   p_mkt_carry: same with the offer discounted at the 3-month T-bill rate over the expected remaining
                   time to closing (training-period median duration minus days elapsed, floor 30 days)
  stock and mixed  exchange ratio = stock part of the offer per share / acquirer close on tm1 (fixed-ratio assumption);
                   offer value at t1 = cash part + exchange ratio x acquirer close(t1); spread and p_mkt from that value
  offer            The export stores the *final* terms. For cash deals the offer used at t1 is the per-share price stated in the
                   announcement press release (EDGAR) when it differs from the export by more than 1% and is plausible
                   against P0 (between -30% and +200% premium); otherwise the export's offer. Using the final offer
                   would put a later price increase into the day-1 spread.
  validity         |P0 / (export offer / (1 + premium 1 day prior)) - 1| <= 15%: the CRSP price before announcement agrees with
                   the price implied by the export. Fails when the link is wrong or the export's offer was revised.
Backtest legs (total returns: CRSP daily returns include dividends)
  completed deal   product of daily returns from t1 to the CRSP delisting record (which carries the delisting return).
                   No delisting record near O (CRSP ends 31 Dec 2025, or the security kept trading): cash deals are paid
                   the offer on O, stock and mixed deals exit at the close on O.
  broken deal      exit at the close of the first trading day after the cancellation date (`ret`), and at the close
                   five trading days after it (`ret5`).
  price source     CRSP; Compustat daily prices continue a CRSP series that stops early and stand in where a target
                   has no CRSP link; the Capital IQ monthly close is the last resort for the exit of a broken deal.
Pre-announcement features (known at A): 20-day run-up, one-year volatility and market beta of the target.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import market_context as MC

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
TOL = 0.15
D = "datetime64[D]"
PRICE = re.compile(r"\$\s?(\d{1,4}(?:,\d{3})*(?:\.\d{1,4})?)\s+(?:per|a|for each|in cash per|net per|in cash for each)\s+"
                   r"(?:common\s+|ordinary\s+|outstanding\s+|class a\s+)*(?:share|unit|ads|american depositary share)", re.I)


def stated_prices(txt) -> list:
    """Per-share dollar amounts in the opening of the announcement, most frequent first."""
    if not isinstance(txt, str) or not txt:
        return []
    c = Counter(float(m.group(1).replace(",", "")) for m in PRICE.finditer(txt[:6000]))
    return [v for v, _ in c.most_common()]


def announced_offer(export_offer, P0, cands):
    """(offer known on the announcement day, source). See the module docstring."""
    if np.isfinite(export_offer) and any(abs(v / export_offer - 1) <= 0.01 for v in cands):
        return export_offer, "export, confirmed by the press release"
    for v in cands:
        if np.isfinite(P0) and P0 > 0 and 0.7 <= v / P0 <= 3.0:
            return v, "press release (differs from the export)"
    return export_offer, "export, not checked (no text or no price found)"


class Series:
    """Daily series of one security: dates, close, total return, flags."""
    __slots__ = ("dates", "prc", "ret", "delist", "payout", "src")

    def __init__(self, dates, prc, ret, delist=None, payout=None, src="crsp"):
        self.dates, self.prc, self.ret = dates, prc, ret
        self.delist = delist if delist is not None else np.zeros(len(dates), bool)
        self.payout = payout if payout is not None else np.zeros(len(dates))
        self.src = np.full(len(dates), src, dtype=object)

    def at(self, day):
        i = np.searchsorted(self.dates, day)
        return self.prc[i] if i < len(self.dates) and self.dates[i] == day and np.isfinite(self.prc[i]) and self.prc[i] > 0 else np.nan

    def last_before(self, day, max_back=7):
        """Last positive close on or before `day`, at most `max_back` calendar days earlier."""
        i = np.searchsorted(self.dates, day, side="right") - 1
        while i >= 0 and (day - self.dates[i]).astype(int) <= max_back:
            if np.isfinite(self.prc[i]) and self.prc[i] > 0:
                return self.prc[i], self.dates[i]
            i -= 1
        return np.nan, np.datetime64("NaT", "D")

    def growth(self, a, b):
        """Product of (1 + r) over trading days in (a, b]."""
        i, j = np.searchsorted(self.dates, a, side="right"), np.searchsorted(self.dates, b, side="right")
        return float(np.prod(1 + np.nan_to_num(self.ret[i:j])))


def crsp_series(path) -> dict:
    t = pd.read_parquet(path).sort_values(["permno", "date"])
    t["date"] = pd.to_datetime(t["date"]).to_numpy().astype(D)
    out = {}
    for p, g in t.groupby("permno", sort=False):
        dl = (g["dlydelflg"] == "Y").to_numpy()
        prc = np.where(dl, np.nan, g["dlyprc"].to_numpy(float))
        out[int(p)] = Series(g["date"].to_numpy().astype(D), prc, g["dlyret"].to_numpy(float), dl, g["dlynonorddivamt"].fillna(0).to_numpy(float))
    return out


def comp_series(link: pd.DataFrame) -> dict:
    c = pd.read_parquet(WRDS / "comp_daily_target.parquet")
    c = c[(c["curcdd"] == "USD") & c["prccd"].notna()].copy()
    c["date"] = pd.to_datetime(c["date"]).to_numpy().astype(D)
    lk = pd.read_parquet(WRDS / "ccm_link.parquet").dropna(subset=["lpermno"])
    liid = {(g, int(p)): i for g, p, i in zip(lk["gvkey"], lk["lpermno"], lk["liid"])}
    cc = pd.read_parquet(WRDS / "comp_company.parquet").set_index("gvkey")["priusa"].to_dict()
    want = {}
    for g, p in zip(link["gvkey"], link["permno"]):
        if isinstance(g, str):
            want.setdefault(g, liid.get((g, int(p))) if pd.notna(p) else None)
    out = {}
    for g, x in c.groupby("gvkey", sort=False):
        iids = x["iid"].value_counts()
        pick = want.get(g) if want.get(g) in iids.index else (cc.get(g) if cc.get(g) in iids.index else None)
        if pick is None:
            common = x[x["tpci"] == "0"]["iid"].value_counts()
            pick = (common if len(common) else iids).index[0]
        x = x[x["iid"] == pick].drop_duplicates("date").sort_values("date")
        adj = x["prccd"].to_numpy(float) / x["ajexdi"].replace(0, np.nan).fillna(1).to_numpy(float) * x["trfd"].ffill().fillna(1).to_numpy(float)
        ret = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
        gap = np.r_[0, np.diff(x["date"].to_numpy().astype(D)).astype(int)]
        ret[gap > 10] = np.nan                                  # do not bridge long gaps in the series
        out[g] = Series(x["date"].to_numpy().astype(D), x["prccd"].to_numpy(float), ret, src="comp")
    return out


def splice(a: Series | None, b: Series | None) -> Series | None:
    """CRSP first; Compustat continues it after its last date unless CRSP ended with a delisting record."""
    if a is None or len(a.dates) == 0:
        return b
    if b is None or a.delist.any():
        return a
    k = b.dates > a.dates[-1]
    if not k.any():
        return a
    r = b.ret[k].copy()
    first = np.where(k)[0][0]
    last_p = a.prc[np.isfinite(a.prc)][-1] if np.isfinite(a.prc).any() else np.nan
    r[0] = b.prc[first] / last_p - 1 if np.isfinite(last_p) and abs(b.prc[first] / last_p - 1) < 0.5 else np.nan   # bridge on price
    s = Series(np.r_[a.dates, b.dates[k]], np.r_[a.prc, b.prc[k]], np.r_[a.ret, r], np.r_[a.delist, np.zeros(k.sum(), bool)], np.r_[a.payout, np.zeros(k.sum())])
    s.src = np.r_[a.src, b.src[k]]
    return s


def pre_features(link: pd.DataFrame, cal: np.ndarray) -> pd.DataFrame:
    """Run-up, volatility and beta of the target before the announcement (all dates strictly before A)."""
    a = pd.read_parquet(WRDS / "crsp_daily_target_pre.parquet")[["permno", "date", "dlyret", "dlyprc"]]
    b = pd.read_parquet(WRDS / "crsp_daily_target.parquet")[["permno", "date", "dlyret", "dlyprc"]]
    t = pd.concat([a, b]).drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    t["date"] = pd.to_datetime(t["date"]).to_numpy().astype(D)
    ff = pd.read_parquet(WRDS / "ff_daily.parquet")
    mkt = pd.Series((ff["mktrf"] + ff["rf"]).to_numpy(float), index=pd.to_datetime(ff["date"]).to_numpy().astype(D))
    by = {int(p): (g["date"].to_numpy().astype(D), g["dlyret"].to_numpy(float), g["dlyprc"].to_numpy(float)) for p, g in t.groupby("permno", sort=False)}
    rows = []
    for r in link.itertuples(index=False):
        o = {"deal_id": r.deal_id, "runup_20d": np.nan, "vol_1y": np.nan, "beta_1y": np.nan}
        if pd.notna(r.permno) and int(r.permno) in by:
            dts, ret, prc = by[int(r.permno)]
            A = np.datetime64(r.announced_date, "D")
            j = np.searchsorted(dts, A)                      # first row on/after A
            if j >= 21 and np.isfinite(prc[j - 1]) and np.isfinite(prc[j - 21]) and prc[j - 21] > 0:
                o["runup_20d"] = float(np.prod(1 + np.nan_to_num(ret[j - 20:j])) - 1)
            lo = max(j - 272, 0)
            x, dd = ret[lo:j - 20], dts[lo:j - 20]           # one year ending 20 trading days before A (excludes the run-up)
            ok = np.isfinite(x)
            if ok.sum() >= 120:
                m = mkt.reindex(dd[ok]).to_numpy()
                k = np.isfinite(m)
                o["vol_1y"] = float(np.std(x[ok], ddof=1) * np.sqrt(252))
                if k.sum() >= 120 and np.var(m[k]) > 0:
                    o["beta_1y"] = float(np.cov(x[ok][k], m[k])[0, 1] / np.var(m[k], ddof=1))
        rows.append(o)
    return pd.DataFrame(rows)


def main() -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    o = pd.read_parquet(INTERIM / "outcomes.parquet")[["deal_id", "outcome3"]]
    d = d.merge(o, on="deal_id", how="left")
    link = pd.read_parquet(INTERIM / "crsp_link.parquet").set_index("deal_id").reindex(d["deal_id"]).reset_index()
    cal = np.union1d(pd.to_datetime(pd.read_parquet(WRDS / "calendar_crsp.parquet")["date"]).to_numpy().astype(D),
                     pd.to_datetime(pd.read_parquet(WRDS / "calendar_comp.parquet")["date"]).to_numpy().astype(D))
    cal = cal[(pd.DatetimeIndex(cal).dayofweek < 5)]
    crsp_end = pd.to_datetime(pd.read_parquet(WRDS / "calendar_crsp.parquet")["date"]).max().to_numpy().astype(D)
    T = crsp_series(WRDS / "crsp_daily_target.parquet")
    Aq = crsp_series(WRDS / "crsp_daily_acquirer.parquet")
    C = comp_series(link)
    ciq = pd.read_parquet(INTERIM / "price_features.parquet").set_index("deal_id") if (INTERIM / "price_features.parquet").exists() else None
    tp = INTERIM / "text.parquet"
    press = d[["deal_id"]].merge(pd.read_parquet(tp)[["deal_id", "press_release"]], on="deal_id", how="left")["press_release"].tolist() if tp.exists() else [None] * len(d)

    n = len(d)
    label = d["label"].to_numpy()
    offer = d["offer_per_share"].to_numpy(float)
    prem = d["premium_1d"].to_numpy(float)
    cons = d["consideration"].fillna("").astype(str)
    cash_pct, stock_pct = d["cash_pct"].to_numpy(float), d["stock_pct"].to_numpy(float)
    cash = (cash_pct == 100) | ((cons == "Cash").to_numpy() & np.isnan(cash_pct))
    ann = d["announced_date"].to_numpy().astype(D)
    out = d["closed_date"].fillna(d["cancelled_date"]).to_numpy().astype(D)
    train_closed = (d["year"].to_numpy() <= 2018) & (label == 0)
    med_dur = float(np.median((out - ann).astype(int)[train_closed]))

    rec, daily = [], []
    for k in range(n):
        r = {"deal_id": d["deal_id"].iat[k], "cash_deal": bool(cash[k]), "price_source": "", "tm1": np.datetime64("NaT", "D"), "t1": np.datetime64("NaT", "D")}
        permno, gv = link["permno"].iat[k], link["gvkey"].iat[k]
        a = T.get(int(permno)) if pd.notna(permno) else None
        b = C.get(gv) if isinstance(gv, str) else None
        s = splice(a, b)
        i = np.searchsorted(cal, ann[k])
        j = np.searchsorted(cal, ann[k], side="right")
        if s is None or i == 0 or j >= len(cal):
            rec.append(r); continue
        tm1, t1 = cal[i - 1], cal[j]
        P0, _ = s.last_before(tm1, 7)
        c1 = s.at(t1)
        r.update(tm1=tm1, t1=t1, P0=P0, close_t1=c1, days_to_t1=float((t1 - ann[k]).astype(int)),
                 price_source={"crsp": "CRSP", "comp": "Compustat"}.get(s.src[min(np.searchsorted(s.dates, t1), len(s.dates) - 1)], ""))
        imp = offer[k] / (1 + prem[k] / 100) if np.isfinite(prem[k]) and np.isfinite(offer[k]) else np.nan
        r["ratio_p0"] = P0 / imp if np.isfinite(imp) and imp > 0 else np.nan
        r["price_valid"] = bool(np.isfinite(r["ratio_p0"]) and abs(r["ratio_p0"] - 1) <= TOL)
        t1_ok = np.isfinite(c1) and t1 < out[k]
        r["t1_ok"] = bool(t1_ok)
        # ---- offer value at t1
        value1 = np.nan
        if cash[k]:
            value1, r["offer_source"] = announced_offer(offer[k], P0, stated_prices(press[k]))
            r["offer_t1"], r["spread_kind"] = value1, "cash"
        else:
            aq = Aq.get(int(link["acq_permno"].iat[k])) if pd.notna(link["acq_permno"].iat[k]) else None
            sp, cp = np.nan_to_num(stock_pct[k]), np.nan_to_num(cash_pct[k])
            if aq is not None and sp > 0 and np.isfinite(offer[k]):
                a0, _ = aq.last_before(tm1, 7); a1 = aq.at(t1)
                if np.isfinite(a0) and np.isfinite(a1) and a0 > 0:
                    er = offer[k] * sp / 100 / a0
                    value1 = offer[k] * (1 - sp / 100) + er * a1          # everything that is not stock is treated as fixed value
                    r.update(exchange_ratio=er, acq_ret_to_t1=a1 / a0 - 1, spread_kind="stock" if sp == 100 else "mixed")
        if t1_ok and np.isfinite(value1) and value1 > 0:
            r["spread"] = value1 / c1 - 1
            if np.isfinite(P0) and value1 > P0:
                r["p_mkt"] = float(np.clip((value1 - c1) / (value1 - P0), 0, 1))
                Tm = max(med_dur - (t1 - ann[k]).astype(int), 30) / 365.0
                r["_T"] = Tm; r["_value1"] = value1
        # ---- backtest legs
        if t1_ok:
            if label[k] == 0:
                dl = np.where(s.delist & (s.dates > t1))[0]
                near = [x for x in dl if -15 <= (s.dates[x] - out[k]).astype(int) <= 45]
                if near:
                    e = s.dates[near[0]]
                    r.update(ret=s.growth(t1, e) - 1, exit_date=e, exit_rule="delisting return", delist_payout=float(s.payout[near[0]]))
                else:
                    lp, ld = s.last_before(out[k], 10 ** 6)
                    if np.isfinite(lp) and ld >= t1:
                        g = s.growth(t1, ld)
                        if cash[k] and np.isfinite(offer[k]):
                            r.update(ret=g * offer[k] / lp - 1, exit_date=out[k], exit_rule="offer on the closing date")
                        else:
                            r.update(ret=g - 1, exit_date=max(ld, t1 + np.timedelta64(1, "D")), exit_rule="close on the closing date")
                if "exit_date" in r:
                    daily.append((r["deal_id"], s, t1, min(r["exit_date"], s.dates[-1]), r["ret"]))
            else:
                e1i = np.searchsorted(cal, out[k], side="right")
                for lagname, ei in (("", e1i), ("5", e1i + 4)):
                    if ei >= len(cal):
                        continue
                    e = cal[ei]
                    if s.dates[-1] >= e:
                        r.update({f"ret{lagname}": s.growth(t1, e) - 1, f"exit_date{lagname}": e, f"exit_rule{lagname}": "close after cancellation"})
                    elif s.delist.any() and s.dates[-1] > t1:
                        r.update({f"ret{lagname}": s.growth(t1, s.dates[-1]) - 1, f"exit_date{lagname}": s.dates[-1], f"exit_rule{lagname}": "delisted before the exit date"})
                    elif ciq is not None and r["deal_id"] in ciq.index and np.isfinite(ciq.at[r["deal_id"], "exit_price"]) and label[k] == 1:
                        xd = np.datetime64(ciq.at[r["deal_id"], "exit_date"], "D")
                        if xd > t1:
                            r.update({f"ret{lagname}": ciq.at[r["deal_id"], "exit_price"] / c1 - 1, f"exit_date{lagname}": xd,
                                      f"exit_rule{lagname}": "Capital IQ monthly close after cancellation"})
                if "exit_date5" in r or "exit_date" in r:
                    end = r.get("exit_date5", r.get("exit_date"))
                    daily.append((r["deal_id"], s, t1, min(end, s.dates[-1]), np.nan))
        rec.append(r)
    PF = pd.DataFrame(rec)
    for c in ("P0", "close_t1", "days_to_t1", "ratio_p0", "spread", "p_mkt", "_T", "_value1", "ret", "ret5", "exchange_ratio", "acq_ret_to_t1", "delist_payout", "offer_t1"):
        if c not in PF:
            PF[c] = np.nan
    PF["offer_source"] = PF["offer_source"].fillna("") if "offer_source" in PF else ""
    PF["spread_export_offer"] = np.where(PF["cash_deal"], pd.Series(offer, index=PF.index) / PF["close_t1"] - 1, np.nan)
    for c in ("price_valid", "t1_ok"):
        PF[c] = PF[c].fillna(False).astype(bool)
    for c in ("tm1", "t1", "exit_date", "exit_date5"):
        PF[c] = pd.to_datetime(PF[c]) if c in PF else pd.NaT
    rf = np.nan_to_num(MC.risk_free(PF["t1"]), nan=0.0)
    pv = PF["_value1"] / (1 + rf * PF["_T"])
    PF["p_mkt_carry"] = np.where(pv > PF["P0"], np.clip((pv - PF["close_t1"]) / (pv - PF["P0"]), 0, 1), np.nan)
    PF["p_mkt_carry"] = PF["p_mkt_carry"].where(PF["p_mkt"].notna())
    PF = PF.drop(columns=["_T", "_value1"])
    PF["hold_days"] = (PF["exit_date"] - PF["t1"]).dt.days.where(lambda x: x >= 1)
    PF["hold_days5"] = (PF["exit_date5"] - PF["t1"]).dt.days.where(lambda x: x >= 1)
    sane = PF["spread"].between(-0.5, 1.0, inclusive="neither") & PF["ret"].notna() & PF["hold_days"].notna()
    PF["backtest_ok"] = sane & PF["price_valid"]
    PF = PF.merge(pre_features(link, cal), on="deal_id", how="left")
    PF.to_parquet(INTERIM / "price_features_crsp.parquet", index=False)

    rows = []
    for deal_id, s, a, e, _ in daily:
        i, j = np.searchsorted(s.dates, a, side="right"), np.searchsorted(s.dates, e, side="right")
        if j > i:
            rows.append(pd.DataFrame({"deal_id": deal_id, "date": s.dates[i:j].astype("datetime64[ns]"), "ret": np.nan_to_num(s.ret[i:j]), "source": s.src[i:j]}))
    pd.concat(rows, ignore_index=True).to_parquet(INTERIM / "deal_daily_returns.parquet", index=False)

    report(d, link, PF, T, crsp_end, ciq)


def report(d, link, PF, T, crsp_end, ciq) -> None:
    label = d["label"].to_numpy()
    reverse = d["secondary_features"].fillna("").astype(str).str.contains("Reverse Merger").to_numpy()
    cls = np.where(d["outcome3"].notna().to_numpy(), d["outcome3"].astype(object).to_numpy(), np.where(label == 1, "adverse_break", "closed"))
    per = np.where(d["year"] <= 2012, "2001-2012", np.where(d["year"] <= 2018, "2013-2018", "2019-2025"))
    core = ~reverse
    cash = PF["cash_deal"].to_numpy()
    ok = PF["backtest_ok"].to_numpy()
    L = ["# Daily prices from CRSP: coverage, checks, day-1 spread", "",
         "Source: WRDS. CRSP daily stock file (`crsp.dsf_v2`, current format, prices and total returns including the delisting return) for linked targets and acquirers; "
         "Compustat daily security prices (`comp.secd`) as the second source. Licensed data: this file contains aggregates only. "
         f"CRSP ends on {pd.Timestamp(crsp_end).date()}.", "",
         "Entry is at the close of t1, the first trading day after the announcement date. P0 is the close on the last trading day before the announcement. "
         "A deal is usable when it is still open at t1, the price before announcement agrees within 15% with the price implied by the export's offer and "
         "one-day premium, the entry spread lies between -50% and +100%, and an exit exists.", "",
         "## Coverage by period and outcome (core sample)", "",
         "| Period | Outcome | Deals | Linked to CRSP | Close at t1 and deal still open | Price agrees with the export | All-cash | Usable all-cash | Usable, any consideration |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    cov = pd.DataFrame({"per": per, "cls": cls, "linked": link["permno"].notna().to_numpy(), "t1": PF["t1_ok"].to_numpy(), "valid": PF["price_valid"].to_numpy(),
                        "cash": cash, "bt_cash": ok & cash, "bt_all": ok})[core]
    for (a, b), g in cov.groupby(["per", "cls"], sort=True):
        L.append(f"| {a} | {b} | {len(g)} | {g['linked'].mean():.0%} | {g['t1'].mean():.0%} | {g['valid'].mean():.0%} | {int(g['cash'].sum())} | {int(g['bt_cash'].sum())} | {int(g['bt_all'].sum())} |")
    tot = {}
    for name, m in (("test period 2019-2025", d["year"].to_numpy() >= 2019), ("pooled 2013-2025", d["year"].to_numpy() >= 2013), ("training period 2001-2018", d["year"].to_numpy() <= 2018)):
        mm = core & ok & cash & m
        tot[name] = (int(mm.sum()), int((mm & (cls == "adverse_break")).sum()), int((mm & (cls == "topped")).sum()))
    L += ["", "Usable all-cash deals: " + "; ".join(f"{k}: {v[0]} deals, **{v[1]} adverse breaks**, {v[2]} topped" for k, v in tot.items()) + ".", ""]
    if ciq is not None:
        c_ok = ciq["backtest_ok"].reindex(d["deal_id"]).fillna(False).to_numpy() & ciq["cash_deal"].reindex(d["deal_id"]).fillna(False).to_numpy()
        mm = core & c_ok & (d["year"].to_numpy() >= 2019)
        L += [f"For comparison, the Capital IQ monthly closes gave {int(mm.sum())} usable all-cash test deals with {int((mm & (cls == 'adverse_break')).sum())} adverse breaks. "
              "Daily data helps mainly because a deal no longer needs a monthly grid date between announcement and outcome, and because entry is one day after the announcement.", ""]
    src = PF.loc[core & ok, "price_source"].value_counts()
    rules = pd.concat([PF.loc[core & ok & (label == 0), "exit_rule"], PF.loc[core & ok & (label == 1), "exit_rule"]]).value_counts()
    L += ["## Price source and exit rule (usable core deals)", "",
          "Entry price source: " + ", ".join(f"{k} {v}" for k, v in src.items()) + ".", "",
          "Exit rule: " + ", ".join(f"{k} {v}" for k, v in rules.items()) + ".", "",
          f"Deals whose outcome date is after the end of CRSP: {int((core & (pd.to_datetime(link['outcome_date']).to_numpy().astype(D) > crsp_end)).sum())}. Completed cash deals among them are paid "
          "the offer on the closing date; broken deals among them exit on Compustat daily prices or, failing that, on the Capital IQ monthly close.", ""]
    # ---- link check by method
    L += ["## Does the CRSP price before announcement agree with the export?", "",
          "| Link method | Deals with both prices | Median CRSP P0 / implied price | Within 2% | Within 15% |", "|---|---:|---:|---:|---:|"]
    lm = link["link_method"].replace("", "none (Compustat prices only)").to_numpy()
    rr = PF["ratio_p0"].to_numpy()
    for m in pd.unique(lm):
        k = (lm == m) & np.isfinite(rr)
        if k.sum():
            L.append(f"| {m} | {int(k.sum())} | {np.median(rr[k]):.3f} | {(np.abs(rr[k] - 1) <= 0.02).mean():.0%} | {(np.abs(rr[k] - 1) <= TOL).mean():.0%} |")
    k = np.isfinite(rr)
    L += ["", f"Overall {(np.abs(rr[k] - 1) <= 0.02).mean():.0%} of linked deals agree within 2% and {(np.abs(rr[k] - 1) <= TOL).mean():.0%} within 15%. "
          f"Of the deals outside 15%, {int((label[k] == 1)[np.abs(rr[k] - 1) > TOL].sum())} are broken deals. Deals outside the tolerance are excluded from all price-based analyses.", ""]
    # ---- cross-check with Capital IQ monthly closes
    pp = INTERIM / "prices.parquet"
    if pp.exists():
        m = pd.read_parquet(pp).merge(link[["deal_id", "permno"]].dropna(), on="deal_id")
        m["date"] = m["date"].to_numpy().astype(D)
        diffs, per_deal = [], {}
        for (deal, p), g in m.groupby(["deal_id", "permno"], sort=False):
            s = T.get(int(p))
            if s is None:
                continue
            for dt, cl in zip(g["date"].to_numpy(), g["close"].to_numpy()):
                if dt < s.dates[0] or dt > s.dates[-1] or not np.isfinite(cl) or cl <= 0:
                    continue
                cp, _ = s.last_before(dt, 4)
                if np.isfinite(cp):
                    diffs.append(cl / cp - 1); per_deal.setdefault(deal, []).append(abs(cl / cp - 1))
        diffs = np.array(diffs)
        allok = np.mean([max(v) <= 0.02 for v in per_deal.values()]) if per_deal else np.nan
        L += ["## Cross-check against the Capital IQ monthly closes", "",
              f"{len(diffs):,} month-end-style observations (the 20th of each month, or the prior trading day) from {len(per_deal):,} deals fall inside the CRSP window of the same deal. "
              f"Capital IQ close / CRSP close - 1: median {np.median(diffs):+.4f}; within 0.5%: {(np.abs(diffs) <= 0.005).mean():.1%}; within 2%: {(np.abs(diffs) <= 0.02).mean():.1%}; "
              f"more than 10% apart: {(np.abs(diffs) > 0.10).mean():.1%}. For {allok:.0%} of deals every observation agrees within 2%. "
              "Large differences are split-adjusted Capital IQ histories and a few wrong links; the 15% rule above removes the affected deals.", ""]
    # ---- delisting payout vs offer
    k = core & ok & cash & (label == 0) & np.isfinite(PF["delist_payout"].to_numpy()) & (PF["delist_payout"].to_numpy() > 0)
    if k.sum():
        dv = PF["delist_payout"].to_numpy()[k] / d["offer_per_share"].to_numpy(float)[k] - 1
        L += ["## Cross-check of the amount paid at closing (completed all-cash deals)", "",
              f"For {int(k.sum()):,} completed all-cash deals with a CRSP delisting payment: payment / export offer - 1 has median {np.median(dv):+.2%}, "
              f"5th to 95th percentile {np.percentile(dv, 5):+.1%} to {np.percentile(dv, 95):+.1%}; {(np.abs(dv) > 0.05).mean():.1%} differ by more than 5% "
              "(revised offers and contingent payments). The backtest uses the CRSP payment, the spread uses the export's offer.", ""]
    # ---- which offer
    use = core & ok & cash
    osrc = PF.loc[use, "offer_source"].value_counts()
    se, sn = PF["spread_export_offer"].to_numpy(), PF["spread"].to_numpy()
    diff = use & (PF["offer_source"].to_numpy() == "press release (differs from the export)")
    L += ["## Which offer price is known on day 1?", "",
          "The export stores the final terms of a deal. If the price was raised later, the final offer divided by the day-1 close looks like a very wide spread "
          "that no investor could have earned as a spread. For cash deals the per-share price in the announcement press release is therefore used when it differs "
          "from the export:", "",
          "| Offer used at t1 | Usable all-cash deals |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in osrc.items()]
    L += ["", f"In the {int(diff.sum())} deals where the press release differs, the median press-release offer is {np.nanmedian(PF['offer_t1'].to_numpy()[diff] / d['offer_per_share'].to_numpy(float)[diff]):.2f} "
          f"times the export's. Usable all-cash deals with an entry spread above 25%: {int((use & (se > 0.25)).sum())} with the export's offer, {int((use & (sn > 0.25)).sum())} with the announcement-day offer. "
          "Deals without announcement text keep the export's offer and can still carry this error; the backtest therefore also shows a panel restricted to spreads between -5% and +25%.", ""]
    # ---- spread and p_mkt
    S = pd.DataFrame({"per": np.where(d["year"] <= 2018, "train 2001-2018", "test 2019-2025"), "cls": cls, "spread": PF["spread"], "p": PF["p_mkt"], "d": PF["days_to_t1"],
                      "ret": PF["ret"], "hold": PF["hold_days"]})[use]
    L += ["## Day-1 spread and market-implied break probability (usable all-cash deals)", "",
          "| Period | Outcome | Deals | Median spread | Mean spread | Median p_mkt | Mean p_mkt | Mean return to exit | Median holding days |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (a, b), g in S.groupby(["per", "cls"], sort=True):
        L.append(f"| {a} | {b} | {len(g)} | {g.spread.median():.1%} | {g.spread.mean():.1%} | {g.p.median():.2f} | {g.p.mean():.2f} | {g.ret.mean():+.1%} | {g.hold.median():.0f} |")
    L += ["", f"Median calendar days from announcement to t1: {np.nanmedian(PF['days_to_t1']):.0f}. With monthly data it was about two weeks, so the market now has a one-day head start "
          "over the announcement-date model, not a month.", ""]
    st = core & ok & ~cash
    L += ["## Stock and mixed deals (robustness only)", "",
          f"{int(st.sum())} usable stock and mixed deals have an acquirer price ({int((st & (cls == 'adverse_break')).sum())} adverse breaks). The exchange ratio is implied from the export's "
          "stock part of the offer and the acquirer's close before announcement, which assumes a fixed ratio (no collars, no fixed-value deals) and that the "
          "export's offer value was struck at that close. The backtest return for these deals is the unhedged target return.", ""]
    (ROOT / "results/crsp_prices.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

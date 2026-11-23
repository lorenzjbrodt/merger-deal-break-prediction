"""Price-based quantities per deal from the Capital IQ monthly closes (target shares).

Builds data/interim/price_features.parquet (licensed, never committed) and the aggregate
report results/price_coverage.md.

Definitions
  g0      last grid date strictly before the announcement
  t1      first grid date strictly after the announcement and strictly before the outcome date
  spread  offer per share / close(t1) - 1
  P0      offer per share / (1 + premium 1 day prior)        (pre-announcement price)
  p_mkt   (offer - close(t1)) / (offer - P0), clipped to [0, 1]
          Reading: if the deal breaks the price falls back to P0, if it closes it pays the offer,
          so the discount to the offer as a share of the premium is the market's break probability.
          Ignores time value and assumes a fall back to exactly P0.
  p_mkt_carry  same with the offer discounted at the 3-month T-bill rate over the expected
          remaining time to closing (training-period median duration minus days elapsed, floor 30 days)
  validity     |close(g0) / implied pre-announcement price - 1| <= 15%, where the implied price is
          offer / (1 + premium) for the premium horizon closest in time to g0 (1 day, 1 week, 1 month)
Backtest legs (used by backtest.py)
  entry   close(t1)
  exit    closed cash deal: the offer price on the closing date
          closed stock / mixed deal: last available close on or before the closing date (+31 days)
          broken deal: first grid close strictly after the cancellation date (else the last available close)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import market_context as MC

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
TOL = 0.15


def main() -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    o = pd.read_parquet(INTERIM / "outcomes.parquet")[["deal_id", "outcome3", "revised_terms"]]
    d = d.merge(o, on="deal_id", how="left")
    p = pd.read_parquet(INTERIM / "prices.parquet")
    W = p[p["deal_id"].isin(d["deal_id"])].pivot(index="deal_id", columns="date", values="close").reindex(d["deal_id"])
    dates = W.columns.to_numpy().astype("datetime64[ns]")
    A = W.to_numpy(dtype=float)
    n = len(d)
    rows = np.arange(n)
    ann = d["announced_date"].to_numpy().astype("datetime64[ns]")
    out = d["closed_date"].fillna(d["cancelled_date"]).to_numpy().astype("datetime64[ns]")
    label = d["label"].to_numpy()
    offer = d["offer_per_share"].to_numpy(dtype=float)

    i0 = np.searchsorted(dates, ann, side="left") - 1
    i1 = np.searchsorted(dates, ann, side="right")
    ok0, ok1 = i0 >= 0, i1 < len(dates)
    pre = np.where(ok0, A[rows, np.clip(i0, 0, None)], np.nan)
    g0 = np.where(ok0, dates[np.clip(i0, 0, None)], np.datetime64("NaT", "ns"))
    t1 = np.where(ok1, dates[np.clip(i1, 0, len(dates) - 1)], np.datetime64("NaT", "ns"))
    c1 = np.where(ok1, A[rows, np.clip(i1, 0, len(dates) - 1)], np.nan)
    t1_ok = ok1 & (t1 < out)
    c1 = np.where(t1_ok, c1, np.nan)

    days_g0 = (ann - g0) / np.timedelta64(1, "D")
    imp = {k: offer / (1 + d[c].to_numpy(dtype=float) / 100) for k, c in (("1d", "premium_1d"), ("1w", "premium_1w"), ("1m", "premium_1m"))}
    ratio_1m = pre / imp["1m"]
    ratio_ref = pre / np.where(days_g0 <= 3, imp["1d"], np.where(days_g0 <= 12, imp["1w"], imp["1m"]))
    valid = np.abs(ratio_ref - 1) <= TOL
    valid_1m = np.abs(ratio_1m - 1) <= TOL

    cons = d["consideration"].fillna("").astype(str)
    cash = (d["cash_pct"].to_numpy(dtype=float) == 100) | ((cons == "Cash").to_numpy() & d["cash_pct"].isna().to_numpy())

    spread = offer / c1 - 1
    P0 = imp["1d"]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mkt = np.where(offer > P0, (offer - c1) / (offer - P0), np.nan)
    p_mkt = np.clip(p_mkt, 0, 1)
    days_t1 = (t1 - ann) / np.timedelta64(1, "D")
    train_closed = (d["year"].to_numpy() <= 2018) & (label == 0)
    med_dur = float(np.median(((out - ann) / np.timedelta64(1, "D"))[train_closed]))
    T = np.maximum(med_dur - days_t1, 30) / 365.0
    rf = MC.risk_free(pd.Series(t1))
    offer_pv = offer / (1 + np.nan_to_num(rf, nan=0.0) * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_carry = np.where(offer_pv > P0, (offer_pv - c1) / (offer_pv - P0), np.nan)
    p_carry = np.clip(p_carry, 0, 1)

    # ---- exits
    last_idx = np.array([np.max(np.where(np.isfinite(A[r]) & (dates <= out[r] + np.timedelta64(31, "D")))[0], initial=-1) for r in rows])
    last_close = np.where(last_idx >= 0, A[rows, np.clip(last_idx, 0, None)], np.nan)
    last_date = np.where(last_idx >= 0, dates[np.clip(last_idx, 0, None)], np.datetime64("NaT", "ns"))
    ia = np.searchsorted(dates, out, side="right")
    after_ok = ia < len(dates)
    after_close = np.where(after_ok, A[rows, np.clip(ia, 0, len(dates) - 1)], np.nan)
    after_date = np.where(after_ok, dates[np.clip(ia, 0, len(dates) - 1)], np.datetime64("NaT", "ns"))
    any_last_idx = np.array([np.max(np.where(np.isfinite(A[r]))[0], initial=-1) for r in rows])
    any_last = np.where(any_last_idx >= 0, A[rows, np.clip(any_last_idx, 0, None)], np.nan)
    any_last_date = np.where(any_last_idx >= 0, dates[np.clip(any_last_idx, 0, None)], np.datetime64("NaT", "ns"))

    exit_price = np.full(n, np.nan); exit_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"); exit_rule = np.full(n, "", dtype=object)
    m = (label == 0) & cash
    exit_price[m], exit_date[m], exit_rule[m] = offer[m], out[m], "offer at closing"
    m = (label == 0) & ~cash
    exit_price[m], exit_date[m], exit_rule[m] = last_close[m], last_date[m], "last close before delisting"
    m = (label == 1) & np.isfinite(after_close)
    exit_price[m], exit_date[m], exit_rule[m] = after_close[m], after_date[m], "first close after cancellation"
    m = (label == 1) & ~np.isfinite(after_close) & np.isfinite(any_last) & (any_last_date > t1)
    exit_price[m], exit_date[m], exit_rule[m] = any_last[m], any_last_date[m], "last available close"
    ret = exit_price / c1 - 1
    hold = (exit_date - t1) / np.timedelta64(1, "D")
    hold = np.where(hold < 1, np.nan, hold)

    sane = np.isfinite(spread) & (spread > -0.5) & (spread < 1.0) & np.isfinite(ret) & np.isfinite(hold)
    PF = pd.DataFrame({
        "deal_id": d["deal_id"], "pre_close": pre, "ratio_ref": ratio_ref, "ratio_1m": ratio_1m, "price_valid": valid,
        "price_valid_1m_rule": valid_1m, "cash_deal": cash, "t1": t1, "close_t1": c1, "days_to_t1": np.where(t1_ok, days_t1, np.nan),
        "spread": spread, "p_mkt": p_mkt, "p_mkt_carry": p_carry, "exit_price": exit_price, "exit_date": exit_date,
        "exit_rule": exit_rule, "ret": ret, "hold_days": hold, "offer_vs_last_close": offer / last_close - 1,
        "backtest_ok": sane & valid,
    })
    PF.to_parquet(INTERIM / "price_features.parquet", index=False)

    # ---------------------------------------------------------------- report (aggregates only)
    grid = json.loads((ROOT / "results/price_grid_check.json").read_text())
    per = np.where(d["year"] <= 2018, "train 2001-2018", "test 2019-2025")
    cls = np.where(d["outcome3"].notna().to_numpy(), d["outcome3"].astype(object).to_numpy(), np.where(label == 1, "adverse_break", "closed"))
    has_any = np.isfinite(A).any(1)
    L = ["# Price data: coverage and checks", "",
         "Source: Capital IQ transaction screen with the display columns reduced to IDs plus the target's `Day Close Price` on a "
         "monthly grid (the 20th of each month). Three exports, parsed by `src/prices.py`. Licensed data: only aggregates appear here.", "",
         "## Grid", "",
         f"- {grid['n_grid_dates']} monthly dates from {grid['grid_first']} to {grid['grid_last']}; overlapping dates between files: "
         f"{len(grid['overlapping_dates'])}; missing months: {len(grid['missing_months'])}.",
         f"- {grid['grid_dates_on_weekend']} grid dates fall on a weekend. When the months before and after are priced, the weekend date is "
         f"priced in {grid.get('weekend_20th_filled_share_when_neighbours_priced', float('nan')):.1%} of cases, so a non-trading 20th returns the prior close.",
         f"- {int(has_any.sum()):,} of {n:,} sample deals have at least one price.", ""]
    both = np.isfinite(ratio_1m)
    bad = both & ~valid_1m
    near = {k: int((np.abs(ratio_1m[bad] / k - 1) < 0.08).sum()) for k in (0.1, 0.2, 0.25, 1 / 3, 0.5, 2, 3, 4, 5, 10)}
    L += ["## Is the close consistent with the export's offer and premium?", "",
          f"For {int(both.sum()):,} deals both the last monthly close before announcement and the implied pre-announcement price "
          "(offer per share / (1 + premium 1 month prior)) exist. Median ratio close / implied: "
          f"{np.nanmedian(ratio_1m):.3f}; within 15%: {int(valid_1m.sum()):,} ({valid_1m.sum() / both.sum():.0%}).", "",
          "The grid close can be up to a month away from the 1-month-prior reference, so part of the mismatch is ordinary price movement "
          f"(for example a run-up before the announcement). Using the premium horizon closest in time to the grid date (1 day, 1 week or 1 month) "
          f"raises the share within 15% to {int(valid.sum()):,} ({valid.sum() / np.isfinite(ratio_ref).sum():.0%}). "
          "The price-based analyses use this second rule; the count under the literal 1-month rule is shown for transparency.", "",
          f"Split adjustment: of the {int(bad.sum())} deals outside 15% under the 1-month rule, only "
          f"{sum(near.values())} sit within 8% of a clean split factor (2:1 {near[2]}, 1:2 {near[0.5]}, 10:1 {near[10]}, 5:1 {near[5]}, others {sum(near.values()) - near[2] - near[0.5] - near[10] - near[5]}), "
          f"and {int(((label == 1) & bad).sum())} of the {int(bad.sum())} are cancelled deals, whose targets kept trading. "
          "The closes therefore look like unadjusted historical prices for the large majority of deals, with a small number of adjusted or mis-matched series. "
          "All deals outside the tolerance are excluded from price-based analyses, which removes those series whatever the cause.", ""]
    L += ["## Coverage by outcome class and period", "",
          "| Period | Outcome | Deals | Any price | t1 exists (grid date between announcement and outcome) | Price valid | All-cash | Usable all-cash (backtest) | Usable any consideration |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    cov = pd.DataFrame({"per": per, "cls": cls, "any": has_any, "t1": t1_ok & np.isfinite(c1), "valid": valid, "cash": cash,
                        "bt_cash": PF["backtest_ok"].to_numpy() & cash, "bt_all": PF["backtest_ok"].to_numpy()})
    for (a, b), g in cov.groupby(["per", "cls"], sort=True):
        L.append(f"| {a} | {b} | {len(g)} | {g['any'].mean():.0%} | {g['t1'].mean():.0%} | {g['valid'].mean():.0%} | {g['cash'].mean():.0%} | {int(g['bt_cash'].sum())} | {int(g['bt_all'].sum())} |")
    closed = label == 0
    delisted = closed & has_any & (any_last_date <= out + np.timedelta64(62, "D"))
    L += ["", f"Completed deals with delisted targets are covered: {int(delisted.sum()):,} of {int((closed & has_any).sum()):,} closed deals with prices "
          "stop being priced within two months of the closing date, and their pre-closing closes are present. "
          f"Deals with no grid date between announcement and outcome (closed or cancelled within weeks) cannot be entered: {int((~t1_ok).sum()):,} deals.", ""]
    ck = closed & cash & PF["backtest_ok"].to_numpy() & np.isfinite(PF["offer_vs_last_close"].to_numpy())
    dv = PF["offer_vs_last_close"].to_numpy()[ck]
    L += ["## Cross-check of the exit price for completed cash deals", "",
          f"For {int(ck.sum()):,} completed all-cash deals, offer per share versus the last available close before delisting: median difference "
          f"{np.median(dv):+.2%}, 5th to 95th percentile {np.percentile(dv, 5):+.1%} to {np.percentile(dv, 95):+.1%}; "
          f"{(np.abs(dv) > 0.05).mean():.1%} differ by more than 5% (revised offers, contingent consideration or stale marks).", ""]
    use = PF["backtest_ok"].to_numpy() & cash
    L += ["## Spread at t1 and market-implied break probability (usable all-cash deals)", "",
          "| Period | Outcome | Deals | Median spread | Mean spread | Median p_mkt | Mean p_mkt | Median days announcement to t1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    S = pd.DataFrame({"per": per, "cls": cls, "spread": spread, "p": p_mkt, "d": days_t1})[use]
    for (a, b), g in S.groupby(["per", "cls"], sort=True):
        L.append(f"| {a} | {b} | {len(g)} | {g.spread.median():.1%} | {g.spread.mean():.1%} | {g.p.median():.2f} | {g.p.mean():.2f} | {g.d.median():.0f} |")
    L += ["", "t1 can be up to a month after the announcement, so the market price already reflects a month of news that the announcement-date "
          "model does not see. This favours the market benchmark.", ""]
    (ROOT / "results/price_coverage.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

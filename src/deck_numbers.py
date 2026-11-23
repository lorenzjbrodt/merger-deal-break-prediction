"""Read-only export of aggregates for the slide deck: results/deck_numbers.json.

Nothing is fitted and no result file is changed. Every number is recomputed from the cached per-deal files with the
functions of the frozen scripts and checked against the published result files before it is written; the script
stops if a check fails. Only aggregates are written (portfolio values, group means, counts).

  1. Calendar-time portfolios, pooled 2013-2026: month-end value of $100               [expected_return.py]
  2. Ranking test: realised annual return of each third with bootstrap intervals        [expected_return.py]
  3. Loss if a deal breaks: mean loss implied by a fall back to the unaffected price    [loss_model.py]
  4. Core sample: adverse breaks and topped deals, total, training and test period
  5. Time to closing, test period: mean absolute error of the product model by segment  [duration.py]
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import duration as DU
import expected_return as ER
import modeling as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
RES = ROOT / "results"
SEED = 0


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def dump(o, ind=0):
    """JSON with lists of scalars on one line."""
    pad = " " * ind
    if isinstance(o, dict):
        return "{\n" + ",\n".join(f"{pad}  {json.dumps(k)}: {dump(v, ind + 2)}" for k, v in o.items()) + f"\n{pad}}}"
    if isinstance(o, list) and any(isinstance(v, (dict, list)) for v in o):
        return "[\n" + ",\n".join(f"{pad}  {dump(v, ind + 2)}" for v in o) + f"\n{pad}]"
    return json.dumps(o)


def third_intervals(ret, hold, yr, score, by_year, rng):
    """Realised annual return of each third and a 95% interval; resampling as in expected_return.tercile_table:
    announcement years (pooled) or deals (test period), thirds fixed as assigned in the full sample."""
    t = ER.terciles(score, yr)
    blocks = [np.where(yr == v)[0] for v in np.unique(yr)]
    draws = {k: [] for k in (2, 1, 0)}
    for _ in range(ER.N_BOOT):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) if by_year else rng.integers(0, len(ret), len(ret))
        tt = t[ix]
        for k in draws:
            if (tt == k).sum() > 3:
                draws[k].append(ER.portfolio(ret[ix][tt == k], hold[ix][tt == k]))
    out = {}
    for k, lab in ((2, "top"), (1, "middle"), (0, "bottom")):
        m = t == k
        out[lab] = {"deals": int(m.sum()), "annual_return": round(ER.portfolio(ret[m], hold[m]), 6),
                    "ci95": [round(float(np.percentile(draws[k], 2.5)), 6), round(float(np.percentile(draws[k], 97.5)), 6)]}
    return out


def main() -> None:
    D = M.load_data("")
    assert D.pf_source == "CRSP daily"
    PF = D.PF
    n = len(D.year)
    ids = D.deals["deal_id"].to_numpy()
    pub = json.loads((RES / "expected_return.json").read_text())["views"]
    out = {"about": "Aggregates for the slide deck, exported read-only from the frozen results. Returns and errors are fractions unless the key says otherwise; "
                    "returns are before costs. 'pooled' = deals announced 2013-2025, each year scored by models refitted on earlier years only; "
                    "'test' = deals announced 2019-2025, models trained on 2001-2018."}

    # ---- 1 and 2: rankings as stored by expected_return.py
    EP = pd.read_parquet(INTERIM / "expected_return.parquet").set_index("deal_id").reindex(ids).reset_index()
    ret, hold, spread = PF["ret"].to_numpy(), PF["hold_days"].to_numpy(), PF["spread"].to_numpy()
    t1v = PF["t1"].to_numpy()
    daily = pd.read_parquet(INTERIM / "deal_daily_returns.parquet")
    daily = daily.merge(pd.DataFrame({"deal_id": ids, "exit_date": PF["exit_date"].to_numpy()}), on="deal_id", how="left")
    daily = daily[daily["date"] <= daily["exit_date"]]
    ff = pd.read_parquet(WRDS / "ff_daily.parquet")
    rf = pd.Series(ff["rf"].to_numpy(float), index=pd.to_datetime(ff["date"]))
    out["ranking_test"] = {"about": "All-cash core deals with CRSP prices, sorted into thirds within each announcement year, bought at the day-1 close and held to the outcome. "
                                    "annual_return = total P&L over total capital-years. ci95: bootstrap with 2,000 draws, over announcement years (pooled) and over deals (test), "
                                    f"thirds fixed as assigned in the full sample, seed {SEED}. The point estimates equal results/expected_return.md; the intervals per third are new."}
    for view, by_year in (("pooled", True), ("test", False)):
        ear = EP[f"{view}__ear"].to_numpy()
        rows = np.isfinite(ear) & np.isfinite(ret) & np.isfinite(hold)
        V = pub[view]
        assert rows.sum() == V["n"]
        rng = np.random.default_rng(SEED)
        o = {"deals": int(rows.sum()), "adverse_breaks": V["adverse"], "resampling": "announcement years" if by_year else "deals"}
        for key, lab, score in (("ear", "model", ear), ("spread", "raw_spread", spread)):
            o[lab] = third_intervals(ret[rows], hold[rows], D.year[rows], score[rows], by_year, rng)
            for th in ("top", "middle", "bottom"):
                close(o[lab][th]["annual_return"], V["rankings"][key]["terciles"][th]["annual_return"], 1e-6)
                assert o[lab][th]["deals"] == V["rankings"][key]["terciles"][th]["n"]
        out["ranking_test"][view] = o
        if view != "pooled":
            continue

        def top_half(score):
            th = np.zeros(n, bool)
            for i in np.where(rows)[0]:
                prior = rows & (t1v < t1v[i]) & (t1v >= t1v[i] - np.timedelta64(365, "D"))
                th[i] = (score[i] >= np.nanmedian(score[prior])) if prior.sum() >= 20 else True
            return th
        sets = {"all": rows, "top_model": rows & top_half(ear), "top_spread": rows & top_half(np.where(rows, spread, np.nan))}
        ser = {k: ER.calendar_portfolio(daily, set(ids[m]), rf) for k, m in sets.items()}
        ra = ser["all"][0]
        idxd = pd.DatetimeIndex(sorted(set(rf.index[(rf.index >= ra.index.min()) & (rf.index <= ra.index.max())]) | set(ra.index)))
        rr = {k: v[0].reindex(idxd).fillna(0.0) for k, v in ser.items()}
        C = V["calendar"]
        assert str(idxd.min().date()) == C["start"] and str(idxd.max().date()) == C["end"]
        cal = {"about": "Daily calendar-time portfolios of the pooled set (deals announced 2013-2025, held to the outcome, so the series runs into 2026): equal weights across open "
                        "deals with a 10% position cap, the rest in cash at zero, rebalanced daily, before costs. Value of $100 invested before the first day, at the last "
                        "trading day of each month; the last month ends on the last day of the series. With fewer than 10 open deals part of the portfolio is in cash.",
               "start": C["start"], "end": C["end"], "start_value": 100.0,
               "portfolios": {"all": "All open deals", "top_model": "Top half by expected annualized return (model)", "top_spread": "Top half by raw day-1 spread"},
               "summary": {}, "month": None, "month_end_value": {}, "open_deals_at_month_end": {}}
        for k in sets:
            close(float(rr[k].mean() * 252), C[k]["annual_return"])
            assert int(sets[k].sum()) == C[k]["deals"]
            v = (1 + rr[k]).cumprod() * 100.0
            me = v.groupby(v.index.to_period("M")).last()
            cal["month"] = [str(p) for p in me.index]
            cal["month_end_value"][k] = [round(float(x), 2) for x in me.to_numpy()]
            no = ser[k][1].reindex(idxd).fillna(0)
            cal["open_deals_at_month_end"][k] = [int(x) for x in no.groupby(no.index.to_period("M")).last().to_numpy()]
            cal["summary"][k] = {"deals": C[k]["deals"], "annual_return": round(C[k]["annual_return"], 6), "volatility": round(C[k]["volatility"], 6), "sharpe": round(C[k]["sharpe"], 4),
                                 "max_drawdown": round(C[k]["max_drawdown"], 6), "final_value": round(float(v.iloc[-1]), 2)}
        out["calendar_time_portfolios_pooled"] = cal

    # ---- 3: fall back to the unaffected price, same sample as loss_model.py
    fb = (PF["P0"] / PF["close_t1"] - 1).to_numpy(float)
    ok = D.core & PF["t1_ok"].fillna(False).to_numpy() & PF["price_valid"].fillna(False).to_numpy() & np.isfinite(fb) & (fb > -0.9) & (fb < 1.0)
    m = ok & (D.outcome3 == "adverse_break") & np.isfinite(ret)
    LG = json.loads((RES / "loss_given_break.json").read_text())["realised"]["adverse_break | all years"]
    assert m.sum() == LG["n"]
    close(float(ret[m].mean()), LG["mean"])
    out["loss_if_break"] = {"about": "Adverse breaks in the core sample with CRSP prices. Fall-back = pre-announcement close over the day-1 close minus 1 "
                                     "(the loss if the price returned to its level before the announcement). Realised = day-1 close to the close one trading day after the cancellation.",
                            "adverse_breaks": int(m.sum()), "mean_fallback_pct": round(float(fb[m].mean()) * 100, 2), "median_fallback_pct": round(float(np.median(fb[m])) * 100, 2),
                            "mean_realised_pct": round(float(ret[m].mean()) * 100, 2)}

    # ---- 4: core sample counts
    SF = json.loads((RES / "sample_filters.json").read_text())
    o3 = D.outcome3
    per = {"total": np.ones(n, bool), "train_2001_2018": D.year <= M.TRAIN_END, "test_2019_2025": D.year >= M.TEST_START}
    out["core_sample"] = {"about": "Core sample = signed deals without reverse mergers. Adverse break = cancelled for a reason other than a higher competing bid.",
                          "all_signed_deals_before_excluding_reverse_mergers": {"deals": int(n), "adverse_breaks": int((o3 == "adverse_break").sum()), "topped_by_higher_bid": int((o3 == "topped").sum()),
                                                                                "completed": int((o3 == "closed").sum())},
                          **{k: {"deals": int((D.core & p).sum()), "adverse_breaks": int((D.core & p & (o3 == "adverse_break")).sum()), "topped_by_higher_bid": int((D.core & p & (o3 == "topped")).sum()),
                                 "completed": int((D.core & p & (o3 == "closed")).sum())} for k, p in per.items()}}
    c = out["core_sample"]
    assert n == SF["n_deals"] and c["all_signed_deals_before_excluding_reverse_mergers"]["adverse_breaks"] + c["all_signed_deals_before_excluding_reverse_mergers"]["topped_by_higher_bid"] == SF["n_breaks"]
    assert c["total"]["deals"] == c["train_2001_2018"]["deals"] + c["test_2019_2025"]["deals"]
    assert all(c[k]["deals"] == c[k]["adverse_breaks"] + c[k]["topped_by_higher_bid"] + c[k]["completed"] for k in per)

    # ---- 5: time to closing by segment, product model, test period
    P = pickle.load(open(INTERIM / "duration.pkl", "rb"))["pred"]
    DP = pd.read_parquet(INTERIM / "duration_product.parquet").set_index("deal_id").reindex(P["deal_id"]).reset_index()
    done = (P["core"] & P["closed"]).to_numpy()
    okd = done & DP["split__mu"].notna().to_numpy() & P["split__full__final"].notna().to_numpy()
    X, mu = P[okd], DP["split__mu"].to_numpy()[okd]
    days, bench, fin = X["days"].to_numpy(), X["split__full__median_structure"].to_numpy(), X["split__full__final"].to_numpy()
    sic = (X["sic"].fillna(0).astype(int) // 100).to_numpy()
    tend, xb = X["tender"].to_numpy(bool), X["cross_border"].to_numpy(bool)
    segs = {"all completed deals": np.ones(len(X), bool), "mergers": ~tend, "tender offers": tend, "bank deals (target SIC 60-61)": np.isin(sic, [60, 61]), "cross-border deals": xb}
    PD = json.loads((RES / "duration.json").read_text())
    seg_out = {}
    for lab, s in segs.items():
        a, b, f = DU.met(days[s], bench[s]), DU.met(days[s], mu[s]), DU.met(days[s], fin[s])
        seg_out[lab] = {"deals": int(s.sum()), "median_days": float(np.median(days[s])), "mae_days_deal_type_median": round(a[0], 2), "mae_days_product_model": round(b[0], 2),
                        "mae_days_model_without_guidance": round(f[0], 2), "within_30_days_product_model": round(b[3], 4)}
    close(DU.met(days, mu)[0], PD["product"]["split"]["scores"]["product"]["mae"])
    assert len(X) == PD["product"]["split"]["n"]
    for lab, key in (("mergers", "Structure: merger"), ("tender offers", "Structure: tender offer"), ("bank deals (target SIC 60-61)", "Regulated industry: banks"), ("cross-border deals", "Cross-border: cross-border")):
        assert seg_out[lab]["deals"] == PD["segments"][key]["n"]
        close(seg_out[lab]["mae_days_deal_type_median"], PD["segments"][key]["mae_benchmark"], 0.006)
    out["time_to_closing_test_segments"] = {"about": "Completed core deals announced 2019-2025, models trained on 2001-2018. Mean absolute error in calendar days. Deal-type median = "
                                                     "training-period median of tender offers or of mergers. Product model = also reads the stated expected closing where there is one. "
                                                     "Segments overlap (a bank deal is also a merger or a tender offer).", **seg_out}
    (RES / "deck_numbers.json").write_text(dump(out) + "\n")
    print((RES / "deck_numbers.json").read_text())


if __name__ == "__main__":
    main()

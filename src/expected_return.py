"""Expected annualized return per deal on day 1, and whether ranking deals by it makes money out of sample.

Per all-cash deal at t1 (first trading day after the announcement), from out-of-time components only:
  P(adverse break)   calibrated break model (variant: model + market stack)                      [daily_eval.py]
  P(topped)          small logistic regression on the day-1 spread, p_mkt and deal size, fitted on earlier years
  P(close)           1 - P(adverse break) - P(topped)
  time to closing    time-to-closing model, mean of the predicted log-normal                      [duration.py]
  loss if it breaks  loss model                                                                  [loss_model.py]
  bump if topped     mean extra return of topped deals over their day-1 spread, earlier years
  expected return            = P(close) x spread + P(topped) x (spread + bump) + P(adverse) x loss
  expected time to resolution = P(close) x time to closing + P(topped) x mean time to a topping bid + P(adverse) x mean time to a break
  expected annualized return  = expected return / (expected time / 365)
Two evaluation sets: pooled rolling origin 2013-2025 (every component fitted on earlier years only) and the test
period 2019-2025 (components fitted on 2001-2018).

Tests: terciles of the ranking within each announcement year; a calendar-time portfolio on daily CRSP returns
(all open deals versus the top half by rank at entry); attribution (each component replaced by its unconditional
average in turn). Ranking benchmarks: raw spread, spread annualized with the median duration by structure, and the
market-implied probability alone.

Outputs: results/expected_return.md, .json, figures/expected_return.png, data/interim/expected_return.parquet.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import modeling as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
YEARS = list(range(2013, 2026))
CAPITAL = 100e6
N_BOOT = 2000
RANKS = {"ear": "Expected annualized return (model)", "ear_stack": "Expected annualized return (break probability from model + market)",
         "spread": "Benchmark: raw day-1 spread", "spread_ann": "Benchmark: spread annualized with the median duration by structure",
         "mkt": "Benchmark: lowest market-implied break probability", "ear_no_break": "Attribution: break probability replaced by its average",
         "ear_no_duration": "Attribution: time to closing replaced by its average", "ear_no_loss": "Attribution: loss if it breaks replaced by its average"}


def portfolio(ret, hold):
    cy = np.sum(hold) / 365.0
    return float(np.sum(ret) / cy) if cy > 0 else float("nan")


def terciles(score, year, min_n=6):
    t = np.full(len(score), np.nan)
    for v in np.unique(year):
        g = np.where((year == v) & np.isfinite(score))[0]
        if len(g) >= min_n:
            r = pd.Series(score[g]).rank(method="first").to_numpy()
            t[g] = np.minimum((r - 1) * 3 // len(g), 2)
    return t


def components(E, cut_rows, te, p_adv, dur_mu, dur_sd, loss_pred):
    """Expected-return components for rows `te` from training rows `cut_rows` (boolean arrays)."""
    a = cut_rows
    sp = E["spread"].to_numpy()
    cls = E["cls"].to_numpy()
    ret, hold = E["ret"].to_numpy(), E["hold"].to_numpy()
    out = {}
    # P(topped): tiny logistic on day-1 market variables
    Z = np.column_stack([np.clip(sp, -0.2, 0.6), E["p_mkt"].fillna(0).to_numpy(), E["log_value"].fillna(E["log_value"].median()).to_numpy()])
    yt = (cls == "topped").astype(int)
    if yt[a].sum() >= 8:
        out["p_top"] = LogisticRegression(C=1.0, max_iter=1000).fit(Z[a], yt[a]).predict_proba(Z[te])[:, 1]
    else:
        out["p_top"] = np.full(te.sum(), yt[a].mean())
    out["p_adv"] = np.clip(p_adv[te], 1e-4, 0.95)
    out["p_close"] = np.clip(1 - out["p_adv"] - out["p_top"], 0.0, 1.0)
    adv, top, clo = a & (cls == "adverse_break"), a & (cls == "topped"), a & (cls == "closed")
    out["bump"] = float(np.mean(ret[top] - sp[top])) if top.sum() >= 5 else 0.0
    loss_bar = float(np.mean(ret[adv])) if adv.sum() >= 5 else -0.3
    out["loss"] = np.where(np.isfinite(loss_pred[te]), loss_pred[te], loss_bar)
    out["loss_bar"] = loss_bar
    out["p_adv_bar"] = float(np.mean(cls[a] == "adverse_break"))
    t_close = np.exp(dur_mu[te] + 0.5 * dur_sd ** 2) - E["days_to_t1"].fillna(1).to_numpy()[te]
    out["t_close"] = np.clip(t_close, 10, None)
    out["t_close_bar"] = float(np.mean(hold[clo]))
    out["t_top"] = float(np.mean(hold[top])) if top.sum() >= 5 else out["t_close_bar"]
    out["t_adv"] = float(np.mean(hold[adv])) if adv.sum() >= 5 else out["t_close_bar"]
    tend = E["tender"].to_numpy()
    med = {k: float(np.median(hold[clo & (tend == k)])) if (clo & (tend == k)).sum() >= 20 else float(np.median(hold[clo])) for k in (True, False)}
    out["t_med_structure"] = np.where(tend[te], med[True], med[False])
    return out


def ear(c, sp, p_adv=None, t_close=None, loss=None):
    p_adv = c["p_adv"] if p_adv is None else p_adv
    t_close = c["t_close"] if t_close is None else t_close
    loss = c["loss"] if loss is None else loss
    p_close = np.clip(1 - p_adv - c["p_top"], 0.0, 1.0)
    er = p_close * sp + c["p_top"] * (sp + c["bump"]) + p_adv * loss
    et = np.clip(p_close * t_close + c["p_top"] * c["t_top"] + p_adv * c["t_adv"], 15, None)
    return er, et, er / (et / 365.0)


def tercile_table(E, rows, score, by_year, rng):
    X = E[rows]
    ret, hold, yr, cls = X["ret"].to_numpy(), X["hold"].to_numpy(), X["year"].to_numpy(), X["cls"].to_numpy()
    t = terciles(score[rows], yr)
    res = {"terciles": {}}
    for k, lab in ((2, "top"), (1, "middle"), (0, "bottom")):
        m = t == k
        res["terciles"][lab] = {"n": int(m.sum()), "annual_return": portfolio(ret[m], hold[m]), "mean_ret": float(ret[m].mean()), "adverse_rate": float((cls[m] == "adverse_break").mean()),
                                "adverse": int((cls[m] == "adverse_break").sum()), "topped": int((cls[m] == "topped").sum()), "median_hold": float(np.median(hold[m])),
                                "mean_score": float(np.nanmean(score[rows][m]))}
    blocks = [np.where(yr == v)[0] for v in np.unique(yr)]
    ds, da = [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) if by_year else rng.integers(0, len(X), len(X))
        tt = t[ix]
        if (tt == 2).sum() > 3 and (tt == 0).sum() > 3:
            ds.append(portfolio(ret[ix][tt == 2], hold[ix][tt == 2]) - portfolio(ret[ix][tt == 0], hold[ix][tt == 0]))
            da.append(portfolio(ret[ix][tt == 2], hold[ix][tt == 2]) - portfolio(ret[ix][np.isfinite(tt)], hold[ix][np.isfinite(tt)]))
    res["top_minus_bottom"] = res["terciles"]["top"]["annual_return"] - res["terciles"]["bottom"]["annual_return"]
    res["top_minus_bottom_ci95"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    allm = np.isfinite(t)
    res["all"] = portfolio(ret[allm], hold[allm])
    res["top_minus_all"] = res["terciles"]["top"]["annual_return"] - res["all"]
    res["top_minus_all_ci95"] = [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]
    res["share_gt0"] = float((np.array(ds) > 0).mean())
    return res


def calendar_portfolio(daily, ids, rf):
    """Equal-weighted daily return of all open positions among `ids` (0 on days without a position)."""
    x = daily[daily["deal_id"].isin(ids)]
    r = x.groupby("date")["ret"].mean()
    n = x.groupby("date")["ret"].size()
    return r, n


def port_stats(r, rf):
    ex = r - rf.reindex(r.index).ffill().fillna(0)
    cum = (1 + r).cumprod()
    return {"annual_return": float(r.mean() * 252), "volatility": float(r.std() * np.sqrt(252)), "sharpe": float(ex.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
            "max_drawdown": float((cum / cum.cummax() - 1).min()), "days": int(len(r))}


def block_ci(a, b, rf, rng, block=21, n_boot=1000):
    """Stationary-block bootstrap interval for the difference in annual return and Sharpe ratio between two daily series."""
    n = len(a)
    exa, exb = (a - rf).to_numpy(), (b - rf).to_numpy()
    av, bv = a.to_numpy(), b.to_numpy()
    dr, dsr = [], []
    for _ in range(n_boot):
        st = rng.integers(0, n, n // block + 1)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        dr.append((av[ix].mean() - bv[ix].mean()) * 252)
        sa = exa[ix].mean() / av[ix].std() if av[ix].std() > 0 else np.nan
        sb = exb[ix].mean() / bv[ix].std() if bv[ix].std() > 0 else np.nan
        dsr.append((sa - sb) * np.sqrt(252))
    return [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))], [float(np.nanpercentile(dsr, 2.5)), float(np.nanpercentile(dsr, 97.5))]


def main() -> None:
    res, fig = ROOT / "results", ROOT / "figures"
    D = M.load_data("")
    assert D.pf_source == "CRSP daily"
    PF = D.PF
    n = len(D.year)
    DE = pd.read_parquet(INTERIM / "daily_eval.parquet").set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    ZD = pickle.load(open(INTERIM / "duration.pkl", "rb"))
    DU, dsel = ZD["pred"].set_index("deal_id").reindex(D.deals["deal_id"]).reset_index(), ZD["selection"]
    LO = pd.read_parquet(INTERIM / "loss_oot.parquet").set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    E = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "year": D.year, "cls": D.outcome3, "spread": PF["spread"].to_numpy(), "p_mkt": PF["p_mkt"].to_numpy(),
                      "ret": PF["ret"].to_numpy(), "hold": PF["hold_days"].to_numpy(), "days_to_t1": PF["days_to_t1"].to_numpy(),
                      "log_value": D.X["log_deal_value"].to_numpy(float), "tender": (D.X["feat_tender_offer"].fillna(0).to_numpy() == 1),
                      "t1": PF["t1"].to_numpy(), "exit_date": PF["exit_date"].to_numpy()})
    priced = D.core & PF["backtest_ok"].fillna(False).to_numpy() & PF["cash_deal"].fillna(False).to_numpy() & np.isfinite(E["p_mkt"].to_numpy())
    canc = D.deals["closed_date"].fillna(D.deals["cancelled_date"]).to_numpy().astype("datetime64[D]")
    rng = np.random.default_rng(0)
    out = {"views": {}}
    S = {k: np.full(n, np.nan) for k in RANKS}
    S2 = {v: {k: np.full(n, np.nan) for k in RANKS} for v in ("pooled", "test")}
    comp_cols = {v: {k: np.full(n, np.nan) for k in ("p_adv", "p_top", "p_close", "loss", "t_close", "exp_ret", "exp_days", "ear", "ear_stack")} for v in ("pooled", "test")}
    for view, origins in (("pooled", YEARS), ("test", [None])):
        for Y in origins:
            te = priced & ((D.year == Y) if Y else (D.year >= 2019)) & DE[f"{view}__rows"].fillna(False).to_numpy(bool)
            cut = np.datetime64(f"{Y}-01-01") if Y else np.datetime64("2019-01-01")
            tr = priced & (D.year <= ((Y - 1) if Y else 2018)) & (canc < cut)              # outcomes known before the cut-off
            tag = "oot" if Y else "split"
            I = dsel.get((tag, Y, "full"))
            if te.sum() == 0 or I is None or tr.sum() < 200:
                continue
            c = components(E, tr, te, DE[f"{view}__cal"].to_numpy(), DU[f"{tag}__full__final"].to_numpy(), I["final"]["oof_resid_sd"], LO[f"{tag}__adverse"].to_numpy())
            sp = E["spread"].to_numpy()[te]
            er, et, e1 = ear(c, sp)
            R = S2[view]
            R["ear"][te] = e1
            R["ear_stack"][te] = ear(c, sp, p_adv=np.clip(DE[f"{view}__stack"].to_numpy()[te], 1e-4, 0.95))[2]
            R["spread"][te] = sp
            R["spread_ann"][te] = sp / (c["t_med_structure"] / 365.0)
            R["mkt"][te] = -E["p_mkt"].to_numpy()[te]
            R["ear_no_break"][te] = ear(c, sp, p_adv=np.full(te.sum(), c["p_adv_bar"]))[2]
            R["ear_no_duration"][te] = ear(c, sp, t_close=np.full(te.sum(), c["t_close_bar"]))[2]
            R["ear_no_loss"][te] = ear(c, sp, loss=np.full(te.sum(), c["loss_bar"]))[2]
            cc = comp_cols[view]
            for k in ("p_adv", "p_top", "p_close", "loss", "t_close"):
                cc[k][te] = c[k]
            cc["exp_ret"][te], cc["exp_days"][te], cc["ear"][te], cc["ear_stack"][te] = er, et, e1, R["ear_stack"][te]

    L = ["# Expected annualized return per deal, and the ranking test", "",
         "All-cash core deals with CRSP prices. On day 1 each deal gets: a probability of an adverse break, of being topped and of closing; an expected time to closing; an expected "
         "loss if it breaks; and from these an expected return and an expected annualized return. Every component comes from models fitted on earlier years only "
         "(`results/pooled_oot.md`, `results/duration.md`, `results/loss_given_break.md`).", "",
         "The test: sort the deals of each announcement year into thirds by the ranking, buy at the day-1 close, hold to the outcome, and compare realised returns. "
         "Annual return = total P&L over total capital-years. Intervals: bootstrap over announcement years (pooled) and over deals (test period). "
         "**Limits:** no transaction or financing costs, equal weights, unlimited capacity, every deal bought at the day-1 close.", ""]
    daily = pd.read_parquet(INTERIM / "deal_daily_returns.parquet")
    daily = daily.merge(pd.DataFrame({"deal_id": E["deal_id"], "exit_date": E["exit_date"]}), on="deal_id", how="left")
    daily = daily[daily["date"] <= daily["exit_date"]]
    ff = pd.read_parquet(WRDS / "ff_daily.parquet")
    rf = pd.Series(ff["rf"].to_numpy(float), index=pd.to_datetime(ff["date"]))
    fig_data = {}
    for view, title, by_year in (("pooled", "Pooled rolling origin, 2013-2025", True), ("test", "Test period, 2019-2025", False)):
        R = S2[view]
        rows = np.isfinite(R["ear"]) & np.isfinite(E["ret"].to_numpy()) & np.isfinite(E["hold"].to_numpy())
        if rows.sum() < 60:
            continue
        V = {"n": int(rows.sum()), "adverse": int((E["cls"][rows] == "adverse_break").sum()), "topped": int((E["cls"][rows] == "topped").sum()), "rankings": {}}
        L += [f"## {title}: {V['n']} deals, {V['adverse']} adverse breaks, {V['topped']} topped", "",
              "| Ranking | Top third: annual return | Middle | Bottom | Top minus bottom [95% CI] | Top third minus all deals [95% CI] | Adverse breaks top / middle / bottom |", "|---|---:|---:|---:|---|---|---|"]
        for k, lab in RANKS.items():
            r = tercile_table(E, rows, R[k], by_year, rng)
            V["rankings"][k] = r
            t = r["terciles"]
            L.append(f"| {lab} | {t['top']['annual_return']:+.1%} | {t['middle']['annual_return']:+.1%} | {t['bottom']['annual_return']:+.1%} | "
                     f"{r['top_minus_bottom'] * 100:+.1f} [{r['top_minus_bottom_ci95'][0] * 100:+.1f}, {r['top_minus_bottom_ci95'][1] * 100:+.1f}] | "
                     f"{r['top_minus_all'] * 100:+.1f} [{r['top_minus_all_ci95'][0] * 100:+.1f}, {r['top_minus_all_ci95'][1] * 100:+.1f}] | {t['top']['adverse']} / {t['middle']['adverse']} / {t['bottom']['adverse']} |")
        r = V["rankings"]["ear"]; t = r["terciles"]
        L += ["", "Differences are in points of annual return, equal to $ million per $100 million per year. "
              f"All ranked deals together returned {r['all']:+.1%} a year.", "",
              "Detail for the model ranking:", "", "| Third | Deals | Mean expected annualized return | Realised annual return | $ million per $100 million per year | Mean return per deal | Adverse break rate | Topped | Median holding days |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lab in ("top", "middle", "bottom"):
            x = t[lab]
            L.append(f"| {lab} | {x['n']} | {x['mean_score']:+.1%} | {x['annual_return']:+.1%} | {x['annual_return'] * 100:+.2f} | {x['mean_ret']:+.2%} | {x['adverse_rate']:.1%} | {x['topped']} | {x['median_hold']:.0f} |")
        L.append("")
        # rank agreement for the attribution
        base = pd.Series(R["ear"][rows])
        att = {k: float(base.corr(pd.Series(R[k][rows]), method="spearman")) for k in ("ear_no_break", "ear_no_duration", "ear_no_loss", "spread", "spread_ann", "mkt", "ear_stack")}
        V["rank_correlation_with_model"] = att
        L += ["Attribution: rank correlation of the model ranking with the ranking obtained when one component is replaced by its unconditional average "
              f"(1 = that component does not move the ranking): break probability {att['ear_no_break']:.2f}, time to closing {att['ear_no_duration']:.2f}, loss if it breaks {att['ear_no_loss']:.2f}. "
              f"Correlation with the raw spread ranking: {att['spread']:.2f}; with the annualized spread: {att['spread_ann']:.2f}.", ""]
        # calendar-time portfolio
        ids_all = E["deal_id"].to_numpy()[rows]
        e = R["ear"]
        order = np.argsort(E["t1"].to_numpy())
        top_half = np.zeros(n, bool)
        t1v = E["t1"].to_numpy()
        for i in np.where(rows)[0]:
            prior = rows & (t1v < t1v[i]) & (t1v >= t1v[i] - np.timedelta64(365, "D"))
            top_half[i] = (e[i] >= np.nanmedian(e[prior])) if prior.sum() >= 20 else True
        ra, na = calendar_portfolio(daily, set(ids_all), rf)
        rb, nb = calendar_portfolio(daily, set(E["deal_id"].to_numpy()[rows & top_half]), rf)
        idxd = pd.DatetimeIndex(sorted(set(rf.index[(rf.index >= ra.index.min()) & (rf.index <= ra.index.max())]) | set(ra.index)))
        ra, rb = ra.reindex(idxd).fillna(0.0), rb.reindex(idxd).fillna(0.0)
        rfd = rf.reindex(idxd).ffill().fillna(0.0)
        sa, sb = port_stats(ra, rfd), port_stats(rb, rfd)
        ci_r, ci_s = block_ci(rb, ra, rfd, rng)
        V["calendar"] = {"all": sa, "top_half": sb, "mean_open_all": float(na.reindex(idxd).fillna(0).mean()), "mean_open_top": float(nb.reindex(idxd).fillna(0).mean()),
                         "deals_all": int(rows.sum()), "deals_top": int((rows & top_half).sum()), "diff_return": sb["annual_return"] - sa["annual_return"], "diff_return_ci95": ci_r,
                         "diff_sharpe": sb["sharpe"] - sa["sharpe"], "diff_sharpe_ci95": ci_s, "start": str(idxd.min().date()), "end": str(idxd.max().date())}
        fig_data[view] = (idxd, (1 + ra).cumprod(), (1 + rb).cumprod(), V)
        L += [f"Calendar-time portfolio, {idxd.min().date()} to {idxd.max().date()}, daily CRSP total returns, equal weights across open deals, rebalanced daily, cash earns nothing on days without a position. "
              "\"Top half\" holds a deal only if its expected annualized return at entry is at or above the median of the deals announced in the previous twelve months.", "",
              "| Portfolio | Deals | Average open positions | Annual return | Volatility | Sharpe ratio | Maximum drawdown |", "|---|---:|---:|---:|---:|---:|---:|",
              f"| All open deals | {int(rows.sum())} | {V['calendar']['mean_open_all']:.1f} | {sa['annual_return']:+.1%} | {sa['volatility']:.1%} | {sa['sharpe']:.2f} | {sa['max_drawdown']:.1%} |",
              f"| Top half by expected annualized return | {int((rows & top_half).sum())} | {V['calendar']['mean_open_top']:.1f} | {sb['annual_return']:+.1%} | {sb['volatility']:.1%} | {sb['sharpe']:.2f} | {sb['max_drawdown']:.1%} |", "",
              f"Top half minus all: annual return {V['calendar']['diff_return'] * 100:+.2f} points [95% CI {ci_r[0] * 100:+.2f}, {ci_r[1] * 100:+.2f}]; Sharpe ratio {V['calendar']['diff_sharpe']:+.2f} "
              f"[{ci_s[0]:+.2f}, {ci_s[1]:+.2f}] (block bootstrap of daily returns, 21-day blocks). The Sharpe ratio is over the 1-month T-bill rate.", ""]
        out["views"][view] = V
    L += ["## Reading", "",
          "- A ranking adds value only if the top third beats the bottom third (or all deals) by more than the interval. Compare the model rows with the spread rows: the spread is free.",
          "- The attribution rows are not separate strategies; they show which of the three numbers moves the ranking and what is lost without it.",
          "- Returns are before all costs. Equal weighting and daily rebalancing are not achievable at scale in small deals.", ""]
    (res / "expected_return.md").write_text("\n".join(L) + "\n")
    (res / "expected_return.json").write_text(json.dumps(out, indent=2, default=float))
    sv = pd.DataFrame({"deal_id": E["deal_id"]})
    for v in ("pooled", "test"):
        for k, a in comp_cols[v].items():
            sv[f"{v}__{k}"] = a
    sv.to_parquet(INTERIM / "expected_return.parquet", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    if "pooled" in out["views"]:
        f_, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
        V = out["views"]["pooled"]
        ks = ["ear", "ear_stack", "spread", "spread_ann", "mkt"]
        ypos = np.arange(len(ks))[::-1]
        d = [V["rankings"][k]["top_minus_bottom"] * 100 for k in ks]
        lo = [(V["rankings"][k]["top_minus_bottom"] - V["rankings"][k]["top_minus_bottom_ci95"][0]) * 100 for k in ks]
        hi = [(V["rankings"][k]["top_minus_bottom_ci95"][1] - V["rankings"][k]["top_minus_bottom"]) * 100 for k in ks]
        ax[0].errorbar(d, ypos, xerr=[lo, hi], fmt="o", color="#08519c", capsize=3); ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set_yticks(ypos); ax[0].set_yticklabels([RANKS[k].replace("Benchmark: ", "").replace("Expected annualized return", "Expected ann. return") for k in ks], fontsize=8); ax[0].grid(axis="y", alpha=0)
        ax[0].set_xlabel("Top third minus bottom third, points of annual return (95% CI)")
        ax[0].set_title(f"Ranking test, pooled 2013-2025: {V['n']} all-cash deals, {V['adverse']} adverse breaks", loc="left", fontsize=9)
        idxd, ca, cb, _ = fig_data["pooled"]
        ax[1].plot(idxd, ca, color="#9e9e9e", label="all open deals"); ax[1].plot(idxd, cb, color="#08519c", label="top half by expected annualized return")
        ax[1].set_ylabel("Growth of 1 (before costs)"); ax[1].legend(frameon=False, fontsize=8)
        ax[1].set_title("Calendar-time portfolio, daily CRSP returns", loc="left", fontsize=9)
        f_.tight_layout(); f_.savefig(fig / "expected_return.png"); plt.close(f_)
    print("\n".join(L))


if __name__ == "__main__":
    main()

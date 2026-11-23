"""Model versus market and the backtest, on CRSP daily prices (needs the WRDS pull; see DATA.md).

Two evaluation sets, both core sample with adverse break as the target:
  test     deals announced 2019-2025, model trained on 2001-2018 (train_eval.py)
  pooled   deals announced 2013-2025, each scored by a model fitted on earlier years only (pooled_oot.py)

For every set the market side is built from training-period deals only:
  market logistic   logistic regression of the outcome on p_mkt and the day-1 spread
  stack             logistic regression on the model's out-of-fold score, p_mkt and the spread
  rule threshold    "skip the deal when the model sees more risk than the market": the gap between the calibrated
                    model probability and the market logistic (in log-odds) above a threshold; the threshold is the
                    training-period quantile that gave the best training-period annual return (candidates: skip
                    0%, 5%, ..., 50%). If skipping nothing was best in training, the rule holds everything.
In the pooled set all of this is refitted for every test year on the years before it.

Also here: the ladder step "day 1 after announcement" (final model plus day-1 spread and p_mkt as features). It is a
different decision time from the announcement-date models and is reported separately.

Writes results/market_benchmark.md, results/backtest.md, results/backtest.json, results/market_benchmark.json,
figures/model_vs_market.png, figures/backtest.png and data/interim/daily_eval.parquet (deal level, git-ignored).
"""
from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import modeling as M
import pooled_oot as PO
import train_eval as TE

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
CAPITAL = 100e6
N_BOOT = 2000
QGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CONFIRMED = ("export, confirmed by the press release", "press release (differs from the export)")


def portfolio(ret, hold):
    """Capital-time-weighted annual return: total P&L over total capital-years."""
    cy = np.sum(hold) / 365.0
    return float(np.sum(ret) / cy) if cy > 0 else float("nan")


def lr(Z, y, C=1.0):
    return LogisticRegression(C=C, max_iter=2000).fit(Z, y)


def market_view(E, oof, pred, tr, te, mask):
    """Fit market logistic, stack, calibration and the rule threshold on `tr`, apply to `te`. Arrays are full length."""
    y = E["y"].to_numpy()
    pm, sp = E["p_mkt"].to_numpy(), E["spread"].clip(-0.2, 0.6).to_numpy()
    a = tr & mask & np.isfinite(oof) & np.isfinite(pm)
    b = te & mask & np.isfinite(pred) & np.isfinite(pm)
    out = {k: np.full(len(E), np.nan) for k in ("mkt", "stack", "cal", "delta")}
    out["hold_rule"] = np.zeros(len(E), bool)
    info = {"n_train": int(a.sum()), "pos_train": int(y[a].sum()), "n_test": int(b.sum()), "pos_test": int(y[b].sum())}
    if y[a].sum() < 8 or b.sum() == 0:
        return out, b & False, info
    Zm = np.column_stack([pm, sp])
    mk = lr(Zm[a], y[a])
    st = lr(np.column_stack([M.logit(oof[a]), Zm[a]]), y[a])
    allo = tr & np.isfinite(oof)
    pl = lr(M.logit(oof[allo])[:, None], y[allo], C=1e6)             # Platt scaling on all out-of-fold rows
    cal_a, cal_b = pl.predict_proba(M.logit(oof[a])[:, None])[:, 1], pl.predict_proba(M.logit(pred[b])[:, None])[:, 1]
    mk_a, mk_b = mk.predict_proba(Zm[a])[:, 1], mk.predict_proba(Zm[b])[:, 1]
    d_a, d_b = M.logit(cal_a) - M.logit(mk_a), M.logit(cal_b) - M.logit(mk_b)
    ret, hold = E["ret"].to_numpy(), E["hold"].to_numpy()
    best = (-np.inf, 0.0, np.inf)
    for q in QGRID:
        thr = np.inf if q == 0 else float(np.quantile(d_a, 1 - q))
        h = d_a <= thr
        ann = portfolio(ret[a][h], hold[a][h])
        if ann > best[0] + 1e-12:
            best = (ann, q, thr)
    out["mkt"][b], out["cal"][b], out["delta"][b] = mk_b, cal_b, d_b
    out["stack"][b] = st.predict_proba(np.column_stack([M.logit(pred[b]), Zm[b]]))[:, 1]
    out["hold_rule"][b] = d_b <= best[2]
    info.update(stack_coef=st.coef_[0].tolist(), market_coef=mk.coef_[0].tolist(), rule_q=best[1], rule_threshold=None if not np.isfinite(best[2]) else best[2],
                rule_train_gain=best[0] - portfolio(ret[a], hold[a]))
    return out, b, info


def build_views(E, Z, P, final, mask):
    """(test view, pooled view): dicts of full-length arrays plus the evaluated rows and fit information."""
    yr = E["year"].to_numpy()
    t_out, t_rows, t_info = market_view(E, P[f"oof__primary__full__{final}"].to_numpy(), P[f"pred__primary__full__{final}"].to_numpy(), yr <= 2018, yr >= 2019, mask)
    t_out["model"] = P[f"pred__primary__full__{final}"].to_numpy()
    p_out = {k: np.full(len(E), np.nan) for k in ("mkt", "stack", "cal", "delta")}
    p_out["hold_rule"] = np.zeros(len(E), bool)
    p_rows = np.zeros(len(E), bool)
    p_info = {}
    pos = pd.Series(np.arange(len(E)), index=E["deal_id"].to_numpy())
    pred = Z["pred"]["pred__full__final_proc"].to_numpy()
    for Y in Z["years"]:
        ids, op = Z["oof"].get((Y, "full", "final_proc"), (np.array([]), np.array([])))
        oof = np.full(len(E), np.nan)
        if len(ids):
            oof[pos.reindex(ids).to_numpy()] = op
        o, rows, info = market_view(E, oof, pred, yr <= Y - 1, yr == Y, mask)
        for k in p_out:
            p_out[k][rows] = o[k][rows]
        p_rows |= rows
        p_info[Y] = info
    p_out["model"] = pred
    return (t_out, t_rows, t_info), (p_out, p_rows, p_info)


# ----------------------------------------------------------------------------- market benchmark
SCORES = [("p_mkt", "Market: p_mkt"), ("p_mkt_carry", "Market: p_mkt with carry"), ("spread", "Market: raw day-1 spread"),
          ("mkt", "Market: logistic on p_mkt and spread (fitted on earlier years)"), ("model", "Model (announcement date)"),
          ("stack", "Model + market (stack fitted on earlier years)")]
PAIRS = [("stack", "mkt"), ("stack", "model"), ("model", "p_mkt"), ("stack", "p_mkt")]


def bench(E, view, ci_key):
    o, rows, _ = view
    y, yr = E["y"].to_numpy()[rows], E["year"].to_numpy()[rows]
    S = {"p_mkt": E["p_mkt"].to_numpy()[rows], "p_mkt_carry": E["p_mkt_carry"].fillna(E["p_mkt"]).to_numpy()[rows], "spread": E["spread"].to_numpy()[rows],
         "mkt": o["mkt"][rows], "model": o["model"][rows], "stack": o["stack"][rows]}
    ev = PO.evaluate(y, yr, S, PAIRS, n_boot=N_BOOT, auc_only=True)
    ev["n"], ev["pos"] = int(rows.sum()), int(y.sum())
    L = ["| Score | ROC AUC [95% CI] | Within-year AUC [95% CI] | Adverse breaks in riskiest 20% |", "|---|---|---|---:|"]
    for k, lab in SCORES:
        r = ev["scores"][k]
        L.append(f"| {lab} | {PO.f3(r['roc_auc'], r[ci_key]['roc_auc'])} | {PO.f3(r['within_year_auc'], r[ci_key]['within_year_auc'])} | {M.capture_at(y, S[k], 0.2):.2f} |")
    L.append("")
    names = {"stack minus mkt": "Model added to the market (stack minus market logistic)", "stack minus model": "Market added to the model (stack minus model)",
             "model minus p_mkt": "Model minus p_mkt", "stack minus p_mkt": "Stack minus p_mkt"}
    for k, lab in names.items():
        r = ev["pairs"][k]
        other = "ci95_deal" if ci_key == "ci95_year" else "ci95_year"
        L.append(f"- {lab}: **{PO.d3(r['roc_auc'], r[ci_key]['roc_auc'])}** AUC" + (f" (resampling deals instead of years: [{r[other]['roc_auc'][0]:+.3f}, {r[other]['roc_auc'][1]:+.3f}])" if ci_key == "ci95_year" else "")
                 + f"; positive in {r['share_gt0_' + ci_key.split('_')[1]]:.0%} of resamples. Within-year AUC: {PO.d3(r['within_year_auc'], r[ci_key]['within_year_auc'])}.")
    L.append("")
    return ev, L


# ----------------------------------------------------------------------------- backtest
def run_bt(E, rows, score, hold_rule, mkt, by_year, v5=False, seed=0):
    X = E[rows]
    ret = np.where(X["broken"], X["ret5"], X["ret"]) if v5 else X["ret"].to_numpy()
    hold = np.where(X["broken"], X["hold5"], X["hold"]) if v5 else X["hold"].to_numpy()
    k = np.isfinite(ret) & np.isfinite(hold)
    X, ret, hold = X[k], ret[k], hold[k]
    sc, yr, cls, sp = score[rows][k], X["year"].to_numpy(), X["cls"].to_numpy(), X["spread"].to_numpy()
    rule = hold_rule[rows][k]
    mk = mkt[rows][k]
    n = len(X)

    def top(vals, frac, within_year):
        f = np.zeros(n, bool)
        groups = [np.where(yr == v)[0] for v in np.unique(yr)] if within_year else [np.arange(n)]
        for g in groups:
            kk = int(round(frac * len(g)))
            if kk:
                f[g[np.argsort(-vals[g], kind="stable")[:kk]]] = True
        return f
    ref = np.zeros(n, bool)                                   # same number of deals as the rule, chosen by the market alone
    for g in ([np.where(yr == v)[0] for v in np.unique(yr)] if by_year else [np.arange(n)]):
        kk = int(rule[g].sum())
        if kk:
            ref[g[np.argsort(-mk[g], kind="stable")[:kk]]] = True
    S = {"Hold all deals": np.ones(n, bool), "Skip riskiest 10% by model": ~top(sc, 0.10, by_year), "Skip riskiest 20% by model": ~top(sc, 0.20, by_year),
         "Model-versus-market rule (threshold from earlier years)": rule,
         "Reference: same number of deals, highest market-implied risk first (no model)": ref, "Reference: skip the widest 10% of spreads": ~top(sp, 0.10, by_year)}
    rng = np.random.default_rng(seed)
    base = portfolio(ret, hold)
    blocks = [np.where(yr == v)[0] for v in np.unique(yr)]
    res = {"n": n, "adverse": int((cls == "adverse_break").sum()), "topped": int((cls == "topped").sum()), "strategies": {}, "by_outcome": {}}
    draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) if by_year else rng.integers(0, n, n) for _ in range(N_BOOT)]
    base_draws = np.array([portfolio(ret[ix], hold[ix]) for ix in draws])
    res["hold_all_ci95"] = [float(np.percentile(base_draws, 2.5)), float(np.percentile(base_draws, 97.5))]
    for name, h in S.items():
        ann = portfolio(ret[h], hold[h])
        ci = None
        if name != "Hold all deals":
            ds = []
            for ix, b0 in zip(draws, base_draws):
                if "riskiest" in name and not by_year:                # re-rank inside each resample
                    frac = 0.10 if "10%" in name else 0.20
                    hh = np.ones(n, bool); hh[np.argsort(-sc[ix], kind="stable")[:int(round(frac * n))]] = False
                else:
                    hh = h[ix]
                if hh.sum() > 5:
                    ds.append(portfolio(ret[ix][hh], hold[ix][hh]) - b0)
            ci = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
        res["strategies"][name] = {"held": int(h.sum()), "adverse_held": int((cls[h] == "adverse_break").sum()), "topped_held": int((cls[h] == "topped").sum()),
                                   "mean_ret": float(ret[h].mean()) if h.any() else float("nan"), "median_hold": float(np.median(hold[h])) if h.any() else float("nan"),
                                   "annual_return": ann, "usd_per_100m": ann * CAPITAL, "delta_vs_all": ann - base, "delta_ci95": ci}
    dd = [portfolio(ret[ix][rule[ix]], hold[ix][rule[ix]]) - portfolio(ret[ix][ref[ix]], hold[ix][ref[ix]]) for ix in draws if rule[ix].sum() > 5 and ref[ix].sum() > 5]
    res["rule_minus_market_only"] = {"point": portfolio(ret[rule], hold[rule]) - portfolio(ret[ref], hold[ref]) if rule.sum() > 5 and ref.sum() > 5 else float("nan"),
                                     "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))] if dd else None, "overlap": float((rule & ref).sum() / max(rule.sum(), 1)),
                                     "mean_spread_rule": float(np.nanmean(sp[rule])) if rule.any() else float("nan"), "mean_spread_all": float(np.nanmean(sp))}
    for c in ("closed", "topped", "adverse_break"):
        m = cls == c
        if m.any():
            res["by_outcome"][c] = {"n": int(m.sum()), "mean_ret": float(ret[m].mean()), "median_ret": float(np.median(ret[m])), "min_ret": float(ret[m].min()),
                                    "max_ret": float(ret[m].max()), "median_hold": float(np.median(hold[m])), "mean_score": float(sc[m].mean()),
                                    "mean_p_mkt": float(np.nanmean(X["p_mkt"].to_numpy()[m]))}
    return res


def bt_table(r, by_year):
    L = [f"{r['n']} deals: {r['n'] - r['adverse'] - r['topped']} closed, {r['adverse']} adverse breaks, {r['topped']} topped by a higher bid. "
         f"Hold-all annual return {r['strategies']['Hold all deals']['annual_return']:+.1%} [95% CI {r['hold_all_ci95'][0]:+.1%}, {r['hold_all_ci95'][1]:+.1%}].", "",
         "| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, s in r["strategies"].items():
        d = "reference" if s["delta_ci95"] is None else f"{s['delta_vs_all'] * 100:+.2f} [{s['delta_ci95'][0] * 100:+.2f}, {s['delta_ci95'][1] * 100:+.2f}]"
        L.append(f"| {k} | {s['held']} | {s['adverse_held']} | {s['topped_held']} | {s['mean_ret']:+.2%} | {s['median_hold']:.0f} | {s['annual_return']:+.1%} | {s['annual_return'] * 100:+.2f} | {d} |")
    x = r["rule_minus_market_only"]
    if x["ci95"]:
        L += ["", f"Model-versus-market rule minus the model-free reference with the same number of deals: **{x['point'] * 100:+.2f} points [95% CI {x['ci95'][0] * 100:+.2f}, {x['ci95'][1] * 100:+.2f}]**; "
              f"the two hold {x['overlap']:.0%} of the same deals. Mean day-1 spread of the deals the rule holds: {x['mean_spread_rule']:.1%}, of all deals: {x['mean_spread_all']:.1%}."]
    L.append("")
    return L


def outcome_table(r):
    L = ["| Outcome | Deals | Mean return per deal | Median return | Worst | Best | Median holding days | Mean model score | Mean p_mkt |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, lab in (("closed", "Closed"), ("topped", "Topped by a higher bid"), ("adverse_break", "Adverse break")):
        if c in r["by_outcome"]:
            o = r["by_outcome"][c]
            L.append(f"| {lab} | {o['n']} | {o['mean_ret']:+.1%} | {o['median_ret']:+.1%} | {o['min_ret']:+.1%} | {o['max_ret']:+.1%} | {o['median_hold']:.0f} | {o['mean_score']:.3f} | {o['mean_p_mkt']:.2f} |")
    L.append("")
    return L


# ----------------------------------------------------------------------------- main
def main() -> None:
    res, fig = ROOT / "results", ROOT / "figures"
    D = M.load_data("")
    assert D.pf_source == "CRSP daily", "run src/crsp_prices.py first (needs the WRDS pull)"
    Mj = json.loads((res / "metrics.json").read_text())
    final = Mj["final_models"]["overall"]
    P = pd.read_parquet(INTERIM / "predictions.parquet").set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    Z = pickle.load(open(INTERIM / "pooled_oot.pkl", "rb"))
    Z["pred"] = Z["pred"].set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    PF = D.PF
    E = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "year": D.year, "core": D.core, "y": D.y["label_adverse"], "broken": D.y["label"] == 1, "cls": D.outcome3,
                      "cash": PF["cash_deal"].fillna(False).to_numpy(bool), "ok": PF["backtest_ok"].fillna(False).to_numpy(bool),
                      "spread": PF["spread"].to_numpy(), "p_mkt": PF["p_mkt"].to_numpy(), "p_mkt_carry": PF["p_mkt_carry"].to_numpy(),
                      "ret": PF["ret"].to_numpy(), "hold": PF["hold_days"].to_numpy(), "ret5": PF["ret5"].to_numpy(), "hold5": PF["hold_days5"].to_numpy(),
                      "offer_source": PF["offer_source"].fillna("").to_numpy()})
    core, ok, cash = E["core"].to_numpy(), E["ok"].to_numpy(), E["cash"].to_numpy()
    masks = {"cash": core & ok & cash, "all": core & ok, "confirmed": core & ok & cash & E["offer_source"].isin(CONFIRMED).to_numpy()}
    V = {k: build_views(E, Z, P, final, m) for k, m in masks.items()}
    out = {"final_model": final, "benchmark": {}, "day1": {}}

    # ------------------------------------------------------------------ market benchmark
    (tv, pv) = V["cash"]
    ev_t, tab_t = bench(E, tv, "ci95_deal")
    ev_p, tab_p = bench(E, pv, "ci95_year")
    out["benchmark"] = {"test_cash": ev_t, "pooled_cash": ev_p, "test_fit": tv[2], "pooled_fit": {str(k): v for k, v in pv[2].items()}}
    old = json.loads((res / "monthly/backtest.json").read_text())["benchmark"].get("All-cash deals, core sample (main) | label_adverse") if (res / "monthly/backtest.json").exists() else None
    old_txt = (f" The earlier version on monthly closes had {old['n_test']} priced test deals and {old['pos_test']} adverse breaks (`results/monthly/market_benchmark.md`)." if old else "")
    MB = ["# Market benchmark: does the model add information beyond the market?", "",
          f"Model: {final} at the announcement date. Market: the day-1 spread and `p_mkt` = (offer - close(t1)) / (offer - P0) from CRSP daily prices, where t1 is the first trading day "
          "after the announcement and P0 the close on the last trading day before it. Core sample, all-cash deals, target adverse break. "
          "`p_mkt` ignores time value and assumes a broken deal falls back to exactly P0; `p_mkt_carry` discounts the offer at the 3-month T-bill rate over the expected remaining time.", "",
          "The market sees one more trading day than the model. The market logistic, the stack and everything else on the market side are fitted on earlier years only.", "",
          f"## Test period 2019-2025: {ev_t['n']} priced all-cash deals, {ev_t['pos']} adverse breaks", "",
          f"Fitted on {tv[2]['n_train']} training-period deals ({tv[2]['pos_train']} adverse breaks) with out-of-fold model scores.{old_txt} Intervals resample deals.", ""] + tab_t
    MB += [f"## Pooled out-of-time 2013-2025: {ev_p['n']} priced all-cash deals, {ev_p['pos']} adverse breaks", "",
           "Each deal is scored by a model fitted on earlier years only (13 origins, `results/pooled_oot.md`); the market logistic and the stack are refitted for every test year on "
           "the years before it. Intervals resample announcement years.", ""] + tab_p
    MB += ["## Robustness", "", "| Set | Deals | Adverse breaks | p_mkt | Model | Stack | Stack minus market logistic [95% CI] |", "|---|---:|---:|---:|---:|---:|---|"]
    for key, lab in (("all", "all considerations (stock and mixed spreads from an implied exchange ratio)"), ("confirmed", "all-cash, offer confirmed by or taken from the press release")):
        for (vw, ck, nm) in ((V[key][0], "ci95_deal", "Test 2019-2025"), (V[key][1], "ci95_year", "Pooled 2013-2025")):
            if vw[1].sum() == 0:
                continue
            ev, _ = bench(E, vw, ck)
            out["benchmark"][f"{nm} | {key}"] = ev
            r = ev["pairs"]["stack minus mkt"]
            MB.append(f"| {nm}, {lab} | {ev['n']} | {ev['pos']} | {ev['scores']['p_mkt']['roc_auc']:.3f} | {ev['scores']['model']['roc_auc']:.3f} | {ev['scores']['stack']['roc_auc']:.3f} | {PO.d3(r['roc_auc'], r[ck]['roc_auc'])} |")
    MB.append("")

    # ------------------------------------------------------------------ day-1 ladder step
    specs = pickle.load(open(INTERIM / "model_specs.pkl", "rb"))[("primary", "full")]
    kind, ft, bp = specs["feats"][final]
    tr, te = specs["tr"], specs["te"]
    y = D.y["label_adverse"]
    day1 = np.full(len(E), np.nan)
    day1[te] = M.fit_lgbm(D, y, dataclasses.replace(ft, spread=True), tr, te, bp)[0]
    day1_p = np.full(len(E), np.nan)
    idx = np.arange(len(E))
    for Y in Z["years"]:
        s = Z["selection"].get((Y, "full"))
        if not s or "final_proc" not in s:
            continue
        ch = s["final_proc"]["chosen"]; par = s[ch]
        f = M.Feat(base_only=True, year=par["use_year"], spread=True) if ch == "lgbm_base" else \
            M.Feat(groups=TE.EXT if ch == "lgbm_ext" else TE.EXTM, year=par["use_year"], spread=True)
        a, b = idx[D.core & (D.year <= Y - 1)], idx[D.core & (D.year == Y)]
        day1_p[b] = M.fit_lgbm(D, y, f, a, b, (par["num_leaves"], par["n_estimators"]))[0]
    d1 = {}
    for nm, sc, base, rows0, ck in (("test", day1, P[f"pred__primary__full__{final}"].to_numpy(), np.isin(idx, te), "ci95_deal"),
                                    ("pooled", day1_p, Z["pred"]["pred__full__final_proc"].to_numpy(), np.isfinite(day1_p), "ci95_year")):
        for sub, rows in (("all core deals", rows0), ("priced all-cash deals", rows0 & masks["cash"] & np.isfinite(E["p_mkt"].to_numpy()))):
            S = {"model": base[rows], "day1": sc[rows]}
            pairs = [("day1", "model")]
            if sub.startswith("priced"):
                S["p_mkt"] = E["p_mkt"].to_numpy()[rows]; pairs.append(("day1", "p_mkt"))
            ev = PO.evaluate(y[rows], D.year[rows], S, pairs, n_boot=N_BOOT, auc_only=True)
            ev["n"], ev["pos"], ev["ci"] = int(rows.sum()), int(y[rows].sum()), ck
            d1[f"{nm} | {sub}"] = ev
    out["day1"] = d1
    D1 = ["## Different decision time: day 1 after announcement", "",
          "**Not comparable with the rows above as a like-for-like model step.** The models above decide on the announcement date. This step adds what the market shows one trading day later "
          "(day-1 spread, `p_mkt`, `p_mkt` with carry, days to t1) to the final model's features; deals without a usable price keep missing values. "
          "Hyper-parameters are those of the final model.", "",
          "| Evaluation set | Deals | Adverse breaks | Final model, announcement date | Final model + day-1 market features | Difference [95% CI] | Day-1 model minus p_mkt [95% CI] |", "|---|---:|---:|---:|---:|---|---|"]
    for k, ev in d1.items():
        nm, sub = k.split(" | ")
        ck = ev["ci"]; r = ev["pairs"]["day1 minus model"]
        extra = PO.d3(ev["pairs"]["day1 minus p_mkt"]["roc_auc"], ev["pairs"]["day1 minus p_mkt"][ck]["roc_auc"]) if "day1 minus p_mkt" in ev["pairs"] else "n/a"
        D1.append(f"| {'Test 2019-2025' if nm == 'test' else 'Pooled out-of-time 2013-2025'}, {sub} | {ev['n']} | {ev['pos']} | {ev['scores']['model']['roc_auc']:.3f} | "
                  f"{PO.f3(ev['scores']['day1']['roc_auc'], ev['scores']['day1'][ck]['roc_auc'])} | {PO.d3(r['roc_auc'], r[ck]['roc_auc'])} | {extra} |")
    D1 += ["", "Intervals resample deals for the test period and announcement years for the pooled set.", ""]
    MB += D1
    MB += ["Reading the tables: an increment whose interval includes zero means the data cannot tell whether one source adds to the other.", ""]
    (res / "market_benchmark.md").write_text("\n".join(MB) + "\n")
    (res / "market_benchmark.json").write_text(json.dumps(out, indent=2, default=float))
    bt_path = res / "benchmark_table.md"                              # append the clearly separated day-1 step to the ladder file
    if bt_path.exists():
        txt = bt_path.read_text().split("\n## Different decision time: day 1 after announcement")[0].rstrip("\n")
        bt_path.write_text(txt + "\n\n" + "\n".join(D1) + "\nDetails and the comparison with the market: `results/market_benchmark.md`.\n")

    # ------------------------------------------------------------------ backtest
    plaus = E["spread"].between(-0.05, 0.25).to_numpy()
    B = {"final_model": final, "panels": {}}
    BT = ["# Backtest with daily marks (headline impact)", "",
          "Core sample, CRSP daily prices. Entry: target close on t1, the first trading day after the announcement. Completed deals are held to the last trading day and earn "
          "CRSP total returns including dividends and the delisting return (the cash paid at closing). Broken deals are sold at the close of the first trading day after the "
          "cancellation date; a variant sells five trading days after it. Equal weights. Annual return = total P&L divided by total capital-years, so "
          "`$ per $100 million per year` assumes the capital is continuously redeployed into deals of the same kind.", "",
          "**Limits.** No transaction costs, no financing costs, no borrow, no position or capacity limits; every deal can be bought at the day-1 close in equal size. "
          "A few deals without announcement text keep the export's final offer in the spread (`results/crsp_prices.md`); this affects the spread-based rules, not the returns. "
          f"Model: {final} for the test period, the same procedure refitted each year for the pooled set. \"Riskiest 10%\" ranks within the test period, and within each announcement "
          "year in the pooled set. Intervals: bootstrap over deals (test period) and over announcement years (pooled).", ""]
    spec = [("All-cash deals (main)", "cash", None, False), ("All-cash deals with a day-1 spread between -5% and +25%", "cash", plaus, False),
            ("All-cash deals, broken deals sold five trading days after the cancellation date", "cash", None, True),
            ("All considerations, including stock and mixed deals, unhedged (robustness)", "all", None, False)]
    for title, key, extra, v5 in spec:
        BT += [f"## {title}", ""]
        for nm, vw, by_year in (("Test period 2019-2025", V[key][0], False), ("Pooled out-of-time 2013-2025", V[key][1], True)):
            o, rows, info = vw
            rows = rows & (extra if extra is not None else True)
            if rows.sum() < 30:
                continue
            r = run_bt(E, rows, o["model"], o["hold_rule"], o["mkt"], by_year, v5=v5)
            B["panels"][f"{title} | {nm}"] = r
            BT += [f"### {nm}", ""] + bt_table(r, by_year)
            if key == "cash" and extra is None and not v5:
                BT += ["Per-deal returns by outcome:", ""] + outcome_table(r)
    ti = V["cash"][0][2]
    qs = [v.get("rule_q") for v in V["cash"][1][2].values() if "rule_q" in v]
    B["rule"] = {"test": ti, "pooled_q": qs}
    BT += ["## The model-versus-market rule", "",
           "The rule skips a deal when the calibrated model probability exceeds the market logistic by more than a threshold (in log-odds). The threshold is the training-period "
           "quantile that maximised the training-period annual return, searched over skipping 0% to 50% of deals in steps of 5 points.", "",
           f"- Test period: the training years chose to skip the top {ti.get('rule_q', 0):.0%} of deals by that gap (training-period gain {ti.get('rule_train_gain', 0) * 100:+.2f} points a year)."
           + (" Skipping nothing was best in training, so the rule holds every deal." if ti.get("rule_q", 0) == 0 else ""),
           f"- Pooled set: the chosen share varies by origin year: {', '.join(f'{q:.0%}' for q in qs)}.", "",
           "A threshold picked as the best of eleven candidates on a few dozen training breaks is itself noisy; read the row as a test of the procedure, not of one cut-off.", "",
           "**The rule is mostly a bet on wide spreads.** Deals where the model sees less risk than the market are, almost by construction, deals with wide spreads, and wide spreads "
           "earned more than narrow ones in this sample (see the reference row that skips the widest 10%). The fair test of the model is therefore the line under each table that "
           "compares the rule with a model-free rule holding the same number of deals, chosen by the market-implied risk alone.", ""]
    (res / "backtest.md").write_text("\n".join(BT) + "\n")
    (res / "backtest.json").write_text(json.dumps(B, indent=2, default=float))

    # deal-level outputs for later steps (git-ignored)
    sv = pd.DataFrame({"deal_id": E["deal_id"], "day1_test": day1, "day1_pooled": day1_p})
    for nm, vw in (("test", V["cash"][0]), ("pooled", V["cash"][1]), ("test_all", V["all"][0]), ("pooled_all", V["all"][1])):
        for k in ("model", "cal", "mkt", "stack", "delta"):
            sv[f"{nm}__{k}"] = vw[0][k]
        sv[f"{nm}__rows"] = vw[1]
    sv.to_parquet(INTERIM / "daily_eval.parquet", index=False)

    figures(out, B, fig)
    print("\n".join(MB)); print("\n".join(BT[:60]))


def figures(out, B, fig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    f_, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharex=True)
    rowsdef = [("p_mkt", "Market-implied probability (day 1)"), ("model", "Model (announcement date)"), ("stack", "Model + market")]
    for ax, (key, ck, title) in zip(axes, (("test_cash", "ci95_deal", "Test period 2019-2025"), ("pooled_cash", "ci95_year", "Pooled out-of-time 2013-2025"))):
        ev = out["benchmark"][key]
        ypos = np.arange(len(rowsdef))[::-1]
        v = [ev["scores"][k]["roc_auc"] for k, _ in rowsdef]
        lo = [ev["scores"][k]["roc_auc"] - ev["scores"][k][ck]["roc_auc"][0] for k, _ in rowsdef]
        hi = [ev["scores"][k][ck]["roc_auc"][1] - ev["scores"][k]["roc_auc"] for k, _ in rowsdef]
        ax.errorbar(v, ypos, xerr=[lo, hi], fmt="o", color="#08519c", capsize=3)
        ax.axvline(0.5, color="k", lw=0.8)
        ax.set_yticks(ypos); ax.set_yticklabels([l for _, l in rowsdef] if ax is axes[0] else []); ax.grid(axis="y", alpha=0)
        ax.set_xlabel("ROC AUC (95% CI)")
        ax.set_title(f"{title}\n{ev['n']} priced all-cash deals, {ev['pos']} adverse breaks", loc="left", fontsize=9)
    f_.suptitle("Ranking adverse breaks: model versus the market's day-1 spread", x=0.01, ha="left", fontsize=10)
    f_.tight_layout(); f_.savefig(fig / "model_vs_market.png"); plt.close(f_)

    f_, axes = plt.subplots(1, 2, figsize=(11.5, 3.4), sharey=True)
    for ax, nm in zip(axes, ("Test period 2019-2025", "Pooled out-of-time 2013-2025")):
        r = B["panels"][f"All-cash deals (main) | {nm}"]["strategies"]
        ks = [k for k in r if k != "Hold all deals"]
        ypos = np.arange(len(ks))[::-1]
        d = [r[k]["delta_vs_all"] * 100 for k in ks]
        lo = [(r[k]["delta_vs_all"] - r[k]["delta_ci95"][0]) * 100 for k in ks]; hi = [(r[k]["delta_ci95"][1] - r[k]["delta_vs_all"]) * 100 for k in ks]
        ax.errorbar(d, ypos, xerr=[lo, hi], fmt="o", color="#08519c", capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ypos); ax.set_yticklabels([k.replace("Reference: ", "").replace(" (threshold from earlier years)", "").replace("same number of deals, highest market-implied risk first (no model)", "same number of deals by market risk, no model") for k in ks], fontsize=8); ax.grid(axis="y", alpha=0)
        ax.set_xlabel("Change in annual return vs hold-all (points, 95% CI)")
        b = r["Hold all deals"]
        ax.set_title(f"{nm}\nhold-all: {b['annual_return']:.1%} a year, {b['held']} deals, {b['adverse_held']} adverse breaks", loc="left", fontsize=9)
    f_.suptitle("Backtest with daily marks, all-cash deals, before costs", x=0.01, ha="left", fontsize=10)
    f_.tight_layout(); f_.savefig(fig / "backtest.png"); plt.close(f_)


if __name__ == "__main__":
    main()

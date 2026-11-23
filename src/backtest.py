"""Market benchmark and price-based backtest on Capital IQ MONTHLY closes, test period 2019-2025.

This is the documented fallback that runs without WRDS. With CRSP daily data the headline versions are
produced by src/daily_eval.py (results/market_benchmark.md, results/backtest.md); this script writes to
results/monthly/ and figures/monthly_*.png.

Part 1 - does the model add information beyond the market?
  On test deals with a usable price, compare the market-implied break probability p_mkt
  (and the raw spread) with the announcement-date model, and with a stack of both. The stack
  and the market-only benchmark are fitted on training-period deals only, using the model's
  out-of-fold scores there.

Part 2 - backtest (headline impact)
  Buy the target at close(t1), the first monthly grid close after the announcement. Exit at
  the offer price on the closing date for completed cash deals, and at the first grid close
  after cancellation for broken deals. Strategies: hold all; skip the riskiest 10% / 20% by
  model score; hold only deals where the calibrated model probability is below p_mkt.

Limits: monthly marks (entry up to a month after announcement, exits on the next grid date),
no transaction costs, no dividends, no financing cost, equal weights, no capacity limits.

Usage: python src/backtest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import modeling as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
CAPITAL = 100e6
N_BOOT = 2000


def auc_ci(y, s, rng, n=N_BOOT):
    bs = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            bs.append(roc_auc_score(y[ix], s[ix]))
    return float(roc_auc_score(y, s)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def diff_ci(y, a, b, rng, n=N_BOOT):
    bs = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            bs.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float((np.array(bs) > 0).mean())


def portfolio(ret, hold, w=None):
    """Capital-time-weighted annual return: total P&L over total capital-years."""
    w = np.ones(len(ret)) if w is None else w
    cap_years = np.sum(w * hold / 365.0)
    return float(np.sum(w * ret) / cap_years) if cap_years > 0 else float("nan")


def main() -> None:
    res0 = ROOT / "results"; res = res0 / "monthly"; fig = ROOT / "figures"
    res.mkdir(exist_ok=True)
    Mj = json.loads((res0 / "metrics.json").read_text())
    final = Mj["final_models"]["overall"]
    P = pd.read_parquet(INTERIM / "predictions.parquet").merge(pd.read_parquet(INTERIM / "price_features.parquet"), on="deal_id", how="left")
    rng = np.random.default_rng(0)
    pcol, ocol, ccol = f"pred__primary__full__{final}", f"oof__primary__full__{final}", f"cal__primary__full__{final}"
    P["model"] = P[pcol]
    P["model_cal"] = P[ccol] if ccol in P else P[pcol]
    out = {"final_model": final}
    MB = ["# Market benchmark on monthly closes (fallback without CRSP)", "",
          "The headline version uses CRSP daily prices: `results/market_benchmark.md`. This file is what the repository produces from the Capital IQ monthly closes alone. "
          "It uses the export's final offer, so later price increases inflate some spreads (see `results/crsp_prices.md`).", "",
          f"Model: {final} (announcement-date features only; trained on 2001-2018). Market: `p_mkt` = (offer - close(t1)) / (offer - P0), the share of the "
          "premium the market has not yet paid, where t1 is the first monthly grid date after the announcement and P0 the price one day before it. "
          "`p_mkt` ignores time value and assumes a broken deal falls back to exactly P0; `p_mkt_carry` discounts the offer at the 3-month T-bill rate "
          "over the expected remaining time to closing.", "",
          "**Caveat that favours the market:** t1 is up to a month after the announcement (median about two weeks), so the market price reflects news the "
          "announcement-date model cannot see. A model that only matches the market here would still be useful on the announcement day; a model that adds to "
          "it despite the head start is stronger evidence.", ""]

    def benchmark(mask_name, mask, target):
        y_all = P[target].to_numpy()
        tr = (mask & (P["year"] <= 2018) & P[ocol].notna() & P["p_mkt"].notna()).to_numpy()
        te = (mask & (P["year"] >= 2019) & P["model"].notna() & P["p_mkt"].notna()).to_numpy()
        if y_all[te].sum() < 5 or y_all[tr].sum() < 10:
            return None, [f"### {mask_name}, target `{target}`: too few breaks with usable prices (test {int(y_all[te].sum())}, train {int(y_all[tr].sum())}); not evaluated.", ""]
        sp = P["spread"].clip(-0.2, 0.6).to_numpy()
        Ztr_mkt = np.column_stack([P["p_mkt"].to_numpy()[tr], sp[tr]]); Zte_mkt = np.column_stack([P["p_mkt"].to_numpy()[te], sp[te]])
        mk = LogisticRegression(C=1.0, max_iter=1000).fit(Ztr_mkt, y_all[tr])
        Ztr = np.column_stack([M.logit(P[ocol].to_numpy()[tr]), Ztr_mkt]); Zte = np.column_stack([M.logit(P["model"].to_numpy()[te]), Zte_mkt])
        st = LogisticRegression(C=1.0, max_iter=1000).fit(Ztr, y_all[tr])
        y = y_all[te]
        S = {"Market: p_mkt": P["p_mkt"].to_numpy()[te], "Market: p_mkt with carry": P["p_mkt_carry"].fillna(P["p_mkt"]).to_numpy()[te],
             "Market: raw spread": sp[te], "Market: logistic on p_mkt + spread (fitted on train)": mk.predict_proba(Zte_mkt)[:, 1],
             "Model (announcement date)": P["model"].to_numpy()[te], "Model + market (stack fitted on train)": st.predict_proba(Zte)[:, 1]}
        rows = [f"### {mask_name}, target `{target}`", "",
                f"Test deals with usable prices: {int(te.sum())} ({int(y.sum())} positives). Stack fitted on {int(tr.sum())} training-period deals "
                f"({int(y_all[tr].sum())} positives) with out-of-fold model scores.", "",
                "| Score | ROC AUC [95% CI] | Positives in riskiest 20% |", "|---|---|---:|"]
        r = {"n_test": int(te.sum()), "pos_test": int(y.sum()), "n_train": int(tr.sum()), "pos_train": int(y_all[tr].sum()), "auc": {}}
        for k, s in S.items():
            a, ci = auc_ci(y, s, rng)
            r["auc"][k] = {"auc": a, "ci95": ci}
            rows.append(f"| {k} | {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {M.capture_at(y, s, 0.2):.2f} |")
        d1 = diff_ci(y, S["Model + market (stack fitted on train)"], S["Market: logistic on p_mkt + spread (fitted on train)"], rng)
        d2 = diff_ci(y, S["Model + market (stack fitted on train)"], S["Model (announcement date)"], rng)
        d3 = diff_ci(y, S["Model (announcement date)"], S["Market: p_mkt"], rng)
        r["incremental"] = {"stack_minus_market": d1, "stack_minus_model": d2, "model_minus_pmkt": d3, "stack_coef": st.coef_[0].tolist()}
        rows += ["", f"- Incremental gain of the model over the market (stack minus market-only): **{d1[0]:+.3f} AUC** [95% CI {d1[1][0]:+.3f}, {d1[1][1]:+.3f}]; positive in {d1[2]:.0%} of resamples.",
                 f"- Incremental gain of the market over the model (stack minus model-only): {d2[0]:+.3f} AUC [{d2[1][0]:+.3f}, {d2[1][1]:+.3f}].",
                 f"- Model minus p_mkt: {d3[0]:+.3f} AUC [{d3[1][0]:+.3f}, {d3[1][1]:+.3f}].",
                 f"- Stack coefficients (logit of model score, p_mkt, spread): {', '.join(f'{c:.2f}' for c in st.coef_[0])}.", ""]
        return r, rows

    core = P["core"].to_numpy(bool); ok = P["backtest_ok"].fillna(False).to_numpy(bool); cash = P["cash_deal"].fillna(False).to_numpy(bool)
    out["benchmark"] = {}
    for name, mask in (("All-cash deals, core sample (main)", core & ok & cash), ("All considerations, core sample (robustness)", core & ok)):
        for target in ("label_adverse", "label"):
            r, rows = benchmark(name, pd.Series(mask), target)
            MB += rows
            out["benchmark"][f"{name} | {target}"] = r
    MB += ["Reading the table: with this few breaks among priced test deals the intervals are wide. A positive incremental gain whose interval includes zero "
           "means the data cannot tell whether the model adds to the market.", ""]
    (res / "market_benchmark.md").write_text("\n".join(MB) + "\n")

    # ------------------------------------------------------------------ backtest
    BT = ["# Backtest on Capital IQ monthly closes (fallback without CRSP)", "",
          "The headline version uses CRSP daily prices: `results/backtest.md`.", "",
          "Test period 2019-2025, core sample. Entry: target close at t1, the first monthly grid date (the 20th) after the announcement. Exit: the offer price on the closing "
          "date for completed all-cash deals; the first grid close after the cancellation date for broken deals. Equal weights. "
          "Annual return = total P&L divided by total capital-years (each deal ties up one unit of capital for its holding period), "
          "so `$ per $100 million per year` assumes the capital is continuously redeployed into deals of the same kind.", "",
          "**Limits.** Monthly marks only: entry is up to a month after the announcement (part of the spread is already gone and some early news is in the price), "
          "and broken deals are exited on the next grid date, not at the first print after the news. No transaction costs, dividends, financing cost, borrow or "
          "position limits. Deals without a usable price are missing (see `results/price_coverage.md`), and that loss is larger for broken deals. "
          f"Model: {final}; scores from the model trained on 2001-2018.", ""]

    def run_bt(name, mask):
        te = P[mask & (P["year"] >= 2019) & P["model"].notna()].copy()
        if len(te) < 30:
            return None, [f"## {name}: too few deals.", ""]
        ret, hold = te["ret"].to_numpy(), te["hold_days"].to_numpy()
        sc, cal, pm = te["model"].to_numpy(), te["model_cal"].to_numpy(), te["p_mkt"].to_numpy()
        cls = te["outcome3"].to_numpy()
        n = len(te)

        def top(frac):
            f = np.zeros(n, bool); f[np.argsort(-sc, kind="stable")[:max(int(round(frac * n)), 1)]] = True
            return f
        strategies = {"Hold all deals": np.ones(n, bool), "Skip riskiest 10% by model": ~top(0.10), "Skip riskiest 20% by model": ~top(0.20),
                      "Hold only if model probability < market-implied": np.isfinite(pm) & (cal < pm),
                      "Hold only if model risk percentile < market risk percentile": np.isfinite(pm) & (pd.Series(sc).rank(pct=True).to_numpy() < pd.Series(pm).rank(pct=True).to_numpy()),
                      "Reference: skip widest-spread 10% (market's own ranking)": ~np.isin(np.arange(n), np.argsort(-te["spread"].to_numpy())[:max(int(round(0.1 * n)), 1)])}
        rows = [f"## {name}", "", f"{n} test deals with usable prices: {int((cls == 'closed').sum())} closed, {int((cls == 'adverse_break').sum())} adverse breaks, "
                f"{int((cls == 'topped').sum())} topped by a higher bid.", "",
                "| Strategy | Deals held | Adverse breaks held | Topped held | Mean return per deal | Median holding days | Annual return | $ million per $100 million per year | Change vs hold-all [95% CI] |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        r = {"n": n, "strategies": {}}
        base = portfolio(ret, hold)
        boot = rng.integers(0, n, (N_BOOT, n))
        for k, h in strategies.items():
            ann = portfolio(ret[h], hold[h])
            if k == "Hold all deals":
                dtxt = "reference"
                ci = None
            else:
                ds = []
                for ix in boot:
                    if "riskiest" in k:      # re-rank inside each resample
                        frac = 0.10 if "10%" in k else 0.20
                        o = np.argsort(-sc[ix], kind="stable")[:max(int(round(frac * n)), 1)]
                        hh = np.ones(n, bool); hh[o] = False
                    else:
                        hh = h[ix]
                    if hh.sum() > 5:
                        ds.append(portfolio(ret[ix][hh], hold[ix][hh]) - portfolio(ret[ix], hold[ix]))
                ci = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
                dtxt = f"{(ann - base) * 100:+.2f} [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]"
            r["strategies"][k] = {"held": int(h.sum()), "adverse_held": int((cls[h] == "adverse_break").sum()), "topped_held": int((cls[h] == "topped").sum()),
                                  "mean_ret": float(ret[h].mean()), "median_hold": float(np.median(hold[h])), "annual_return": ann,
                                  "usd_per_100m": ann * CAPITAL, "delta_vs_all": ann - base, "delta_ci95": ci}
            rows.append(f"| {k} | {int(h.sum())} | {int((cls[h] == 'adverse_break').sum())} | {int((cls[h] == 'topped').sum())} | {ret[h].mean():+.2%} | "
                        f"{np.median(hold[h]):.0f} | {ann:+.1%} | {ann * 100:+.2f} | {dtxt} |")
        rows += ["", "Change vs hold-all is in $ million per $100 million per year (equivalently percentage points of annual return).", "",
                 "By outcome (all usable deals):", "", "| Outcome | Deals | Mean return per deal | Median return | Worst | Median holding days | Mean model score | Mean p_mkt |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        r["by_outcome"] = {}
        for c, lab in (("closed", "Closed"), ("topped", "Topped by a higher bid"), ("adverse_break", "Adverse break")):
            mk = cls == c
            if mk.sum() == 0:
                continue
            r["by_outcome"][c] = {"n": int(mk.sum()), "mean_ret": float(ret[mk].mean()), "median_ret": float(np.median(ret[mk])), "min_ret": float(ret[mk].min())}
            rows.append(f"| {lab} | {int(mk.sum())} | {ret[mk].mean():+.1%} | {np.median(ret[mk]):+.1%} | {ret[mk].min():+.1%} | {np.median(hold[mk]):.0f} | "
                        f"{sc[mk].mean():.3f} | {np.nanmean(pm[mk]):.2f} |")
        rows.append("")
        return r, rows, te

    out["backtest"] = {}
    main_te = None
    for name, mask in (("All-cash deals (main)", pd.Series(core & ok & cash)), ("All considerations, including stock and mixed (robustness)", pd.Series(core & ok))):
        got = run_bt(name, mask)
        if got[0] is None:
            BT += got[1]; continue
        r, rows, te = got
        BT += rows; out["backtest"][name] = r
        if main_te is None:
            main_te = te
    plaus = (P["spread"] >= -0.05) & (P["spread"] <= 0.25)
    got = run_bt("All-cash deals with an entry spread between -5% and +25% (robustness)", pd.Series(core & ok & cash) & plaus)
    if got[0] is not None:
        BT += got[1]; out["backtest"]["All-cash, plausible spread"] = got[0]
        BT += ["This panel was added after the first backtest showed that a few deals with very wide entry spreads dominate the averages. A spread above 25% on a "
               "cash deal is either a genuinely distressed situation or a mismatch between the exported offer and the price series (revised offers, contingent "
               "consideration); a negative spread below -5% means the market expected a higher bid. Neither can be told apart from the export, so both panels are shown.", ""]
    BT += ["**How to read the model-versus-market rows.** `p_mkt` is not calibrated: it averages about 0.19 for deals that closed, several times the actual break rate, "
           "because part of every spread is time value and compensation for risk. The rule \"model probability below market-implied\" therefore mostly selects deals "
           "with wide spreads, and its return comes from holding those, not from the model. The percentile version compares ranks instead of levels and is the fairer test.", ""]
    BT += ["For stock and mixed deals the exit for completed deals is the last available close before delisting and the spread uses the exported offer value per share, "
           "which moves with the acquirer's share price; no hedge of the acquirer leg is modelled. Treat that panel as a rough check only.", ""]
    (res / "backtest.md").write_text("\n".join(BT) + "\n")
    (res / "backtest.json").write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------------ figures
    if main_te is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
        panels = [("All priced all-cash deals", out["backtest"]["All-cash deals (main)"])]
        if "All-cash, plausible spread" in out["backtest"]:
            panels.append(("Entry spread between -5% and +25% only", out["backtest"]["All-cash, plausible spread"]))
        f_, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, pan) in zip(axes, panels):
            r = pan["strategies"]; ks = [k for k in r if k != "Hold all deals"]
            ypos = np.arange(len(ks))[::-1]
            d = [r[k]["delta_vs_all"] * 100 for k in ks]
            lo = [r[k]["delta_vs_all"] * 100 - r[k]["delta_ci95"][0] * 100 for k in ks]; hi = [r[k]["delta_ci95"][1] * 100 - r[k]["delta_vs_all"] * 100 for k in ks]
            ax.errorbar(d, ypos, xerr=[lo, hi], fmt="o", color="#08519c", capsize=3)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(ypos); ax.set_yticklabels([k.replace("Reference: ", "") for k in ks], fontsize=8)
            ax.set_xlabel("Change in annual return vs holding all deals (points, 95% CI)"); ax.grid(axis="y", alpha=0)
            base = r["Hold all deals"]
            ax.set_title(f"{title}\nhold-all: {base['annual_return']:.1%} a year, {base['held']} deals, {base['adverse_held']} adverse breaks", loc="left", fontsize=9)
        f_.suptitle("Backtest on monthly closes, all-cash deals, test 2019-2025, before costs", x=0.01, ha="left", fontsize=10)
        f_.tight_layout()
        f_.savefig(fig / "monthly_backtest.png"); plt.close(f_)

        f_, ax = plt.subplots(figsize=(5.4, 4.6))
        colors = {"closed": "#9ecae1", "topped": "#31a354", "adverse_break": "#de2d26"}
        for c, lab in (("closed", "closed"), ("topped", "topped by a higher bid"), ("adverse_break", "adverse break")):
            mk = (main_te["outcome3"] == c).to_numpy()
            ax.scatter(main_te["p_mkt"].to_numpy()[mk], main_te["model_cal"].to_numpy()[mk], s=14 if c == "closed" else 36, color=colors[c],
                       alpha=0.5 if c == "closed" else 0.95, lw=0, label=lab, zorder=1 if c == "closed" else 3)
        lim = float(np.nanpercentile(main_te["model_cal"], 99.5)) * 1.1
        ax.plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1)
        ax.set(xlabel="Market-implied break probability p_mkt (at t1)", ylabel="Model probability of an adverse break (calibrated)", xlim=(-0.02, 1.02), ylim=(0, max(lim, 0.2)))
        ax.legend(fontsize=8, frameon=False); ax.set_title("Model versus market, all-cash test deals", loc="left", fontsize=10)
        f_.savefig(fig / "monthly_model_vs_market.png"); plt.close(f_)
    print("\n".join(MB[-30:])); print("\n".join(BT[5:40]))


if __name__ == "__main__":
    main()

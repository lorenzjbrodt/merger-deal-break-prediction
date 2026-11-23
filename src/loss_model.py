"""Loss if the deal breaks (and gain if it is topped), from CRSP daily prices.

Realised outcome: total return of the target from the close on t1 (first trading day after the announcement) to the
close one trading day after the cancellation date (`ret`) and five trading days after it (`ret5`), raw and net of the
market return over the same days. Adverse breaks and deals topped by a higher bid are analysed separately.

Prediction, from information available on t1 only: the price jump at announcement (which defines the benchmark
"falls back to the unaffected price": P0 / close(t1) - 1), premium, pre-announcement run-up, day-1 spread and p_mkt,
target volatility and beta, size, consideration and structure. The sample is a few hundred breaks, so the models are
deliberately simple: a constant, the fall-back benchmark, a one-variable regression on it, ridge regression and a
shallow boosted model. Evaluation: leave-one-year-out, the time split, and rolling-origin predictions (a model for
year Y only sees breaks that were announced before Y and cancelled before 1 January of Y).

Outputs: results/loss_given_break.md, .json, figures/loss_given_break.png, data/interim/loss_oot.parquet (git-ignored).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge

import modeling as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
YEARS = list(range(2013, 2026))
FEATS = ["fallback", "premium_1d", "runup_20d", "spread", "p_mkt", "vol_1y", "beta_1y", "log_deal_value", "log_target_mktcap", "cash_pct", "stock_pct",
         "feat_tender_offer", "buyer_financial", "pct_52w_high", "days_to_t1"]
MODELS = {"mean": "Training mean", "fallback": "Falls back to the unaffected price (P0 / close(t1) - 1)", "fallback_fit": "One-variable regression on the fall-back",
          "ridge": "Ridge regression, all day-1 features", "gbm": "Shallow boosted trees, all day-1 features"}
N_BOOT = 2000


def market_growth(t1, exit_):
    ff = pd.read_parquet(WRDS / "ff_daily.parquet")
    idx = pd.to_datetime(ff["date"]).to_numpy()
    g = np.cumsum(np.log1p((ff["mktrf"] + ff["rf"]).to_numpy(float)))
    a = np.searchsorted(idx, t1.to_numpy(), side="right") - 1
    b = np.searchsorted(idx, exit_.to_numpy(), side="right") - 1
    ok = t1.notna().to_numpy() & exit_.notna().to_numpy() & (exit_.to_numpy() <= idx[-1]) & (a >= 0)
    return np.where(ok, np.exp(g[np.clip(b, 0, len(g) - 1)] - g[np.clip(a, 0, len(g) - 1)]), np.nan)


def design(X, tr, te):
    """Median-impute and standardise on the training rows; add missing indicators for the market features."""
    A, B = X.iloc[tr].copy(), X.iloc[te].copy()
    med = A.median()
    for c in ("spread", "p_mkt", "vol_1y"):
        A[f"{c}_missing"], B[f"{c}_missing"] = A[c].isna().astype(float), B[c].isna().astype(float)
    A, B = A.fillna(med).fillna(0), B.fillna(med).fillna(0)
    mu, sd = A.mean(), A.std().replace(0, 1).fillna(1)
    return ((A - mu) / sd).to_numpy(), ((B - mu) / sd).to_numpy(), list(A.columns)


def predict_all(X, y, tr, te, year_tr=None):
    """Predictions of every model for rows `te` from rows `tr`. Ridge strength is picked by leave-one-year-out inside `tr`."""
    P = {"mean": np.full(len(te), y[tr].mean()), "fallback": X["fallback"].to_numpy()[te]}
    fb = X["fallback"].to_numpy()
    P["fallback_fit"] = LinearRegression().fit(fb[tr, None], y[tr]).predict(fb[te, None])
    A, B, _ = design(X, tr, te)
    alpha = 30.0
    if year_tr is not None and len(np.unique(year_tr)) >= 4:
        best = (np.inf, alpha)
        for a in (3.0, 10.0, 30.0, 100.0, 300.0):
            err = []
            for v in np.unique(year_tr):
                k = year_tr == v
                if k.sum() and (~k).sum() > 20:
                    a_, b_, _ = design(X, tr[~k], tr[k])
                    err.append(np.abs(Ridge(alpha=a).fit(a_, y[tr[~k]]).predict(b_) - y[tr[k]]))
            e = np.concatenate(err).mean()
            if e < best[0]:
                best = (e, a)
        alpha = best[1]
    P["ridge"] = Ridge(alpha=alpha).fit(A, y[tr]).predict(B)
    Xt, Xe = X.iloc[tr].to_numpy(float), X.iloc[te].to_numpy(float)
    P["gbm"] = np.mean([LGBMRegressor(objective="regression", num_leaves=3, n_estimators=100, learning_rate=0.03, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=s, n_jobs=4).fit(Xt, y[tr]).predict(Xe) for s in range(3)], axis=0)
    return P, alpha


def scores(y, P, ref, rng):
    """MAE, RMSE and out-of-sample R2 against `ref` (the training-mean forecast), with bootstrap intervals over deals."""
    out = {}
    n = len(y)
    draws = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    for k, p in P.items():
        mae = float(np.mean(np.abs(p - y))); r2 = float(1 - np.sum((p - y) ** 2) / np.sum((ref - y) ** 2))
        b = np.array([[np.mean(np.abs(p[ix] - y[ix])), 1 - np.sum((p[ix] - y[ix]) ** 2) / np.sum((ref[ix] - y[ix]) ** 2),
                       np.mean(np.abs(p[ix] - y[ix])) - np.mean(np.abs(ref[ix] - y[ix]))] for ix in draws])
        out[k] = {"mae": mae, "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "r2_vs_mean": r2, "corr": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else float("nan"),
                  "mae_ci95": [float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))], "r2_ci95": [float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))],
                  "mae_minus_mean": mae - float(np.mean(np.abs(ref - y))), "mae_minus_mean_ci95": [float(np.percentile(b[:, 2], 2.5)), float(np.percentile(b[:, 2], 97.5))]}
    return out


def table(sc):
    L = ["| Forecast | MAE [95% CI] | MAE minus training mean [95% CI] | Out-of-sample R2 vs training mean [95% CI] | Correlation |", "|---|---|---|---|---:|"]
    for k, lab in MODELS.items():
        if k in sc:
            r = sc[k]
            L.append(f"| {lab} | {r['mae']:.3f} [{r['mae_ci95'][0]:.3f}, {r['mae_ci95'][1]:.3f}] | {r['mae_minus_mean']:+.3f} [{r['mae_minus_mean_ci95'][0]:+.3f}, {r['mae_minus_mean_ci95'][1]:+.3f}] | "
                     f"{r['r2_vs_mean']:+.2f} [{r['r2_ci95'][0]:+.2f}, {r['r2_ci95'][1]:+.2f}] | {'n/a' if k == 'mean' else format(r['corr'], '.2f')} |")
    return L + [""]


def main() -> None:
    res, fig = ROOT / "results", ROOT / "figures"
    D = M.load_data("")
    assert D.pf_source == "CRSP daily"
    PF = D.PF
    n = len(D.year)
    idx = np.arange(n)
    X = pd.DataFrame({"fallback": PF["P0"] / PF["close_t1"] - 1, "premium_1d": D.X["premium_1d"] / 100, "runup_20d": PF["runup_20d"], "spread": PF["spread"], "p_mkt": PF["p_mkt"],
                      "vol_1y": PF["vol_1y"], "beta_1y": PF["beta_1y"], "days_to_t1": PF["days_to_t1"]})
    for c in ("log_deal_value", "log_target_mktcap", "cash_pct", "stock_pct", "feat_tender_offer", "buyer_financial", "pct_52w_high"):
        X[c] = D.X[c].to_numpy(float) if c in D.X else np.nan
    X["cash_pct"] = X["cash_pct"].fillna(pd.Series(np.where(PF["cash_deal"].fillna(False), 100.0, np.nan)))
    X = X[FEATS]
    ok = (D.core & PF["t1_ok"].fillna(False).to_numpy() & PF["price_valid"].fillna(False).to_numpy() & np.isfinite(X["fallback"].to_numpy())
          & (X["fallback"].to_numpy() > -0.9) & (X["fallback"].to_numpy() < 1.0))
    canc = D.deals["cancelled_date"].to_numpy().astype("datetime64[D]")
    mg1, mg5 = market_growth(PF["t1"], PF["exit_date"]), market_growth(PF["t1"], PF["exit_date5"])
    R = {"ret": PF["ret"].to_numpy(), "ret5": PF["ret5"].to_numpy(), "ret_adj": (1 + PF["ret"].to_numpy()) / mg1 - 1, "ret5_adj": (1 + PF["ret5"].to_numpy()) / mg5 - 1}
    rng = np.random.default_rng(0)
    out = {"realised": {}, "adverse": {}, "topped": {}}
    L = ["# Loss if the deal breaks", "",
         "Core sample, CRSP daily prices. Return of the target from the close on t1 (the first trading day after the announcement) to the close one and five trading days "
         "after the cancellation date, total return including dividends; \"net of market\" divides by the market's return over the same days (CRSP value-weighted market from "
         "the Fama-French file). All considerations: the long target leg is the same in cash and stock deals; no acquirer hedge is modelled.", "",
         "## What happened", "",
         "| Outcome | Period | Deals | Mean, 1 day after | Median | 10th pct | 90th pct | Mean, 5 days after | Mean net of market, 1 day | Mean net of market, 5 days | Median days held |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    hold = PF["hold_days"].to_numpy()
    for cls, lab in (("adverse_break", "Adverse break"), ("topped", "Topped by a higher bid")):
        for per, pm in (("2001-2018", D.year <= 2018), ("2019-2025", D.year >= 2019), ("all years", np.ones(n, bool))):
            m = ok & (D.outcome3 == cls) & pm & np.isfinite(R["ret"])
            if m.sum() < 3:
                continue
            r = R["ret"][m]
            out["realised"][f"{cls} | {per}"] = {"n": int(m.sum()), "mean": float(r.mean()), "median": float(np.median(r)), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90)),
                                                 "mean5": float(np.nanmean(R["ret5"][m])), "mean_adj": float(np.nanmean(R["ret_adj"][m])), "mean5_adj": float(np.nanmean(R["ret5_adj"][m]))}
            L.append(f"| {lab} | {per} | {int(m.sum())} | {r.mean():+.1%} | {np.median(r):+.1%} | {np.percentile(r, 10):+.1%} | {np.percentile(r, 90):+.1%} | {np.nanmean(R['ret5'][m]):+.1%} | "
                     f"{np.nanmean(R['ret_adj'][m]):+.1%} | {np.nanmean(R['ret5_adj'][m]):+.1%} | {np.median(hold[m]):.0f} |")
    m = ok & (D.outcome3 == "adverse_break") & np.isfinite(R["ret"])
    fb = X["fallback"].to_numpy()
    cashm = PF["cash_deal"].fillna(False).to_numpy()
    L += ["", f"For adverse breaks the fall-back benchmark (the price returns to the pre-announcement close) implies a mean loss of {fb[m].mean():+.1%}; the realised mean is {R['ret'][m].mean():+.1%} "
          f"raw and {np.nanmean(R['ret_adj'][m]):+.1%} net of market. All-cash deals: {int((m & cashm).sum())} adverse breaks, mean {R['ret'][m & cashm].mean():+.1%}; "
          f"stock and mixed: {int((m & ~cashm).sum())}, mean {R['ret'][m & ~cashm].mean():+.1%}. "
          "Breaks take months to happen, so the raw return also contains whatever the market and the company did in between; that is why raw and market-adjusted differ.", ""]

    pred_cols = {}
    for cls, lab, key in (("adverse_break", "Adverse breaks: predicting the loss", "adverse"), ("topped", "Deals topped by a higher bid: predicting the gain", "topped")):
        for tgt, tlab in (("ret", "raw return to one day after the cancellation"), ("ret_adj", "return net of market")):
            y = R[tgt]
            rows = idx[ok & (D.outcome3 == cls) & np.isfinite(y)]
            if len(rows) < 40:
                continue
            yr = D.year[rows]
            # leave-one-year-out
            P = {k: np.full(len(rows), np.nan) for k in MODELS}
            for v in np.unique(yr):
                te = yr == v
                p, _ = predict_all(X, y, rows[~te], rows[te], yr[~te])
                for k in MODELS:
                    P[k][te] = p[k]
            ref = P["mean"]
            sc_loyo = scores(y[rows], P, ref, rng)
            # time split
            tr, te = rows[yr <= 2018], rows[yr >= 2019]
            sc_split = None
            if len(te) >= 10 and len(tr) >= 30:
                p, alpha = predict_all(X, y, tr, te, D.year[tr])
                sc_split = scores(y[te], p, p["mean"], rng)
            slope = LinearRegression().fit(fb[rows, None], y[rows])
            out[key][tgt] = {"n": int(len(rows)), "loyo": sc_loyo, "split": sc_split, "n_test": int(len(te)), "slope_on_fallback": float(slope.coef_[0]), "intercept": float(slope.intercept_)}
            L += [f"## {lab} ({tlab}; {len(rows)} deals)", "", "Leave-one-year-out (every year predicted from all other years):", ""] + table(sc_loyo)
            if sc_split:
                L += [f"Time split (trained on {len(tr)} breaks announced 2001-2018, tested on {len(te)} announced 2019-2025):", ""] + table(sc_split)
            L += [f"One-variable fit on all {len(rows)} deals: realised = {slope.intercept_:+.3f} {slope.coef_[0]:+.2f} x fall-back. A slope of 1 with intercept 0 would mean prices return exactly to the unaffected price.", ""]
        # rolling-origin predictions for the expected-return model (raw return to one day after the cancellation)
        y = R["ret"]
        rows = idx[ok & (D.outcome3 == cls) & np.isfinite(y)]
        for tag, origins in (("oot", YEARS), ("split", [None])):
            col = np.full(n, np.nan); chosen = {}
            for Y in origins:
                cut = np.datetime64(f"{Y}-01-01") if Y else np.datetime64("2019-01-01")
                tr = rows[(D.year[rows] <= ((Y - 1) if Y else 2018)) & (canc[rows] < cut)]
                te = idx[D.core & ((D.year == Y) if Y else (D.year >= 2019)) & np.isfinite(fb)]
                if len(tr) < 25 or len(te) == 0:
                    continue
                # pick the forecast by leave-one-year-out inside the training breaks
                yt = D.year[tr]; inner = {k: [] for k in MODELS}
                for v in np.unique(yt):
                    k_ = yt == v
                    if k_.sum() and (~k_).sum() >= 20:
                        p, _ = predict_all(X, y, tr[~k_], tr[k_], None)
                        for k in MODELS:
                            inner[k].append(np.abs(p[k] - y[tr[k_]]))
                best = min(MODELS, key=lambda k: np.concatenate(inner[k]).mean() if inner[k] else np.inf)
                p, _ = predict_all(X, y, tr, te, yt)
                col[te] = p[best]; chosen[Y or "split"] = best
            pred_cols[f"{tag}__{key}"] = col
            out[key][f"chosen_{tag}"] = {str(k): v for k, v in chosen.items()}
            if tag == "oot" and chosen:
                e = rows[np.isfinite(col[rows])]
                if len(e) >= 10:
                    mae = float(np.mean(np.abs(col[e] - y[e]))); base = float(np.mean(np.abs(y[e] - y[e].mean())))
                    out[key]["oot_eval"] = {"n": int(len(e)), "mae": mae, "mae_constant": base, "corr": float(np.corrcoef(col[e], y[e])[0, 1])}
                    cnt = pd.Series(list(chosen.values())).value_counts()
                    L += [f"Rolling origin, 2013-2025 ({len(e)} {lab.split(':')[0].lower()}): each year's forecast comes from breaks announced and cancelled before that year, and the forecast "
                          f"type is chosen by leave-one-year-out inside those. Chosen: {', '.join(f'{MODELS[k].split(chr(40))[0].strip().lower()} x{v}' for k, v in cnt.items())}. "
                          f"MAE {mae:.3f} against {base:.3f} for the pooled mean of the same deals; correlation with the realised return {np.corrcoef(col[e], y[e])[0, 1]:.2f}.", ""]
    L += ["## Reading", "",
          "- The numbers above are returns of an unhedged long position entered one day after the announcement. They are the loss leg of an arbitrageur's payoff, not a prediction of where the price settles in the long run.",
          "- Samples are small and the losses are widely dispersed, so the intervals are wide. A forecast whose MAE interval against the training mean includes zero has not been shown to help.", ""]
    (res / "loss_given_break.md").write_text("\n".join(L) + "\n")
    (res / "loss_given_break.json").write_text(json.dumps(out, indent=2, default=float))
    sv = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "fallback": fb, "ret_adj": R["ret_adj"], "ret5_adj": R["ret5_adj"], **pred_cols})
    sv.to_parquet(INTERIM / "loss_oot.parquet", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    f_, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cls, col_, lab in (("adverse_break", "#de2d26", "adverse break"), ("topped", "#31a354", "topped by a higher bid")):
        m = ok & (D.outcome3 == cls) & np.isfinite(R["ret"])
        ax[0].hist(np.clip(R["ret"][m], -1, 1), bins=np.linspace(-1, 1, 41), alpha=0.6, color=col_, label=f"{lab} ({int(m.sum())})")
    ax[0].axvline(0, color="k", lw=0.8); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set(xlabel="Target return from the day-1 close to one day after the cancellation", ylabel="Deals"); ax[0].set_title("What a broken deal costs", loc="left", fontsize=10)
    m = ok & (D.outcome3 == "adverse_break") & np.isfinite(R["ret"])
    ax[1].scatter(fb[m], R["ret"][m], s=14, color="#de2d26", alpha=0.6, lw=0)
    ax[1].plot([-0.7, 0.1], [-0.7, 0.1], color="#9e9e9e", ls="--", lw=1, label="falls back exactly to the unaffected price")
    ax[1].set(xlabel="Fall-back benchmark: P0 / close(t1) - 1", ylabel="Realised return of adverse breaks", xlim=(-0.75, 0.15), ylim=(-1.0, 0.8)); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Adverse breaks: realised loss against the fall-back benchmark", loc="left", fontsize=10)
    f_.tight_layout(); f_.savefig(fig / "loss_given_break.png"); plt.close(f_)
    print("\n".join(L))


if __name__ == "__main__":
    main()

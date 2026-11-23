"""Day-1 market signals beyond the target spread.

Question: does anything observable at the close of the first trading day after the announcement (t1) predict adverse
breaks better than the target spread alone?

SPECIFICATION, FIXED BEFORE ANY OUT-OF-TIME RESULT WAS LOOKED AT (two models, no tuning, no iteration):
  inputs      p_mkt, day-1 spread (clipped to [-0.2, 0.6]), the announcement-date model score (log-odds), and
              acq_adj_ret       acquirer return from the close before announcement to t1, net of the market (listed acquirers)
              acq_ret_x_stock   the same times the stock share of the consideration
              abvol_ann, abvol_t1   log target volume on the announcement day and on t1 relative to its 60-day average
              neg_spread        1 if the target closed above the offer on t1
              runup_30d         target return over the 30 trading days before the announcement
              idio_vol          target idiosyncratic volatility (market-model residual, one year ending 30 days before)
              spread_rel        spread minus the median day-1 spread of deals announced in the prior 90 days
              rel_size          deal value relative to the acquirer's market capitalisation
              overlap_x_value   Hoberg-Phillips product-similarity score times log deal value
  cleaning    signals are clipped to fixed limits chosen from their distributions alone (CLIP below), before any model was fitted
  model 1     logistic regression, L2 with C = 0.1, inputs standardised, medians imputed, missing-value indicators
  model 2     LightGBM, 3 leaves, 200 trees, learning rate 0.03, minimum 20 deals per leaf, average of 5 seeds
  fitting     per rolling origin on earlier years only; the announcement-date score of training deals is the
              out-of-fold score of that origin
  sample      priced all-cash core deals (the sample of the market benchmark); all considerations as one robustness set
  evaluation  once, on the pooled out-of-time set 2013-2025 and on the test period 2019-2025, against the market-only
              benchmark: ROC AUC difference and share of adverse breaks in the riskiest 20%, intervals by announcement year
  use in the expected-return step: the day-1 model replaces the market-implied probability for a test year only if the
              cross-validation inside that origin's training years prefers it (higher mean AUC than p_mkt on
              time-ordered validation blocks); the better of the two models by that CV is the one used.

  python src/day1_signals.py build   data/interim/day1_features.parquet
  python src/day1_signals.py run     results/day1_signals.md, .json, data/interim/day1_signals.parquet
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import crsp_prices as CP
import modeling as M
import pooled_oot as PO

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
D8 = "datetime64[D]"
FEATS = ["acq_adj_ret", "acq_ret_x_stock", "abvol_ann", "abvol_t1", "neg_spread", "runup_30d", "idio_vol", "spread_rel", "rel_size", "overlap_x_value"]
NICE = {"p_mkt": "market-implied probability", "spread": "day-1 spread", "score": "announcement-date model score (log-odds)", "acq_adj_ret": "acquirer return to t1, net of market",
        "acq_ret_x_stock": "acquirer return x stock share", "abvol_ann": "abnormal target volume, announcement day", "abvol_t1": "abnormal target volume, t1",
        "neg_spread": "target closed above the offer", "runup_30d": "pre-announcement run-up (30 days)", "idio_vol": "target idiosyncratic volatility",
        "spread_rel": "spread minus recent median spread", "rel_size": "deal value / acquirer market cap", "overlap_x_value": "product overlap x deal size"}
LOGIT_C = 0.1
CLIP = {"acq_adj_ret": (-0.5, 0.5), "abvol_ann": (-3, 8), "abvol_t1": (-3, 8), "runup_30d": (-0.9, 2.0), "idio_vol": (0, 2.0)}
LGBM = dict(objective="binary", num_leaves=3, n_estimators=200, learning_rate=0.03, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=5.0, verbose=-1, n_jobs=4)
N_BOOT = 2000
SCORES = {"p_mkt": "Market-only benchmark: p_mkt", "mkt": "Market logistic (p_mkt and spread)", "logit": "Day-1 model 1: logistic regression",
          "lgbm": "Day-1 model 2: LightGBM", "selected": "Probability passed to the expected-return step"}


# ----------------------------------------------------------------------------- features
def build() -> None:
    D = M.load_data("")
    PF = D.PF
    link = pd.read_parquet(INTERIM / "crsp_link.parquet").set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    t = pd.concat([pd.read_parquet(WRDS / "crsp_daily_target_pre.parquet"), pd.read_parquet(WRDS / "crsp_daily_target.parquet")[["permno", "date", "dlyprc", "dlyret", "dlyvol"]]])
    t = t.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    t["date"] = pd.to_datetime(t["date"]).to_numpy().astype(D8)
    by = {int(p): (g["date"].to_numpy().astype(D8), g["dlyvol"].to_numpy(float), g["dlyret"].to_numpy(float)) for p, g in t.groupby("permno", sort=False)}
    Aq = CP.crsp_series(WRDS / "crsp_daily_acquirer.parquet")
    ff = pd.read_parquet(WRDS / "ff_daily.parquet")
    fd = pd.to_datetime(ff["date"]).to_numpy().astype(D8)
    mret = (ff["mktrf"] + ff["rf"]).to_numpy(float)
    mcum = np.cumsum(np.log1p(mret))
    mser = pd.Series(mret, index=fd)

    def mkt_growth(a, b):
        i, j = np.searchsorted(fd, a, side="right") - 1, np.searchsorted(fd, b, side="right") - 1
        return float(np.exp(mcum[j] - mcum[i])) if 0 <= i < j < len(fd) else np.nan
    ann = D.deals["announced_date"].to_numpy().astype(D8)
    tm1, t1 = PF["tm1"].to_numpy().astype(D8), PF["t1"].to_numpy().astype(D8)
    stock = np.nan_to_num(D.deals["stock_pct"].to_numpy(float)) / 100
    rows = []
    for k in range(len(D.year)):
        o = {f: np.nan for f in FEATS}
        o["deal_id"] = D.deals["deal_id"].iat[k]
        if np.isnat(tm1[k]) or np.isnat(t1[k]):
            rows.append(o); continue
        p = link["permno"].iat[k]
        if pd.notna(p) and int(p) in by:
            dts, vol, ret = by[int(p)]
            i0 = np.searchsorted(dts, tm1[k], side="right") - 1                    # last row on or before tm1
            if i0 >= 59:
                avg = np.nanmean(vol[i0 - 59:i0 + 1])
                ia = np.searchsorted(dts, ann[k])                                   # first trading row on or after the announcement
                i1 = np.searchsorted(dts, t1[k])
                if avg > 0 and ia < len(dts) and dts[ia] <= t1[k] and np.isfinite(vol[ia]):
                    o["abvol_ann"] = float(np.log((vol[ia] + 1) / (avg + 1)))
                if avg > 0 and i1 < len(dts) and dts[i1] == t1[k] and np.isfinite(vol[i1]):
                    o["abvol_t1"] = float(np.log((vol[i1] + 1) / (avg + 1)))
            if i0 >= 29:
                o["runup_30d"] = float(np.prod(1 + np.nan_to_num(ret[i0 - 29:i0 + 1])) - 1)
            lo, hi = max(i0 - 281, 0), i0 - 29
            if hi - lo >= 120:
                x, dd = ret[lo:hi], dts[lo:hi]
                okx = np.isfinite(x)
                m = mser.reindex(dd[okx]).to_numpy()
                kk = np.isfinite(m)
                if kk.sum() >= 120 and np.var(m[kk]) > 0:
                    b = np.cov(x[okx][kk], m[kk])[0, 1] / np.var(m[kk], ddof=1)
                    o["idio_vol"] = float(np.std(x[okx][kk] - b * m[kk], ddof=1) * np.sqrt(252))
        ap = link["acq_permno"].iat[k]
        s = Aq.get(int(ap)) if pd.notna(ap) else None
        if s is not None:
            a0, d0 = s.last_before(tm1[k], 7); a1 = s.at(t1[k])
            g = mkt_growth(tm1[k], t1[k])
            if np.isfinite(a0) and np.isfinite(a1) and np.isfinite(g):
                o["acq_adj_ret"] = float(s.growth(d0, t1[k]) / g - 1)
                o["acq_ret_x_stock"] = o["acq_adj_ret"] * stock[k]
        rows.append(o)
    F = pd.DataFrame(rows)
    sp = PF["spread"].clip(-0.2, 0.6).to_numpy()
    F["neg_spread"] = np.where(np.isfinite(sp), (sp < 0).astype(float), np.nan)
    # spread relative to the median day-1 spread of usable all-cash deals announced in the prior 90 days
    use = (PF["backtest_ok"].fillna(False) & PF["cash_deal"].fillna(False)).to_numpy() & D.core & np.isfinite(sp)
    rel = np.full(len(F), np.nan)
    order = np.argsort(ann)
    a_s, s_s, u_s = ann[order], sp[order], use[order]
    for pos, k in enumerate(order):
        if not np.isfinite(sp[k]):
            continue
        lo = np.searchsorted(a_s, ann[k] - np.timedelta64(90, "D")); hi = np.searchsorted(a_s, ann[k])      # strictly before this announcement
        w = s_s[lo:hi][u_s[lo:hi]]
        if len(w) >= 5:
            rel[k] = sp[k] - np.median(w)
    F["spread_rel"] = rel
    F["rel_size"] = D.X["rel_size_deal_to_acq"].to_numpy(float) if "rel_size_deal_to_acq" in D.X else np.nan
    F["overlap_x_value"] = (D.X["ov_tnic_score"] * D.X["log_deal_value"]).to_numpy(float) if "ov_tnic_score" in D.X else np.nan
    for c, (lo_, hi_) in CLIP.items():                           # limits set from the feature distributions alone (bad links, splits)
        F[c] = F[c].clip(lo_, hi_)
    F["acq_ret_x_stock"] = F["acq_adj_ret"] * stock
    F[["deal_id"] + FEATS].to_parquet(INTERIM / "day1_features.parquet", index=False)
    print(F[FEATS].notna().sum().to_string())


# ----------------------------------------------------------------------------- models
def design(X: pd.DataFrame, tr, te):
    A, B = X.iloc[tr].copy(), X.iloc[te].copy()
    med = A.median()
    for c in FEATS:
        if A[c].isna().any() or B[c].isna().any():
            A[f"{c}_missing"], B[f"{c}_missing"] = A[c].isna().astype(float), B[c].isna().astype(float)
    A, B = A.fillna(med).fillna(0), B.fillna(med).fillna(0)
    mu, sd = A.mean(), A.std().replace(0, 1).fillna(1)
    return ((A - mu) / sd).to_numpy(), ((B - mu) / sd).to_numpy(), list(A.columns)


def fit_logit(X, y, tr, te):
    A, B, names = design(X, tr, te)
    m = LogisticRegression(C=LOGIT_C, max_iter=3000).fit(A, y[tr])
    return m.predict_proba(B)[:, 1], dict(zip(names, m.coef_[0].tolist()))


def fit_lgbm(X, y, tr, te):
    A, B = X.iloc[tr].to_numpy(float), X.iloc[te].to_numpy(float)
    return np.mean([LGBMClassifier(random_state=s, **LGBM).fit(A, y[tr]).predict_proba(B)[:, 1] for s in M.SEEDS], axis=0)


def one_origin(X, y, year, pm, sp, a, b):
    """Fit both models and the market logistic on rows `a`, predict rows `b`; CV inside `a` decides what M uses."""
    out = {}
    Zm = np.column_stack([pm, sp])
    out["mkt"] = LogisticRegression(C=1.0, max_iter=2000).fit(Zm[a], y[a]).predict_proba(Zm[b])[:, 1]
    out["logit"], coef = fit_logit(X, y, a, b)
    out["lgbm"] = fit_lgbm(X, y, a, b)
    cv = {"logit": [], "lgbm": [], "p_mkt": []}
    for ftr, fva in M.cv_folds(year, a, n_blocks=3, block=3):
        if y[ftr].sum() < 8 or y[fva].sum() < 2:
            continue
        cv["logit"].append(PO.fast_auc(y[fva], fit_logit(X, y, ftr, fva)[0]))
        cv["lgbm"].append(PO.fast_auc(y[fva], fit_lgbm(X, y, ftr, fva)))
        cv["p_mkt"].append(PO.fast_auc(y[fva], pm[fva]))
    info = {"n_train": int(len(a)), "pos_train": int(y[a].sum()), "folds": len(cv["p_mkt"]), "cv_auc": {k: float(np.mean(v)) if v else None for k, v in cv.items()}, "coef": coef}
    best = max(("logit", "lgbm"), key=lambda k: info["cv_auc"][k] if info["cv_auc"][k] is not None else -1)
    info["best_day1"] = best
    info["use_day1"] = bool(info["folds"] > 0 and info["cv_auc"][best] > info["cv_auc"]["p_mkt"])
    out["selected"] = out[best] if info["use_day1"] else out["mkt"]
    return out, info


def boot(y, yr, S, pairs, seed=0):
    """AUC and top-20% capture with intervals over announcement years (and over deals, for comparison)."""
    stat = lambda yy, s: [PO.fast_auc(yy, s), M.capture_at(yy, s, 0.20)]
    pt = {k: np.array(stat(y, s)) for k, s in S.items()}
    res = {"scores": {k: {"auc": float(v[0]), "cap20": float(v[1])} for k, v in pt.items()}, "pairs": {f"{a} minus {b}": {"auc": float(pt[a][0] - pt[b][0]), "cap20": float(pt[a][1] - pt[b][1])} for a, b in pairs}}
    blocks = [np.where(yr == v)[0] for v in np.unique(yr)]
    for mode in ("year", "deal"):
        rng = np.random.default_rng(seed)
        acc = {k: [] for k in S}
        for _ in range(N_BOOT):
            ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) if mode == "year" else rng.integers(0, len(y), len(y))
            if not 1 < y[ix].sum() < len(ix):
                continue
            for k, s in S.items():
                acc[k].append(stat(y[ix], s[ix]))
        arr = {k: np.array(v) for k, v in acc.items()}
        for k in S:
            res["scores"][k][f"ci_{mode}"] = {"auc": [float(np.percentile(arr[k][:, 0], 2.5)), float(np.percentile(arr[k][:, 0], 97.5))],
                                              "cap20": [float(np.percentile(arr[k][:, 1], 2.5)), float(np.percentile(arr[k][:, 1], 97.5))]}
        for a, b in pairs:
            d = arr[a] - arr[b]
            res["pairs"][f"{a} minus {b}"][f"ci_{mode}"] = {"auc": [float(np.percentile(d[:, 0], 2.5)), float(np.percentile(d[:, 0], 97.5))],
                                                           "cap20": [float(np.percentile(d[:, 1], 2.5)), float(np.percentile(d[:, 1], 97.5))], "share_gt0": float((d[:, 0] > 0).mean())}
    return res


def run() -> None:
    res = ROOT / "results"
    D = M.load_data("")
    PF = D.PF
    n = len(D.year)
    idx = np.arange(n)
    y = D.y["label_adverse"]
    F = D.deals[["deal_id"]].merge(pd.read_parquet(INTERIM / "day1_features.parquet"), on="deal_id", how="left")
    pm, sp = PF["p_mkt"].to_numpy(), PF["spread"].clip(-0.2, 0.6).to_numpy()
    final = json.loads((res / "metrics.json").read_text())["final_models"]["overall"]
    P = pd.read_parquet(INTERIM / "predictions.parquet").set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    Z = pickle.load(open(INTERIM / "pooled_oot.pkl", "rb"))
    Zp = Z["pred"].set_index("deal_id").reindex(D.deals["deal_id"]).reset_index()
    pos = pd.Series(idx, index=D.deals["deal_id"].to_numpy())
    ok = PF["backtest_ok"].fillna(False).to_numpy(bool) & D.core & np.isfinite(pm)
    cash = PF["cash_deal"].fillna(False).to_numpy(bool)
    masks = {"cash": ok & cash, "all": ok}
    out = {"spec": {"logit_C": LOGIT_C, "lgbm": {k: v for k, v in LGBM.items() if k != "n_jobs"}, "features": FEATS, "specifications_tried": 2}, "sets": {}, "origins": {}}
    sv = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy()})

    def xmat(score):
        X = pd.DataFrame({"p_mkt": pm, "spread": sp, "score": M.logit(np.where(np.isfinite(score), score, 0.5))})
        for c in FEATS:
            X[c] = F[c].to_numpy(float)
        return X
    for key, mask in masks.items():
        for view in ("pooled", "test"):
            S = {k: np.full(n, np.nan) for k in ("mkt", "logit", "lgbm", "selected")}
            rows = np.zeros(n, bool)
            origins = Z["years"] if view == "pooled" else [None]
            for Y in origins:
                if Y is None:
                    oof, pred = P[f"oof__primary__full__{final}"].to_numpy(), P[f"pred__primary__full__{final}"].to_numpy()
                    tr_m, te_m = D.year <= 2018, D.year >= 2019
                else:
                    ids, op = Z["oof"].get((Y, "full", "final_proc"), (np.array([]), np.array([])))
                    oof = np.full(n, np.nan)
                    if len(ids):
                        oof[pos.reindex(ids).to_numpy()] = op
                    pred = Zp["pred__full__final_proc"].to_numpy()
                    tr_m, te_m = D.year <= Y - 1, D.year == Y
                a, b = idx[tr_m & mask & np.isfinite(oof)], idx[te_m & mask & np.isfinite(pred)]
                if y[a].sum() < 8 or len(b) == 0:
                    continue
                score = np.where(tr_m, oof, pred)
                o, info = one_origin(xmat(score), y, D.year, pm, sp, a, b)
                for k in S:
                    S[k][b] = o[k]
                rows[b] = True
                out["origins"][f"{key} | {view} | {Y or 'split'}"] = info
            yy, yr = y[rows], D.year[rows]
            SS = {"p_mkt": pm[rows], **{k: v[rows] for k, v in S.items()}}
            ev = boot(yy, yr, SS, [("logit", "p_mkt"), ("lgbm", "p_mkt"), ("logit", "mkt"), ("lgbm", "mkt"), ("selected", "p_mkt")])
            ev["n"], ev["pos"] = int(rows.sum()), int(yy.sum())
            out["sets"][f"{key} | {view}"] = ev
            for k, v in S.items():
                sv[f"{key}__{view}__{k}"] = v
            sv[f"{key}__{view}__rows"] = rows
    sv.to_parquet(INTERIM / "day1_signals.parquet", index=False)

    L = ["# Day-1 market signals beyond the target spread", "",
         "Question: does anything observable at the close of the first trading day after the announcement predict adverse breaks better than the target spread alone?", "",
         "**Specification, fixed before any out-of-time result was looked at.** Two models, both on the same inputs: the market-implied probability, the day-1 spread, the "
         "announcement-date model score, and ten day-1 signals (" + "; ".join(NICE[f] for f in FEATS) + "). Model 1 is a logistic regression (L2, C = 0.1, standardised inputs, "
         "missing-value indicators); model 2 is a LightGBM with 3 leaves and 200 trees. No hyper-parameter was tuned. Both are refitted for every test year on earlier years only. "
         "**Specifications tried: these two.** They were evaluated once; nothing was changed after seeing the results below.", "",
         "Sample: priced all-cash core deals, as in `results/market_benchmark.md`. Benchmark: the market-implied probability `p_mkt` by itself. "
         "Intervals resample announcement years (13 for the pooled set, 7 for the test period, so the test-period intervals are rough).", ""]
    cov = F[FEATS][masks["cash"]].notna().mean()
    L += ["Share of priced all-cash deals with each signal: " + ", ".join(f"{NICE[f]} {cov[f]:.0%}" for f in FEATS) + ". "
          "For all-cash deals the acquirer-return-times-stock-share term is zero by construction; it matters only in the robustness set with stock and mixed deals.", ""]
    for key, title in (("cash", "All-cash deals (main)"), ("all", "All considerations (robustness; spreads of stock and mixed deals from an implied exchange ratio)")):
        L += [f"## {title}", ""]
        for view, vt in (("pooled", "Pooled out-of-time 2013-2025"), ("test", "Test period 2019-2025")):
            ev = out["sets"].get(f"{key} | {view}")
            if not ev:
                continue
            L += [f"### {vt}: {ev['n']} deals, {ev['pos']} adverse breaks", "", "| Score | ROC AUC [95% CI by year] | Adverse breaks in riskiest 20% [95% CI] |", "|---|---|---|"]
            for k, lab in SCORES.items():
                r = ev["scores"][k]
                L.append(f"| {lab} | {r['auc']:.3f} [{r['ci_year']['auc'][0]:.3f}, {r['ci_year']['auc'][1]:.3f}] | {r['cap20']:.2f} [{r['ci_year']['cap20'][0]:.2f}, {r['ci_year']['cap20'][1]:.2f}] |")
            L += ["", "| Difference | ROC AUC [95% CI by year] | [95% CI by deal] | Riskiest-20% capture [95% CI by year] |", "|---|---|---|---|"]
            for pk, lab in (("logit minus p_mkt", "Logistic day-1 model minus p_mkt"), ("lgbm minus p_mkt", "LightGBM day-1 model minus p_mkt"),
                            ("logit minus mkt", "Logistic day-1 model minus market logistic"), ("lgbm minus mkt", "LightGBM day-1 model minus market logistic"),
                            ("selected minus p_mkt", "Probability passed on minus p_mkt")):
                r = ev["pairs"][pk]
                L.append(f"| {lab} | {r['auc']:+.3f} [{r['ci_year']['auc'][0]:+.3f}, {r['ci_year']['auc'][1]:+.3f}] | [{r['ci_deal']['auc'][0]:+.3f}, {r['ci_deal']['auc'][1]:+.3f}] | "
                         f"{r['cap20']:+.2f} [{r['ci_year']['cap20'][0]:+.2f}, {r['ci_year']['cap20'][1]:+.2f}] |")
            L.append("")
    # CV decisions
    L += ["## What the training-period cross-validation preferred (all-cash deals)", "",
          "For every origin the two day-1 models and `p_mkt` are compared on time-ordered validation blocks inside the training years. The day-1 model is passed to the "
          "expected-return step for that year only if its mean validation AUC is higher than that of `p_mkt`; otherwise the market-implied probability (through the market logistic) is used.", "",
          "| Origin | Training deals | Adverse breaks | Folds | CV AUC p_mkt | CV AUC logistic | CV AUC LightGBM | Passed to the expected-return step |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    used = []
    for k, info in out["origins"].items():
        key, view, Y = k.split(" | ")
        if key != "cash":
            continue
        c = info["cv_auc"]
        what = (f"day-1 model ({'logistic' if info['best_day1'] == 'logit' else 'LightGBM'})" if info["use_day1"] else "market-implied probability")
        used.append((view, Y, info["use_day1"]))
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        L.append(f"| {'test period (2001-2018)' if Y == 'split' else 'year ' + Y} | {info['n_train']} | {info['pos_train']} | {info['folds']} | {f(c['p_mkt'])} | {f(c['logit'])} | {f(c['lgbm'])} | {what} |")
    np_ = sum(u for v, _, u in used if v == "pooled"); nt = [u for v, _, u in used if v == "test"]
    out["used_in_M"] = {"pooled_origins_using_day1": int(np_), "pooled_origins": int(sum(1 for v, _, _ in used if v == "pooled")), "test_uses_day1": bool(nt and nt[0])}
    L += ["", f"**Used in the expected-return step:** the day-1 model in {np_} of {out['used_in_M']['pooled_origins']} pooled origins and "
          f"{'the day-1 model' if out['used_in_M']['test_uses_day1'] else 'the market-implied probability'} for the test period.", ""]
    info = out["origins"].get("cash | test | split")
    if info:
        co = sorted(((k, v) for k, v in info["coef"].items()), key=lambda kv: -abs(kv[1]))
        L += ["## Logistic model fitted on 2001-2018 (standardised coefficients, all-cash deals)", "", "| Input | Coefficient |", "|---|---:|"]
        L += [f"| {NICE.get(k, NICE.get(k.replace('_missing', ''), k) + ' is missing' if k.endswith('_missing') else k)} | {v:+.2f} |" for k, v in co]
        L += ["", "Positive = more adverse-break risk. Coefficients are shrunk (C = 0.1) and describe the training period only.", ""]
    (res / "day1_signals.md").write_text("\n".join(L) + "\n")
    (res / "day1_signals.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    a = ap.parse_args()
    {"build": build, "run": run}[a.cmd]()

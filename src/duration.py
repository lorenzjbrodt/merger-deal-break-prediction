"""Time to closing: how many calendar days from announcement to closing, for deals that complete.

Target: log(days from announcement to closing) of completed core-sample deals. The closing date is a label only.
Ladder: median; median by structure (tender offer versus merger); ridge regression on base terms; LightGBM on base
terms; plus extended features; plus market context; plus LLM-extracted terms; plus text embeddings; plus the parties'
own stated expected closing as a feature. Benchmark: that stated expected closing by itself ("management guidance").
Intervals: LightGBM quantile models (10 / 50 / 90). Robustness: a log-normal survival model in which broken deals are
censored at their cancellation date.

Evaluation: the time split (train 2001-2018, test 2019-2025) and pooled rolling-origin predictions for 2013-2025
(each year predicted by models fitted on earlier years, hyper-parameters re-selected inside each origin).

  python src/duration.py fit      data/interim/duration.pkl (predictions for every core deal, git-ignored); resumable:
                                  one checkpoint per origin in data/interim/duration_ckpt/ (delete the folder to refit)
  python src/duration.py product  product forecast (model 6 where the closing is stated and CV prefers it) and the conformal 80% interval
  python src/duration.py report   results/duration.md, results/duration.json, figures/duration.png
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import optimize, stats
from sklearn.linear_model import Ridge

import modeling as M
import train_eval as TE
from modeling import Feat

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
CKPT = INTERIM / "duration_ckpt"
YEARS = list(range(2013, 2026))
REG = dict(learning_rate=0.03, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, n_jobs=M.N_JOBS)
LEAVES, TREES = (6, 12, 24), (200, 400, 800)
GRID = [(l, t) for l in LEAVES for t in TREES]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
N_BOOT = 2000
LABELS = {
    "median": "0. Median of the training period", "median_structure": "1. Median by structure (tender offer / merger)",
    "ridge_base": "2. Ridge regression, base terms", "lgbm_base": "3. LightGBM, base terms", "lgbm_ext": "3x. LightGBM + extended features",
    "lgbm_extm": "3m. 3x + market context", "lgbm_llm": "4. 3m + LLM-extracted terms", "lgbm_emb": "5. 3m + LLM terms + text embeddings",
    "lgbm_guide": "6. 3m + LLM terms + stated expected closing", "guidance": "Benchmark: the parties' stated expected closing",
    "final": "Final model (chosen by CV inside the training years)", "aft_base": "R. Log-normal survival model, base terms, broken deals censored",
}
ORDER = ["median", "median_structure", "ridge_base", "lgbm_base", "lgbm_ext", "lgbm_extm", "lgbm_llm", "lgbm_emb", "lgbm_guide", "final", "aft_base"]


# ----------------------------------------------------------------------------- fitting helpers
def cv_reg(D, y, feat, folds):
    """Mean fold MAE on log days per (leaves, trees); returns best params, its CV MAE and out-of-fold predictions."""
    sc = {g: [] for g in GRID}; oof = {g: [] for g in GRID}; oi = []
    for tr, va in folds:
        if len(tr) < 200 or len(va) < 30:
            continue
        A, B, _ = M.design_tree(D, feat, tr, va)
        oi.append(va)
        for l in LEAVES:
            m = LGBMRegressor(objective="regression", num_leaves=l, n_estimators=max(TREES), random_state=0, **REG).fit(A, y[tr])
            for t in TREES:
                p = m.predict(B, num_iteration=t)
                sc[(l, t)].append(np.mean(np.abs(p - y[va]))); oof[(l, t)].append(p)
    mean = {k: float(np.mean(v)) for k, v in sc.items() if v}
    best = min(mean, key=mean.get)
    return best, mean[best], np.concatenate(oi), np.concatenate(oof[best])


def fit_reg(D, y, feat, tr, te, best, quantiles=()):
    A, B, names = M.design_tree(D, feat, tr, te)
    p = np.mean([LGBMRegressor(objective="regression", num_leaves=best[0], n_estimators=best[1], random_state=s, **REG).fit(A, y[tr]).predict(B) for s in M.SEEDS], axis=0)
    q = {a: LGBMRegressor(objective="quantile", alpha=a, num_leaves=best[0], n_estimators=best[1], random_state=0, **REG).fit(A, y[tr]).predict(B) for a in quantiles}
    return p, q


def aft_lognormal(X, t, event, l2=1.0):
    """Log-normal accelerated-failure-time model with right-censoring, by penalised maximum likelihood."""
    n, k = X.shape

    def nll(w):
        b, ls = w[:k + 1], w[-1]
        s = np.exp(ls)
        z = (t - b[0] - X @ b[1:]) / s
        ll = np.where(event, stats.norm.logpdf(z) - ls, stats.norm.logsf(z))
        lam = np.where(event, z, np.exp(stats.norm.logpdf(z) - stats.norm.logsf(z)))
        g_mu = -lam / s
        g = np.r_[g_mu.sum(), X.T @ g_mu + l2 * b[1:], np.sum(np.where(event, 1 - z * z, -lam * z))]
        return -ll.sum() + 0.5 * l2 * np.sum(b[1:] ** 2), g
    w0 = np.r_[np.mean(t[event]), np.zeros(k), np.log(np.std(t[event]))]
    r = optimize.minimize(nll, w0, jac=True, method="L-BFGS-B")
    return r.x[:k + 1], float(np.exp(r.x[-1]))


def ladder(D, y, closed, tr_all, te, subset, log, cens=None):
    """Fit the ladder on completed deals in `tr_all`, predict log days for every deal in `te`."""
    tr = tr_all[closed[tr_all]]
    folds = M.cv_folds(D.year, tr)
    P, I, O = {}, {}, {}
    P["median"] = np.full(len(te), np.median(y[tr]))
    tend = D.X["feat_tender_offer"].fillna(0).to_numpy() == 1 if "feat_tender_offer" in D.X else np.zeros(len(y), bool)
    med = {k: np.median(y[tr][tend[tr] == k]) if (tend[tr] == k).sum() >= 20 else np.median(y[tr]) for k in (True, False)}
    P["median_structure"] = np.where(tend[te], med[True], med[False])
    # ridge on base terms
    best = None
    for a in ALPHAS:
        errs = []
        for f_tr, f_va in folds:
            A, B, _ = M.design_logit(D, Feat(base_only=True), f_tr, f_va)
            errs.append(np.mean(np.abs(Ridge(alpha=a).fit(A, y[f_tr]).predict(B) - y[f_va])))
        if best is None or np.mean(errs) < best[1]:
            best = (a, float(np.mean(errs)))
    A, B, _ = M.design_logit(D, Feat(base_only=True), tr, te)
    P["ridge_base"] = Ridge(alpha=best[0]).fit(A, y[tr]).predict(B); I["ridge_base"] = {"alpha": best[0], "cv_mae_log": best[1]}
    S = {"lgbm_base": Feat(base_only=True), "lgbm_ext": Feat(groups=TE.EXT), "lgbm_extm": Feat(groups=TE.EXTM)}
    if subset == "text":
        S["lgbm_llm"] = Feat(groups=TE.EXTM, llm=True)
        S["lgbm_emb"] = Feat(groups=TE.EXTM, llm=True, emb=True)
    else:
        S["lgbm_llm"] = Feat(groups=TE.EXTM, has_text_flag=True, llm=True)        # usable for every deal: missing where there is no text
    feats = {}
    for name, ft in S.items():
        bp, s, oi, op = cv_reg(D, y, ft, folds)
        P[name], _ = fit_reg(D, y, ft, tr, te, bp)
        I[name] = {"num_leaves": bp[0], "n_estimators": bp[1], "cv_mae_log": s}; O[name] = (oi, op); feats[name] = (ft, bp)
        log(f"    {name}: {bp} cv MAE(log) {s:.3f}")
    if subset == "text" and D.cache.get("guide") is not None:
        L0, n0 = D.L, D.L_names
        D.L, D.L_names = np.column_stack([L0, D.cache["guide"]]), n0 + ["guide_days", "guide_lo_days", "guide_hi_days"]
        ft = Feat(groups=TE.EXTM, llm=True)
        bp, s, oi, op = cv_reg(D, y, ft, folds)
        P["lgbm_guide"], _ = fit_reg(D, y, ft, tr, te, bp)
        I["lgbm_guide"] = {"num_leaves": bp[0], "n_estimators": bp[1], "cv_mae_log": s}
        D.L, D.L_names = L0, n0
    if subset == "full":
        cands = [m for m in ("lgbm_base", "lgbm_ext", "lgbm_extm", "lgbm_llm")]
        ch = min(cands, key=lambda m: I[m]["cv_mae_log"])
        ft, bp = feats[ch]
        p, q = fit_reg(D, y, ft, tr, te, bp, quantiles=(0.1, 0.5, 0.9))
        P["final"] = p; I["final"] = {"chosen": ch}
        for a, v in q.items():
            P[f"final_q{int(a * 100)}"] = v
        oi, op = O[ch]
        I["final"]["oof_resid_sd"] = float(np.std(y[oi] - op))
        if cens is not None:                                     # survival robustness: broken deals censored at cancellation
            trc = tr_all[np.isfinite(cens[tr_all])]
            A, B, _ = M.design_logit(D, Feat(base_only=True), trc, te)
            A, B = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A), np.asarray(B.todense()) if hasattr(B, "todense") else np.asarray(B)
            b, s = aft_lognormal(A, cens[trc], closed[trc], l2=best[0] / 10)
            P["aft_base"] = b[0] + B @ b[1:]; I["aft_base"] = {"sigma": s, "censored": int((~closed[trc]).sum())}
    return P, I


def fit(suffix: str, reverse: bool = False) -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    D = M.load_data(suffix)
    n = len(D.year)
    idx = np.arange(n)
    closed = D.y["label"] == 0
    days = np.clip(D.days_out.astype(float), 1, None)
    y = np.log(days)
    cens = np.where(np.isfinite(days), y, np.nan)                 # log days to closing, or to cancellation for broken deals
    gp = INTERIM / f"closing_guidance{suffix}.parquet"
    guide = None
    if gp.exists():
        G = D.deals[["deal_id"]].merge(pd.read_parquet(gp), on="deal_id", how="left")
        guide = G[["guide_days", "guide_lo_days", "guide_hi_days"]].to_numpy(float)
        D.cache["guide"] = guide
    cols = {}

    def put(name, te, p):
        cols.setdefault(name, np.full(n, np.nan))[te] = p
    sel = {}
    for tag, origins in (("split", [None]), ("oot", YEARS[::-1] if reverse else YEARS)):   # --reverse: a second worker from the other end
        for Y in origins:
            for subset in ("full", "text"):
                sm = D.core & (D.has_text if subset == "text" else True)
                tr = idx[sm & (D.year <= (M.TRAIN_END if Y is None else Y - 1))]
                te = idx[sm & ((D.year >= M.TEST_START) if Y is None else (D.year == Y))]
                if len(te) == 0 or closed[tr].sum() < 300:
                    continue
                ck = CKPT / f"{tag}_{Y or 'split'}_{subset}{suffix}.pkl"      # one checkpoint per origin: a stopped run resumes here
                if ck.exists():
                    P, I = pickle.load(open(ck, "rb"))
                    log(f"{tag} {Y or ''} / {subset}: from checkpoint")
                else:
                    log(f"{tag} {Y or ''} / {subset}: train {int(closed[tr].sum())} completed, test {len(te)} deals ({int(closed[te].sum())} completed)")
                    P, I = ladder(D, y, closed, tr, te, subset, log, cens=cens)
                    CKPT.mkdir(parents=True, exist_ok=True)
                    tmp = ck.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        pickle.dump((P, I), fh)
                    tmp.rename(ck)
                for m, p in P.items():
                    put(f"{tag}__{subset}__{m}", te, p)
                sel[(tag, Y, subset)] = I
    out = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "year": D.year, "core": D.core, "has_text": D.has_text, "closed": closed, "days": days,
                        "tender": (D.X["feat_tender_offer"].fillna(0).to_numpy() == 1) if "feat_tender_offer" in D.X else False,
                        "cross_border": (D.X["feat_cross_border"].fillna(0).to_numpy() == 1) if "feat_cross_border" in D.X else False,
                        "deal_value": D.deals["deal_value"].to_numpy(float), "sic": pd.to_numeric(D.deals["sic"], errors="coerce").to_numpy() if "sic" in D.deals else np.nan})
    if guide is not None:
        out["guide_days"] = guide[:, 0]
    out = pd.concat([out, pd.DataFrame(cols)], axis=1)
    with open(INTERIM / f"duration{suffix}.pkl", "wb") as fh:
        pickle.dump({"pred": out, "selection": sel}, fh)
    log("wrote duration.pkl")


# ----------------------------------------------------------------------------- product model and calibrated interval
ALPHA = 0.20          # 80% interval
CALIB_YEARS = 3       # the last three training years are held out for the conformal calibration


def conformal_q(scores):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))


def product(suffix: str) -> None:
    """Time to closing as used in the product, per origin and on training-period data only.

    Point forecast: model 6 (also reads the stated expected closing) for deals that state one, if the CV inside the
    origin's training years prefers model 6 over model 4 on the deals with text; the final model otherwise.
    80% interval: split conformal calibration of the quantile models (conformalized quantile regression). The quantile
    models are fitted on the training years except the last three; the last three are the calibration set; the
    interval is [q10 - Q, q90 + Q] with Q the conformal quantile of the calibration scores max(q10 - y, y - q90).
    """
    D = M.load_data(suffix)
    n = len(D.year)
    idx = np.arange(n)
    closed = D.y["label"] == 0
    y = np.log(np.clip(D.days_out.astype(float), 1, None))
    Z = pickle.load(open(INTERIM / f"duration{suffix}.pkl", "rb"))
    P, sel = Z["pred"], Z["selection"]
    assert (P["deal_id"].to_numpy() == D.deals["deal_id"].to_numpy()).all()
    G = D.deals[["deal_id"]].merge(pd.read_parquet(INTERIM / f"closing_guidance{suffix}.parquet"), on="deal_id", how="left")
    guide = G[["guide_days", "guide_lo_days", "guide_hi_days"]].to_numpy(float)
    has_g = np.isfinite(guide[:, 0]) & D.has_text
    base = {"lgbm_base": Feat(base_only=True), "lgbm_ext": Feat(groups=TE.EXT), "lgbm_extm": Feat(groups=TE.EXTM), "lgbm_llm": Feat(groups=TE.EXTM, has_text_flag=True, llm=True)}
    cols = {f"{t}__{k}": np.full(n, np.nan) for t in ("split", "oot") for k in ("mu", "sigma", "lo", "hi", "lo_final", "hi_final", "use_guide")}
    info = {}
    L0, n0 = D.L, D.L_names
    for (tag, Y, subset), I in sel.items():
        if subset != "full" or "final" not in I:
            continue
        cut = M.TRAIN_END if Y is None else Y - 1
        te = idx[D.core & ((D.year >= M.TEST_START) if Y is None else (D.year == Y))]
        tr = idx[D.core & closed & (D.year <= cut)]
        proper, calib = tr[D.year[tr] <= cut - CALIB_YEARS], tr[D.year[tr] > cut - CALIB_YEARS]
        ch = I["final"]["chosen"]
        _, q = fit_reg(D, y, base[ch], proper, np.r_[calib, te], (I[ch]["num_leaves"], I[ch]["n_estimators"]), quantiles=(0.1, 0.9))
        k = len(calib)
        Q = conformal_q(np.maximum(q[0.1][:k] - y[calib], y[calib] - q[0.9][:k]))
        cols[f"{tag}__lo_final"][te], cols[f"{tag}__hi_final"][te] = q[0.1][k:] - Q, q[0.9][k:] + Q
        cols[f"{tag}__mu"][te] = P[f"{tag}__full__final"].to_numpy()[te]
        cols[f"{tag}__sigma"][te] = I["final"]["oof_resid_sd"]
        cols[f"{tag}__lo"][te], cols[f"{tag}__hi"][te] = cols[f"{tag}__lo_final"][te], cols[f"{tag}__hi_final"][te]
        cols[f"{tag}__use_guide"][te] = 0.0
        It = sel.get((tag, Y, "text"), {})
        o = {"final": ch, "calibration_deals": int(k), "Q_final": Q, "cv_model4": It.get("lgbm_llm", {}).get("cv_mae_log"), "cv_model6": It.get("lgbm_guide", {}).get("cv_mae_log")}
        o["use_model6"] = bool(o["cv_model6"] is not None and o["cv_model4"] is not None and o["cv_model6"] < o["cv_model4"])
        if o["use_model6"]:
            D.L, D.L_names = np.column_stack([L0, guide]), n0 + ["guide_days", "guide_lo_days", "guide_hi_days"]
            f6 = Feat(groups=TE.EXTM, llm=True)
            pr6, ca6, te6 = proper[D.has_text[proper]], calib[has_g[calib]], te[has_g[te]]
            p6, q6 = fit_reg(D, y, f6, pr6, np.r_[ca6, te6], (It["lgbm_guide"]["num_leaves"], It["lgbm_guide"]["n_estimators"]), quantiles=(0.1, 0.9))
            D.L, D.L_names = L0, n0
            k6 = len(ca6)
            Q6 = conformal_q(np.maximum(q6[0.1][:k6] - y[ca6], y[ca6] - q6[0.9][:k6]))
            cols[f"{tag}__mu"][te6] = P[f"{tag}__text__lgbm_guide"].to_numpy()[te6]
            cols[f"{tag}__sigma"][te6] = float(np.std(y[ca6] - p6[:k6]))
            cols[f"{tag}__lo"][te6], cols[f"{tag}__hi"][te6] = q6[0.1][k6:] - Q6, q6[0.9][k6:] + Q6
            cols[f"{tag}__use_guide"][te6] = 1.0
            o.update(calibration_deals_model6=int(k6), Q_model6=Q6, sigma_model6=float(np.std(y[ca6] - p6[:k6])), test_deals_with_guidance=int(len(te6)))
        info[f"{tag} {Y or 'split'}"] = o
        print(f"{tag} {Y or 'split'}: model 6 {'used' if o['use_model6'] else 'not used'} (CV {o['cv_model6']} vs {o['cv_model4']}), Q {Q:.3f}", flush=True)
    out = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), **cols})
    out.to_parquet(INTERIM / f"duration_product{suffix}.parquet", index=False)
    (INTERIM / f"duration_product{suffix}.json").write_text(json.dumps(info, indent=2, default=float))


# ----------------------------------------------------------------------------- report
def met(days, pred_log):
    p = np.exp(pred_log)
    ae = np.abs(p - days)
    ly = np.log(days)
    return [float(ae.mean()), float(np.median(ae)), float(1 - np.sum((ly - pred_log) ** 2) / np.sum((ly - ly.mean()) ** 2)), float((ae <= 30).mean())]


KEYS = ["mae", "medae", "r2_log", "within30"]


def evaluate(days, yr, S, pairs, by_year, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [np.where(yr == v)[0] for v in np.unique(yr)]
    pt = {k: np.array(met(days, s)) for k, s in S.items()}
    acc = {k: [] for k in S}
    for _ in range(N_BOOT):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) if by_year else rng.integers(0, len(days), len(days))
        for k, s in S.items():
            acc[k].append(met(days[ix], s[ix]))
    arr = {k: np.array(v) for k, v in acc.items()}
    out = {"scores": {k: {**{m: float(pt[k][j]) for j, m in enumerate(KEYS)},
                          "ci95": {m: [float(np.percentile(arr[k][:, j], 2.5)), float(np.percentile(arr[k][:, j], 97.5))] for j, m in enumerate(KEYS)}} for k in S}, "pairs": {}}
    for a, b in pairs:
        if a in S and b in S:
            d = arr[a] - arr[b]
            out["pairs"][f"{a} minus {b}"] = {**{m: float(pt[a][j] - pt[b][j]) for j, m in enumerate(KEYS)},
                                              "ci95": {m: [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))] for j, m in enumerate(KEYS)}}
    return out


def table(ev, order):
    L = ["| Model | MAE, days [95% CI] | Median abs. error, days | R2 on log days [95% CI] | Within 30 days |", "|---|---|---:|---|---:|"]
    for m in order:
        if m in ev["scores"]:
            r = ev["scores"][m]
            L.append(f"| {LABELS[m]} | {r['mae']:.1f} [{r['ci95']['mae'][0]:.1f}, {r['ci95']['mae'][1]:.1f}] | {r['medae']:.1f} | "
                     f"{r['r2_log']:.3f} [{r['ci95']['r2_log'][0]:.3f}, {r['ci95']['r2_log'][1]:.3f}] | {r['within30']:.0%} |")
    return L + [""]


def pair_lines(ev, names):
    L = []
    for k, lab in names:
        if k in ev["pairs"]:
            r = ev["pairs"][k]
            L.append(f"- {lab}: MAE {r['mae']:+.1f} days [{r['ci95']['mae'][0]:+.1f}, {r['ci95']['mae'][1]:+.1f}]; "
                     f"R2 on log days {r['r2_log']:+.3f} [{r['ci95']['r2_log'][0]:+.3f}, {r['ci95']['r2_log'][1]:+.3f}].")
    return L + [""]


def report(suffix: str) -> None:
    synthetic = suffix != ""
    res = ROOT / ("results" if not synthetic else f"results{suffix}"); fig = ROOT / ("figures" if not synthetic else f"figures{suffix}")
    res.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)
    Z = pickle.load(open(INTERIM / f"duration{suffix}.pkl", "rb"))
    P, sel = Z["pred"], Z["selection"]
    out = {}
    done = P["core"] & P["closed"]
    trn = done & (P["year"] <= 2018)
    L = ["# Time to closing" + (" (SYNTHETIC SAMPLE: not real results)" if synthetic else ""), "",
         "Target: calendar days from announcement to closing, completed core-sample deals. Models predict log days; the prediction in days is the exponential "
         "(a median forecast). Training period 2001-2018: "
         f"{int(trn.sum()):,} completed deals, median {P.loc[trn, 'days'].median():.0f} days, 10th to 90th percentile {P.loc[trn, 'days'].quantile(0.1):.0f} to {P.loc[trn, 'days'].quantile(0.9):.0f}. "
         "Intervals: bootstrap over deals for the time split, over announcement years for the pooled set.", ""]
    names_full = [("lgbm_base minus ridge_base", "LightGBM minus ridge, base terms"), ("lgbm_ext minus lgbm_base", "Extended features (3x minus 3)"),
                  ("lgbm_extm minus lgbm_ext", "Market context (3m minus 3x)"), ("lgbm_llm minus lgbm_extm", "LLM-extracted terms (4 minus 3m)"),
                  ("final minus median_structure", "Final model minus median by structure"), ("aft_base minus ridge_base", "Survival model minus ridge (effect of censoring)")]
    names_text = [("lgbm_llm minus lgbm_extm", "**LLM-extracted terms** (4 minus 3m)"), ("lgbm_emb minus lgbm_llm", "**Text embeddings** (5 minus 4)"),
                  ("lgbm_emb minus lgbm_extm", "All text (5 minus 3m)"), ("lgbm_guide minus lgbm_llm", "Stated expected closing as a feature (6 minus 4)")]
    for tag, title, by_year in (("split", "Time split: test period 2019-2025", False), ("oot", "Pooled rolling origin, 2013-2025", True)):
        for subset, sub_title, names in (("full", "all core deals", names_full), ("text", "deals with announcement text", names_text)):
            cols = {m: f"{tag}__{subset}__{m}" for m in ORDER if f"{tag}__{subset}__{m}" in P}
            if not cols:
                continue
            ok = (done & P[list(cols.values())].notna().all(axis=1)).to_numpy()
            if ok.sum() < 30:
                continue
            days, yr = P["days"].to_numpy()[ok], P["year"].to_numpy()[ok]
            S = {m: P[c].to_numpy()[ok] for m, c in cols.items()}
            ev = evaluate(days, yr, S, [tuple(k.split(" minus ")) for k, _ in names], by_year)
            ev["n"] = int(ok.sum())
            out[f"{tag}_{subset}"] = ev
            L += [f"## {title}, {sub_title}: {ev['n']:,} completed deals", ""] + table(ev, ORDER) + ["Paired differences (negative MAE = better):", ""] + pair_lines(ev, names)
    # ---- interval coverage
    for tag, title in (("split", "test period"), ("oot", "pooled 2013-2025")):
        c = [f"{tag}__full__final_q10", f"{tag}__full__final_q50", f"{tag}__full__final_q90"]
        if all(x in P for x in c):
            ok = (done & P[c].notna().all(axis=1)).to_numpy()
            ly = np.log(P["days"].to_numpy()[ok]); lo, hi = P[c[0]].to_numpy()[ok], P[c[2]].to_numpy()[ok]
            cov = float(((ly >= lo) & (ly <= hi)).mean()); wid = float(np.median(np.exp(hi) - np.exp(lo)))
            out[f"coverage_{tag}"] = {"coverage80": cov, "median_width_days": wid, "below": float((ly < lo).mean()), "above": float((ly > hi).mean())}
            L += [f"- 80% interval from the quantile models, {title}: covers {cov:.0%} of completed deals ({(ly < lo).mean():.0%} closed faster, {(ly > hi).mean():.0%} slower); median width {wid:.0f} days."]
    L.append("")
    # ---- management guidance
    if "guide_days" in P:
        for tag, title, by_year in (("split", "test period 2019-2025", False), ("oot", "pooled 2013-2025", True)):
            need = [c for c in (f"{tag}__text__lgbm_llm", f"{tag}__text__lgbm_guide", f"{tag}__text__lgbm_extm", f"{tag}__text__median") if c in P]
            ok = (done & P["guide_days"].notna() & P[need].notna().all(axis=1)).to_numpy()
            if ok.sum() < 30:
                continue
            days, yr = P["days"].to_numpy()[ok], P["year"].to_numpy()[ok]
            S = {"guidance": np.log(P["guide_days"].to_numpy()[ok])}
            S.update({c.split("__")[-1]: P[c].to_numpy()[ok] for c in need})
            ev = evaluate(days, yr, S, [("lgbm_llm", "guidance"), ("lgbm_guide", "guidance"), ("lgbm_extm", "guidance")], by_year)
            ev["n"] = int(ok.sum()); ev["share_with_guidance"] = float(ok.sum() / (done & P[need].notna().all(axis=1)).sum())
            g = P["guide_days"].to_numpy()[ok]
            ev["bias"] = {"median_actual_minus_guidance": float(np.median(days - g)), "share_later_than_guidance": float((days > g).mean())}
            out[f"guidance_{tag}"] = ev
            L += [f"## Does the model beat management guidance? ({title}, {ev['n']:,} completed deals with stated guidance, {ev['share_with_guidance']:.0%} of deals with text)", "",
                  "Guidance = the expected closing stated in the announcement, converted to days (midpoint of the stated window; see `results/closing_guidance_validation.md`).", ""]
            L += table(ev, ["median", "guidance", "lgbm_extm", "lgbm_llm", "lgbm_guide"])
            L += pair_lines(ev, [("lgbm_extm minus guidance", "Model without any text (3m) minus guidance"), ("lgbm_llm minus guidance", "Model with LLM terms (4) minus guidance"),
                                 ("lgbm_guide minus guidance", "Model that also sees the guidance (6) minus guidance")])
            L += [f"Deals closed a median of {ev['bias']['median_actual_minus_guidance']:+.0f} days relative to guidance; {ev['bias']['share_later_than_guidance']:.0%} closed later than the stated midpoint.", ""]
    # ---- segments (test period, final model vs median by structure)
    c_f, c_m = "split__full__final", "split__full__median_structure"
    if c_f in P:
        ok = (done & P[c_f].notna()).to_numpy()
        X = P[ok]
        sic = X["sic"].fillna(0).astype(int) // 100
        reg = np.where(sic.isin([60, 61]), "banks", np.where(sic == 63, "insurance", np.where(sic == 49, "utilities", np.where(sic == 48, "telecom and media", "other industries"))))
        size = pd.qcut(X["deal_value"], 3, labels=["small (bottom third by value)", "mid", "large (top third)"]).astype(str)
        seg = {"Structure": np.where(X["tender"], "tender offer", "merger"), "Size": size.to_numpy(), "Cross-border": np.where(X["cross_border"], "cross-border", "domestic"), "Regulated industry": reg}
        L += ["## Segments (test period, final model against the median by structure)", "", "| Segment | Deals | Median days | MAE, median by structure | MAE, final model | Within 30 days, final model |", "|---|---:|---:|---:|---:|---:|"]
        segs = {}
        for g, lab in seg.items():
            for v in pd.unique(lab):
                m = lab == v
                if m.sum() < 15:
                    continue
                a, b = met(X["days"].to_numpy()[m], X[c_m].to_numpy()[m]), met(X["days"].to_numpy()[m], X[c_f].to_numpy()[m])
                segs[f"{g}: {v}"] = {"n": int(m.sum()), "median_days": float(np.median(X["days"].to_numpy()[m])), "mae_benchmark": a[0], "mae_final": b[0], "within30": b[3]}
                L.append(f"| {g}: {v} | {int(m.sum())} | {np.median(X['days'].to_numpy()[m]):.0f} | {a[0]:.1f} | {b[0]:.1f} | {b[3]:.0%} |")
        out["segments"] = segs
        L.append("")
    # ---- product model and calibrated interval
    pp = INTERIM / f"duration_product{suffix}.parquet"
    if pp.exists():
        DP = pd.read_parquet(pp).set_index("deal_id").reindex(P["deal_id"]).reset_index()
        pinfo = json.loads((INTERIM / f"duration_product{suffix}.json").read_text())
        used = [k for k, v in pinfo.items() if v["use_model6"]]
        L += ["## Product model: the forecast shown in the demo and used in the expected-return step", "",
              "Where the announcement states an expected closing, the product uses model 6 (which also reads it); otherwise the final model. The switch is decided per origin "
              "by the CV inside its training years (model 6 against model 4 on the deals with text). "
              f"CV preferred model 6 in {len(used)} of {len(pinfo)} origins.", "",
              "| Evaluation | Completed deals | With stated closing | MAE days, final model | MAE days, product model | Difference [95% CI] | R2 on log days: difference [95% CI] |", "|---|---:|---:|---:|---:|---|---|"]
        out["product"] = {"origins": pinfo}
        for tag, title, by_year in (("split", "Test period 2019-2025", False), ("oot", "Pooled rolling origin 2013-2025", True)):
            ok = (done & DP[f"{tag}__mu"].notna() & P[f"{tag}__full__final"].notna()).to_numpy()
            e = evaluate(P["days"].to_numpy()[ok], P["year"].to_numpy()[ok], {"final": P[f"{tag}__full__final"].to_numpy()[ok], "product": DP[f"{tag}__mu"].to_numpy()[ok]}, [("product", "final")], by_year)
            e["n"] = int(ok.sum()); e["with_guidance"] = int((DP[f"{tag}__use_guide"].to_numpy()[ok] == 1).sum())
            out["product"][tag] = e
            r = e["pairs"]["product minus final"]
            L.append(f"| {title} | {e['n']} | {e['with_guidance']} | {e['scores']['final']['mae']:.1f} | {e['scores']['product']['mae']:.1f} | {r['mae']:+.1f} [{r['ci95']['mae'][0]:+.1f}, {r['ci95']['mae'][1]:+.1f}] | "
                     f"{r['r2_log']:+.3f} [{r['ci95']['r2_log'][0]:+.3f}, {r['ci95']['r2_log'][1]:+.3f}] |")
        L += ["", "## Calibrated 80% interval (split conformal)", "",
              "The quantile models are refitted on the training years except the last three; those three years are the calibration set. The interval is widened (or narrowed) on "
              "both sides by the conformal quantile of the calibration errors. Nothing from the evaluated years is used.", "",
              "| Evaluation | Interval | Coverage of completed deals | Closed faster | Closed slower | Median width, days |", "|---|---|---:|---:|---:|---:|"]
        out["conformal"] = {}
        for tag, title in (("split", "Test period 2019-2025"), ("oot", "Pooled rolling origin 2013-2025")):
            ly = np.log(P["days"].to_numpy())
            for lab, lo, hi in (("quantile models, uncalibrated", P[f"{tag}__full__final_q10"].to_numpy(), P[f"{tag}__full__final_q90"].to_numpy()),
                                ("final model, conformal", DP[f"{tag}__lo_final"].to_numpy(), DP[f"{tag}__hi_final"].to_numpy()),
                                ("product model, conformal (shown in the demo)", DP[f"{tag}__lo"].to_numpy(), DP[f"{tag}__hi"].to_numpy())):
                ok = done.to_numpy() & np.isfinite(lo) & np.isfinite(hi)
                cov = float(((ly[ok] >= lo[ok]) & (ly[ok] <= hi[ok])).mean())
                out["conformal"][f"{tag} | {lab}"] = {"n": int(ok.sum()), "coverage": cov, "below": float((ly[ok] < lo[ok]).mean()), "above": float((ly[ok] > hi[ok]).mean()),
                                                      "median_width_days": float(np.median(np.exp(hi[ok]) - np.exp(lo[ok])))}
                L.append(f"| {title} | {lab} | {cov:.0%} | {(ly[ok] < lo[ok]).mean():.0%} | {(ly[ok] > hi[ok]).mean():.0%} | {np.median(np.exp(hi[ok]) - np.exp(lo[ok])):.0f} |")
        L.append("")
    ch = [v["final"]["chosen"] for k, v in sel.items() if k[0] == "oot" and k[2] == "full" and "final" in v]
    out["final_chosen"] = {str(k[1] or "split"): v["final"]["chosen"] for k, v in sel.items() if k[2] == "full" and "final" in v}
    L += ["## Notes", "",
          f"- Final model: the feature set with the lowest CV error inside the training years. Time split: {LABELS[sel[('split', None, 'full')]['final']['chosen']].split('.')[0]}; "
          f"rolling origins: {', '.join(f'{LABELS[c].split(chr(46))[0]} x{ch.count(c)}' for c in dict.fromkeys(ch))}.",
          "- The survival model treats broken deals as censored at their cancellation date instead of dropping them. Dropping them is the main specification because the "
          "quantity an arbitrageur needs is time to closing given that the deal closes; the survival row shows the effect of that choice.",
          "- A model for year Y is trained on completed deals announced by Y-1, some of which closed during Y. Deals still open at the training cut-off are longer than average, "
          "so a live forecast would have slightly fewer long deals to learn from.", ""]
    (res / "duration.md").write_text("\n".join(L) + "\n")
    (res / "duration.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(L))
    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    if c_f in P:
        ok = (done & P[c_f].notna()).to_numpy()
        f_, ax = plt.subplots(1, 2, figsize=(11, 4))
        d, p = P["days"].to_numpy()[ok], np.exp(P[c_f].to_numpy()[ok])
        ax[0].scatter(p, d, s=8, alpha=0.35, color="#08519c", lw=0)
        ax[0].plot([10, 1000], [10, 1000], color="#9e9e9e", ls="--", lw=1)
        ax[0].set(xscale="log", yscale="log", xlabel="Predicted days to closing (final model)", ylabel="Actual days to closing", xlim=(15, 800), ylim=(15, 1200))
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        for axis in (ax[0].xaxis, ax[0].yaxis):
            axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
        ax[0].set_xticks([20, 50, 100, 200, 500]); ax[0].set_yticks([20, 50, 100, 200, 500, 1000])
        ax[0].set_title("Test period 2019-2025, completed deals", loc="left", fontsize=10)
        ev = out.get("split_full")
        if ev:
            ks = [m for m in ORDER if m in ev["scores"] and m not in ("aft_base", "final")]
            ypos = np.arange(len(ks))[::-1]
            v = [ev["scores"][k]["mae"] for k in ks]
            ax[1].errorbar(v, ypos, xerr=[[ev["scores"][k]["mae"] - ev["scores"][k]["ci95"]["mae"][0] for k in ks], [ev["scores"][k]["ci95"]["mae"][1] - ev["scores"][k]["mae"] for k in ks]],
                           fmt="o", color="#08519c", capsize=3)
            ax[1].set_yticks(ypos); ax[1].set_yticklabels([LABELS[k].split(" (")[0] for k in ks], fontsize=8); ax[1].grid(axis="y", alpha=0)
            ax[1].set_xlabel("Mean absolute error, days (95% CI)"); ax[1].set_title("Time-to-closing ladder, test period", loc="left", fontsize=10)
        f_.tight_layout(); f_.savefig(fig / "duration.png"); plt.close(f_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "product", "report"])
    ap.add_argument("--tag", default="real")
    ap.add_argument("--reverse", action="store_true", help="fit the rolling origins from the last year backwards (second worker; shares the checkpoints)")
    a = ap.parse_args()
    sfx = "" if a.tag == "real" else f"_{a.tag}"
    fit(sfx, a.reverse) if a.cmd == "fit" else (product(sfx) if a.cmd == "product" else report(sfx))

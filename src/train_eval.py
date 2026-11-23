"""Train the model ladder, evaluate on the 2019-2025 test period, write tables and figures.

Split: train on announcements 2001-2018, test on 2019-2025. Hyper-parameters are chosen
with expanding-window, year-blocked CV inside the training years only. Every fitted
transform (imputer, scaler, one-hot, TF-IDF vocabulary, SVD, PCA) is fitted on the
training rows of the fold it is used in. The test set is touched once, for the final
numbers.

Usage:
    python src/train_eval.py                  # real data
    python src/train_eval.py --tag synthetic  # smoke test on the synthetic sample
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve

import features as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
TRAIN_END, TEST_START = 2018, 2019
N_BOOT = 2000
SEEDS = [0, 1, 2, 3, 4]
N_JOBS = 8

LGBM_BASE = dict(objective="binary", learning_rate=0.03, min_child_samples=25, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, n_jobs=N_JOBS)
LGBM_LEAVES = [3, 6, 12]
LGBM_TREES = [100, 200, 400]
LOGIT_C = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TFIDF_LOGIT_C = [0.1, 0.3, 1.0, 3.0, 10.0]
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"     # letters only: no dates / years / prices in the vocabulary


# ----------------------------------------------------------------------------- data
@dataclass
class Data:
    deals: pd.DataFrame
    X: pd.DataFrame
    K: np.ndarray | None
    press: np.ndarray | None
    agr: np.ndarray | None
    e_press: np.ndarray | None
    e_agr: np.ndarray | None
    has_text: np.ndarray
    y: np.ndarray
    year: np.ndarray


def load_data(suffix: str) -> Data:
    deals = pd.read_parquet(INTERIM / f"deals{suffix}.parquet").reset_index(drop=True)
    X = F.build_tabular(deals)
    assert not set(X.columns) & set(F.FORBIDDEN), "forbidden column leaked into features"
    n = len(deals)
    K = press = agr = e_press = e_agr = None
    has_text = np.zeros(n, dtype=bool)
    tp = INTERIM / f"text{suffix}.parquet"
    if tp.exists():
        text = deals[["deal_id"]].merge(pd.read_parquet(tp), on="deal_id", how="left")
        press = text["press_release"].fillna("").astype(str).to_numpy(dtype=object)
        agr = text["agreement"].fillna("").astype(str).to_numpy(dtype=object)
        has_text = text["has_text"].fillna(False).to_numpy(dtype=bool)
        tt = pd.DataFrame({"press_release": np.where(has_text, press, ""), "agreement": np.where(has_text, agr, "")})
        K = F.keyword_features(tt).to_numpy(dtype=float)
        ep = INTERIM / f"embeddings{suffix}.npz"
        if ep.exists():
            z = np.load(ep, allow_pickle=False)
            pos = pd.Series(np.arange(len(z["deal_id"])), index=z["deal_id"]).reindex(deals["deal_id"].astype(str))
            ok = pos.notna().to_numpy()
            e_press = np.full((n, z["press"].shape[1]), np.nan, dtype=np.float32)
            e_agr = np.full((n, z["agreement"].shape[1]), np.nan, dtype=np.float32)
            e_press[ok] = z["press"][pos[ok].astype(int)]
            e_agr[ok] = z["agreement"][pos[ok].astype(int)]
            e_press[~has_text] = np.nan
            e_agr[~has_text] = np.nan
    X["has_text"] = has_text.astype(int)
    return Data(deals, X, K, press, agr, e_press, e_agr, has_text,
                deals["label"].to_numpy(dtype=int), deals["year"].to_numpy(dtype=int))


def cv_folds(years: np.ndarray, idx: np.ndarray, n_blocks: int = 4, block: int = 3):
    """Expanding-window folds: validate on consecutive 3-year blocks at the end of the
    training period, train on all earlier years."""
    ys = np.sort(np.unique(years[idx]))
    folds = []
    for b in range(n_blocks, 0, -1):
        val_years = ys[len(ys) - b * block: len(ys) - (b - 1) * block]
        if len(val_years) == 0:
            continue
        tr = idx[years[idx] < val_years.min()]
        va = idx[np.isin(years[idx], val_years)]
        folds.append((tr, va))
    return folds


# ----------------------------------------------------------------------------- designs
@dataclass(frozen=True)
class Feat:
    tab: bool = True
    year: bool = False
    suspects: bool = False
    impute: bool = False        # median-impute and drop missingness flags (robustness)
    has_text_flag: bool = False
    kw: bool = False
    tfidf: bool = False
    emb: bool = False


def design_tree(D: Data, feat: Feat, tr: np.ndarray, te: np.ndarray):
    """Dense design matrices for LightGBM; everything fitted on `tr` only."""
    A, B, names = [], [], []
    if feat.tab:
        num = list(F.NUMERIC) + (F.YEAR if feat.year else []) + (F.SUSPECT_FEATURES if feat.suspects else [])
        if feat.impute:
            num = [c for c in num if c != "conditions_missing"]
        pre = F.tree_preprocessor(num, F.CATEGORICAL, impute=feat.impute)
        A.append(np.asarray(pre.fit_transform(D.X.iloc[tr]), dtype=float))
        B.append(np.asarray(pre.transform(D.X.iloc[te]), dtype=float))
        names += [str(n).replace("_infrequent_sklearn", "_other") for n in pre.get_feature_names_out()]
    if feat.has_text_flag:
        A.append(D.has_text[tr, None].astype(float))
        B.append(D.has_text[te, None].astype(float))
        names.append("has_text")
    if feat.kw:
        A.append(D.K[tr]); B.append(D.K[te]); names += F.KEYWORD_FEATURES
    if feat.tfidf:
        for field, docs, k, kw in (("press", D.press, 30, dict(ngram_range=(1, 2), min_df=5, max_features=30000)),
                                   ("agreement", D.agr, 10, dict(ngram_range=(1, 1), min_df=10, max_features=20000))):
            has = np.array([len(t) > 0 for t in docs]) & D.has_text
            a = np.full((len(tr), k), np.nan); b = np.full((len(te), k), np.nan)
            if has[tr].sum() > 5 * k:
                vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=TOKEN, max_df=0.9,
                                      sublinear_tf=True, dtype=np.float32, **kw)
                Ttr = vec.fit_transform([t[:200_000] for t in docs[tr][has[tr]]])
                kk = min(k, Ttr.shape[1] - 1)
                svd = TruncatedSVD(kk, random_state=0)
                a[has[tr], :kk] = svd.fit_transform(Ttr)
                if has[te].any():
                    b[has[te], :kk] = svd.transform(vec.transform([t[:200_000] for t in docs[te][has[te]]]))
            A.append(a); B.append(b); names += [f"tfidf_{field}_svd{i + 1}" for i in range(k)]
    if feat.emb:
        for field, E, k in (("press", D.e_press, 20), ("agreement", D.e_agr, 10)):
            ok = np.isfinite(E[:, 0])
            a = np.full((len(tr), k), np.nan); b = np.full((len(te), k), np.nan)
            if ok[tr].sum() > 5 * k:
                pca = PCA(k, random_state=0).fit(E[tr][ok[tr]])
                a[ok[tr]] = pca.transform(E[tr][ok[tr]])
                if ok[te].any():
                    b[ok[te]] = pca.transform(E[te][ok[te]])
            A.append(a); B.append(b); names += [f"emb_{field}_pc{i + 1}" for i in range(k)]
    return np.hstack(A), np.hstack(B), names


def design_logit(D: Data, feat: Feat, tr, te):
    num = list(F.NUMERIC) + (F.YEAR if feat.year else [])
    pre = F.logistic_preprocessor(num, F.CATEGORICAL)
    return pre.fit_transform(D.X.iloc[tr]), pre.transform(D.X.iloc[te]), [str(n) for n in pre.get_feature_names_out()]


def design_tfidf_sparse(D: Data, tr, te):
    vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=TOKEN, ngram_range=(1, 2),
                          min_df=5, max_df=0.9, max_features=30000, sublinear_tf=True)
    return vec.fit_transform(D.press[tr]), vec.transform(D.press[te]), list(vec.get_feature_names_out())


# ----------------------------------------------------------------------------- fitting
def auc_safe(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def cv_lgbm(D: Data, feat: Feat, folds):
    """Mean fold AUC for every (num_leaves, n_estimators); returns best params and OOF scores."""
    scores = {(l, t): [] for l in LGBM_LEAVES for t in LGBM_TREES}
    oof = {k: [] for k in scores}
    for tr, va in folds:
        if D.y[tr].sum() < 5 or D.y[va].sum() < 2:
            continue
        A, B, _ = design_tree(D, feat, tr, va)
        for l in LGBM_LEAVES:
            m = LGBMClassifier(**LGBM_BASE, num_leaves=l, n_estimators=max(LGBM_TREES), random_state=0).fit(A, D.y[tr])
            for t in LGBM_TREES:
                p = m.predict_proba(B, num_iteration=t)[:, 1]
                scores[(l, t)].append(auc_safe(D.y[va], p))
                oof[(l, t)].append(p)
    if not any(scores.values()):
        return (LGBM_LEAVES[0], LGBM_TREES[0]), np.nan, np.array([])
    mean = {k: float(np.nanmean(v)) for k, v in scores.items() if v}
    best = max(mean, key=mean.get)
    return best, mean[best], np.concatenate(oof[best])


def fit_lgbm(D: Data, feat: Feat, tr, te, best):
    A, B, names = design_tree(D, feat, tr, te)
    p = np.zeros(len(te)); gain = np.zeros(A.shape[1])
    for seed in SEEDS:
        m = LGBMClassifier(**LGBM_BASE, num_leaves=best[0], n_estimators=best[1], random_state=seed).fit(A, D.y[tr])
        p += m.predict_proba(B)[:, 1] / len(SEEDS)
        gain += m.booster_.feature_importance(importance_type="gain") / len(SEEDS)
    return p, dict(zip(names, gain.tolist()))


def cv_linear(D: Data, folds, grid, design):
    scores = {c: [] for c in grid}
    oof = {c: [] for c in grid}
    for tr, va in folds:
        if D.y[tr].sum() < 5 or D.y[va].sum() < 2:
            continue
        A, B, _ = design(tr, va)
        for c in grid:
            m = LogisticRegression(C=c, max_iter=3000).fit(A, D.y[tr])
            p = m.predict_proba(B)[:, 1]
            scores[c].append(auc_safe(D.y[va], p)); oof[c].append(p)
    if not any(scores.values()):
        return grid[0], np.nan, np.array([])
    mean = {k: float(np.nanmean(v)) for k, v in scores.items() if v}
    best = max(mean, key=mean.get)
    return best, mean[best], np.concatenate(oof[best])


ONE_RULES = {
    "stock-only consideration": lambda X: (X["consideration_type"] == "stock").to_numpy(),
    "any stock in consideration": lambda X: (X["stock_pct"].fillna(0) > 0).to_numpy(),
    "solicited approach": lambda X: (X["solicited"] == 1).to_numpy(),
    "buyer not listed": lambda X: (X["buyer_listed"] == 0).to_numpy(),
    "financial condition attached": lambda X: (X["cond_financial"] == 1).to_numpy(),
    "not yet signed at announcement": lambda X: (X["signed_after_announcement"] == 1).to_numpy(),
    "more than one buyer": lambda X: (X["buyer_count"] > 1).to_numpy(),
}


def one_rule(D: Data, tr, te, folds):
    """Pick the single binary rule with the best CV AUC; score = training break rate per group."""
    def score(rule, a, b):
        r_tr, r_te = rule(D.X.iloc[a]), rule(D.X.iloc[b])
        p1 = D.y[a][r_tr].mean() if r_tr.any() else D.y[a].mean()
        p0 = D.y[a][~r_tr].mean() if (~r_tr).any() else D.y[a].mean()
        return np.where(r_te, p1, p0)
    cv = {}
    for name, rule in ONE_RULES.items():
        aucs = [auc_safe(D.y[va], score(rule, a, va)) for a, va in folds if D.y[a].sum() >= 5 and D.y[va].sum() >= 2]
        cv[name] = float(np.nanmean(aucs)) if aucs else 0.5
    best = max(cv, key=cv.get)
    return score(ONE_RULES[best], tr, te), best, cv


# ----------------------------------------------------------------------------- metrics
def capture_at(y, s, frac):
    """Share of all breaks that sit in the riskiest `frac` of deals (ties split pro rata)."""
    n = len(y); k = max(int(round(frac * n)), 1)
    order = np.argsort(-s, kind="stable"); ss, ys = s[order], y[order]
    thr = ss[k - 1]
    above, tied = ss > thr, ss == thr
    got = ys[above].sum() + ys[tied].sum() * (k - above.sum()) / tied.sum()
    return float(got / max(ys.sum(), 1))


def metrics(y, s):
    return {"roc_auc": float(auc_safe(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "brier": float(brier_score_loss(y, s)),
            "capture_top10": capture_at(y, s, 0.10), "capture_top20": capture_at(y, s, 0.20)}


def bootstrap(y, preds: dict[str, np.ndarray], pairs: list[tuple[str, str]], n_boot=N_BOOT, seed=0):
    """Nonparametric bootstrap over test deals: 95% CIs per model and for paired differences."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: [] for m in preds}
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        yb = y[ix]
        if yb.sum() < 2:
            continue
        for m, p in preds.items():
            draws[m].append(list(metrics(yb, p[ix]).values()))
    keys = list(metrics(y, next(iter(preds.values()))).keys())
    arr = {m: np.array(v) for m, v in draws.items()}
    ci = {m: {k: [float(np.percentile(a[:, j], 2.5)), float(np.percentile(a[:, j], 97.5))] for j, k in enumerate(keys)}
          for m, a in arr.items()}
    diffs = {}
    for a, b in pairs:
        if a in arr and b in arr:
            d = arr[a] - arr[b]
            diffs[f"{a} minus {b}"] = {
                k: {"point": float(metrics(y, preds[a])[k] - metrics(y, preds[b])[k]),
                    "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))],
                    "share_boot_gt0": float((d[:, j] > 0).mean())}
                for j, k in enumerate(keys)}
    return ci, diffs


# ----------------------------------------------------------------------------- ladder
LABELS = {
    "base_rate": "0. Base rate",
    "one_rule": "1. One rule",
    "logit_tab": "2. Logistic, tabular",
    "lgbm_tab": "3. LightGBM, tabular",
    "lgbm_tab_kw": "3k. LightGBM, tabular + keywords",
    "lgbm_tab_tfidf": "4. LightGBM, tabular + keywords + TF-IDF",
    "lgbm_tab_emb": "5. LightGBM, tabular + keywords + embeddings",
    "tfidf_logit": "6a. Text only: TF-IDF logistic",
    "lgbm_text_only": "6b. Text only: keywords + embeddings LightGBM",
    "lgbm_tab_hastext": "3t. LightGBM, tabular + has_text flag",
    "lgbm_full_text": "5f. LightGBM, tabular + has_text + keywords + embeddings",
    "lgbm_tab_suspects": "A1. Ablation: tabular + Deal Responses/Attitude (leakage suspects)",
    "lgbm_tab_imputed": "A2. Ablation: tabular, median-imputed, no missingness signal",
}


def run_ladder(D: Data, tr, te, text_models: bool, full_extras: bool, log):
    """Fit the ladder on `tr`, predict `te`. Returns predictions and bookkeeping."""
    folds = cv_folds(D.year, tr)
    preds, info, oof, gains = {}, {}, {}, {}
    y_tr = D.y[tr]

    preds["base_rate"] = np.full(len(te), y_tr.mean())
    info["base_rate"] = {"train_break_rate": float(y_tr.mean())}

    p, rule, cv = one_rule(D, tr, te, folds)
    preds["one_rule"] = p; info["one_rule"] = {"rule": rule, "cv_auc_by_rule": cv}
    log(f"  one rule: {rule} (cv auc {cv[rule]:.3f})")

    # 2. logistic regression; `year` in or out is chosen by CV
    best = None
    for use_year in (False, True):
        ft = Feat(year=use_year)
        c, s, o = cv_linear(D, folds, LOGIT_C, lambda a, b, ft=ft: design_logit(D, ft, a, b))
        if best is None or (s == s and s > best[1]):
            best = (c, s, o, use_year)
    c, s, o, use_year = best
    A, B, names = design_logit(D, Feat(year=use_year), tr, te)
    m = LogisticRegression(C=c, max_iter=3000).fit(A, y_tr)
    preds["logit_tab"] = m.predict_proba(B)[:, 1]; oof["logit_tab"] = o
    info["logit_tab"] = {"C": c, "use_year": use_year, "cv_auc": s,
                         "top_coefficients": dict(sorted(zip(names, m.coef_[0].tolist()), key=lambda kv: -abs(kv[1]))[:15])}
    log(f"  logit: C={c} year={use_year} cv auc {s:.3f}")

    # 3. LightGBM tabular; `year` in or out chosen by CV and then held fixed for all LightGBM variants
    best = None
    for use_year in (False, True):
        bp, s, o = cv_lgbm(D, Feat(year=use_year), folds)
        if best is None or (s == s and s > best[1]):
            best = (bp, s, o, use_year)
    bp, s, o, use_year = best
    preds["lgbm_tab"], gains["lgbm_tab"] = fit_lgbm(D, Feat(year=use_year), tr, te, bp)
    oof["lgbm_tab"] = o
    info["lgbm_tab"] = {"num_leaves": bp[0], "n_estimators": bp[1], "use_year": use_year, "cv_auc": s}
    log(f"  lgbm tab: leaves={bp[0]} trees={bp[1]} year={use_year} cv auc {s:.3f}")

    variants = {}
    if text_models and D.K is not None:
        variants["lgbm_tab_kw"] = Feat(year=use_year, kw=True)
        variants["lgbm_tab_tfidf"] = Feat(year=use_year, kw=True, tfidf=True)
        if D.e_press is not None:
            variants["lgbm_tab_emb"] = Feat(year=use_year, kw=True, emb=True)
            variants["lgbm_text_only"] = Feat(tab=False, kw=True, emb=True)
    if full_extras:
        variants["lgbm_tab_suspects"] = Feat(year=use_year, suspects=True)
        variants["lgbm_tab_imputed"] = Feat(year=use_year, impute=True)
        if D.K is not None and D.has_text.any():
            variants["lgbm_tab_hastext"] = Feat(year=use_year, has_text_flag=True)
            if D.e_press is not None:
                variants["lgbm_full_text"] = Feat(year=use_year, has_text_flag=True, kw=True, emb=True)
    for name, ft in variants.items():
        bp, s, o = cv_lgbm(D, ft, folds)
        preds[name], gains[name] = fit_lgbm(D, ft, tr, te, bp)
        oof[name] = o
        info[name] = {"num_leaves": bp[0], "n_estimators": bp[1], "use_year": ft.year, "cv_auc": s}
        log(f"  {name}: leaves={bp[0]} trees={bp[1]} cv auc {s:.3f}")

    if text_models and D.press is not None:
        c, s, o = cv_linear(D, folds, TFIDF_LOGIT_C, lambda a, b: design_tfidf_sparse(D, a, b))
        A, B, names = design_tfidf_sparse(D, tr, te)
        m = LogisticRegression(C=c, max_iter=3000).fit(A, y_tr)
        preds["tfidf_logit"] = m.predict_proba(B)[:, 1]; oof["tfidf_logit"] = o
        order = np.argsort(m.coef_[0])
        info["tfidf_logit"] = {"C": c, "cv_auc": s,
                               "top_terms_break": [names[i] for i in order[::-1][:25]],
                               "top_terms_close": [names[i] for i in order[:25]]}
        log(f"  tfidf logit: C={c} cv auc {s:.3f}")

    for name, o in oof.items():   # ex-ante thresholds for the impact simulation
        if len(o):
            info[name]["oof_quantiles"] = {"q80": float(np.quantile(o, 0.8)), "q90": float(np.quantile(o, 0.9))}
    return preds, info, gains


# ----------------------------------------------------------------------------- reporting
def fmt_ci(v, ci, d=3):
    return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def table(y, preds, ci, order):
    rows = ["| Model | ROC AUC | PR AUC | Brier | Breaks in riskiest 10% | Breaks in riskiest 20% |",
            "|---|---|---|---|---|---|"]
    for m in order:
        if m not in preds:
            continue
        mt = metrics(y, preds[m])
        rows.append(f"| {LABELS[m]} | {fmt_ci(mt['roc_auc'], ci[m]['roc_auc'])} | {fmt_ci(mt['pr_auc'], ci[m]['pr_auc'])} | "
                    f"{fmt_ci(mt['brier'], ci[m]['brier'], 4)} | {fmt_ci(mt['capture_top10'], ci[m]['capture_top10'], 2)} | "
                    f"{fmt_ci(mt['capture_top20'], ci[m]['capture_top20'], 2)} |")
    return rows


def segment_table(seg: pd.Series, y, preds, models):
    rows = ["| Segment | Deals | Breaks | Break rate | " + " | ".join(f"AUC {LABELS[m].split('.')[0]}" for m in models) + " | "
            + " | ".join(f"Top-20% capture {LABELS[m].split('.')[0]}" for m in models) + " |",
            "|---|---:|---:|---:|" + "---:|" * (2 * len(models))]
    for s in list(pd.unique(seg)):
        mk = (seg == s).to_numpy()
        yy = y[mk]
        aucs = [f"{auc_safe(yy, preds[m][mk]):.3f}" if yy.sum() >= 5 else "n/a" for m in models]
        caps = [f"{capture_at(yy, preds[m][mk], 0.2):.2f}" if yy.sum() >= 5 else "n/a" for m in models]
        rows.append(f"| {s} | {mk.sum()} | {int(yy.sum())} | {yy.mean():.1%} | " + " | ".join(aucs) + " | " + " | ".join(caps) + " |")
    return rows


def make_figures(D, fig_dir, y_text, preds_text, y_full, preds_full, gains_full, gains_text, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    note = " (synthetic sample)" if synthetic else ""
    colors = {"base_rate": "#9e9e9e", "one_rule": "#bdbdbd", "logit_tab": "#6baed6", "lgbm_tab": "#08519c",
              "lgbm_tab_tfidf": "#fd8d3c", "lgbm_tab_emb": "#d94801", "tfidf_logit": "#74c476", "lgbm_text_only": "#238b45"}
    short = {"base_rate": "0 base rate", "one_rule": "1 one rule", "logit_tab": "2 logistic tabular",
             "lgbm_tab": "3 LightGBM tabular", "lgbm_tab_tfidf": "4 tabular + TF-IDF", "lgbm_tab_emb": "5 tabular + embeddings",
             "tfidf_logit": "6a text only (TF-IDF)", "lgbm_text_only": "6b text only (embeddings)"}

    # 1. break rate by year
    g = D.deals.groupby("year")["label"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(g.index, g["size"], color="#c6dbef", label="Deals (left)")
    ax.set_ylabel("Deals announced"); ax.set_xlabel("Announcement year")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot(g.index, 100 * g["mean"], color="#08519c", marker="o", ms=4, label="Break rate (right)")
    ax2.set_ylabel("Break rate (%)"); ax2.set_ylim(0, max(15, 100 * g["mean"].max() * 1.15))
    ax.axvline(TRAIN_END + 0.5, color="k", ls="--", lw=1)
    ax.text(TRAIN_END + 0.6, ax.get_ylim()[1] * 0.95, "test", va="top", fontsize=9)
    ax.text(TRAIN_END + 0.4, ax.get_ylim()[1] * 0.95, "train", va="top", ha="right", fontsize=9)
    ax.set_title("Deals and break rate by announcement year" + note, loc="left", fontsize=10)
    fig.savefig(fig_dir / "break_rate_by_year.png"); plt.close(fig)

    ladder = [m for m in ["one_rule", "logit_tab", "lgbm_tab", "lgbm_tab_tfidf", "lgbm_tab_emb", "tfidf_logit", "lgbm_text_only"]
              if m in preds_text]
    y, P = (y_text, preds_text) if len(ladder) > 3 else (y_full, preds_full)
    ladder = [m for m in ladder if m in P]

    # 2. ROC and PR
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in ladder:
        fpr, tpr, _ = roc_curve(y, P[m]); pr, rc, _ = precision_recall_curve(y, P[m])
        axes[0].plot(fpr, tpr, color=colors[m], lw=1.6, label=f"{short[m]} ({roc_auc_score(y, P[m]):.2f})")
        axes[1].plot(rc, pr, color=colors[m], lw=1.6, label=f"{short[m]} ({average_precision_score(y, P[m]):.2f})")
    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1)
    axes[1].axhline(y.mean(), color="#9e9e9e", ls="--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate"); axes[0].set_title("ROC, test 2019-2025" + note, loc="left", fontsize=10)
    axes[1].set(xlabel="Recall (share of breaks found)", ylabel="Precision", ylim=(0, min(1, max(0.6, 3 * y.mean()))))
    axes[1].set_title("Precision-recall (dashed: base rate)", loc="left", fontsize=10)
    axes[0].legend(fontsize=7.5, title="ROC AUC", title_fontsize=8, loc="lower right")
    axes[1].legend(fontsize=7.5, title="PR AUC", title_fontsize=8, loc="upper right")
    fig.savefig(fig_dir / "roc_pr_ladder.png"); plt.close(fig)

    # 3. calibration (quintile bins: few breaks, so few bins)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lim = 0.0
    for m in [m for m in ["logit_tab", "lgbm_tab", "lgbm_tab_emb"] if m in P]:
        q = pd.qcut(pd.Series(P[m]).rank(method="first"), 5, labels=False).to_numpy()
        xs = [P[m][q == b].mean() for b in range(5)]; ys = [y[q == b].mean() for b in range(5)]
        se = [np.sqrt(max(v * (1 - v), 1e-9) / (q == b).sum()) for b, v in enumerate(ys)]
        ax.errorbar(xs, ys, yerr=1.96 * np.array(se), color=colors[m], marker="o", ms=4, lw=1.4, capsize=2, label=short[m])
        lim = max(lim, max(xs), max(ys))
    lim = min(1.0, lim * 1.25 + 0.02)
    ax.plot([0, lim], [0, lim], color="#9e9e9e", ls="--", lw=1)
    ax.set(xlabel="Mean predicted break probability", ylabel="Observed break rate", xlim=(0, lim), ylim=(0, lim))
    ax.set_title("Calibration by score quintile, test" + note, loc="left", fontsize=10); ax.legend(fontsize=8)
    fig.savefig(fig_dir / "calibration.png"); plt.close(fig)

    # 4. top features (LightGBM gain share)
    panels = [(gains_full.get("lgbm_tab"), "LightGBM tabular (full sample)")]
    if gains_text.get("lgbm_tab_emb"):
        panels.append((gains_text["lgbm_tab_emb"], "Tabular + keywords + embeddings (text subset)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (g, title) in zip(axes, panels):
        s = pd.Series(g).sort_values(ascending=False); s = 100 * s / s.sum()
        top = s.head(15)[::-1]
        cols = ["#d94801" if n.startswith(("emb_", "tfidf_")) else "#fdae6b" if n.startswith(("kw_", "log_press", "log_agreement", "has_agreement")) else "#08519c" for n in top.index]
        ax.barh(top.index, top.values, color=cols); ax.set_xlabel("Share of total split gain (%)")
        ax.set_title(title + note, loc="left", fontsize=10); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(fig_dir / "top_features.png"); plt.close(fig)

    # 5. lift chart: cumulative share of breaks captured
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for m in ladder:
        order = np.argsort(-P[m], kind="stable")
        ax.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[order]) / y.sum(), color=colors[m], lw=1.6, label=short[m])
    ax.plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1, label="random")
    for x in (0.1, 0.2):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ax.set(xlabel="Share of deals flagged (riskiest first)", ylabel="Share of breaks captured", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Cumulative breaks captured, test" + note, loc="left", fontsize=10); ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(fig_dir / "lift_chart.png"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()
    synthetic = args.tag != "real"
    suffix = "" if not synthetic else f"_{args.tag}"
    res_dir = ROOT / ("results" if not synthetic else f"results_{args.tag}")
    fig_dir = ROOT / ("figures" if not synthetic else f"figures_{args.tag}")
    res_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    D = load_data(suffix)
    idx = np.arange(len(D.y))
    tr_full, te_full = idx[D.year <= TRAIN_END], idx[D.year >= TEST_START]
    tr_text, te_text = tr_full[D.has_text[tr_full]], te_full[D.has_text[te_full]]
    have_text = len(tr_text) > 50 and len(te_text) > 20 and D.y[te_text].sum() >= 2
    log(f"full sample: train {len(tr_full)} ({D.y[tr_full].sum()} breaks), test {len(te_full)} ({D.y[te_full].sum()} breaks)")
    log(f"text subset: train {len(tr_text)} ({D.y[tr_text].sum()} breaks), test {len(te_text)} ({D.y[te_text].sum()} breaks)")

    log("ladder on the full sample")
    P_full, I_full, G_full = run_ladder(D, tr_full, te_full, text_models=False, full_extras=True, log=log)
    P_text, I_text, G_text = {}, {}, {}
    if have_text:
        log("ladder on the text subset")
        P_text, I_text, G_text = run_ladder(D, tr_text, te_text, text_models=True, full_extras=False, log=log)

    y_full, y_text = D.y[te_full], D.y[te_text]
    log("bootstrap")
    ci_full, diff_full = bootstrap(y_full, P_full, [("lgbm_tab", "logit_tab"), ("lgbm_tab", "one_rule"),
                                                    ("lgbm_tab_hastext", "lgbm_tab"), ("lgbm_full_text", "lgbm_tab"),
                                                    ("lgbm_tab_suspects", "lgbm_tab"), ("lgbm_tab_imputed", "lgbm_tab")])
    ci_text, diff_text = ({}, {})
    if have_text:
        ci_text, diff_text = bootstrap(y_text, P_text, [("lgbm_tab_emb", "lgbm_tab"), ("lgbm_tab_tfidf", "lgbm_tab"),
                                                        ("lgbm_tab_kw", "lgbm_tab"), ("lgbm_tab", "logit_tab")])

    # ---- robustness: drop test deals that resolved within 10 days of announcement
    days_out = (D.deals["closed_date"].fillna(D.deals["cancelled_date"]) - D.deals["announced_date"]).dt.days.to_numpy()
    robust = {}
    for name, te, P in (("full", te_full, P_full), ("text", te_text, P_text)):
        if not P:
            continue
        keep = days_out[te] > 10
        robust[name] = {"n": int(keep.sum()), "breaks": int(D.y[te][keep].sum()),
                        "metrics": {m: metrics(D.y[te][keep], p[keep]) for m, p in P.items()
                                    if m in ("lgbm_tab", "lgbm_tab_emb", "lgbm_tab_tfidf", "lgbm_full_text")}}

    # ---- tables
    head = ["# Benchmark table" + (" (SYNTHETIC SAMPLE: not real results)" if synthetic else ""), "",
            f"Train: announcements 2001-{TRAIN_END}. Test: {TEST_START}-2025. Brackets are bootstrap 95% confidence "
            f"intervals ({N_BOOT} resamples of test deals). Label 1 = deal cancelled after a definitive agreement.", ""]
    order_text = ["base_rate", "one_rule", "logit_tab", "lgbm_tab", "lgbm_tab_kw", "lgbm_tab_tfidf", "lgbm_tab_emb",
                  "tfidf_logit", "lgbm_text_only"]
    order_full = ["base_rate", "one_rule", "logit_tab", "lgbm_tab", "lgbm_tab_hastext", "lgbm_full_text",
                  "lgbm_tab_suspects", "lgbm_tab_imputed"]
    lines = list(head)
    if have_text:
        lines += [f"## Text subset: deals with an announcement text (train {len(tr_text)} deals / {int(D.y[tr_text].sum())} breaks, "
                  f"test {len(te_text)} deals / {int(y_text.sum())} breaks)", "",
                  "All rows are trained and tested on the same deals, so the text rows are comparable with the tabular rows.", ""]
        lines += table(y_text, P_text, ci_text, order_text) + [""]
        lines += ["### Uplift on the text subset (paired bootstrap, same resamples)", "",
                  "| Comparison | ROC AUC | PR AUC | Brier (negative = better) | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---|---|---|---|---|"]
        for k, d in diff_text.items():
            a, b = k.split(" minus ")
            lines.append(f"| {LABELS[a].split('.')[0]} minus {LABELS[b].split('.')[0]} | " + " | ".join(
                f"{d[m]['point']:+.{4 if m == 'brier' else 3}f} [{d[m]['ci95'][0]:+.{4 if m == 'brier' else 3}f}, {d[m]['ci95'][1]:+.{4 if m == 'brier' else 3}f}]"
                for m in ["roc_auc", "pr_auc", "brier", "capture_top10", "capture_top20"]) + " |")
        lines.append("")
    lines += [f"## Full sample (train {len(tr_full)} deals / {int(D.y[tr_full].sum())} breaks, "
              f"test {len(te_full)} deals / {int(y_full.sum())} breaks)", ""]
    lines += table(y_full, P_full, ci_full, order_full) + [""]
    lines += ["### Differences on the full sample (paired bootstrap)", "",
              "| Comparison | ROC AUC | PR AUC | Brier (negative = better) | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---|---|---|---|---|"]
    for k, d in diff_full.items():
        a, b = k.split(" minus ")
        lines.append(f"| {LABELS[a].split('.')[0]} minus {LABELS[b].split('.')[0]} | " + " | ".join(
            f"{d[m]['point']:+.{4 if m == 'brier' else 3}f} [{d[m]['ci95'][0]:+.{4 if m == 'brier' else 3}f}, {d[m]['ci95'][1]:+.{4 if m == 'brier' else 3}f}]"
            for m in ["roc_auc", "pr_auc", "brier", "capture_top10", "capture_top20"]) + " |")
    lines += ["", f"One-rule benchmark chosen by CV on the training years: full sample = \"{I_full['one_rule']['rule']}\""
              + (f", text subset = \"{I_text['one_rule']['rule']}\"." if have_text else "."), ""]
    (res_dir / "benchmark_table.md").write_text("\n".join(lines) + "\n")

    # ---- segments (test set)
    Xte = D.X.iloc[te_full]
    seg_cs = F.cash_stock_segment(Xte).reset_index(drop=True)
    dv = D.deals["deal_value"].to_numpy()
    cuts = np.quantile(dv[tr_full], [1 / 3, 2 / 3])       # tercile cut-offs from the training years
    seg_size = pd.Series(np.select([dv[te_full] <= cuts[0], dv[te_full] <= cuts[1]],
                                   [f"small (<= ${cuts[0]:,.0f}m)", f"mid (<= ${cuts[1]:,.0f}m)"],
                                   f"large (> ${cuts[1]:,.0f}m)"))
    yr = D.year[te_full]
    seg_per = pd.Series(np.select([yr <= 2020, yr <= 2022], ["2019-2020", "2021-2022"], "2023-2025"))
    seg_lines = ["# Segment results on the test set" + (" (SYNTHETIC SAMPLE)" if synthetic else ""), "",
                 "AUC and capture are computed inside each segment; segments with fewer than 5 breaks show n/a. "
                 "Segment-level numbers rest on few breaks and are noisy. Deal-size terciles use cut-offs from the training years.", ""]
    full_models = [m for m in ["logit_tab", "lgbm_tab", "lgbm_full_text"] if m in P_full]
    for title, seg in (("Consideration", seg_cs), ("Deal size tercile (total transaction value)", seg_size), ("Period", seg_per)):
        seg_lines += [f"## Full sample, by {title.lower()}", ""] + segment_table(seg, y_full, P_full, full_models) + [""]
    if have_text:
        in_text = D.has_text[te_full]
        text_models_ = [m for m in ["lgbm_tab", "lgbm_tab_tfidf", "lgbm_tab_emb"] if m in P_text]
        for title, seg in (("Consideration", seg_cs), ("Deal size tercile (total transaction value)", seg_size), ("Period", seg_per)):
            seg_lines += [f"## Text subset, by {title.lower()}", ""] + \
                segment_table(seg[in_text].reset_index(drop=True), y_text, P_text, text_models_) + [""]
    seg_lines += ["Model keys: " + "; ".join(f"{LABELS[m].split('.')[0]} = {LABELS[m].split('. ', 1)[1]}"
                                             for m in dict.fromkeys(full_models + (text_models_ if have_text else []))), ""]
    (res_dir / "segments.md").write_text("\n".join(seg_lines) + "\n")

    # ---- metrics.json
    def block_share(g):
        s = pd.Series(g); tot = s.sum() or 1.0
        kw = s[[n for n in s.index if n in F.KEYWORD_FEATURES]].sum()
        tx = s[[n for n in s.index if n.startswith(("emb_", "tfidf_"))]].sum()
        return {"tabular": float((tot - kw - tx) / tot), "keywords": float(kw / tot), "text_components": float(tx / tot)}

    out = {
        "tag": args.tag, "split": {"train_years": [2001, TRAIN_END], "test_years": [TEST_START, 2025]},
        "full_sample": {"n_train": int(len(tr_full)), "breaks_train": int(D.y[tr_full].sum()),
                        "n_test": int(len(te_full)), "breaks_test": int(y_full.sum()),
                        "metrics": {m: metrics(y_full, p) for m, p in P_full.items()}, "ci95": ci_full,
                        "paired_differences": diff_full, "selection": I_full,
                        "gain_share_by_block": {m: block_share(g) for m, g in G_full.items()}},
        "text_subset": {"n_train": int(len(tr_text)), "breaks_train": int(D.y[tr_text].sum()),
                        "n_test": int(len(te_text)), "breaks_test": int(y_text.sum()),
                        "metrics": {m: metrics(y_text, p) for m, p in P_text.items()}, "ci95": ci_text,
                        "paired_differences": diff_text, "selection": I_text,
                        "gain_share_by_block": {m: block_share(g) for m, g in G_text.items()}} if have_text else None,
        "robustness_excluding_deals_resolved_within_10_days": robust,
        "n_bootstrap": N_BOOT,
    }
    (res_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- deal-level test predictions for impact.py (not committed)
    pred = pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy()[te_full], "year": yr, "label": y_full,
                         "in_text_subset": D.has_text[te_full]})
    for m, p in P_full.items():
        pred[f"full__{m}"] = p
    for m, p in P_text.items():
        col = np.full(len(te_full), np.nan); col[D.has_text[te_full]] = p
        pred[f"text__{m}"] = col
    pred.to_parquet(INTERIM / f"test_predictions{suffix}.parquet", index=False)

    make_figures(D, fig_dir, y_text, P_text, y_full, P_full, G_full, G_text, synthetic)
    log(f"wrote {res_dir.name}/ and {fig_dir.name}/")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

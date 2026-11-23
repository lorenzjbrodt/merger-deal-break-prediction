"""Shared modelling code: data assembly, in-fold design matrices, CV, fitting, metrics.

Everything fitted (imputer, scaler, one-hot, top-advisor lists, TF-IDF vocabulary, SVD, PCA)
is fitted on the training rows passed to the design function. Outcome fields never enter a
design matrix: labels are passed separately as `y`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import features as F
import market_context as MC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
TRAIN_END, TEST_START = 2018, 2019
SEEDS = [0, 1, 2, 3, 4]
N_JOBS = 8
LGBM_BASE = dict(objective="binary", learning_rate=0.03, min_child_samples=25, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, verbose=-1, n_jobs=N_JOBS)
LGBM_LEAVES = [3, 6, 12]
LGBM_TREES = [100, 200, 400]
LOGIT_C = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TFIDF_LOGIT_C = [0.1, 0.3, 1.0, 3.0, 10.0]
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"     # letters only: no dates / years / prices in the vocabulary
SPREAD_FEATURES = ["spread", "p_mkt", "p_mkt_carry", "days_to_t1"]
TARGETS = {"label_adverse": "adverse break (break excluding superior competing bids)", "label": "any break"}


# ----------------------------------------------------------------------------- data
@dataclass
class Data:
    deals: pd.DataFrame
    X: pd.DataFrame
    y: dict
    year: np.ndarray
    has_text: np.ndarray
    core: np.ndarray                 # not a reverse merger (primary sample)
    revised: np.ndarray              # notes mention a revised offer (analysis only)
    days_out: np.ndarray             # days from announcement to outcome (analysis only)
    outcome3: np.ndarray
    K: np.ndarray | None = None
    press: np.ndarray | None = None
    agr: np.ndarray | None = None
    e_press: np.ndarray | None = None
    e_agr: np.ndarray | None = None
    L: np.ndarray | None = None
    L_names: list = field(default_factory=list)
    ft: dict = field(default_factory=dict)
    PF: pd.DataFrame | None = None
    extended: bool = False
    has_market: bool = False
    cache: dict = field(default_factory=dict)


def load_data(suffix: str = "") -> Data:
    deals = pd.read_parquet(INTERIM / f"deals{suffix}.parquet").reset_index(drop=True)
    n = len(deals)
    extended = "secondary_features" in deals.columns
    op = INTERIM / f"outcomes{suffix}.parquet"
    if op.exists():
        deals = deals.merge(pd.read_parquet(op)[["deal_id", "reason", "outcome3", "label_adverse", "revised_terms"]], on="deal_id", how="left")
    else:   # synthetic sample: no notes, so the two targets coincide
        deals["reason"] = np.where(deals["label"] == 1, "unknown", None)
        deals["outcome3"] = np.where(deals["label"] == 1, "adverse_break", "closed")
        deals["label_adverse"] = deals["label"]
        deals["revised_terms"] = False
    X = F.build_tabular(deals)
    if extended:
        X = pd.concat([X, F.build_extended(deals)], axis=1)
    has_market = False
    try:
        X = pd.concat([X, MC.market_features(deals)], axis=1)
        has_market = True
    except FileNotFoundError:
        pass
    assert not set(X.columns) & set(F.FORBIDDEN), "forbidden column leaked into features"
    reverse = deals["secondary_features"].fillna("").astype(str).str.contains("Reverse Merger").to_numpy() if extended else np.zeros(n, bool)
    D = Data(deals=deals, X=X, y={"label": deals["label"].to_numpy(int), "label_adverse": deals["label_adverse"].fillna(0).to_numpy(int)},
             year=deals["year"].to_numpy(int), has_text=np.zeros(n, bool), core=~reverse,
             revised=deals["revised_terms"].fillna(False).to_numpy(bool),
             days_out=(deals["closed_date"].fillna(deals["cancelled_date"]) - deals["announced_date"]).dt.days.to_numpy(),
             outcome3=deals["outcome3"].astype(object).to_numpy(), extended=extended, has_market=has_market)
    tp = INTERIM / f"text{suffix}.parquet"
    if tp.exists():
        text = deals[["deal_id"]].merge(pd.read_parquet(tp), on="deal_id", how="left")
        D.press = text["press_release"].fillna("").astype(str).to_numpy(dtype=object)
        D.agr = text["agreement"].fillna("").astype(str).to_numpy(dtype=object)
        D.has_text = text["has_text"].fillna(False).to_numpy(dtype=bool)
        kp = INTERIM / f"keywords{suffix}.parquet"      # keyword flags are label-free; cached because the regex pass is slow
        if kp.exists() and kp.stat().st_mtime >= tp.stat().st_mtime and len(pd.read_parquet(kp)) == n:
            D.K = np.array(pd.read_parquet(kp).to_numpy(dtype=float), copy=True)
        else:
            tt = pd.DataFrame({"press_release": np.where(D.has_text, D.press, ""), "agreement": np.where(D.has_text, D.agr, "")})
            Kdf = F.keyword_features(tt)
            Kdf.to_parquet(kp, index=False)
            D.K = Kdf.to_numpy(dtype=float)
        ep = INTERIM / f"embeddings{suffix}.npz"
        if ep.exists():
            z = np.load(ep, allow_pickle=False)
            pos = pd.Series(np.arange(len(z["deal_id"])), index=z["deal_id"]).reindex(deals["deal_id"].astype(str))
            ok = pos.notna().to_numpy()
            D.e_press = np.full((n, z["press"].shape[1]), np.nan, dtype=np.float32)
            D.e_agr = np.full((n, z["agreement"].shape[1]), np.nan, dtype=np.float32)
            D.e_press[ok] = z["press"][pos[ok].astype(int)]
            D.e_agr[ok] = z["agreement"][pos[ok].astype(int)]
            D.e_press[~D.has_text] = np.nan
            D.e_agr[~D.has_text] = np.nan
    lp = INTERIM / f"llm_terms{suffix}.parquet"
    if lp.exists():
        Lt = deals[["deal_id"]].merge(pd.read_parquet(lp), on="deal_id", how="left").drop(columns="deal_id")
        D.L, D.L_names = np.array(Lt.to_numpy(dtype=float), copy=True), list(Lt.columns)
        D.L[~D.has_text] = np.nan
    for t in TARGETS:
        fp = INTERIM / f"ft_scores_{t}{suffix}.parquet"
        if fp.exists():
            D.ft[t] = deals[["deal_id"]].merge(pd.read_parquet(fp), on="deal_id", how="left")["ft_score"].to_numpy(dtype=float)
    pp = INTERIM / f"price_features{suffix}.parquet"
    if pp.exists():
        D.PF = deals[["deal_id"]].merge(pd.read_parquet(pp), on="deal_id", how="left")
    D.X["has_text"] = D.has_text.astype(int)
    return D


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
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


# ----------------------------------------------------------------------------- designs
@dataclass(frozen=True)
class Feat:
    groups: tuple = ()           # deal_terms, premium_price, conditions, financials, market_context, advisors
    base_only: bool = False      # the base-run tabular feature set (40-column export)
    year: bool = False
    suspects: bool = False
    impute: bool = False
    has_text_flag: bool = False
    kw: bool = False
    tfidf: bool = False
    emb: bool = False
    llm: bool = False
    spread: bool = False


TAB_GROUPS = ("deal_terms", "premium_price", "conditions", "financials", "advisors")


def feat_columns(D: Data, feat: Feat):
    num, cat = [], []
    if feat.base_only:
        num, cat = list(F.NUMERIC), list(F.CATEGORICAL)
    for g in feat.groups:
        if g in F.GROUPS:
            num += F.GROUPS[g][0]; cat += F.GROUPS[g][1]
        elif g == "market_context":
            num += MC.MARKET_FEATURES
    if feat.year:
        num += F.YEAR
    if feat.suspects:
        num += F.SUSPECT_FEATURES
    if feat.impute:
        num = [c for c in num if c != "conditions_missing"]
    num = [c for c in dict.fromkeys(num) if c in D.X.columns]
    cat = [c for c in dict.fromkeys(cat) if c in D.X.columns]
    return num, cat


def _key(name, tr, te):
    return (name, len(tr), int(tr[:50].sum()), int(tr[-50:].sum()), len(te), int(te[:50].sum()), int(te[-50:].sum()))


def _tfidf_block(D: Data, tr, te):
    k_ = _key("tfidf", tr, te)
    if k_ in D.cache:
        return D.cache[k_]
    A, B, names = [], [], []
    for fld, docs, k, kw in (("press", D.press, 30, dict(ngram_range=(1, 2), min_df=5, max_features=30000)),
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
        A.append(a); B.append(b); names += [f"tfidf_{fld}_svd{i + 1}" for i in range(k)]
    D.cache[k_] = (np.hstack(A), np.hstack(B), names)
    return D.cache[k_]


def _emb_block(D: Data, tr, te):
    A, B, names = [], [], []
    for fld, E, k in (("press", D.e_press, 20), ("agreement", D.e_agr, 10)):
        ok = np.isfinite(E[:, 0])
        a = np.full((len(tr), k), np.nan); b = np.full((len(te), k), np.nan)
        if ok[tr].sum() > 5 * k:
            pca = PCA(k, random_state=0).fit(E[tr][ok[tr]])
            a[ok[tr]] = pca.transform(E[tr][ok[tr]])
            if ok[te].any():
                b[ok[te]] = pca.transform(E[te][ok[te]])
        A.append(a); B.append(b); names += [f"emb_{fld}_pc{i + 1}" for i in range(k)]
    return np.hstack(A), np.hstack(B), names


def design_tree(D: Data, feat: Feat, tr: np.ndarray, te: np.ndarray):
    """Dense design matrices for LightGBM; everything fitted on `tr` only."""
    A, B, names = [], [], []
    num, cat = feat_columns(D, feat)
    if num or cat:
        pre = F.tree_preprocessor(num, cat, impute=feat.impute)
        A.append(np.asarray(pre.fit_transform(D.X.iloc[tr]), dtype=float))
        B.append(np.asarray(pre.transform(D.X.iloc[te]), dtype=float))
        names += [str(n).replace("_infrequent_sklearn", "_other") for n in pre.get_feature_names_out()]
    if "advisors" in feat.groups and D.extended:
        ta = F.TopAdvisorFlags().fit(D.deals.iloc[tr])
        A.append(ta.transform(D.deals.iloc[tr])); B.append(ta.transform(D.deals.iloc[te])); names += F.TopAdvisorFlags.NAMES
    if feat.has_text_flag:
        A.append(D.has_text[tr, None].astype(float)); B.append(D.has_text[te, None].astype(float)); names.append("has_text")
    if feat.kw and D.K is not None:
        A.append(D.K[tr]); B.append(D.K[te]); names += F.KEYWORD_FEATURES
    if feat.tfidf and D.press is not None:
        a, b, nm = _tfidf_block(D, tr, te); A.append(a); B.append(b); names += nm
    if feat.emb and D.e_press is not None:
        a, b, nm = _emb_block(D, tr, te); A.append(a); B.append(b); names += nm
    if feat.llm and D.L is not None:
        A.append(D.L[tr]); B.append(D.L[te]); names += D.L_names
    if feat.spread and D.PF is not None:
        S = D.PF[SPREAD_FEATURES].to_numpy(dtype=float)
        A.append(S[tr]); B.append(S[te]); names += SPREAD_FEATURES
    return np.hstack(A), np.hstack(B), names


def design_logit(D: Data, feat: Feat, tr, te):
    num, cat = feat_columns(D, feat)
    pre = F.logistic_preprocessor(num, cat)
    return pre.fit_transform(D.X.iloc[tr]), pre.transform(D.X.iloc[te]), [str(n) for n in pre.get_feature_names_out()]


def design_tfidf_sparse(D: Data, tr, te):
    vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=TOKEN, ngram_range=(1, 2),
                          min_df=5, max_df=0.9, max_features=30000, sublinear_tf=True)
    return vec.fit_transform(D.press[tr]), vec.transform(D.press[te]), list(vec.get_feature_names_out())


# ----------------------------------------------------------------------------- fitting
def auc_safe(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def cv_lgbm(D: Data, y, feat: Feat, folds, grid=None):
    """Mean fold AUC per (num_leaves, n_estimators). Returns best params, its CV AUC, and the
    out-of-fold predictions (row index, score) of the best configuration."""
    grid = grid or [(l, t) for l in LGBM_LEAVES for t in LGBM_TREES]
    leaves = sorted({l for l, _ in grid})
    scores = {g: [] for g in grid}
    oof = {g: [] for g in grid}
    oof_idx = []
    for tr, va in folds:
        if y[tr].sum() < 5 or y[va].sum() < 2:
            continue
        A, B, _ = design_tree(D, feat, tr, va)
        oof_idx.append(va)
        for l in leaves:
            trees = sorted(t for ll, t in grid if ll == l)
            m = LGBMClassifier(**LGBM_BASE, num_leaves=l, n_estimators=max(trees), random_state=0).fit(A, y[tr])
            for t in trees:
                p = m.predict_proba(B, num_iteration=t)[:, 1]
                scores[(l, t)].append(auc_safe(y[va], p)); oof[(l, t)].append(p)
    if not oof_idx:
        return grid[0], np.nan, np.array([], int), np.array([])
    mean = {k: float(np.nanmean(v)) for k, v in scores.items() if v}
    best = max(mean, key=mean.get)
    return best, mean[best], np.concatenate(oof_idx), np.concatenate(oof[best])


def fit_lgbm(D: Data, y, feat: Feat, tr, te, best, return_models=False):
    A, B, names = design_tree(D, feat, tr, te)
    p = np.zeros(len(te)); gain = np.zeros(A.shape[1]); models = []
    for seed in SEEDS:
        m = LGBMClassifier(**LGBM_BASE, num_leaves=best[0], n_estimators=best[1], random_state=seed).fit(A, y[tr])
        p += m.predict_proba(B)[:, 1] / len(SEEDS)
        gain += m.booster_.feature_importance(importance_type="gain") / len(SEEDS)
        models.append(m)
    if return_models:
        return p, dict(zip(names, gain.tolist())), models, A, B, names
    return p, dict(zip(names, gain.tolist()))


def cv_linear(y, folds, grid, design):
    scores = {c: [] for c in grid}
    oof = {c: [] for c in grid}
    oof_idx = []
    for tr, va in folds:
        if y[tr].sum() < 5 or y[va].sum() < 2:
            continue
        A, B, _ = design(tr, va)
        oof_idx.append(va)
        for c in grid:
            m = LogisticRegression(C=c, max_iter=3000).fit(A, y[tr])
            p = m.predict_proba(B)[:, 1]
            scores[c].append(auc_safe(y[va], p)); oof[c].append(p)
    if not oof_idx:
        return grid[0], np.nan, np.array([], int), np.array([])
    mean = {k: float(np.nanmean(v)) for k, v in scores.items() if v}
    best = max(mean, key=mean.get)
    return best, mean[best], np.concatenate(oof_idx), np.concatenate(oof[best])


ONE_RULES = {
    "stock-only consideration": lambda X: (X["consideration_type"] == "stock").to_numpy(),
    "any stock in consideration": lambda X: (X["stock_pct"].fillna(0) > 0).to_numpy(),
    "solicited approach": lambda X: (X["solicited"] == 1).to_numpy(),
    "buyer not listed": lambda X: (X["buyer_listed"] == 0).to_numpy(),
    "financial condition attached": lambda X: (X["cond_financial"] == 1).to_numpy(),
    "not yet signed at announcement": lambda X: (X["signed_after_announcement"] == 1).to_numpy(),
    "more than one buyer": lambda X: (X["buyer_count"] > 1).to_numpy(),
}


def one_rule(D: Data, y, tr, te, folds):
    def score(rule, a, b):
        r_tr, r_te = rule(D.X.iloc[a]), rule(D.X.iloc[b])
        p1 = y[a][r_tr].mean() if r_tr.any() else y[a].mean()
        p0 = y[a][~r_tr].mean() if (~r_tr).any() else y[a].mean()
        return np.where(r_te, p1, p0)
    cv = {}
    for name, rule in ONE_RULES.items():
        aucs = [auc_safe(y[va], score(rule, a, va)) for a, va in folds if y[a].sum() >= 5 and y[va].sum() >= 2]
        cv[name] = float(np.nanmean(aucs)) if aucs else 0.5
    best = max(cv, key=cv.get)
    return score(ONE_RULES[best], tr, te), best, cv


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def platt(oof_pred, oof_y):
    """Platt scaling fitted on out-of-fold (validation) predictions only."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof_pred)[:, None], oof_y)
    return lambda p: m.predict_proba(logit(np.asarray(p))[:, None])[:, 1]


# ----------------------------------------------------------------------------- metrics
def capture_at(y, s, frac):
    """Share of all positives that sit in the riskiest `frac` of deals (ties split pro rata)."""
    n = len(y); k = max(int(round(frac * n)), 1)
    order = np.argsort(-s, kind="stable"); ss, ys = s[order], y[order]
    thr = ss[k - 1]
    above, tied = ss > thr, ss == thr
    got = ys[above].sum() + ys[tied].sum() * (k - above.sum()) / tied.sum()
    return float(got / max(ys.sum(), 1))


def metrics(y, s):
    return {"roc_auc": float(auc_safe(y, s)), "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
            "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
            "capture_top10": capture_at(y, s, 0.10), "capture_top20": capture_at(y, s, 0.20)}


METRIC_KEYS = ["roc_auc", "pr_auc", "brier", "capture_top10", "capture_top20"]


def bootstrap(y, preds: dict, pairs: list, n_boot=2000, seed=0):
    """Nonparametric bootstrap over test deals: 95% CIs per model and for paired differences."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: [] for m in preds}
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        yb = y[ix]
        if yb.sum() < 2 or yb.sum() == n:
            continue
        for m, p in preds.items():
            draws[m].append(list(metrics(yb, p[ix]).values()))
    arr = {m: np.array(v) for m, v in draws.items()}
    ci = {m: {k: [float(np.nanpercentile(a[:, j], 2.5)), float(np.nanpercentile(a[:, j], 97.5))] for j, k in enumerate(METRIC_KEYS)}
          for m, a in arr.items()}
    diffs = {}
    for a, b in pairs:
        if a in arr and b in arr:
            d = arr[a] - arr[b]
            pa, pb = metrics(y, preds[a]), metrics(y, preds[b])
            diffs[f"{a} minus {b}"] = {k: {"point": float(pa[k] - pb[k]),
                                           "ci95": [float(np.nanpercentile(d[:, j], 2.5)), float(np.nanpercentile(d[:, j], 97.5))],
                                           "share_boot_gt0": float((d[:, j] > 0).mean())} for j, k in enumerate(METRIC_KEYS)}
    return ci, diffs

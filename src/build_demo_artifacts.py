"""Build the offline artefacts for the Streamlit demo (app/demo.py).

Writes artifacts/demo/demo.pkl (git-ignored: it holds deal-level rows). Two modes:

  python src/build_demo_artifacts.py              real data (needs the full pipeline outputs)
  python src/build_demo_artifacts.py --synthetic  built from data/sample/synthetic_ciq_export.csv only

Contents
  deals      test-period deals with terms, model score, percentile, market-implied probability,
             top drivers (SHAP), extracted terms, the opening of the press release, and the underwriting
             view (expected time to closing with an 80% interval, loss if it breaks, expected annualized return)
  compact    a small LightGBM on terms a user can type in, plus keyword flags from pasted text,
             used for the "enter your own deal" mode (clearly weaker than the full model)
  similar    TF-IDF index of historical broken deals for "most similar broken deals"
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F          # noqa: E402
import modeling as M          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/demo"
COMPACT = ["log_deal_value", "premium_1d", "cash_pct", "solicited", "buyer_listed", "buyer_financial", "feat_tender_offer", "feat_lbo",
           "feat_go_shop", "feat_cross_border", "feat_going_private", "sell_fee_pct_value", "buy_fee_pct_value", "rel_size_deal_to_acq"]
KW = list(F.KEYWORDS)


def compact_matrix(X: pd.DataFrame, K: np.ndarray | None) -> np.ndarray:
    A = X.reindex(columns=COMPACT).to_numpy(dtype=float)
    k = K[:, :len(KW)] if K is not None else np.full((len(X), len(KW)), np.nan)
    return np.hstack([A, k])


def fit_compact(D: M.Data, y, tr):
    A = compact_matrix(D.X.iloc[tr], D.K[tr] if D.K is not None else None)
    models = [LGBMClassifier(**M.LGBM_BASE, num_leaves=3, n_estimators=200, random_state=s).fit(A, y[tr]) for s in M.SEEDS]
    # Platt scaling on out-of-fold predictions from the time-ordered folds
    oi, op = [], []
    for a, v in M.cv_folds(D.year, tr):
        if y[a].sum() < 5:
            continue
        m = LGBMClassifier(**M.LGBM_BASE, num_leaves=3, n_estimators=200, random_state=0).fit(compact_matrix(D.X.iloc[a], D.K[a] if D.K is not None else None), y[a])
        oi.append(v); op.append(m.predict_proba(compact_matrix(D.X.iloc[v], D.K[v] if D.K is not None else None))[:, 1])
    cal = None
    if oi:
        oi, op = np.concatenate(oi), np.concatenate(op)
        if 0 < y[oi].sum() < len(oi):
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(C=1e6, max_iter=1000).fit(M.logit(op)[:, None], y[oi])
    return {"models": models, "features": COMPACT + KW, "calibrator": cal}


def compact_predict(compact: dict, A: np.ndarray):
    p = np.mean([m.predict_proba(A)[:, 1] for m in compact["models"]], axis=0)
    contrib = np.mean([m.booster_.predict(A, pred_contrib=True) for m in compact["models"]], axis=0)[:, :-1]
    if compact["calibrator"] is not None:
        p = compact["calibrator"].predict_proba(M.logit(p)[:, None])[:, 1]
    return p, contrib


def build(synthetic: bool) -> Path:
    suffix = "_synthetic" if synthetic else ""
    if synthetic and not (M.INTERIM / "deals_synthetic.parquet").exists():
        import subprocess
        py = sys.executable
        subprocess.run([py, str(ROOT / "src/prepare.py"), "--input", str(ROOT / "data/sample/synthetic_ciq_export.csv"), "--tag", "synthetic"], check=True)
        subprocess.run([py, str(ROOT / "src/build_text.py"), "--tag", "synthetic"], check=True)
    D = M.load_data(suffix)
    y = D.y["label_adverse"]
    idx = np.arange(len(y))
    tr, te = idx[D.core & (D.year <= M.TRAIN_END)], idx[D.core & (D.year >= M.TEST_START)]
    compact = fit_compact(D, y, tr)
    A_te = compact_matrix(D.X.iloc[te], D.K[te] if D.K is not None else None)
    p_c, contrib_c = compact_predict(compact, A_te)
    A_tr = compact_matrix(D.X.iloc[tr], D.K[tr] if D.K is not None else None)
    hist_compact = compact_predict(compact, A_tr)[0]

    score, hist, drivers, source = p_c, hist_compact, None, "compact model"
    pp = M.INTERIM / f"predictions{suffix}.parquet"
    mj = ROOT / ("results" if not synthetic else "results_synthetic") / "metrics.json"
    if pp.exists() and mj.exists() and not synthetic:
        P = pd.read_parquet(pp).set_index("deal_id").reindex(D.deals["deal_id"])
        final = json.loads(mj.read_text())["final_models"]["overall"]
        ftab = json.loads(mj.read_text())["final_models"]["tabular"]
        col, ocol = f"cal__primary__full__{final}", f"oof__primary__full__{final}"
        if col in P and P[col].iloc[te].notna().all():
            score, source = P[col].to_numpy()[te], f"final model ({final}), Platt-calibrated"
            ho = P[ocol].dropna().to_numpy() if ocol in P else hist_compact
            hist = ho
            # put history on the calibrated scale too
            raw_te = P[f"pred__primary__full__{final}"].to_numpy()[te]
            order = np.argsort(raw_te)
            hist = np.interp(ho, raw_te[order], score[order])
        sp = ROOT / "artifacts/final_tabular_shap.pkl"
        if sp.exists():
            sh = pickle.load(open(sp, "rb"))
            pos = {d: i for i, d in enumerate(sh["deal_id"])}
            drivers = {"names": sh["names"], "pos": pos, "contrib": sh["contrib"], "values": sh["values"], "model": ftab}

    # underwriting view (real data only): time to closing with an 80% interval, loss if it breaks, expected annualized return
    uw = {}
    if not synthetic and (M.INTERIM / "duration_product.parquet").exists():
        du = pd.read_parquet(M.INTERIM / "duration_product.parquet").set_index("deal_id")      # product forecast and conformal 80% interval
        lo = pd.read_parquet(M.INTERIM / "loss_oot.parquet").set_index("deal_id") if (M.INTERIM / "loss_oot.parquet").exists() else None
        er = pd.read_parquet(M.INTERIM / "expected_return.parquet").set_index("deal_id") if (M.INTERIM / "expected_return.parquet").exists() else None
        ear_all = er["test__ear"].dropna().to_numpy() if er is not None else np.array([])

        def num(frame, did, col, f=lambda v: v):
            if frame is None or did not in frame.index or col not in frame or pd.isna(frame.at[did, col]):
                return None
            return float(f(frame.at[did, col]))
        for did in D.deals["deal_id"].iloc[te]:
            e = num(er, did, "test__ear")
            uw[did] = {"days": num(du, did, "split__mu", np.exp), "days_lo": num(du, did, "split__lo", np.exp), "days_hi": num(du, did, "split__hi", np.exp),
                       "uses_guidance": bool(num(du, did, "split__use_guide") == 1), "loss": num(lo, did, "split__adverse"),
                       "ear": e, "exp_ret": num(er, did, "test__exp_ret"), "p_top": num(er, did, "test__p_top"),
                       "ear_pct": None if e is None or not len(ear_all) else float((ear_all < e).mean())}
    d = D.deals.iloc[te]
    rows = []
    llm_dir = M.INTERIM / "llm_extract/out/haiku"
    kf = M.INTERIM / "llm_extract/kept_fields.json"
    kept = (set(json.loads(kf.read_text())) | {"regulatory_approvals"}) if kf.exists() else None   # hide fields that failed validation
    for j, i in enumerate(te):
        r = D.deals.iloc[i]
        if drivers is not None and r["deal_id"] in drivers["pos"]:
            k = drivers["pos"][r["deal_id"]]; c = drivers["contrib"][k]; v = drivers["values"][k]; nm = drivers["names"]
        else:
            c, v, nm = contrib_c[j], A_te[j], compact["features"]
        top = np.argsort(-np.abs(c))[:8]
        terms = {}
        lp = llm_dir / f"{r['deal_id']}.json"
        if lp.exists():
            terms = {k: v_ for k, v_ in json.loads(lp.read_text()).items() if k != "doc_id" and v_ not in (None, [], "") and (kept is None or k in kept)}
            if kept is not None and isinstance(terms.get("regulatory_approvals"), list):
                terms["regulatory_approvals"] = [a for a in terms["regulatory_approvals"] if f"reg_{a}" in kept]
        elif D.K is not None and D.has_text[i]:
            terms = {k.replace("kw_", "mentions "): bool(x) for k, x in zip(KW, D.K[i][:len(KW)]) if x == 1}
        pf = D.PF.iloc[i] if D.PF is not None else None
        rows.append({
            "deal_id": r["deal_id"], "target": r["target_clean"], "buyer": str(r["buyers"]) if pd.notna(r["buyers"]) else "", "announced": str(pd.Timestamp(r["announced_date"]).date()),
            "deal_value": float(r["deal_value"]), "premium_1d": None if pd.isna(r["premium_1d"]) else float(r["premium_1d"]),
            "consideration": str(r["consideration"]), "score": float(score[j]), "score_compact": float(p_c[j]),
            "percentile": float((hist < score[j]).mean()),
            "p_mkt": None if pf is None or pd.isna(pf["p_mkt"]) or not bool(pf["backtest_ok"]) or not bool(pf["cash_deal"]) else float(pf["p_mkt"]),      # all-cash deals only:
            "spread": None if pf is None or pd.isna(pf["spread"]) or not bool(pf["backtest_ok"]) or not bool(pf["cash_deal"]) else float(pf["spread"]),   # stock-deal spreads are rough
            "outcome": str(D.outcome3[i]), "drivers": [(nm[t], None if not np.isfinite(v[t]) else float(v[t]), float(c[t])) for t in top],
            "terms": terms, "press": (D.press[i][:1800] if D.press is not None and D.has_text[i] else ""),
            "compact_row": A_te[j].astype(float).tolist(), "underwriting": uw.get(r["deal_id"]),
        })
    if synthetic:
        # Made-up underwriting values so that the fallback shows the same layout as the real product. They follow no
        # data: time to closing is drawn around 45 days for tender offers and 120 for mergers, spread and loss from
        # fixed distributions, and the expected annualized return is computed from them and the synthetic break score.
        rng = np.random.default_rng(15071)
        it = COMPACT.index("feat_tender_offer")
        for r_ in rows:
            days = float(np.exp(rng.normal(np.log(45 if r_["compact_row"][it] == 1 else 120), 0.25)))
            spread = float(np.clip(rng.normal(0.03, 0.02), 0.003, 0.12))
            loss = -float(np.clip(rng.normal(0.25, 0.08), 0.05, 0.60))
            er_ = (1 - r_["score"]) * spread + r_["score"] * loss
            r_["spread"], r_["p_mkt"] = spread, float(np.clip(rng.beta(2, 12), 0.01, 0.9))
            r_["underwriting"] = {"days": days, "days_lo": days * 0.62, "days_hi": days * 1.55, "uses_guidance": False, "loss": loss,
                                  "ear": er_ / (days / 365), "exp_ret": er_, "p_top": None, "ear_pct": None}
        ears = np.array([r_["underwriting"]["ear"] for r_ in rows])
        for r_ in rows:
            r_["underwriting"]["ear_pct"] = float((ears < r_["underwriting"]["ear"]).mean())
    # similarity index over historical broken deals (training period)
    similar = None
    if D.press is not None:
        broke = tr[(y[tr] == 1) & D.has_text[tr]]
        fit_rows = tr[D.has_text[tr]]
        if len(broke) >= 3:
            vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=M.TOKEN, ngram_range=(1, 2), min_df=2 if synthetic else 5,
                                  max_df=0.9, max_features=30000, sublinear_tf=True).fit(D.press[fit_rows])
            similar = {"vectorizer": vec, "matrix": vec.transform(D.press[broke]),
                       "meta": [{"target": D.deals["target_clean"].iloc[i], "buyer": str(D.deals["buyers"].iloc[i]), "year": int(D.year[i]),
                                 "reason": str(D.deals["reason"].iloc[i])} for i in broke]}
    art = {"synthetic": synthetic, "source": source, "deals": rows, "compact": compact, "history": np.sort(hist).astype(np.float32),
           "history_compact": np.sort(hist_compact).astype(np.float32), "similar": similar, "base_rate": float(y[tr].mean()),
           "compact_medians": np.nanmedian(A_tr, axis=0).tolist(), "price_source": "synthetic" if synthetic else getattr(D, "pf_source", "")}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("demo_synthetic.pkl" if synthetic else "demo.pkl")
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    print(f"wrote {path.relative_to(ROOT)}: {len(rows)} test deals, score source: {source}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    build(ap.parse_args().synthetic)

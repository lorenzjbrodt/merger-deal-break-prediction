"""Train the extended model ladder and evaluate it.

Split: train on announcements 2001-2018, test on 2019-2025. Hyper-parameters are chosen by
expanding-window CV inside the training years. All fitted transforms are fitted in-fold
(see modeling.py). The test period is used for reporting only.

Runs (target x sample):
  primary     adverse break, core sample (reverse mergers excluded)
  any_core    any break, core sample
  any_spec    any break, sample exactly as specified (comparable with the base run)
  adv_spec    adverse break, sample as specified

Usage: python src/train_eval.py [--tag synthetic] [--quick]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import features as F
import market_context as MC
import modeling as M
from modeling import Feat

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
N_BOOT = 2000
EXT = ("deal_terms", "premium_price", "conditions", "financials", "advisors")
EXTM = EXT + ("market_context",)

LABELS = {
    "base_rate": "0. Base rate",
    "one_rule": "1. One rule",
    "logit_base": "2. Logistic, base tabular",
    "lgbm_base": "3. LightGBM, base tabular (as in the base run)",
    "lgbm_ext": "3x. LightGBM + extended deal, buyer and target features",
    "lgbm_extm": "3m. LightGBM + extended features + market context",
    "lgbm_kw": "3k. 3m + keyword flags",
    "lgbm_tfidf": "4. 3m + keywords + TF-IDF",
    "lgbm_emb": "5. 3m + keywords + frozen embeddings",
    "lgbm_llm": "5L. 3m + keywords + LLM-extracted terms",
    "lgbm_alltext": "5A. 3m + keywords + TF-IDF + embeddings + LLM terms",
    "tfidf_logit": "6a. Text only: TF-IDF logistic",
    "lgbm_text_only": "6b. Text only: keywords + embeddings + LLM terms",
    "ft_encoder": "6c. Text only: fine-tuned encoder",
    "fusion_ft": "6d. 3m fused with the fine-tuned encoder score",
    "lgbm_full_text": "5f. 3m + has_text + keywords + embeddings + LLM terms",
    "lgbm_suspects": "A1. Ablation: 3m + Deal Responses / Attitude (leakage suspects)",
    "lgbm_imputed": "A2. Ablation: 3m, median-imputed, no missingness signal",
}
SHORT = {k: v.split(".")[0] for k, v in LABELS.items()}
RUNS = {
    "primary": ("label_adverse", "core", "Adverse break, core sample (reverse mergers excluded)"),
    "any_core": ("label", "core", "Any break, core sample (reverse mergers excluded)"),
    "any_spec": ("label", "spec", "Any break, sample as specified (comparable with the base run)"),
    "adv_spec": ("label_adverse", "spec", "Adverse break, sample as specified"),
}


def specs(D, uy: bool, subset: str, reduced: bool) -> dict:
    """Model variants (all LightGBM) for a subset: 'full' sample or 'text' subset."""
    ext = EXT if D.extended else ()
    extm = ext + (("market_context",) if D.has_market else ())
    if not ext:   # base schema only (should not happen with the extended export)
        return {}
    S = {"lgbm_ext": Feat(groups=ext, year=uy), "lgbm_extm": Feat(groups=extm, year=uy)}
    have_llm, have_emb = D.L is not None, D.e_press is not None
    if subset == "text" and D.K is not None:
        if not reduced:
            S["lgbm_kw"] = Feat(groups=extm, year=uy, kw=True)
        S["lgbm_tfidf"] = Feat(groups=extm, year=uy, kw=True, tfidf=True)
        if have_emb:
            S["lgbm_emb"] = Feat(groups=extm, year=uy, kw=True, emb=True)
        if have_llm:
            S["lgbm_llm"] = Feat(groups=extm, year=uy, kw=True, llm=True)
        S["lgbm_alltext"] = Feat(groups=extm, year=uy, kw=True, tfidf=True, emb=have_emb, llm=have_llm)
        if not reduced:
            S["lgbm_text_only"] = Feat(kw=True, emb=have_emb, llm=have_llm)
    if subset == "full":
        if D.K is not None and D.has_text.any():
            S["lgbm_full_text"] = Feat(groups=extm, year=uy, has_text_flag=True, kw=True, emb=have_emb, llm=have_llm)
        if not reduced:
            S["lgbm_suspects"] = Feat(groups=extm, year=uy, suspects=True)
            S["lgbm_imputed"] = Feat(groups=extm, year=uy, impute=True)
    return S


def run_ladder(D, target: str, tr, te, subset: str, reduced: bool, log):
    y = D.y[target]
    folds = M.cv_folds(D.year, tr)
    P, I, G, O, feats = {}, {}, {}, {}, {}
    P["base_rate"] = np.full(len(te), y[tr].mean()); I["base_rate"] = {"train_rate": float(y[tr].mean())}
    P["one_rule"], rule, cv = M.one_rule(D, y, tr, te, folds); I["one_rule"] = {"rule": rule, "cv_auc_by_rule": cv}

    best = None
    for uy in (False, True):
        ft = Feat(base_only=True, year=uy)
        c, s, oi, op = M.cv_linear(y, folds, M.LOGIT_C, lambda a, b, ft=ft: M.design_logit(D, ft, a, b))
        if best is None or (s == s and s > best[1]):
            best = (c, s, oi, op, uy)
    c, s, oi, op, uy_l = best
    A, B, names = M.design_logit(D, Feat(base_only=True, year=uy_l), tr, te)
    m = LogisticRegression(C=c, max_iter=3000).fit(A, y[tr])
    P["logit_base"] = m.predict_proba(B)[:, 1]; O["logit_base"] = (oi, op)
    I["logit_base"] = {"C": c, "use_year": uy_l, "cv_auc": s}
    feats["logit_base"] = ("logit", Feat(base_only=True, year=uy_l), c)

    best = None
    for uy in (False, True):
        bp, s, oi, op = M.cv_lgbm(D, y, Feat(base_only=True, year=uy), folds)
        if best is None or (s == s and s > best[1]):
            best = (bp, s, oi, op, uy)
    bp, s, oi, op, uy = best
    P["lgbm_base"], G["lgbm_base"] = M.fit_lgbm(D, y, Feat(base_only=True, year=uy), tr, te, bp)
    O["lgbm_base"] = (oi, op); I["lgbm_base"] = {"num_leaves": bp[0], "n_estimators": bp[1], "use_year": uy, "cv_auc": s}
    feats["lgbm_base"] = ("lgbm", Feat(base_only=True, year=uy), bp)
    log(f"    logit C={c} cv {I['logit_base']['cv_auc']:.3f} | lgbm_base {bp} year={uy} cv {s:.3f}")

    for name, ft in specs(D, uy, subset, reduced).items():
        bp, s, oi, op = M.cv_lgbm(D, y, ft, folds)
        P[name], G[name] = M.fit_lgbm(D, y, ft, tr, te, bp)
        O[name] = (oi, op); I[name] = {"num_leaves": bp[0], "n_estimators": bp[1], "use_year": ft.year, "cv_auc": s}
        feats[name] = ("lgbm", ft, bp)
        log(f"    {name}: {bp} cv {s:.3f}")

    if subset == "text" and D.press is not None:
        c, s, oi, op = M.cv_linear(y, folds, M.TFIDF_LOGIT_C, lambda a, b: M.design_tfidf_sparse(D, a, b))
        A, B, names = M.design_tfidf_sparse(D, tr, te)
        m = LogisticRegression(C=c, max_iter=3000).fit(A, y[tr])
        P["tfidf_logit"] = m.predict_proba(B)[:, 1]; O["tfidf_logit"] = (oi, op)
        order = np.argsort(m.coef_[0])
        I["tfidf_logit"] = {"C": c, "cv_auc": s, "top_terms_break": [names[i] for i in order[::-1][:25]],
                            "top_terms_close": [names[i] for i in order[:25]]}
        # fine-tuned encoder: trained on <=2015, early-stopped on 2016-2018 (see finetune_encoder.py)
        if target in D.ft and np.isfinite(D.ft[target][te]).all() and "lgbm_extm" in O:
            P["ft_encoder"] = D.ft[target][te]; I["ft_encoder"] = {"note": "trained <=2015, early stopping on 2016-2018"}
            oi, op = O["lgbm_extm"]
            val = D.year[oi] >= 2016                        # rows where both scores are out-of-sample
            if val.sum() > 50 and y[oi][val].sum() >= 5:
                Z = np.column_stack([M.logit(op[val]), M.logit(D.ft[target][oi][val])])
                st = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y[oi][val])
                P["fusion_ft"] = st.predict_proba(np.column_stack([M.logit(P["lgbm_extm"]), M.logit(P["ft_encoder"])]))[:, 1]
                I["fusion_ft"] = {"stacker_coef": st.coef_[0].tolist(), "fitted_on": "2016-2018 validation years",
                                  "n": int(val.sum()), "positives": int(y[oi][val].sum())}
    for name, (oi, op) in O.items():
        if len(op):
            I[name]["oof_quantiles"] = {"q80": float(np.quantile(op, 0.8)), "q90": float(np.quantile(op, 0.9))}
    return P, I, G, O, feats


# ----------------------------------------------------------------------------- reporting helpers
def fmt_ci(v, ci, d=3):
    return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def table(y, preds, ci, order):
    rows = ["| Model | ROC AUC | PR AUC | Brier | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---|---|---|---|---|"]
    for m in order:
        if m in preds:
            mt = M.metrics(y, preds[m])
            rows.append(f"| {LABELS[m]} | {fmt_ci(mt['roc_auc'], ci[m]['roc_auc'])} | {fmt_ci(mt['pr_auc'], ci[m]['pr_auc'])} | "
                        f"{fmt_ci(mt['brier'], ci[m]['brier'], 4)} | {fmt_ci(mt['capture_top10'], ci[m]['capture_top10'], 2)} | "
                        f"{fmt_ci(mt['capture_top20'], ci[m]['capture_top20'], 2)} |")
    return rows


def diff_table(diffs):
    rows = ["| Comparison | ROC AUC | PR AUC | Brier (negative = better) | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---|---|---|---|---|"]
    for k, d in diffs.items():
        a, b = k.split(" minus ")
        rows.append(f"| {SHORT.get(a, a)} minus {SHORT.get(b, b)} | " + " | ".join(
            f"{d[m]['point']:+.{4 if m == 'brier' else 3}f} [{d[m]['ci95'][0]:+.{4 if m == 'brier' else 3}f}, {d[m]['ci95'][1]:+.{4 if m == 'brier' else 3}f}]"
            for m in M.METRIC_KEYS) + " |")
    return rows


ORDER_TEXT = ["base_rate", "one_rule", "logit_base", "lgbm_base", "lgbm_ext", "lgbm_extm", "lgbm_kw", "lgbm_tfidf", "lgbm_emb",
              "lgbm_llm", "lgbm_alltext", "tfidf_logit", "lgbm_text_only", "ft_encoder", "fusion_ft"]
ORDER_FULL = ["base_rate", "one_rule", "logit_base", "lgbm_base", "lgbm_ext", "lgbm_extm", "lgbm_full_text", "lgbm_suspects", "lgbm_imputed"]
PAIRS_TEXT = [("lgbm_ext", "lgbm_base"), ("lgbm_extm", "lgbm_ext"), ("lgbm_tfidf", "lgbm_extm"), ("lgbm_emb", "lgbm_extm"),
              ("lgbm_llm", "lgbm_extm"), ("lgbm_alltext", "lgbm_extm"), ("fusion_ft", "lgbm_extm"), ("ft_encoder", "tfidf_logit"),
              ("lgbm_alltext", "lgbm_base")]
PAIRS_FULL = [("lgbm_base", "logit_base"), ("lgbm_ext", "lgbm_base"), ("lgbm_extm", "lgbm_ext"), ("lgbm_extm", "lgbm_base"),
              ("lgbm_full_text", "lgbm_extm"), ("lgbm_suspects", "lgbm_extm"), ("lgbm_imputed", "lgbm_extm")]

FEATURE_TEXT = {
    "premium_1d": "offer premium over the prior day's price", "premium_1w": "offer premium over the price a week earlier",
    "premium_1m": "offer premium over the price a month earlier", "pct_52w_high": "target price as % of its 52-week high",
    "pct_52w_low": "target price as % of its 52-week low", "target_ltm_return": "target's 12-month return before the deal",
    "log_deal_value": "deal size", "log_target_mktcap": "target market cap", "log_implied_equity_value": "implied equity value",
    "log_implied_ev": "implied enterprise value", "value_to_mktcap": "deal value relative to target market cap",
    "cash_pct": "cash share of the consideration", "stock_pct": "stock share of the consideration", "solicited": "solicited approach (target ran a process)",
    "buyer_listed": "buyer is a listed company", "buyer_financial": "financial buyer (private equity)", "feat_lbo": "leveraged buyout",
    "feat_going_private": "going-private transaction", "feat_go_shop": "go-shop provision", "feat_tender_offer": "tender offer structure",
    "feat_cross_border": "cross-border deal", "feat_management_buyout": "management buyout", "feat_management_participated": "management participates",
    "feat_merger_of_equals": "merger of equals", "feat_cash_merger": "cash merger", "feat_stock_merger": "stock merger",
    "sell_fee_pct_value": "target termination fee as % of deal value", "buy_fee_pct_value": "buyer (reverse) termination fee as % of deal value",
    "has_buy_side_fee": "a buyer termination fee is recorded", "log_sell_fee_usd": "target termination fee (USD)", "log_buy_fee_usd": "buyer termination fee (USD)",
    "rel_size_deal_to_acq": "deal value relative to acquirer market cap", "log_acq_mktcap": "acquirer market cap", "acq_has_mktcap": "acquirer has a market cap (is listed)",
    "acq_cash_to_deal_value": "acquirer cash relative to deal value", "acq_leverage": "acquirer leverage", "acq_ltm_return": "acquirer's 12-month return",
    "acq_pct_52w_high": "acquirer price as % of its 52-week high", "tgt_leverage": "target leverage", "tgt_roa": "target return on assets",
    "tgt_loss_making": "target is loss-making", "ev_ebitda": "EV / EBITDA multiple", "ev_revenue": "EV / revenue multiple", "pb_multiple": "price / book multiple",
    "pe_multiple": "price / earnings multiple", "same_sector": "buyer and target in the same sector", "mkt_vix": "VIX level before announcement",
    "mkt_vix_chg_30d": "30-day change in the VIX", "mkt_baa_aaa_spread": "Baa minus Aaa credit spread", "mkt_baa_10y_spread": "Baa minus 10-year Treasury spread",
    "mkt_ust10y": "10-year Treasury yield", "mkt_equity_ret_3m": "equity market 3-month return", "mkt_equity_ret_12m": "equity market 12-month return",
    "mkt_deals_90d": "number of deals announced in the prior 90 days", "mkt_break_rate_12m": "break rate of deals resolved in the prior 12 months",
    "days_signed_before_announcement": "days between signing and announcement", "signed_after_announcement": "not yet signed at announcement",
    "cond_financial": "financing-related condition recorded", "n_conditions": "number of deal conditions recorded", "conditions_missing": "no deal conditions recorded",
    "buyer_count": "number of buyers", "year": "announcement year", "n_fin_adv_target": "number of target financial advisers",
    "n_legal_adv_buyer": "number of buyer legal advisers", "log_offer_per_share": "offer price per share", "pct_sought": "percent of shares sought",
    "cash_share_of_value": "cash consideration as a share of deal value", "log_tgt_revenue": "target revenue", "log_tgt_assets": "target total assets",
}


def nice(name: str) -> str:
    return FEATURE_TEXT.get(name, name.replace("_", " "))


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--quick", action="store_true", help="primary run only, fewer bootstrap draws (debugging)")
    args = ap.parse_args()
    synthetic = args.tag != "real"
    suffix = "" if not synthetic else f"_{args.tag}"
    res = ROOT / ("results" if not synthetic else f"results_{args.tag}")
    fig = ROOT / ("figures" if not synthetic else f"figures_{args.tag}")
    res.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)
    n_boot = 300 if args.quick else N_BOOT
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    D = M.load_data(suffix)
    idx = np.arange(len(D.year))
    log(f"{len(idx)} deals; extended={D.extended} market={D.has_market} llm={D.L is not None} ft={list(D.ft)} prices={D.PF is not None}; "
        f"core sample {int(D.core.sum())}, text {int(D.has_text.sum())}")

    out = {"tag": args.tag, "n_bootstrap": n_boot, "runs": {}, "sample": {
        "n_spec": int(len(idx)), "n_core": int(D.core.sum()), "reverse_mergers": int((~D.core).sum()),
        "reverse_mergers_test": int(((~D.core) & (D.year >= M.TEST_START)).sum()),
        "reverse_merger_breaks_test": int((D.y["label"][(~D.core) & (D.year >= M.TEST_START)]).sum())}}
    pred = pd.DataFrame({"deal_id": D.deals["deal_id"], "year": D.year, "core": D.core, "has_text": D.has_text,
                         "label": D.y["label"], "label_adverse": D.y["label_adverse"], "outcome3": D.outcome3})
    keep, cols = {}, {}
    run_keys = ["primary"] if args.quick else list(RUNS)
    for rk in run_keys:
        target, sample, title = RUNS[rk]
        y = D.y[target]
        mask = D.core if sample == "core" else np.ones(len(idx), bool)
        R = {"target": target, "sample": sample, "title": title}
        for subset in ("full", "text"):
            sm = mask & (D.has_text if subset == "text" else True)
            tr, te = idx[sm & (D.year <= M.TRAIN_END)], idx[sm & (D.year >= M.TEST_START)]
            if len(tr) < 60 or y[te].sum() < 2 or y[tr].sum() < 8:
                continue
            log(f"{rk} / {subset}: train {len(tr)} ({int(y[tr].sum())} pos), test {len(te)} ({int(y[te].sum())} pos)")
            P, I, G, O, feats = run_ladder(D, target, tr, te, subset, reduced=(rk != "primary"), log=log)
            ci, diffs = M.bootstrap(y[te], P, PAIRS_FULL if subset == "full" else PAIRS_TEXT, n_boot=n_boot)
            R[subset] = {"n_train": int(len(tr)), "pos_train": int(y[tr].sum()), "n_test": int(len(te)), "pos_test": int(y[te].sum()),
                         "metrics": {m: M.metrics(y[te], p) for m, p in P.items()}, "ci95": ci, "paired_differences": diffs, "selection": I}
            keep[(rk, subset)] = dict(P=P, I=I, G=G, O=O, feats=feats, tr=tr, te=te, ci=ci, diffs=diffs)
            for m, p in P.items():
                col = np.full(len(idx), np.nan); col[te] = p; cols[f"pred__{rk}__{subset}__{m}"] = col
            for m, (oi, op) in O.items():
                if len(op):
                    col = np.full(len(idx), np.nan); col[oi] = op; cols[f"oof__{rk}__{subset}__{m}"] = col
        out["runs"][rk] = R
    pred = pd.concat([pred, pd.DataFrame(cols, index=pred.index)], axis=1)

    # ------------------------------------------------------------------ final model choice (CV only, never the test set)
    pf = keep[("primary", "full")]
    cands = [m for m in ("lgbm_base", "lgbm_ext", "lgbm_extm") if m in pf["I"]]
    final_tab = max(cands, key=lambda m: pf["I"][m]["cv_auc"])
    cands2 = cands + [m for m in ("lgbm_full_text",) if m in pf["I"]]
    final = max(cands2, key=lambda m: pf["I"][m]["cv_auc"])
    out["final_models"] = {"tabular": final_tab, "overall": final, "rule": "highest CV AUC on the training years (full sample, primary run)",
                           "cv_auc": {m: pf["I"][m]["cv_auc"] for m in cands2}}
    log(f"final tabular model: {final_tab}; final overall: {final}")
    y = D.y["label_adverse"]; tr, te = pf["tr"], pf["te"]

    # ------------------------------------------------------------------ calibration (fitted on validation predictions only)
    cal = {}
    for m in dict.fromkeys([final_tab, final, "logit_base"]):
        oi, op = pf["O"][m]
        f = M.platt(op, y[oi])
        raw, adj = pf["P"][m], f(pf["P"][m])
        cc = np.full(len(idx), np.nan); cc[te] = adj
        pred[f"cal__primary__full__{m}"] = cc

        def ece(p):
            q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False).to_numpy()
            return float(sum(abs(p[q == b].mean() - y[te][q == b].mean()) * (q == b).mean() for b in range(5)))
        cal[m] = {"brier_raw": M.metrics(y[te], raw)["brier"], "brier_calibrated": M.metrics(y[te], adj)["brier"],
                  "mean_pred_raw": float(raw.mean()), "mean_pred_calibrated": float(adj.mean()), "observed_rate": float(y[te].mean()),
                  "ece5_raw": ece(raw), "ece5_calibrated": ece(adj), "n_validation_predictions": int(len(op)),
                  "validation_positives": int(y[oi].sum())}
        keep[("cal", m)] = (raw, adj, f)
    out["calibration"] = cal

    extras = {}
    if not args.quick:
        # -------------------------------------------------------------- ablation by feature group (text subset)
        if ("primary", "text") in keep and "lgbm_alltext" in keep[("primary", "text")]["feats"]:
            kt = keep[("primary", "text")]
            _, full_feat, bp = kt["feats"]["lgbm_alltext"]
            ttr, tte = kt["tr"], kt["te"]
            groups = {"deal_terms": dict(g=("deal_terms", "advisors")), "premium_price": dict(g=("premium_price",)), "conditions": dict(g=("conditions",)),
                      "financials": dict(g=("financials",)), "market_context": dict(g=("market_context",)), "kw": dict(kw=True),
                      "tfidf": dict(tfidf=True), "emb": dict(emb=True), "llm_terms": dict(llm=True)}
            AP = {"all groups": kt["P"]["lgbm_alltext"]}
            for gname, spec in groups.items():
                drop_groups = tuple(g for g in full_feat.groups if g not in spec.get("g", ()))
                f_drop = Feat(groups=drop_groups, year=full_feat.year, kw=full_feat.kw and not spec.get("kw"), tfidf=full_feat.tfidf and not spec.get("tfidf"),
                              emb=full_feat.emb and not spec.get("emb"), llm=full_feat.llm and not spec.get("llm"))
                f_only = Feat(groups=tuple(g for g in full_feat.groups if g in spec.get("g", ())), kw=bool(spec.get("kw")) and full_feat.kw,
                              tfidf=bool(spec.get("tfidf")) and full_feat.tfidf, emb=bool(spec.get("emb")) and full_feat.emb, llm=bool(spec.get("llm")) and full_feat.llm)
                if f_drop == full_feat:
                    continue
                AP[f"without {gname}"] = M.fit_lgbm(D, y, f_drop, ttr, tte, bp)[0]
                AP[f"only {gname}"] = M.fit_lgbm(D, y, f_only, ttr, tte, bp)[0]
                log(f"ablation {gname} done")
            aci, adiff = M.bootstrap(y[tte], AP, [("all groups", k) for k in AP if k.startswith("without")], n_boot=n_boot)
            extras["ablation"] = {"metrics": {k: M.metrics(y[tte], p) for k, p in AP.items()}, "ci95": aci, "drop_one": adiff,
                                  "params": list(bp), "n_test": int(len(tte)), "pos_test": int(y[tte].sum())}

        # -------------------------------------------------------------- rolling-origin evaluation (full core sample)
        ro = []
        ro_models = [m for m in ("logit_base", "lgbm_base", "lgbm_extm", "lgbm_full_text") if m in pf["feats"]]
        core_idx = idx[D.core]
        for T in range(2012, 2024):
            rtr, rte = core_idx[D.year[core_idx] <= T], core_idx[(D.year[core_idx] > T) & (D.year[core_idx] <= T + 2)]
            if y[rte].sum() < 3 or y[rtr].sum() < 20:
                continue
            row = {"T": T, "test_years": f"{T + 1}-{min(T + 2, 2025)}", "n_train": int(len(rtr)), "n_test": int(len(rte)), "pos_test": int(y[rte].sum())}
            for m in ro_models:
                kind, ft, par = pf["feats"][m]
                if kind == "logit":
                    A, B, _ = M.design_logit(D, ft, rtr, rte)
                    p = LogisticRegression(C=par, max_iter=3000).fit(A, y[rtr]).predict_proba(B)[:, 1]
                else:
                    p = M.fit_lgbm(D, y, ft, rtr, rte, par)[0]
                row[m] = M.metrics(y[rte], p)
            ro.append(row); log(f"rolling origin T={T} done")
        extras["rolling_origin"] = ro

        # -------------------------------------------------------------- SHAP for the final tabular model
        kind, ft, bp = pf["feats"][final_tab]
        p, gains, models, A, B, names = M.fit_lgbm(D, y, ft, tr, te, bp, return_models=True)
        contrib = np.mean([m.booster_.predict(B, pred_contrib=True) for m in models], axis=0)[:, :-1]
        imp = np.abs(contrib).mean(0)
        top = np.argsort(-imp)[:15]
        shap_rows = []
        for j in top:
            v = B[:, j]; ok = np.isfinite(v)
            rho = float(pd.Series(v[ok]).corr(pd.Series(contrib[ok, j]), method="spearman")) if ok.sum() > 20 and np.nanstd(v[ok]) > 0 else float("nan")
            shap_rows.append({"feature": names[j], "mean_abs_shap": float(imp[j]), "direction_rho": rho, "share_missing": float(1 - ok.mean())})
        extras["shap"] = {"model": final_tab, "rows": shap_rows, "total_mean_abs": float(imp.sum())}
        keep["shap"] = (contrib, B, names, top)
        # artefacts for the demo (git-ignored)
        art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
        with open(art / "final_tabular_shap.pkl", "wb") as fh:
            pickle.dump({"deal_id": D.deals["deal_id"].to_numpy()[te], "contrib": contrib.astype(np.float32), "names": names,
                         "values": B.astype(np.float32)}, fh)

        # -------------------------------------------------------------- three-class view
        cls_map = {"closed": 0, "topped": 1, "adverse_break": 2}
        y3 = np.array([cls_map.get(c, 0) for c in D.outcome3])
        if (y3[tr] == 1).sum() >= 10 and (y3[te] == 1).sum() >= 3:
            A3, B3, _ = M.design_tree(D, ft, tr, te)
            p3 = np.zeros((len(te), 3))
            for seed in M.SEEDS:
                kw = {k: v for k, v in M.LGBM_BASE.items() if k != "objective"}
                p3 += LGBMClassifier(objective="multiclass", num_leaves=bp[0], n_estimators=bp[1], random_state=seed, **kw).fit(A3, y3[tr]).predict_proba(B3) / len(M.SEEDS)
            rng = np.random.default_rng(0)
            tc = {"counts_train": {k: int((y3[tr] == v).sum()) for k, v in cls_map.items()}, "counts_test": {k: int((y3[te] == v).sum()) for k, v in cls_map.items()}, "ovr_auc": {}}
            for k, v in cls_map.items():
                yb = (y3[te] == v).astype(int)
                bs = []
                for _ in range(1000):
                    ix = rng.integers(0, len(te), len(te))
                    if 0 < yb[ix].sum() < len(ix):
                        bs.append(roc_auc_score(yb[ix], p3[ix, v]))
                tc["ovr_auc"][k] = {"auc": float(roc_auc_score(yb, p3[:, v])), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            tc["mean_binary_score_by_class"] = {k: float(pf["P"][final_tab][y3[te] == v].mean()) for k, v in cls_map.items()}
            tc["mean_p3_by_class"] = {k: [float(x) for x in p3[y3[te] == v].mean(0)] for k, v in cls_map.items()}
            extras["three_class"] = tc
            for j, k in enumerate(["p3_closed", "p3_topped", "p3_adverse"]):
                pred[k] = np.nan; pred.loc[te, k] = p3[:, j]

        # -------------------------------------------------------------- robustness
        rb = {}
        kind, ftf, bpf = pf["feats"][final]
        base_m = M.metrics(y[te], pf["P"][final])
        rb["baseline"] = {"n_test": int(len(te)), "pos_test": int(y[te].sum()), **base_m}
        k10 = D.days_out[te] > 10
        rb["exclude_resolved_within_10_days (evaluation only)"] = {"n_test": int(k10.sum()), "pos_test": int(y[te][k10].sum()), **M.metrics(y[te][k10], pf["P"][final][k10])}
        nr_tr, nr_te = tr[~D.revised[tr]], te[~D.revised[te]]
        p_nr = M.fit_lgbm(D, y, ftf, nr_tr, nr_te, bpf)[0]
        rb["exclude_revised_offers (train and test)"] = {"n_test": int(len(nr_te)), "pos_test": int(y[nr_te].sum()), **M.metrics(y[nr_te], p_nr)}
        kr = ~D.revised[te]
        rb["exclude_revised_offers (evaluation only)"] = {"n_test": int(kr.sum()), "pos_test": int(y[te][kr].sum()), **M.metrics(y[te][kr], pf["P"][final][kr])}
        if "lgbm_imputed" in pf["P"]:
            rb["no missingness signal (A2, median-imputed 3m)"] = {"n_test": int(len(te)), "pos_test": int(y[te].sum()), **M.metrics(y[te], pf["P"]["lgbm_imputed"])}
        if "lgbm_suspects" in pf["P"]:
            rb["with leakage suspects (A1)"] = {"n_test": int(len(te)), "pos_test": int(y[te].sum()), **M.metrics(y[te], pf["P"]["lgbm_suspects"])}
        if ("adv_spec", "full") in keep and final in keep[("adv_spec", "full")]["P"]:
            ks = keep[("adv_spec", "full")]
            rb["sample as specified, reverse mergers included"] = {"n_test": int(len(ks["te"])), "pos_test": int(y[ks["te"]].sum()), **M.metrics(y[ks["te"]], ks["P"][final])}
            rev = ~D.core[ks["te"]]
            if y[ks["te"]][rev].sum() >= 3:
                rb["reverse mergers only (model trained on the specified sample)"] = {"n_test": int(rev.sum()), "pos_test": int(y[ks["te"]][rev].sum()),
                                                                                       **M.metrics(y[ks["te"]][rev], ks["P"][final][rev])}
        extras["robustness"] = {"model": final, "rows": rb}
    out["extras"] = extras
    (res / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    pred.to_parquet(INTERIM / f"predictions{suffix}.parquet", index=False)
    with open(INTERIM / f"model_specs{suffix}.pkl", "wb") as fh:
        pickle.dump({k: {"feats": v["feats"], "tr": v["tr"], "te": v["te"]} for k, v in keep.items() if isinstance(k, tuple) and k[0] in RUNS}, fh)

    write_reports(D, out, keep, res, synthetic, final_tab, final)
    make_figures(D, out, keep, fig, synthetic, final_tab, final)
    log("done")


def write_reports(D, out, keep, res, synthetic, final_tab, final):
    tag = " (SYNTHETIC SAMPLE: not real results)" if synthetic else ""
    s = out["sample"]
    L = ["# Benchmark table" + tag, "",
         f"Train: announcements 2001-{M.TRAIN_END}. Test: {M.TEST_START}-2025. Brackets are bootstrap 95% confidence intervals "
         f"({out['n_bootstrap']} resamples of test deals). Hyper-parameters are chosen by expanding-window CV inside the training years.", "",
         f"Samples: the sample as specified has {s['n_spec']:,} deals. {s['reverse_mergers']} of them are reverse mergers (de-SPACs and shell mergers), "
         f"{s['reverse_mergers_test']} of which fall in the test period and account for {s['reverse_merger_breaks_test']} test-period breaks; the training period has almost none. "
         f"The **core sample** ({s['n_core']:,} deals) excludes them and is the primary sample. Targets: **adverse break** = cancelled for a reason other than a superior "
         "competing bid (primary); **any break** = cancelled.", ""]
    for rk, R in out["runs"].items():
        L += [f"## {R['title']}" + (" (PRIMARY)" if rk == "primary" else ""), ""]
        for subset, title in (("text", "Text subset: deals with an announcement text"), ("full", "Full sample")):
            if subset not in R:
                continue
            S = R[subset]; k = keep[(rk, subset)]
            y = D.y[R["target"]][k["te"]]
            L += [f"### {title} (train {S['n_train']:,} deals / {S['pos_train']} positives, test {S['n_test']:,} deals / {S['pos_test']} positives)", ""]
            L += table(y, k["P"], k["ci"], ORDER_TEXT if subset == "text" else ORDER_FULL) + [""]
            if rk in ("primary", "any_spec"):
                L += ["Paired differences (same bootstrap resamples):", ""] + diff_table(k["diffs"]) + [""]
            L += [f"One-rule benchmark chosen by CV: \"{S['selection']['one_rule']['rule']}\".", ""]
    L += [f"Final models (chosen by CV AUC on the training years, primary run, full sample): tabular = {LABELS[final_tab]}; overall = {LABELS[final]}.", ""]
    (res / "benchmark_table.md").write_text("\n".join(L) + "\n")

    ex = out.get("extras", {})
    # ---- calibration
    C = ["# Calibration" + tag, "",
         "Platt scaling (a logistic fit on the logit of the score) is fitted on the training-period out-of-fold predictions only "
         "(validation blocks 2007-2018) and then applied to the test predictions. Target: adverse break, core sample.", "",
         "| Model | Validation predictions (positives) | Mean predicted, raw | Mean predicted, calibrated | Observed test rate | Brier raw | Brier calibrated | ECE (5 bins) raw | ECE calibrated |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, c in out.get("calibration", {}).items():
        C.append(f"| {LABELS[m]} | {c['n_validation_predictions']} ({c['validation_positives']}) | {c['mean_pred_raw']:.3f} | {c['mean_pred_calibrated']:.3f} | "
                 f"{c['observed_rate']:.3f} | {c['brier_raw']:.4f} | {c['brier_calibrated']:.4f} | {c['ece5_raw']:.3f} | {c['ece5_calibrated']:.3f} |")
    C += ["", "See `figures/calibration.png` for the reliability plot (quintile bins with 95% intervals).", ""]
    (res / "calibration.md").write_text("\n".join(C) + "\n")
    if not ex:
        return
    # ---- ablation
    if "ablation" in ex:
        a = ex["ablation"]
        A = ["# Ablation by feature group" + tag, "",
             f"Text subset, core sample, adverse-break target (test: {a['n_test']} deals, {a['pos_test']} adverse breaks). The reference is LightGBM on all groups "
             f"(model 5A) with its CV-selected hyper-parameters (num_leaves {a['params'][0]}, {a['params'][1]} trees), kept fixed for every row. "
             "\"Without\" drops one group from the full model; \"only\" trains on that group alone. Brackets: bootstrap 95% intervals.", "",
             "| Feature group | AUC of the group alone | AUC without the group | Loss in AUC when dropped (all minus without) |", "|---|---|---|---|"]
        allm = a["metrics"]["all groups"]["roc_auc"]
        for g in ["deal_terms", "premium_price", "conditions", "financials", "market_context", "kw", "tfidf", "emb", "llm_terms"]:
            if f"without {g}" not in a["metrics"]:
                continue
            d = a["drop_one"][f"all groups minus without {g}"]["roc_auc"]
            A.append(f"| {F.GROUP_LABELS[g]} | {fmt_ci(a['metrics'][f'only {g}']['roc_auc'], a['ci95'][f'only {g}']['roc_auc'])} | "
                     f"{fmt_ci(a['metrics'][f'without {g}']['roc_auc'], a['ci95'][f'without {g}']['roc_auc'])} | {d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
        A += ["", f"All groups together: AUC {fmt_ci(allm, a['ci95']['all groups']['roc_auc'])}.",
              "The market-spread group is dated t1 (up to a month after announcement) and is evaluated separately on the price-covered subset in `results/market_benchmark.md`.", ""]
        (res / "ablation.md").write_text("\n".join(A) + "\n")
    # ---- rolling origin
    if ex.get("rolling_origin"):
        ro = ex["rolling_origin"]; ms = [m for m in ("logit_base", "lgbm_base", "lgbm_extm", "lgbm_full_text") if m in ro[0]]
        Rr = ["# Rolling-origin evaluation" + tag, "",
              "Train on announcements through year T, test on T+1 to T+2, for T = 2012 to 2023. Core sample, adverse-break target, full sample "
              "(text features missing where a deal has no text). Hyper-parameters are those selected in the main run (training years 2001-2018) and are held "
              "fixed across origins, so origins before 2018 carry a mild look-ahead in the hyper-parameters only; all model fitting uses data up to T.", "",
              "| Train through | Test years | Test deals | Adverse breaks | " + " | ".join(f"AUC {SHORT[m]}" for m in ms) + " | " + " | ".join(f"Top-20% capture {SHORT[m]}" for m in ms) + " |",
              "|---|---|---:|---:|" + "---:|" * (2 * len(ms))]
        for r in ro:
            Rr.append(f"| {r['T']} | {r['test_years']} | {r['n_test']} | {r['pos_test']} | " + " | ".join(f"{r[m]['roc_auc']:.3f}" for m in ms) + " | "
                      + " | ".join(f"{r[m]['capture_top20']:.2f}" for m in ms) + " |")
        Rr.append("| **Mean** | | | | " + " | ".join(f"**{np.mean([r[m]['roc_auc'] for r in ro]):.3f}**" for m in ms) + " | "
                  + " | ".join(f"**{np.mean([r[m]['capture_top20'] for r in ro]):.2f}**" for m in ms) + " |")
        Rr.append("| Std. dev. across origins | | | | " + " | ".join(f"{np.std([r[m]['roc_auc'] for r in ro]):.3f}" for m in ms) + " | "
                  + " | ".join(f"{np.std([r[m]['capture_top20'] for r in ro]):.2f}" for m in ms) + " |")
        wins = {m: sum(r[m]["roc_auc"] > r["logit_base"]["roc_auc"] for r in ro) for m in ms if m != "logit_base"}
        Rr += ["", "Origins overlap (each test year appears in two windows), so the rows are not independent. "
               + "; ".join(f"{SHORT[m]} beats the logistic benchmark in {w} of {len(ro)} origins" for m, w in wins.items()) + ".",
               "Model keys: " + "; ".join(f"{SHORT[m]} = {LABELS[m].split('. ', 1)[1]}" for m in ms) + ".", ""]
        (res / "rolling_origin.md").write_text("\n".join(Rr) + "\n")
    # ---- SHAP
    if "shap" in ex:
        sh = ex["shap"]
        S_ = ["# SHAP reading of the tabular model" + tag, "",
              f"Model: {LABELS[sh['model']]}, trained on 2001-2018 (core sample, adverse-break target). TreeSHAP values on the test deals, averaged over the five seeds. "
              "Mean |SHAP| is in log-odds units. Direction is the rank correlation between the feature value and its SHAP value on the test deals: "
              "positive means higher values push the predicted break risk up.", "",
              "| Rank | Feature | Mean abs. SHAP | Share of total | Direction | Plain reading |", "|---:|---|---:|---:|---|---|"]
        for i, r in enumerate(sh["rows"], 1):
            rho = r["direction_rho"]
            if rho != rho:
                direction, reading = "n/a", f"{nice(r['feature'])}: effect not monotone or feature mostly missing"
            elif abs(rho) < 0.25:
                direction, reading = "mixed", f"{nice(r['feature'])}: matters, but not in one direction (interaction or U-shape)"
            else:
                direction = "higher = riskier" if rho > 0 else "higher = safer"
                if r["feature"].startswith(("feat_", "buyer_", "has_", "cond_", "tgt_loss", "solicited", "same_sector", "signed_after")):
                    reading = f"Deals with this flag ({nice(r['feature'])}) get a {'higher' if rho > 0 else 'lower'} predicted break risk"
                else:
                    reading = f"{'More / higher' if rho > 0 else 'Less / lower'} {nice(r['feature'])} goes with higher predicted break risk"
            S_.append(f"| {i} | `{r['feature']}` | {r['mean_abs_shap']:.3f} | {r['mean_abs_shap'] / sh['total_mean_abs']:.0%} | {direction} (rho {rho:+.2f}) | {reading} |")
        S_ += ["", "These are associations the model uses, not causal effects. Features with many missing values also carry a missingness signal "
               "(see `results/robustness.md`). Figure: `figures/shap_summary.png`.", ""]
        (res / "shap.md").write_text("\n".join(S_) + "\n")
    # ---- three-class
    if "three_class" in ex:
        tc = ex["three_class"]
        T_ = ["# Three-class view: closes, topped by a higher bid, adverse break" + tag, "",
              "A multiclass LightGBM with the features and hyper-parameters of the final tabular model, trained on 2001-2018 (core sample).", "",
              "| Class | Train deals | Test deals | One-vs-rest AUC on test [95% CI] | Mean binary adverse-break score |", "|---|---:|---:|---|---:|"]
        for k, lab in (("closed", "Closes"), ("topped", "Topped by a higher bid"), ("adverse_break", "Adverse break")):
            a = tc["ovr_auc"][k]
            T_.append(f"| {lab} | {tc['counts_train'][k]} | {tc['counts_test'][k]} | {a['auc']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {tc['mean_binary_score_by_class'][k]:.3f} |")
        T_ += ["", "Mean predicted class probabilities by true class (rows: true class; columns: P(closes), P(topped), P(adverse)):", "",
               "| True class | P(closes) | P(topped) | P(adverse) |", "|---|---:|---:|---:|"]
        for k, lab in (("closed", "Closes"), ("topped", "Topped"), ("adverse_break", "Adverse break")):
            T_.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in tc["mean_p3_by_class"][k]) + " |")
        T_ += ["", "Topped deals are good outcomes for an arbitrageur (the price usually rises), which is why the adverse-break target excludes them. "
               "The number of topped deals in the test period is small, so its AUC interval is wide.", ""]
        (res / "three_class.md").write_text("\n".join(T_) + "\n")
    # ---- robustness
    if "robustness" in ex:
        rb = ex["robustness"]
        Rb = ["# Robustness checks" + tag, "", f"Model: {LABELS[rb['model']]}, adverse-break target, test period 2019-2025.", "",
              "| Variant | Test deals | Adverse breaks | ROC AUC | PR AUC | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k, r in rb["rows"].items():
            Rb.append(f"| {k} | {r['n_test']} | {r['pos_test']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['capture_top10']:.2f} | {r['capture_top20']:.2f} |")
        Rb += ["", "- *Revised offers*: deals whose Capital IQ notes mention a raised, lowered or amended offer after announcement. For these the exported price, premium and "
               "value can be final terms. Excluding them removes that possible look-ahead.",
               "- *Resolved within 10 days*: deals that closed or were cancelled almost immediately, where the text window could overlap the outcome.",
               "- *No missingness signal*: Capital IQ fills fewer fields for cancelled deals, so missing values are themselves predictive. A2 removes that signal.",
               "- *Leakage suspects*: Deal Responses and Deal Attitude can be updated after announcement; A1 shows what they would add.", ""]
        (res / "robustness.md").write_text("\n".join(Rb) + "\n")
    # ---- segments
    if ("primary", "full") in keep:
        k = keep[("primary", "full")]; te = k["te"]; y = D.y["label_adverse"][te]
        Xte = D.X.iloc[te]
        seg_cs = F.cash_stock_segment(Xte).reset_index(drop=True)
        dv = D.deals["deal_value"].to_numpy()
        cuts = np.quantile(dv[k["tr"]], [1 / 3, 2 / 3])
        seg_size = pd.Series(np.select([dv[te] <= cuts[0], dv[te] <= cuts[1]], [f"small (<= ${cuts[0]:,.0f}m)", f"mid (<= ${cuts[1]:,.0f}m)"], f"large (> ${cuts[1]:,.0f}m)"))
        yr = D.year[te]
        seg_per = pd.Series(np.select([yr <= 2020, yr <= 2022], ["2019-2020", "2021-2022"], "2023-2025"))
        fb = D.X["buyer_financial"].iloc[te].to_numpy() if "buyer_financial" in D.X else np.zeros(len(te))
        seg_buyer = pd.Series(np.where(fb == 1, "financial buyer", "strategic / other"))
        ms = [m for m in ("logit_base", "lgbm_base", final_tab, final) if m in k["P"]]
        ms = list(dict.fromkeys(ms))
        SL = ["# Segment results on the test set" + tag, "",
              "Core sample, adverse-break target. AUC and capture are computed inside each segment; segments with fewer than 5 adverse breaks show n/a. "
              "Segment numbers rest on few breaks and are noisy. Deal-size terciles use cut-offs from the training years.", ""]
        for title, seg in (("consideration", seg_cs), ("deal size tercile", seg_size), ("period", seg_per), ("buyer type", seg_buyer)):
            SL += [f"## By {title}", "", "| Segment | Deals | Adverse breaks | Rate | " + " | ".join(f"AUC {SHORT[m]}" for m in ms) + " | "
                   + " | ".join(f"Top-20% capture {SHORT[m]}" for m in ms) + " |", "|---|---:|---:|---:|" + "---:|" * (2 * len(ms))]
            for sname in list(pd.unique(seg)):
                mk = (seg == sname).to_numpy(); yy = y[mk]
                aucs = [f"{M.auc_safe(yy, k['P'][m][mk]):.3f}" if yy.sum() >= 5 else "n/a" for m in ms]
                caps = [f"{M.capture_at(yy, k['P'][m][mk], 0.2):.2f}" if yy.sum() >= 5 else "n/a" for m in ms]
                SL.append(f"| {sname} | {mk.sum()} | {int(yy.sum())} | {yy.mean():.1%} | " + " | ".join(aucs) + " | " + " | ".join(caps) + " |")
            SL.append("")
        SL += ["Model keys: " + "; ".join(f"{SHORT[m]} = {LABELS[m].split('. ', 1)[1]}" for m in ms) + ".", ""]
        (res / "segments.md").write_text("\n".join(SL) + "\n")


def make_figures(D, out, keep, fig, synthetic, final_tab, final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})
    note = " (synthetic sample)" if synthetic else ""
    col = {"one_rule": "#bdbdbd", "logit_base": "#6baed6", "lgbm_base": "#3182bd", "lgbm_extm": "#08306b", "lgbm_tfidf": "#fdae6b",
           "lgbm_emb": "#f16913", "lgbm_llm": "#a63603", "lgbm_alltext": "#7f2704", "tfidf_logit": "#74c476", "ft_encoder": "#238b45",
           "lgbm_full_text": "#f16913", "lgbm_ext": "#6a51a3"}
    short = {"one_rule": "1 one rule", "logit_base": "2 logistic, base tabular", "lgbm_base": "3 LightGBM, base tabular", "lgbm_ext": "3x + extended features",
             "lgbm_extm": "3m + extended + market", "lgbm_tfidf": "4 + TF-IDF", "lgbm_emb": "5 + embeddings", "lgbm_llm": "5L + LLM terms",
             "lgbm_alltext": "5A + all text", "tfidf_logit": "6a text only (TF-IDF)", "ft_encoder": "6c text only (fine-tuned)", "lgbm_full_text": "5f + text (full sample)"}

    # 1. deals and break rates by year
    g = D.deals.assign(core=D.core, adv=D.y["label_adverse"]).groupby("year").agg(n=("label", "size"), any=("label", "mean"))
    gc = D.deals.assign(adv=D.y["label_adverse"])[D.core].groupby("year").agg(n=("label", "size"), adv=("adv", "mean"), any=("label", "mean"))
    f_, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(g.index, g["n"], color="#deebf7", label="Deals, sample as specified")
    ax.bar(gc.index, gc["n"], color="#9ecae1", label="Deals, core sample")
    ax.set_ylabel("Deals announced"); ax.set_xlabel("Announcement year")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot(g.index, 100 * g["any"], color="#9e9e9e", marker="o", ms=3, lw=1.2, label="Any break, as specified (%)")
    ax2.plot(gc.index, 100 * gc["any"], color="#3182bd", marker="o", ms=3, lw=1.2, label="Any break, core (%)")
    ax2.plot(gc.index, 100 * gc["adv"], color="#a63603", marker="o", ms=3, lw=1.6, label="Adverse break, core (%)")
    ax2.set_ylabel("Break rate (%)"); ax2.set_ylim(0, max(16, 100 * g["any"].max() * 1.15))
    ax.axvline(M.TRAIN_END + 0.5, color="k", ls="--", lw=1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=2, frameon=False)
    ax.set_title("Deals and break rates by announcement year (dashed line: train / test split)" + note, loc="left", fontsize=10)
    f_.savefig(fig / "break_rate_by_year.png"); plt.close(f_)

    kt = keep.get(("primary", "text")) or keep[("primary", "full")]
    y = D.y["label_adverse"][kt["te"]]; P = kt["P"]
    ladder = [m for m in ["one_rule", "logit_base", "lgbm_base", "lgbm_extm", "lgbm_tfidf", "lgbm_emb", "lgbm_llm", "lgbm_alltext", "tfidf_logit", "ft_encoder"] if m in P]
    # 2. ROC / PR
    f_, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for m in ladder:
        fpr, tpr, _ = roc_curve(y, P[m]); pr, rc, _ = precision_recall_curve(y, P[m])
        axes[0].plot(fpr, tpr, color=col[m], lw=1.5, label=f"{short[m]} ({roc_auc_score(y, P[m]):.2f})")
        axes[1].plot(rc, pr, color=col[m], lw=1.5, label=f"{short[m]} ({average_precision_score(y, P[m]):.2f})")
    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1); axes[1].axhline(y.mean(), color="#9e9e9e", ls="--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate"); axes[0].set_title("ROC, adverse breaks, test 2019-2025" + note, loc="left", fontsize=10)
    axes[1].set(xlabel="Recall (share of adverse breaks found)", ylabel="Precision", ylim=(0, min(1, max(0.5, 6 * y.mean()))))
    axes[1].set_title("Precision-recall (dashed: base rate)", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, title="ROC AUC", title_fontsize=8, loc="lower right"); axes[1].legend(fontsize=7, title="PR AUC", title_fontsize=8, loc="upper right")
    f_.savefig(fig / "roc_pr_ladder.png"); plt.close(f_)

    # 3. reliability before / after calibration
    kf = keep[("primary", "full")]; yf = D.y["label_adverse"][kf["te"]]
    f_, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    lim = 0.0
    for m, c_ in ((final, "#08306b"), ("logit_base", "#6baed6")):
        if ("cal", m) not in keep:
            continue
        raw, adj, _ = keep[("cal", m)]
        for ax, p_, in ((axes[0], raw), (axes[1], adj)):
            q = pd.qcut(pd.Series(p_).rank(method="first"), 5, labels=False).to_numpy()
            xs = [p_[q == b].mean() for b in range(5)]; ys = [yf[q == b].mean() for b in range(5)]
            se = [np.sqrt(max(v * (1 - v), 1e-9) / (q == b).sum()) for b, v in enumerate(ys)]
            ax.errorbar(xs, ys, yerr=1.96 * np.array(se), color=c_, marker="o", ms=4, lw=1.4, capsize=2, label=short.get(m, m))
            lim = max(lim, max(xs), max(ys) + 2 * max(se))
    lim = min(1.0, lim * 1.1 + 0.01)
    for ax, t in ((axes[0], "Raw model output"), (axes[1], "After Platt scaling fitted on validation predictions")):
        ax.plot([0, lim], [0, lim], color="#9e9e9e", ls="--", lw=1); ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability of an adverse break")
        ax.set_title(t + note, loc="left", fontsize=10)
    axes[0].set_ylabel("Observed adverse-break rate (test)"); axes[0].legend(fontsize=8, loc="upper left")
    f_.savefig(fig / "calibration.png"); plt.close(f_)

    # 4. top features by gain
    panels = [(kf["G"].get(final_tab), f"{short.get(final_tab, final_tab)} (full sample)")]
    if ("primary", "text") in keep and "lgbm_alltext" in keep[("primary", "text")]["G"]:
        panels.append((keep[("primary", "text")]["G"]["lgbm_alltext"], "5A all groups (text subset)"))
    f_, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4)); axes = np.atleast_1d(axes)
    for ax, (g_, title) in zip(axes, panels):
        s_ = pd.Series(g_).sort_values(ascending=False); s_ = 100 * s_ / s_.sum(); top = s_.head(15)[::-1]
        cols = ["#d94801" if n.startswith(("emb_", "tfidf_")) else "#fdae6b" if n.startswith(("kw_", "llm_", "log_press", "log_agreement", "has_agreement"))
                else "#6a51a3" if n.startswith("mkt_") else "#08519c" for n in top.index]
        ax.barh(top.index, top.values, color=cols); ax.set_xlabel("Share of total split gain (%)"); ax.set_title(title + note, loc="left", fontsize=10); ax.grid(axis="y", alpha=0)
    f_.tight_layout(); f_.savefig(fig / "top_features.png"); plt.close(f_)

    # 5. lift chart
    f_, ax = plt.subplots(figsize=(5.6, 4.5))
    for m in ladder:
        o_ = np.argsort(-P[m], kind="stable")
        ax.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[o_]) / y.sum(), color=col[m], lw=1.5, label=short[m])
    ax.plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1, label="random")
    for x in (0.1, 0.2):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ax.set(xlabel="Share of deals flagged (riskiest first)", ylabel="Share of adverse breaks captured", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Cumulative adverse breaks captured, test" + note, loc="left", fontsize=10); ax.legend(fontsize=7, loc="lower right")
    f_.savefig(fig / "lift_chart.png"); plt.close(f_)

    ex = out.get("extras", {})
    # 6. SHAP summary (beeswarm-style)
    if "shap" in keep:
        contrib, B, names, top = keep["shap"]
        rng = np.random.default_rng(0)
        f_, ax = plt.subplots(figsize=(7.4, 5.6))
        for r, j in enumerate(top[::-1]):
            v = B[:, j]; ok = np.isfinite(v)
            rk = pd.Series(v[ok]).rank(pct=True).to_numpy()
            ax.scatter(contrib[~ok, j], r + rng.uniform(-0.25, 0.25, (~ok).sum()), s=4, color="#bdbdbd", alpha=0.6, lw=0)
            sc = ax.scatter(contrib[ok, j], r + rng.uniform(-0.25, 0.25, ok.sum()), s=4, c=rk, cmap="coolwarm", alpha=0.7, lw=0, vmin=0, vmax=1)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([names[j] for j in top[::-1]], fontsize=8)
        ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("SHAP value (log-odds of an adverse break; right = riskier)"); ax.grid(axis="y", alpha=0)
        cb = f_.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Feature value (percentile rank; grey = missing)", fontsize=8); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
        ax.set_title("SHAP values of the tabular model on test deals" + note, loc="left", fontsize=10)
        f_.savefig(fig / "shap_summary.png"); plt.close(f_)
    # 7. ablation
    if "ablation" in ex:
        a = ex["ablation"]; gs = [g_ for g_ in ["deal_terms", "premium_price", "conditions", "financials", "market_context", "kw", "tfidf", "emb", "llm_terms"] if f"without {g_}" in a["metrics"]]
        f_, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        ypos = np.arange(len(gs))[::-1]
        only = [a["metrics"][f"only {g_}"]["roc_auc"] for g_ in gs]; oc = [a["ci95"][f"only {g_}"]["roc_auc"] for g_ in gs]
        axes[0].errorbar(only, ypos, xerr=[[o - c[0] for o, c in zip(only, oc)], [c[1] - o for o, c in zip(only, oc)]], fmt="o", color="#08519c", capsize=3)
        axes[0].axvline(0.5, color="#9e9e9e", ls="--", lw=1); axes[0].axvline(a["metrics"]["all groups"]["roc_auc"], color="k", ls=":", lw=1)
        axes[0].set_yticks(ypos); axes[0].set_yticklabels([F.GROUP_LABELS[g_] for g_ in gs], fontsize=8); axes[0].set_xlabel("Test AUC of the group alone (dotted: all groups)")
        d_ = [a["drop_one"][f"all groups minus without {g_}"]["roc_auc"] for g_ in gs]
        axes[1].errorbar([x["point"] for x in d_], ypos, xerr=[[x["point"] - x["ci95"][0] for x in d_], [x["ci95"][1] - x["point"] for x in d_]], fmt="o", color="#a63603", capsize=3)
        axes[1].axvline(0, color="k", lw=0.8); axes[1].set_xlabel("Loss in test AUC when the group is dropped")
        axes[0].set_title("Feature groups alone" + note, loc="left", fontsize=10); axes[1].set_title("Drop-one-group", loc="left", fontsize=10)
        for ax in axes:
            ax.grid(axis="y", alpha=0)
        f_.savefig(fig / "ablation_groups.png"); plt.close(f_)
    # 8. rolling origin
    if ex.get("rolling_origin"):
        ro = ex["rolling_origin"]
        f_, ax = plt.subplots(figsize=(7.6, 3.8))
        for m in [m for m in ("logit_base", "lgbm_base", "lgbm_extm", "lgbm_full_text") if m in ro[0]]:
            ax.plot([r["T"] for r in ro], [r[m]["roc_auc"] for r in ro], marker="o", ms=4, lw=1.5, color=col[m], label=short[m])
        ax.axhline(0.5, color="#9e9e9e", ls="--", lw=1)
        ax.set(xlabel="Last training year T (test on T+1 to T+2)", ylabel="Test ROC AUC, adverse breaks"); ax.legend(fontsize=7.5, ncol=2, frameon=False)
        ax.set_title("Rolling-origin evaluation" + note, loc="left", fontsize=10)
        f_.savefig(fig / "rolling_origin.png"); plt.close(f_)


if __name__ == "__main__":
    main()

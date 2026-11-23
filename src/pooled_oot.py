"""Pooled out-of-time evaluation, 2013-2025 (more breaks than the single 2019-2025 test period).

For every test year Y from 2013 to 2025 the whole ladder is refitted on core-sample deals announced
up to Y-1 and scored on the deals announced in Y. Test years do not overlap, so every deal from
2013 on gets exactly one out-of-time score per model. Everything is chosen inside the training
years of each origin: hyper-parameters, the one rule, whether the year enters, and which of the
three tabular feature sets is "the final model" (highest expanding-window CV AUC, the same rule
that picked the final model in train_eval.py). Nothing from 2013-2025 leaks backwards.

  python src/pooled_oot.py fit      refit all origins (about half an hour); writes data/interim/pooled_oot.pkl
  python src/pooled_oot.py report   pooled metrics with intervals; writes results/pooled_oot.md, .json

Intervals: bootstrap that resamples announcement years (13 clusters), 2,000 draws. Deals of one
year share a market environment and one fitted model, so resampling deals alone would understate
the uncertainty. With 13 clusters the percentile interval is itself rough; the deal-level
interval is reported next to it.

Convention (same as the main split): a model for year Y trains on all deals announced by Y-1,
including a few that were still pending on 1 January of Y. The `embargo` variant trains on deals
announced by Y-2 only and shows how much that matters.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import modeling as M
import train_eval as TE

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
YEARS = list(range(2013, 2026))
TARGET = "label_adverse"
TAB = ("lgbm_base", "lgbm_ext", "lgbm_extm")
N_BOOT = 2000

LABELS = {
    "base_rate": "0. Base rate (training-period break rate)", "one_rule": "1. One rule", "logit_base": "2. Logistic, base terms",
    "lgbm_base": "3. LightGBM, base terms", "lgbm_ext": "3x. LightGBM + extended features", "lgbm_extm": "3m. 3x + market context",
    "final_proc": "Final-model procedure (3, 3x or 3m, chosen by CV inside each origin)",
    "lgbm_full_text": "5f. 3m + text groups (has_text, keywords, embeddings, LLM terms)",
    "lgbm_tfidf": "4. 3m + keywords + TF-IDF", "lgbm_emb": "5. 3m + keywords + embeddings", "lgbm_llm": "5L. 3m + keywords + LLM terms",
    "lgbm_alltext": "5A. 3m + all text groups", "tfidf_logit": "6a. Text only: TF-IDF logistic",
}
ORDER_FULL = ["base_rate", "one_rule", "logit_base", "lgbm_base", "lgbm_ext", "lgbm_extm", "final_proc", "lgbm_full_text"]
ORDER_TEXT = ["base_rate", "one_rule", "logit_base", "lgbm_base", "lgbm_ext", "lgbm_extm", "lgbm_tfidf", "lgbm_emb", "lgbm_llm", "lgbm_alltext", "tfidf_logit"]
PAIRS_FULL = [("lgbm_base", "logit_base"), ("lgbm_ext", "lgbm_base"), ("lgbm_extm", "lgbm_ext"), ("final_proc", "logit_base"),
              ("final_proc", "lgbm_base"), ("lgbm_full_text", "lgbm_extm")]
PAIRS_TEXT = [("lgbm_ext", "lgbm_base"), ("lgbm_tfidf", "lgbm_extm"), ("lgbm_emb", "lgbm_extm"), ("lgbm_llm", "lgbm_extm"),
              ("lgbm_alltext", "lgbm_extm"), ("lgbm_alltext", "lgbm_base")]


# ----------------------------------------------------------------------------- fit
def fit(suffix: str, years) -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    D = M.load_data(suffix)
    D.ft = {}                                   # the fine-tuned encoder was trained on <=2015: not out-of-time before 2019
    y = D.y[TARGET]
    idx = np.arange(len(y))
    n = len(idx)
    cols, oof, sel = {}, {}, {}

    def put(name, te, p):
        cols.setdefault(name, np.full(n, np.nan))[te] = p

    for Y in years:
        for subset in ("full", "text"):
            sm = D.core & (D.has_text if subset == "text" else True)
            tr, te = idx[sm & (D.year <= Y - 1)], idx[sm & (D.year == Y)]
            if len(te) == 0 or y[tr].sum() < 20:
                continue
            log(f"{Y} / {subset}: train {len(tr)} ({int(y[tr].sum())} pos), test {len(te)} ({int(y[te].sum())} pos)")
            P, I, G, O, feats = TE.run_ladder(D, TARGET, tr, te, subset, reduced=True, log=log)
            for m, p in P.items():
                put(f"pred__{subset}__{m}", te, p)
            sel[(Y, subset)] = {m: {k: v for k, v in I[m].items() if k in ("cv_auc", "num_leaves", "n_estimators", "use_year", "C", "rule")} for m in I}
            for m, (oi, op) in O.items():
                if len(op):
                    oof[(Y, subset, m)] = (D.deals["deal_id"].to_numpy()[oi], np.asarray(op, dtype=float))
            if subset == "full":
                best = max((m for m in TAB if m in I), key=lambda m: I[m]["cv_auc"])
                sel[(Y, "full")]["final_proc"] = {"chosen": best}
                put("pred__full__final_proc", te, P[best])
                oof[(Y, "full", "final_proc")] = oof.get((Y, "full", best), (np.array([]), np.array([])))
                # embargo variant: same specification, trained on deals announced by Y-2
                tr2 = idx[sm & (D.year <= Y - 2)]
                kind, ft, par = feats[best]
                put("pred__embargo__final_proc", te, M.fit_lgbm(D, y, ft, tr2, te, par)[0])
    ids = D.deals["deal_id"].to_numpy()
    base = pd.DataFrame({"deal_id": ids, "year": D.year, "core": D.core, "has_text": D.has_text, "label": D.y["label"],
                         "label_adverse": D.y["label_adverse"], "outcome3": D.outcome3})
    pred = pd.concat([base, pd.DataFrame(cols)], axis=1)
    with open(INTERIM / f"pooled_oot{suffix}.pkl", "wb") as fh:
        pickle.dump({"pred": pred, "oof": oof, "selection": sel, "years": list(years), "target": TARGET}, fh)
    log(f"wrote pooled_oot{suffix}.pkl: {int(pred['pred__full__final_proc'].notna().sum())} scored deals")


# ----------------------------------------------------------------------------- pooled metrics
def within_year_auc(y, s, yr):
    """AUC over pairs of deals announced in the same year (removes drift in score levels between origins)."""
    num = den = 0.0
    for v in np.unique(yr):
        m = yr == v
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        if pos and neg:
            num += roc_auc_score(y[m], s[m]) * pos * neg; den += pos * neg
    return num / den if den else np.nan


def fast_auc(y, s):
    """ROC AUC from ranks (ties get average ranks); same value as sklearn, much faster inside a bootstrap."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats(y, s, yr, auc_only=False):
    if auc_only:
        num = den = 0.0
        for v in np.unique(yr):
            m = yr == v
            pos = int(y[m].sum()); neg = int(m.sum()) - pos
            if pos and neg:
                num += fast_auc(y[m], s[m]) * pos * neg; den += pos * neg
        return [fast_auc(y, s), num / den if den else np.nan, np.nan, np.nan, np.nan]
    return [M.auc_safe(y, s), within_year_auc(y, s, yr), average_precision_score(y, s) if y.sum() else np.nan,
            M.capture_at(y, s, 0.10), M.capture_at(y, s, 0.20)]


STAT_KEYS = ["roc_auc", "within_year_auc", "pr_auc", "capture_top10", "capture_top20"]


def year_blocks(yr):
    return {v: np.where(yr == v)[0] for v in np.unique(yr)}


def draws(yr, n_boot, rng, by_year=True):
    blocks = year_blocks(yr)
    keys = list(blocks)
    for _ in range(n_boot):
        if by_year:
            yield np.concatenate([blocks[k] for k in rng.choice(keys, len(keys))])
        else:
            yield rng.integers(0, len(yr), len(yr))


def evaluate(y, yr, S: dict, pairs, n_boot=N_BOOT, seed=0, auc_only=False):
    """Point estimates plus year-cluster and deal-level bootstrap intervals for every score and paired difference.
    With auc_only the bootstrap covers the pooled and the within-year ROC AUC only (the other intervals come back as NaN)."""
    out = {"scores": {}, "pairs": {}}
    point = {k: np.array(stats(y, s, yr)) for k, s in S.items()}
    for mode in ("year", "deal"):
        rng = np.random.default_rng(seed)
        acc = {k: [] for k in S}
        for ix in draws(yr, n_boot, rng, by_year=(mode == "year")):
            if not 1 < y[ix].sum() < len(ix):
                continue
            for k, s in S.items():
                acc[k].append(stats(y[ix], s[ix], yr[ix], auc_only))
        arr = {k: np.array(v) for k, v in acc.items()}
        for k in S:
            d = out["scores"].setdefault(k, {m: float(point[k][j]) for j, m in enumerate(STAT_KEYS)})
            d[f"ci95_{mode}"] = {m: [float(np.nanpercentile(arr[k][:, j], 2.5)), float(np.nanpercentile(arr[k][:, j], 97.5))] for j, m in enumerate(STAT_KEYS)}
        for a, b in pairs:
            if a in S and b in S:
                dd = arr[a] - arr[b]
                d = out["pairs"].setdefault(f"{a} minus {b}", {m: float(point[a][j] - point[b][j]) for j, m in enumerate(STAT_KEYS)})
                d[f"ci95_{mode}"] = {m: [float(np.nanpercentile(dd[:, j], 2.5)), float(np.nanpercentile(dd[:, j], 97.5))] for j, m in enumerate(STAT_KEYS)}
                d[f"share_gt0_{mode}"] = float((dd[:, 0] > 0).mean())
    return out


def f3(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def d3(v, ci):
    return f"{v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report(suffix: str) -> None:
    synthetic = suffix != ""
    res = ROOT / ("results" if not synthetic else f"results{suffix}")
    res.mkdir(exist_ok=True)
    Z = pickle.load(open(INTERIM / f"pooled_oot{suffix}.pkl", "rb"))
    P, sel = Z["pred"], Z["selection"]
    out = {}
    L = ["# Pooled out-of-time evaluation, 2013-2025" + (" (SYNTHETIC SAMPLE: not real results)" if synthetic else ""), "",
         "Every deal announced from 2013 to 2025 is scored by models fitted only on deals announced in earlier years (13 origins, one test year each, "
         "no overlap). Hyper-parameters, the one rule and the choice of feature set are all re-selected inside each origin's training years. "
         "Target: adverse break. Sample: core (reverse mergers excluded).", "",
         "Why: the 2019-2025 test period has 40 adverse breaks. Pooling 13 out-of-time years gives more, at the price of mixing 13 different fitted models. "
         "`ROC AUC` pools raw scores across years (what a manager using one cut-off over time would experience); `within-year AUC` only compares deals "
         "announced in the same year and is immune to drift in score levels between origins.", "",
         "Intervals: bootstrap resampling announcement years (13 clusters, 2,000 draws). The deal-level interval is shown for the paired differences as a comparison; "
         "it ignores that deals of one year share a model and a market, and is narrower for that reason.", ""]
    for subset, order, pairs, title in (("full", ORDER_FULL, PAIRS_FULL, "All core deals"), ("text", ORDER_TEXT, PAIRS_TEXT, "Deals with announcement text")):
        cols = {m: f"pred__{subset}__{m}" for m in order if f"pred__{subset}__{m}" in P}
        if not cols:
            continue
        ok = P[list(cols.values())].notna().all(axis=1).to_numpy()
        y, yr = P[TARGET].to_numpy()[ok], P["year"].to_numpy()[ok]
        if y.sum() < 3:
            continue
        S = {m: P[c].to_numpy()[ok] for m, c in cols.items()}
        ev = evaluate(y, yr, S, pairs)
        ev["n"], ev["pos"], ev["years"] = int(ok.sum()), int(y.sum()), [int(yr.min()), int(yr.max())]
        out[subset] = ev
        L += [f"## {title}: {ev['n']:,} deals, {ev['pos']} adverse breaks ({yr.min()}-{yr.max()})", "",
              "| Model | ROC AUC [95% CI by year] | Within-year AUC | PR AUC | Breaks in riskiest 10% | Breaks in riskiest 20% |", "|---|---|---|---|---|---|"]
        for m in order:
            if m in S:
                r = ev["scores"][m]; c = r["ci95_year"]
                L.append(f"| {LABELS[m]} | {f3(r['roc_auc'], c['roc_auc'])} | {f3(r['within_year_auc'], c['within_year_auc'])} | {f3(r['pr_auc'], c['pr_auc'])} | "
                         f"{r['capture_top10']:.2f} [{c['capture_top10'][0]:.2f}, {c['capture_top10'][1]:.2f}] | {r['capture_top20']:.2f} [{c['capture_top20'][0]:.2f}, {c['capture_top20'][1]:.2f}] |")
        L += ["", "Paired differences:", "", "| Comparison | ROC AUC [95% CI by year] | ROC AUC [95% CI by deal] | Within-year AUC [by year] | Breaks in riskiest 20% [by year] |", "|---|---|---|---|---|"]
        for k, r in ev["pairs"].items():
            a, b = k.split(" minus ")
            L.append(f"| {LABELS[a].split('.')[0].split(' (')[0]} minus {LABELS[b].split('.')[0].split(' (')[0]} | {d3(r['roc_auc'], r['ci95_year']['roc_auc'])} | "
                     f"{d3(r['roc_auc'], r['ci95_deal']['roc_auc'])} | {d3(r['within_year_auc'], r['ci95_year']['within_year_auc'])} | "
                     f"{r['capture_top20']:+.2f} [{r['ci95_year']['capture_top20'][0]:+.2f}, {r['ci95_year']['capture_top20'][1]:+.2f}] |")
        L.append("")
    # choice of the final model per origin, per-year detail, embargo
    if "pred__full__final_proc" in P:
        ok = P["pred__full__final_proc"].notna().to_numpy()
        y, yr = P[TARGET].to_numpy()[ok], P["year"].to_numpy()[ok]
        L += ["## By test year (final-model procedure)", "", "| Test year | Deals | Adverse breaks | Chosen feature set | ROC AUC | Mean score | Break rate |", "|---|---:|---:|---|---:|---:|---:|"]
        s = P["pred__full__final_proc"].to_numpy()[ok]
        per = []
        for v in np.unique(yr):
            m = yr == v
            ch = sel.get((int(v), "full"), {}).get("final_proc", {}).get("chosen", "")
            a = M.auc_safe(y[m], s[m])
            per.append({"year": int(v), "n": int(m.sum()), "pos": int(y[m].sum()), "chosen": ch, "auc": float(a)})
            L.append(f"| {v} | {int(m.sum())} | {int(y[m].sum())} | {LABELS.get(ch, ch).split('.')[0]} | {a:.3f} | {s[m].mean():.3f} | {y[m].mean():.3f} |")
        out["by_year"] = per
        if "pred__embargo__final_proc" in P:
            e = P["pred__embargo__final_proc"].to_numpy()[ok]
            ev = evaluate(y, yr, {"final_proc": s, "embargo": e}, [("embargo", "final_proc")], n_boot=1000)
            out["embargo"] = ev
            r = ev["scores"]["embargo"]; p = ev["pairs"]["embargo minus final_proc"]
            L += ["", "## Embargo check", "",
                  "A model for year Y trains on deals announced by Y-1, a few of which were still pending on 1 January of Y. Training on deals announced by Y-2 "
                  f"instead (same specification) gives a pooled AUC of {f3(r['roc_auc'], r['ci95_year']['roc_auc'])}, a change of "
                  f"{d3(p['roc_auc'], p['ci95_year']['roc_auc'])}. The convention does not drive the result.", ""]
    (res / "pooled_oot.md").write_text("\n".join(L) + "\n")
    (res / "pooled_oot.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "report"])
    ap.add_argument("--tag", default="real")
    ap.add_argument("--years", default=None, help="e.g. 2013-2025")
    a = ap.parse_args()
    sfx = "" if a.tag == "real" else f"_{a.tag}"
    if a.cmd == "fit":
        ys = YEARS
        if a.years:
            lo, hi = a.years.split("-"); ys = list(range(int(lo), int(hi) + 1))
        fit(sfx, ys)
    else:
        report(sfx)

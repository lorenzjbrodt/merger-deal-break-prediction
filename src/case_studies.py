"""Pick six test-period cases by a fixed rule and collect their facts for the write-up.

Selection (core sample, test period, deals with an announcement text, final model score):
  two flagged breaks   the adverse breaks with the highest model score
  two missed breaks    the adverse breaks with the lowest model score
  two false alarms     the completed deals with the highest model score

Writes data/interim/case_facts.json (git-ignored). The published case studies in
results/case_studies.md are written from these facts using public filing text only.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"


def main() -> None:
    mj = json.loads((ROOT / "results/metrics.json").read_text())
    final = mj["final_models"]["overall"]
    P = pd.read_parquet(INTERIM / "predictions.parquet")
    d = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "target_clean", "buyers", "announced_date", "cancelled_date", "closed_date"])
    t = pd.read_parquet(INTERIM / "text.parquet", columns=["deal_id", "press_release", "press_source"])
    o = pd.read_parquet(INTERIM / "outcomes.parquet")[["deal_id", "reason"]]
    col = f"pred__primary__full__{final}"
    m = P.merge(d, on="deal_id").merge(t, on="deal_id").merge(o, on="deal_id", how="left")
    m = m[m["core"] & (m["year"] >= 2019) & m["has_text"] & m[col].notna()].copy()
    m["pct"] = m[col].rank(pct=True)
    adv, closed = m[m["label_adverse"] == 1].sort_values(col), m[m["outcome3"] == "closed"].sort_values(col)
    picks = [("flagged break", r) for r in adv.tail(2).iloc[::-1].itertuples(index=False)] + \
            [("missed break", r) for r in adv.head(2).itertuples(index=False)] + \
            [("false alarm", r) for r in closed.tail(2).iloc[::-1].itertuples(index=False)]
    sh = pickle.load(open(ROOT / "artifacts/final_tabular_shap.pkl", "rb")) if (ROOT / "artifacts/final_tabular_shap.pkl").exists() else None
    pos = {x: i for i, x in enumerate(sh["deal_id"])} if sh else {}
    pf = pd.read_parquet(INTERIM / "price_features.parquet").set_index("deal_id") if (INTERIM / "price_features.parquet").exists() else None
    facts = []
    for kind, r in picks:
        f = {"kind": kind, "target": r.target_clean, "buyer": str(r.buyers), "announced": str(pd.Timestamp(r.announced_date).date()),
             "outcome_date": str(pd.Timestamp(r.cancelled_date if pd.notna(r.cancelled_date) else r.closed_date).date()),
             "outcome": r.outcome3, "reason_class": r.reason, "score_percentile_in_test": round(float(r.pct), 3), "score": round(float(getattr(r, col)), 4),
             "press_source": r.press_source, "press_opening": r.press_release[:1500]}
        if r.deal_id in pos:
            c = sh["contrib"][pos[r.deal_id]]; v = sh["values"][pos[r.deal_id]]
            top = np.argsort(-np.abs(c))[:6]
            f["drivers"] = [{"feature": sh["names"][j], "pushes_risk": "up" if c[j] > 0 else "down", "shap": round(float(c[j]), 3),
                             "value": None if not np.isfinite(v[j]) else round(float(v[j]), 3)} for j in top]
        lp = INTERIM / "llm_extract/out/haiku" / f"{r.deal_id}.json"
        if lp.exists():
            f["extracted_terms"] = {k: v for k, v in json.loads(lp.read_text()).items() if k != "doc_id" and v not in (None, [], "")}
        if pf is not None and r.deal_id in pf.index and bool(pf.loc[r.deal_id, "backtest_ok"]):
            f["p_mkt"] = round(float(pf.loc[r.deal_id, "p_mkt"]), 3); f["spread_at_t1"] = round(float(pf.loc[r.deal_id, "spread"]), 4)
        facts.append(f)
    (INTERIM / "case_facts.json").write_text(json.dumps(facts, indent=1))
    for f in facts:
        print(json.dumps({k: v for k, v in f.items() if k != "press_opening"})[:900]); print("   PRESS:", f["press_opening"][:500].replace("\n", " "), "\n")


if __name__ == "__main__":
    main()

"""Label audit: check Capital IQ's Closed / Cancelled status against the target's EDGAR filing record.

A random sample of 40 Cancelled and 40 Closed deals (with a target CIK) is checked against the
SEC submissions index (cached by edgar_download.py; older index pages are fetched when needed):

  Closed deals     confirmed by a Form 15 / Form 25 (deregistration or delisting) or an 8-K
                   Item 2.01 / 3.01 / 5.01 / 3.03 between 15 days before and 150 days after the closing
                   date. Contradicted if the target is still filing 10-K / 10-Q more than 15 months
                   after the closing date and never filed a Form 15 / 25 in between.
  Cancelled deals  confirmed by an 8-K Item 1.02 (termination of a material definitive agreement)
                   within 5 days before to 20 days after the cancellation date, or by the target still
                   filing periodic reports at least 6 months later. A Form 15 / 25 soon after the
                   cancellation is NOT a contradiction when another buyer acquired the target.
                   Contradicted only if a Form 15 / 25 follows within 150 days, no periodic report
                   follows, and the break is not classified as a competing bid.

This is an audit of the label, never an input to any model. Only aggregates are written.

Usage: python src/label_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import edgar_download as E

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
N_PER_CLASS, SEED = 40, 11
DEREG = ("15-12B", "15-12G", "15-15D", "15", "25", "25-NSE", "15-12B/A", "15-12G/A", "25/A")
PERIODIC = ("10-K", "10-Q", "10-K405", "10-KSB", "10-QSB", "20-F", "40-F", "10-KT")


def all_filings(cik: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    c = E.cik10(cik)
    sub = E.get_json_cached(f"https://data.sec.gov/submissions/CIK{c}.json", E.SUB_DIR / f"CIK{c}.json")
    if not sub:
        return pd.DataFrame(columns=["date", "form", "items"])
    blocks = [sub.get("filings", {}).get("recent", {})]
    for f in sub.get("filings", {}).get("files", []):
        try:
            f_from, f_to = pd.Timestamp(f["filingFrom"]), pd.Timestamp(f["filingTo"])
        except Exception:
            continue
        if f_from <= end and f_to >= start:
            page = E.get_json_cached(f"https://data.sec.gov/submissions/{f['name']}", E.SUB_DIR / f["name"])
            if page:
                blocks.append(page)
    rows = []
    for b in blocks:
        n = len(b.get("accessionNumber", []))
        items = b.get("items", [""] * n)
        rows += [(b["filingDate"][i], b["form"][i], items[i] if i < len(items) else "") for i in range(n)]
    df = pd.DataFrame(rows, columns=["date", "form", "items"]).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)]


def audit_one(r) -> dict:
    out_date = r.closed_date if r.label == 0 else r.cancelled_date
    if pd.isna(out_date) or not isinstance(r.cik, str):
        return {"verdict": "no evidence", "why": "no CIK or outcome date"}
    fil = []
    for c in [x.strip() for x in r.cik.split(";") if x.strip().isdigit()]:
        try:
            fil.append(all_filings(c, out_date - pd.Timedelta(days=30), out_date + pd.Timedelta(days=900)))
        except Exception as e:
            return {"verdict": "no evidence", "why": f"index error {type(e).__name__}"}
    f = pd.concat(fil) if fil else pd.DataFrame(columns=["date", "form", "items"])
    if f.empty:
        return {"verdict": "no evidence", "why": "no filings in the window"}
    d = (f["date"] - out_date).dt.days
    dereg = f["form"].isin(DEREG)
    periodic = f["form"].isin(PERIODIC)
    k8 = f["form"].isin(["8-K", "8-K/A"])
    items = f["items"].fillna("").astype(str)
    if r.label == 0:
        near = (d >= -15) & (d <= 150)
        if (dereg & near).any():
            return {"verdict": "confirmed", "why": "Form 15 / 25 after closing"}
        if (k8 & near & items.str.contains(r"2\.01|3\.01|5\.01|3\.03")).any():
            return {"verdict": "confirmed", "why": "8-K completion / delisting / change-of-control item"}
        if (periodic & (d > 460)).any() and not (dereg & (d <= 460)).any():
            return {"verdict": "contradicted", "why": "target still files periodic reports 15+ months after the closing date"}
        if not (periodic & (d > 150)).any():
            return {"verdict": "consistent", "why": "periodic reports stop after closing (no Form 15 / 25 found)"}
        return {"verdict": "unclear", "why": "periodic reports continue for a while, no deregistration found"}
    if (k8 & (d >= -5) & (d <= 20) & items.str.contains(r"1\.02")).any():
        return {"verdict": "confirmed", "why": "8-K Item 1.02 at the cancellation date"}
    if (periodic & (d >= 180)).any():
        return {"verdict": "confirmed", "why": "target still files periodic reports 6+ months later"}
    if (dereg & (d >= 0) & (d <= 150)).any():
        if r.reason == "competing_bid":
            return {"verdict": "consistent", "why": "deregistered after cancellation; break classified as a competing bid (acquired by another buyer)"}
        return {"verdict": "contradicted", "why": "Form 15 / 25 within 150 days of the cancellation and no later periodic report"}
    if (k8 & (d >= -5) & (d <= 20)).any():
        return {"verdict": "consistent", "why": "8-K at the cancellation date (no item codes available)"}
    return {"verdict": "unclear", "why": "no termination 8-K and no later periodic report in the index"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96; p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    d = d.merge(pd.read_parquet(INTERIM / "outcomes.parquet")[["deal_id", "reason"]], on="deal_id", how="left")
    d = d[d["cik"].notna()]
    rng = np.random.default_rng(SEED)
    rows = []
    for lab, name in ((1, "Cancelled"), (0, "Closed")):
        g = d[d["label"] == lab]
        for r in g.iloc[rng.choice(len(g), N_PER_CLASS, replace=False)].itertuples(index=False):
            rows.append({"status": name, "year": int(r.year), **audit_one(r)})
    A = pd.DataFrame(rows)
    A.to_parquet(INTERIM / "label_audit.parquet", index=False)
    L = ["# Label audit against EDGAR", "",
         f"{N_PER_CLASS} Cancelled and {N_PER_CLASS} Closed deals with a target CIK were drawn at random (seed {SEED}) and their Capital IQ status was checked "
         "against the target's SEC filing index: Form 15 / Form 25 and 8-K completion items for closed deals; 8-K Item 1.02 and continued periodic "
         "reporting for cancelled deals. The rules are in the header of `src/label_audit.py`. The check is automatic and uses filing metadata, not the "
         "text of the filings; 8-K item codes exist only from late 2004.", "",
         "| Capital IQ status | Confirmed | Consistent | Unclear | No evidence | Contradicted |", "|---|---:|---:|---:|---:|---:|"]
    res = {}
    for name in ("Cancelled", "Closed"):
        g = A[A["status"] == name]["verdict"].value_counts()
        res[name] = {k: int(g.get(k, 0)) for k in ("confirmed", "consistent", "unclear", "no evidence", "contradicted")}
        L.append(f"| {name} | " + " | ".join(str(res[name][k]) for k in ("confirmed", "consistent", "unclear", "no evidence", "contradicted")) + " |")
    L += ["", "Reasons recorded:", ""]
    for (st, v, why), n in A.groupby(["status", "verdict", "why"]).size().items():
        L.append(f"- {st}, {v}: {why} ({n})")
    L += ["", "## Estimated label noise", ""]
    for name in ("Cancelled", "Closed"):
        r = res[name]; checked = r["confirmed"] + r["consistent"] + r["contradicted"]
        lo, hi = wilson(r["contradicted"], checked)
        L.append(f"- {name}: {r['contradicted']} of {checked} checkable deals contradicted = {r['contradicted'] / max(checked, 1):.1%} "
                 f"(95% Wilson interval {lo:.1%} to {hi:.1%}); {r['unclear'] + r['no evidence']} of {N_PER_CLASS} could not be checked.")
    L += ["", "A contradiction means the filing record does not fit the status, not that the status is proven wrong: a target with public debt can keep "
          "filing after it is acquired, and a cancelled target can be acquired by someone else soon after. The upper ends of the intervals are the "
          "relevant figures for how much label noise the models may be facing. The reason for a break (the outcome taxonomy) is a separate source of "
          "noise; see `results/outcome_taxonomy.md`.", ""]
    (ROOT / "results/label_audit.md").write_text("\n".join(L) + "\n")
    (ROOT / "results/label_audit.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()

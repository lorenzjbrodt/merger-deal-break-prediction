"""EXPLORATORY benchmark: an LLM's zero-shot break-risk score from anonymised announcement text.

This is not part of the main models. It answers one side question: how well does a general LLM,
with no training on this sample, rank deals by break risk from the announcement alone?

Caveat that cannot be removed: the LLM may recognise well-known deals despite anonymisation
(deal size, industry, price and wording can identify them) and may know how they ended from its
training data. Any skill shown here is an UPPER bound on what the text contains, and deals after
the model's knowledge cut-off are reported separately for that reason.

Anonymisation: target and buyer names, tickers and four-digit years are replaced; the text is
truncated to the first 6,000 characters of the press release.

Usage: python src/llm_zero_shot.py run [--model sonnet] ; python src/llm_zero_shot.py report
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WORK = INTERIM / "llm_zero_shot"
SYSTEM = """You assess anonymised merger announcements for a merger-arbitrage desk.

You will receive one announcement in which company names, tickers and years have been replaced by placeholders.
Estimate the probability, from 0 to 100, that this deal will NOT be completed for a reason other than a superior
competing bid (regulatory block, financing failure, shareholder vote failure, buyer walking away, mutual termination).

Rules: use only the text provided. Do not try to identify the companies and do not use any outside knowledge of
what happened. Base the estimate on the stated terms and risks (conditions, approvals needed, financing, buyer type,
structure, premium, timing). Most announced deals close; calibrate accordingly.

Return: doc_id (copied exactly), break_probability_pct (number 0-100), main_risk (at most 12 words)."""
SCHEMA = {"type": "object", "properties": {"doc_id": {"type": "string"}, "break_probability_pct": {"type": "number"},
                                           "main_risk": {"type": "string"}}, "required": ["doc_id", "break_probability_pct"]}
SUFFIX_RE = re.compile(r",?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.p\.|lp|plc|holdings?|group|n\.v\.|s\.a\.|ag)\.?$", re.I)


def name_variants(name: str) -> list[str]:
    name = re.sub(r"\s*\([^)]*\)", "", str(name)).strip()
    out = {name}
    core = name
    for _ in range(3):
        core = SUFFIX_RE.sub("", core).strip(" ,.")
    out.add(core)
    return sorted((v for v in out if len(v) >= 4), key=len, reverse=True)


def anonymise(text: str, target: str, buyers: str) -> str:
    t = text[:6000]
    for v in name_variants(target):
        t = re.sub(re.escape(v), "TARGET", t, flags=re.I)
    for i, b in enumerate([b for b in str(buyers).split(";") if b.strip()][:4]):
        for v in name_variants(b):
            t = re.sub(re.escape(v), "BUYER" if i == 0 else f"BUYER{i + 1}", t, flags=re.I)
    t = re.sub(r"\((NYSE|NASDAQ|Nasdaq|AMEX|NYSE American|OTC|TSX|LSE)[^)]{0,25}\)", "(TICKER)", t)
    t = re.sub(r"\b(19|20)\d{2}\b", "YEAR", t)
    return t


def cmd_run(args) -> None:
    out = WORK / "out"; out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "year", "target_clean", "buyers", "secondary_features"])
    t = pd.read_parquet(INTERIM / "text.parquet", columns=["deal_id", "press_release", "has_text"])
    d = d.merge(t, on="deal_id")
    d = d[(d["year"] >= 2019) & d["has_text"] & ~d["secondary_features"].fillna("").str.contains("Reverse Merger")]
    todo = [r for r in d.itertuples(index=False) if not (out / f"{r.deal_id}.json").exists()]
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA)]

    def one(r) -> int:
        doc = f"doc_id: {r.deal_id}\n\n{anonymise(r.press_release, r.target_clean, r.buyers)}"
        for attempt in range(4):
            try:
                res = subprocess.run(cmd, input=doc, capture_output=True, text=True, timeout=600, cwd=str(WORK))
                env = json.loads(res.stdout)
                if env.get("is_error"):
                    raise RuntimeError(str(env.get("result"))[:200])
                o = env.get("structured_output") or json.loads(env.get("result") or "{}")
                if o.get("doc_id") == r.deal_id and isinstance(o.get("break_probability_pct"), (int, float)):
                    (out / f"{r.deal_id}.json").write_text(json.dumps(o)); return 1
            except Exception as e:
                time.sleep(120 if "limit" in str(e).lower() else 5 * (attempt + 1))
        return 0

    t0 = time.time(); n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, k in enumerate(ex.map(one, todo), 1):
            n += k
            if i % 100 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] saved={n} elapsed={(time.time() - t0) / 60:.1f} min", flush=True)


ID_SYSTEM = """You will receive a merger announcement in which company names, tickers and years were replaced by placeholders
(TARGET, BUYER, TICKER, YEAR). If you can tell with confidence which real companies TARGET and BUYER are, name them.
If you cannot, answer "unknown". Do not guess. Return: doc_id (copied exactly), target_name, buyer_name, confident (true/false)."""
ID_SCHEMA = {"type": "object", "properties": {"doc_id": {"type": "string"}, "target_name": {"type": "string"}, "buyer_name": {"type": "string"},
                                              "confident": {"type": "boolean"}}, "required": ["doc_id", "target_name"]}


def cmd_identify(args) -> None:
    """Leakage probe: can the LLM name the companies behind the anonymised text?"""
    out = WORK / "identify"; out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "year", "target_clean", "buyers", "secondary_features"])
    t = pd.read_parquet(INTERIM / "text.parquet", columns=["deal_id", "press_release", "has_text"])
    d = d.merge(t, on="deal_id")
    d = d[(d["year"] >= 2019) & d["has_text"] & ~d["secondary_features"].fillna("").str.contains("Reverse Merger")]
    todo = [r for r in d.itertuples(index=False) if not (out / f"{r.deal_id}.json").exists()]
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", ID_SYSTEM, "--json-schema", json.dumps(ID_SCHEMA)]

    def one(r) -> int:
        doc = f"doc_id: {r.deal_id}\n\n{anonymise(r.press_release, r.target_clean, r.buyers)}"
        for attempt in range(4):
            try:
                res = subprocess.run(cmd, input=doc, capture_output=True, text=True, timeout=600, cwd=str(WORK))
                env = json.loads(res.stdout)
                if env.get("is_error"):
                    raise RuntimeError(str(env.get("result"))[:200])
                o = env.get("structured_output") or json.loads(env.get("result") or "{}")
                if o.get("doc_id") == r.deal_id:
                    (out / f"{r.deal_id}.json").write_text(json.dumps(o)); return 1
            except Exception as e:
                time.sleep(120 if "limit" in str(e).lower() else 5 * (attempt + 1))
        return 0
    with ThreadPoolExecutor(args.workers) as ex:
        n = sum(ex.map(one, todo))
    print(f"identified-probe saved {n} of {len(todo)}")


def _name_hit(guess: str, truth: str) -> bool:
    """True if the guessed name shares a distinctive leading word with the real name."""
    stop = {"the", "inc", "corp", "corporation", "company", "co", "group", "holdings", "ltd", "llc", "plc", "of", "and", "international", "unknown", "financial", "first", "american", "national", "bancorp", "energy", "pharmaceuticals", "therapeutics", "technologies"}
    g = [w for w in re.findall(r"[a-z0-9]+", str(guess).lower()) if w not in stop and len(w) > 2]
    tr = [w for w in re.findall(r"[a-z0-9]+", str(truth).lower()) if w not in stop and len(w) > 2]
    return bool(g) and bool(tr) and (tr[0] in g or g[0] in tr)


def cmd_report(args) -> None:
    rows = [json.loads(p.read_text()) for p in (WORK / "out").glob("*.json")]
    Z = pd.DataFrame(rows).rename(columns={"doc_id": "deal_id"})
    P = pd.read_parquet(INTERIM / "predictions.parquet")
    mj = json.loads((ROOT / "results/metrics.json").read_text())
    final = mj["final_models"]["overall"]
    m = P.merge(Z, on="deal_id")
    m = m[m["core"] & (m["year"] >= 2019)]
    rng = np.random.default_rng(0)

    def auc(y, s):
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(y), len(y))
            if 0 < y[ix].sum() < len(ix):
                bs.append(roc_auc_score(y[ix], s[ix]))
        return f"{roc_auc_score(y, s):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
    L = ["# Exploratory: LLM zero-shot break-risk score from anonymised text", "",
         "**Not part of the main models.** A general LLM (Claude Sonnet, no fitting on this sample) reads the anonymised press release of each test-period "
         "deal and returns a probability that the deal will not be completed for a reason other than a superior bid.", "",
         "**Caveat.** Anonymisation replaces company names, tickers and years, but size, industry, price and wording can still identify well-known deals, and "
         "the LLM may know how they ended from its training data. Treat any skill as an upper bound on what the text itself contains. "
         "Deals announced in 2025 are shown separately because they are closest to (or beyond) the model's knowledge cut-off; the count of breaks there is small.", "",
         "| Test deals (core sample, with text) | Deals | Adverse breaks | LLM zero-shot AUC [95% CI] | Final model AUC on the same deals |", "|---|---:|---:|---|---|"]
    col = f"pred__primary__full__{final}"
    for name, mk in (("All test years 2019-2025", m["year"] >= 2019), ("2019-2023", m["year"] <= 2023), ("2024-2025", m["year"] >= 2024), ("2025 only", m["year"] == 2025)):
        g = m[mk]
        y = g["label_adverse"].to_numpy()
        if y.sum() < 3:
            L.append(f"| {name} | {len(g)} | {int(y.sum())} | too few breaks | too few breaks |"); continue
        L.append(f"| {name} | {len(g)} | {int(y.sum())} | {auc(y, g['break_probability_pct'].to_numpy())} | {auc(y, g[col].to_numpy())} |")
    y = m["label_adverse"].to_numpy(); s = m["break_probability_pct"].to_numpy()
    L += ["", f"Mean stated probability: {s.mean():.1f}% (observed adverse-break rate {y.mean():.1%}); "
          f"mean for deals that broke {s[y == 1].mean():.1f}%, for the rest {s[y == 0].mean():.1f}%. "
          f"Rank correlation with the final model's score: {pd.Series(s).corr(pd.Series(m[col].to_numpy()), method='spearman'):.2f}.", ""]
    idd = WORK / "identify"
    if idd.exists() and any(idd.glob("*.json")):
        I = pd.DataFrame([json.loads(p.read_text()) for p in idd.glob("*.json")]).rename(columns={"doc_id": "deal_id"})
        names = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "target_clean"])
        mi = m.merge(I, on="deal_id").merge(names, on="deal_id")
        mi["identified"] = [_name_hit(g, t) for g, t in zip(mi["target_name"], mi["target_clean"])]
        L += ["## Leakage probe: can the LLM name the companies?", "",
              "The same anonymised texts were given to the same LLM with the question \"which companies are TARGET and BUYER?\". A deal counts as identified when the "
              "guessed target name matches the real one on its leading distinctive word.", "",
              "| Group | Deals | Adverse breaks | LLM zero-shot AUC [95% CI] | Final model AUC |", "|---|---:|---:|---|---|"]
        for name, mk in (("Target identified from the anonymised text", mi["identified"]), ("Target not identified", ~mi["identified"])):
            g = mi[mk]; yy = g["label_adverse"].to_numpy()
            if yy.sum() >= 3:
                L.append(f"| {name} | {len(g)} | {int(yy.sum())} | {auc(yy, g['break_probability_pct'].to_numpy())} | {auc(yy, g[col].to_numpy())} |")
            else:
                L.append(f"| {name} | {len(g)} | {int(yy.sum())} | too few breaks | too few breaks |")
        L += ["", f"Identified: {mi['identified'].mean():.0%} of deals overall; {mi.loc[mi.label_adverse == 1, 'identified'].mean():.0%} of the deals that broke and "
              f"{mi.loc[mi.label_adverse == 0, 'identified'].mean():.0%} of the rest. If the LLM names broken deals more often than completed ones, part of its ranking "
              "skill can come from memory of the outcome, not from the text. Skill among deals it could not name is the cleaner estimate, and even that is not proof: "
              "not naming a company is not the same as not recognising the situation.", ""]
    (ROOT / "results/llm_zero_shot.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("identify"); p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=6); p.set_defaults(fn=cmd_identify)
    p = sub.add_parser("report"); p.set_defaults(fn=cmd_report)
    a = ap.parse_args(); a.fn(a)

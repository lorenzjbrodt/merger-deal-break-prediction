"""Management guidance on closing: the expected closing time the parties state in the announcement.

Used only as a benchmark for the time-to-closing models ("does the model beat what the parties said?").
The LLM is a reader: it converts the timing sentence(s) of the press release into a date range. It sees the
announcement date and the sentences, nothing else, and is never asked how long the deal will take.

  python src/llm_closing.py prepare                       sentences that state expected closing, one small document per deal
  python src/llm_closing.py run --run sonnet --model sonnet
  python src/llm_closing.py run --run strong --model fable --sample 60     independent reading for validation
  python src/llm_closing.py finalize                      data/interim/closing_guidance.parquet + results/closing_guidance_validation.md

Everything is cached under data/interim/llm_closing/ (git-ignored) and resumable.
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

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WORK = INTERIM / "llm_closing"
TIMING = re.compile(r"(expect|anticipat|schedul|target|intend|estimat|likely|plan)\w*\b[^.;]{0,160}\b(clos|complet|consummat|finaliz)\w*|"
                    r"\b(clos|complet|consummat)\w*\b[^.;]{0,120}\b(expect|anticipat)\w*\b[^.;]{0,120}\b(quarter|half|year|month|20\d\d|end of|by )", re.I)
WHEN = re.compile(r"quarter|half|year[\s\-]end|end of|20\d\d|month|week|days|january|february|march|april|may|june|july|august|september|october|november|december|summer|fall|spring|winter|early|late|mid", re.I)

SCHEMA = {"type": "object", "properties": {
    "doc_id": {"type": "string"},
    "stated": {"type": "boolean"},
    "earliest_date": {"type": ["string", "null"]},
    "latest_date": {"type": ["string", "null"]},
    "basis": {"type": ["string", "null"], "enum": ["calendar", "fiscal", "relative", None]},
    "quote": {"type": ["string", "null"]},
}, "required": ["doc_id", "stated"]}

INSTRUCTIONS = """You read sentences from merger announcements and record WHEN THE PARTIES SAY THEY EXPECT THE DEAL TO CLOSE.

Rules
- Use only the sentences provided and the announcement date in the header. Never use outside knowledge about the deal
  and never give your own view of how long it will take.
- stated: true only if the text states an expected time of closing/completion of this transaction (a quarter, half-year,
  month, season, date, "by year-end", "within 60 days", ...). Otherwise false with null dates. Statements about a
  tender offer commencing, a meeting date, an outside/termination date or financing dates do not count.
- earliest_date / latest_date: the stated window as calendar dates YYYY-MM-DD.
  "third quarter of 2015" -> 2015-07-01 to 2015-09-30. "first half of 2016" -> 2016-01-01 to 2016-06-30.
  "mid-2014" -> 2014-05-01 to 2014-08-31. "early 2012" -> 2012-01-01 to 2012-03-31. "late 2012" -> 2012-10-01 to 2012-12-31.
  "summer" -> June 1 to August 31; "fall" -> September 1 to November 30; "spring" -> March 1 to May 31.
  "by the end of 2013" / "by year-end" / "before September 30" -> earliest_date null, latest_date the stated end.
  "in the third or fourth quarter" -> from the start of the third to the end of the fourth quarter.
  "within 60 days" / "in approximately three months" -> count from the announcement date: both dates equal to that day
  for "approximately", earliest null for "within".
  A year that is not written is the first one that makes the window end on or after the announcement date.
- basis: "calendar" for calendar periods, "fiscal" if the text says fiscal quarter/year (then convert only if the fiscal
  year end is stated in the sentences; otherwise treat the fiscal period as a calendar period and still mark "fiscal"),
  "relative" for windows counted from the announcement.
- quote: the few words that state the timing, copied from the text.
Return one result per document, as structured output."""


def sentences(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    flat = re.sub(r"\s+", " ", text[:40_000])
    out = []
    for m in TIMING.finditer(flat):
        a = max(flat.rfind(". ", 0, m.start()) + 2, m.start() - 350, 0)
        b = flat.find(". ", m.end())
        b = min(b + 1 if b != -1 else len(flat), m.end() + 350)
        s = flat[a:b].strip()
        if WHEN.search(s) and not any(s in o or o in s for o in out):
            out.append(s)
        if len(out) >= 4:
            break
    return "\n".join(f"- {s}" for s in out)


def prepare(_args) -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")[["deal_id", "announced_date"]]
    t = pd.read_parquet(INTERIM / "text.parquet")[["deal_id", "press_release", "has_text"]]
    d = d.merge(t, on="deal_id", how="inner")
    d = d[d["has_text"].fillna(False)]
    docs = {}
    for r in d.itertuples(index=False):
        s = sentences(r.press_release)
        if s:
            docs[r.deal_id] = f"### DOCUMENT doc_id={r.deal_id}\nAnnouncement date: {pd.Timestamp(r.announced_date).date()}\nSentences:\n{s}\n"
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "docs.json").write_text(json.dumps(docs))
    print(f"{len(d):,} deals with text; {len(docs):,} have a sentence about expected closing")


def run(args) -> None:
    docs = json.loads((WORK / "docs.json").read_text())
    out = WORK / "out" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(docs)
    if args.sample:
        rng = np.random.default_rng(args.seed)
        ids = sorted(rng.choice(ids, size=min(args.sample, len(ids)), replace=False).tolist())
    todo = [i for i in ids if not (out / f"{i}.json").exists()]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    schema = json.dumps({"type": "object", "properties": {"results": {"type": "array", "items": SCHEMA}}, "required": ["results"]})
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", INSTRUCTIONS, "--json-schema", schema]
    t0 = time.time()
    stat = {"ok": 0, "fail": 0, "cost": 0.0}

    def one(b):
        doc = "\n".join(docs[i] for i in b)
        for attempt in range(4):
            try:
                r = subprocess.run(cmd, input=doc, capture_output=True, text=True, timeout=600, cwd=str(WORK))
                env = json.loads(r.stdout)
                if env.get("is_error"):
                    raise RuntimeError(str(env.get("result"))[:200])
                res = env.get("structured_output") or json.loads(env.get("result") or "{}")
                n = 0
                for o in res.get("results", []):
                    if o.get("doc_id") in b:
                        (out / f"{o['doc_id']}.json").write_text(json.dumps(o)); n += 1
                stat["cost"] += float(env.get("total_cost_usd") or 0)
                if n:
                    return n
            except Exception as e:
                msg = str(e).lower()
                time.sleep(120 if ("limit" in msg or "overloaded" in msg or "429" in msg) else 5 * (attempt + 1))
        return 0

    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(one, batches), 1):
            stat["ok"] += n; stat["fail"] += (n == 0)
            if i % 25 == 0 or i == len(batches):
                print(f"[{i}/{len(batches)}] saved={stat['ok']} failed_batches={stat['fail']} cost=${stat['cost']:.2f} elapsed={(time.time() - t0) / 60:.1f} min", flush=True)


def load(run_name: str) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in (WORK / "out" / run_name).glob("*.json")]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["doc_id", "stated", "earliest_date", "latest_date", "basis", "quote"])


def to_days(df: pd.DataFrame, ann: pd.Series) -> pd.DataFrame:
    """Guidance in days from announcement: midpoint of the window, or the only bound given."""
    a = df["doc_id"].map(ann)
    lo = pd.to_datetime(df["earliest_date"], errors="coerce")
    hi = pd.to_datetime(df["latest_date"], errors="coerce")
    lo_d, hi_d = (lo - a).dt.days, (hi - a).dt.days
    mid = np.where(lo_d.notna() & hi_d.notna(), (lo_d.clip(lower=0) + hi_d) / 2, np.where(hi_d.notna(), hi_d, lo_d))
    out = pd.DataFrame({"deal_id": df["doc_id"], "guide_stated": df["stated"].fillna(False).astype(bool), "guide_lo_days": lo_d, "guide_hi_days": hi_d,
                        "guide_days": mid, "guide_basis": df["basis"], "guide_window": np.where(lo_d.notna() & hi_d.notna(), "window", np.where(hi_d.notna(), "by a date", "other"))})
    bad = ~out["guide_stated"] | out["guide_days"].isna() | (out["guide_days"] < 1) | (out["guide_days"] > 1500)
    out.loc[bad, ["guide_days", "guide_lo_days", "guide_hi_days"]] = np.nan
    out.loc[bad, "guide_stated"] = False
    return out


def finalize(args) -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    ann = d.set_index("deal_id")["announced_date"]
    main, strong = load(args.run), load(args.reference)
    G = to_days(main, ann)
    G.to_parquet(INTERIM / "closing_guidance.parquet", index=False)
    docs = json.loads((WORK / "docs.json").read_text())
    n_text = int(pd.read_parquet(INTERIM / "text.parquet")["has_text"].fillna(False).sum())
    L = ["# Management guidance on closing: extraction and validation", "",
         f"Source: the announcement press release (EDGAR). Sentences that state an expected closing time are found by pattern ({len(docs):,} of {n_text:,} deals with text), "
         f"then read by Claude ({args.run}) into a date window. The reader sees the announcement date and the sentences only.", "",
         f"- Deals with usable guidance: {int(G['guide_stated'].sum()):,} ({G['guide_stated'].sum() / n_text:.0%} of deals with text).",
         f"- Form: " + ", ".join(f"{k} {v}" for k, v in G.loc[G['guide_stated'], 'guide_window'].value_counts().items()) + "; "
         f"fiscal-period wording in {int((G.loc[G['guide_stated'], 'guide_basis'] == 'fiscal').sum())} deals.",
         f"- Guidance in days from announcement (midpoint of the window, or the stated bound): median {G['guide_days'].median():.0f}, "
         f"10th to 90th percentile {G['guide_days'].quantile(0.1):.0f} to {G['guide_days'].quantile(0.9):.0f}.", ""]
    if len(strong):
        S = to_days(strong, ann).set_index("deal_id")
        m = G.set_index("deal_id").reindex(S.index)
        both = m["guide_stated"].fillna(False) & S["guide_stated"]
        diff = (m.loc[both, "guide_days"] - S.loc[both, "guide_days"]).abs()
        L += [f"## Validation against the strongest model ({args.reference}, {len(S)} deals, independent reading)", "",
              f"- Agreement on whether guidance is stated: {(m['guide_stated'].fillna(False) == S['guide_stated']).mean():.0%}.",
              f"- Where both give a window ({int(both.sum())} deals): identical within 7 days in {(diff <= 7).mean():.0%}, within 31 days in {(diff <= 31).mean():.0%}; "
              f"median absolute difference {diff.median():.0f} days.", "",
              "The guidance is the benchmark for the time-to-closing models. One separately labelled ladder step also uses it as an input; no other model sees it.", ""]
    (ROOT / "results/closing_guidance_validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--run", default="sonnet"); r.add_argument("--model", default="sonnet"); r.add_argument("--batch-size", type=int, default=10)
    r.add_argument("--workers", type=int, default=6); r.add_argument("--sample", type=int, default=0); r.add_argument("--seed", type=int, default=15071)
    f = sub.add_parser("finalize")
    f.add_argument("--run", default="sonnet"); f.add_argument("--reference", default="strong")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "finalize": finalize}[a.cmd](a)

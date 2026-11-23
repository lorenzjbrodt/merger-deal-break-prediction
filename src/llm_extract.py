"""LLM extraction of deal terms from the EDGAR announcement text.

The LLM is used only as a reader: it fills a fixed JSON schema with terms STATED in the
documents (null when a term is not stated). It is never asked to predict the outcome.

Steps
  prepare   build one input document per deal (press release + key agreement excerpts) and
            group documents into batch files under data/interim/llm_extract/batches/
  run-cli   run the extraction with headless Claude Code (`claude -p`), parallel, resumable,
            one cached JSON per document under data/interim/llm_extract/out/<run>/
  collect   import results produced by an earlier batch run into the same per-document
            cache
  features  turn the cache into a feature table data/interim/llm_terms.parquet
  validate  field-level agreement between two runs (e.g. small model vs strongest model)

All inputs and outputs live in git-ignored folders.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WORK = INTERIM / "llm_extract"
MAX_PRESS = 12_000
MAX_AGREEMENT = 24_000
WINDOW = 700

REG_ENUM = ["HSR", "foreign_antitrust", "CFIUS", "FCC", "banking", "insurance", "FERC_or_state_utility", "other"]
SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string"},
        "buyer_type": {"type": ["string", "null"], "enum": ["strategic", "financial_sponsor", "management_led", "consortium", "other", None]},
        "financing_condition": {"type": ["boolean", "null"]},
        "committed_financing_mentioned": {"type": ["boolean", "null"]},
        "regulatory_approvals": {"type": "array", "items": {"type": "string", "enum": REG_ENUM}},
        "target_termination_fee_usd_m": {"type": ["number", "null"]},
        "reverse_termination_fee_usd_m": {"type": ["number", "null"]},
        "go_shop": {"type": ["boolean", "null"]},
        "go_shop_days": {"type": ["number", "null"]},
        "structure": {"type": ["string", "null"], "enum": ["tender_offer", "one_step_merger", "other", None]},
        "minimum_tender_condition": {"type": ["boolean", "null"]},
        "target_shareholder_vote_required": {"type": ["boolean", "null"]},
        "acquirer_shareholder_vote_required": {"type": ["boolean", "null"]},
        "expected_closing_months": {"type": ["number", "null"]},
        "outside_date_months": {"type": ["number", "null"]},
        "voting_support_agreements": {"type": ["boolean", "null"]},
        "support_agreement_pct": {"type": ["number", "null"]},
        "collar": {"type": ["boolean", "null"]},
        "cvr": {"type": ["boolean", "null"]},
        "antitrust_covenant": {"type": ["string", "null"], "enum": ["hell_or_high_water", "divestiture_commitment", "efforts_only", None]},
        "specific_performance": {"type": ["boolean", "null"]},
        "cross_border_acquirer": {"type": ["boolean", "null"]},
        "acquirer_country": {"type": ["string", "null"]},
        "stated_premium_pct": {"type": ["number", "null"]},
    },
    "required": ["doc_id"],
}
FIELDS = [k for k in SCHEMA["properties"] if k != "doc_id"]

INSTRUCTIONS = """You are extracting deal terms from merger announcement documents filed with the SEC.

Rules
- Use ONLY the text of the document provided. Never use outside knowledge about the companies or the deal,
  and never guess what happened to the deal later.
- If a term is not stated in the text, return null for it (or an empty list for regulatory_approvals).
  Null means "not stated". Do not infer a value from what is typical.
- Do not predict or comment on whether the deal will close.

Fields (one object per document)
- doc_id: the id shown in the document header, copied exactly.
- buyer_type: "strategic" (operating company), "financial_sponsor" (private equity / investment fund),
  "management_led" (management or controlling holder buyout), "consortium" (several unrelated buyers), "other"; null if unclear.
- financing_condition: true if closing is conditioned on the buyer obtaining financing; false if the text says the
  deal is NOT subject to a financing condition; null if not stated.
- committed_financing_mentioned: true if debt/equity commitment letters or committed financing are mentioned;
  false if the text says the buyer will use cash on hand / no financing needed; null if not stated.
- regulatory_approvals: every approval the text says is required, from this list only:
  HSR (Hart-Scott-Rodino / U.S. antitrust), foreign_antitrust (EU, UK, China, other non-U.S. competition approval),
  CFIUS, FCC, banking (Federal Reserve, OCC, FDIC, state banking), insurance (state insurance regulators),
  FERC_or_state_utility (FERC, NRC, state public utility commissions), other (any other named regulator or
  unspecified "regulatory approvals").
- target_termination_fee_usd_m: fee payable BY THE TARGET, in USD millions; null if no amount is stated.
- reverse_termination_fee_usd_m: fee payable BY THE BUYER/PARENT, in USD millions; null if no amount is stated.
- go_shop: true if the target may actively solicit competing offers for a period; false if the text states a
  no-solicitation ("no-shop") covenant without a go-shop; null if not stated. go_shop_days: length in days, else null.
- structure: "tender_offer" (tender or exchange offer followed by a merger), "one_step_merger", "other"; null if not stated.
- minimum_tender_condition: true if a minimum tender condition is stated; null otherwise (false only if expressly waived/absent).
- target_shareholder_vote_required / acquirer_shareholder_vote_required: true if the text says that party's
  shareholders must approve; false if it says no vote of that party is required; null if not stated.
- expected_closing_months: months from the announcement to the expected closing stated in the text
  (e.g. announced in March, "expected to close in the third quarter" -> about 5); null if not stated.
- outside_date_months: months from signing to the outside / end / termination date (before extensions); null if not stated.
- voting_support_agreements: true if voting, support or tender agreements with shareholders are mentioned; null if not.
  support_agreement_pct: share of the target's votes/shares covered, in percent; null if not stated.
- collar: true if the exchange ratio or price has a collar; null if not stated. cvr: true if a contingent value right is part of the consideration; null if not stated.
- antitrust_covenant: "hell_or_high_water" (buyer must take any and all actions, including any divestiture or litigation),
  "divestiture_commitment" (buyer must accept divestitures/remedies up to a stated limit), "efforts_only"
  (reasonable best efforts or similar with no divestiture obligation stated); null if the covenant is not described.
- specific_performance: true if the text says a party may seek specific performance to force closing; null if not stated.
- cross_border_acquirer: true if the acquirer is organised or headquartered outside the United States; false if it is
  stated to be a U.S. company; null if not stated. acquirer_country: country as stated, else null.
- stated_premium_pct: the premium to the target's share price stated in the text (the first / headline one), in percent; null if not stated.
"""

KEY_PATTERNS = [
    r"termination fee", r"reverse termination|parent termination fee|buyer termination fee", r"\"?(outside|end|termination|drop[\s\-]dead) date\"?\s+(means|shall mean)|on or before [A-Z][a-z]+ \d{1,2}, \d{4}",
    r"financing (condition|commitment)|commitment letter|debt financing|equity financing", r"hart[\s\-]*scott|\bHSR\b", r"\bCFIUS\b|committee on foreign investment",
    r"antitrust|competition law|divest", r"specific performance", r"go[\s\-]shop|no[\s\-]shop period|solicit(ation)? period", r"minimum (tender )?condition",
    r"(stockholder|shareholder) approval|requisite (company )?vote", r"\bcollar\b|contingent value right", r"federal communications|federal reserve|insurance (commissioner|regulator|department)|\bFERC\b|public (utility|service) commission",
    r"voting agreement|support agreement|tender and support", r"conditions to (the )?(obligations|closing|merger)",
]
_KEY_RE = [re.compile(p, re.I) for p in KEY_PATTERNS]


def agreement_excerpts(text: str) -> str:
    """Key passages of the merger agreement: the opening plus windows around key terms,
    taken round-robin over the patterns until the character budget is used."""
    if not text:
        return ""
    spans = [(0, min(1500, len(text)))]
    hits = [[m.start() for m in rx.finditer(text)][:6] for rx in _KEY_RE]
    budget = MAX_AGREEMENT - 1500
    for rnd in range(6):
        for h in hits:
            if rnd < len(h) and budget > 0:
                a, b = max(0, h[rnd] - WINDOW), min(len(text), h[rnd] + WINDOW)
                if not any(a < e and b > s for s, e in spans):
                    spans.append((a, b)); budget -= b - a
    spans.sort()
    return "\n[...]\n".join(text[a:b] for a, b in spans)


def build_document(doc_id: str, announced: str, press: str, agreement: str) -> str:
    parts = [f"===== DOCUMENT {doc_id} =====", f"Announcement date: {announced}", "", "--- PRESS RELEASE / ANNOUNCEMENT ---", press[:MAX_PRESS]]
    ex = agreement_excerpts(agreement)
    if ex:
        parts += ["", "--- MERGER AGREEMENT (EXCERPTS) ---", ex]
    return "\n".join(parts) + f"\n===== END OF DOCUMENT {doc_id} =====\n"


def out_dir(run: str) -> Path:
    d = WORK / "out" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_prepare(args) -> None:
    text = pd.read_parquet(INTERIM / "text.parquet")
    deals = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "announced_date", "year"])
    t = deals.merge(text, on="deal_id")
    t = t[t["has_text"]].sort_values(["year", "deal_id"], ascending=[False, True])
    if args.sample:
        t = t.sample(args.sample, random_state=args.seed)
    done = {p.stem for p in out_dir(args.run).glob("*.json")}
    t = t[~t["deal_id"].isin(done)]
    bdir = WORK / "batches" / args.run
    bdir.mkdir(parents=True, exist_ok=True)
    for old in bdir.glob("batch_*.txt"):
        old.unlink()
    from multiprocessing import Pool
    jobs = [(r.deal_id, str(pd.Timestamp(r.announced_date).date()), r.press_release, r.agreement) for r in t.itertuples(index=False)]
    with Pool(8) as pool:
        docs = pool.starmap(build_document, jobs, chunksize=16)
    ids = t["deal_id"].tolist()
    manifest = []
    for b in range(0, len(docs), args.batch_size):
        path = bdir / f"batch_{b // args.batch_size:04d}.txt"
        path.write_text("\n".join(docs[b:b + args.batch_size]))
        manifest.append({"path": str(path), "doc_ids": ids[b:b + args.batch_size]})
    (bdir / "manifest.json").write_text(json.dumps(manifest))
    (WORK / "instructions.txt").write_text(INSTRUCTIONS)
    chars = sum(len(d) for d in docs)
    print(f"run={args.run}: {len(docs)} documents to do ({len(done)} cached), {len(manifest)} batches, "
          f"{chars / 1e6:.1f}m chars (~{chars / 4e6:.1f}m tokens)")


def save_result(run: str, obj: dict) -> bool:
    doc_id = obj.get("doc_id")
    if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", doc_id):
        return False
    clean = {k: obj.get(k) for k in FIELDS}
    if not isinstance(clean.get("regulatory_approvals"), list):
        clean["regulatory_approvals"] = []
    clean["regulatory_approvals"] = [r for r in clean["regulatory_approvals"] if r in REG_ENUM]
    (out_dir(run) / f"{doc_id}.json").write_text(json.dumps({"doc_id": doc_id, **clean}))
    return True


def cmd_run_cli(args) -> None:
    """Headless Claude Code (`claude -p`), one call per batch file, parallel and resumable.
    The instructions go in the system prompt, the document(s) on stdin; no tools are enabled."""
    import time
    manifest = json.loads((WORK / "batches" / args.run / "manifest.json").read_text())
    done = {p.stem for p in out_dir(args.run).glob("*.json")}
    manifest = [b for b in manifest if not set(b["doc_ids"]) <= done]
    schema = json.dumps({"type": "object", "properties": {"results": {"type": "array", "items": SCHEMA}}, "required": ["results"]})
    system = INSTRUCTIONS + "\nReturn one result object per document, as structured output."
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--output-format", "json", "--system-prompt", system, "--json-schema", schema]
    stats = {"ok": 0, "fail": 0, "cost": 0.0}
    t0 = time.time()

    def one(b: dict) -> int:
        doc = Path(b["path"]).read_text()
        for attempt in range(4):
            try:
                r = subprocess.run(cmd, input=doc, capture_output=True, text=True, timeout=600, cwd=str(WORK))
                env = json.loads(r.stdout)
                if env.get("is_error"):
                    raise RuntimeError(str(env.get("result"))[:200])
                res = env.get("structured_output") or json.loads(env.get("result") or "{}")
                n = sum(save_result(args.run, o) for o in res.get("results", []) if o.get("doc_id") in b["doc_ids"])
                stats["cost"] += float(env.get("total_cost_usd") or 0)
                if n:
                    return n
            except Exception as e:
                msg = str(e).lower()
                time.sleep(120 if ("limit" in msg or "overloaded" in msg or "429" in msg) else 5 * (attempt + 1))
        return 0

    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(one, manifest), 1):
            stats["ok"] += n; stats["fail"] += (n == 0)
            if i % 50 == 0 or i == len(manifest):
                print(f"[{i}/{len(manifest)}] saved={stats['ok']} failed_batches={stats['fail']} cost=${stats['cost']:.2f} "
                      f"elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
    print("done; rerun `prepare` then `run-cli` to retry anything missing", flush=True)


def cmd_collect(args) -> None:
    """Import structured results from a workflow journal (one JSON object per line)."""
    valid = set()
    mf = WORK / "batches" / args.run / "manifest.json"
    if mf.exists():
        valid = {d for b in json.loads(mf.read_text()) for d in b["doc_ids"]}
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if "doc_id" in o and any(k in o for k in FIELDS):
                if not valid or o["doc_id"] in valid:
                    n += save_result(args.run, o)
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and '"doc_id"' in o:
            try:
                walk(json.loads(o))
            except Exception:
                pass

    for path in args.journal:
        for line in Path(path).read_text().splitlines():
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"imported {n} result objects into {out_dir(args.run)}; cached documents: {len(list(out_dir(args.run).glob('*.json')))}")


def load_run(run: str) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(out_dir(run).glob("*.json"))]
    return pd.DataFrame(rows)


def to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature table. Booleans -> 1/0/NaN (NaN = not stated)."""
    F = pd.DataFrame({"deal_id": df["doc_id"]})
    for c in ["financing_condition", "committed_financing_mentioned", "go_shop", "minimum_tender_condition",
              "target_shareholder_vote_required", "acquirer_shareholder_vote_required", "voting_support_agreements",
              "collar", "cvr", "specific_performance", "cross_border_acquirer"]:
        F[f"llm_{c}"] = df[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
    for c in ["target_termination_fee_usd_m", "reverse_termination_fee_usd_m", "go_shop_days", "expected_closing_months",
              "outside_date_months", "support_agreement_pct", "stated_premium_pct"]:
        F[f"llm_{c}"] = pd.to_numeric(df[c], errors="coerce")
    for v in ["strategic", "financial_sponsor", "management_led", "consortium"]:
        F[f"llm_buyer_{v}"] = (df["buyer_type"] == v).astype(float).where(df["buyer_type"].notna())
    F["llm_structure_tender"] = (df["structure"] == "tender_offer").astype(float).where(df["structure"].notna())
    for v in ["hell_or_high_water", "divestiture_commitment", "efforts_only"]:
        F[f"llm_antitrust_{v}"] = (df["antitrust_covenant"] == v).astype(float)
    regs = df["regulatory_approvals"].map(lambda v: v if isinstance(v, (list, np.ndarray)) else [])
    for r in REG_ENUM:
        F[f"llm_reg_{r.lower()}"] = regs.map(lambda v, r=r: float(r in list(v)))
    F["llm_n_regulatory_approvals"] = regs.map(lambda v: float(len(list(v))))
    return F


def cmd_features(args) -> None:
    df = load_run(args.run)
    F = to_features(df)
    kf = WORK / "kept_fields.json"
    if kf.exists():   # drop features built from fields that failed validation
        keep = set(json.loads(kf.read_text()))
        prefix = {"buyer_type": ["llm_buyer_"], "structure": ["llm_structure_"], "antitrust_covenant": ["llm_antitrust_"],
                  "regulatory_approvals": ["llm_n_regulatory_approvals"]}
        prefix.update({f"reg_{r}": [f"llm_reg_{r.lower()}"] for r in REG_ENUM})
        drop = []
        for field in FIELDS + [f"reg_{r}" for r in REG_ENUM]:
            if field in keep:
                continue
            pats = prefix.get(field, [f"llm_{field}"])
            drop += [c for c in F.columns if any(c == p or (p.endswith("_") and c.startswith(p)) for p in pats)]
        F = F.drop(columns=sorted(set(drop)))
    F.to_parquet(INTERIM / "llm_terms.parquet", index=False)
    print(f"{len(F)} deals, {F.shape[1] - 1} LLM term features -> data/interim/llm_terms.parquet")
    print(", ".join(c for c in F.columns if c != "deal_id"))


def cmd_validate(args) -> None:
    a, b = load_run(args.run).set_index("doc_id"), load_run(args.reference).set_index("doc_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    rows = []
    for f in FIELDS:
        if f == "regulatory_approvals":
            sa = a[f].map(lambda v: frozenset(v)); sb = b[f].map(lambda v: frozenset(v))
            agree = (sa == sb).mean()
            jac = np.mean([len(x & y) / len(x | y) if (x | y) else 1.0 for x, y in zip(sa, sb)])
            rows.append({"field": f, "n": len(ids), "agreement": agree, "both_null": float(((sa.map(len) == 0) & (sb.map(len) == 0)).mean()),
                         "agreement_when_either_stated": float((sa == sb)[(sa.map(len) > 0) | (sb.map(len) > 0)].mean()),
                         "note": f"exact match of the whole list; mean Jaccard {jac:.2f}. Judged per approval below"})
            for r in REG_ENUM:      # each approval type as its own yes/no field
                xa, xb = sa.map(lambda v, r=r: r in v), sb.map(lambda v, r=r: r in v)
                either = xa | xb
                rows.append({"field": f"reg_{r}", "n": len(ids), "agreement": float((xa == xb).mean()), "both_null": float((~either).mean()),
                             "agreement_when_either_stated": float((xa == xb)[either].mean()) if either.any() else np.nan,
                             "note": "approval listed: yes / no"})
            continue
        x, y = a[f], b[f]
        xn, yn = x.isna(), y.isna()
        if pd.api.types.is_numeric_dtype(pd.to_numeric(x.dropna(), errors="coerce")) and f.endswith(("_usd_m", "_days", "_months", "_pct")):
            xv, yv = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            same = (xn & yn) | ((~xn & ~yn) & ((xv - yv).abs() <= np.maximum(0.1 * yv.abs(), 1.0)))
            note = "numeric: equal within 10% or 1 unit"
        else:
            same = (xn & yn) | ((~xn & ~yn) & (x.astype(str) == y.astype(str)))
            note = ""
        either = ~(xn & yn)
        rows.append({"field": f, "n": len(ids), "agreement": float(same.mean()), "both_null": float((xn & yn).mean()),
                     "agreement_when_either_stated": float(same[either].mean()) if either.any() else np.nan, "note": note})
    R = pd.DataFrame(rows)
    R["keep"] = (R["agreement"] >= args.min_agreement) & ((R["agreement_when_either_stated"] >= args.min_stated) | R["agreement_when_either_stated"].isna())
    (WORK / "kept_fields.json").write_text(json.dumps(R.loc[R["keep"], "field"].tolist()))
    lines = ["# LLM term extraction: validation against the strongest model", "",
             f"{len(ids)} randomly sampled documents were extracted twice: by the small production model (run `{args.run}`) and "
             f"by the strongest model (run `{args.reference}`). Agreement = identical value, both-null counting as agreement; "
             "numeric fields agree when within 10% or one unit. A field is kept as a model feature when overall agreement is at "
             f"least {args.min_agreement:.0%} and agreement on documents where either model found a value is at least {args.min_stated:.0%}.", "",
             "| Field | Agreement | Both not stated | Agreement when either stated | Kept | Note |", "|---|---:|---:|---:|---|---|"]
    for r in R.itertuples():
        st = "n/a" if pd.isna(r.agreement_when_either_stated) else f"{r.agreement_when_either_stated:.0%}"
        lines.append(f"| {r.field} | {r.agreement:.0%} | {r.both_null:.0%} | {st} | {'yes' if r.keep else 'dropped'} | {r.note} |")
    (ROOT / "results/llm_extraction_validation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--run", default="haiku"); p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--sample", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run-cli"); p.add_argument("--run", default="haiku"); p.add_argument("--model", default="haiku")
    p.add_argument("--workers", type=int, default=6); p.set_defaults(fn=cmd_run_cli)
    p = sub.add_parser("collect"); p.add_argument("--run", default="haiku"); p.add_argument("journal", nargs="+"); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("features"); p.add_argument("--run", default="haiku"); p.set_defaults(fn=cmd_features)
    p = sub.add_parser("validate"); p.add_argument("--run", default="haiku"); p.add_argument("--reference", default="strong")
    p.add_argument("--min-agreement", type=float, default=0.85); p.add_argument("--min-stated", type=float, default=0.6); p.set_defaults(fn=cmd_validate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

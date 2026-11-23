"""Outcome taxonomy for cancelled deals (labels and analysis only, never model features).

Capital IQ's "Deal Resolution", "Transaction Comments", "Pre-Deal Situation" and "Related
Transactions" were written or updated after the fact. They are used here ONLY to classify
why a cancelled deal broke and to flag deals whose exported terms reflect a revised offer.
Nothing produced by this module is allowed into a feature matrix.

Classes: competing_bid (superior competing bid accepted), regulatory (regulatory or antitrust
block), financing (financing failure), shareholder_vote (vote failed / insufficient tenders),
buyer_walked_mac (buyer walked or MAC dispute), mutual_other (mutual termination or other
stated reason), unknown (no reason stated).

Steps
  prepare    batch files for LLM classification of every Cancelled deal in the sample
  collect    import LLM results from a workflow journal into a per-deal cache
  finalize   combine LLM labels with rule checks -> data/interim/outcomes.parquet, results/outcome_taxonomy.md
  review     print N random cases (text + label) for manual re-reading
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WORK = INTERIM / "outcomes"
CLASSES = ["competing_bid", "regulatory", "financing", "shareholder_vote", "buyer_walked_mac", "mutual_other", "unknown"]
CLASS_LABEL = {"competing_bid": "Superior competing bid accepted", "regulatory": "Regulatory or antitrust block",
               "financing": "Financing failure", "shareholder_vote": "Shareholder vote failed / insufficient tenders",
               "buyer_walked_mac": "Buyer walked or MAC dispute", "mutual_other": "Mutual termination or other stated reason",
               "unknown": "Unknown (no reason stated)"}

INSTRUCTIONS = """You are classifying why announced M&A deals were cancelled, from a data vendor's deal notes.

For each case, read the notes and return:
- case_id: copied exactly from the case header.
- reason: exactly one of
    competing_bid     the target accepted (or the deal lost to) a superior / higher competing offer from another bidder,
                      or the original buyer was outbid. Includes a buyer declining to match a topping bid.
    regulatory        blocked, challenged or abandoned because of antitrust or another regulator / government
                      (FTC, DOJ, European Commission, CFIUS, FCC, banking or insurance regulators, court injunction sought by a regulator),
                      including failure to obtain a required regulatory approval in time.
    financing         the buyer could not obtain or complete financing.
    shareholder_vote  target or acquirer shareholders voted the deal down, the vote could not be obtained, or too few shares were tendered.
    buyer_walked_mac  the buyer terminated or walked away, alleged a material adverse change / breach by the target,
                      or the parties litigated over the buyer's obligation to close. Also a buyer withdrawing after due diligence or market deterioration.
    mutual_other      a stated reason that fits none of the above (mutual termination citing market conditions, the target
                      terminated for a non-bid reason, the target's board changed its recommendation without a rival bid, the target went bankrupt, and so on).
    unknown           the notes give no reason at all (for example only "X cancelled the acquisition of Y on <date>").
  If several reasons are given, choose the one the notes present as the decisive cause. If a rival bid exists AND the deal was terminated
  so the target could take it, choose competing_bid.
- terminated_by: "target", "buyer", "mutual", "regulator_or_court", "unknown".
- topping_bidder_was_different_party: true if the target ended up agreeing a deal with a different buyer at a higher price; false or null otherwise.
- terms_revised_after_announcement: true if the notes say the offer price or consideration was changed (raised, lowered, amended)
  after the original announcement of THIS deal; false if not mentioned.
- evidence: a short quote (max 25 words) from the notes that supports the reason; empty string for unknown.

Use only the notes provided. Do not use outside knowledge about these companies.
"""
SCHEMA = {"type": "object", "properties": {
    "case_id": {"type": "string"}, "reason": {"type": "string", "enum": CLASSES},
    "terminated_by": {"type": "string", "enum": ["target", "buyer", "mutual", "regulator_or_court", "unknown"]},
    "topping_bidder_was_different_party": {"type": ["boolean", "null"]},
    "terms_revised_after_announcement": {"type": ["boolean", "null"]},
    "evidence": {"type": "string"}}, "required": ["case_id", "reason"]}

# Rule checks: independent keyword evidence in the resolution text (first match in this order wins).
RULES = [
    ("competing_bid", r"superior (proposal|offer|bid)|competing (bid|offer|proposal)|higher (bid|offer)|topping (bid|offer)|rival (bid|offer)|"
                      r"in favou?r of|accepted (an?|the) (offer|proposal|bid) from|outbid|to enter into .{0,60}agreement with|unsolicited (offer|proposal|bid) from"),
    ("regulatory", r"antitrust|anti-trust|federal trade commission|\bFTC\b|department of justice|\bDOJ\b|european commission|\bCFIUS\b|"
                   r"regulatory (approval|clearance|opposition|hurdle|concern)|regulator|\bFCC\b|federal reserve|injunction|competition (authority|bureau|commission)"),
    ("financing", r"financing|unable to (obtain|secure|raise|arrange) (the )?(necessary )?(funding|funds|debt)|credit markets?|lenders?"),
    ("shareholder_vote", r"(shareholders|stockholders|unitholders) (voted against|rejected|did not approve|failed to approve|voted down)|"
                         r"(failed|failure) to (obtain|receive|get) .{0,40}(shareholder|stockholder|unitholder) approval|not approved by .{0,30}(shareholders|stockholders)|"
                         r"insufficient (number of )?(shares|votes|tenders)|minimum (tender )?condition|did not receive (the )?(requisite|required|sufficient)"),
    ("buyer_walked_mac", r"material adverse|\bMAC\b|\bMAE\b|breach|lawsuit|sued|litigation|walked away|withdrew (its|the) (offer|bid|proposal)|due diligence"),
    ("mutual_other", r"mutual|market conditions|bankrupt|chapter 11"),
]
REVISED_RE = re.compile(
    r"(revised|increased|raised|sweetened|improved|reduced|lowered|decreased|cut|amended) (its |the |their )?(offer|bid|proposal|purchase price|price|consideration|terms)|"
    r"(offer|bid|price|consideration|exchange ratio) (was|has been|were) (subsequently |later )?(revised|increased|raised|amended|reduced|lowered|decreased)|"
    r"(revised|amended|increased|improved|sweetened) (offer|bid|proposal|terms|agreement)|amended (the |its )?(merger |definitive )?agreement|"
    r"amendment to the (merger |definitive )?agreement|from \$[\d.,]+ (per share )?to \$[\d.,]+ per share", re.I)


def rule_label(resolution: str) -> str:
    t = resolution if isinstance(resolution, str) else ""
    for name, pat in RULES:
        if re.search(pat, t, re.I):
            return name
    return "unknown"


def rival_deals(sample: pd.DataFrame) -> pd.DataFrame:
    """For each cancelled sample deal: is there another majority deal for the same target, by a different
    buyer, announced between this deal's announcement and 45 days after its cancellation? Uses the full export."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare import load_export
    raw = load_export(ROOT / "data/raw/ciq_transactions_v2.xls")
    raw = raw[(raw["pct_sought"] >= 50) & raw["company_id"].notna()][["deal_id", "company_id", "announced_date", "buyers"]]
    by_co = {k: g for k, g in raw.groupby("company_id")}
    rows = []
    for r in sample[sample["label"] == 1].itertuples(index=False):
        g = by_co.get(r.company_id)
        hit = False
        if g is not None and pd.notna(r.cancelled_date):
            other = g[(g["deal_id"] != r.deal_id) & (g["announced_date"] >= r.announced_date) &
                      (g["announced_date"] <= r.cancelled_date + pd.Timedelta(days=45))]
            mine = str(r.buyers).lower()[:25]
            hit = any(str(b).lower()[:25] != mine for b in other["buyers"])
        rows.append({"deal_id": r.deal_id, "rival_deal": hit})
    return pd.DataFrame(rows)


def load_deals() -> pd.DataFrame:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    need = ["deal_resolution", "transaction_comments", "pre_deal_situation", "related_transactions"]
    if any(c not in d.columns for c in need):
        raise SystemExit("deals.parquet lacks the v2 text columns; run prepare.py on the extended export first")
    return d


def case_text(r) -> str:
    def f(v, n):
        return (v if isinstance(v, str) else "")[:n]
    return (f"===== CASE {r.deal_id} =====\nAnnounced: {pd.Timestamp(r.announced_date).date()}\n"
            f"DEAL RESOLUTION: {f(r.deal_resolution, 2500)}\n\nTRANSACTION COMMENTS: {f(r.transaction_comments, 5000)}\n\n"
            f"PRE-DEAL SITUATION: {f(r.pre_deal_situation, 1200)}\n\nRELATED TRANSACTIONS: {f(r.related_transactions, 600)}\n"
            f"===== END OF CASE {r.deal_id} =====\n")


def cmd_prepare(args) -> None:
    d = load_deals()
    c = d[d["label"] == 1]
    out = WORK / "out"; out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out.glob("*.json")}
    c = c[~c["deal_id"].isin(done)]
    bdir = WORK / "batches"; bdir.mkdir(parents=True, exist_ok=True)
    for old in bdir.glob("batch_*.txt"):
        old.unlink()
    rows = list(c.itertuples(index=False)); manifest = []
    for b in range(0, len(rows), args.batch_size):
        path = bdir / f"batch_{b // args.batch_size:03d}.txt"
        path.write_text("\n".join(case_text(r) for r in rows[b:b + args.batch_size]))
        manifest.append({"path": str(path), "case_ids": [r.deal_id for r in rows[b:b + args.batch_size]]})
    (bdir / "manifest.json").write_text(json.dumps(manifest))
    (WORK / "instructions.txt").write_text(INSTRUCTIONS)
    print(f"{len(rows)} cancelled deals to classify ({len(done)} cached) in {len(manifest)} batches")


def cmd_run_cli(args) -> None:
    """Classify with headless Claude Code (`claude -p`), one call per batch, resumable."""
    import subprocess, time
    from concurrent.futures import ThreadPoolExecutor
    out = WORK / "out"; out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((WORK / "batches" / "manifest.json").read_text())
    schema = json.dumps({"type": "object", "properties": {"results": {"type": "array", "items": SCHEMA}}, "required": ["results"]})
    cmd = ["claude", "-p", "--model", args.model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--output-format", "json", "--system-prompt",
           INSTRUCTIONS + "\nReturn one result object per case, as structured output.", "--json-schema", schema]

    def one(b: dict) -> int:
        if all((out / f"{c}.json").exists() for c in b["case_ids"]):
            return len(b["case_ids"])
        for attempt in range(4):
            try:
                r = subprocess.run(cmd, input=Path(b["path"]).read_text(), capture_output=True, text=True, timeout=900, cwd=str(WORK))
                env = json.loads(r.stdout)
                if env.get("is_error"):
                    raise RuntimeError(str(env.get("result"))[:200])
                res = env.get("structured_output") or json.loads(env.get("result") or "{}")
                n = 0
                for o in res.get("results", []):
                    if o.get("case_id") in b["case_ids"] and o.get("reason") in CLASSES:
                        (out / f"{o['case_id']}.json").write_text(json.dumps(o)); n += 1
                if n:
                    return n
            except Exception as e:
                time.sleep(120 if "limit" in str(e).lower() else 5 * (attempt + 1))
        return 0

    with ThreadPoolExecutor(args.workers) as ex:
        n = sum(ex.map(one, manifest))
    print(f"classified {n} cases; cached: {len(list(out.glob('*.json')))}")


def cmd_collect(args) -> None:
    out = WORK / "out"; out.mkdir(parents=True, exist_ok=True)
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if "case_id" in o and o.get("reason") in CLASSES and re.fullmatch(r"[A-Za-z0-9_]+", str(o["case_id"])):
                (out / f"{o['case_id']}.json").write_text(json.dumps(o)); n += 1
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and '"case_id"' in o:
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
    print(f"imported {n} classifications; cached: {len(list(out.glob('*.json')))}")


def cmd_finalize(args) -> None:
    d = load_deals()
    llm = pd.DataFrame([json.loads(p.read_text()) for p in (WORK / "out").glob("*.json")])
    llm = llm.rename(columns={"case_id": "deal_id", "reason": "reason_llm"}) if len(llm) else pd.DataFrame(columns=["deal_id", "reason_llm"])
    o = d[["deal_id", "label", "year", "deal_resolution", "transaction_comments"]].merge(llm, on="deal_id", how="left")
    # rules read the resolution text plus the tail of the comments, where the outcome is usually described
    rule_text = o["deal_resolution"].fillna("").astype(str) + " " + o["transaction_comments"].fillna("").astype(str).str[-1200:]
    o["reason_rule"] = rule_text.map(rule_label).where(o["label"] == 1)
    o = o.merge(rival_deals(d), on="deal_id", how="left")
    o["rival_deal"] = o["rival_deal"].fillna(False).astype(bool)
    # Final label: the LLM reading, unless a manual review overrides it.
    o["reason"] = o["reason_llm"].where(o["label"] == 1)
    mr = WORK / "manual_review.json"
    review = json.loads(mr.read_text()) if mr.exists() else {}
    for k, v in review.items():
        o.loc[o["deal_id"] == k, "reason"] = v
    o.loc[(o["label"] == 1) & o["reason"].isna(), "reason"] = o["reason_rule"]
    # Structured rule check: the notes of a topped deal are often silent on the reason ("X cancelled the
    # acquisition of Y"). If the export itself shows another buyer's majority deal for the same target
    # announced between this deal's announcement and 45 days after its cancellation, treat the break as a
    # competing bid when the notes gave no specific adverse reason.
    relabel = (o["label"] == 1) & o["rival_deal"] & o["reason"].isin(["unknown", "mutual_other"]) & ~o["deal_id"].isin(review)
    o["relabelled_by_rival_rule"] = relabel
    o.loc[relabel, "reason"] = "competing_bid"
    o["outcome3"] = np.where(o["label"] == 0, "closed", np.where(o["reason"] == "competing_bid", "topped", "adverse_break"))
    o["label_adverse"] = (o["outcome3"] == "adverse_break").astype(int)
    # revised terms: comments regex for every deal, OR the LLM flag for cancelled deals
    rev_rx = o["transaction_comments"].map(lambda t: bool(REVISED_RE.search(t)) if isinstance(t, str) else False)
    rev_llm = o.get("terms_revised_after_announcement", pd.Series(False, index=o.index)).fillna(False).astype(bool)
    o["revised_terms"] = rev_rx | rev_llm
    keep = ["deal_id", "reason", "reason_llm", "reason_rule", "rival_deal", "relabelled_by_rival_rule", "outcome3", "label_adverse", "revised_terms"]
    for c in ["terminated_by", "topping_bidder_was_different_party", "evidence"]:
        if c in o.columns:
            keep.append(c)
    o[keep].to_parquet(INTERIM / "outcomes.parquet", index=False)

    c = o[o["label"] == 1]
    test = c["year"] >= 2019
    L = ["# Outcome taxonomy for cancelled deals", "",
         "Source: Capital IQ deal notes (Deal Resolution, Transaction Comments, Pre-Deal Situation, Related Transactions). "
         "These fields are written after the fact. They are used for labels and analysis only and never enter a feature matrix.", "",
         f"Cancelled deals in the sample: {len(c)}. Classified by an LLM reading of the notes, checked against keyword rules, "
         "and validated by re-reading a random sample (below).", "",
         "| Reason | All | Train 2001-2018 | Test 2019-2025 | Share of breaks |", "|---|---:|---:|---:|---:|"]
    for k in CLASSES:
        m = c["reason"] == k
        L.append(f"| {CLASS_LABEL[k]} | {int(m.sum())} | {int((m & ~test).sum())} | {int((m & test).sum())} | {m.mean():.0%} |")
    L += ["", "## Targets", "",
          "| Target | Positives (all) | Train | Test | Rate |", "|---|---:|---:|---:|---:|",
          f"| (a) Any break | {int(o.label.sum())} | {int(o.label[o.year <= 2018].sum())} | {int(o.label[o.year >= 2019].sum())} | {o.label.mean():.1%} |",
          f"| (b) Adverse break (break excluding superior competing bids) | {int(o.label_adverse.sum())} | {int(o.label_adverse[o.year <= 2018].sum())} | "
          f"{int(o.label_adverse[o.year >= 2019].sum())} | {o.label_adverse.mean():.1%} |", "",
          "Three-class view: " + ", ".join(f"{k} {int(v)}" for k, v in o["outcome3"].value_counts().items()) + ".", ""]
    both = c.dropna(subset=["reason_llm"])
    agree = (both["reason_llm"] == both["reason_rule"]).mean() if len(both) else float("nan")
    cb = ((both["reason_llm"] == "competing_bid") == (both["reason_rule"] == "competing_bid")).mean() if len(both) else float("nan")
    L += ["## Rule check", "",
          f"Keyword rules applied to the Deal Resolution text and the last 1,200 characters of the comments agree with the LLM label on {agree:.0%} of cancelled deals across the seven classes "
          f"and on {cb:.0%} for the binary question that defines the adverse-break target (competing bid or not). "
          "The rules are a coarse cross-check (first keyword match wins); the LLM label is the one used.", "",
          "| LLM label \\ rule label | " + " | ".join(CLASSES) + " |", "|---|" + "---:|" * len(CLASSES)]
    ct = pd.crosstab(both["reason_llm"], both["reason_rule"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    for k in CLASSES:
        L.append(f"| {k} | " + " | ".join(str(int(v)) for v in ct.loc[k]) + " |")
    n_rel = int(o["relabelled_by_rival_rule"].sum())
    L += ["", "## Rival-deal rule", "",
          f"{int(c['rival_deal'].sum()) if 'rival_deal' in c else int(o.rival_deal[o.label == 1].sum())} cancelled deals have another buyer's majority deal for the same target in the export, announced between "
          "the deal's announcement and 45 days after its cancellation. "
          f"{n_rel} of them had been labelled `unknown` or `mutual_other` from the notes and were relabelled as competing bids. "
          "All counts above are after this rule.", ""]
    vf = WORK / "validation_50.json"
    if vf.exists():
        v = json.loads(vf.read_text())
        L += ["", "## Validation by re-reading", "", v["summary"], ""]
    n_rev = int(o["revised_terms"].sum())
    L += ["", "## Deals whose exported terms may reflect a revised offer", "",
          f"{n_rev} of {len(o)} deals ({n_rev / len(o):.1%}) have notes that mention a revised, raised, lowered or amended offer after the "
          f"announcement ({int(o.revised_terms[o.label == 1].sum())} cancelled, {int(o.revised_terms[o.label == 0].sum())} closed). "
          "For these deals the exported offer price, premium and deal value can be the final terms, not the announcement terms. "
          "The sensitivity of the results to excluding them is reported in `results/robustness.md`.", ""]
    (ROOT / "results/outcome_taxonomy.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_review(args) -> None:
    d = load_deals()
    o = pd.read_parquet(INTERIM / "outcomes.parquet")
    c = d[d["label"] == 1].merge(o, on="deal_id").sample(args.n, random_state=args.seed)
    for r in c.itertuples(index=False):
        print(f"### {r.deal_id} | LLM={r.reason_llm} | RULE={r.reason_rule}\nRES: {str(r.deal_resolution)[:600]}\nCOM(tail): ...{str(r.transaction_comments)[-380:]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--batch-size", type=int, default=12); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run-cli"); p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_run_cli)
    p = sub.add_parser("collect"); p.add_argument("journal", nargs="+"); p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("finalize"); p.set_defaults(fn=cmd_finalize)
    p = sub.add_parser("review"); p.add_argument("--n", type=int, default=50); p.add_argument("--seed", type=int, default=7); p.set_defaults(fn=cmd_review)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

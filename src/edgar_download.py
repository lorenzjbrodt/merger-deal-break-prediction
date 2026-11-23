"""Download the target's SEC filings around each deal announcement.

For every deal in data/interim/deals.parquet:
  1. list the target's filings via the SEC submissions API
     (data.sec.gov/submissions/CIK##########.json plus the older "files" pages),
  2. keep forms 8-K, 425, DEFA14A, SC TO-C, SC 14D9, SC TO-T, PREM14A filed from one
     calendar day before to five business days after the announcement date,
  3. download each filing's complete submission text file and keep the primary document
     and the exhibits EX-99.* and EX-2.*.

Everything is cached under data/edgar_cache/ and the script is resumable: rerunning it
skips whatever is already on disk. Requests are globally throttled to <= 5 per second.

The SEC asks automated clients to identify themselves in the User-Agent header
("Name contact-address"). Set it in the environment before running:

    export SEC_USER_AGENT="Your Name your-contact-address"

If the variable is not set, a generic client name is sent instead.

Usage: python src/edgar_download.py [--limit N] [--workers 6]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/edgar_cache"
SUB_DIR = CACHE / "submissions"
FIL_DIR = CACHE / "filings"
MANIFEST = CACHE / "deal_filings.jsonl"

USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip() or "academic-research-script/1.0"
FORMS = ["8-K", "SC TO-T", "SC 14D9", "PREM14A", "SC TO-C", "425", "DEFA14A"]  # priority order
MAX_REQ_PER_SEC = 4.5          # hard ceiling requested: 5/s
MAX_FILINGS_PER_DEAL = 8
MAX_DOC_CHARS = 6_000_000      # skip absurdly large single documents
BINARY_EXT = (".pdf", ".jpg", ".jpeg", ".gif", ".png", ".xls", ".xlsx", ".zip", ".xml", ".xsd", ".js", ".css")


class RateLimiter:
    """Global minimum spacing between requests, shared by all worker threads."""

    def __init__(self, per_sec: float):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


LIMITER = RateLimiter(MAX_REQ_PER_SEC)
_local = threading.local()
STATS = {"requests": 0, "errors": 0, "throttled": 0}


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        _local.s = s
    return _local.s


def fetch(url: str, tries: int = 6) -> bytes | None:
    """GET with throttling and backoff. Returns None on 404."""
    delay = 2.0
    for attempt in range(tries):
        LIMITER.wait()
        try:
            r = session().get(url, timeout=60)
            STATS["requests"] += 1
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429):
                # SEC rate-limit / access response: back off hard.
                STATS["throttled"] += 1
                time.sleep(min(60 * (attempt + 1), 600))
            else:
                time.sleep(delay)
        except requests.RequestException:
            STATS["errors"] += 1
            time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"giving up on {url}")


def cik10(cik: str) -> str:
    return str(int(cik)).zfill(10)


def get_json_cached(url: str, path: Path) -> dict | None:
    if path.exists():
        txt = path.read_text()
        return json.loads(txt) if txt else None
    raw = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_bytes(raw or b"")
    tmp.replace(path)
    return json.loads(raw) if raw else None


def list_filings(cik: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Filings of `cik` with filingDate in [start, end], from the submissions API."""
    c = cik10(cik)
    sub = get_json_cached(f"https://data.sec.gov/submissions/CIK{c}.json", SUB_DIR / f"CIK{c}.json")
    if not sub:
        return []
    blocks = [sub.get("filings", {}).get("recent", {})]
    for f in sub.get("filings", {}).get("files", []):
        try:
            f_from, f_to = pd.Timestamp(f["filingFrom"]), pd.Timestamp(f["filingTo"])
        except Exception:
            f_from, f_to = pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")
        if f_from <= end and f_to >= start:
            page = get_json_cached(f"https://data.sec.gov/submissions/{f['name']}", SUB_DIR / f["name"])
            if page:
                blocks.append(page)
    out = []
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    for b in blocks:
        acc = b.get("accessionNumber", [])
        for i in range(len(acc)):
            d = b["filingDate"][i]
            if s <= d <= e and b["form"][i] in FORMS:
                out.append({
                    "cik": c, "accession": acc[i], "form": b["form"][i], "filing_date": d,
                    "primary_document": b.get("primaryDocument", [""] * len(acc))[i],
                    "items": b.get("items", [""] * len(acc))[i],
                })
    return out


DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)


def _tag(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}>([^\n<]*)", block, re.I)
    return m.group(1).strip() if m else ""


def parse_submission(raw: str, form: str) -> list[dict]:
    """Split a complete submission text file into documents; keep primary, EX-99.*, EX-2.*."""
    docs = []
    for m in DOC_RE.finditer(raw):
        block = m.group(1)
        head = block[:2000]
        dtype = _tag(head, "TYPE").upper()
        seq = _tag(head, "SEQUENCE")
        fname = _tag(head, "FILENAME")
        desc = _tag(head, "DESCRIPTION")
        is_primary = seq == "1" or dtype == form.upper()
        is_ex99 = dtype.startswith("EX-99")
        is_ex2 = dtype == "EX-2" or dtype.startswith("EX-2.")
        if not (is_primary or is_ex99 or is_ex2):
            continue
        if fname.lower().endswith(BINARY_EXT):
            continue
        tm = re.search(r"<TEXT>(.*)", block, re.S | re.I)
        text = tm.group(1) if tm else ""
        text = re.sub(r"</TEXT>\s*$", "", text, flags=re.I)
        stripped = text.lstrip()[:200].lower()
        if stripped.startswith("begin 6") or stripped.startswith("<pdf>") or "<xbrl" in stripped:
            continue
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS]
        docs.append({"type": dtype, "sequence": seq, "filename": fname, "description": desc,
                     "primary": bool(is_primary), "text": text})
    return docs


def download_filing(f: dict) -> str:
    """Fetch one filing; cache the kept documents as gzipped JSON. Returns a status string."""
    acc = f["accession"]
    path = FIL_DIR / f["cik"] / f"{acc}.json.gz"
    if path.exists():
        return "cached"
    url = f"https://www.sec.gov/Archives/edgar/data/{int(f['cik'])}/{acc.replace('-', '')}/{acc}.txt"
    raw = fetch(url)
    docs = parse_submission(raw.decode("latin-1", errors="replace"), f["form"]) if raw else []
    payload = {**f, "url": url, "missing": raw is None, "docs": docs}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return "missing" if raw is None else "ok"


def window(announced: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """One calendar day before to five business days (Mon-Fri) after the announcement."""
    start = announced - pd.Timedelta(days=1)
    end = pd.Timestamp(np.busday_offset(announced.date(), 5, roll="backward"))
    return start, end


def process_deal(deal_id: str, ciks: list[str], announced: pd.Timestamp) -> dict:
    start, end = window(announced)
    filings, seen = [], set()
    for c in ciks:
        try:
            for f in list_filings(c, start, end):
                if f["accession"] not in seen:   # the same filing can sit under several CIKs
                    seen.add(f["accession"])
                    filings.append(f)
        except Exception as e:  # keep going; a rerun retries
            return {"deal_id": deal_id, "error": f"list:{c}:{e}", "filings": []}
    # Earliest filing of each form first (round-robin over forms), so the cap keeps the
    # announcement-day 8-K / tender / proxy material rather than a run of duplicates.
    filings.sort(key=lambda f: (FORMS.index(f["form"]), f["filing_date"], f["accession"]))
    rank: dict[str, int] = {}
    for f in filings:
        f["_rank"] = rank.get(f["form"], 0)
        rank[f["form"]] = f["_rank"] + 1
    filings.sort(key=lambda f: (f["_rank"] if f["form"] != "8-K" else f["_rank"] // 2, FORMS.index(f["form"])))
    for f in filings:
        del f["_rank"]
    n_found = len(filings)
    filings = filings[:MAX_FILINGS_PER_DEAL]
    for f in filings:
        try:
            f["status"] = download_filing(f)
        except Exception as e:
            return {"deal_id": deal_id, "error": f"filing:{f['accession']}:{e}", "filings": []}
    return {"deal_id": deal_id, "window": [str(start.date()), str(end.date())],
            "n_found": n_found, "filings": filings}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", default=str(ROOT / "data/interim/deals.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    deals = pd.read_parquet(args.deals, columns=["deal_id", "cik", "announced_date"])
    deals = deals[deals["cik"].notna()]
    if args.limit:
        deals = deals.sample(args.limit, random_state=0)

    done: set[str] = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["deal_id"])
            except json.JSONDecodeError:
                pass
    todo = deals[~deals["deal_id"].isin(done)]
    print(f"{len(deals)} deals with a CIK, {len(done)} already done, {len(todo)} to do", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    t0, n_ok, n_err = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as ex, open(MANIFEST, "a") as mf:
        futs = []
        for r in todo.itertuples(index=False):
            ciks = [c.strip() for c in str(r.cik).split(";") if c.strip().isdigit()]
            futs.append(ex.submit(process_deal, r.deal_id, ciks, pd.Timestamp(r.announced_date)))
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            mf.write(json.dumps(res) + "\n")
            mf.flush()
            n_err += "error" in res
            n_ok += "error" not in res
            if i % 100 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"[{i}/{len(futs)}] ok={n_ok} err={n_err} requests={STATS['requests']} "
                      f"throttled={STATS['throttled']} ({STATS['requests'] / max(el, 1):.2f}/s) "
                      f"elapsed={el / 60:.1f} min", flush=True)
    print("done" if n_err == 0 else f"done with {n_err} deal-level errors; rerun to retry", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

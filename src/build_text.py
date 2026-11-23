"""Turn the cached EDGAR filings into two text fields per deal and (optionally) embed them.

  press_release : the announcement press release (EX-99.*), else another announcement
                  communication (425 / DEFA14A / SC TO-C / SC 14D9 body), else the 8-K body
  agreement     : the merger agreement (largest EX-2.* exhibit)

Only filings dated inside the download window (one calendar day before to five business
days after the announcement) exist in the cache, so the text predates the deal outcome.

Usage:
    python src/build_text.py                 # build text + coverage table
    python src/build_text.py --embed         # also compute frozen sentence embeddings
    python src/build_text.py --tag synthetic # synthetic placeholder text for the sample data
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/edgar_cache"
INTERIM = ROOT / "data/interim"

MAX_PRESS_CHARS = 40_000
MAX_AGREEMENT_CHARS = 600_000
MIN_PRESS_CHARS = 400
EMB_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS = 300
MAX_PRESS_CHUNKS = 10
MAX_AGREEMENT_CHUNKS = 24

DEAL_RE = re.compile(
    r"merger|acquir|acquisition|tender offer|definitive agreement|business combination|going[\s\-]private", re.I)
ANNOUNCE_RE = re.compile(
    r"(announced today|today announced|jointly announced|announce[sd]? (that|the signing|a definitive|an agreement)|"
    r"(entered|entering) into a(n)? (definitive|merger|agreement)|definitive (merger )?agreement|"
    r"agreement and plan of merger)", re.I)
# Section headers that start trailing boilerplate in press releases.
BOILER_RE = re.compile(
    r"^\s*(cautionary (note|statement)s?|forward[\s\-]looking statements?|safe harbor|"
    r"(important )?additional information( and where to find it| about)?|where to find additional information|"
    r"important information|participants in (the )?solicitation|notice to investors|"
    r"about [A-Z][\w&.,' \-]{2,60}|contacts?:?|investor contacts?:?|media contacts?:?)\s*:?\s*$", re.I | re.M)
BOILER_INLINE_RE = re.compile(
    r"(this (press release|communication|document|news release) (contains|includes|may contain)[^.]{0,80}"
    r"forward[\s\-]looking statements|statements (in|contained in) this[^.]{0,80}forward[\s\-]looking|"
    r"safe harbor (statement|for forward))", re.I)
PRESS_RE = re.compile(r"for immediate release|press release|news release|\bcontacts?:|announced today|today announced|"
                      r"\(nyse|\(nasdaq|\(amex", re.I)
LEGAL_RE = re.compile(r"\bWHEREAS\b|\bRECITALS\b|WITNESSETH|NOW, THEREFORE")
ITEM_RE = re.compile(r"item\s+\d+(\.\d+)?\.?\s", re.I)
SIGNATURE_RE = re.compile(r"\n\s*SIGNATURES?\s*\n")
TAG_RE = re.compile(r"<[^>]+>")


def strip_html(raw: str) -> str:
    """HTML / SGML -> plain text with paragraph breaks."""
    if re.search(r"<(html|body|div|p|font|table|td|br)\b", raw[:20000], re.I):
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(raw, "lxml")
        for t in soup(["script", "style", "head", "title"]):
            t.decompose()
        for br in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"]):
            br.insert_before("\n")
        text = soup.get_text(" ")
    else:
        text = TAG_RE.sub(" ", raw)
    text = html.unescape(text).replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    # drop page-number / table-of-contents artefacts
    text = re.sub(r"\n(\s*-?\s*\d{1,3}\s*-?\s*|\s*table of contents\s*|\s*page \d+( of \d+)?\s*)(?=\n)", "\n", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_press_boilerplate(text: str) -> str:
    """Cut trailing forward-looking / legal / 'About' sections from a press release."""
    cut = len(text)
    floor = min(800, len(text) // 4)
    for rx in (BOILER_RE, BOILER_INLINE_RE):
        for m in rx.finditer(text):
            if m.start() >= floor:
                cut = min(cut, m.start())
                break
    return text[:cut].strip()


def strip_8k_cover(text: str) -> str:
    """Remove the 8-K cover page (before the first Item) and the signature block."""
    m = ITEM_RE.search(text)
    if m and m.start() < 6000:
        text = text[m.start():]
    s = SIGNATURE_RE.search(text)
    if s and s.start() > 200:
        text = text[:s.start()]
    return text.strip()


def build_deal(args: tuple[str, list[dict]]) -> dict:
    deal_id, filings = args
    press_cands, agr_cands = [], []
    n_docs = 0
    for f in filings:
        path = CACHE / "filings" / f["cik"] / f"{f['accession']}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        for doc in payload.get("docs", []):
            n_docs += 1
            dtype = doc["type"]
            raw = doc["text"]
            if dtype.startswith("EX-2"):
                if len(raw) > 5000:
                    txt = strip_html(raw[:4_000_000])
                    if re.search(r"agreement", txt[:5000], re.I):
                        agr_cands.append((-len(txt), f["filing_date"], txt))
                continue
            txt = strip_html(raw[:1_500_000])
            head = txt[:4000]
            legal = bool(LEGAL_RE.search(txt[:6000])) and not PRESS_RE.search(txt[:1500])
            if dtype.startswith("EX-99"):
                txt = strip_press_boilerplate(txt)
                if legal:                       # legal agreements filed as EX-99
                    if len(txt) > 50_000 and re.search(r"agreement and plan of|merger agreement|acquisition agreement",
                                                       txt[:1500], re.I):
                        agr_cands.append((-len(txt), f["filing_date"], txt))   # merger agreement filed as EX-99
                    continue
                elif ANNOUNCE_RE.search(head) and PRESS_RE.search(head):
                    tier = 0
                elif ANNOUNCE_RE.search(head):
                    tier = 1
                else:                           # e.g. an earnings release that mentions the deal
                    tier = 4
            elif f["form"] == "8-K":
                txt = strip_press_boilerplate(strip_8k_cover(txt))
                tier = 3 if ANNOUNCE_RE.search(txt[:8000]) else 5
            elif f["form"] in ("425", "DEFA14A", "SC TO-C", "SC 14D9"):
                m = PRESS_RE.search(txt[:8000])
                if m and m.start() > 200:       # drop the SEC cover page in front of the release
                    txt = txt[m.start():]
                txt = strip_press_boilerplate(txt)
                tier = 2 if ANNOUNCE_RE.search(txt[:4000]) and not legal else 5
            else:  # SC TO-T, PREM14A bodies: long legal documents, last resort
                tier = 6
            n_deal = len(DEAL_RE.findall(txt[:20000]))
            if len(txt) < MIN_PRESS_CHARS or n_deal < (2 if tier <= 3 else 4):
                continue
            press_cands.append((tier, f["filing_date"], dtype, f["accession"], f["form"], txt))
    out = {"deal_id": deal_id, "n_filings": len(filings), "n_docs": n_docs,
           "press_release": "", "press_source": "", "agreement": "", "forms": ";".join(sorted({f["form"] for f in filings}))}
    if press_cands:
        press_cands.sort(key=lambda c: c[:4])
        tier, date, dtype, acc, form, txt = press_cands[0]
        out["press_release"] = txt[:MAX_PRESS_CHARS]
        out["press_source"] = f"{form}:{dtype}:tier{tier}"
    if agr_cands:
        agr_cands.sort(key=lambda c: c[:2])
        out["agreement"] = agr_cands[0][2][:MAX_AGREEMENT_CHARS]
    return out


def load_manifest() -> dict[str, list[dict]]:
    deals: dict[str, list[dict]] = {}
    mf = CACHE / "deal_filings.jsonl"
    if not mf.exists():
        return deals
    for line in mf.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        deals[r["deal_id"]] = r["filings"]   # later lines win (reruns)
    return deals


def synthetic_text(deals: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Placeholder text for the synthetic sample so the text models can run. Not real filings."""
    rng = np.random.default_rng(seed)
    bits = [
        "The transaction is not subject to a financing condition.",
        "The transaction is subject to a financing condition and receipt of committed financing.",
        "Closing is subject to expiration of the waiting period under the Hart-Scott-Rodino Antitrust Improvements Act.",
        "The merger requires approval by the stockholders of the company at a special meeting.",
        "The agreement includes a 40-day go-shop period.",
        "The parties will seek CFIUS clearance and other regulatory approvals.",
        "The buyer will pay a reverse termination fee under specified circumstances.",
        "The board of directors unanimously approved the merger agreement.",
        "The buyer will commence a tender offer for all outstanding shares.",
    ]
    rows = []
    for r in deals.itertuples(index=False):
        k = rng.integers(2, 6)
        body = " ".join(rng.choice(bits, size=k, replace=False))
        has = rng.random() < 0.85
        press = (f"SYNTHETIC PRESS RELEASE. The company today announced that it has entered into a definitive "
                 f"merger agreement to be acquired. {body}") if has else ""
        agr = (f"SYNTHETIC AGREEMENT AND PLAN OF MERGER. {body} " * 5) if has and rng.random() < 0.7 else ""
        rows.append({"deal_id": r.deal_id, "n_filings": int(has), "n_docs": int(has), "press_release": press,
                     "press_source": "synthetic" if has else "", "agreement": agr, "forms": "8-K" if has else ""})
    return pd.DataFrame(rows)


def chunk_words(text: str, max_chunks: int, spread: bool) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    if len(chunks) <= max_chunks:
        return chunks
    if spread:   # evenly spaced over the document (long agreements)
        idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        return [chunks[i] for i in idx]
    return chunks[:max_chunks]


def embed_field(model, texts: pd.Series, max_chunks: int, spread: bool, batch_size: int = 32) -> np.ndarray:
    """Mean-pooled, L2-normalised chunk embeddings; rows without text are NaN."""
    owners, chunks = [], []
    for i, t in enumerate(texts.fillna("")):
        for c in chunk_words(t, max_chunks, spread):
            owners.append(i)
            chunks.append(c)
    dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)()
    out = np.full((len(texts), dim), np.nan, dtype=np.float32)
    if not chunks:
        return out
    emb = model.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    owners = np.asarray(owners)
    sums = np.zeros((len(texts), dim), dtype=np.float64)
    np.add.at(sums, owners, emb)
    counts = np.bincount(owners, minlength=len(texts))
    has = counts > 0
    mean = sums[has] / counts[has, None]
    out[has] = (mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    suffix = "" if args.tag == "real" else f"_{args.tag}"
    deals = pd.read_parquet(INTERIM / f"deals{suffix}.parquet",
                            columns=["deal_id", "announced_date", "year", "cik"])
    text_path = INTERIM / f"text{suffix}.parquet"

    if not args.embed_only:
        if args.tag == "synthetic":
            text = synthetic_text(deals)
        else:
            manifest = load_manifest()
            jobs = [(d, manifest.get(d, [])) for d in deals["deal_id"]]
            with Pool(args.workers) as pool:
                rows = pool.map(build_deal, jobs, chunksize=20)
            text = pd.DataFrame(rows)
            text["downloaded"] = text["deal_id"].isin(manifest)
        text = deals[["deal_id"]].merge(text, on="deal_id", how="left")
        text["press_release"] = text["press_release"].fillna("")
        text["agreement"] = text["agreement"].fillna("")
        text["has_press"] = text["press_release"].str.len() >= MIN_PRESS_CHARS if args.tag == "real" \
            else text["press_release"].str.len() > 0
        text["has_agreement"] = text["agreement"].str.len() > 0
        text["has_text"] = text["has_press"]
        text.to_parquet(text_path, index=False)

        cov = deals.merge(text[["deal_id", "n_filings", "has_press", "has_agreement"]], on="deal_id")
        cov["has_cik"] = cov["cik"].notna()
        cov["has_filing"] = cov["n_filings"].fillna(0) > 0
        g = cov.groupby("year").agg(deals=("deal_id", "size"), with_cik=("has_cik", "mean"),
                                    with_filing=("has_filing", "mean"), press_release=("has_press", "mean"),
                                    agreement=("has_agreement", "mean"))
        tot = pd.DataFrame({"deals": [len(cov)], "with_cik": [cov.has_cik.mean()], "with_filing": [cov.has_filing.mean()],
                            "press_release": [cov.has_press.mean()], "agreement": [cov.has_agreement.mean()]}, index=["all"])
        g = pd.concat([g, tot])
        lines = ["# Text coverage by announcement year" + (" (SYNTHETIC SAMPLE)" if args.tag != "real" else ""), "",
                 "Share of deals with a target CIK, with at least one in-window EDGAR filing, with a usable "
                 "press-release text, and with a merger agreement (EX-2). The text subset used for the text "
                 "models is the `press_release` column.", "",
                 "| Year | Deals | Target CIK | Any filing in window | Press release | Agreement (EX-2) |",
                 "|---|---:|---:|---:|---:|---:|"]
        for y, r in g.iterrows():
            lines.append(f"| {y} | {int(r.deals)} | {r.with_cik:.0%} | {r.with_filing:.0%} | "
                         f"{r.press_release:.0%} | {r.agreement:.0%} |")
        res = ROOT / ("results" if args.tag == "real" else f"results_{args.tag}")
        res.mkdir(exist_ok=True)
        (res / "text_coverage.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[-6:]))
        print(text["press_source"].str.replace(r":EX-99[^:]*", ":EX-99", regex=True).value_counts().head(12))

    if args.embed or args.embed_only:
        import torch
        from sentence_transformers import SentenceTransformer
        text = pd.read_parquet(text_path)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(EMB_MODEL, device=device)   # inference only, weights frozen
        model.max_seq_length = 512
        model.eval()
        with torch.no_grad():
            ep = embed_field(model, text["press_release"], MAX_PRESS_CHUNKS, spread=False)
            ea = embed_field(model, text["agreement"], MAX_AGREEMENT_CHUNKS, spread=True)
        np.savez_compressed(INTERIM / f"embeddings{suffix}.npz", deal_id=text["deal_id"].to_numpy(dtype=str),
                            press=ep, agreement=ea)
        print(f"embeddings saved: press {np.isfinite(ep[:, 0]).sum()} rows, agreement {np.isfinite(ea[:, 0]).sum()} rows, "
              f"device={device}")


if __name__ == "__main__":
    main()

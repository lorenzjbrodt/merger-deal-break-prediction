"""Link every deal to CRSP: the target's permno and, where the buyer is US-listed, the acquirer's permno.

Target (in this order, first hit wins)
  1. cik_ccm    SEC CIK -> Compustat gvkey (comp.company) -> CRSP permno through the CCM link table,
                link types LU / LC / LS, valid on the announcement date; primary links (P, C) preferred
  2. ticker     ticker in the Capital IQ target string and CRSP names history valid on the announcement date
  3. name       exact match of the normalised company name with a CRSP issuer name valid on the announcement
                date, or with a Compustat name whose CCM link is valid on that date (unique matches only)
Acquirer (Capital IQ shows the buyer's *current* ticker, which may differ from the one used at the time,\nand no ticker at all for buyers that have since been delisted)
  1. ticker_date    ticker valid on the announcement date and the CRSP issuer name resembles the buyer name
  2. ticker_permno  a security that carried the ticker at any time, resembles the buyer name and traded on
                    the announcement date (catches later ticker changes)
  3. name           exact normalised name match with a CRSP issuer name valid on the announcement date, or with a\n                    Compustat name (short or legal) whose CCM link is valid on that date; unique matches only
Links are checked against prices later (src/crsp_prices.py): a link whose pre-announcement CRSP close is
far from the price implied by the export's offer and premium is dropped there.

Outputs: data/interim/crsp_link.parquet (deal level, git-ignored), results/crsp_link.md (aggregates only).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
US_EXCH = {"NYSE", "NasdaqGS", "NasdaqGM", "NasdaqCM", "NYSEAM", "NYSEArca", "ARCA", "BATS", "AMEX", "Nasdaq", "NasdaqNM", "NasdaqSC"}
TICK = re.compile(r"\(([A-Za-z]+):([A-Z0-9\.\-]+)\)")
STOP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COMPANIES", "COS", "LTD", "LIMITED", "LLC", "LP", "PLC", "HOLDINGS", "HOLDING",
        "HLDGS", "HLDG", "GROUP", "GRP", "THE", "NEW", "DEL", "DE", "AND", "OF", "TRUST", "TR", "SA", "NV", "AG", "CL", "A", "B", "COM"}
ABBR = {"INTERNATIONAL": "INTL", "FINANCIAL": "FINL", "TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH", "BANCORPORATION": "BANCORP", "INDUSTRIES": "INDS",
        "SYSTEMS": "SYS", "SERVICES": "SVCS", "COMMUNICATIONS": "COMM", "PHARMACEUTICALS": "PHARMA", "PHARMACEUTICAL": "PHARMA", "RESOURCES": "RES",
        "ENTERPRISES": "ENTS", "PROPERTIES": "PPTYS", "LABORATORIES": "LABS", "AMERICAN": "AMERN", "NATIONAL": "NATL", "MANUFACTURING": "MFG",
        "ASSOCIATES": "ASSOC", "PRODUCTS": "PRODS", "ENERGY": "ENERGY", "SOLUTIONS": "SOLUTIONS", "BANCSHARES": "BANCSHARES"}


def tokens(name) -> list:
    if not isinstance(name, str):
        return []
    s = re.sub(r"\(.*?\)", " ", name.upper())
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return [ABBR.get(t, t) for t in s.split() if t not in STOP]


def norm(name) -> str:
    return " ".join(tokens(name))


def tok_match(a: str, b: str) -> bool:
    if a == b:
        return True
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k >= 4 or (k >= 3 and k == min(len(a), len(b)))


def similar(a, b) -> bool:
    """Abbreviation-tolerant resemblance: first tokens match and most tokens of the shorter name match in the other."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb or not tok_match(ta[0], tb[0]):
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    hit = sum(any(tok_match(x, y) for y in long_) for x in short)
    return hit / len(short) >= 0.6


def main() -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    o = pd.read_parquet(INTERIM / "outcomes.parquet")[["deal_id", "outcome3"]]
    d = d.merge(o, on="deal_id", how="left")
    d["outcome3"] = d["outcome3"].fillna(pd.Series(np.where(d["label"] == 1, "adverse_break", "closed"), index=d.index))
    cc = pd.read_parquet(WRDS / "comp_company.parquet")
    lk = pd.read_parquet(WRDS / "ccm_link.parquet")
    nm = pd.read_parquet(WRDS / "crsp_names.parquet")
    for c in ("linkdt", "linkenddt"):
        lk[c] = pd.to_datetime(lk[c])
    for c in ("namedt", "nameenddt", "securitybegdt", "securityenddt"):
        nm[c] = pd.to_datetime(nm[c])
    lk = lk[lk["linktype"].isin(["LU", "LC", "LS"]) & lk["lpermno"].notna()].copy()
    lk["linkenddt"] = lk["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    lk["prio"] = lk["linkprim"].map({"P": 0, "C": 1, "J": 2, "N": 3}).fillna(4)
    lk_by = {g: x for g, x in lk.groupby("gvkey")}
    cik2gv = cc.dropna(subset=["cik"]).set_index("cik")["gvkey"].to_dict()
    nm["norm"] = nm["issuernm"].map(norm)
    comp_by_norm = {}
    for col in ("conm", "conml"):
        for k, g in zip(cc[col].map(norm), cc["gvkey"]):
            if k and g not in comp_by_norm.setdefault(k, []):
                comp_by_norm[k].append(g)
    nm_by_tic = {t: x for t, x in nm.dropna(subset=["ticker"]).groupby("ticker")}
    nm_by_norm = {t: x for t, x in nm.groupby("norm")}
    crsp_end = nm["nameenddt"].max()

    def ccm(gv, ann):
        x = lk_by.get(gv)
        if x is None:
            return None
        x = x[(x["linkdt"] <= ann) & (x["linkenddt"] >= ann)]
        if x.empty:
            return None
        return int(x.sort_values(["prio", "linkdt"]).iloc[0]["lpermno"])

    def active(x, ann):
        return x[(x["namedt"] <= ann) & (x["nameenddt"] >= ann)]

    rows = []
    for r in d.itertuples(index=False):
        ann = pd.Timestamp(r.announced_date)
        out = {"deal_id": r.deal_id, "announced_date": ann, "outcome_date": r.closed_date if pd.notna(r.closed_date) else r.cancelled_date,
               "gvkey": None, "permno": np.nan, "link_method": "", "acq_permno": np.nan, "acq_method": "", "acq_ticker": None}
        # ---- target
        gvs = [cik2gv[c.strip()] for c in str(r.cik).split(";") if c.strip() in cik2gv] if isinstance(r.cik, str) else []
        for gv in gvs:
            p = ccm(gv, ann)
            if p is not None:
                out.update(gvkey=gv, permno=p, link_method="cik_ccm"); break
        if not out["link_method"]:
            if gvs:
                out["gvkey"] = gvs[0]
            tics = [t for src in (r.target, r.exchange_ticker, r.security_tickers) if isinstance(src, str) for e, t in TICK.findall(src if "(" in src else f"({src})") if e in US_EXCH]
            for t in dict.fromkeys(tics):
                x = nm_by_tic.get(t)
                if x is not None:
                    a = active(x, ann)
                    if a["permno"].nunique() == 1:
                        out.update(permno=int(a["permno"].iloc[0]), link_method="ticker"); break
        if not out["link_method"]:
            key = norm(r.target_clean)
            if key:
                x = nm_by_norm.get(key)
                a = active(x, ann) if x is not None else None
                if a is not None and a["permno"].nunique() == 1:
                    out.update(permno=int(a["permno"].iloc[0]), link_method="name")
                else:
                    ps = {(g, ccm(g, ann)) for g in comp_by_norm.get(key, [])}
                    ps = {(g, p) for g, p in ps if p is not None}
                    if len(ps) == 1:
                        g, p = next(iter(ps))
                        out.update(gvkey=out["gvkey"] or g, permno=p, link_method="name")
                    elif out["gvkey"] is None and len(comp_by_norm.get(key, [])) == 1:
                        out["gvkey"] = comp_by_norm[key][0]          # Compustat prices only
        # ---- acquirer
        if isinstance(r.buyers, str):
            parts = [p_.strip() for p_ in r.buyers.split(";") if p_.strip()]
            for part in parts:                                   # buyers with a US ticker
                m = TICK.search(part)
                if not m or m.group(1) not in US_EXCH:
                    continue
                t = m.group(2)
                out["acq_ticker"] = t
                x = nm_by_tic.get(t)
                if x is None:
                    continue
                a = active(x, ann)
                a = a[[similar(part, n) for n in a["issuernm"]]] if len(a) else a
                if a["permno"].nunique() == 1:
                    out.update(acq_permno=int(a["permno"].iloc[0]), acq_method="ticker_date"); break
                cand = [p for p in x["permno"].unique()
                        if any(similar(part, n) for n in nm.loc[nm["permno"] == p, "issuernm"].unique())
                        and len(active(nm[nm["permno"] == p], ann))]
                if len(cand) == 1:
                    out.update(acq_permno=int(cand[0]), acq_method="ticker_permno"); break
            if not out["acq_method"]:                            # exact name routes (CRSP name on the date, or Compustat name -> CCM)
                for part in parts:
                    key = norm(part)
                    if not key:
                        continue
                    x = nm_by_norm.get(key)
                    a = active(x, ann) if x is not None else None
                    if a is not None and a["permno"].nunique() == 1:
                        out.update(acq_permno=int(a["permno"].iloc[0]), acq_method="name"); break
                    ps = {ccm(g, ann) for g in comp_by_norm.get(key, [])} - {None}
                    if len(ps) == 1:
                        out.update(acq_permno=int(next(iter(ps))), acq_method="name"); break
        rows.append(out)
    L = pd.DataFrame(rows)
    L["outcome_date"] = pd.to_datetime(L["outcome_date"])
    L.to_parquet(INTERIM / "crsp_link.parquet", index=False)

    # ---------------------------------------------------------------- report (aggregates only)
    reverse = d["secondary_features"].fillna("").astype(str).str.contains("Reverse Merger").to_numpy()
    cash = ((d["cash_pct"] == 100) | ((d["consideration"].fillna("") == "Cash") & d["cash_pct"].isna())).to_numpy()
    T = pd.DataFrame({"period": np.where(d["year"] <= 2012, "2001-2012", np.where(d["year"] <= 2018, "2013-2018", "2019-2025")), "outcome": d["outcome3"].to_numpy(),
                      "core": ~reverse, "cash": cash, "linked": L["permno"].notna().to_numpy(), "method": L["link_method"].to_numpy(),
                      "gvkey": L["gvkey"].notna().to_numpy(), "acq": L["acq_permno"].notna().to_numpy(), "acq_tic": L["acq_ticker"].notna().to_numpy(),
                      "beyond": (L["outcome_date"] > crsp_end).to_numpy(), "ann_beyond": (L["announced_date"] > crsp_end).to_numpy()})
    R = ["# Linking deals to CRSP", "",
         "Source: WRDS (CRSP daily stock file in the current format, `crsp.dsf_v2`; Compustat `comp.company`; CRSP/Compustat Merged link table `crsp.ccmxpf_lnkhist`). "
         "Licensed data: this file contains counts only.", "",
         f"**CRSP coverage ends on {crsp_end.date()}.** Deals whose outcome date is later ({int(T['beyond'].sum())} deals, {int((T['beyond'] & (T['outcome'] != 'closed')).sum())} of them broken) "
         "cannot be followed to the end in CRSP; for those the exit uses Compustat daily prices where available and otherwise the Capital IQ monthly closes "
         "(see `results/crsp_prices.md`).", "",
         "## Target link, by period and outcome (core sample)", "",
         "| Period | Outcome | Deals | Linked to a permno | via CIK and CCM | via ticker | via name | Not linked |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (a, b), g in T[T["core"]].groupby(["period", "outcome"], sort=True):
        R.append(f"| {a} | {b} | {len(g)} | {int(g['linked'].sum())} ({g['linked'].mean():.0%}) | {int((g['method'] == 'cik_ccm').sum())} | "
                 f"{int((g['method'] == 'ticker').sum())} | {int((g['method'] == 'name').sum())} | {int((~g['linked']).sum())} |")
    g = T[T["core"]]
    R += [f"| All | all | {len(g)} | {int(g['linked'].sum())} ({g['linked'].mean():.0%}) | {int((g['method'] == 'cik_ccm').sum())} | {int((g['method'] == 'ticker').sum())} | "
          f"{int((g['method'] == 'name').sum())} | {int((~g['linked']).sum())} |", "",
          f"Reverse mergers (outside the core sample): {int((~T['core']).sum())} deals, {int((T['linked'] & ~T['core']).sum())} linked.", "",
          "Targets without a link are mostly companies whose shares did not trade on NYSE, NYSE American or Nasdaq (CRSP does not cover OTC stocks) "
          "and deals for which the export has no CIK.", "",
          "## Acquirer link (needed for stock and mixed deals)", "",
          "| Period | Consideration | Deals | Buyer shown with a US ticker | Acquirer linked | Target and acquirer linked |", "|---|---|---:|---:|---:|---:|"]
    for (a, c), g in T[T["core"]].assign(cons=np.where(T.loc[T["core"], "cash"], "all cash", "stock or mixed")).groupby(["period", "cons"], sort=True):
        R.append(f"| {a} | {c} | {len(g)} | {int(g['acq_tic'].sum())} | {int(g['acq'].sum())} | {int((g['acq'] & g['linked']).sum())} |")
    am = L["acq_method"].value_counts()
    R += ["", "Acquirer link methods: " + ", ".join(f"{k or 'none'} {v}" for k, v in am.items()) + ". Capital IQ shows a ticker only for buyers listed today, so buyers "
          "that were later acquired or delisted are found by name. The remaining unlinked buyers are private companies, financial sponsors or foreign-listed "
          "companies; a stock deal with such a buyer has no CRSP acquirer price.", ""]
    (ROOT / "results/crsp_link.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()

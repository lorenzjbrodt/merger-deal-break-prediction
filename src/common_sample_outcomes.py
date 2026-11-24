"""Read-only export: outcome shares, day-1 spread and realised returns on ONE common sample.

    .venv/bin/python src/common_sample_outcomes.py

Nothing is fitted and no existing result file is changed. The script reads the cached per-deal price file
(data/interim/, git-ignored, licensed) and writes aggregates only:
results/common_sample_outcomes.json and results/common_sample_outcomes.md. It is not part of run_all.sh.

Common sample: the priced all-cash core deals announced 2013-2025 that the ranking test of expected_return.py uses.
The filter is the `priced` mask of expected_return.py and nothing else: core sample (no reverse mergers), all-cash,
usable CRSP price (`backtest_ok`) and a defined market-implied probability (offer above the pre-announcement close).
The script stops if this does not reproduce 1,205 deals with 1,150 closed, 38 adverse breaks and 17 topped, or if
the rows differ from the rows ranked in expected_return.py. Two comparison samples use the same filter with other
announcement years: 2019-2025 (the test period) and 2001-2025 (all years).

Definitions, all from crsp_prices.py:
  day-1 spread   offer known on day 1 / close(t1) - 1, t1 = first trading day after the announcement
  return         total return of the target from the close on t1 to the exit (`ret`): completed deals to the closing
                 (CRSP delisting return, or the offer paid on the closing date), cancelled deals to the close of the
                 first trading day after the cancellation date. Before costs, unhedged, not annualized.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import modeling as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
RES = ROOT / "results"
EXPECTED = {"deals": 1205, "closed": 1150, "adverse_break": 38, "topped": 17}
SAMPLES = (("pooled_2013_2025", "Common sample: priced all-cash deals announced 2013-2025 (the ranking-test sample)", 2013),
           ("test_2019_2025", "Comparison: priced all-cash deals announced 2019-2025", 2019),
           ("all_2001_2025", "Comparison: priced all-cash deals announced 2001-2025", 2001))
CLS = (("closed", "Closed"), ("adverse_break", "Adverse break"), ("topped", "Topped by a higher bid"))
NAMES = {"sample_filter": "Sample filter", "outcomes": "Outcomes", "spread_day1": "Day-1 spread", "return": "Return", "loss_to_spread": "Loss to spread"}


def stop(msg: str) -> None:
    raise SystemExit(f"STOPPED, nothing written: {msg}")


def stats(x) -> dict:
    """n, mean and median in percent, two decimals; the unrounded values are kept under '_' for the ratios."""
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean_pct": round(float(x.mean()) * 100, 2), "median_pct": round(float(np.median(x)) * 100, 2),
            "_": (float(x.mean()) * 100, float(np.median(x)) * 100)}


def ratio(lab, loss, lname, sp, sname) -> dict:
    """Loss over spread as a positive multiple. The ratio is computed from the unrounded values; the calculation is
    shown with the two-decimal inputs, and the result from those inputs is given when it differs."""
    r, rr = round(-loss / sp, 2), round(-round(loss, 2) / round(sp, 2), 2)
    return {"label": lab, "ratio": r, "calculation": f"{-round(loss, 2):.2f} / {round(sp, 2):.2f} = {rr:.2f}",
            "inputs": f"{lname} of adverse breaks (sign reversed) / {sname} of closed deals, both in %", "ratio_from_rounded_inputs": rr}


def summarise(ret, sp, cls, m) -> dict:
    n = int(m.sum())
    k = {c: m & (cls == c) for c, _ in CLS}
    o = {"deals": n, "outcomes": {c: {"n": int(k[c].sum()), "share_pct": round(100.0 * k[c].sum() / n, 2)} for c, _ in CLS},
         "spread_day1_closed": stats(sp[k["closed"]]), "return_closed": stats(ret[k["closed"]]),
         "return_adverse_break": stats(ret[k["adverse_break"]]), "return_topped": stats(ret[k["topped"]])}
    (lm, lmed), (sm, smed) = o["return_adverse_break"]["_"], o["spread_day1_closed"]["_"]
    o["loss_to_spread"] = [ratio("mean loss / median spread", lm, "mean return", smed, "median day-1 spread"),
                           ratio("mean loss / mean spread", lm, "mean return", sm, "mean day-1 spread"),
                           ratio("median loss / median spread", lmed, "median return", smed, "median day-1 spread")]
    for v in o.values():
        if isinstance(v, dict):
            v.pop("_", None)
    return o


def check_crsp_prices_table(PF, D, ret, sp) -> int:
    """The spread and return columns must reproduce the published table in results/crsp_prices.md (usable all-cash
    deals by period and outcome, one decimal). Returns the number of table rows checked."""
    use = D.core & PF["backtest_ok"].fillna(False).to_numpy() & PF["cash_deal"].fillna(False).to_numpy()
    pat = re.compile(r"^\| (test 2019-2025|train 2001-2018) \| (\w+) \| (\d+) \| (-?[\d.]+)% \| (-?[\d.]+)% \| [\d.]+ \| [\d.]+ \| ([+-][\d.]+)% \|")
    n = 0
    for line in (RES / "crsp_prices.md").read_text().splitlines():
        g = pat.match(line)
        if not g:
            continue
        m = use & ((D.year >= 2019) if g.group(1).startswith("test") else (D.year <= 2018)) & (D.outcome3 == g.group(2))
        got = (int(m.sum()), f"{np.median(sp[m]):.1%}", f"{sp[m].mean():.1%}", f"{ret[m].mean():+.1%}")
        want = (int(g.group(3)), g.group(4) + "%", g.group(5) + "%", g.group(6) + "%")
        if got != want:
            stop(f"results/crsp_prices.md row '{g.group(1)} | {g.group(2)}' not reproduced: {got} against {want}")
        n += 1
    if n != 6:
        stop(f"expected 6 rows of the day-1 spread table in results/crsp_prices.md, found {n}")
    return n


def main() -> None:
    D = M.load_data("")
    if D.pf_source != "CRSP daily":
        stop("the CRSP daily price file is not available (data/interim/price_features_crsp.parquet)")
    PF = D.PF
    ids = D.deals["deal_id"].to_numpy()
    ret, sp, cls = PF["ret"].to_numpy(float), PF["spread"].to_numpy(float), D.outcome3
    priced = D.core & PF["backtest_ok"].fillna(False).to_numpy() & PF["cash_deal"].fillna(False).to_numpy() & np.isfinite(PF["p_mkt"].to_numpy(float))
    if not (np.isfinite(ret[priced]).all() and np.isfinite(sp[priced]).all() and np.isin(cls[priced], [c for c, _ in CLS]).all()):
        stop("a priced deal has no return, no spread or no outcome class")
    masks = {key: priced & (D.year >= y0) & (D.year <= 2025) for key, _, y0 in SAMPLES}

    # ---- the common sample must be the ranking-test sample, row for row and count for count
    got = {"deals": int(masks["pooled_2013_2025"].sum()), **{c: int((masks["pooled_2013_2025"] & (cls == c)).sum()) for c, _ in CLS}}
    if got != EXPECTED:
        stop(f"common sample counts differ. Expected {EXPECTED}, found {got}; difference {({k: got[k] - EXPECTED[k] for k in EXPECTED})}")
    EP = pd.read_parquet(INTERIM / "expected_return.parquet").set_index("deal_id").reindex(ids).reset_index()
    pub = json.loads((RES / "expected_return.json").read_text())["views"]
    hold = PF["hold_days"].to_numpy(float)
    checks = []
    for key, view in (("pooled_2013_2025", "pooled"), ("test_2019_2025", "test")):
        rows = np.isfinite(EP[f"{view}__ear"].to_numpy()) & np.isfinite(ret) & np.isfinite(hold)
        m, V = masks[key], pub[view]
        if (rows != m).any():
            stop(f"{key}: {int((rows & ~m).sum())} ranked deals are outside the filter and {int((m & ~rows).sum())} filtered deals are not ranked")
        if (int(m.sum()), int((m & (cls == "adverse_break")).sum()), int((m & (cls == "topped")).sum())) != (V["n"], V["adverse"], V["topped"]):
            stop(f"{key}: counts differ from results/expected_return.json")
        T = V["rankings"]["ear"]["terciles"]
        w = sum(T[t]["n"] * T[t]["mean_ret"] for t in T) / sum(T[t]["n"] for t in T)
        if sum(T[t]["n"] for t in T) != int(m.sum()) or abs(w - ret[m].mean()) > 1e-9:
            stop(f"{key}: mean return per deal differs from the thirds in results/expected_return.json")
        S = V["rankings"]["spread"]["terciles"]
        ws = sum(S[t]["n"] * S[t]["mean_score"] for t in S) / sum(S[t]["n"] for t in S)
        if abs(ws - sp[m].mean()) > 1e-9:
            stop(f"{key}: mean day-1 spread differs from the thirds in results/expected_return.json")
        checks.append(f"{key}: same rows as the ranking test in expected_return.py; deals, adverse breaks and topped deals equal results/expected_return.json "
                      f"({V['n']}, {V['adverse']}, {V['topped']}); mean return per deal ({ret[m].mean():+.2%}) and mean day-1 spread ({sp[m].mean():.2%}) of all deals equal "
                      "the deal-weighted means of the thirds published there")
    nrows = check_crsp_prices_table(PF, D, ret, sp)
    checks.append(f"spread and return columns reproduce all {nrows} rows of the day-1 spread table in results/crsp_prices.md (usable all-cash deals, which also "
                  "include deals without a market-implied probability: deals, median and mean spread, mean return)")
    checks.append("all_2001_2025: same filter, no published count to compare with; it contains the two samples above as subsets")

    defs = {"sample_filter": "Core sample (signed deals without reverse mergers), all-cash consideration, usable CRSP price (deal still open at the day-1 close, pre-announcement "
                             "price within 15% of the price implied by the vendor's offer and premium, day-1 spread between -50% and +100%, an exit exists), and a defined "
                             "market-implied break probability (offer above the pre-announcement close). Identical to the `priced` filter of src/expected_return.py; the three "
                             "samples differ only in the announcement years.",
            "outcomes": "Closed = completed. Adverse break = cancelled for a reason other than a higher competing bid. Topped = cancelled because a higher competing bid won.",
            "spread_day1": "Offer known on day 1 / close on t1 - 1, where t1 is the first trading day after the announcement date (results/crsp_prices.md). The offer is the "
                           "per-share price in the announcement press release where it differs from the vendor's final offer.",
            "return": "Total return of the target share (dividends included) from the close on t1 to the exit, the field behind loss_if_break in results/deck_numbers.json. "
                      "Closed deals: to the closing (CRSP delisting return; the offer paid on the closing date where CRSP has no delisting record). Adverse breaks and topped "
                      "deals: to the close of the first trading day after the cancellation date. Not annualized, unhedged, before costs.",
            "loss_to_spread": "Loss = return of adverse breaks with the sign reversed; spread = day-1 spread of closed deals; both from the same sample. The ratio is computed "
                              "from unrounded values; 'calculation' shows it with the two-decimal inputs."}
    out = {"about": "Outcome shares, day-1 spread and realised returns on one common sample, so that a statement about how often deals break, what a closed deal pays and what a "
                    "break costs rests on the same deals. Read-only export from the frozen results; aggregates only. Percentages with two decimals.",
           "definitions": defs, "checks": checks, "samples": {}}
    for key, title, _ in SAMPLES:
        out["samples"][key] = {"title": title, **summarise(ret, sp, cls, masks[key])}
    (RES / "common_sample_outcomes.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---- markdown
    L = ["# Outcomes, day-1 spread and loss on one common sample", "",
         "Read-only export (`src/common_sample_outcomes.py`); no model was refitted and no other result file changed. Aggregates only. "
         "Returns are per deal, not annualized, unhedged and before costs.", "",
         "**Why.** Figures quoted elsewhere come from different samples, for example break rates from the core sample, the spread from usable all-cash deals and the 28% "
         "loss from 184 adverse breaks of any consideration. Here outcome shares, spread and loss are computed on the same deals.", "",
         "## Definitions", ""] + [f"- **{NAMES[k]}.** {v}" for k, v in defs.items()] + [""]
    for key, title, _ in SAMPLES:
        o = out["samples"][key]
        L += [f"## {title}", "", f"{o['deals']:,} deals.", "", "| | Deals | Share of deals | Mean | Median |", "|---|---:|---:|---:|---:|"]
        for c, lab in CLS:
            L.append(f"| Outcome: {lab.lower()} | {o['outcomes'][c]['n']:,} | {o['outcomes'][c]['share_pct']:.2f}% | | |")
        for k, lab in (("spread_day1_closed", "Day-1 spread, closed deals"), ("return_closed", "Return to closing, closed deals"),
                       ("return_adverse_break", "Return, adverse breaks"), ("return_topped", "Return, topped deals")):
            L.append(f"| {lab} | {o[k]['n']:,} | | {o[k]['mean_pct']:+.2f}% | {o[k]['median_pct']:+.2f}% |")
        L += ["", "| Loss to spread | Calculation (inputs in %) | Ratio |", "|---|---|---:|"]
        for r in o["loss_to_spread"]:
            note = "" if r["ratio"] == r["ratio_from_rounded_inputs"] else f" ({r['ratio']:.2f} from unrounded inputs)"
            L.append(f"| {r['label']} | {r['calculation']}{note} | {r['ratio']:.2f} |")
        L.append("")
    L += ["## Side by side", "", "| | " + " | ".join(t.split(": ")[1].replace("priced all-cash deals announced ", "").replace(" (the ranking-test sample)", "") for _, t, _ in SAMPLES) + " |",
          "|---|" + "---:|" * len(SAMPLES)]
    S = [out["samples"][k] for k, _, _ in SAMPLES]
    L.append("| Deals | " + " | ".join(f"{o['deals']:,}" for o in S) + " |")
    for c, lab in CLS:
        L.append(f"| {lab}, share of deals | " + " | ".join(f"{o['outcomes'][c]['share_pct']:.2f}% ({o['outcomes'][c]['n']:,})" for o in S) + " |")
    for k, lab in (("spread_day1_closed", "Day-1 spread, closed deals"), ("return_closed", "Return to closing, closed deals"),
                   ("return_adverse_break", "Return, adverse breaks"), ("return_topped", "Return, topped deals")):
        L.append(f"| {lab}: mean / median | " + " | ".join(f"{o[k]['mean_pct']:+.2f}% / {o[k]['median_pct']:+.2f}%" for o in S) + " |")
    for i in range(3):
        L.append(f"| {S[0]['loss_to_spread'][i]['label'].capitalize()} | " + " | ".join(f"{o['loss_to_spread'][i]['ratio']:.2f}" for o in S) + " |")
    L += ["", "## Checks", ""] + [f"- {c}." for c in checks] + ["",
          "## Reading", "",
          "- The adverse breaks of the 2019-2025 sample are a subset of those of the 2013-2025 sample, and both are subsets of the 2001-2025 sample; the three columns are not "
          "independent.",
          "- Means of the break returns rest on 22 to 81 deals and move by several points between samples; the ratios move with them.",
          "- The return of closed deals is not the same number as their day-1 spread: it is the realised total return and includes dividends and any later price increase, "
          "while the spread uses the offer known on day 1.",
          "- The 28% average loss quoted elsewhere is from 184 adverse breaks of any consideration under a different price filter (`results/loss_given_break.md`); it is not "
          "the loss of this sample.", ""]
    (RES / "common_sample_outcomes.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

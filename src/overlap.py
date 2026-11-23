"""Antitrust overlap between acquirer and target: product similarity and industry shares.

Sources
  Hoberg-Phillips TNIC-3 (public; Hoberg and Phillips 2010, 2016): pairs of Compustat firms whose 10-K product
      descriptions are similar, with a similarity score, by year. File: data/external/hoberg_phillips/extracted/tnic3_data.txt
      (download: https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip). The data end in 2023.
  Compustat annual (WRDS pull): sales and historical SIC code per firm-year.
Everything is taken from the year before the announcement (TNIC: the latest available year up to three years back),
so it is known on the announcement date.

Features (prefix ov_; missing when the acquirer is not a Compustat firm)
  ov_tnic_pair        1 if acquirer and target are TNIC-3 peers, 0 if both are in the TNIC universe but not peers
  ov_tnic_score       their similarity score (0 if in the universe but not peers)
  ov_tnic_tgt_rivals  log(1 + number of TNIC-3 peers of the target)
  ov_same_sic4 / ov_same_sic3 / ov_same_sic2
  ov_tgt_share / ov_acq_share   each firm's share of Compustat sales in its own 4-digit SIC industry
  ov_hhi              Herfindahl index of the target's 4-digit SIC industry (Compustat sales)
  ov_delta_hhi        2 x target share x acquirer share if both are in the same 4-digit industry, else 0
  ov_acq_compustat    1 if the acquirer was matched to Compustat

  python src/overlap.py build   data/interim/overlap_features.parquet (git-ignored)
  python src/overlap.py eval    results/overlap.md, results/overlap.json (after train_eval, pooled_oot and duration)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
WRDS = ROOT / "data/wrds"
TNIC = ROOT / "data/external/hoberg_phillips/extracted/tnic3_data.txt"
FEATURES = ["ov_tnic_pair", "ov_tnic_score", "ov_tnic_tgt_rivals", "ov_same_sic4", "ov_same_sic3", "ov_same_sic2", "ov_tgt_share", "ov_acq_share", "ov_hhi",
            "ov_delta_hhi", "ov_acq_compustat"]
SIC_ONLY = [f for f in FEATURES if "tnic" not in f]


def build() -> None:
    d = pd.read_parquet(INTERIM / "deals.parquet")
    link = pd.read_parquet(INTERIM / "crsp_link.parquet").set_index("deal_id").reindex(d["deal_id"]).reset_index()
    lk = pd.read_parquet(WRDS / "ccm_link.parquet").dropna(subset=["lpermno"])
    lk = lk[lk["linktype"].isin(["LU", "LC", "LS"])].copy()
    lk["linkdt"] = pd.to_datetime(lk["linkdt"]); lk["linkenddt"] = pd.to_datetime(lk["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    by_perm = {int(p): g for p, g in lk.groupby("lpermno")}

    def gv_of(permno, ann):
        g = by_perm.get(int(permno)) if pd.notna(permno) else None
        if g is None:
            return None
        x = g[(g["linkdt"] <= ann) & (g["linkenddt"] >= ann)]
        x = x if len(x) else g
        return x.sort_values("linkdt").iloc[-1]["gvkey"]
    year = d["year"].to_numpy(int)
    tg = link["gvkey"].to_numpy(object)
    ag = np.array([gv_of(p, a) for p, a in zip(link["acq_permno"], link["announced_date"])], dtype=object)

    # ---- Compustat: SIC and sales in fiscal year Y-1
    fa = pd.read_parquet(WRDS / "comp_funda.parquet").dropna(subset=["fyear"])
    fa["fyear"] = fa["fyear"].astype(int)
    hdr = pd.read_parquet(WRDS / "comp_company.parquet").set_index("gvkey")["sic"]
    fa["sic4"] = fa["sich"].fillna(pd.to_numeric(fa["gvkey"].map(hdr), errors="coerce"))
    fa = fa.dropna(subset=["sic4"]).sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    fa["sic4"] = fa["sic4"].astype(int)
    pos = fa[fa["sale"] > 0]
    tot = pos.groupby(["sic4", "fyear"])["sale"].sum()
    pos = pos.assign(share=pos["sale"] / pd.MultiIndex.from_arrays([pos["sic4"], pos["fyear"]]).map(tot))
    hhi = pos.assign(s2=pos["share"] ** 2).groupby(["sic4", "fyear"])["s2"].sum().to_dict()
    info = {(g, y): (s, sh) for g, y, s, sh in zip(fa["gvkey"], fa["fyear"], fa["sic4"], fa["gvkey"].map(lambda _: np.nan))}
    share = {(g, y): sh for g, y, sh in zip(pos["gvkey"], pos["fyear"], pos["share"])}
    sic_of = {(g, y): s for g, y, s in zip(fa["gvkey"], fa["fyear"], fa["sic4"])}

    def firm(g, y):
        """(sic4, sales share) for the latest fiscal year in [y-2, y]."""
        if not isinstance(g, str):
            return np.nan, np.nan
        for yy in (y, y - 1, y - 2):
            if (g, yy) in sic_of:
                return sic_of[(g, yy)], share.get((g, yy), np.nan), yy
        s = pd.to_numeric(hdr.get(g), errors="coerce")
        return (s, np.nan, y) if pd.notna(s) else (np.nan, np.nan, y)
    del info
    ciq_sic = pd.to_numeric(d["sic"], errors="coerce").to_numpy() if "sic" in d else np.full(len(d), np.nan)

    # ---- TNIC: keep rows that involve any deal firm
    want = {int(g) for g in list(tg) + list(ag) if isinstance(g, str)}
    tn = pd.read_csv(TNIC, sep="\t", dtype={"year": "int16", "gvkey1": "int32", "gvkey2": "int32", "score": "float32"})
    tmax = int(tn["year"].max())
    tn = tn[tn["gvkey1"].isin(want)]
    universe = set(zip(tn.loc[tn["gvkey1"] == tn["gvkey2"], "gvkey1"], tn.loc[tn["gvkey1"] == tn["gvkey2"], "year"]))
    peers = tn[tn["gvkey1"] != tn["gvkey2"]]
    n_peers = peers.groupby(["gvkey1", "year"]).size().to_dict()
    score = {(a, b, y): s for a, b, y, s in zip(peers["gvkey1"], peers["gvkey2"], peers["year"], peers["score"])}

    rows = []
    for k in range(len(d)):
        y = int(year[k]) - 1
        o = {f: np.nan for f in FEATURES}
        o["deal_id"] = d["deal_id"].iat[k]
        t, a = tg[k], ag[k]
        o["ov_acq_compustat"] = float(isinstance(a, str))
        ft = firm(t, y) if isinstance(t, str) else (ciq_sic[k], np.nan, y)
        s_t, sh_t = ft[0], ft[1]
        if pd.isna(s_t) and pd.notna(ciq_sic[k]):
            s_t = ciq_sic[k]
        if pd.notna(s_t):
            o["ov_tgt_share"] = sh_t
            o["ov_hhi"] = hhi.get((int(s_t), ft[2]), np.nan)
        if isinstance(a, str):
            fa_ = firm(a, y)
            s_a, sh_a = fa_[0], fa_[1]
            o["ov_acq_share"] = sh_a
            if pd.notna(s_t) and pd.notna(s_a):
                s_t, s_a = int(s_t), int(s_a)
                o["ov_same_sic4"], o["ov_same_sic3"], o["ov_same_sic2"] = float(s_t == s_a), float(s_t // 10 == s_a // 10), float(s_t // 100 == s_a // 100)
                o["ov_delta_hhi"] = 2 * sh_t * sh_a if (s_t == s_a and pd.notna(sh_t) and pd.notna(sh_a)) else (0.0 if s_t != s_a else np.nan)
        if isinstance(t, str):
            ti = int(t)
            for yy in (y, y - 1, y - 2, y - 3):                     # latest TNIC year available before the announcement
                if yy > tmax:
                    continue
                if (ti, yy) in universe:
                    o["ov_tnic_tgt_rivals"] = float(np.log1p(n_peers.get((ti, yy), 0)))
                    if isinstance(a, str) and (int(a), yy) in universe:
                        sc = score.get((ti, int(a), yy))
                        o["ov_tnic_pair"], o["ov_tnic_score"] = float(sc is not None), float(sc) if sc is not None else 0.0
                    break
        rows.append(o)
    O = pd.DataFrame(rows)[["deal_id"] + FEATURES]
    O.to_parquet(INTERIM / "overlap_features.parquet", index=False)
    print(f"TNIC years to {tmax}; deals {len(O):,}; acquirer in Compustat {int(O['ov_acq_compustat'].sum()):,}; "
          f"TNIC pair known {int(O['ov_tnic_pair'].notna().sum()):,} (peers {int((O['ov_tnic_pair'] == 1).sum()):,}); same SIC4 known {int(O['ov_same_sic4'].notna().sum()):,} "
          f"(same {int((O['ov_same_sic4'] == 1).sum()):,})")


def evaluate() -> None:
    import modeling as M
    import pooled_oot as PO
    import duration as DU
    import train_eval as TE
    res = ROOT / "results"
    D = M.load_data("")
    assert D.has_overlap, "run `python src/overlap.py build` first"
    y = D.y["label_adverse"]
    idx = np.arange(len(y))
    O = D.X[FEATURES]
    out = {}
    cls = D.outcome3
    L = ["# Antitrust overlap: product similarity and industry shares", "",
         "Feature group added to the break model and the time-to-closing model. Sources: Hoberg-Phillips TNIC-3 product similarity of acquirer and target (public data, "
         "latest year before the announcement, data end in 2023) and Compustat sales shares by 4-digit SIC industry. The acquirer must be a Compustat firm; "
         "for private, sponsor and most foreign buyers the pair features are missing, which the tree models treat as its own category.", "",
         "## Coverage (core sample)", "",
         f"- Acquirer matched to Compustat: {int((O['ov_acq_compustat'][D.core] == 1).sum()):,} of {int(D.core.sum()):,} deals.",
         f"- TNIC pair status known (both firms in the TNIC universe): {int(O['ov_tnic_pair'][D.core].notna().sum()):,}; of these {int((O['ov_tnic_pair'][D.core] == 1).sum()):,} are TNIC-3 peers.",
         f"- Same 4-digit SIC known: {int(O['ov_same_sic4'][D.core].notna().sum()):,}; same industry in {int((O['ov_same_sic4'][D.core] == 1).sum()):,}.", "",
         "## What the raw data say (core sample, all years; descriptive)", "",
         "| Group | Deals | Adverse break rate | Median days to closing (completed deals) |", "|---|---:|---:|---:|"]
    sc = O["ov_tnic_score"].to_numpy()
    hi = np.nanmedian(sc[(O["ov_tnic_pair"] == 1).to_numpy() & D.core])
    groups = {"Acquirer not in Compustat (private, sponsor, foreign)": (O["ov_acq_compustat"] == 0).to_numpy(),
              "Both in TNIC, not peers": (O["ov_tnic_pair"] == 0).to_numpy(), "TNIC peers, similarity below the peer median": ((O["ov_tnic_pair"] == 1) & (sc < hi)).to_numpy(),
              "TNIC peers, similarity above the peer median": ((O["ov_tnic_pair"] == 1) & (sc >= hi)).to_numpy(),
              "Same 4-digit SIC": (O["ov_same_sic4"] == 1).to_numpy(), "Different 4-digit SIC": (O["ov_same_sic4"] == 0).to_numpy()}
    desc = {}
    for g, m in groups.items():
        m = m & D.core
        dd = D.days_out[m & (D.y["label"] == 0)]
        desc[g] = {"n": int(m.sum()), "adverse_rate": float(y[m].mean()), "median_days": float(np.nanmedian(dd))}
        L.append(f"| {g} | {int(m.sum())} | {y[m].mean():.1%} | {np.nanmedian(dd):.0f} |")
    out["descriptive"] = desc
    L.append("")

    # ---------------------------------------------------------------- break model
    specs = pickle.load(open(INTERIM / "model_specs.pkl", "rb"))[("primary", "full")]
    final = json.loads((res / "metrics.json").read_text())["final_models"]["overall"]
    kind, ft, bp = specs["feats"][final]
    tr, te = specs["tr"], specs["te"]
    folds = M.cv_folds(D.year, tr)
    variants = {"overlap": dataclasses.replace(ft, groups=ft.groups + ("overlap",)), "overlap_sic": dataclasses.replace(ft, groups=ft.groups + ("overlap_sic",))}
    P = {"final": pd.read_parquet(INTERIM / "predictions.parquet").set_index("deal_id").reindex(D.deals["deal_id"])[f"pred__primary__full__{final}"].to_numpy()[te]}
    cv = {"final": json.loads((res / "metrics.json").read_text())["final_models"]["cv_auc"][final]}
    for name, f in variants.items():
        b2, s, _, _ = M.cv_lgbm(D, y, f, folds)
        P[name] = M.fit_lgbm(D, y, f, tr, te, b2)[0]; cv[name] = s
    ci, diffs = M.bootstrap(y[te], P, [("overlap", "final"), ("overlap_sic", "final")], n_boot=2000)
    pub = (O["ov_acq_compustat"].to_numpy()[te] == 1)
    ci_p, diffs_p = M.bootstrap(y[te][pub], {k: v[pub] for k, v in P.items()}, [("overlap", "final")], n_boot=2000)
    out["break_test"] = {"metrics": {k: M.metrics(y[te], p) for k, p in P.items()}, "ci95": ci, "diffs": diffs, "cv_auc": cv,
                         "public_acquirer": {"n": int(pub.sum()), "pos": int(y[te][pub].sum()), "metrics": {k: M.metrics(y[te][pub], p[pub]) for k, p in P.items()}, "diffs": diffs_p}}
    # pooled out-of-time
    Z = pickle.load(open(INTERIM / "pooled_oot.pkl", "rb"))
    base = Z["pred"].set_index("deal_id").reindex(D.deals["deal_id"])["pred__full__final_proc"].to_numpy()
    po = {k: np.full(len(y), np.nan) for k in variants}
    for Y in Z["years"]:
        s = Z["selection"].get((Y, "full"))
        if not s or "final_proc" not in s:
            continue
        ch = s["final_proc"]["chosen"]; par = s[ch]
        g0 = () if ch == "lgbm_base" else (TE.EXT if ch == "lgbm_ext" else TE.EXTM)
        a, b = idx[D.core & (D.year <= Y - 1)], idx[D.core & (D.year == Y)]
        for k, extra in (("overlap", "overlap"), ("overlap_sic", "overlap_sic")):
            f = M.Feat(groups=g0 + (extra,), base_only=(ch == "lgbm_base"), year=par["use_year"])
            po[k][b] = M.fit_lgbm(D, y, f, a, b, (par["num_leaves"], par["n_estimators"]))[0]
    ok = np.isfinite(base) & np.isfinite(po["overlap"])
    ev = PO.evaluate(y[ok], D.year[ok], {"final": base[ok], "overlap": po["overlap"][ok], "overlap_sic": po["overlap_sic"][ok]}, [("overlap", "final"), ("overlap_sic", "final")], auc_only=True)
    ev["n"], ev["pos"] = int(ok.sum()), int(y[ok].sum())
    out["break_pooled"] = ev
    pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "oot__overlap": po["overlap"], "oot__overlap_sic": po["overlap_sic"]}).to_parquet(INTERIM / "overlap_preds.parquet", index=False)
    mt, d1, d2 = out["break_test"]["metrics"], diffs["overlap minus final"]["roc_auc"], diffs["overlap_sic minus final"]["roc_auc"]
    dp = diffs_p["overlap minus final"]["roc_auc"]
    L += ["## Break model (adverse break, core sample)", "",
          "| Evaluation | Deals | Adverse breaks | Final model | + overlap group | Difference [95% CI] | + SIC and sales measures only | Difference [95% CI] |", "|---|---:|---:|---:|---:|---|---:|---|",
          f"| Test period 2019-2025 | {len(te)} | {int(y[te].sum())} | {mt['final']['roc_auc']:.3f} | {mt['overlap']['roc_auc']:.3f} | {d1['point']:+.3f} [{d1['ci95'][0]:+.3f}, {d1['ci95'][1]:+.3f}] | "
          f"{mt['overlap_sic']['roc_auc']:.3f} | {d2['point']:+.3f} [{d2['ci95'][0]:+.3f}, {d2['ci95'][1]:+.3f}] |",
          f"| Test period, acquirer in Compustat | {int(pub.sum())} | {int(y[te][pub].sum())} | {M.metrics(y[te][pub], P['final'][pub])['roc_auc']:.3f} | {M.metrics(y[te][pub], P['overlap'][pub])['roc_auc']:.3f} | "
          f"{dp['point']:+.3f} [{dp['ci95'][0]:+.3f}, {dp['ci95'][1]:+.3f}] | n/a | n/a |"]
    r1, r2 = ev["pairs"]["overlap minus final"], ev["pairs"]["overlap_sic minus final"]
    L += [f"| Pooled out-of-time 2013-2025 | {ev['n']} | {ev['pos']} | {ev['scores']['final']['roc_auc']:.3f} | {ev['scores']['overlap']['roc_auc']:.3f} | {PO.d3(r1['roc_auc'], r1['ci95_year']['roc_auc'])} | "
          f"{ev['scores']['overlap_sic']['roc_auc']:.3f} | {PO.d3(r2['roc_auc'], r2['ci95_year']['roc_auc'])} |", "",
          f"CV AUC inside the training years (2001-2018): final model {cv['final']:.3f}, with the overlap group {cv['overlap']:.3f}, with SIC and sales measures only {cv['overlap_sic']:.3f}. "
          "Intervals: bootstrap over deals (test period) and over announcement years (pooled).", ""]

    # ---------------------------------------------------------------- duration model
    ZD = pickle.load(open(INTERIM / "duration.pkl", "rb"))
    PD_, sel = ZD["pred"].set_index("deal_id").reindex(D.deals["deal_id"]).reset_index(), ZD["selection"]
    closed = D.y["label"] == 0
    days = np.clip(D.days_out.astype(float), 1, None)
    ly = np.log(days)
    base_feats = {"lgbm_base": M.Feat(base_only=True), "lgbm_ext": M.Feat(groups=TE.EXT), "lgbm_extm": M.Feat(groups=TE.EXTM), "lgbm_llm": M.Feat(groups=TE.EXTM, has_text_flag=True, llm=True)}
    dur = {"split": np.full(len(y), np.nan), "oot": np.full(len(y), np.nan)}
    for (tag, Y, subset), I in sel.items():
        if subset != "full" or "final" not in I:
            continue
        ch = I["final"]["chosen"]
        f = dataclasses.replace(base_feats[ch], groups=base_feats[ch].groups + ("overlap",))
        a = idx[D.core & closed & (D.year <= (M.TRAIN_END if Y is None else Y - 1))]
        b = idx[D.core & ((D.year >= M.TEST_START) if Y is None else (D.year == Y))]
        dur[tag][b] = DU.fit_reg(D, ly, f, a, b, (I[ch]["num_leaves"], I[ch]["n_estimators"]))[0]
    L += ["## Time-to-closing model (completed deals)", "", "| Evaluation | Completed deals | MAE days, final model | MAE days, + overlap group | Difference [95% CI] | R2 on log days: difference [95% CI] |", "|---|---:|---:|---:|---|---|"]
    for tag, title, by_year in (("split", "Test period 2019-2025", False), ("oot", "Pooled out-of-time 2013-2025", True)):
        c = f"{tag}__full__final"
        ok = (D.core & closed & PD_[c].notna().to_numpy() & np.isfinite(dur[tag]))
        e = DU.evaluate(days[ok], D.year[ok], {"final": PD_[c].to_numpy()[ok], "overlap": dur[tag][ok]}, [("overlap", "final")], by_year)
        out[f"duration_{tag}"] = e
        r = e["pairs"]["overlap minus final"]
        L.append(f"| {title} | {int(ok.sum())} | {e['scores']['final']['mae']:.1f} | {e['scores']['overlap']['mae']:.1f} | {r['mae']:+.1f} [{r['ci95']['mae'][0]:+.1f}, {r['ci95']['mae'][1]:+.1f}] | "
                 f"{r['r2_log']:+.3f} [{r['ci95']['r2_log'][0]:+.3f}, {r['ci95']['r2_log'][1]:+.3f}] |")
    pd.DataFrame({"deal_id": D.deals["deal_id"].to_numpy(), "split__overlap": dur["split"], "oot__overlap": dur["oot"]}).to_parquet(INTERIM / "overlap_duration_preds.parquet", index=False)
    L += ["", "Same hyper-parameters as the final time-to-closing model of each origin; only the feature group is added.", ""]
    (res / "overlap.md").write_text("\n".join(L) + "\n")
    (res / "overlap.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    a = ap.parse_args()
    {"build": build, "eval": evaluate}[a.cmd]()

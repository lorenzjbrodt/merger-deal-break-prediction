"""Stylized dollar-impact simulation on the test period.

There is no price data in this project, so this is NOT a backtest. It converts the
model's ranking of deals into dollars under explicit, labelled assumptions:

  A1. The fund holds every announced deal in the test period, equal-weighted.
  A2. A deal that closes earns a gross spread s; a deal that breaks loses L. Both are the
      same for every deal (in reality the market charges a wider spread for riskier deals).
  A3. Capital is turned once per year: $100 million of deal exposure is put on per year
      and spread equally over the deals held. No leverage, financing cost, fees or taxes.
  A4. "Skip" removes the flagged deals and redeploys their capital equally over the
      remaining deals; "halve" gives flagged deals half weight. A no-redeployment variant
      (skipped capital sits in cash at 0%) is also shown.
  A5. Flagged = the riskiest 10% / 20% of test-period deals by model score. This ranks
      inside the test period (an ex-post cut-off); a variant with a cut-off fixed in
      advance from training-period out-of-fold scores is reported as a check.

Usage: python src/impact.py [--tag synthetic]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPITAL = 100e6
SPREADS = [0.02, 0.03, 0.04, 0.05]
LOSSES = [0.15, 0.20, 0.25, 0.30]
BASE_S, BASE_L = 0.03, 0.20
STRATS = [("skip", 0.10), ("skip", 0.20), ("halve", 0.10), ("halve", 0.20)]
N_BOOT = 2000


def flag_top(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(int(round(frac * len(score))), 1)
    f = np.zeros(len(score), dtype=bool)
    f[np.argsort(-score, kind="stable")[:k]] = True
    return f


def weights(flag: np.ndarray, how: str, redeploy: bool = True) -> np.ndarray:
    w = np.ones(len(flag))
    if how == "skip":
        w[flag] = 0.0
    elif how == "halve":
        w[flag] = 0.5
    return w / (w.sum() if redeploy else len(flag))


def pnl(y: np.ndarray, w: np.ndarray, s: float, L: float) -> float:
    """Dollars per $100m of exposure put on per year."""
    return float(CAPITAL * np.sum(w * np.where(y == 1, -L, s)))


def delta(y, score, how, frac, s, L, redeploy=True, flag=None) -> float:
    f = flag_top(score, frac) if flag is None else flag
    return pnl(y, weights(f, how, redeploy), s, L) - pnl(y, weights(np.zeros(len(y), bool), "hold"), s, L)


def grid_md(fn, title: str) -> list[str]:
    rows = [f"**{title}** ($ million per $100 million per year)", "",
            "| Spread s \\ Loss L | " + " | ".join(f"{L:.0%}" for L in LOSSES) + " |", "|---|" + "---:|" * len(LOSSES)]
    for s in SPREADS:
        rows.append(f"| {s:.0%} | " + " | ".join(f"{fn(s, L) / 1e6:+.2f}" for L in LOSSES) + " |")
    return rows + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()
    synthetic = args.tag != "real"
    suffix = "" if not synthetic else f"_{args.tag}"
    res_dir = ROOT / ("results" if not synthetic else f"results_{args.tag}")
    fig_dir = ROOT / ("figures" if not synthetic else f"figures_{args.tag}")
    R = pd.read_parquet(ROOT / f"data/interim/predictions{suffix}.parquet")
    M = json.loads((res_dir / "metrics.json").read_text())
    final, final_tab = M["final_models"]["overall"], M["final_models"]["tabular"]
    # primary run: core sample, adverse-break target, test period. A deal topped by a higher bid is NOT a break here.
    R = R[R["core"] & (R["year"] >= 2019) & R[f"pred__primary__full__{final}"].notna()].reset_index(drop=True)
    P = pd.DataFrame({"label": R["label_adverse"], "year": R["year"], "in_text_subset": R["has_text"]})
    for c in R.columns:
        if c.startswith("pred__primary__full__"):
            P["full__" + c.split("__")[-1]] = R[c]
        elif c.startswith("pred__primary__text__"):
            P["text__" + c.split("__")[-1]] = R[c]
    y = P["label"].to_numpy()
    n, b = len(y), y.mean()
    years = P["year"].nunique()
    rng = np.random.default_rng(0)

    names = {"full__one_rule": "1. One rule", "full__logit_base": "2. Logistic, base tabular", "full__lgbm_base": "3. LightGBM, base tabular",
             f"full__{final_tab}": f"Final tabular model ({final_tab})", f"full__{final}": f"Final model ({final})"}
    names = {k: v for k, v in names.items() if k in P.columns}
    main_model = f"full__{final}"
    sc = P[main_model].to_numpy()

    out = {"assumptions": {"capital_per_year": CAPITAL, "base_spread": BASE_S, "base_loss": BASE_L,
                           "test_deals": int(n), "test_breaks": int(y.sum()), "test_years": int(years)},
           "hold_all": {}, "delta_vs_hold_all": {}, "base_case": {}}
    L_ = ["# Dollar impact: stylized simulation" + (" (SYNTHETIC SAMPLE: not real results)" if synthetic else ""), "",
          "**Secondary view.** The headline impact is the price-based backtest in `results/backtest.md`, which covers the deals with usable prices. "
          "This grid covers every test-period deal instead, at the cost of assumed payoffs: it translates the model's ranking into dollars under the "
          "assumptions listed here, and every figure is conditional on them.", "",
          "## Assumptions", "",
          f"- **Assumption 1 (portfolio).** The fund holds every deal announced in the test period (2019-2025, core sample without reverse mergers: {n:,} deals, "
          f"{int(y.sum())} adverse breaks, rate {b:.1%}), equal-weighted. A deal topped by a higher bid is treated like a completed deal (it earns *s*); in reality it usually earns more.",
          "- **Assumption 2 (payoffs).** A deal that closes earns a gross spread *s*; a deal that breaks loses *L*. "
          "Both are identical across deals. Real spreads are wider for deals the market already sees as risky, so a "
          "constant *s* overstates the benefit of skipping flagged deals. See the break-even spread below.",
          "- **Assumption 3 (capital).** $100 million of deal exposure is put on per year and turned once per year. "
          "No leverage, financing cost, transaction cost, fees or taxes. Deals are pooled over the test period, so the "
          "figures are per-year averages, not a year-by-year path.",
          "- **Assumption 4 (actions).** \"Skip\" removes flagged deals and redeploys their capital equally into the "
          "remaining deals. \"Halve\" gives flagged deals half weight, again fully invested.",
          "- **Assumption 5 (flagging).** Flagged deals are the riskiest 10% or 20% of test-period deals by model score. "
          "This ranks deals inside the test period. A check with a cut-off fixed in advance is shown at the end.",
          f"- **Sensitivity grid.** *s* from 2% to 5%, *L* from 15% to 30%. Base case for the summary rows: *s* = {BASE_S:.0%}, *L* = {BASE_L:.0%}.",
          "- Scores come from models trained on 2001-2018 only.", ""]

    L_ += ["## Hold every deal", ""]
    hold = lambda s, L: pnl(y, weights(np.zeros(n, bool), "hold"), s, L)
    L_ += grid_md(hold, "Hold all deals: P&L")
    out["hold_all"] = {f"s={s:.2f},L={L:.2f}": hold(s, L) for s in SPREADS for L in LOSSES}
    be = b * BASE_L / (1 - b)
    L_ += [f"At the test-period break rate of {b:.1%} and *L* = {BASE_L:.0%}, holding everything breaks even at a spread of {be:.2%}.", ""]

    L_ += [f"## Change versus holding every deal, using the final model ({final})", ""]
    for how, frac in STRATS:
        f = flag_top(sc, frac)
        cap = y[f].sum() / max(y.sum(), 1)
        title = f"{how.capitalize()} the riskiest {frac:.0%} ({int(f.sum())} deals, {int(y[f].sum())} of {int(y.sum())} adverse breaks = {cap:.0%})"
        fn = lambda s, L, how=how, frac=frac: delta(y, sc, how, frac, s, L)
        L_ += grid_md(fn, title + ": change in P&L")
        out["delta_vs_hold_all"][f"{how}_top{int(frac * 100)}"] = {f"s={s:.2f},L={L:.2f}": fn(s, L) for s in SPREADS for L in LOSSES}

    # ---- base case across models, with bootstrap CIs
    L_ += [f"## Base case (*s* = {BASE_S:.0%}, *L* = {BASE_L:.0%}): change versus holding every deal, by model", "",
           f"Hold-all P&L in the base case: ${hold(BASE_S, BASE_L) / 1e6:.2f} million per $100 million per year. "
           f"Brackets: bootstrap 95% interval over test deals ({N_BOOT} resamples; deals re-ranked in each resample).", "",
           "| Model | " + " | ".join(f"{h.capitalize()} riskiest {f:.0%}" for h, f in STRATS) + " |", "|---|" + "---|" * len(STRATS)]
    boot_ix = rng.integers(0, n, (N_BOOT, n))
    for col, label in names.items():
        s_ = P[col].to_numpy()
        cells = []
        for how, frac in STRATS:
            pt = delta(y, s_, how, frac, BASE_S, BASE_L)
            bs = [delta(y[ix], s_[ix], how, frac, BASE_S, BASE_L) for ix in boot_ix]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out["base_case"].setdefault(col, {})[f"{how}_top{int(frac * 100)}"] = {"delta": pt, "ci95": [float(lo), float(hi)]}
            cells.append(f"{pt / 1e6:+.2f} [{lo / 1e6:+.2f}, {hi / 1e6:+.2f}]")
        L_.append(f"| {label} | " + " | ".join(cells) + " |")
    L_ += ["", "A random flag (no model) changes expected P&L by $0 in this setup, because capital is redeployed into "
           "deals with the same expected payoff.", ""]

    # ---- text subset: does text change the dollars?
    if "text__lgbm_alltext" in P.columns and P["text__lgbm_alltext"].notna().sum() > 20:
        T = P[P["text__lgbm_alltext"].notna()]
        yt = T["label"].to_numpy(); nt = len(yt)
        ix_t = rng.integers(0, nt, (N_BOOT, nt))
        L_ += ["## Text subset: extended tabular versus tabular + text", "",
               f"Deals with an announcement text only ({nt:,} test deals, {int(yt.sum())} breaks). Base case, change versus "
               "holding every deal in this subset. The last row is the paired difference between the two models.", "",
               "| Model | " + " | ".join(f"{h.capitalize()} riskiest {f:.0%}" for h, f in STRATS) + " |", "|---|" + "---|" * len(STRATS)]
        cols = {"text__lgbm_extm": "3m. Extended tabular + market context", "text__lgbm_tfidf": "4. + keywords + TF-IDF",
                "text__lgbm_emb": "5. + keywords + embeddings", "text__lgbm_llm": "5L. + keywords + LLM terms", "text__lgbm_alltext": "5A. + all text groups"}
        draws = {}
        for col, label in cols.items():
            if col not in T.columns:
                continue
            s_ = T[col].to_numpy(); cells = []
            for how, frac in STRATS:
                pt = delta(yt, s_, how, frac, BASE_S, BASE_L)
                bs = np.array([delta(yt[ix], s_[ix], how, frac, BASE_S, BASE_L) for ix in ix_t])
                draws[(col, how, frac)] = (pt, bs)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out["base_case"].setdefault(col, {})[f"{how}_top{int(frac * 100)}"] = {"delta": pt, "ci95": [float(lo), float(hi)]}
                cells.append(f"{pt / 1e6:+.2f} [{lo / 1e6:+.2f}, {hi / 1e6:+.2f}]")
            L_.append(f"| {label} | " + " | ".join(cells) + " |")
        cells = []
        for how, frac in STRATS:
            p5, b5 = draws[("text__lgbm_alltext", how, frac)]; p3, b3 = draws[("text__lgbm_extm", how, frac)]
            lo, hi = np.percentile(b5 - b3, [2.5, 97.5])
            out["base_case"].setdefault("text__alltext_minus_extm", {})[f"{how}_top{int(frac * 100)}"] = {"delta": p5 - p3, "ci95": [float(lo), float(hi)]}
            cells.append(f"{(p5 - p3) / 1e6:+.2f} [{lo / 1e6:+.2f}, {hi / 1e6:+.2f}]")
        L_ += ["| 5A minus 3m (paired) | " + " | ".join(cells) + " |", ""]

    # ---- checks
    L_ += ["## Checks on the assumptions", ""]
    f10, f20 = flag_top(sc, 0.10), flag_top(sc, 0.20)
    rows = ["| Flagged group (final model) | Deals | Break rate | Break-even spread on flagged deals |", "|---|---:|---:|---:|"]
    out["break_even_spread"] = {}
    for lab, f in (("riskiest 10%", f10), ("riskiest 20%", f20)):
        bf, br = y[f].mean(), y[~f].mean()
        r_rest = (1 - br) * BASE_S - br * BASE_L
        be_f = (r_rest + bf * BASE_L) / (1 - bf) if bf < 1 else np.nan
        out["break_even_spread"][lab] = {"break_rate_flagged": float(bf), "break_rate_rest": float(br), "break_even_spread": float(be_f)}
        rows.append(f"| {lab} | {int(f.sum())} | {bf:.1%} (rest: {br:.1%}) | {be_f:.1%} |")
    L_ += rows + ["", f"Reading: with *L* = {BASE_L:.0%} and the remaining deals earning *s* = {BASE_S:.0%}, skipping the flagged deals "
                  "only pays if their actual market spread is below the break-even value. If the market already prices these "
                  "deals at a wider spread than that, the fund is being paid for the risk and skipping destroys value. "
                  "This is the main reason the numbers above are an upper-end view.", ""]

    rows = ["| Variant (final model, base case) | " + " | ".join(f"{h.capitalize()} riskiest {f:.0%}" for h, f in STRATS) + " |", "|---|" + "---:|" * len(STRATS)]
    rows.append("| Redeploy skipped capital (main) | " + " | ".join(f"{delta(y, sc, h, f, BASE_S, BASE_L) / 1e6:+.2f}" for h, f in STRATS) + " |")
    rows.append("| Skipped capital idle at 0% | " + " | ".join(f"{delta(y, sc, h, f, BASE_S, BASE_L, redeploy=False) / 1e6:+.2f}" for h, f in STRATS) + " |")
    q = (M["runs"]["primary"]["full"]["selection"].get(final, {}) or {}).get("oof_quantiles")
    if q:
        cells = []
        for how, frac in STRATS:
            f = sc >= (q["q90"] if frac == 0.10 else q["q80"])
            cells.append(f"{delta(y, sc, how, frac, BASE_S, BASE_L, flag=f) / 1e6:+.2f} ({f.mean():.0%} flagged)")
        rows.append("| Cut-off fixed in advance (training-period out-of-fold score quantile) | " + " | ".join(cells) + " |")
        out["ex_ante_threshold"] = {"q80": q["q80"], "q90": q["q90"],
                                    "share_flagged_q90": float((sc >= q["q90"]).mean()), "share_flagged_q80": float((sc >= q["q80"]).mean())}
    L_ += rows + ["", "$ million per $100 million per year, change versus holding every deal.", ""]
    (res_dir / "impact.md").write_text("\n".join(L_) + "\n")
    (res_dir / "impact.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("Hold every deal: P&L", lambda s, L: hold(s, L)),
              ("Skip riskiest 10%: change vs hold-all", lambda s, L: delta(y, sc, "skip", 0.10, s, L)),
              ("Skip riskiest 20%: change vs hold-all", lambda s, L: delta(y, sc, "skip", 0.20, s, L))]
    for ax, (title, fn) in zip(axes, panels):
        Z = np.array([[fn(s, L) / 1e6 for L in LOSSES] for s in SPREADS])
        v = max(np.abs(Z).max(), 0.5)
        ax.imshow(Z, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(LOSSES)), [f"{L:.0%}" for L in LOSSES]); ax.set_yticks(range(len(SPREADS)), [f"{s:.0%}" for s in SPREADS])
        ax.set_xlabel("Loss on a broken deal, L (assumption)"); ax.set_ylabel("Gross spread on a closed deal, s (assumption)")
        for i in range(len(SPREADS)):
            for j in range(len(LOSSES)):
                ax.text(j, i, f"{Z[i, j]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(r"Stylized simulation, test 2019-2025, final model: \$ million per \$100 million per year"
                 + (" (synthetic sample)" if synthetic else ""), x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "impact_sensitivity.png"); plt.close(fig)
    print("\n".join(L_))


if __name__ == "__main__":
    main()

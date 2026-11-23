"""Local demo: break risk and the underwriting view (time to closing, loss if it breaks, expected annualized return) for a signed M&A deal.

    streamlit run app/demo.py

Runs offline from artifacts/demo/demo.pkl (built by src/build_demo_artifacts.py, git-ignored).
If that file is missing it falls back to the SYNTHETIC sample and says so on the page.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_demo_artifacts as B   # noqa: E402
import features as F               # noqa: E402

st.set_page_config(page_title="Deal break risk", layout="wide")

NICE = {
    "log_deal_value": "Deal size", "premium_1d": "Premium (1 day)", "cash_pct": "Cash share of consideration", "solicited": "Solicited approach",
    "buyer_listed": "Buyer is listed", "buyer_financial": "Financial buyer", "feat_tender_offer": "Tender offer", "feat_lbo": "Leveraged buyout",
    "feat_go_shop": "Go-shop provision", "feat_cross_border": "Cross-border", "feat_going_private": "Going private",
    "sell_fee_pct_value": "Target termination fee (% of value)", "buy_fee_pct_value": "Reverse termination fee (% of value)",
    "rel_size_deal_to_acq": "Deal value / acquirer market cap",
}
REASON = {"competing_bid": "superior competing bid", "regulatory": "regulatory or antitrust block", "financing": "financing failure",
          "shareholder_vote": "shareholder vote failed", "buyer_walked_mac": "buyer walked / MAC dispute", "mutual_other": "mutual termination or other",
          "unknown": "reason not stated"}


@st.cache_resource
def load():
    real = ROOT / "artifacts/demo/demo.pkl"
    path = real if real.exists() else ROOT / "artifacts/demo/demo_synthetic.pkl"
    if not path.exists():
        path = B.build(synthetic=True)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def nice(name: str) -> str:
    return NICE.get(name, name.replace("kw_", "text mentions: ").replace("_", " "))


def drivers_chart(drivers):
    df = pd.DataFrame(drivers, columns=["feature", "value", "contribution"])
    df["label"] = [f"{nice(f)}" + ("" if v is None else f" = {v:,.2f}") for f, v in zip(df["feature"], df["value"])]
    df = df.iloc[::-1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(df["label"], df["contribution"], color=["#de2d26" if c > 0 else "#3182bd" for c in df["contribution"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Contribution to break risk (log-odds; red raises, blue lowers)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def similar_deals(art, text: str, k: int = 5) -> pd.DataFrame:
    sim = art.get("similar")
    if not sim or not text.strip():
        return pd.DataFrame()
    q = sim["vectorizer"].transform([text])
    s = (sim["matrix"] @ q.T).toarray().ravel()
    top = np.argsort(-s)[:k]
    return pd.DataFrame([{"Target": sim["meta"][i]["target"], "Buyer": sim["meta"][i]["buyer"][:60], "Year": sim["meta"][i]["year"],
                          "Why it broke": REASON.get(sim["meta"][i]["reason"], sim["meta"][i]["reason"]), "Text similarity": round(float(s[i]), 2)} for i in top])


def headline(p: float, pct: float, base: float, p_mkt):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model break probability", f"{p:.1%}", f"{p / base:.1f}x the base rate", delta_color="off")
    c2.metric("Percentile versus history", f"{pct:.0%}", "share of past deals scored lower", delta_color="off")
    c3.metric("Market-implied probability", "n/a" if p_mkt is None else f"{p_mkt:.0%}",
              "no usable price" if p_mkt is None else ("model sees more risk" if p > p_mkt else "market sees more risk"), delta_color="off")
    c4.metric("Base rate (training period)", f"{base:.1%}")


def underwriting(u, spread):
    """Second row: the three numbers an arbitrageur underwrites and the expected annualized return they imply."""
    if not u:
        st.caption("Underwriting view (time to closing, loss if it breaks, expected annualized return): not available for this deal.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected time to closing", "n/a" if u["days"] is None else f"{u['days']:.0f} days",
              "" if u["days_lo"] is None else f"80% interval {u['days_lo']:.0f} to {u['days_hi']:.0f} days" + (" (reads the stated closing)" if u.get("uses_guidance") else ""), delta_color="off")
    c2.metric("Expected loss if it breaks", "n/a" if u["loss"] is None else f"{u['loss']:.0%}", "target return from the day-1 close", delta_color="off")
    c3.metric("Expected annualized return", "n/a" if u["ear"] is None else f"{u['ear']:.1%}",
              "all-cash deals with a CRSP price only" if u["ear"] is None else ("day-1 spread n/a" if spread is None else f"day-1 spread {spread:.1%}"), delta_color="off")
    c4.metric("Rank by expected annualized return", "n/a" if u["ear_pct"] is None else f"{u['ear_pct']:.0%}", "share of test-period deals ranked lower", delta_color="off")


art = load()
st.title("Will this deal close? Break risk at announcement")
if art["synthetic"]:
    st.warning("SYNTHETIC SAMPLE. The real artefacts are not on this machine, so the app runs on made-up deals. Numbers are meaningless.")
st.caption(f"Break = cancelled for a reason other than a superior competing bid. Scores: {art['source']}. "
           "This is a course project, not investment advice.")

mode = st.sidebar.radio("Mode", ["Pick a test-period deal", "Enter your own deal"])
deals = art["deals"]

if mode == "Pick a test-period deal":
    order = st.sidebar.selectbox("Sort deals by", ["Highest model risk", "Announcement date", "Name"])
    key = {"Highest model risk": lambda d: -d["score"], "Announcement date": lambda d: d["announced"], "Name": lambda d: d["target"]}[order]
    lst = sorted(deals, key=key)
    d = st.sidebar.selectbox("Deal", lst, format_func=lambda d: f"{d['target']} ({d['announced'][:4]})")
    show = st.sidebar.checkbox("Reveal what happened", value=False)
    st.subheader(f"{d['target']}  <-  {d['buyer'][:90]}")
    st.write(f"Announced {d['announced']} | deal value ${d['deal_value']:,.0f}m | consideration: {d['consideration']}"
             + ("" if d["premium_1d"] is None else f" | premium {d['premium_1d']:.0f}%")
             + ("" if d["spread"] is None else f" | {'spread at first monthly mark' if art.get('price_source') == 'Capital IQ monthly' else 'day-1 spread'} {d['spread']:.1%}"))
    headline(d["score"], d["percentile"], art["base_rate"], d["p_mkt"])
    underwriting(d.get("underwriting"), d["spread"])
    if show:
        out = {"closed": "The deal closed.", "topped": "The deal was topped by a higher bid.", "adverse_break": "The deal broke."}[d["outcome"]]
        (st.error if d["outcome"] == "adverse_break" else st.success)(f"Outcome: {out}")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Top drivers of the score**")
        st.pyplot(drivers_chart(d["drivers"]))
        st.markdown("**Most similar historical broken deals (by announcement text)**")
        sd = similar_deals(art, d["press"])
        if len(sd):
            st.dataframe(sd, hide_index=True)
        else:
            st.write("No text available for this deal.")
    with right:
        st.markdown("**Terms extracted from the filing text**")
        if d["terms"]:
            st.dataframe(pd.DataFrame([{"Term": k.replace("_", " "), "Value": ", ".join(v) if isinstance(v, list) else str(v)} for k, v in d["terms"].items()]), hide_index=True)
        else:
            st.write("No extracted terms for this deal.")
        with st.expander("Opening of the announcement"):
            st.write(d["press"] or "No text.")
else:
    st.sidebar.info("This mode uses a compact model on a handful of terms plus keyword flags from the pasted text. It is weaker than the full model.")
    med = art["compact_medians"]
    c1, c2, c3 = st.columns(3)
    value = c1.number_input("Deal value ($ million)", 100.0, 200000.0, 1500.0, step=100.0)
    prem = c2.number_input("Premium to prior-day price (%)", -20.0, 200.0, 30.0, step=1.0)
    cash = c3.slider("Cash share of consideration (%)", 0, 100, 100)
    c1, c2, c3 = st.columns(3)
    sell_fee = c1.number_input("Target termination fee (% of value)", 0.0, 15.0, 3.0, step=0.1)
    buy_fee = c2.number_input("Reverse termination fee (% of value, 0 = none)", 0.0, 25.0, 0.0, step=0.1)
    rel = c3.number_input("Deal value / acquirer market cap (0 = buyer not listed)", 0.0, 10.0, 0.3, step=0.05)
    c1, c2, c3, c4 = st.columns(4)
    fin = c1.checkbox("Financial buyer"); lbo = c2.checkbox("Leveraged buyout"); tender = c3.checkbox("Tender offer"); goshop = c4.checkbox("Go-shop")
    c1, c2, c3, c4 = st.columns(4)
    xb = c1.checkbox("Cross-border"); gp = c2.checkbox("Going private"); sol = c3.checkbox("Solicited approach"); listed = c4.checkbox("Buyer is listed", value=True)
    text = st.text_area("Paste the press release (optional)", height=180)
    row = dict(zip(B.COMPACT, [np.log1p(value), prem, float(cash), float(sol), float(listed), float(fin), float(tender), float(lbo), float(goshop),
                               float(xb), float(gp), sell_fee, buy_fee if buy_fee > 0 else np.nan, rel if rel > 0 else np.nan]))
    kw = [float(bool(re.search(p, text, re.I))) if text.strip() else np.nan for p in F.KEYWORDS.values()]
    A = np.array([[row[c] for c in B.COMPACT] + kw])
    p, contrib = B.compact_predict(art["compact"], A)
    hist = art["history_compact"]
    headline(float(p[0]), float((hist < p[0]).mean()), art["base_rate"], None)
    top = np.argsort(-np.abs(contrib[0]))[:8]
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Top drivers of the score (compact model)**")
        st.pyplot(drivers_chart([(art["compact"]["features"][t], None if not np.isfinite(A[0, t]) else float(A[0, t]), float(contrib[0, t])) for t in top]))
    with right:
        st.markdown("**Terms found in the pasted text (keyword flags, offline)**")
        if text.strip():
            found = [k.replace("kw_", "").replace("_", " ") for k, v in zip(F.KEYWORDS, kw) if v == 1]
            st.write(", ".join(found) if found else "None of the tracked terms found.")
            st.markdown("**Most similar historical broken deals**")
            st.dataframe(similar_deals(art, text), hide_index=True)
        else:
            st.write("Paste a press release to see keyword flags and similar broken deals.")

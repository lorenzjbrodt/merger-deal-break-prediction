#!/usr/bin/env bash
# Rebuild everything from the raw exports.
#   ./run_all.sh              real data (needs the Capital IQ exports in data/raw/, see DATA.md)
#   ./run_all.sh synthetic    smoke test on data/sample/synthetic_ciq_export.csv (no Capital IQ, no EDGAR, no LLM)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  python3.13 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
MODE="${1:-real}"
mkdir -p logs results figures

if [ "$MODE" = "synthetic" ]; then
  $PY src/make_synthetic.py
  $PY src/prepare.py --input data/sample/synthetic_ciq_export.csv --tag synthetic
  $PY src/build_text.py --tag synthetic --embed
  $PY src/train_eval.py --tag synthetic
  $PY src/build_demo_artifacts.py --synthetic
  echo "Synthetic smoke test done: see results_synthetic/ and figures_synthetic/ (not real results)."
  exit 0
fi

# 1. sample and public text
$PY src/prepare.py                                   # data/raw/ciq_transactions_v2.xls -> data/interim/deals.parquet
# Resumable; about 100 minutes on a first run at <= 5 requests per second.
# Set SEC_USER_AGENT="Your Name your-contact-address" first (SEC fair-access policy).
$PY src/edgar_download.py 2>&1 | tee logs/edgar_download.log
$PY src/build_text.py --embed

# 2. labels and extra data (LLM steps need a logged-in `claude` CLI; all are resumable and cached)
$PY src/outcomes.py prepare && $PY src/outcomes.py run-cli --model sonnet && $PY src/outcomes.py finalize
$PY src/market_context.py                            # needs the FRED CSVs in data/external/fred/ (see DATA.md)
$PY src/prices.py && $PY src/price_analysis.py       # data/raw/ciq_prices_{1,2,3}.xls (monthly closes: the fallback without WRDS)
# CRSP / Compustat through WRDS (optional; one-off setup in DATA.md). Skipped when ~/.pgpass has no WRDS line and nothing is cached.
WRDS=0
if [ -d data/wrds ] || grep -q "wrds-pgdata" "$HOME/.pgpass" 2>/dev/null; then
  WRDS=1
  $PY src/wrds_pull.py lookups && $PY src/crsp_link.py && $PY src/wrds_pull.py daily && $PY src/crsp_prices.py
  [ -f data/external/hoberg_phillips/extracted/tnic3_data.txt ] && $PY src/overlap.py build
fi
$PY src/llm_extract.py prepare --run haiku --batch-size 1 && $PY src/llm_extract.py run-cli --run haiku --model haiku
$PY src/llm_extract.py prepare --run strong --sample 60 --seed 15071 --batch-size 1 && $PY src/llm_extract.py run-cli --run strong --model fable
$PY src/llm_extract.py validate --run haiku --reference strong && $PY src/llm_extract.py features --run haiku
$PY src/finetune_encoder.py --target label_adverse && $PY src/finetune_encoder.py --target label

# 3. models, evaluation, impact
$PY src/train_eval.py 2>&1 | tee logs/train_eval.log
$PY src/backtest.py > logs/backtest.log               # monthly closes -> results/monthly/
$PY src/impact.py > logs/impact.log
$PY src/pooled_oot.py fit 2>&1 | tee logs/pooled_oot_fit.log && $PY src/pooled_oot.py report > logs/pooled_oot.log   # 13 rolling origins, about an hour
# time to closing (needs no prices); the stated expected closing is read from the press releases first
$PY src/llm_closing.py prepare && $PY src/llm_closing.py run --run sonnet --model sonnet
$PY src/llm_closing.py run --run strong --model fable --sample 60 --batch-size 5 && $PY src/llm_closing.py finalize
$PY src/duration.py fit 2>&1 | tee logs/duration_fit.log && $PY src/duration.py product && $PY src/duration.py report > logs/duration.log
if [ "$WRDS" = 1 ]; then
  $PY src/daily_eval.py > logs/daily_eval.log         # model versus market, backtest with daily marks, day-1 ladder step
  $PY src/loss_model.py > logs/loss_model.log         # loss if the deal breaks
  [ -f data/interim/overlap_features.parquet ] && $PY src/overlap.py eval > logs/overlap.log
  $PY src/day1_signals.py build && $PY src/day1_signals.py run > logs/day1_signals.log   # pre-specified; decides the break probability used next
  $PY src/expected_return.py > logs/expected_return.log
fi
$PY src/label_audit.py > logs/label_audit.log
$PY src/llm_zero_shot.py run && $PY src/llm_zero_shot.py report     # exploratory
$PY src/case_studies.py > logs/case_studies.log
$PY src/build_demo_artifacts.py
echo "Done: see results/ and figures/. Demo: .venv/bin/streamlit run app/demo.py"

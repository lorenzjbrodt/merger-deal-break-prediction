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
$PY src/prices.py && $PY src/price_analysis.py       # data/raw/ciq_prices_{1,2,3}.xls
$PY src/llm_extract.py prepare --run haiku --batch-size 1 && $PY src/llm_extract.py run-cli --run haiku --model haiku
$PY src/llm_extract.py prepare --run strong --sample 60 --seed 15071 --batch-size 1 && $PY src/llm_extract.py run-cli --run strong --model fable
$PY src/llm_extract.py validate --run haiku --reference strong && $PY src/llm_extract.py features --run haiku
$PY src/finetune_encoder.py --target label_adverse && $PY src/finetune_encoder.py --target label

# 3. models, evaluation, impact
$PY src/train_eval.py 2>&1 | tee logs/train_eval.log
$PY src/backtest.py > logs/backtest.log
$PY src/impact.py > logs/impact.log
$PY src/label_audit.py > logs/label_audit.log
$PY src/llm_zero_shot.py run && $PY src/llm_zero_shot.py report     # exploratory
$PY src/case_studies.py > logs/case_studies.log
$PY src/build_demo_artifacts.py
echo "Done: see results/ and figures/. Demo: .venv/bin/streamlit run app/demo.py"

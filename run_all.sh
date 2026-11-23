#!/usr/bin/env bash
# Rebuild everything from the raw export.
#   ./run_all.sh              real data (needs data/raw/ciq_transactions.xls, see DATA.md)
#   ./run_all.sh synthetic    smoke test on data/sample/synthetic_ciq_export.csv (no Capital IQ, no EDGAR)
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
  $PY src/impact.py --tag synthetic
  echo "Synthetic smoke test done: see results_synthetic/ and figures_synthetic/ (not real results)."
  exit 0
fi

$PY src/prepare.py
# Resumable; about 75 minutes on a first run at <= 5 requests per second.
# Set SEC_USER_AGENT="Your Name your-contact-address" first (SEC fair-access policy).
$PY src/edgar_download.py 2>&1 | tee logs/edgar_download.log
$PY src/build_text.py --embed
$PY src/train_eval.py 2>&1 | tee logs/train_eval.log
$PY src/impact.py > logs/impact.log
echo "Done: see results/ and figures/."

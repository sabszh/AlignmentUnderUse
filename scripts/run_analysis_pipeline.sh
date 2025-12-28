#!/usr/bin/env bash
set -euo pipefail
set -o igncr 2>/dev/null || true

INPUT_PATH="${1:-data/processed/conversations_english.jsonl}"
DERIVED_DIR="${2:-data/derived}"
OUTPUTS_DIR="${3:-data/outputs}"
SKIP_TOPICS="${4:-false}"
VERBOSE="${5:-true}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[run_analysis_pipeline] Input file not found: $INPUT_PATH" >&2
  exit 1
fi

mkdir -p "$DERIVED_DIR"
mkdir -p "$OUTPUTS_DIR"

SEMANTIC_OUT="$DERIVED_DIR/semantic_alignment.csv"
SENTIMENT_OUT="$DERIVED_DIR/sentiment_alignment.csv"
LSM_OUT="$DERIVED_DIR/lsm_scores.csv"
TOPICS_OUT="$OUTPUTS_DIR/topics"
ARCHIVE_PATH="$OUTPUTS_DIR/analysis_outputs.zip"

echo "[run_analysis_pipeline] Input: $INPUT_PATH"
echo "[run_analysis_pipeline] Derived dir: $DERIVED_DIR"
echo "[run_analysis_pipeline] Outputs dir: $OUTPUTS_DIR"
echo "[run_analysis_pipeline] Skip topics: $SKIP_TOPICS"
echo "[run_analysis_pipeline] Started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ "$VERBOSE" == "true" ]]; then
  echo
  echo "[run_analysis_pipeline] Python: $(python --version 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run_analysis_pipeline] GPU info:"
    nvidia-smi || true
  else
    echo "[run_analysis_pipeline] GPU info: nvidia-smi not found"
  fi
fi

echo
if [[ -f "$SEMANTIC_OUT" ]]; then
  echo "[run_analysis_pipeline] Semantic alignment: skip (exists)"
else
  echo "[run_analysis_pipeline] Semantic alignment..."
  python -u -m src.analysis.semantic_alignment \
    --input "$INPUT_PATH" \
    --output "$SEMANTIC_OUT" \
    --embeddings-cache-dir "$DERIVED_DIR" \
    --device auto
fi

echo
if [[ -f "$SENTIMENT_OUT" ]]; then
  echo "[run_analysis_pipeline] Sentiment alignment: skip (exists)"
else
  echo "[run_analysis_pipeline] Sentiment alignment..."
  python -u -m src.analysis.sentiment_alignment \
    --input "$SEMANTIC_OUT" \
    --output "$SENTIMENT_OUT" \
    --cache-dir "$DERIVED_DIR" \
    --conversations "$INPUT_PATH" \
    --device auto
fi

echo
if [[ -f "$LSM_OUT" ]]; then
  echo "[run_analysis_pipeline] LSM scoring: skip (exists)"
else
  echo "[run_analysis_pipeline] LSM scoring..."
  python -u -m src.analysis.lsm_scoring \
    --input "$INPUT_PATH" \
    --output "$LSM_OUT"
fi

if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo
  if [[ -f "$TOPICS_OUT/conversations_with_topics.csv" ]]; then
    echo "[run_analysis_pipeline] Topic modeling: skip (exists)"
  else
    echo "[run_analysis_pipeline] Topic modeling..."
    if python - <<'PY'
from src.analysis.topic_modeling import KeyNMF
raise SystemExit(0 if KeyNMF is not None else 1)
PY
    then
      python -u -m src.analysis.topic_modeling \
        --input "$INPUT_PATH" \
        --output-dir "$TOPICS_OUT"
    else
      echo "[run_analysis_pipeline] Topic modeling: skip (turftopic import failed)"
    fi
  fi
fi

echo
echo "[run_analysis_pipeline] Zipping outputs..."
zip -r "$ARCHIVE_PATH" "$DERIVED_DIR" "$OUTPUTS_DIR" >/dev/null

echo
echo "[run_analysis_pipeline] Finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[run_analysis_pipeline] Done."
echo "  $SEMANTIC_OUT"
echo "  $SENTIMENT_OUT"
echo "  $LSM_OUT"
if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo "  $TOPICS_OUT"
fi
echo "  $ARCHIVE_PATH"

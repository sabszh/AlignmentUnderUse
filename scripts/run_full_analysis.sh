#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-data/processed/conversations_english.jsonl}"
DERIVED_DIR="${2:-data/derived}"
OUTPUTS_DIR="${3:-data/outputs}"
SKIP_TOPICS="${4:-false}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[run_full_analysis] Input file not found: $INPUT_PATH" >&2
  exit 1
fi

mkdir -p "$DERIVED_DIR"
mkdir -p "$OUTPUTS_DIR"

SEMANTIC_OUT="$DERIVED_DIR/semantic_alignment.csv"
SENTIMENT_OUT="$DERIVED_DIR/sentiment_alignment.csv"
LSM_OUT="$DERIVED_DIR/lsm_scores.csv"
TOPICS_OUT="$OUTPUTS_DIR/topics"
ARCHIVE_PATH="$OUTPUTS_DIR/analysis_outputs.zip"

echo "[run_full_analysis] Input: $INPUT_PATH"
echo "[run_full_analysis] Derived dir: $DERIVED_DIR"
echo "[run_full_analysis] Outputs dir: $OUTPUTS_DIR"

echo
echo "[run_full_analysis] Semantic alignment..."
python -m analysis.semantic_alignment \
  --input "$INPUT_PATH" \
  --output "$SEMANTIC_OUT" \
  --embeddings-cache-dir "$DERIVED_DIR"

echo
echo "[run_full_analysis] Sentiment alignment..."
python -m analysis.sentiment_alignment \
  --input "$SEMANTIC_OUT" \
  --output "$SENTIMENT_OUT" \
  --cache-dir "$DERIVED_DIR" \
  --conversations "$INPUT_PATH"

echo
echo "[run_full_analysis] LSM scoring..."
python -m analysis.lsm_scoring \
  --input "$INPUT_PATH" \
  --output "$LSM_OUT"

if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo
  echo "[run_full_analysis] Topic modeling..."
  python -m analysis.topic_modeling \
    --input "$INPUT_PATH" \
    --output-dir "$TOPICS_OUT"
fi

echo
echo "[run_full_analysis] Zipping outputs..."
zip -r "$ARCHIVE_PATH" "$DERIVED_DIR" "$OUTPUTS_DIR" >/dev/null

echo
echo "[run_full_analysis] Done."
echo "  $SEMANTIC_OUT"
echo "  $SENTIMENT_OUT"
echo "  $LSM_OUT"
if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo "  $TOPICS_OUT"
fi
echo "  $ARCHIVE_PATH"

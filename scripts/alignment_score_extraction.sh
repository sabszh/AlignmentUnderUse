#!/usr/bin/env bash
set -euo pipefail
set -o igncr 2>/dev/null || true

INPUT_PATH="${1:-data/processed/conversations_english.jsonl}"
DERIVED_DIR="${2:-data/derived}"
OUTPUTS_DIR="${3:-data/outputs}"
SKIP_TOPICS="${4:-false}"
VERBOSE="${5:-true}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[alignment_score_extraction] Input file not found: $INPUT_PATH" >&2
  exit 1
fi

mkdir -p "$DERIVED_DIR"
mkdir -p "$OUTPUTS_DIR"

SEMANTIC_OUT="$DERIVED_DIR/semantic_alignment.csv"
SENTIMENT_OUT="$DERIVED_DIR/sentiment_alignment.csv"
LSM_OUT="$DERIVED_DIR/lsm_scores.csv"
TOPICS_OUT="$OUTPUTS_DIR/topics"
ARCHIVE_PATH="$OUTPUTS_DIR/analysis_outputs.zip"

echo "[alignment_score_extraction] Input: $INPUT_PATH"
echo "[alignment_score_extraction] Derived dir: $DERIVED_DIR"
echo "[alignment_score_extraction] Outputs dir: $OUTPUTS_DIR"
echo "[alignment_score_extraction] Skip topics: $SKIP_TOPICS"
echo "[alignment_score_extraction] Started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ "$VERBOSE" == "true" ]]; then
  echo
  echo "[alignment_score_extraction] Python: $(python --version 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[alignment_score_extraction] GPU info:"
    nvidia-smi || true
  else
    echo "[alignment_score_extraction] GPU info: nvidia-smi not found"
  fi
fi

echo
if [[ -f "$SEMANTIC_OUT" ]]; then
  echo "[alignment_score_extraction] Semantic alignment: skip (exists)"
else
  echo "[alignment_score_extraction] Semantic alignment..."
  python -u -m src.measures.semantic_alignment \
    --input "$INPUT_PATH" \
    --output "$SEMANTIC_OUT" \
    --embeddings-cache-dir "$DERIVED_DIR" \
    --device auto
fi

echo
if [[ -f "$SENTIMENT_OUT" ]]; then
  echo "[alignment_score_extraction] Sentiment alignment: skip (exists)"
else
  echo "[alignment_score_extraction] Sentiment alignment..."
  python -u -m src.measures.sentiment_alignment \
    --input "$SEMANTIC_OUT" \
    --output "$SENTIMENT_OUT" \
    --cache-dir "$DERIVED_DIR" \
    --conversations "$INPUT_PATH" \
    --device auto
fi

echo
if [[ -f "$LSM_OUT" ]]; then
  echo "[alignment_score_extraction] LSM scoring: skip (exists)"
else
  echo "[alignment_score_extraction] LSM scoring..."
  python -u -m src.measures.lsm_scoring \
    --input "$INPUT_PATH" \
    --output "$LSM_OUT"
fi

if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo
  if [[ -f "$TOPICS_OUT/conversations_with_topics.csv" ]]; then
    echo "[alignment_score_extraction] Topic modeling: skip (exists)"
  else
    echo "[alignment_score_extraction] Topic modeling..."
    if python - <<'PY'
from src.measures.topic_modeling import KeyNMF
raise SystemExit(0 if KeyNMF is not None else 1)
PY
    then
      python -u -m src.measures.topic_modeling \
        --input "$INPUT_PATH" \
        --output-dir "$TOPICS_OUT"
    else
      echo "[alignment_score_extraction] Topic modeling: skip (turftopic import failed)"
    fi
  fi
fi

echo
echo "[alignment_score_extraction] Zipping outputs..."
zip -r "$ARCHIVE_PATH" "$DERIVED_DIR" "$OUTPUTS_DIR" >/dev/null

echo
echo "[alignment_score_extraction] Finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[alignment_score_extraction] Done."
echo "  $SEMANTIC_OUT"
echo "  $SENTIMENT_OUT"
echo "  $LSM_OUT"
if [[ "$SKIP_TOPICS" != "true" ]]; then
  echo "  $TOPICS_OUT"
fi
echo "  $ARCHIVE_PATH"

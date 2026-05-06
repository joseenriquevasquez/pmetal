#!/usr/bin/env bash
# Inference Example
# Run text generation with a base model

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PROMPT="${PROMPT:-Explain machine learning in one concise paragraph.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

echo "=== PMetal Inference ==="
echo "Model: $MODEL"
echo ""

"$PMETAL_BIN" infer \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --chat \
    --no-thinking \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P"

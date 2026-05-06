#!/usr/bin/env bash
# End-to-end workload benchmark example.

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-TeichAI/gemini-3-pro-preview-high-reasoning-250x}"
OUTPUT="${OUTPUT:-./output/bench/workload.json}"
PRESET="${PRESET:-dense-qwen3}"
PROMPT_SAMPLES="${PROMPT_SAMPLES:-16}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-16}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== PMetal Workload Benchmark ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Preset: $PRESET"
echo "Output: $OUTPUT"
echo ""

"$PMETAL_BIN" bench-workload \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --preset "$PRESET" \
    --prompt-samples "$PROMPT_SAMPLES" \
    --train-samples "$TRAIN_SAMPLES" \
    --json \
    --output "$OUTPUT"

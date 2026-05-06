#!/usr/bin/env bash
# GGUF quantization example.

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
OUTPUT="${OUTPUT:-./output/quantized/qwen3-0.6b-q4_k_m.gguf}"
METHOD="${METHOD:-q4_k_m}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== PMetal GGUF Quantization ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "Method: $METHOD"
echo ""

"$PMETAL_BIN" quantize \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --format gguf \
    --method "$METHOD"

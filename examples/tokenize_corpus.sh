#!/usr/bin/env bash
# Tokenize a JSONL text corpus into PMetal shards.

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
INPUT="${INPUT:-./examples/sample_corpus.jsonl}"
OUTPUT="${OUTPUT:-./output/tokenized}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-0.6B}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
DOCS_PER_SHARD="${DOCS_PER_SHARD:-1000}"

echo "=== PMetal Tokenization ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Tokenizer: $TOKENIZER"
echo ""

"$PMETAL_BIN" tokenize \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --tokenizer "$TOKENIZER" \
    --text-column "$TEXT_COLUMN" \
    --docs-per-shard "$DOCS_PER_SHARD"

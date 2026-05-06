# PMetal Examples

This directory contains runnable shell examples and small sample datasets for common PMetal workflows. All scripts use `./target/release/pmetal` by default; override `PMETAL_BIN` to point at another binary.

## Examples

### Training

- `lora_finetune.sh` - Basic LoRA fine-tuning workflow
- `qlora_finetune.sh` - QLoRA fine-tuning with 4-bit quantization

### Inference

- `inference.sh` - Text generation with base model
- `inference_lora.sh` - Text generation with LoRA adapter
- `serve_openai.sh` - OpenAI-compatible local server with continuous batching

Build the server example with `cargo build -p pmetal --release --features serve`; the default binary does not include the `serve` subcommand.

### Quantization and Benchmarking

- `quantize_gguf.sh` - GGUF quantization with a configurable method
- `bench_workload.sh` - End-to-end inference and LoRA workload benchmark

### Data Preparation

- `sample_dataset.jsonl` - Example training data format
- `sample_corpus.jsonl` - Example text corpus for tokenizer sharding
- `tokenize_corpus.sh` - Tokenize a JSONL corpus into PMetal shards

## Quick Start

```bash
# 1. Build PMetal
cargo build --release

# 2. Run inference
./examples/inference.sh

# 3. Fine-tune a model
./examples/lora_finetune.sh

# 4. Run inference with the adapter
./examples/inference_lora.sh
```

## Dataset Formats

PMetal supports multiple dataset formats. See `sample_dataset.jsonl` for examples of:

- **ShareGPT**: Multi-turn conversations
- **Alpaca**: Instruction/input/output format
- **Messages**: OpenAI-style chat format

## Configuration

Most examples can be customized with environment variables:

| Parameter | Description |
|-----------|-------------|
| `PMETAL_BIN` | Path to the `pmetal` binary |
| `MODEL` | HuggingFace model ID or local path |
| `OUTPUT` | Output file or directory |
| `PROMPT` | Inference prompt |
| `MAX_TOKENS` | Generated token budget |
| `HOST` | Server bind host |
| `PORT` | Server port |
| `DATASET` | Training JSONL file or workload dataset |
| `TOKENIZER` | Tokenizer model ID or local path |
| `METHOD` | GGUF quantization method |
| `PRESET` | Workload benchmark preset |

Example:

```bash
MODEL=Qwen/Qwen3-4B MAX_TOKENS=128 ./examples/inference.sh
```

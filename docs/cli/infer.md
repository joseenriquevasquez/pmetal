# pmetal infer

Run interactive inference with chat, tool use, thinking mode, and LoRA adapters.

Run inference on a loaded model. Supports interactive chat, tool/function calling, thinking mode, FP8 quantization, and LoRA adapter loading.

## Usage

```bash
pmetal infer \
  --model <MODEL> \
  [--prompt <PROMPT>] \
  [OPTIONS]
```

## Examples

```bash
# Simple generation
pmetal infer --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"

# Interactive chat with LoRA
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --chat --show-thinking

# FP8 quantized inference (2× memory reduction)
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat

# With tool definitions
pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --tools tools.json --chat

# ANE-optimized inference
pmetal infer --model Qwen/Qwen3-0.6B --ane-max-seq-len 2048

# JIT-compiled sampling
pmetal infer --model Qwen/Qwen3-0.6B --compiled --chat
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--prompt` | — | Input prompt (omit for stdin) |
| `--lora` | — | Path to LoRA adapter weights |
| `--temperature` | model default | Sampling temperature |
| `--top-k` | model default | Top-k sampling |
| `--top-p` | model default | Nucleus sampling |
| `--min-p` | model default | Min-p dynamic sampling |
| `--max-tokens` | `256` | Maximum generation length |
| `--repetition-penalty` | `1.0` | Repetition penalty |
| `--frequency-penalty` | `0.0` | Frequency penalty |
| `--presence-penalty` | `0.0` | Presence penalty |
| `--chat` | `false` | Apply chat template |
| `--show-thinking` | `false` | Show reasoning content |
| `--fp8` | `false` | FP8 weights (~2× mem reduction) |
| `--compiled` | `false` | JIT-compiled sampling |
| `--no-ane` | `false` | Disable ANE inference |
| `--ane-max-seq-len` | `1024` | Max ANE kernel sequence length |
| `--tools` | — | Tool definitions file (OpenAI format) |
| `--system` | — | System message |

## Chat Mode

With `--chat`, PMetal applies the model's chat template and starts an interactive session:

```
> What is quantum entanglement?
Quantum entanglement is a phenomenon where two particles...

> Can you explain it more simply?
Think of it like two coins that always land on opposite sides...
```

## Tool Use

Pass OpenAI-format tool definitions with `--tools`:

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  }
]
```

Supported for Qwen, Llama 3.1+, Mistral v3+, and DeepSeek models.

## See Also

- [pmetal serve](/cli/serve/) — OpenAI-compatible inference server
- [Rust SDK](/sdk/easy-api/) — Programmatic inference
- [Python SDK](/python/quick-start/) — Python inference

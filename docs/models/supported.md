# Supported Models

All model architectures supported for inference and LoRA training in PMetal.

PMetal supports a wide range of model architectures. Models are loaded from HuggingFace Hub or local safetensors with automatic architecture detection.

## Inference Support

All models below work with the CLI (`pmetal infer`), TUI, GUI, and SDK.

| Family | Architecture | Variants | `model_type` values |
|--------|-------------|----------|-------------------|
| Llama | `Llama` | 2, 3, 3.1, 3.2, 3.3 | `llama`, `llama3` |
| Llama 4 | `Llama4` | Scout, Maverick | `llama4` |
| Qwen 2 | `Qwen2` | 2, 2.5 | `qwen2`, `qwen2_5` |
| Qwen 3 | `Qwen3` | 3 | `qwen3` |
| Qwen 3 MoE | `Qwen3MoE` | 3-MoE | `qwen3_moe` |
| Qwen 3.5 | `Qwen3Next` | 3.5 (Next) | `qwen3_next`, `qwen3_5` |
| DeepSeek | `DeepSeek` | V3, V3.2, V3.2-Speciale | `deepseek`, `deepseek_v3` |
| Mistral | `Mistral` | 7B, Mixtral 8×7B | `mistral`, `mixtral` |
| Gemma | `Gemma` | 2, 3 | `gemma`, `gemma2`, `gemma3` |
| Phi 3 | `Phi` | 3, 3.5 | `phi`, `phi3` |
| Phi 4 | `Phi4` | 4 | `phi4` |
| Cohere | `Cohere` | Command R | `cohere`, `command_r` |
| Granite | `Granite` | 3.0, 3.1, Hybrid MoE | `granite`, `granitehybrid` |
| NemotronH | `NemotronH` | Hybrid (Mamba+Attention) | `nemotron_h` |
| StarCoder2 | `StarCoder2` | 3B, 7B, 15B | `starcoder2` |
| RecurrentGemma | `RecurrentGemma` | Griffin | `recurrentgemma`, `griffin` |
| Jamba | `Jamba` | 1.5 | `jamba` |
| Flux | `Flux` | 1-dev, 1-schnell | `flux` |

## LoRA / QLoRA Training Support

| Architecture | LoRA | QLoRA | Notes |
|-------------|------|-------|-------|
| Llama | Yes | Yes | Covers Llama 2–3.3. Gradient checkpointing supported. |
| Qwen 2 | Yes | — | Uses Qwen3 LoRA implementation internally. |
| Qwen 3 | Yes | Yes | Gradient checkpointing supported. |
| Qwen 3.5 (Next) | Yes | — | Hybrid architecture with nested `text_config`. |
| Gemma | Yes | Yes | GeGLU activation, special RMSNorm. |
| Mistral | Yes | Yes | Sliding window attention support. |
| Phi 3 | Yes | — | Partial RoPE, fused gate_up projection. |

Architectures not listed (Llama 4, Qwen 3 MoE, DeepSeek, Cohere, Granite, NemotronH, Phi 4, StarCoder2, RecurrentGemma, Jamba) support inference only.

## Architecture Modules (Not Yet in Dispatcher)

These have implementations in `pmetal-models` but are not in the `DynamicModel` dispatcher:

| Family | Module | Notes |
|--------|--------|-------|
| GPT-OSS | `gpt_oss` | MoE with Top-4 sigmoid routing, 20B/120B |
| Pixtral | `pixtral` | 12B vision-language |
| Qwen2-VL | `qwen2_vl` | 2B, 7B vision-language |
| MLlama | `mllama` | Llama 3.2-Vision |
| CLIP | `clip` | ViT-L/14 vision encoder |
| Whisper | `whisper` | Base–Large speech models |
| T5 | `t5` | Encoder-decoder architecture |

These can be used directly via their Rust types (e.g., `pmetal_models::architectures::gpt_oss::GptOssForCausalLM`).

## See Also

- [Model Merging](/models/merging/) — Merge strategies
- [Quantization](/models/quantization/) — GGUF quantization

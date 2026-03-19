# Feature Flags

Cargo feature flags for enabling optional PMetal functionality.

PMetal uses Cargo feature flags to control which crates and capabilities are compiled.

## Feature Matrix

| Feature | Default | Crate | Description |
|---------|---------|-------|-------------|
| `core` | Yes | `pmetal-core` | Foundation types, configs, traits |
| `gguf` | Yes | `pmetal-gguf` | GGUF format support |
| `metal` | Yes | `pmetal-metal` | Metal GPU kernels |
| `hub` | Yes | `pmetal-hub` | HuggingFace Hub integration |
| `mlx` | Yes | `pmetal-mlx` | MLX backend |
| `models` | Yes | `pmetal-models` | LLM architectures |
| `lora` | Yes | `pmetal-lora` | LoRA/QLoRA |
| `trainer` | Yes | `pmetal-trainer` | Training loops (pulls in `data`, `distill`) |
| `easy` | Yes | — | High-level builders (pulls in `trainer`, `hub`, `data`) |
| `ane` | Yes | — | Apple Neural Engine |
| `data` | Yes* | `pmetal-data` | Dataset loading (*via `easy`) |
| `distill` | Yes* | `pmetal-distill` | Knowledge distillation (*via `trainer`) |
| `lora-metal-fused` | **No** | — | ~2× LoRA training speedup via fused Metal kernels |
| `merge` | **No** | `pmetal-merge` | Model merging strategies |
| `vocoder` | **No** | `pmetal-vocoder` | BigVGAN neural vocoder |
| `distributed` | **No** | `pmetal-distributed` | Distributed training |
| `mhc` | **No** | `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `serve` | **No** | `pmetal-serve` | OpenAI-compatible inference server |
| `full` | **No** | — | All features |

## Usage

```bash
# Default features
cargo install pmetal

# With specific features
cargo install pmetal --features "merge,serve"

# All features
cargo install pmetal --features full

# Minimal build (no ANE)
cargo build --release --no-default-features --features dashboard
```

## As a Library Dependency

```toml
[dependencies]
pmetal = { version = "0.1", features = ["easy"] }

# Or specific crates
pmetal-models = "0.1"
pmetal-trainer = "0.1"
pmetal-lora = "0.1"
```

## See Also

- [Installation](/installation/) — Build options
- [Advanced SDK Usage](/sdk/advanced/) — Crate-level API

# Advanced SDK Usage

Lower-level crate APIs for full control over training loops, models, and pipelines.

For full control, use the underlying crates directly instead of the `easy` module.

## Architecture

PMetal is a Rust workspace with 18 specialized crates:

| Crate | Purpose |
|-------|---------|
| `pmetal-core` | Foundation: configs, traits, types, error handling |
| `pmetal-metal` | Custom Metal GPU kernels + ANE runtime |
| `pmetal-mlx` | MLX backend integration |
| `pmetal-models` | LLM architectures (Llama, Qwen, DeepSeek, etc.) |
| `pmetal-lora` | LoRA/QLoRA training implementations |
| `pmetal-trainer` | Training loops (SFT, DPO, SimPO, ORPO, KTO, GRPO) |
| `pmetal-data` | Dataset loading, chat templates, tokenization |
| `pmetal-hub` | HuggingFace Hub integration + model fit estimation |
| `pmetal-distill` | Knowledge distillation (online, offline, TAID) |
| `pmetal-merge` | Model merging (14 strategies) |
| `pmetal-gguf` | GGUF format with imatrix quantization |
| `pmetal-mhc` | Manifold-Constrained Hyper-Connections |
| `pmetal-distributed` | Distributed training (mDNS, Ring All-Reduce) |
| `pmetal-vocoder` | BigVGAN neural vocoder |
| `pmetal-serve` | OpenAI-compatible inference server |
| `pmetal-py` | Python bindings (maturin/PyO3) |

## Manual Training Loop

```rust
use pmetal_trainer::TrainingLoop;
use pmetal_models::DynamicModel;
use pmetal_lora::DynamicLoraModel;
use pmetal_data::DataLoader;

// Load model with LoRA
let model = DynamicLoraModel::load("Qwen/Qwen3-0.6B", lora_config)?;

// Create data loader
let loader = DataLoader::from_file("train.jsonl")?;

// Run training loop
let mut loop = TrainingLoop::new(model, loader, training_config);
loop.add_callback(MyCustomCallback);
let result = loop.run().await?;
```

## Model Loading

```rust
use pmetal_models::DynamicModel;

// Load from HuggingFace
let model = DynamicModel::load("Qwen/Qwen3-0.6B").await?;

// Load from local path
let model = DynamicModel::load("./my-model/").await?;
```

## Callback System

The `TrainingCallback` trait provides lifecycle hooks:

```rust
use pmetal_trainer::TrainingCallback;

struct MyCallback;

impl TrainingCallback for MyCallback {
    fn on_step_start(&mut self, step: usize) { /* ... */ }
    fn on_step_end(&mut self, step: usize, loss: f32) { /* ... */ }
    fn should_stop(&self) -> bool { false }
}
```

## See Also

- [Easy API](/sdk/easy-api/) — High-level builders
- [`examples/`](https://github.com/Epistates/pmetal/tree/main/crates/pmetal/examples/) — Complete working examples

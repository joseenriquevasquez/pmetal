# Easy API

High-level builder API for training and inference — get started in a few lines of Rust.

The `easy` module provides high-level builders that wrap PMetal's full pipeline into ergonomic one-liner APIs.

## Fine-Tuning

```rust
use pmetal::easy;

let result = easy::finetune("Qwen/Qwen3-0.6B", "train.jsonl")
    .lora(16, 32.0)           // rank, alpha
    .learning_rate(2e-4)
    .epochs(3)
    .batch_size(4)
    .output("./output")
    .run()
    .await?;

println!("Final loss: {}", result.final_loss);
println!("Total steps: {}", result.total_steps);
```

## Preference Optimization

```rust
// DPO
let result = easy::dpo("Qwen/Qwen3-0.6B", "preferences.jsonl")
    .dpo_beta(0.1)
    .reference_model("Qwen/Qwen3-0.6B")
    .run()
    .await?;

// SimPO
let result = easy::simpo("Qwen/Qwen3-0.6B", "preferences.jsonl")
    .run()
    .await?;

// ORPO
let result = easy::orpo("Qwen/Qwen3-0.6B", "preferences.jsonl")
    .run()
    .await?;

// KTO
let result = easy::kto("Qwen/Qwen3-0.6B", "feedback.jsonl")
    .run()
    .await?;
```

## Inference

```rust
// Single generation
let output = easy::infer("Qwen/Qwen3-0.6B")
    .temperature(0.7)
    .max_tokens(256)
    .lora("./output/lora_weights.safetensors")
    .generate("What is 2+2?")
    .await?;

println!("{output}");
```

## Streaming Inference

```rust
easy::infer("Qwen/Qwen3-0.6B")
    .temperature(0.7)
    .generate_streaming("Tell me a story", |delta| {
        print!("{delta}");
        true // return false to stop early
    })
    .await?;
```

## Available Builders

| Builder | Description |
|---------|-------------|
| `easy::finetune()` | SFT with LoRA/QLoRA/DoRA |
| `easy::dpo()` | Direct Preference Optimization |
| `easy::simpo()` | Simple Preference Optimization |
| `easy::orpo()` | Odds-Ratio Preference Optimization |
| `easy::kto()` | Kahneman-Tversky Optimization |
| `easy::infer()` | Inference with optional LoRA |

## See Also

- [Advanced SDK Usage](/sdk/advanced/) — Lower-level crate APIs
- [Python SDK](/python/quick-start/) — Python equivalent

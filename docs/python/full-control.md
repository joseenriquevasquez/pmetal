# Full Control

Advanced Python SDK usage — custom training loops, callbacks, and model operations.

For full control over training and inference, use the lower-level Python classes.

## Custom Training

```python
import pmetal

# Configure training components
lora_config = pmetal.LoraConfig(r=16, alpha=32.0)
training_config = pmetal.TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    max_seq_len=2048,
)

# Create trainer
trainer = pmetal.Trainer(
    model_id="Qwen/Qwen3-0.6B",
    lora_config=lora_config,
    training_config=training_config,
    dataset_path="train.jsonl",
)

# Add callbacks
trainer.add_callback(pmetal.ProgressCallback())
trainer.add_callback(pmetal.LoggingCallback())

# Train
result = trainer.train()
```

## Model Loading

```python
# Load model
model = pmetal.Model.load("Qwen/Qwen3-0.6B")

# Generate
output = model.generate("Hello world", temperature=0.7)
print(output)
```

## Configuration Classes

| Class | Description |
|-------|-------------|
| `pmetal.LoraConfig` | LoRA rank, alpha, target modules |
| `pmetal.TrainingConfig` | Learning rate, epochs, batch size, scheduler |
| `pmetal.GenerationConfig` | Temperature, top-k, top-p, max tokens |
| `pmetal.DataLoaderConfig` | Dataset format, sequence packing |

## Callbacks

| Callback | Description |
|----------|-------------|
| `pmetal.ProgressCallback()` | Progress bar display |
| `pmetal.LoggingCallback()` | Console logging |
| `pmetal.MetricsJsonCallback(path)` | JSONL metrics file |
| Custom | Subclass with `on_step_end(step, loss)` |

## Hub Operations

```python
# Download a model
pmetal.download_model("Qwen/Qwen3-0.6B")

# Download a specific file
pmetal.download_file("Qwen/Qwen3-0.6B", "config.json")
```

## Tokenizer

```python
tokenizer = pmetal.Tokenizer("Qwen/Qwen3-0.6B")
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

## See Also

- [Quick Start](/python/quick-start/) — Simplified API
- [Rust SDK](/sdk/advanced/) — Rust equivalent

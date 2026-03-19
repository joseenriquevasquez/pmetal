# Configuration File

YAML configuration file format for PMetal training runs.

PMetal supports YAML configuration files as an alternative to CLI flags. Generate a sample config with `pmetal init`.

## Example

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  dtype: bfloat16
  device: gpu
  max_seq_len: 2048

lora:
  r: 16
  alpha: 32.0

training:
  learning_rate: 2e-4
  batch_size: 1
  epochs: 3
  lr_schedule: cosine
  warmup_steps: 100
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  weight_decay: 0.01

dataset:
  path: "./train.jsonl"
  format: jsonl
```

## Usage

```bash
pmetal train --config training.yaml
```

CLI flags override config file values when both are specified.

## Sections

### model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *required* | HuggingFace ID or local path |
| `dtype` | string | `bfloat16` | Data type: `float32`, `float16`, `bfloat16` |
| `device` | string | `gpu` | Device: `gpu`, `cpu` |
| `max_seq_len` | integer | `0` | Max sequence length (0 = auto) |

### lora

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | integer | `16` | LoRA rank |
| `alpha` | float | `32.0` | LoRA scaling factor |

### training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | `2e-4` | Learning rate |
| `batch_size` | integer | `1` | Micro-batch size |
| `epochs` | integer | `1` | Training epochs |
| `lr_schedule` | string | `cosine` | LR schedule type |
| `warmup_steps` | integer | `100` | Warmup steps |
| `gradient_accumulation_steps` | integer | `4` | Gradient accumulation |
| `max_grad_norm` | float | `1.0` | Gradient clipping |
| `weight_decay` | float | `0.01` | AdamW weight decay |

### dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | *required* | Dataset file path |
| `format` | string | auto | Format: `jsonl`, `json`, `parquet`, `csv` |

## See Also

- [pmetal init](/cli/init/) — Generate a sample config
- [pmetal train](/cli/train/) — CLI training parameters

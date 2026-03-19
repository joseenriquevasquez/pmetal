# pmetal fuse

Fuse LoRA adapter weights into the base model.

Merge LoRA adapter weights into the base model weights, producing a standalone model without adapter overhead.

## Usage

```bash
pmetal fuse \
  --model <BASE_MODEL> \
  --lora <LORA_WEIGHTS> \
  [OPTIONS]
```

## Examples

```bash
# Fuse LoRA weights
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors

# Accurate mode (higher precision)
pmetal fuse \
  --model Qwen/Qwen3-0.6B \
  --lora ./output/lora_weights.safetensors \
  --accurate
```

## Modes

| Mode | Description |
|------|-------------|
| Standard | Fast fusing with standard precision |
| Accurate | Higher precision fusing for sensitive weights |

## See Also

- [pmetal train](/cli/train/) — Generate LoRA weights
- [pmetal quantize](/cli/quantize/) — Quantize the fused model

# pmetal quantize

Quantize models to GGUF format with 13 quantization options.

Quantize a model to GGUF format for efficient inference. Supports importance matrix for quality-preserving quantization.

## Usage

```bash
pmetal quantize \
  --model <MODEL> \
  --output <OUTPUT_FILE> \
  --type <QUANT_TYPE> \
  [OPTIONS]
```

## Examples

```bash
# 4-bit quantization
pmetal quantize \
  --model ./output \
  --output model.gguf --type q4km

# With importance matrix
pmetal quantize \
  --model ./output \
  --output model.gguf --type q4km \
  --imatrix calibration.jsonl

# Dynamic per-layer quantization
pmetal quantize \
  --model ./output \
  --output model.gguf --type dynamic

# KL-calibrated quantization (per-tensor type selection)
pmetal quantize \
  --model ./output \
  --output model.gguf \
  --kl-calibrate --target-bpw 4.5
```

## Quantization Types

| Format | Description |
|--------|-------------|
| `dynamic` | Auto-select per layer |
| `q8_0` | 8-bit quantization |
| `q6k` | 6-bit k-quant |
| `q5km` | 5-bit k-quant (medium) |
| `q5ks` | 5-bit k-quant (small) |
| `q4km` | 4-bit k-quant (medium) |
| `q4ks` | 4-bit k-quant (small) |
| `q3km` | 3-bit k-quant (medium) |
| `q3ks` | 3-bit k-quant (small) |
| `q3kl` | 3-bit k-quant (large) |
| `q2k` | 2-bit k-quant |
| `f16` | Float16 |
| `f32` | Float32 |

## See Also

- [Quantization](/models/quantization/) — Detailed quantization guide

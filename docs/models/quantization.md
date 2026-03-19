# Quantization

GGUF quantization with 13 format options and importance matrix support.

PMetal provides GGUF quantization with 13 format options and importance matrix support for quality-preserving compression.

## Quantization Formats

| Format | Description | Typical Size Reduction |
|--------|-------------|----------------------|
| `f32` | Float32 (no quantization) | 1× |
| `f16` | Float16 | 2× |
| `q8_0` | 8-bit quantization | 4× |
| `q6k` | 6-bit k-quant | ~5× |
| `q5km` | 5-bit k-quant (medium) | ~6× |
| `q5ks` | 5-bit k-quant (small) | ~6× |
| `q4km` | 4-bit k-quant (medium) | ~8× |
| `q4ks` | 4-bit k-quant (small) | ~8× |
| `q3km` | 3-bit k-quant (medium) | ~10× |
| `q3ks` | 3-bit k-quant (small) | ~10× |
| `q3kl` | 3-bit k-quant (large) | ~10× |
| `q2k` | 2-bit k-quant | ~16× |
| `dynamic` | Auto-select per layer | varies |

## Importance Matrix

Use `--imatrix` with a calibration dataset to preserve quality on important weights:

```bash
pmetal quantize \
  --model ./output \
  --output model.gguf \
  --type q4km \
  --imatrix calibration.jsonl
```

## FP8 Runtime Quantization

For inference-time memory reduction without GGUF conversion:

```bash
pmetal infer --model Qwen/Qwen3-4B --fp8 --chat
```

Converts to FP8 (E4M3) at load time for approximately 2× memory reduction.

## See Also

- [pmetal quantize](/cli/quantize/) — CLI reference
- [Supported Models](/models/supported/) — Compatible architectures

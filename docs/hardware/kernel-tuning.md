# Kernel Tuning

Per-tier Metal kernel tuning — matrix tiles, FlashAttention blocks, threadgroup sizes, and batch multipliers.

PMetal automatically selects kernel parameters based on your device tier and GPU family.

## Matrix Tile Size (GEMM, LoRA forward)

| Tier | Apple7–9 | Apple10 (M5+, NAX) |
|------|----------|-------------------|
| Base | 32×32×32 | 64×32×32 |
| Pro | 64×32×32 | 64×64×32 |
| Max | 64×64×32 | 128×64×32 |
| Ultra | 64×64×32 | 128×64×32 |

## FlashAttention Block Sizes

Block size selection per head dimension:

| Head Dim | Base | Pro | Max | Ultra |
|----------|------|-----|-----|-------|
| 64 | 64×32 | 64×32 | 64×64 | 64×64 |
| 80 | 64×32 | 64×32 | 64×64 | 64×64 |
| 96 | 64×32 | 64×32 | 64×64 | 64×64 |
| 128 | 32×32 | 32×32 | 64×64 | 64×64 |
| 256 | 32×16 | 32×16 | 32×32 | 32×32 |

## Fused Kernel Threadgroup Sizes

### Fused RMSNorm + LoRA

| Tier | Threadgroup Size |
|------|-----------------|
| Base | 128 |
| Pro | 128 |
| Max | 256 |
| Ultra | 256 |

### Fused SwiGLU

| Tier | Threadgroup Size |
|------|-----------------|
| Base | 256 |
| Pro | 256 |
| Max | 512 |
| Ultra | 512 |

## Batch Size Multiplier

| Tier | Multiplier |
|------|-----------|
| Base | 1× |
| Pro | 2× |
| Max | 4× |
| Ultra | 8× |

## Metal GPU Optimizations

| Kernel | Description |
|--------|-------------|
| FlashAttention | O(n) memory attention with fused softmax, tier-aware block sizes |
| Fused GDN | Gated Delta Network recurrence kernel — single-pass state update |
| Fused LoRA | Combined forward pass for adapter layers (~2× speedup) |
| Fused Cross-Entropy | Unsloth-style chunked loss computation |
| Fused Linear Cross-Entropy | Skips logits materialization entirely |
| Fused RoPE | Rotary position embeddings in-kernel |
| Fused SwiGLU | Fused gate + activation with tier-tuned threadgroups |
| Fused RMSNorm + LoRA | Combined normalization and adapter projection |
| Fused Sampler | JIT-compiled token sampling |
| Fused MLP | Combined gate/up/down projections |
| Async Scheduler | Double/triple-buffered GPU command scheduling |

## See Also

- [Apple Silicon Support](/hardware/apple-silicon/) — Hardware detection matrix
- [pmetal bench](/cli/bench/) — Benchmark on your hardware

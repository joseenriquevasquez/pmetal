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

These tier tables apply to the standard Metal GEMM/LoRA kernels. On Apple10/M5 hardware, the Metal 4 / MPP dispatcher now auto-tunes and persists among `32×32` / `1`-simdgroup, `64×32` / `2`-simdgroup, `32×64` / `2`-simdgroup, and `64×64` / `4`-simdgroup variants, plus Morton-vs-linear tile walk order. Aligned M/N tiles use static full-tile extents. Apple7-9 continue to use the standard Metal kernels. For 4-bit affine quantized linear inference, Apple10/M5 also benchmarks and persists MLX `quantized_matmul` versus the MPP path per device and problem shape. For supported `head_dim = 128` inference attention shapes, Apple10/M5 now benchmarks and persists MLX fast SDPA vs Metal FlashAttention vs MPP FlashAttention, rejecting MPP candidates that diverge numerically from the MLX reference.

## FlashAttention Block Sizes

Block size selection per head dimension:

| Head Dim | Base | Pro | Max | Ultra |
|----------|------|-----|-----|-------|
| 64 | 64×32 | 64×64 | 64×64 | 64×64 |
| 80 | 32×32 | 64×32 | 64×32 | 64×32 |
| 96 | 32×32 | 64×32 | 64×32 | 64×32 |
| 128 | 32×32 | 32×32 | 64×32 | 64×32 |
| 256 | 16×16 | 16×16 | 32×16 | 32×16 |

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
| Fused Cross-Entropy | Chunked vocabulary loss computation |
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

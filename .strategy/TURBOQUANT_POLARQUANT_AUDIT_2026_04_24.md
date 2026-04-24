# TurboQuant / PolarQuant Diligence Audit — 2026-04-24

Audit of pmetal's TurboQuant implementation against turboquant.pdf, and PolarQuant status vs polarquant.pdf.

## Verdict

**TurboQuant core algorithm is correct.** No critical math bugs. Several defensive gaps and performance wins worth addressing. PolarQuant is **not implemented** — only referenced by name in one comment.

## Confirmed correct (don't touch)

| Claim | Status |
|---|---|
| QJL constant `√(π/2)/d = 1.2533141373155 / d` matches Theorem 2 | ✓ |
| Rotation applied as `x · Π^T` via row-major orthogonal matrix (confusingly named `rotation` but equivalent to `Π^T`) | ✓ |
| Keys use 2-stage MSE + QJL residual; values are MSE-only | ✓ (matches paper) |
| Norm rescaling after MSE+QJL reconstruction | ✓ |
| `as_type<float>((uint)attn_scale_bits)` in D256 kernels | ✓ intentional bit-reinterpret, not UB — `attn_scale_bits: uint32_t` carries the bit pattern of a float through Metal's uint constant slot |

## Actionable findings

### Correctness / robustness (MODERATE)

**A1. Missing `residual_norm` clamp on encode.** Theorem 2's guarantee assumes `||r|| ≤ √MSE_{b-1}`. If upstream fp16 corruption produces a NaN/negative/huge value, the QJL term in `bridge_turboquant_score.cpp:46-49` and `bridge_turboquant_attn_d256.cpp:72` can blow up. Guard `if (residual > 0.0)` catches negatives but not NaN or pathological magnitudes. Recommendation: clamp at encode time to `[0, ceil_from_codebook_max]`, computed once from the Beta codebook.

**A2. Mixed-bit outlier channels may violate Theorem 1 independence.** `TurboQuantTensorConfig::Mixed` (turboquant.rs:94-187) quantizes regular channels at one bit-width and outlier channels at another. The paper's distortion bound assumes identical per-coordinate codebook. If outliers are a disjoint subspace the bound recovers by superposition, but this isn't proven or asserted in code. Recommendation: either add a comment + unit test demonstrating subspace disjointness, or document the divergence from the paper's guarantee.

### Performance (MODERATE)

**P1. Redundant reshape copies in `gpu_quantize_mse`** (turboquant.rs:443-461): `[B,H,S,D] → [N,D] → quantize → [B,H,S,D]` allocates intermediate. Teach the kernel to accept arbitrary rank + flatten internally (standard MLX pattern). ~3-8% throughput on mid-batch inference.

**P2. No fused rotate-and-encode kernel.** Current path issues `matmul(rotation)` then `encode` as two dispatches. KV append is dominated by this on short sequences. A fused `turboquant_fused_rotate_encode` over the 256-dim tile would land a measurable win on decode-phase KV writes.

### Test coverage (MODERATE)

**T1. No round-trip error-bound test.** There is no test asserting that `E[||x - Π^T · decode(encode(Π · x))||²] ≤ distortion_from_Theorem_1` — the single most important correctness invariant. mlx-vlm has `test_turboquant_prod_is_nearly_unbiased_across_seeds`; port its equivalent.

**T2. No long-context numeric validation** vs CPU golden reference for the 2-pass attention (`bridge_turboquant_attn_d256.cpp` log-sum-exp reduction in pass-1/pass-2 merge is the fragile bit).

### Gap (CRITICAL)

**G1. PolarQuant not implemented.** Only reference is a comment in `crates/pmetal-bridge/src/qwen3_native/weights.rs:1-5`. PolarQuant's recursive polar transform (radii + log₂d angles, Lloyd-Max per level) is orthogonal to TurboQuant's direct codebook lookup and targets **weight** quantization where it should beat NF4 / AWQ on per-channel-correlated layers. Implementing it is a substantial piece of work (not a quick fix) and should be a standalone project, not a diligence-pass drive-by.

## Recommended next actions (in priority order)

1. **A1 + T1 together** (~1 day): add encode-time clamp + port the unbiasedness test. Small, low-risk, catches real edge cases, validates the invariant that matters most.
2. **T2** (~1 day): CPU golden for 2-pass attention. Catches future numeric regressions.
3. **P1** (~half day): reshape-in-kernel refactor.
4. **P2** (~2-3 days): fused rotate-encode Metal kernel. Needs benchmarking harness.
5. **A2** (~half day): document mixed-bit guarantees or add disjointness assertion.
6. **G1 PolarQuant**: scope as separate project — read the paper end-to-end, design weight-quant integration point, decide whether to host in pmetal-bridge (parallel to turboquant.rs) or a new crate. Don't touch until dedicated sprint.

## Files surveyed

- crates/pmetal-bridge/cpp/bridge_turboquant_{attn_d128,attn_d256,encode,pack,score,weighted,internal.h}
- crates/pmetal-bridge/cpp/bridge/turboquant.h
- crates/pmetal-bridge/src/turboquant.rs
- crates/pmetal-bridge/src/qwen3_native/weights.rs
- .strategy/10-turboquant-integration.md, TURBOQUANT_REDESIGN_CONTEXT_2026-04-02.md, 06-diligence-pass-2.md

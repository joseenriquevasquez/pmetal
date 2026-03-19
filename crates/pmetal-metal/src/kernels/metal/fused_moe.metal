/// Fused MoE (Mixture of Experts) Metal compute kernels.
///
/// Provides optimized dequant kernels for MoE inference using the qdot technique
/// (pre-scaled activations, eliminating shifts from inner loop):
///
/// 1. `fused_gate_up_swiglu` — Fused gate+up+SwiGLU for a single quantized expert
/// 2. `dequant_matvec_4bit` — Optimized 4-bit dequant matrix-vector multiply
/// 3. `dequant_matvec_2bit` — 2-bit variant for further compressed experts
/// 4. `gather_qmm_swiglu` — Fused gather + quantized matmul + SwiGLU for resident mode
/// 5. `gather_dequant_matvec` — Down projection for resident mode
///
/// Quantization format (MLX affine, group_size via params/FC):
///   - Weights stored as uint32, each holding pack_factor values
///   - Per-group scale and bias in bfloat16
///   - qdot: pre-scale activations to eliminate per-nibble shifts
///   - result = scale * qdot_accum + bias * x_sum  (factored outer FMA)
///
/// Thread model: 2 simdgroups × 32 threads = 64 threads per threadgroup
/// Each simdgroup computes RESULTS_PER_SG output rows (register-only, no shared memory)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

// ============================================================================
// Threadgroup geometry
// ============================================================================

// 4 output rows per simdgroup, 2 simdgroups per threadgroup
#define RESULTS_PER_SG 4
#define NUM_SIMDGROUPS  2
#define ROWS_PER_TG    (RESULTS_PER_SG * NUM_SIMDGROUPS)  // 8
#define TG_SIZE        (NUM_SIMDGROUPS * 32)                // 64

// ============================================================================
// Function constants for compile-time specialization (gather kernels)
// ============================================================================

constant uint FC_GROUP_SIZE     [[function_constant(0)]];
constant uint FC_BITS           [[function_constant(1)]]; // 2 or 4

// ============================================================================
// 4-bit qdot helper: pre-scaled dot product for one uint32 (8 nibbles)
// ============================================================================
//
// Computes sum_i[ q_i * x_i ] where q_i is the i-th 4-bit value in `packed`.
// Instead of shifting, we pre-scale x values and use mask-only extraction:
//   (packed & 0x00F0) * (x[1] / 16) == ((packed >> 4) & 0xF) * x[1]
//
// Returns (qdot_accum, x_sum) where:
//   qdot_accum = sum_i[ q_i * x_i ]  (via pre-scaled activations)
//   x_sum = sum_i[ x_i ]             (for factored bias: bias * x_sum)

inline float2 qdot4(device const float* x_ptr, uint32_t packed) {
    float x0 = x_ptr[0], x1 = x_ptr[1], x2 = x_ptr[2], x3 = x_ptr[3];
    float x4 = x_ptr[4], x5 = x_ptr[5], x6 = x_ptr[6], x7 = x_ptr[7];

    // Pre-scale: divide by shift factor to eliminate >> from inner product
    // uint16 low half: nibbles at bit positions 0, 4, 8, 12
    // uint16 high half: nibbles at bit positions 0, 4, 8, 12 (after >>16)
    uint16_t lo = uint16_t(packed);
    uint16_t hi = uint16_t(packed >> 16);

    float accum = float(lo & 0x000fu) * x0
                + float(lo & 0x00f0u) * (x1 * (1.0f / 16))
                + float(lo & 0x0f00u) * (x2 * (1.0f / 256))
                + float(lo & 0xf000u) * (x3 * (1.0f / 4096))
                + float(hi & 0x000fu) * x4
                + float(hi & 0x00f0u) * (x5 * (1.0f / 16))
                + float(hi & 0x0f00u) * (x6 * (1.0f / 256))
                + float(hi & 0xf000u) * (x7 * (1.0f / 4096));

    float x_sum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

    return float2(accum, x_sum);
}

// ============================================================================
// 2-bit qdot helper: pre-scaled dot product for one uint32 (16 crumbs)
// ============================================================================

inline float2 qdot2(device const float* x_ptr, uint32_t packed) {
    uint16_t lo = uint16_t(packed);
    uint16_t hi = uint16_t(packed >> 16);

    // 2-bit: 8 values per uint16 at bit positions 0,2,4,6,8,10,12,14
    // Pre-scale factors: 1, 1/4, 1/16, 1/64, 1/256, 1/1024, 1/4096, 1/16384
    float accum = float(lo & 0x0003u) * x_ptr[0]
                + float(lo & 0x000cu) * (x_ptr[1]  * (1.0f / 4))
                + float(lo & 0x0030u) * (x_ptr[2]  * (1.0f / 16))
                + float(lo & 0x00c0u) * (x_ptr[3]  * (1.0f / 64))
                + float(lo & 0x0300u) * (x_ptr[4]  * (1.0f / 256))
                + float(lo & 0x0c00u) * (x_ptr[5]  * (1.0f / 1024))
                + float(lo & 0x3000u) * (x_ptr[6]  * (1.0f / 4096))
                + float(lo & 0xc000u) * (x_ptr[7]  * (1.0f / 16384))
                + float(hi & 0x0003u) * x_ptr[8]
                + float(hi & 0x000cu) * (x_ptr[9]  * (1.0f / 4))
                + float(hi & 0x0030u) * (x_ptr[10] * (1.0f / 16))
                + float(hi & 0x00c0u) * (x_ptr[11] * (1.0f / 64))
                + float(hi & 0x0300u) * (x_ptr[12] * (1.0f / 256))
                + float(hi & 0x0c00u) * (x_ptr[13] * (1.0f / 1024))
                + float(hi & 0x3000u) * (x_ptr[14] * (1.0f / 4096))
                + float(hi & 0xc000u) * (x_ptr[15] * (1.0f / 16384));

    float x_sum = 0.0f;
    for (int i = 0; i < 16; i++) x_sum += x_ptr[i];

    return float2(accum, x_sum);
}


// ============================================================================
// Kernel 1: Fused Gate+Up+SwiGLU (single expert, quantized 4-bit)
// ============================================================================
//
// For a single quantized expert: reads input x ONCE per column, computes:
//   gate_out = gate_W @ x   (dequant 4-bit via qdot)
//   up_out   = up_W @ x     (dequant 4-bit via qdot)
//   output   = silu(gate_out) * up_out
//
// Each simdgroup computes RESULTS_PER_SG=4 output rows.
// 64 threads = 2 simdgroups = 8 output rows per threadgroup.
// Thread-private registers only — no threadgroup shared memory.

kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W      [[buffer(0)]],   // [out_dim, in_dim/8] packed
    device const uint16_t* gate_scales [[buffer(1)]],   // [out_dim, num_groups] bf16
    device const uint16_t* gate_biases [[buffer(2)]],   // [out_dim, num_groups] bf16
    device const uint32_t* up_W        [[buffer(3)]],   // [out_dim, in_dim/8] packed
    device const uint16_t* up_scales   [[buffer(4)]],   // [out_dim, num_groups] bf16
    device const uint16_t* up_biases   [[buffer(5)]],   // [out_dim, num_groups] bf16
    device const float*    x           [[buffer(6)]],   // [in_dim]
    device float*          out         [[buffer(7)]],   // [out_dim]
    constant uint&         out_dim     [[buffer(8)]],
    constant uint&         in_dim      [[buffer(9)]],
    constant uint&         group_size  [[buffer(10)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 8; // packed columns per quantization group

    float ga[RESULTS_PER_SG] = {0, 0, 0, 0};
    float ua[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        device const float* x_ptr = x + col * 8;
        uint g = col / pf_gs;

        // Compute qdot for each row (amortizes x load across RESULTS_PER_SG rows)
        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            uint w_idx = row * packed_cols + col;
            uint sb_idx = row * num_groups + g;

            // Gate qdot
            float2 gqd = qdot4(x_ptr, gate_W[w_idx]);
            float gsc = bf16_to_f32(gate_scales[sb_idx]);
            float gbi = bf16_to_f32(gate_biases[sb_idx]);
            ga[r] += gsc * gqd.x + gbi * gqd.y;

            // Up qdot
            float2 uqd = qdot4(x_ptr, up_W[w_idx]);
            float usc = bf16_to_f32(up_scales[sb_idx]);
            float ubi = bf16_to_f32(up_biases[sb_idx]);
            ua[r] += usc * uqd.x + ubi * uqd.y;
        }
    }

    // SIMD reduction + SwiGLU write
    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float rg = simd_sum(ga[r]);
        float ru = simd_sum(ua[r]);
        if (simd_lane == 0) {
            out[row] = (rg / (1.0f + exp(-rg))) * ru;
        }
    }
}


// ============================================================================
// Kernel 2: Optimized 4-bit dequant matrix-vector multiply (qdot)
// ============================================================================
//
// For down projection (or any single matvec with quantized weights).
// Each simdgroup computes RESULTS_PER_SG=4 output rows via qdot.
// Thread-private registers only — no threadgroup shared memory.

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 8;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        device const float* x_ptr = x + col * 8;
        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            float2 qd = qdot4(x_ptr, W_packed[row * packed_cols + col]);
            float scale = bf16_to_f32(scales[row * num_groups + g]);
            float bias  = bf16_to_f32(biases[row * num_groups + g]);
            acc[r] += scale * qd.x + bias * qd.y;
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            out[row] = sum;
        }
    }
}


// ============================================================================
// Kernel 3: 2-bit dequant matrix-vector multiply (qdot)
// ============================================================================
//
// Same structure as 4-bit but packs 16 x 2-bit values per uint32.
// ~44% smaller expert files for proportionally faster SSD streaming.

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 16;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        device const float* x_ptr = x + col * 16;
        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            float2 qd = qdot2(x_ptr, W_packed[row * packed_cols + col]);
            float scale = bf16_to_f32(scales[row * num_groups + g]);
            float bias  = bf16_to_f32(biases[row * num_groups + g]);
            acc[r] += scale * qd.x + bias * qd.y;
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            out[row] = sum;
        }
    }
}


// ============================================================================
// Kernel 4: Gather + Quantized MatMul + SwiGLU (resident mode, qdot)
// ============================================================================
//
// For resident inference where all expert weights fit in GPU memory.
// Replaces the 3x gather_mm + silu + multiply pattern.
//
// Uses FC_BITS function constant for PSO specialization (no runtime branch).
// Each simdgroup computes RESULTS_PER_SG output rows for one token-expert pair.
// Thread-private registers — no threadgroup shared memory.
//
// Grid: ceil(intermediate / ROWS_PER_TG) * N * K
// Where N = number of tokens, K = top-k experts per token.

struct GatherQmmSwigluParams {
    uint hidden_dim;            // Input dimension (D)
    uint intermediate_dim;      // Output dimension per gate/up (I)
    uint num_tokens;            // N
    uint topk;                  // K
};

kernel void gather_qmm_swiglu(
    device const uint32_t* gate_weights  [[buffer(0)]],  // [E, I, D/pack_factor]
    device const uint16_t* gate_scales   [[buffer(1)]],  // [E, I, D/group_size]
    device const uint16_t* gate_biases   [[buffer(2)]],  // [E, I, D/group_size]
    device const uint32_t* up_weights    [[buffer(3)]],  // [E, I, D/pack_factor]
    device const uint16_t* up_scales     [[buffer(4)]],  // [E, I, D/group_size]
    device const uint16_t* up_biases     [[buffer(5)]],  // [E, I, D/group_size]
    device const float*    input         [[buffer(6)]],  // [N, D]
    device const uint*     expert_ids    [[buffer(7)]],  // [N, K]
    device float*          output        [[buffer(8)]],  // [N, K, I]
    constant GatherQmmSwigluParams& params [[buffer(9)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint D = params.hidden_dim;
    uint I = params.intermediate_dim;
    uint K = params.topk;

    // Determine which (token, expert_slot, output_row_tile) this threadgroup handles
    uint tiles_per_token_expert = (I + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint token_expert_idx = tgid / tiles_per_token_expert;
    uint tile_idx = tgid % tiles_per_token_expert;

    uint n = token_expert_idx / K;  // token index
    uint k = token_expert_idx % K;  // expert slot
    uint row_base = tile_idx * ROWS_PER_TG + simd_group * RESULTS_PER_SG;

    if (n >= params.num_tokens || row_base >= I) return;

    // Look up which expert this token-slot maps to
    uint expert_id = expert_ids[n * K + k];

    // Compute layout constants using function constants
    uint pack_factor = (FC_BITS == 2) ? 16 : 8;
    uint packed_cols = D / pack_factor;
    uint num_groups = D / FC_GROUP_SIZE;
    uint pf_gs = FC_GROUP_SIZE / pack_factor;

    // Per-expert weight base offsets
    uint expert_weight_offset = expert_id * I * packed_cols;
    uint expert_scale_offset = expert_id * I * num_groups;

    device const float* x_in = input + n * D;

    float ga[RESULTS_PER_SG] = {0, 0, 0, 0};
    float ua[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        device const float* x_ptr = x_in + col * pack_factor;
        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= I) break;

            uint w_idx = expert_weight_offset + row * packed_cols + col;
            uint sb_idx = expert_scale_offset + row * num_groups + g;

            if (FC_BITS == 4) {
                // Gate qdot (4-bit)
                float2 gqd = qdot4(x_ptr, gate_weights[w_idx]);
                ga[r] += bf16_to_f32(gate_scales[sb_idx]) * gqd.x
                       + bf16_to_f32(gate_biases[sb_idx]) * gqd.y;

                // Up qdot (4-bit)
                float2 uqd = qdot4(x_ptr, up_weights[w_idx]);
                ua[r] += bf16_to_f32(up_scales[sb_idx]) * uqd.x
                       + bf16_to_f32(up_biases[sb_idx]) * uqd.y;
            } else {
                // Gate qdot (2-bit)
                float2 gqd = qdot2(x_ptr, gate_weights[w_idx]);
                ga[r] += bf16_to_f32(gate_scales[sb_idx]) * gqd.x
                       + bf16_to_f32(gate_biases[sb_idx]) * gqd.y;

                // Up qdot (2-bit)
                float2 uqd = qdot2(x_ptr, up_weights[w_idx]);
                ua[r] += bf16_to_f32(up_scales[sb_idx]) * uqd.x
                       + bf16_to_f32(up_biases[sb_idx]) * uqd.y;
            }
        }
    }

    // SIMD reduction + SwiGLU write
    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= I) break;
        float rg = simd_sum(ga[r]);
        float ru = simd_sum(ua[r]);
        if (simd_lane == 0) {
            output[(n * K + k) * I + row] = (rg / (1.0f + exp(-rg))) * ru;
        }
    }
}


// ============================================================================
// Kernel 5: Gather + Dequant MatVec (down projection for resident mode, qdot)
// ============================================================================
//
// Phase 2 of gather_qmm_swiglu: applies down projection per expert.
// Uses FC_BITS function constant for PSO specialization.
// Thread-private registers — no threadgroup shared memory.

struct GatherDequantMatvecParams {
    uint in_dim;       // Intermediate dimension (I) — input to down proj
    uint out_dim;      // Hidden dimension (D) — output of down proj
    uint num_tokens;   // N
    uint topk;         // K
};

kernel void gather_dequant_matvec(
    device const uint32_t* down_weights [[buffer(0)]],  // [E, D, I/pack_factor]
    device const uint16_t* down_scales  [[buffer(1)]],  // [E, D, I/group_size]
    device const uint16_t* down_biases  [[buffer(2)]],  // [E, D, I/group_size]
    device const float*    input        [[buffer(3)]],  // [N, K, I] — SwiGLU output
    device const uint*     expert_ids   [[buffer(4)]],  // [N, K]
    device float*          output       [[buffer(5)]],  // [N, K, D]
    constant GatherDequantMatvecParams& params [[buffer(6)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint I = params.in_dim;   // intermediate
    uint D = params.out_dim;  // hidden
    uint K = params.topk;

    uint tiles_per_token_expert = (D + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint token_expert_idx = tgid / tiles_per_token_expert;
    uint tile_idx = tgid % tiles_per_token_expert;

    uint n = token_expert_idx / K;
    uint k = token_expert_idx % K;
    uint row_base = tile_idx * ROWS_PER_TG + simd_group * RESULTS_PER_SG;

    if (n >= params.num_tokens || row_base >= D) return;

    uint expert_id = expert_ids[n * K + k];

    uint pack_factor = (FC_BITS == 2) ? 16 : 8;
    uint packed_cols = I / pack_factor;
    uint num_groups = I / FC_GROUP_SIZE;
    uint pf_gs = FC_GROUP_SIZE / pack_factor;

    uint expert_weight_offset = expert_id * D * packed_cols;
    uint expert_scale_offset = expert_id * D * num_groups;

    device const float* x_in = input + (n * K + k) * I;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        device const float* x_ptr = x_in + col * pack_factor;
        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= D) break;

            uint w_idx = expert_weight_offset + row * packed_cols + col;
            uint sb_idx = expert_scale_offset + row * num_groups + g;

            if (FC_BITS == 4) {
                float2 qd = qdot4(x_ptr, down_weights[w_idx]);
                acc[r] += bf16_to_f32(down_scales[sb_idx]) * qd.x
                        + bf16_to_f32(down_biases[sb_idx]) * qd.y;
            } else {
                float2 qd = qdot2(x_ptr, down_weights[w_idx]);
                acc[r] += bf16_to_f32(down_scales[sb_idx]) * qd.x
                        + bf16_to_f32(down_biases[sb_idx]) * qd.y;
            }
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= D) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            output[(n * K + k) * D + row] = sum;
        }
    }
}

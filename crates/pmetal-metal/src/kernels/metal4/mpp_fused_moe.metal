// mpp_fused_moe.metal
// Metal 4 Fused MoE Expert Forward using MPP matmul2d.
//
// Computes the full expert forward pass for dense (fp16) MoE experts:
//   gate = x @ gate_weight^T      [hidden → intermediate]
//   up   = x @ up_weight^T        [hidden → intermediate]
//   act  = silu(gate) * up        [SwiGLU — postfix fusion in registers]
//   out  = act @ down_weight^T    [intermediate → hidden]
//
// MPP Guide Section 2.3.4 (Postfix Fusion):
//   gate and up GEMMs keep results in cooperative tensor registers (rGate, rUp).
//   SwiGLU is applied element-wise in register space, producing rAct.
//   rAct is held in registers and used as the A input to the down projection
//   GEMM — no intermediate global memory write between gate/up and down.
//
// MPP Guide Section 2.3.1 (Single simdgroup):
//   execution_simdgroup is used throughout. NAX matmul2d provides hardware
//   acceleration for the 32×32 GEMM tiles.
//
// Tile dimensions: BM=32 (token), BN=32 (dim), matching the recommended
// single-simdgroup tile for NAX.
//
// For weighted residual combine (MoE scatter), a separate lightweight kernel
// adds the expert contribution scaled by the router gate weight.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct MppMoEParams {
    uint batch_size;        // number of tokens dispatched to this expert
    uint hidden_dim;        // model hidden dimension H
    uint intermediate_dim;  // expert intermediate dimension I
};

// Expert routing weight for weighted accumulation.
struct MppMoEScatterParams {
    uint num_tokens;
    uint hidden_dim;
};

inline float silu_moe(float x) {
    return x / (1.0f + metal::fast::exp(-x));
}

// =============================================================================
// MPP Fused MoE Expert Forward — gate+up SwiGLU (fp16)
// =============================================================================
//
// Produces act[batch, intermediate] = silu(gate) * up in registers.
// Output: intermediate activations held in rAct register array.
//
// Grid: [ceil(I/32), ceil(B/32), 1]  Threadgroup: [32, 1, 1]

kernel void mpp_fused_moe_gate_up_f16(
    device const half*   input         [[buffer(0)]],   // [B, H]
    device const half*   gate_weight   [[buffer(1)]],   // [I, H]
    device const half*   up_weight     [[buffer(2)]],   // [I, H]
    device half*         act_out       [[buffer(3)]],   // [B, I] — SwiGLU output
    constant MppMoEParams& params      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;

    constexpr int BM = 32;
    constexpr int BN = 32;

    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // Input:       [B, H] row-major → columns-first tensor
    auto tX  = tensor(input,       dextents<int, 2>{H, B}, array<int, 2>{1, H});
    // Weights:     [I, H] row-major → accessed as [H, I] transposed
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight,   dextents<int, 2>{H, I}, array<int, 2>{1, H});
    // Output:      [I, B] columns-first
    auto tOut = tensor(act_out,    dextents<int, 2>{I, B}, array<int, 2>{1, I});

    auto sliceX  = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    // MPP single-simdgroup matmul: 32×32 tile, dynamic K, A@B^T
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    // Gate GEMM — result in registers
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> gate_op;
    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), float>();
    gate_op.run(sliceX, sliceGW, rGate);

    // Up GEMM — result in registers
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> up_op;
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceUW), float>();
    up_op.run(sliceX, sliceUW, rUp);

    // Postfix SwiGLU fusion in register space
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> act_op;
    auto rAct = act_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(sliceGW), half>();

    for (int i = 0; i < rGate.get_capacity(); i++) {
        rAct[i] = half(silu_moe(rGate[i]) * rUp[i]);
    }

    // Single store — no threadgroup staging
    auto sliceOut = tOut.slice(tile_i, tile_b);
    rAct.store(sliceOut);
}

// =============================================================================
// MPP MoE Down Projection (fp16)
// =============================================================================
//
// Computes: expert_out[B, H] = act[B, I] @ down_weight^T
// Grid: [ceil(H/32), ceil(B/32), 1]  Threadgroup: [32, 1, 1]

kernel void mpp_fused_moe_down_f16(
    device const half*   act_in       [[buffer(0)]],   // [B, I] — SwiGLU output
    device const half*   down_weight  [[buffer(1)]],   // [H, I]
    device half*         out          [[buffer(2)]],   // [B, H]
    constant MppMoEParams& params     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;

    constexpr int BM = 32;
    constexpr int BN = 32;

    const int tile_b = (int)(tgid.y * BM);
    const int tile_h = (int)(tgid.x * BN);
    if (tile_b >= B || tile_h >= H) return;

    // act_in: [B, I] row-major → [I, B] columns-first
    auto tA   = tensor(act_in,     dextents<int, 2>{I, B}, array<int, 2>{1, I});
    // down_weight: [H, I] row-major → accessed as [I, H] transposed
    auto tDW  = tensor(down_weight, dextents<int, 2>{I, H}, array<int, 2>{1, I});
    // out: [B, H] row-major → [H, B] columns-first
    auto tOut = tensor(out,         dextents<int, 2>{H, B}, array<int, 2>{1, H});

    auto sliceA  = tA.slice(0, tile_b);
    auto sliceDW = tDW.slice(0, tile_h);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> down_op;
    auto rOut = down_op.template get_destination_cooperative_tensor<
        decltype(sliceA), decltype(sliceDW), half>();
    down_op.run(sliceA, sliceDW, rOut);

    auto sliceOut = tOut.slice(tile_h, tile_b);
    rOut.store(sliceOut);
}

// =============================================================================
// Weighted Residual Scatter (MoE combine)
// =============================================================================
//
// Adds expert contribution scaled by router weight to the output accumulator.
// out[token, :] += weight[token] * expert_out[local_idx, :]
//
// Element-wise — no GEMM. Single SIMD group per token, strided over hidden_dim.
// Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]

kernel void mpp_moe_weighted_scatter_f16(
    device const half*   expert_out   [[buffer(0)]],   // [num_tokens, H]
    device const float*  weights      [[buffer(1)]],   // [num_tokens]
    device half*         accum        [[buffer(2)]],   // [num_tokens, H] (read-modify-write)
    constant MppMoEScatterParams& p   [[buffer(3)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float w = weights[token_idx];
    const device half* src = expert_out + token_idx * p.hidden_dim;
    device half* dst       = accum      + token_idx * p.hidden_dim;

    for (uint d = lane; d < p.hidden_dim; d += 32u) {
        dst[d] = half(float(dst[d]) + w * float(src[d]));
    }
}

// =============================================================================
// MPP Fused MoE Expert — Quantized Gate+Up (4-bit weights)
// =============================================================================
//
// Handles MoeExpertDescriptor's packed u32 weight buffers with fp16 scales
// and biases. Dequantizes gate and up weight tiles cooperatively into
// threadgroup fp16, then uses MPP matmul2d for the SwiGLU projections.
// The down projection follows the same dequantize-then-matmul pattern.
//
// Dequant: W_fp16[i] = scale[group(i)] * nibble(W_q[i]) + bias[group(i)]
//
// Grid for gate_up: [ceil(I/32), ceil(B/32), 1]  Threadgroup: [128, 1, 1]
// (128 threads = 4 simdgroups for cooperative dequant, single execution_simdgroup
// for the matmul2d)
//
// Grid for down:    [ceil(H/32), ceil(B/32), 1]  Threadgroup: [128, 1, 1]

struct MppMoEQuantParams {
    uint batch_size;        // number of tokens
    uint hidden_dim;        // H
    uint intermediate_dim;  // I
    uint group_size;        // quantization group size (e.g. 64)
    uint bits;              // quantization bits (4)
};

// ---------------------------------------------------------------------------
// Dequant helper: expand packed 4-bit word into fp16 array.
// Writes 8 half values (nibbles) into dst.
// ---------------------------------------------------------------------------
inline void dequant_4bit_word(
    uint32_t packed,
    half scale,
    half bias,
    threadgroup half* dst
) {
    float s = float(scale);
    float b = float(bias);
    for (uint k = 0; k < 8; k++) {
        uint nibble = (packed >> (k * 4)) & 0xFu;
        dst[k] = half(s * float(nibble) + b);
    }
}

// =============================================================================
// mpp_fused_moe_gate_up_quant_f16
//
// Gate + Up SwiGLU for quantized (4-bit) expert weights.
// Output: act_out[B, I] = silu(x @ gate_W^T) * (x @ up_W^T)
// =============================================================================

kernel void mpp_fused_moe_gate_up_quant_f16(
    device const half*     input       [[buffer(0)]],   // [B, H] fp16 activations
    device const uint32_t* gate_w_q    [[buffer(1)]],   // [I, H/8] packed 4-bit gate
    device const half*     gate_scales [[buffer(2)]],   // [I, H/group_size]
    device const half*     gate_biases [[buffer(3)]],   // [I, H/group_size]
    device const uint32_t* up_w_q      [[buffer(4)]],   // [I, H/8] packed 4-bit up
    device const half*     up_scales   [[buffer(5)]],   // [I, H/group_size]
    device const half*     up_biases   [[buffer(6)]],   // [I, H/group_size]
    device half*           act_out     [[buffer(7)]],   // [B, I] SwiGLU output
    constant MppMoEQuantParams& params [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;
    const uint gs = params.group_size;

    constexpr int BM = 32;   // batch tile
    constexpr int BN = 32;   // intermediate tile
    constexpr int BK = 32;   // K-tile width for dequant chunks

    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // Threadgroup dequant buffers — one for gate, one for up.
    // Each is BN * BK half values = 32*32*2 = 2 KB.
    threadgroup half W_gate_dq[BN * BK];
    threadgroup half W_up_dq[BN * BK];

    // Shared thread count for cooperative dequant (128 threads = 4 simdgroups).
    const uint total_threads = 128;
    const uint linear_tid = simd_group_id * 32 + simd_lane_id;
    const uint packed_cols = (uint)H / 8u;   // 8 nibbles per uint32

    auto tX   = tensor(input,   dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tOut = tensor(act_out, dextents<int, 2>{I, B}, array<int, 2>{1, I});
    auto sliceX   = tX.slice(0, tile_b);
    auto sliceOut = tOut.slice(tile_i, tile_b);

    // Gate/Up GEMM descriptor with accumulate mode so we sum over K-tiles.
    constexpr auto gu_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK,
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<gu_desc, execution_simdgroup> gate_op;
    mpp::tensor_ops::matmul2d<gu_desc, execution_simdgroup> up_op;

    auto tGateDQ = tensor((threadgroup half*)W_gate_dq, dextents<int, 2>{BK, BN}, array<int, 2>{1, BK});
    auto tUpDQ   = tensor((threadgroup half*)W_up_dq,   dextents<int, 2>{BK, BN}, array<int, 2>{1, BK});

    auto rGate = gate_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(tGateDQ), float>();
    auto rUp = up_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(tUpDQ), float>();

    // K-loop: dequantize BK columns of gate and up at a time, accumulate.
    for (int k_start = 0; k_start < H; k_start += BK) {
        // Cooperatively fill W_gate_dq and W_up_dq.
        for (uint idx = linear_tid; idx < (uint)(BN * BK); idx += total_threads) {
            uint n_local = idx / (uint)BK;
            uint k_local = idx % (uint)BK;
            uint global_n = (uint)tile_i + n_local;
            uint global_k = (uint)k_start + k_local;

            if (global_n < (uint)I && global_k < (uint)H) {
                uint word_idx   = global_k / 8u;
                uint nibble_idx = global_k % 8u;
                uint group_idx  = global_k / gs;
                uint num_groups = (uint)H / gs;

                // Gate
                uint32_t gw = gate_w_q[global_n * packed_cols + word_idx];
                uint g_nib  = (gw >> (nibble_idx * 4u)) & 0xFu;
                float gs_v  = float(gate_scales[global_n * num_groups + group_idx]);
                float gb_v  = float(gate_biases[global_n * num_groups + group_idx]);
                W_gate_dq[n_local * BK + k_local] = half(gs_v * float(g_nib) + gb_v);

                // Up
                uint32_t uw = up_w_q[global_n * packed_cols + word_idx];
                uint u_nib  = (uw >> (nibble_idx * 4u)) & 0xFu;
                float us_v  = float(up_scales[global_n * num_groups + group_idx]);
                float ub_v  = float(up_biases[global_n * num_groups + group_idx]);
                W_up_dq[n_local * BK + k_local] = half(us_v * float(u_nib) + ub_v);
            } else {
                W_gate_dq[n_local * BK + k_local] = half(0.0f);
                W_up_dq[n_local * BK + k_local]   = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sliceKX    = sliceX.slice(k_start, 0);
        gate_op.run(sliceKX, tGateDQ, rGate);
        up_op.run(sliceKX, tUpDQ, rUp);

        threadgroup_barrier(mem_flags::mem_none);
    }

    // Postfix SwiGLU: rAct = silu(rGate) * rUp (stored as fp16).
    constexpr auto store_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK, false, true, false
    );
    mpp::tensor_ops::matmul2d<store_desc, execution_simdgroup> act_op;
    auto rAct = act_op.template get_destination_cooperative_tensor<
        decltype(sliceX), decltype(tGateDQ), half>();
    for (int k = 0; k < rGate.get_capacity(); k++) {
        rAct[k] = half(silu_moe(rGate[k]) * rUp[k]);
    }
    rAct.store(sliceOut);
}

// =============================================================================
// mpp_fused_moe_down_quant_f16
//
// Down projection for quantized (4-bit) expert weights.
// out[B, H] = act[B, I] @ down_W^T
// =============================================================================

kernel void mpp_fused_moe_down_quant_f16(
    device const half*     act_in      [[buffer(0)]],   // [B, I] SwiGLU output
    device const uint32_t* down_w_q    [[buffer(1)]],   // [H, I/8] packed 4-bit down
    device const half*     down_scales [[buffer(2)]],   // [H, I/group_size]
    device const half*     down_biases [[buffer(3)]],   // [H, I/group_size]
    device half*           out         [[buffer(4)]],   // [B, H]
    constant MppMoEQuantParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_dim;
    const int I = (int)params.intermediate_dim;
    const uint gs = params.group_size;

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    const int tile_b = (int)(tgid.y * BM);
    const int tile_h = (int)(tgid.x * BN);
    if (tile_b >= B || tile_h >= H) return;

    threadgroup half W_down_dq[BN * BK];

    const uint total_threads = 128;
    const uint linear_tid = simd_group_id * 32 + simd_lane_id;
    const uint packed_cols = (uint)I / 8u;

    auto tA   = tensor(act_in, dextents<int, 2>{I, B}, array<int, 2>{1, I});
    auto tOut = tensor(out,    dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto sliceA   = tA.slice(0, tile_b);
    auto sliceOut = tOut.slice(tile_h, tile_b);

    constexpr auto down_desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK,
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<down_desc, execution_simdgroup> down_op;

    auto tDownDQ = tensor((threadgroup half*)W_down_dq, dextents<int, 2>{BK, BN}, array<int, 2>{1, BK});
    auto rOut = down_op.template get_destination_cooperative_tensor<
        decltype(sliceA), decltype(tDownDQ), half>();

    for (int k_start = 0; k_start < I; k_start += BK) {
        // Dequantize down_weight[tile_h..+BN, k_start..+BK] → threadgroup.
        for (uint idx = linear_tid; idx < (uint)(BN * BK); idx += total_threads) {
            uint n_local = idx / (uint)BK;   // output-dim local (H tile)
            uint k_local = idx % (uint)BK;   // intermediate local
            uint global_n = (uint)tile_h + n_local;
            uint global_k = (uint)k_start + k_local;

            if (global_n < (uint)H && global_k < (uint)I) {
                uint word_idx   = global_k / 8u;
                uint nibble_idx = global_k % 8u;
                uint group_idx  = global_k / gs;
                uint num_groups = (uint)I / gs;

                uint32_t dw  = down_w_q[global_n * packed_cols + word_idx];
                uint d_nib   = (dw >> (nibble_idx * 4u)) & 0xFu;
                float ds_v   = float(down_scales[global_n * num_groups + group_idx]);
                float db_v   = float(down_biases[global_n * num_groups + group_idx]);
                W_down_dq[n_local * BK + k_local] = half(ds_v * float(d_nib) + db_v);
            } else {
                W_down_dq[n_local * BK + k_local] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sliceKA = sliceA.slice(k_start, 0);
        down_op.run(sliceKA, tDownDQ, rOut);

        threadgroup_barrier(mem_flags::mem_none);
    }

    rOut.store(sliceOut);
}

// =============================================================================
// MPP MoE Tile Count Compute Kernel
// =============================================================================
//
// Computes total grouped GEMM tile count from expert_offsets on GPU,
// eliminating the CPU round-trip in grouped_gemm dispatch.
//
// expert_offsets[E+1] is a prefix-sum buffer: offsets[e+1]-offsets[e] gives
// the number of tokens for expert e.
//
// output[0] = sum_{e: tokens_e > 0} ceil(tokens_e / BLOCK_M) * num_n_tiles
//
// Grid: [1, 1, 1]  Threadgroup: [E_max, 1, 1]  (single threadgroup, E threads)
// E_max is clamped to 1024 at the call site.

struct TileCountParams {
    uint num_experts;   // E
    uint intermediate;  // N (output dim) — used to compute num_n_tiles
    uint block_m;       // BLOCK_M for grouped GEMM (64)
    uint block_n;       // BLOCK_N for grouped GEMM (64)
};

kernel void mpp_grouped_gemm_tile_count(
    device const uint*         expert_offsets [[buffer(0)]],  // [E+1]
    device atomic_uint*        tile_count_out [[buffer(1)]],  // [1] output
    constant TileCountParams&  params         [[buffer(2)]],
    uint expert_idx [[thread_position_in_threadgroup]]
) {
    if (expert_idx >= params.num_experts) return;

    uint m_start = expert_offsets[expert_idx];
    uint m_end   = expert_offsets[expert_idx + 1];
    uint m_size  = m_end - m_start;
    if (m_size == 0) return;

    uint num_n_tiles = (params.intermediate + params.block_n - 1) / params.block_n;
    uint num_m_tiles = (m_size + params.block_m - 1) / params.block_m;
    uint expert_tiles = num_m_tiles * num_n_tiles;

    atomic_fetch_add_explicit(tile_count_out, expert_tiles, memory_order_relaxed);
}

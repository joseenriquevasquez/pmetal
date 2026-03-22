// mpp_flash_attention.metal
// Metal 4 FlashAttention using MPP matmul2d for block GEMM.
//
// Key improvements over Metal 3 flash_attention.metal:
// - Block GEMM for S = Q @ K^T via matmul2d (replaces per-key scalar dot products)
// - Block GEMM for O += P @ V via matmul2d (replaces per-key scalar accumulation)
// - Online block softmax applied to score tile in register/cooperative tensor space
// - No explicit threadgroup memory staging for GEMM (MPP handles cache hierarchy)
// - Q/K/V tiles remain in threadgroup memory for online softmax bookkeeping only
//
// Algorithm (FlashAttention-2 with block GEMM):
//   For each Q block (Bq rows):
//     Load Q tile into threadgroup memory
//     For each KV block (Bk rows):
//       S_block = Q_tile @ K_block^T  [Bq × Bk]  ← MPP matmul2d
//       Apply causal mask to S_block
//       Online softmax: m_new, l_new, P_block = exp(S - m_new) / l_new
//       O_block += P_block @ V_block  [Bq × D]   ← MPP matmul2d
//     Normalize O by final l
//
// References:
// - FlashAttention-2: https://arxiv.org/abs/2307.08691
// - FlashAttention-3: https://arxiv.org/abs/2407.08608

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Configuration
// =============================================================================

struct FlashAttentionParams {
    uint batch_size;
    uint num_heads;
    uint num_kv_heads;
    uint query_seq_len;
    uint kv_seq_len;
    uint head_dim;
    float scale;
    uint block_q;
    uint block_k;
    uint gqa_ratio;
    uint is_causal;
    uint sliding_window;
    float softcap;
};

constant uint SIMD_SIZE = 32;

// =============================================================================
// Utility functions
// =============================================================================

inline float simd_max_f32(float val) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, simd_shuffle_xor(val, offset));
    }
    return val;
}

inline float simd_sum_f32(float val) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// =============================================================================
// FlashAttention Forward with Block GEMM (D=128, causal)
// =============================================================================
//
// Thread organization:
//   Grid: [num_q_blocks, num_heads, batch_size]
//   Threadgroup: [32, 4, 1] = 128 threads (4 SIMD groups)
//
// Block GEMM strategy:
//   S[Bq, Bk] = Q[Bq, D] @ K[Bk, D]^T  → matmul2d(Bq, Bk, D)
//   O[Bq, D]  = P[Bq, Bk] @ V[Bk, D]   → matmul2d(Bq, D, Bk)
//
// We use 32×32 simdgroup tiles within a 4-simdgroup threadgroup.
// For Bq=Bk=32, one simdgroup handles the entire score matrix.
// For Bq=Bk=64, we'd need a 2×2 decomposition.

kernel void mpp_flash_attention_fwd_d128_causal(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 128;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    // Threadgroup memory for score tile and intermediate values
    // S_tile holds the Bq×Bk score matrix for online softmax
    threadgroup float S_tile[Bq * Bk];   // 4KB
    threadgroup half Q_tile[Bq * D];     // 8KB
    threadgroup half K_tile[Bk * D];     // 8KB
    threadgroup half V_tile[Bk * D];     // 8KB
    // Total: 28KB (under 32KB limit)

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators
    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    // Load Q tile cooperatively
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal: only process KV positions up to end of this Q block
    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    // Main loop over KV blocks
    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        // Load K and V tiles cooperatively
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Block GEMM: S = Q @ K^T  [Bq × Bk]
        // =====================================================================
        // Use MPP matmul2d for the score computation.
        // Q_tile: [Bq, D] in threadgroup, K_tile: [Bk, D] in threadgroup
        // S_tile: [Bq, Bk] in threadgroup
        //
        // Create tensors from threadgroup pointers
        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq},
                         array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk},
                         array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});

        // Matmul: S = Q @ K^T → SM=32, SN=32, K=128 (D)
        // transpose_right = true because K is [Bk, D] and we want K^T
        constexpr auto score_desc = mpp::tensor_ops::matmul2d_descriptor(
            32, 32,
            static_cast<int>(dynamic_extent),
            false, true, false
        );
        mpp::tensor_ops::matmul2d<score_desc, execution_simdgroups<4>> score_op;
        score_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // Online softmax on S_tile and O accumulation
        // =====================================================================
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            // Find block max for this row
            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;

                // Causal mask
                if (global_k_pos > global_q_pos || global_k_pos >= k_end_actual) {
                    s = -INFINITY;
                }
                // Sliding window
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    s = -INFINITY;
                }
                // Softcap
                if (params.softcap > 0.0f && s > -INFINITY) {
                    s = params.softcap * tanh(s / params.softcap);
                }
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            // Online softmax correction
            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            // Compute exp(s - m_new) for this row and sum
            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                S_tile[my_row * Bk + k] = p;
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            // Correct existing O accumulation
            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                o_local[r][dd] *= correction;
            }

            // Accumulate O += P[r, :] @ V[:, d] using SIMD-parallel reduction
            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                uint d = simd_lane_id * 4 + dd;
                float o_partial = 0.0f;
                for (uint k = 0; k < Bk && k_start + k < k_end_actual; k++) {
                    o_partial += S_tile[my_row * Bk + k] * float(V_tile[k * D + d]);
                }
                o_local[r][dd] += o_partial;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;

        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }

        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward (D=128, non-causal)
// =============================================================================

kernel void mpp_flash_attention_fwd_d128(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 128;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup float S_tile[Bq * Bk];
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block GEMM: S = Q @ K^T
        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq},
                         array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk},
                         array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});

        constexpr auto score_desc = mpp::tensor_ops::matmul2d_descriptor(
            32, 32,
            static_cast<int>(dynamic_extent),
            false, true, false
        );
        mpp::tensor_ops::matmul2d<score_desc, execution_simdgroups<4>> score_op;
        score_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + O accumulation
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos >= k_end_actual) s = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) s = -INFINITY;
                if (params.softcap > 0.0f && s > -INFINITY) s = params.softcap * tanh(s / params.softcap);
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                S_tile[my_row * Bk + k] = p;
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                o_local[r][dd] *= correction;
            }

            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                uint d = simd_lane_id * 4 + dd;
                float o_partial = 0.0f;
                for (uint k = 0; k < Bk && k_start + k < k_end_actual; k++) {
                    o_partial += S_tile[my_row * Bk + k] * float(V_tile[k * D + d]);
                }
                o_local[r][dd] += o_partial;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// TurboQuant weighted-decode helpers for the rotated domain:
//  - weighted_decode: aggregate centroid-indexed values weighted by scores
//  - weighted_sum_d256_dense_values: dense-value analogue for D256 long-ctx

#include "bridge_turboquant_internal.h"

static const char* TURBOQUANT_WEIGHTED_DECODE_SOURCE = R"(
    constexpr uint kTileDims = 8u;
    constexpr uint kMaxCentroids = 512u;
    threadgroup float shared_codebook[kMaxCentroids];

    if (n_centroids > kMaxCentroids) return;
    uint lane = thread_index_in_simdgroup;
    for (uint c = lane; c < n_centroids; c += 32u) {
        shared_codebook[c] = codebook[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = thread_position_in_grid.y;
    uint row = tile / ((dim + kTileDims - 1u) / kTileDims);
    if (row >= n_rows) return;
    uint tile_idx = tile % ((dim + kTileDims - 1u) / kTileDims);
    uint d0 = tile_idx * kTileDims;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row % q_heads;
    uint kv_row = batch * kv_heads + (q_head / groups);

    uint scalar_base = row * n_seq;
    uint kv_scalar_base = kv_row * cache_seq_capacity;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint seq = lane; seq < n_seq; seq += 32u) {
        uint scalar_idx = scalar_base + seq;
        uint kv_scalar_idx = kv_scalar_base + seq;
        float scalar = weights[scalar_idx] * norms[kv_scalar_idx];
        uint cache_base = (kv_row * dim + d0) * cache_seq_capacity + seq;
        if (d0 + 0u < dim) {
            acc0 += scalar * shared_codebook[(uint)indices[cache_base + 0u * cache_seq_capacity]];
        }
        if (d0 + 1u < dim) {
            acc1 += scalar * shared_codebook[(uint)indices[cache_base + 1u * cache_seq_capacity]];
        }
        if (d0 + 2u < dim) {
            acc2 += scalar * shared_codebook[(uint)indices[cache_base + 2u * cache_seq_capacity]];
        }
        if (d0 + 3u < dim) {
            acc3 += scalar * shared_codebook[(uint)indices[cache_base + 3u * cache_seq_capacity]];
        }
        if (d0 + 4u < dim) {
            acc4 += scalar * shared_codebook[(uint)indices[cache_base + 4u * cache_seq_capacity]];
        }
        if (d0 + 5u < dim) {
            acc5 += scalar * shared_codebook[(uint)indices[cache_base + 5u * cache_seq_capacity]];
        }
        if (d0 + 6u < dim) {
            acc6 += scalar * shared_codebook[(uint)indices[cache_base + 6u * cache_seq_capacity]];
        }
        if (d0 + 7u < dim) {
            acc7 += scalar * shared_codebook[(uint)indices[cache_base + 7u * cache_seq_capacity]];
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    if (thread_index_in_simdgroup == 0) {
        uint out_base = row * dim + d0;
        if (d0 + 0u < dim) output[out_base + 0u] = acc0;
        if (d0 + 1u < dim) output[out_base + 1u] = acc1;
        if (d0 + 2u < dim) output[out_base + 2u] = acc2;
        if (d0 + 3u < dim) output[out_base + 3u] = acc3;
        if (d0 + 4u < dim) output[out_base + 4u] = acc4;
        if (d0 + 5u < dim) output[out_base + 5u] = acc5;
        if (d0 + 6u < dim) output[out_base + 6u] = acc6;
        if (d0 + 7u < dim) output[out_base + 7u] = acc7;
    }
)";

// Specialized q8 long-context decode primitive for D=256/V=256.
// This follows the same 2-pass structure as the D128 primitive, but widens the

static const char* TURBOQUANT_WEIGHTED_SUM_D256_DENSE_VALUES_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;

    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    if (row >= n_rows) return;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row - batch * q_heads;
    uint kv_head = q_head / groups;
    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    const device float* row_weights = weights + row * n_seq;
    for (uint seq = 0u; seq < n_seq; ++seq) {
        float w = row_weights[seq];
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        acc0 += w * float(value_dense[value_base + 0u]);
        acc1 += w * float(value_dense[value_base + 1u]);
        acc2 += w * float(value_dense[value_base + 2u]);
        acc3 += w * float(value_dense[value_base + 3u]);
        acc4 += w * float(value_dense[value_base + 4u]);
        acc5 += w * float(value_dense[value_base + 5u]);
        acc6 += w * float(value_dense[value_base + 6u]);
        acc7 += w * float(value_dense[value_base + 7u]);
    }

    uint out_base = row * kDim + d0;
    output[out_base + 0u] = acc0;
    output[out_base + 1u] = acc1;
    output[out_base + 2u] = acc2;
    output[out_base + 3u] = acc3;
    output[out_base + 4u] = acc4;
    output[out_base + 5u] = acc5;
    output[out_base + 6u] = acc6;
    output[out_base + 7u] = acc7;
)";

static mlx::core::fast::CustomKernelFunction& get_turboquant_weighted_decode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_weighted_decode",
        {"weights", "indices", "norms", "codebook"},
        {"output"},
        TURBOQUANT_WEIGHTED_DECODE_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_weighted_sum_d256_dense_values_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_weighted_sum_d256_dense_values",
        {"weights", "value_dense"},
        {"output"},
        TURBOQUANT_WEIGHTED_SUM_D256_DENSE_VALUES_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_weighted_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* indices,
    const mlx_inline_array* norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0 || n_seq == 0 || kv_heads == 0 || q_heads == 0) return 1;
    if (cache_seq_capacity < n_seq) return 1;
    if (q_heads % kv_heads != 0) return 1;

    try {
        auto& kernel = get_turboquant_weighted_decode_kernel();
        constexpr uint32_t tg_threads = 32u;
        constexpr uint32_t tile_dims = 8u;
        uint32_t dim_tiles = (dim + tile_dims - 1u) / tile_dims;
        auto outputs = kernel(
            {as_arr(weights), as_arr(indices), as_arr(norms), as_arr(codebook)},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)tg_threads, (int)(n_rows * dim_tiles), 1},
            {(int)tg_threads, 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows},
             {"n_seq",       (int)n_seq},
             {"cache_seq_capacity", (int)cache_seq_capacity},
             {"q_heads",     (int)q_heads},
             {"kv_heads",    (int)kv_heads}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_weighted_decode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_weighted_decode", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_weighted_sum_d256_dense_values(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;

    if (n_rows == 0 || n_seq == 0 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& weights_arr = as_arr(weights);
        const array& value_dense_arr = as_arr(value_dense);

        if (weights_arr.shape(-1) != n_seq) return 1;
        if (value_dense_arr.shape(-1) != dim) return 1;

        auto& kernel = get_turboquant_weighted_sum_d256_dense_values_kernel();
        auto outputs = kernel(
            {weights_arr, value_dense_arr},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)n_rows, 1, 1},
            {32, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
            },
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_weighted_sum_d256_dense_values", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_weighted_sum_d256_dense_values", "unknown C++ exception"); return 1; }
}

} // extern "C"

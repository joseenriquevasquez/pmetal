// TurboQuant packing helpers: sign-bit packing, q8 key-byte packing
// (transposed + seq-major variants), key+value combined packing,
// signed FWHT, and gather/scatter_last_dim layout helpers.

#include "bridge_turboquant_internal.h"

static const char* TURBOQUANT_PACK_SIGN_BITS_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint packed_idx = thread_position_in_grid.x;
    if (row >= n_rows || packed_idx >= packed_dim) return;

    uint base = row * dim;
    uint start = packed_idx * 32u;
    uint word = 0u;
    for (uint bit = 0u; bit < 32u; ++bit) {
        uint d = start + bit;
        if (d < dim && projected[base + d] >= 0.0f) {
            word |= (1u << bit);
        }
    }
    output[row * packed_dim + packed_idx] = word;
)";

static const char* TURBOQUANT_UNPACK_SIGN_BITS_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;
    if (row >= n_rows || d >= dim) return;

    uint word = packed[row * packed_dim + (d >> 5)];
    output[row * dim + d] = ((word >> (d & 31u)) & 1u) == 0 ? -1.0f : 1.0f;
)";

static const char* TURBOQUANT_PACK_Q8_KEYBYTES_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    output[index_base] = index | sign_bit;
)";

static const char* TURBOQUANT_PACK_Q8_KEYBYTES_SEQ_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    uint output_idx = ((row * cache_seq_capacity) + seq) * dim + d;
    output[output_idx] = index | sign_bit;
)";

static const char* TURBOQUANT_PACK_Q8_KVBYTES_SEQ_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar key_index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    uchar key_byte = key_index | sign_bit;
    uint output_idx = ((row * cache_seq_capacity) + seq) * dim + d;
    ushort value_idx = (ushort)value_indices[output_idx];
    output[output_idx] = ((value_idx & 0xffu) << 8) | (ushort)key_byte;
)";

// WEIGHTED_DECODE: aggregate value centroids directly in the rotated domain.
// This avoids materializing a [row, seq, dim] decoded value tensor before the
// attention reduction.
//
// The value cache is stored in a decode-friendly transposed view:
//   indices: [KvRows, D, S_cap]

static const char* TURBOQUANT_GATHER_LAST_DIM_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint out_d = thread_position_in_grid.x;
    if (out_d >= out_dim || row >= n_rows) return;

    int src_d = positions[out_d];
    output[row * out_dim + out_d] = input[row * full_dim + (uint)src_d];
)";

// SCATTER_LAST_DIM: reassemble mixed regular/outlier sub-vectors into the
// original full-dimensional row layout.
static const char* TURBOQUANT_SCATTER_LAST_DIM_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;
    if (d >= full_dim || row >= n_rows) return;

    float value = 0.0f;
    bool found = false;
    for (uint i = 0u; i < regular_dim; ++i) {
        if ((uint)regular_positions[i] == d) {
            value = regular[row * regular_dim + i];
            found = true;
            break;
        }
    }
    if (!found) {
        for (uint i = 0u; i < outlier_dim; ++i) {
            if ((uint)outlier_positions[i] == d) {
                value = outlier[row * outlier_dim + i];
                break;
            }
        }
    }
    output[row * full_dim + d] = value;
)";

// ── TurboQuant kernel getters ────────────────────────────────────────────────

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_sign_bits_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_sign_bits",
        {"projected"},
        {"output"},
        TURBOQUANT_PACK_SIGN_BITS_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_keybytes_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_keybytes",
        {"indices", "qjl_signs"},
        {"output"},
        TURBOQUANT_PACK_Q8_KEYBYTES_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_keybytes_seq_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_keybytes_seq",
        {"indices", "qjl_signs"},
        {"output"},
        TURBOQUANT_PACK_Q8_KEYBYTES_SEQ_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_kvbytes_seq_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_kvbytes_seq",
        {"indices", "qjl_signs", "value_indices"},
        {"output"},
        TURBOQUANT_PACK_Q8_KVBYTES_SEQ_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_unpack_sign_bits_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_unpack_sign_bits",
        {"packed"},
        {"output"},
        TURBOQUANT_UNPACK_SIGN_BITS_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_gather_last_dim_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_gather_last_dim",
        {"input", "positions"},
        {"output"},
        TURBOQUANT_GATHER_LAST_DIM_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_scatter_last_dim_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_scatter_last_dim",
        {"regular", "outlier", "regular_positions", "outlier_positions"},
        {"output"},
        TURBOQUANT_SCATTER_LAST_DIM_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}


extern "C" {

int mlx_inline_turboquant_pack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* projected,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_sign_bits_kernel();
        auto outputs = kernel(
            {as_arr(projected)},
            {{(int)n_rows, (int)packed_dim}},
            {uint32},
            {(int)packed_dim, (int)n_rows, 1},
            {(int)std::min(packed_dim, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_sign_bits", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_sign_bits", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_keybytes(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_keybytes_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs)},
            {{(int)n_rows, (int)dim, (int)cache_seq_capacity}},
            {uint8},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_keybytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_keybytes_seq_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs)},
            {{(int)n_rows, (int)cache_seq_capacity, (int)dim}},
            {uint8},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes_seq", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes_seq", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_kvbytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* value_indices,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_kvbytes_seq_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs), as_arr(value_indices)},
            {{(int)n_rows, (int)cache_seq_capacity, (int)dim}},
            {uint16},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_kvbytes_seq", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_kvbytes_seq", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_unpack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_unpack_sign_bits_kernel();
        auto outputs = kernel(
            {as_arr(packed)},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)dim, (int)n_rows, 1},
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_unpack_sign_bits", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_unpack_sign_bits", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_signed_fwht_pow2_rows(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* left_signs,
    const mlx_inline_array* right_signs,
    uint32_t                dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_rows == 0 || dim == 0 || (dim & (dim - 1)) != 0) return 1;

    try {
        // Walsh-Hadamard transform is unitary up to a 1/sqrt(dim) factor, which
        // mlx::core::hadamard_transform passes through as the scale argument.
        const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        auto input_f32 = mlx::core::astype(as_arr(input), float32);
        auto input_contig = mlx::core::contiguous(input_f32);
        auto signed_input = mlx::core::multiply(input_contig, as_arr(left_signs));
        auto signed_input_contig = mlx::core::contiguous(signed_input);
        auto transformed = mlx::core::hadamard_transform(signed_input_contig, scale);
        auto output_arr = mlx::core::multiply(transformed, as_arr(right_signs));
        new (out->buf) array(output_arr);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_signed_fwht_pow2_rows", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_signed_fwht_pow2_rows", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_gather_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* positions,
    uint32_t                full_dim,
    uint32_t                out_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (full_dim == 0 || out_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_gather_last_dim_kernel();
        auto outputs = kernel(
            {as_arr(input), as_arr(positions)},
            {{(int)n_rows, (int)out_dim}},
            {float32},
            {(int)out_dim, (int)n_rows, 1},
            {(int)std::min(out_dim, 256u), 1, 1},
            {{"full_dim", (int)full_dim},
             {"out_dim",  (int)out_dim},
             {"n_rows",   (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_gather_last_dim", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_gather_last_dim", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_scatter_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* regular,
    const mlx_inline_array* outlier,
    const mlx_inline_array* regular_positions,
    const mlx_inline_array* outlier_positions,
    uint32_t                full_dim,
    uint32_t                regular_dim,
    uint32_t                outlier_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (full_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_scatter_last_dim_kernel();
        auto outputs = kernel(
            {
                as_arr(regular),
                as_arr(outlier),
                as_arr(regular_positions),
                as_arr(outlier_positions),
            },
            {{(int)n_rows, (int)full_dim}},
            {float32},
            {(int)full_dim, (int)n_rows, 1},
            {(int)std::min(full_dim, 256u), 1, 1},
            {{"full_dim",    (int)full_dim},
             {"regular_dim", (int)regular_dim},
             {"outlier_dim", (int)outlier_dim},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_scatter_last_dim", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_scatter_last_dim", "unknown C++ exception"); return 1; }
}

} // extern "C"

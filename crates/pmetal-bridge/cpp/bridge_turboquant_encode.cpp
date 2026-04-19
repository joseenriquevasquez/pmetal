// TurboQuant encode / decode fused kernels.
// Each kernel does a single-dispatch nearest-centroid search over the
// codebook; caller handles norm + rotation outside.

#include "bridge_turboquant_internal.h"

static const char* TURBOQUANT_ENCODE_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d   = thread_position_in_grid.x;
    if (d >= dim || row >= n_rows) return;

    float x = input[row * dim + d];

    // Nearest-centroid search over the codebook.  This stays on-GPU even for
    // larger q8 codebooks, which is still preferable to materializing the
    // expand_dims/subtract/square/argmin fallback graph.
    float best_dist = (x - codebook[0]) * (x - codebook[0]);
    uint  best_idx  = 0u;
    for (uint c = 1u; c < n_centroids; ++c) {
        float dist = (x - codebook[c]) * (x - codebook[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = c;
        }
    }

    indices[row * dim + d] = best_idx;
)";

// DECODE: for each (row, dim) pair, look up the centroid.
// Grid: (D, N), threadgroup: (min(D,256), 1).
// Output is f32 in the rotated domain.  The caller is responsible for the
// subsequent matmul(rotation) to get back to the original input space.
// Norm rescaling is done OUTSIDE the kernel (by the matmul + multiply in Rust)
// to keep the kernel minimal and avoid a dependency on the norms shape layout.
static const char* TURBOQUANT_DECODE_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d   = thread_position_in_grid.x;
    if (d >= dim || row >= n_rows) return;

    output[row * dim + d] = codebook[indices[row * dim + d]];
)";

// SCORE: for each (row, seq) pair, compute the TurboQuant key score directly
// from compressed centroids + QJL signs without materializing a [row, seq, dim]
// decoded tensor.
//
// The key cache is stored in score-friendly transposed views:
//   indices:   [KvRows, D, S_cap]
//   qjl_signs: [KvRows, ceil(D/32), S_cap]

static mlx::core::fast::CustomKernelFunction& get_turboquant_encode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_encode",
        {"input", "codebook"},
        {"indices"},
        TURBOQUANT_ENCODE_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_decode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_decode",
        {"indices", "codebook"},
        {"output"},
        TURBOQUANT_DECODE_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_encode(
    mlx_inline_array*       out_indices,
    mlx_inline_array*       /*out_norms — unused, kept for ABI symmetry*/,
    const mlx_inline_array* input,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0) return 1;

    const array& inp = as_arr(input);
    const array& cb  = as_arr(codebook);

    // Input must be f32 (normalised+rotated via matmul produces f32).
    // Codebook is always f32.
    try {
        auto& kernel = get_turboquant_encode_kernel();
        auto outputs = kernel(
            {inp, cb},
            {{(int)n_rows, (int)dim}},   // output shape: [N, D]
            {uint8},                      // output dtype: uint8
            {(int)dim, (int)n_rows, 1},  // grid (x=D, y=N, z=1)
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out_indices->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_encode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_encode", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* /*norms — unused, kept for ABI symmetry*/,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0) return 1;

    const array& idx = as_arr(indices);
    const array& cb  = as_arr(codebook);

    try {
        auto& kernel = get_turboquant_decode_kernel();
        auto outputs = kernel(
            {idx, cb},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)dim, (int)n_rows, 1},
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_decode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_decode", "unknown C++ exception"); return 1; }
}

} // extern "C"

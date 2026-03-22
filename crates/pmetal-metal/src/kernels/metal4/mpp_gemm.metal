// mpp_gemm.metal
// Metal 4 GEMM kernels using Metal Performance Primitives (MPP).
//
// Uses mpp::tensor_ops::matmul2d for hardware-accelerated matrix multiplication
// on M5 (Apple10) NAX cores.
//
// Key advantages over Metal 3 manual GEMM:
// - Hardware matrix multiply units (NAX)
// - No explicit threadgroup memory staging needed
// - Static tensor extents eliminate bounds checking for aligned tiles
//
// References:
// - Metal Performance Primitives Programming Guide (Apple, 2026)
// - MLX steel/gemm/nax.h (upstream reference implementation)

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Morton ordering for threadgroup walk order
// =============================================================================

/// Decode a linearized index into 2D (x, y) coordinates via Morton Z-curve.
/// MPP Guide Section 2.3.3: Preserves spatial locality for LLC cache reuse.
inline uint2 morton_decode(uint linear) {
    uint x = 0, y = 0;
    for (uint bit = 0; bit < 16; bit++) {
        y |= ((linear >> (2 * bit))     & 1) << bit;
        x |= ((linear >> (2 * bit + 1)) & 1) << bit;
    }
    return uint2(x, y);
}

// =============================================================================
// Configuration
// =============================================================================

/// Function constants for compile-time specialization
constant bool FC_MORTON    [[function_constant(0)]];  // Use Morton ordering

struct MppGemmParams {
    uint M;           // Output rows
    uint N;           // Output columns
    uint K;           // Reduction dimension
    float alpha;      // Scalar multiplier
    float beta;       // Accumulate multiplier (0 = overwrite, 1 = accumulate)
    uint num_tiles_m; // Total M tiles (for Morton decode)
    uint num_tiles_n; // Total N tiles (for Morton decode)
};

// =============================================================================
// MPP GEMM: D = A @ B^T (fp16 input, fp32 accumulation, fp16 output)
// =============================================================================
//
// A: [M, K] row-major, B: [N, K] row-major (transposed via descriptor)
// D: [M, N] row-major
//
// Thread organization:
//   Grid: (total_tiles, 1, batch) — linearized for Morton ordering
//   Threadgroup: (32 * 4, 1, 1) = 128 threads (4 simdgroups)
//
// MPP Guide recommended starting points for M5 fp16:
//   SM=SN=64 threadgroup tile with 4 cooperating simdgroups
//   K handled by MPP internally (dynamic_extent)
//
// Each threadgroup computes a 64x64 output tile using 4 cooperating simdgroups.

kernel void mpp_gemm_nn_f16(
    device half* A [[buffer(0)]],                 // [M, K] row-major
    device half* B [[buffer(1)]],                 // [N, K] row-major (transposed)
    device half* D [[buffer(2)]],                 // [M, N] row-major
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    // Decode threadgroup position — Morton ordering for cache locality
    uint2 tile;
    if (FC_MORTON) {
        tile = morton_decode(tgid.x);
        if (tile.y >= params.num_tiles_m || tile.x >= params.num_tiles_n) {
            tile.y = tgid.x / params.num_tiles_n;
            tile.x = tgid.x % params.num_tiles_n;
        }
    } else {
        tile.y = tgid.x / params.num_tiles_n;
        tile.x = tgid.x % params.num_tiles_n;
    }

    const int tile_m = (int)(tile.y * 64);
    const int tile_n = (int)(tile.x * 64);
    if (tile_m >= M || tile_n >= N) return;

    // Batch offsets
    const uint batch_idx = tgid.z;
    device half* A_batch = A + batch_idx * M * K;
    device half* B_batch = B + batch_idx * N * K;
    device half* D_batch = D + batch_idx * M * N;

    // Create 2D tensors from device pointers
    // A: [M, K] row-major → tensor dims are [K, M] with strides [1, K]
    // B: [N, K] row-major → tensor dims are [K, N] with strides [1, K]
    // D: [M, N] row-major → tensor dims are [N, M] with strides [1, N]
    auto tA = tensor(A_batch, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B_batch, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tD = tensor(D_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    // MPP guide Section 2.3.5: Use static extents for aligned tiles to avoid
    // bounds checking, dynamic extents for edge tiles.
    constexpr int SM = 64;
    constexpr int SN = 64;

    // Matmul descriptor: 64x64 threadgroup tile with 4 cooperating simdgroups
    // K = dynamic_extent → MPP handles full K-dim internally
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        SM,                                     // threadgroup M tile
        SN,                                     // threadgroup N tile
        static_cast<int>(dynamic_extent),       // K: let MPP handle internally
        false,                                  // transpose_left
        true,                                   // transpose_right (B is [N,K])
        false                                   // relaxed_precision
    );

    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    // Slice to this threadgroup's 64x64 tile
    // Dynamic extents with bounds checking — MPP handles edge tiles safely.
    // TODO: Add static_slice path for aligned interior tiles (MPP Guide 2.3.5)
    auto sliceA = tA.slice(0, tile_m);
    auto sliceB = tB.slice(0, tile_n);
    auto sliceD = tD.slice(tile_n, tile_m);
    op.run(sliceA, sliceB, sliceD);
}

// =============================================================================
// MPP GEMM: D = A @ B^T (fp32, for weight gradients)
// =============================================================================

kernel void mpp_gemm_nn_f32(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    uint2 tile;
    if (FC_MORTON) {
        tile = morton_decode(tgid.x);
        if (tile.y >= params.num_tiles_m || tile.x >= params.num_tiles_n) {
            tile.y = tgid.x / params.num_tiles_n;
            tile.x = tgid.x % params.num_tiles_n;
        }
    } else {
        tile.y = tgid.x / params.num_tiles_n;
        tile.x = tgid.x % params.num_tiles_n;
    }

    const int tile_m = (int)(tile.y * 64);
    const int tile_n = (int)(tile.x * 64);
    if (tile_m >= M || tile_n >= N) return;

    const uint batch_idx = tgid.z;
    device float* A_batch = A + batch_idx * M * K;
    device float* B_batch = B + batch_idx * N * K;
    device float* D_batch = D + batch_idx * M * N;

    auto tA = tensor(A_batch, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B_batch, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tD = tensor(D_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64,
        static_cast<int>(dynamic_extent),
        false, true, false
    );

    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    // Slice to this threadgroup's 64x64 tile
    // Dynamic extents with bounds checking — MPP handles edge tiles safely.
    // TODO: Add static_slice path for aligned interior tiles (MPP Guide 2.3.5)
    auto sliceA = tA.slice(0, tile_m);
    auto sliceB = tB.slice(0, tile_n);
    auto sliceD = tD.slice(tile_n, tile_m);
    op.run(sliceA, sliceB, sliceD);
}

// =============================================================================
// MPP GEMM: D = alpha * A @ B^T + beta * C (fp16, with accumulation)
// =============================================================================
//
// Postfix fusion: when beta != 0, loads C and fuses the scaled addition
// via cooperative tensor operations in register space, avoiding a separate
// elementwise kernel pass.

kernel void mpp_gemm_accumulate_f16(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],                 // Existing output (for beta accumulation)
    device half* D [[buffer(3)]],                 // Output (can alias C)
    constant MppGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    uint2 tile;
    if (FC_MORTON) {
        tile = morton_decode(tgid.x);
        if (tile.y >= params.num_tiles_m || tile.x >= params.num_tiles_n) {
            tile.y = tgid.x / params.num_tiles_n;
            tile.x = tgid.x % params.num_tiles_n;
        }
    } else {
        tile.y = tgid.x / params.num_tiles_n;
        tile.x = tgid.x % params.num_tiles_n;
    }

    const int tile_m = (int)(tile.y * 64);
    const int tile_n = (int)(tile.x * 64);
    if (tile_m >= M || tile_n >= N) return;

    const uint batch_idx = tgid.z;
    device half* A_batch = A + batch_idx * M * K;
    device half* B_batch = B + batch_idx * N * K;
    device half* D_batch = D + batch_idx * M * N;

    auto tA = tensor(A_batch, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B_batch, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tD = tensor(D_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    // Use multiply_accumulate mode with K-loop for postfix fusion
    constexpr int BK = 128;  // MPP Guide: BK=128 recommended on M5
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64, BK,
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    // Slice to this threadgroup's tile
    auto sliceA = tA.slice(0, tile_m);
    auto sliceB = tB.slice(0, tile_n);
    auto sliceD = tD.slice(tile_n, tile_m);

    // Get cooperative tensor for accumulation
    auto rD = op.get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), float>();

    // Initialize from C if beta != 0 (postfix fusion with existing output)
    if (params.beta != 0.0f) {
        device half* C_batch = C + batch_idx * M * N;
        auto tC = tensor(C_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});
        auto sliceC = tC.slice(tile_n, tile_m);

        // Load C into cooperative tensor via threadgroup-scoped tensor
        // This is postfix fusion: we pre-load the accumulator before matmul
        auto oC = op.get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
        oC.load(sliceC);

        // Scale by beta and initialize accumulator
        for (int i = 0; i < rD.get_capacity(); i++) {
            rD[i] = float(oC[i]) * params.beta;
        }
    }

    // K-loop with accumulation barriers (MPP Guide Section 2.3.4)
    const int num_k = (K + BK - 1) / BK;
    for (int kk = 0; kk < num_k; kk++) {
        threadgroup_barrier(mem_flags::mem_none);

        auto tkA = sliceA.slice(kk * BK, 0);
        auto tkB = sliceB.slice(kk * BK, 0);

        op.run(tkA, tkB, rD);
    }

    // Postfix: apply alpha scaling on the cooperative tensor (register space)
    if (params.alpha != 1.0f) {
        for (int i = 0; i < rD.get_capacity(); i++) {
            rD[i] = rD[i] * params.alpha;
        }
    }

    // Store: convert f32 accumulator to f16 output via cooperative tensor store
    auto oD = op.get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
    for (int i = 0; i < rD.get_capacity(); i++) {
        oD[i] = half(rD[i]);
    }
    oD.store(sliceD);
}

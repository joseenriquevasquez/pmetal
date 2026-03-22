// mpp_fused_swiglu.metal
// Metal 4 Fused SwiGLU MLP using MPP matmul2d.
//
// Replaces the Metal 3 fused_swiglu kernels which use per-element dot products
// (SIMD-strided reduction per output element). MPP provides hardware matrix
// multiply for the projections, with SwiGLU activation applied as postfix
// fusion on the cooperative tensor result.
//
// Computes: output = silu(x @ gate_W^T) * (x @ up_W^T)
//
// Single kernel launch combines:
//   1. gate = x @ gate_weight^T
//   2. up   = x @ up_weight^T
//   3. output = silu(gate) * up
//
// For LoRA: adds scale * (x @ A^T) @ B^T to each projection.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct FusedSwiGLUParams {
    uint batch_size;
    uint hidden_size;
    uint intermediate_size;
    uint lora_rank;
    float lora_scale;
};

inline float silu(float x) {
    return x / (1.0f + metal::exp(-x));
}

// =============================================================================
// MPP Fused SwiGLU Forward (no LoRA)
// =============================================================================
//
// Strategy: Compute gate and up projections using matmul2d, then fuse
// SwiGLU activation. Each threadgroup handles a tile of the output.
//
// For batch_size=1 (decode), this is memory-bound — matmul2d still helps
// because it avoids the overhead of per-element SIMD reduction loops.
//
// For batch_size>1 (prefill/training), this is compute-bound and matmul2d
// provides significant speedup via hardware MMA units.

kernel void mpp_fused_swiglu_forward_f16(
    device half* input [[buffer(0)]],
    device half* gate_weight [[buffer(1)]],
    device half* up_weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant FusedSwiGLUParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;

    // Grid: [num_intermediate_tiles, num_batch_tiles, 1]
    // Each threadgroup computes a 64×64 tile of the output
    // But for SwiGLU we need BOTH gate and up for the same output positions,
    // so each threadgroup computes gate[B_tile, I_tile] and up[B_tile, I_tile]
    // then fuses silu(gate) * up.

    const int BM = 64;  // batch tile
    const int BN = 64;  // intermediate tile

    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    // Create tensors
    // input: [B, H] row-major
    auto tX = tensor(input, dextents<int, 2>{H, B}, array<int, 2>{1, H});

    // gate_weight: [I, H] row-major → need transpose
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});

    // Threadgroup memory for gate and up results
    // We need both to compute silu(gate) * up
    threadgroup float gate_tile[BM * BN]; // 16KB
    threadgroup float up_tile[BM * BN];   // 16KB
    // Total: 32KB (at threadgroup limit)

    auto tGate = tensor((threadgroup float*)gate_tile,
                        dextents<int, 2>{BN, BM},
                        array<int, 2>{1, BN});
    auto tUp = tensor((threadgroup float*)up_tile,
                      dextents<int, 2>{BN, BM},
                      array<int, 2>{1, BN});

    // Slice to this tile
    auto sliceX = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);
    auto sliceGate = tGate.slice(0, 0);
    auto sliceUp = tUp.slice(0, 0);

    // Gate projection: gate_tile = X @ gate_W^T
    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64,
        static_cast<int>(dynamic_extent),
        false, true, false
    );
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroups<4>> proj_op;

    proj_op.run(sliceX, sliceGW, sliceGate);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up projection: up_tile = X @ up_W^T
    proj_op.run(sliceX, sliceUW, sliceUp);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fuse SwiGLU: output = silu(gate) * up
    // Each thread handles a subset of the tile
    uint total_threads = 128; // 4 simdgroups × 32 lanes
    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint tile_b_size = min((uint)BM, params.batch_size - (uint)tile_b);
    uint tile_i_size = min((uint)BN, params.intermediate_size - (uint)tile_i);

    for (uint idx = linear_tid; idx < tile_b_size * tile_i_size; idx += total_threads) {
        uint m = idx / tile_i_size;
        uint n = idx % tile_i_size;
        float g = gate_tile[m * BN + n];
        float u = up_tile[m * BN + n];
        uint global_b = (uint)tile_b + m;
        uint global_i = (uint)tile_i + n;
        output[global_b * params.intermediate_size + global_i] = half(silu(g) * u);
    }
}

// =============================================================================
// MPP Fused SwiGLU Forward (fp32)
// =============================================================================

kernel void mpp_fused_swiglu_forward_f32(
    device float* input [[buffer(0)]],
    device float* gate_weight [[buffer(1)]],
    device float* up_weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant FusedSwiGLUParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int B = (int)params.batch_size;
    const int H = (int)params.hidden_size;
    const int I = (int)params.intermediate_size;

    const int BM = 64;
    const int BN = 64;
    const int tile_b = (int)(tgid.y * BM);
    const int tile_i = (int)(tgid.x * BN);
    if (tile_b >= B || tile_i >= I) return;

    auto tX = tensor(input, dextents<int, 2>{H, B}, array<int, 2>{1, H});
    auto tGW = tensor(gate_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});
    auto tUW = tensor(up_weight, dextents<int, 2>{H, I}, array<int, 2>{1, H});

    threadgroup float gate_tile[BM * BN];
    threadgroup float up_tile[BM * BN];

    auto tGate = tensor((threadgroup float*)gate_tile,
                        dextents<int, 2>{BN, BM},
                        array<int, 2>{1, BN});
    auto tUp = tensor((threadgroup float*)up_tile,
                      dextents<int, 2>{BN, BM},
                      array<int, 2>{1, BN});

    auto sliceX = tX.slice(0, tile_b);
    auto sliceGW = tGW.slice(0, tile_i);
    auto sliceUW = tUW.slice(0, tile_i);

    constexpr auto proj_desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64,
        static_cast<int>(dynamic_extent),
        false, true, false
    );
    mpp::tensor_ops::matmul2d<proj_desc, execution_simdgroups<4>> proj_op;

    auto sliceGate = tGate.slice(0, 0);
    auto sliceUp = tUp.slice(0, 0);

    proj_op.run(sliceX, sliceGW, sliceGate);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    proj_op.run(sliceX, sliceUW, sliceUp);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint tile_b_size = min((uint)BM, params.batch_size - (uint)tile_b);
    uint tile_i_size = min((uint)BN, params.intermediate_size - (uint)tile_i);

    for (uint idx = linear_tid; idx < tile_b_size * tile_i_size; idx += 128) {
        uint m = idx / tile_i_size;
        uint n = idx % tile_i_size;
        float g = gate_tile[m * BN + n];
        float u = up_tile[m * BN + n];
        uint global_b = (uint)tile_b + m;
        uint global_i = (uint)tile_i + n;
        output[global_b * params.intermediate_size + global_i] = silu(g) * u;
    }
}

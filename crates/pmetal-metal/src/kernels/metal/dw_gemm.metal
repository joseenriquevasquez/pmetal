//  dw_gemm.metal
//  Tiled fp32 SGEMM for weight gradient accumulation in ANE training.
//
//  Computes: C = alpha * A @ B^T + beta * C
//
//  Replaces per-layer cblas_sgemm dispatches on a CPU worker thread.
//  All 7 per-layer dW GEMMs (+ embedding grad GEMM) are encoded into a single
//  Metal command buffer, executing on the GPU while the ANE handles dx propagation.
//
//  Tile: BM=64, BN=64, BK=16  (8KB threadgroup memory for A+B staging)
//  Threadgroup: 16x16 = 256 threads, each accumulates a 4x4 sub-tile in registers.
//  B is stored [N, K] row-major; we load it transposed into [BK, BN] staging.

#include <metal_stdlib>
using namespace metal;

// Tile dimensions
constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;

// Threads per tile dimension (BM/4 = 16, BN/4 = 16)
constant uint TM = 4;   // rows per thread
constant uint TN = 4;   // cols per thread

struct DwGemmParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

// Threadgroup memory layout: A_stage [BM][BK] + B_stage [BK][BN]
// Total = 64*16 + 16*64 = 2048 floats = 8KB

kernel void dw_gemm_accum(
    device const float* A         [[buffer(0)]],   // [M, K] row-major
    device const float* B         [[buffer(1)]],   // [N, K] row-major (transposed on read)
    device float*       C         [[buffer(2)]],   // [M, N] row-major, read-modify-write
    constant DwGemmParams& params [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tid   [[thread_position_in_threadgroup]]
) {
    // Shared memory for cooperative loading
    threadgroup float A_stage[BM * BK];  // [BM, BK]
    threadgroup float B_stage[BK * BN];  // [BK, BN]

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    // This threadgroup handles tile [tg_id.y * BM .. +BM, tg_id.x * BN .. +BN]
    const uint tile_row = tg_id.y * BM;
    const uint tile_col = tg_id.x * BN;

    // Thread position within threadgroup (16x16 = 256 threads)
    const uint local_row = tid.y;  // 0..15
    const uint local_col = tid.x;  // 0..15

    // Register accumulators for TM x TN sub-tile
    float acc[TM][TN] = {{0.0f}};

    // Number of K-tiles
    const uint num_k_tiles = (K + BK - 1) / BK;

    // Linear thread index for cooperative loading
    const uint linear_tid = tid.y * 16 + tid.x; // 0..255

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        const uint k_base = kt * BK;

        // ---- Cooperative load A_stage [BM x BK] ----
        // 256 threads load 64*16 = 1024 elements → 4 elements per thread
        for (uint i = linear_tid; i < BM * BK; i += 256) {
            uint row = i / BK;
            uint col = i % BK;
            uint global_row = tile_row + row;
            uint global_col = k_base + col;
            if (global_row < M && global_col < K) {
                A_stage[row * BK + col] = A[global_row * K + global_col];
            } else {
                A_stage[row * BK + col] = 0.0f;
            }
        }

        // ---- Cooperative load B_stage [BK x BN] (B is [N,K], transposed) ----
        // B[n, k] → B_stage[k_local, n_local] where k_local = k - k_base, n_local = n - tile_col
        for (uint i = linear_tid; i < BK * BN; i += 256) {
            uint k_local = i / BN;
            uint n_local = i % BN;
            uint global_n = tile_col + n_local;
            uint global_k = k_base + k_local;
            if (global_n < N && global_k < K) {
                // B is [N, K] row-major: element [n, k] is at B[n * K + k]
                B_stage[k_local * BN + n_local] = B[global_n * K + global_k];
            } else {
                B_stage[k_local * BN + n_local] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Register-level FMA: each thread computes a TM x TN sub-tile ----
        // Thread (local_row, local_col) handles rows [local_row*TM .. +TM], cols [local_col*TN .. +TN]
        for (uint kk = 0; kk < BK; kk++) {
            // Load TM elements from A_stage column kk
            float a_reg[TM];
            for (uint m = 0; m < TM; m++) {
                a_reg[m] = A_stage[(local_row * TM + m) * BK + kk];
            }

            // Load TN elements from B_stage row kk
            float b_reg[TN];
            for (uint n = 0; n < TN; n++) {
                b_reg[n] = B_stage[kk * BN + local_col * TN + n];
            }

            // Outer product accumulation
            for (uint m = 0; m < TM; m++) {
                for (uint n = 0; n < TN; n++) {
                    acc[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write back: C = alpha * acc + beta * C ----
    const float alpha = params.alpha;
    const float beta  = params.beta;

    for (uint m = 0; m < TM; m++) {
        uint global_row = tile_row + local_row * TM + m;
        if (global_row >= M) continue;

        for (uint n = 0; n < TN; n++) {
            uint global_col = tile_col + local_col * TN + n;
            if (global_col >= N) continue;

            uint idx = global_row * N + global_col;
            float c_val = beta * C[idx] + alpha * acc[m][n];
            C[idx] = c_val;
        }
    }
}

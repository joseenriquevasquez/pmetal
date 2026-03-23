#include <metal_stdlib>

using namespace metal;

// Simple vectorized copy kernel used to estimate sustained unified-memory
// throughput on the current GPU. This is intentionally narrow in scope: it is
// not part of the model runtime, only a startup/device-characterization probe.
kernel void bandwidth_probe_f32(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant uint& vec_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < vec_count) {
        dst[gid] = src[gid];
    }
}

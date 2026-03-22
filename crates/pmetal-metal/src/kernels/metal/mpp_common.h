// mpp_common.h
// Shared utilities for Metal Performance Primitives (Metal 4 / M5+) kernels.
//
// This header provides:
// - Morton ordering for threadgroup walk order (LLC cache locality)
// - Common MPP includes and namespace imports

#pragma once

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Morton Ordering (Z-order curve)
// =============================================================================
//
// Maps a linearized threadgroup index to 2D tile coordinates using bit
// interleaving. This ensures threadgroups that are numerically adjacent
// cover a square region of the output matrix, maximizing L2 cache reuse
// of shared A-row and B-column data across cores.
//
// Usage:
//   1. Dispatch grid as MTLSize(totalThreadgroups, 1, 1) — linearized
//   2. At kernel start: uint2 tile = morton_decode(tgid.x);
//   3. tile.x = column tile index, tile.y = row tile index
//
// MPP Programming Guide Section 2.3.3: "The Morton ordering is a common
// example of a space-filling two-dimensional curve that preserves locality."

/// Decode a linearized index into 2D (x, y) coordinates via Morton curve.
inline uint2 morton_decode(uint linear) {
    uint x = 0, y = 0;
    for (uint bit = 0; bit < 16; bit++) {
        y |= ((linear >> (2 * bit))     & 1) << bit;
        x |= ((linear >> (2 * bit + 1)) & 1) << bit;
    }
    return uint2(x, y);
}

/// Encode 2D (x, y) coordinates into a linearized Morton index.
inline uint morton_encode(uint x, uint y) {
    uint z = 0;
    for (uint bit = 0; bit < 16; bit++) {
        z |= ((y >> bit) & 1) << (2 * bit);
        z |= ((x >> bit) & 1) << (2 * bit + 1);
    }
    return z;
}

/// Map a linear threadgroup index to 2D tile coordinates, clamped to grid bounds.
/// Use when the total number of tiles isn't a perfect square.
inline uint2 morton_decode_clamped(uint linear, uint num_tiles_x, uint num_tiles_y) {
    uint total = num_tiles_x * num_tiles_y;
    if (linear >= total) return uint2(num_tiles_x, num_tiles_y); // OOB sentinel

    uint2 tile = morton_decode(linear);

    // Morton can produce coordinates outside the grid for non-power-of-2 grids.
    // Fall back to row-major for out-of-range Morton coordinates.
    if (tile.x >= num_tiles_x || tile.y >= num_tiles_y) {
        tile.y = linear / num_tiles_x;
        tile.x = linear % num_tiles_x;
    }

    return tile;
}

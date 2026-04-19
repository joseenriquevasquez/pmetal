// Internal helpers shared by the split bridge_turboquant_*.cpp files.
// Not part of the public C interface.
#pragma once

#include "bridge_internal.h"

// Allow PMETAL_TQ_Q8_2PASS_BLOCKS to override the per-family 2-pass block
// count at runtime. Parsed value is clamped to [32, 1024] and rounded down
// to the nearest multiple of 32. Returns `fallback` when the env var is
// unset, empty, or malformed.
static inline uint32_t turboquant_q8_2pass_blocks_override_or(uint32_t fallback) {
    const char* env = std::getenv("PMETAL_TQ_Q8_2PASS_BLOCKS");
    if (!env || !*env) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(env, &end, 10);
    if (end == env || *end != '\0') return fallback;
    if (parsed < 32ul) parsed = 32ul;
    if (parsed > 1024ul) parsed = 1024ul;
    parsed = (parsed / 32ul) * 32ul;
    return parsed ? static_cast<uint32_t>(parsed) : fallback;
}

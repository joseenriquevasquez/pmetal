// Shared typedefs, lifecycle, interop, error reporting.
// Every other bridge/*.h pulls this in for the `mlx_inline_array` typedef.

#ifndef MLX_INLINE_BRIDGE_COMMON_H
#define MLX_INLINE_BRIDGE_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Size/alignment of mlx::core::array — queried at build time via mlx_inline_array_size()
// Conservative upper bound; static_assert in .cpp verifies at compile time.
#define MLX_ARRAY_SIZE 128
#define MLX_ARRAY_ALIGN 8

// Stack-allocated array — NO heap allocation per op.
// Rust creates this on the stack, C++ placement-news into buf.
typedef struct {
    _Alignas(MLX_ARRAY_ALIGN) unsigned char buf[MLX_ARRAY_SIZE];
} mlx_inline_array;

// ── Lifecycle ─────────────────────────────────────────────────────────────
void mlx_inline_init_empty(mlx_inline_array* dst);
void mlx_inline_init_copy(mlx_inline_array* dst, const mlx_inline_array* src);
void mlx_inline_init_move(mlx_inline_array* dst, mlx_inline_array* src);
void mlx_inline_destroy(mlx_inline_array* a);

// ── Interop with legacy mlx_array handles ─────────────────────────────────
void mlx_inline_from_handle(mlx_inline_array* dst, void* handle_ctx);
void* mlx_inline_to_handle(const mlx_inline_array* src);

// ── Size query (for Rust build-time verification) ─────────────────────────
size_t mlx_inline_array_size(void);
size_t mlx_inline_array_align(void);

// ── Error reporting ───────────────────────────────────────────────────────
//
// Every `mlx_inline_*` entry point that can throw a C++ exception writes
// its failure into a thread-local error slot before returning. Rust callers
// can read the slot via these three functions and must copy the message
// string before issuing another bridge call on the same thread.

// Returns 0 on no error, 1 on a caught std::exception, 2 on an unknown
// (non-std) C++ exception. Thread-local.
int32_t pmetal_bridge_last_error_code(void);

// Returns a NUL-terminated message describing the most recent failure on
// this thread. Pointer is valid until the next bridge call on the same
// thread. Always returns a non-NULL pointer (empty string when no error).
const char* pmetal_bridge_last_error_message(void);

// Manually clears the thread-local error slot. Normally unnecessary —
// every successful bridge op clears it automatically.
void pmetal_bridge_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif

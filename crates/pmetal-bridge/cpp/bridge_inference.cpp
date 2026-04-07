// Additional inference operations, sampling, and memory instrumentation.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"
#include <unordered_set>
#include <typeinfo>

extern "C" {

void mlx_inline_concatenate_2(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b, int axis) {
    try { new (dst->buf) array(mlx::core::concatenate({as_arr(a), as_arr(b)}, axis)); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_softplus(mlx_inline_array* dst, const mlx_inline_array* a) {
    // softplus(x) = log(1 + exp(x)) = log1p(exp(x))
    auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::log1p(mlx::core::exp(x)));
}

void mlx_inline_where(mlx_inline_array* dst, const mlx_inline_array* condition,
                       const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::where(as_arr(condition), as_arr(a), as_arr(b)));
}

void mlx_inline_maximum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::maximum(as_arr(a), as_arr(b)));
}

void mlx_inline_zeros(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::zeros(
        mlx::core::Shape(shape, shape + ndim), dtype_from_int(dtype)));
}

void mlx_inline_ones(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::ones(
        mlx::core::Shape(shape, shape + ndim), dtype_from_int(dtype)));
}

void mlx_inline_slice(mlx_inline_array* dst, const mlx_inline_array* a,
                       const int* start, const int* stop, int ndim) {
    try {
        new (dst->buf) array(mlx::core::slice(
            as_arr(a),
            mlx::core::Shape(start, start + ndim),
            mlx::core::Shape(stop, stop + ndim)));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_slice_set(mlx_inline_array* dst, const mlx_inline_array* a,
                            const mlx_inline_array* value,
                            const int* start, const int* stop, int ndim) {
    try {
        new (dst->buf) array(mlx::core::slice_update(
            as_arr(a), as_arr(value),
            mlx::core::Shape(start, start + ndim),
            mlx::core::Shape(stop, stop + ndim)));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_repeat(mlx_inline_array* dst, const mlx_inline_array* a, int repeats, int axis) {
    try {
        new (dst->buf) array(mlx::core::repeat(as_arr(a), repeats, axis));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_squeeze(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    try {
        new (dst->buf) array(mlx::core::squeeze(as_arr(a), axis));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_expand_dims(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    try {
        new (dst->buf) array(mlx::core::expand_dims(as_arr(a), axis));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_transpose_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                                 const int* axes, int ndim) {
    new (dst->buf) array(mlx::core::transpose(
        as_arr(a), std::vector<int>(axes, axes + ndim)));
}

void mlx_inline_cumsum(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::cumsum(as_arr(a), axis));
}

void mlx_inline_log(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::log(as_arr(a)));
}

void mlx_inline_tril(mlx_inline_array* dst, const mlx_inline_array* a, int k) {
    new (dst->buf) array(mlx::core::tril(as_arr(a), k));
}

void mlx_inline_index(mlx_inline_array* dst, const mlx_inline_array* a,
                       const mlx_inline_array* indices) {
    // take(a, indices) — flat gather over all elements (no axis specified)
    new (dst->buf) array(mlx::core::take(as_arr(a), as_arr(indices)));
}

void mlx_inline_softmax_precise(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::softmax(as_arr(a), axis, /*precise=*/true));
}

void mlx_inline_sdpa_with_mask(mlx_inline_array* dst,
                                 const mlx_inline_array* q, const mlx_inline_array* k,
                                 const mlx_inline_array* v, float scale,
                                 const mlx_inline_array* mask) {
    try {
        auto mask_opt = mask
            ? std::optional<array>(as_arr(mask))
            : std::optional<array>(std::nullopt);
        new (dst->buf) array(mlx::core::fast::scaled_dot_product_attention(
            as_arr(q), as_arr(k), as_arr(v), scale, /*mask_mode=*/"", mask_opt));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_eval_2(mlx_inline_array* a, mlx_inline_array* b) {
    try { mlx::core::eval({as_arr(a), as_arr(b)}); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); }
}

void mlx_inline_eval_many(mlx_inline_array** arrays, int count) {
    try {
        std::vector<array> arrs;
        arrs.reserve(count);
        for (int i = 0; i < count; ++i) {
            arrs.push_back(as_arr(arrays[i]));
        }
        mlx::core::eval(std::move(arrs));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); }
}

void mlx_inline_async_eval_many(mlx_inline_array** arrays, int count) {
    try {
        std::vector<array> arrs;
        arrs.reserve(count);
        for (int i = 0; i < count; ++i) {
            arrs.push_back(as_arr(arrays[i]));
        }
        mlx::core::async_eval(std::move(arrs));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); }
}

void mlx_inline_quantized_matmul(mlx_inline_array* dst,
                                   const mlx_inline_array* x, const mlx_inline_array* w,
                                   const mlx_inline_array* scales, const mlx_inline_array* biases,
                                   bool transpose, int group_size, int bits) {
    try {
        auto biases_opt = biases
            ? std::optional<array>(as_arr(biases))
            : std::optional<array>(std::nullopt);
        new (dst->buf) array(mlx::core::quantized_matmul(
            as_arr(x), as_arr(w), as_arr(scales), biases_opt,
            transpose, group_size, bits));
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] quantized_matmul: %s\n", e.what());
        new (dst->buf) array(0.0f);
    }
}

void mlx_inline_gather_qmm(mlx_inline_array* dst,
                              const mlx_inline_array* x, const mlx_inline_array* w,
                              const mlx_inline_array* scales, const mlx_inline_array* biases,
                              const mlx_inline_array* lhs_indices, const mlx_inline_array* rhs_indices,
                              bool transpose, int group_size, int bits, bool sorted) {
    try {
        auto biases_opt = biases
            ? std::optional<array>(as_arr(biases))
            : std::optional<array>(std::nullopt);
        auto lhs_opt = lhs_indices
            ? std::optional<array>(as_arr(lhs_indices))
            : std::optional<array>(std::nullopt);
        auto rhs_opt = rhs_indices
            ? std::optional<array>(as_arr(rhs_indices))
            : std::optional<array>(std::nullopt);
        new (dst->buf) array(mlx::core::gather_qmm(
            as_arr(x), as_arr(w), as_arr(scales), biases_opt,
            lhs_opt, rhs_opt,
            transpose, group_size, bits,
            /*mode=*/"affine", sorted));
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

// GDN Metal kernel source — fuses the entire recurrence into one Metal dispatch.
// Matches the Rust GDN_KERNEL_SOURCE from pmetal-mlx/src/kernels/gated_delta.rs.

void mlx_inline_take_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis) {
    new (dst->buf) array(mlx::core::take(as_arr(a), as_arr(indices), axis));
}

void mlx_inline_kv_cache_append(mlx_inline_array* dst,
    const mlx_inline_array* cached, const mlx_inline_array* new_kv, int axis) {
    new (dst->buf) array(mlx::core::concatenate({as_arr(cached), as_arr(new_kv)}, axis));
}

void mlx_inline_async_eval_arr(const mlx_inline_array* a) {
    mlx::core::async_eval({as_arr(a)});
}

// ── Sampling ops ──

void mlx_inline_argmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::argmax(as_arr(a), axis));
}

void mlx_inline_argmin(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::argmin(as_arr(a), axis));
}

void mlx_inline_abs(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::abs(as_arr(a)));
}

void mlx_inline_logsumexp(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::logsumexp(as_arr(a), axis, keepdims));
}

void mlx_inline_categorical(mlx_inline_array* dst, const mlx_inline_array* logits) {
    new (dst->buf) array(mlx::core::random::categorical(as_arr(logits)));
}

// mlx_inline_gdn_metal_step lives in bridge_native.cpp (uses get_gdn_kernel)

void mlx_inline_arange(mlx_inline_array* dst, int n, int dtype) {
    new (dst->buf) array(mlx::core::arange(0, n, dtype_from_int(dtype)));
}

int mlx_inline_load_safetensors_key(mlx_inline_array* dst, const char* path, const char* key) {
    try {
        auto [arrays, metadata] = mlx::core::load_safetensors(std::string(path));
        auto it = arrays.find(std::string(key));
        if (it == arrays.end()) return 1;
        new (dst->buf) array(it->second);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_key(%s, %s): %s\n", path, key, e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_key(%s, %s): unknown exception\n", path, key);
        return 1;
    }
}

// Load ALL tensors from a safetensors file in a single parse.
// Each entry gets a strdup'd key and a placement-new'd array in the caller buffers.
// Returns the number of entries written, or -1 on error.
int mlx_inline_load_safetensors_all(
        const char* path,
        char** key_buf,
        mlx_inline_array* arr_buf,
        int max_entries) {
    try {
        auto [arrays, metadata] = mlx::core::load_safetensors(std::string(path));
        int count = 0;
        for (auto& [key, arr] : arrays) {
            if (count >= max_entries) break;
            key_buf[count] = strdup(key.c_str());
            new (arr_buf[count].buf) array(arr);
            count++;
        }
        return count;
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_all(%s): %s\n", path, e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_all(%s): unknown exception\n", path);
        return -1;
    }
}

void mlx_inline_free_key_strings(char** keys, int count) {
    for (int i = 0; i < count; ++i) {
        free(keys[i]);
    }
}

void mlx_inline_from_i32_slice(mlx_inline_array* dst, const int32_t* data, int len) {
    new (dst->buf) array(data, {len}, mlx::core::int32);
}

void mlx_inline_detach(mlx_inline_array* a) {
    as_arr(a).detach();
}

// ── Metal memory instrumentation ──

size_t mlx_inline_get_active_memory(void) {
    return mlx::core::get_active_memory();
}

size_t mlx_inline_get_cache_memory(void) {
    return mlx::core::get_cache_memory();
}

size_t mlx_inline_get_peak_memory(void) {
    return mlx::core::get_peak_memory();
}

void mlx_inline_reset_peak_memory(void) {
    mlx::core::reset_peak_memory();
}


} // extern "C"

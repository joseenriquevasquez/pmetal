// Fused compiled operations — @mx.compile(shapeless=True) equivalents.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"

// ============================================================================
// Fused compiled ops — matching Python's @mx.compile(shapeless=True)
// Each creates a compiled closure on first call, caches it, and replays.
// This produces a single Compiled graph node instead of N separate nodes.
// Must be outside extern "C" for C++ template/lambda support.
// ============================================================================

using namespace mlx::core;
using CompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;

// Heap-allocate and intentionally leak compiled functions.  Function-local
// static CompiledFn objects destruct at program exit via __cxa_atexit, calling
// compile_erase() into libmlx.dylib's CompilerCache.  Cross-DSO atexit
// ordering is unspecified — the CompilerCache may already be torn down,
// causing SIGSEGV.  Heap-leaked objects are never destroyed, avoiding the
// problem entirely (the OS reclaims all memory at process exit).
static CompiledFn* make_compiled(CompiledFn fn) {
    return new CompiledFn(mlx::core::compile(std::move(fn), /*shapeless=*/true));
}

// shapeless=false: works with ALL primitives (Split, CustomKernel, etc.)
// but only replays the tape when input shapes match the first trace.
// Perfect for T=1 decode where shapes are always the same.
static CompiledFn* make_compiled_fixed(CompiledFn fn) {
    return new CompiledFn(mlx::core::compile(std::move(fn), /*shapeless=*/false));
}

extern "C" {

void mlx_inline_fused_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up) {
    try {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& g = inputs[0];
                auto& u = inputs[1];
                return {multiply(multiply(g, sigmoid(g)), u)};
            });
        auto result = (*compiled)({as_arr(gate), as_arr(up)});
        new (dst->buf) array(result[0]);
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_fused_silu(mlx_inline_array* dst, const mlx_inline_array* x) {
    try {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& x = inputs[0];
                return {multiply(x, sigmoid(x))};
            });
        auto result = (*compiled)({as_arr(x)});
        new (dst->buf) array(result[0]);
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_fused_compute_g(mlx_inline_array* dst,
    const mlx_inline_array* a_log, const mlx_inline_array* a, const mlx_inline_array* dt_bias) {
    try {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto decay = exp(astype(inputs[0], float32));
                auto sp = log1p(exp(add(inputs[1], inputs[2])));
                return {exp(negative(multiply(decay, sp)))};
            });
        auto result = (*compiled)({as_arr(a_log), as_arr(a), as_arr(dt_bias)});
        new (dst->buf) array(result[0]);
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

void mlx_inline_fused_precise_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* gate) {
    try {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& x = inputs[0];
                auto& g = inputs[1];
                auto g32 = multiply(astype(g, float32), sigmoid(astype(g, float32)));
                auto x32 = astype(x, float32);
                return {astype(multiply(g32, x32), x.dtype())};
            });
        auto result = (*compiled)({as_arr(x), as_arr(gate)});
        new (dst->buf) array(result[0]);
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst->buf) array(0.0f); }
}

// Compiled entire GDN layer forward.
// Uses 4 separate projection weights matching Python's in_proj_qkv/z/b/a.
// Scalar params are captured in the closure (all GDN layers share the same dims).
void mlx_inline_compiled_gdn_layer(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w,
    const mlx_inline_array* z_w,
    const mlx_inline_array* b_w,
    const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw,
    const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log,
    const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w,
    const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in,
    const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps
) {
    // Lazy-init compiled function with captured scalar params.
    // All GDN layers have the same dimensions, so this is compiled once.
    // Heap-leaked to avoid cross-DSO static destructor ordering crash.
    static CompiledFn* compiled = nullptr;
    try {
        if (!compiled) {
            int NV=nv, NK=nk, DK=dk, DV=dv, CD=cd, CK=ck, KD=kd;
            float EPS=norm_eps;
            compiled = make_compiled(
                [NV, NK, DK, DV, CD, CK, KD, EPS](const std::vector<array>& ins) -> std::vector<array> {
                    auto& normed      = ins[0];
                    auto& qkv_w       = ins[1];
                    auto& z_w         = ins[2];
                    auto& b_w         = ins[3];
                    auto& a_w         = ins[4];
                    auto& conv_w      = ins[5];
                    auto& q_nw        = ins[6];
                    auto& k_nw        = ins[7];
                    auto& a_log_arr   = ins[8];
                    auto& dt_bias_arr = ins[9];
                    auto& norm_w_arr  = ins[10];
                    auto& out_w       = ins[11];
                    auto& conv_state  = ins[12];
                    auto& ssm_state   = ins[13];
                    int B = normed.shape(0); int S = normed.shape(1);

                    // 4 separate matmuls — no splitting needed, matches Python exactly
                    auto qkv   = matmul(normed, qkv_w);
                    auto z     = reshape(matmul(normed, z_w), {B, S, NV, DV});
                    auto b_val = matmul(normed, b_w);
                    auto a_val = matmul(normed, a_w);

                    auto conv_in = concatenate({conv_state, qkv}, 1);
                    auto new_conv = slice(conv_in, {0, 1, 0}, {B, CK, CD});
                    auto conv_out = mlx::core::conv1d(conv_in, conv_w, 1, 0, 1, CD);
                    auto conv_act = multiply(conv_out, sigmoid(conv_out));

                    // shapeless=true does NOT support Split — keep slices here.
                    // (compiled_gdn_layer is for variable-length; fixed-shape decode uses
                    //  compiled_gdn_layer_fixed which uses shapeless=false and supports split.)
                    auto q = fast::rms_norm(reshape(slice(conv_act, {0, 0, 0}, {B, S, KD}), {B, S, NK, DK}), q_nw, EPS);
                    auto k = fast::rms_norm(reshape(slice(conv_act, {0, 0, KD}, {B, S, KD * 2}), {B, S, NK, DK}), k_nw, EPS);
                    auto v = reshape(slice(conv_act, {0, 0, KD * 2}, {B, S, CD}), {B, S, NV, DV});

                    auto g = exp(negative(multiply(exp(astype(a_log_arr, float32)),
                                 log1p(exp(add(a_val, dt_bias_arr))))));
                    auto beta = sigmoid(b_val);

                    auto& kernel = get_gdn_kernel();
                    auto kout = kernel(
                        {q, k, v, g, beta, ssm_state, array(S)},
                        {{B, S, NV, DV}, ssm_state.shape()},
                        {q.dtype(), ssm_state.dtype()},
                        {32, DV, B * NV}, {32, 4, 1},
                        {{"InT", q.dtype()}, {"StT", ssm_state.dtype()},
                         {"Dk", DK}, {"Dv", DV}, {"Hk", NK}, {"Hv", NV}},
                        std::nullopt, false, {});

                    auto out_n = fast::rms_norm(kout[0], norm_w_arr, EPS);
                    auto g32 = multiply(astype(z, float32), sigmoid(astype(z, float32)));
                    auto output = matmul(
                        reshape(astype(multiply(g32, astype(out_n, float32)), q.dtype()),
                                {B, S, NV * DV}),
                        out_w);
                    return {output, new_conv, kout[1]};
                });
        }

        auto result = (*compiled)({
            as_arr(normed),
            as_arr(qkv_w), as_arr(z_w), as_arr(b_w), as_arr(a_w),
            as_arr(conv_w),
            as_arr(q_nw), as_arr(k_nw), as_arr(a_log), as_arr(dt_bias),
            as_arr(norm_w), as_arr(out_w), as_arr(conv_state_in), as_arr(ssm_state_in)
        });
        new (dst_out->buf) array(result[0]);
        new (dst_conv_state->buf) array(result[1]);
        new (dst_ssm_state->buf) array(result[2]);
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_conv_state->buf) array(0.0f);
        new (dst_ssm_state->buf) array(0.0f);
    }
}

// shapeless=false version — fixed shapes, works with ALL primitives.
void mlx_inline_compiled_gdn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w, const mlx_inline_array* z_w,
    const mlx_inline_array* b_w, const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw, const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log, const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w, const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in, const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps
) {
    // Heap-leaked to avoid cross-DSO static destructor ordering crash.
    static CompiledFn* compiled = nullptr;
    try {
        if (!compiled) {
            int NV=nv, NK=nk, DK=dk, DV=dv, CD=cd, CK=ck, KD=kd;
            float EPS=norm_eps;
            compiled = make_compiled_fixed(
                [NV, NK, DK, DV, CD, CK, KD, EPS](const std::vector<array>& ins) -> std::vector<array> {
                    auto& normed = ins[0];
                    auto& qkv_w = ins[1]; auto& z_w = ins[2];
                    auto& b_w = ins[3]; auto& a_w = ins[4]; auto& conv_w = ins[5];
                    auto& q_nw = ins[6]; auto& k_nw = ins[7];
                    auto& a_log_arr = ins[8]; auto& dt_bias_arr = ins[9];
                    auto& norm_w_arr = ins[10]; auto& out_w = ins[11];
                    auto& conv_state = ins[12]; auto& ssm_state = ins[13];
                    int B = normed.shape(0); int S = normed.shape(1);

                    auto qkv = matmul(normed, qkv_w);
                    auto z = reshape(matmul(normed, z_w), {B, S, NV, DV});
                    auto b_val = matmul(normed, b_w);
                    auto a_val = matmul(normed, a_w);

                    auto conv_in = concatenate({conv_state, qkv}, 1);
                    auto new_conv = slice(conv_in, {0, 1, 0}, {B, CK, CD});
                    auto conv_out = mlx::core::conv1d(conv_in, conv_w, 1, 0, 1, CD);
                    auto conv_act = multiply(conv_out, sigmoid(conv_out));

                    // Single split → 3 siblings sharing one Split primitive (matches Python).
                    auto conv_parts = split(conv_act, Shape{KD, KD * 2}, -1);
                    auto q = fast::rms_norm(reshape(conv_parts[0], {B, S, NK, DK}), q_nw, EPS);
                    auto k = fast::rms_norm(reshape(conv_parts[1], {B, S, NK, DK}), k_nw, EPS);
                    auto v = reshape(conv_parts[2], {B, S, NV, DV});

                    auto g = exp(negative(multiply(exp(astype(a_log_arr, float32)),
                                 log1p(exp(add(a_val, dt_bias_arr))))));
                    auto beta = sigmoid(b_val);

                    auto& kernel = get_gdn_kernel();
                    auto kout = kernel(
                        {q, k, v, g, beta, ssm_state, array(S)},
                        {{B, S, NV, DV}, ssm_state.shape()},
                        {q.dtype(), ssm_state.dtype()},
                        {32, DV, B * NV}, {32, 4, 1},
                        {{"InT", q.dtype()}, {"StT", ssm_state.dtype()},
                         {"Dk", DK}, {"Dv", DV}, {"Hk", NK}, {"Hv", NV}},
                        std::nullopt, false, {});

                    auto out_n = fast::rms_norm(kout[0], norm_w_arr, EPS);
                    auto g32 = multiply(astype(z, float32), sigmoid(astype(z, float32)));
                    auto output = matmul(
                        reshape(astype(multiply(g32, astype(out_n, float32)), q.dtype()),
                                {B, S, NV * DV}), out_w);
                    return {output, new_conv, kout[1]};
                });
        }

        auto result = (*compiled)({
            as_arr(normed),
            as_arr(qkv_w), as_arr(z_w), as_arr(b_w), as_arr(a_w), as_arr(conv_w),
            as_arr(q_nw), as_arr(k_nw), as_arr(a_log), as_arr(dt_bias),
            as_arr(norm_w), as_arr(out_w), as_arr(conv_state_in), as_arr(ssm_state_in)
        });
        new (dst_out->buf) array(result[0]);
        new (dst_conv_state->buf) array(result[1]);
        new (dst_ssm_state->buf) array(result[2]);
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_conv_state->buf) array(0.0f);
        new (dst_ssm_state->buf) array(0.0f);
    }
}

void mlx_inline_compiled_attn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_cache_keys,
    mlx_inline_array* dst_cache_vals,
    const mlx_inline_array* normed,
    const mlx_inline_array* q_w,
    const mlx_inline_array* k_w,
    const mlx_inline_array* v_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_nw,
    const mlx_inline_array* k_nw,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int kv_offset,
    int rope_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float scale,
    int rope_dims,
    float rope_base,
    float rope_scale,
    float q_norm_eps,
    float k_norm_eps,
    bool gated
) {
    struct Entry {
        int batch;
        int cache_len;
        int n_heads;
        int n_kv;
        int head_dim;
        int rope_dims;
        int gated;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(normed).shape(0);
        int cache_len = as_arr(cache_keys_in).shape(2);

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.cache_len == cache_len
                && entry.n_heads == n_heads
                && entry.n_kv == n_kv
                && entry.head_dim == head_dim
                && entry.rope_dims == rope_dims
                && entry.gated == static_cast<int>(gated)) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int NH = n_heads;
            int NKV = n_kv;
            int HD = head_dim;
            int RD = rope_dims;
            int L = cache_len;
            bool GATED = gated;
            float SCALE = scale;
            float RBASE = rope_base;
            float RSCALE = rope_scale;
            float QEPS = q_norm_eps;
            float KEPS = k_norm_eps;

            entries->push_back(Entry{
                batch,
                cache_len,
                n_heads,
                n_kv,
                head_dim,
                rope_dims,
                static_cast<int>(gated),
                *make_compiled_fixed(
                    [NH, NKV, HD, RD, L, GATED, SCALE, RBASE, RSCALE, QEPS, KEPS]
                    (const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        auto& normed = ins[0];
                        auto& q_w = ins[1];
                        auto& k_w = ins[2];
                        auto& v_w = ins[3];
                        auto& o_w = ins[4];
                        auto& q_nw = ins[5];
                        auto& k_nw = ins[6];
                        auto& cache_keys = ins[7];
                        auto& cache_vals = ins[8];
                        auto& kv_offset_arr = ins[9];
                        auto& rope_offset_arr = ins[10];

                        int B = normed.shape(0);
                        int S = normed.shape(1);

                        auto q_proj = matmul(normed, q_w);
                        array queries(0.0f);
                        array gate(0.0f);
                        if (GATED) {
                            auto qg = reshape(q_proj, {B, S, NH, HD * 2});
                            auto qg_parts = split(qg, Shape{HD}, -1);
                            queries = qg_parts[0];
                            gate = reshape(qg_parts[1], {B, S, NH * HD});
                        } else {
                            queries = reshape(q_proj, {B, S, NH, HD});
                        }

                        auto new_k = matmul(normed, k_w);
                        auto new_v = matmul(normed, v_w);

                        queries = fast::rms_norm(queries, q_nw, QEPS);
                        auto keys = fast::rms_norm(reshape(new_k, {B, S, NKV, HD}), k_nw, KEPS);
                        auto values = reshape(new_v, {B, S, NKV, HD});

                        queries = transpose(queries, {0, 2, 1, 3});
                        keys = transpose(keys, {0, 2, 1, 3});
                        values = transpose(values, {0, 2, 1, 3});

                        queries = fast::rope(queries, RD, false, RBASE, RSCALE, rope_offset_arr);
                        keys = fast::rope(keys, RD, false, RBASE, RSCALE, rope_offset_arr);

                        auto kv_indices = broadcast_to(
                            reshape(kv_offset_arr, {1, 1, 1, 1}),
                            {B, NKV, S, HD});
                        auto updated_keys = put_along_axis(cache_keys, kv_indices, keys, 2);
                        auto updated_vals = put_along_axis(cache_vals, kv_indices, values, 2);

                        auto next_offset = add(kv_offset_arr, array(S));
                        auto positions = reshape(arange(L, int32), {1, 1, 1, L});
                        auto valid_mask = less(positions, reshape(next_offset, {1, 1, 1, 1}));

                        auto output = fast::scaled_dot_product_attention(
                            queries, updated_keys, updated_vals, SCALE, "", valid_mask);
                        output = transpose(output, {0, 2, 1, 3});
                        output = reshape(output, {B, S, NH * HD});
                        if (GATED) {
                            output = multiply(output, sigmoid(gate));
                        }
                        auto result = matmul(output, o_w);
                        return {result, updated_keys, updated_vals};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(normed),
            as_arr(q_w),
            as_arr(k_w),
            as_arr(v_w),
            as_arr(o_w),
            as_arr(q_nw),
            as_arr(k_nw),
            as_arr(cache_keys_in),
            as_arr(cache_vals_in),
            array(kv_offset),
            array(rope_offset),
        });
        new (dst_out->buf) array(result[0]);
        new (dst_cache_keys->buf) array(result[1]);
        new (dst_cache_vals->buf) array(result[2]);
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_cache_keys->buf) array(0.0f);
        new (dst_cache_vals->buf) array(0.0f);
    }
}

void mlx_inline_compiled_moe_layer_fixed(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* router_w,
    const mlx_inline_array* moe_gate_w,
    const mlx_inline_array* moe_up_w,
    const mlx_inline_array* moe_down_w,
    const mlx_inline_array* shared_gate_w,
    const mlx_inline_array* shared_up_w,
    const mlx_inline_array* shared_down_w,
    const mlx_inline_array* shared_expert_gate_w,
    int top_k,
    bool norm_topk_prob
) {
    struct Entry {
        int batch;
        int hidden;
        int num_experts;
        int routed_hidden;
        int shared_hidden;
        int top_k;
        int dtype;
        int norm_topk_prob;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int hidden = as_arr(x).shape(2);
        int num_experts = as_arr(router_w).shape(1);
        int routed_hidden = as_arr(moe_down_w).shape(1);
        int shared_hidden = as_arr(shared_down_w).shape(0);
        int dtype = static_cast<int>(as_arr(x).dtype().val());

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.hidden == hidden
                && entry.num_experts == num_experts
                && entry.routed_hidden == routed_hidden
                && entry.shared_hidden == shared_hidden
                && entry.top_k == top_k
                && entry.dtype == dtype
                && entry.norm_topk_prob == static_cast<int>(norm_topk_prob)) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int TOPK = top_k;
            bool NORM_TOPK = norm_topk_prob;
            entries->push_back(Entry{
                batch,
                hidden,
                num_experts,
                routed_hidden,
                shared_hidden,
                top_k,
                dtype,
                static_cast<int>(norm_topk_prob),
                *make_compiled_fixed(
                    [TOPK, NORM_TOPK](const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        auto& x = ins[0];
                        auto& router_w = ins[1];
                        auto& moe_gate_w = ins[2];
                        auto& moe_up_w = ins[3];
                        auto& moe_down_w = ins[4];
                        auto& shared_gate_w = ins[5];
                        auto& shared_up_w = ins[6];
                        auto& shared_down_w = ins[7];
                        auto& shared_expert_gate_w = ins[8];

                        int B = x.shape(0);
                        int S = x.shape(1);
                        int H = x.shape(2);
                        int expert_count = router_w.shape(1);

                        auto x_flat = reshape(x, {B * S, H});
                        auto gates = softmax(matmul(x_flat, router_w), -1, /*precise=*/true);
                        auto all_inds = argpartition(gates, -TOPK, -1);
                        auto inds = slice(all_inds, {0, expert_count - TOPK}, {B * S, expert_count});
                        auto scores = take_along_axis(gates, inds, -1);
                        if (NORM_TOPK) {
                            scores = divide(scores, sum(scores, {-1}, true));
                        }

                        auto switch_in = expand_dims(expand_dims(x_flat, 1), 2);
                        auto rhs_indices = std::optional<array>(inds);
                        auto x_gate_exp =
                            gather_mm(switch_in, moe_gate_w, std::nullopt, rhs_indices, false);
                        auto x_up_exp =
                            gather_mm(switch_in, moe_up_w, std::nullopt, rhs_indices, false);
                        auto x_act = multiply(multiply(x_gate_exp, sigmoid(x_gate_exp)), x_up_exp);
                        auto y_exp =
                            squeeze(gather_mm(x_act, moe_down_w, std::nullopt, rhs_indices, false), 2);
                        auto scores_exp = reshape(scores, {B * S, TOPK, 1});
                        auto y_routed = sum(multiply(y_exp, scores_exp), {-2}, false);

                        auto sh_gate = matmul(x_flat, shared_gate_w);
                        auto sh_up = matmul(x_flat, shared_up_w);
                        auto sh_act = multiply(multiply(sh_gate, sigmoid(sh_gate)), sh_up);
                        auto sh_out = matmul(sh_act, shared_down_w);
                        auto sh_scale = sigmoid(matmul(x_flat, shared_expert_gate_w));
                        auto y_shared = multiply(sh_out, sh_scale);

                        return {reshape(add(y_routed, y_shared), {B, S, H})};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(x),
            as_arr(router_w),
            as_arr(moe_gate_w),
            as_arr(moe_up_w),
            as_arr(moe_down_w),
            as_arr(shared_gate_w),
            as_arr(shared_up_w),
            as_arr(shared_down_w),
            as_arr(shared_expert_gate_w),
        });
        new (dst_out->buf) array(result[0]);
    } catch (const std::exception& e) { fprintf(stderr, "[C++ EXCEPTION] %s\n", e.what()); new (dst_out->buf) array(0.0f); }
}

} // extern "C"

// ============================================================================
// Full Qwen3.5 forward pass — single C++ function, low FFI overhead.
//
// The entire forward pass (embedding, all N layers, final norm, lm_head) runs
// here as pure C++ MLX ops.  No Rust stack frame is entered between ops;
// intermediate arrays are C++ locals, never placement-new'd through the FFI
// bridge.  This eliminates ~1800 Rust<->C++ FFI round trips per decode step.
//
// Important: this path still rebuilds the MLX graph on every token. It is not
// a traced or tape-replayed whole-model decode session.
// ============================================================================

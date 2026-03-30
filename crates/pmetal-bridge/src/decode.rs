use crate::InlineArray;

/// Create an f32 scalar cast to a specific MLX dtype.
///
/// Any scalar introduced into a decode graph must match the surrounding tensor
/// dtype unless the reference MLX path intentionally keeps it in float32.
#[inline(always)]
pub fn scalar_f32_dtype(value: f32, dtype: i32) -> InlineArray {
    InlineArray::from_f32(value).as_dtype(dtype)
}

/// Create an f32 scalar that matches the dtype of an existing tensor.
#[inline(always)]
pub fn scalar_f32_like(value: f32, like: &InlineArray) -> InlineArray {
    scalar_f32_dtype(value, like.dtype_raw())
}

/// Shared temperature sampling helper for bridge-backed decode paths.
///
/// The inverse-temperature scalar is cast to the logits dtype before the
/// multiply so bf16/f16 decode graphs do not get silently promoted to f32.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    if temperature <= 0.0 {
        logits_2d.argmax(-1)
    } else {
        let inv_temp = scalar_f32_like(1.0 / temperature, logits_2d);
        let lse = logits_2d.logsumexp(-1, true);
        let log_probs = logits_2d.subtract(&lse);
        let scaled = log_probs.multiply(&inv_temp);
        scaled.categorical()
    }
}

/// Match MLX-LM's cache-aware causal-attention behavior.
///
/// Upstream uses `mask=None` for single-token decode (`N == 1`) and `"causal"`
/// for multi-token prefill. Keeping decode on the unmasked fast path matters
/// for apples-to-apples performance against `mlx-lm`.
#[inline(always)]
pub fn sdpa_causal_like_mlx(
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    scale: f32,
    query_len: i32,
) -> InlineArray {
    if query_len == 1 {
        queries.sdpa_with_mask(keys, values, scale, None)
    } else {
        queries.sdpa(keys, values, scale, "causal")
    }
}

/// Shared generation-session setup for bridge-native decode loops.
///
/// `mlx::core::enable_compile()` was benchmarked and shown to regress decode
/// throughput on the active native paths, so the canonical bridge path keeps
/// it disabled here.
fn begin_generation_session_impl(tag: &str, model_dtype: i32, reset_peak_memory: bool) {
    crate::inline_array::clear_cache();
    if reset_peak_memory {
        crate::inline_array::reset_peak_memory();
    }
    static GENERATION_STREAM_INIT: std::sync::Once = std::sync::Once::new();
    GENERATION_STREAM_INIT.call_once(crate::inline_array::new_generation_stream);
    crate::inline_array::set_generation_stream();
    crate::inline_array::set_wired_limit_max();

    eprintln!(
        "[{tag}] generate: dtype={model_dtype} active={:.0}MB",
        crate::inline_array::get_active_memory() as f64 / 1e6,
    );
}

pub fn begin_generation_session(tag: &str, model_dtype: i32) {
    begin_generation_session_impl(tag, model_dtype, true);
}

pub fn begin_generation_session_preserve_peak(tag: &str, model_dtype: i32) {
    begin_generation_session_impl(tag, model_dtype, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_like_matches_reference_dtype() {
        let like = InlineArray::from_f32(1.0).as_dtype(crate::compat::Dtype::Bfloat16.as_i32());
        let scalar = scalar_f32_like(0.5, &like);
        assert_eq!(scalar.dtype_raw(), like.dtype_raw());
    }
}

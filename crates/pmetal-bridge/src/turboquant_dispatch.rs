//! Shared TurboQuant attention dispatch for the bridge native paths.
//!
//! Every native architecture (`qwen3_native`, `gpt_oss_native`,
//! `llama4_native`, …) needs the same decode-time decision tree once it has
//! a `QuantizedKvCache`:
//!
//! - **Decode (S=1)**: try the optimized `append_and_compute_attention` path
//!   inside the cache (cold-only GPU TurboQuant kernels, hot-only fp16 SDPA,
//!   or dequantize-and-concat when mixed). On any error, fall back to a
//!   correctness-first `append → dequantize → standard SDPA` chain.
//! - **Prefill (S>1)**: append into the cache, then run standard SDPA. When
//!   `prev == 0` (first prefill, no prior history) use the fresh keys/values
//!   directly — no dequantization needed. Otherwise dequantize the full
//!   cache and run SDPA over the concatenated history.
//!
//! Shape contract — queries/keys/values must be `[B, H, S, D]` with `H` the
//! query head count and `H_kv` the cache head count. The function does **not**
//! perform GQA expansion: callers that have `H != H_kv` must replicate keys
//! and values along the head axis before invoking the fallback path. The
//! cache's own direct-attention path handles GQA internally.
//!
//! The function does **not** update the per-layer offset counter — callers
//! own that state because the surrounding code paths (e.g. quantized affine,
//! sliding window) update it from a single point at the end of the layer.

use crate::InlineArray;
use crate::turboquant::QuantizedKvCache;

/// Run a TurboQuant attention layer step on the given cache.
///
/// Returns the layer output `[B, H, S, D_v]`. `prev` is the cache offset
/// **before** the new chunk is appended, used to pick the prefill fallback
/// path. `trace_label` is a short upper-case identifier (e.g. `"QWEN"`,
/// `"GPT_OSS"`) prefixed to trace messages emitted via the
/// `PMETAL_TRACE_TURBOQUANT` env var.
pub fn turboquant_attention_step(
    tq_cache: &mut QuantizedKvCache,
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    scale: f32,
    prev: i32,
    trace_label: &'static str,
) -> InlineArray {
    let s = queries.dim(2);
    if s == 1 {
        match tq_cache.append_and_compute_attention(queries, keys, values, scale) {
            Ok(output) => output,
            Err(err) => {
                trace(
                    trace_label,
                    &format!(
                        "decode_fallback=append_and_compute_attention_err prev={prev} err={err}"
                    ),
                );
                tq_cache.append(keys, values).ok();
                let full_keys = tq_cache.dequantize_keys().unwrap_or_else(|| keys.clone());
                let full_values = tq_cache
                    .dequantize_values()
                    .unwrap_or_else(|| values.clone());
                crate::decode::sdpa_causal_like_mlx(queries, &full_keys, &full_values, scale, s)
            }
        }
    } else {
        tq_cache.append(keys, values).ok();
        if prev == 0 {
            trace(
                trace_label,
                &format!("prefill_path=dense_prompt_only seq={s}"),
            );
            crate::decode::sdpa_causal_like_mlx(queries, keys, values, scale, s)
        } else {
            trace(
                trace_label,
                &format!("prefill_fallback=full_dequantized seq={s} prev={prev}"),
            );
            let full_keys = tq_cache.dequantize_keys().unwrap_or_else(|| keys.clone());
            let full_values = tq_cache
                .dequantize_values()
                .unwrap_or_else(|| values.clone());
            crate::decode::sdpa_causal_like_mlx(queries, &full_keys, &full_values, scale, s)
        }
    }
}

/// True when `PMETAL_TRACE_TURBOQUANT` is set in the environment. Cheap;
/// the env probe runs only when this is consulted by `trace`.
#[inline]
fn trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

#[inline]
fn trace(label: &'static str, message: &str) {
    if trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][{label}] {message}");
    }
}

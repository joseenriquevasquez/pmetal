//! KV caches — sliding rotating buffer for sliding-window layers, unbounded
//! growth for full-attention layers (with optional zero-overhead affine
//! quantization), plus the shared `NativeCache` wrapper.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

/// KV cache for one attention layer.
///
/// GPT-OSS uses two kinds:
///   - Full attention: unbounded growth (256-token chunk reallocation strategy)
///   - Sliding attention: rotating window of `sliding_window` tokens
///
/// Zero-overhead affine quantization is supported for full-attention layers only.
/// Sliding window layers use the bf16 rotating buffer path unconditionally (the
/// rotation logic makes quantized buffers incompatible).
pub struct KvLayerCache {
    pub keys: Option<InlineArray>, // [B, H, MAX_T, D] (or [B, H, window, D] for sliding)
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,               // total tokens written
    pub is_sliding: bool,
    pub window: i32, // sliding window size (ignored when is_sliding=false)
    /// Zero-overhead affine-quantized cache (full-attention layers only).
    pub quantized_keys: Option<crate::qwen3_native::QuantizedTuple>,
    pub quantized_values: Option<crate::qwen3_native::QuantizedTuple>,
    /// None on sliding-window layers or when bf16 cache is used.
    pub quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
}

/// Full model cache — one KV entry per layer.
pub struct NativeCache {
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
}

impl NativeCache {
    /// Evaluate and detach all cache state arrays in one GPU submission.
    ///
    /// Must be called after the prefill forward pass and before decode.
    /// Equivalent to Python's `mx.eval([c.state for c in prompt_cache])`.
    pub fn eval_and_detach_states(&mut self) {
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.kv_caches {
            if let Some(k) = c.keys.take() {
                let trimmed = if c.offset > 0 && c.offset < k.dim(2) {
                    k.slice(&[0, 0, 0, 0], &[k.dim(0), k.dim(1), c.offset, k.dim(3)])
                } else {
                    k
                };
                c.keys = Some(trimmed);
            }
            if let Some(v) = c.values.take() {
                let trimmed = if c.offset > 0 && c.offset < v.dim(2) {
                    v.slice(&[0, 0, 0, 0], &[v.dim(0), v.dim(1), c.offset, v.dim(3)])
                } else {
                    v
                };
                c.values = Some(trimmed);
            }
            if let Some(ref mut k) = c.keys {
                to_eval.push(k);
            }
            if let Some(ref mut v) = c.values {
                to_eval.push(v);
            }
        }
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_quant(weights, None)
    }

    /// Create a cache with optional affine KV quantization.
    ///
    /// Quantization is silently disabled for sliding-window layers (the rotation
    /// logic is incompatible with quantized buffers). Only full-attention layers
    /// receive a `quant_config`.
    pub fn new_with_quant(
        weights: &NativeWeights,
        quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
    ) -> Self {
        let kv_caches = weights
            .layers
            .iter()
            .map(|lw| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                is_sliding: lw.attn_is_sliding,
                window: lw.attn_sliding_window,
                quantized_keys: None,
                quantized_values: None,
                // Disable quantization for sliding layers — their rotating buffer
                // is incompatible with the pre-allocated quantized buffer scheme.
                quant_config: if lw.attn_is_sliding {
                    None
                } else {
                    quant_config
                },
            })
            .collect();

        NativeCache {
            kv_caches,
            rope_offset: 0,
        }
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}

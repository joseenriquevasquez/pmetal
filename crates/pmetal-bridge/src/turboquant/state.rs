//! Shared TurboQuant state: per-tensor runtime selection plus the
//! `(key, value)` core pair built once per `(dim, config)` combination.
//!
//! The `TurboQuantState` is expensive to construct (QR decomposition +
//! Lloyd-Max) and immutable after construction; wrap it in `Arc` and
//! share it across every layer/head that uses the same shape.

use std::sync::Arc;

use super::config::{TurboQuantConfig, TurboQuantTensorConfig};
use super::core::TurboQuantCore;

/// Per-tensor runtime — selects between a single Uniform core and a
/// regular/outlier Mixed core pair. Constructed via [`build_tensor_runtime`].
#[derive(Debug, Clone)]
pub(super) enum TensorRuntime {
    Uniform {
        config: TurboQuantTensorConfig,
        core: Arc<TurboQuantCore>,
    },
    Mixed {
        config: TurboQuantTensorConfig,
        regular_core: Arc<TurboQuantCore>,
        outlier_core: Arc<TurboQuantCore>,
    },
}

/// Shared TurboQuant state for a given K/V head dimension and config.
///
/// Expensive to build (QR decomposition + Lloyd-Max).  Wrap in `Arc` and share
/// across all layers and heads that share the same (dim, config) pair.
#[derive(Debug, Clone)]
pub struct TurboQuantState {
    pub(super) keys: TensorRuntime,
    pub(super) values: TensorRuntime,
}

impl TurboQuantState {
    /// Build a new state.  Typical latency: ~50–200 ms for dim=128.
    pub fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Self {
        config.keys.assert_valid(key_dim, "keys");
        config.values.assert_valid(value_dim, "values");

        // Cache cores so (dim, bits) pairs that appear for both keys and values
        // share the same Arc.
        let mut core_cache = std::collections::HashMap::<(usize, u8), Arc<TurboQuantCore>>::new();
        let mut get_core = |subdim: usize, max_mse_bits: u8| {
            core_cache
                .entry((subdim, max_mse_bits))
                .or_insert_with(|| Arc::new(TurboQuantCore::new(subdim, max_mse_bits)))
                .clone()
        };

        let keys = build_tensor_runtime(key_dim, config.keys, true, &mut get_core);
        let values = build_tensor_runtime(value_dim, config.values, false, &mut get_core);

        Self { keys, values }
    }

    /// `true` when the fused GPU attention kernels accept this state's
    /// `(config, dim)` combination. When `false`, the cache still produces
    /// correct results — it just falls back to dequantize+SDPA, which is
    /// roughly 2–4× slower at decode time. Use this to surface "slow path"
    /// warnings in model-load logs.
    ///
    /// Coverage as of 2026-04: fused attention requires Uniform precision
    /// for both keys and values. Mixed precision and head_dims other than
    /// 128 / 256 fall back.
    pub fn has_gpu_attention_support(&self) -> bool {
        self.gpu_attention_unsupported_reason_inner().is_none()
    }

    /// Return a static string describing why fused GPU attention is
    /// unsupported for this state, or a placeholder when it *is* supported.
    pub fn gpu_attention_unsupported_reason(&self) -> &'static str {
        self.gpu_attention_unsupported_reason_inner()
            .unwrap_or("fused GPU attention is supported for this state")
    }

    fn gpu_attention_unsupported_reason_inner(&self) -> Option<&'static str> {
        if !matches!(self.keys, TensorRuntime::Uniform { .. }) {
            return Some("keys use Mixed precision (no fused attention kernel)");
        }
        if !matches!(self.values, TensorRuntime::Uniform { .. }) {
            return Some("values use Mixed precision (no fused attention kernel)");
        }
        None
    }
}

fn build_tensor_runtime<F>(
    total_dim: usize,
    config: TurboQuantTensorConfig,
    _is_keys: bool,
    get_core: &mut F,
) -> TensorRuntime
where
    F: FnMut(usize, u8) -> Arc<TurboQuantCore>,
{
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            TensorRuntime::Uniform {
                config,
                // Build the full MSE codebook ladder even for keys so pure-MSE
                // key paths (for example the full-byte D256 experiments) can
                // reuse the same core without a second cache format.
                core: get_core(total_dim, bits),
            }
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = total_dim - outlier_count;
            TensorRuntime::Mixed {
                config,
                regular_core: get_core(regular_dim, regular_bits),
                outlier_core: get_core(outlier_count, outlier_bits),
            }
        }
    }
}

//! TurboQuant KV cache — zero mlx-rs dependency.
//!
//! Self-contained implementation of the TurboQuant-inspired KV cache using only
//! [`InlineArray`] and pure-Rust math.  The module is intentionally free of any
//! mlx-rs or pmetal-metal imports; all GPU work is driven through
//! `InlineArray::matmul` which dispatches to MLX's Metal backend automatically.
//!
//! ## Storage layout invariants (audit-pinned 2026-04-25)
//!
//! Pack/encode/decode/score MUST agree on these axis orders. This contract is
//! mirrored in `cpp/bridge/turboquant.h`'s top-of-file comment; keep both in
//! sync. The [`layout_invariants`] tests pin the derived dimensions
//! ([`packed_qjl_words`] etc.) so a silent off-by-one in one of the helpers
//! will fail CI rather than corrupt cache reads.
//!
//! Live cache layouts (read by score kernels):
//!
//! * `indices`        — `[N, D, S_cap]` `u8` (7-bit centroid id)
//! * `qjl_signs`      — `[N, ceil(D/32), S_cap]` `u32` (packed sign bits)
//! * `norms`          — `[N, S_cap]` `f32`
//! * `residual_norms` — `[N, S_cap]` `f32`
//! * `codebook`       — `[C]` `f32` (sorted Lloyd-Max centroids)
//!
//! q8 fullbyte (D=256, n_seq>=1024) uses a transposed seq-major shadow:
//!
//! * `q8_fullbyte_seq`     — `[N, S_cap, D]` `u8`
//! * `q8_slot_scales_seq`  — `[N, S_cap]` `f16`
//! * `d256_rot_values_seq` — `[N, S_cap, D]` `bf16`
//!
//! GPU-attention coverage:
//!
//! * Uniform + head_dim==128 → fused d128 attention kernel
//! * Uniform + head_dim==256 → fused d256 attention kernel
//!   (+ q8 fullbyte specialisation when bits==8 and n_seq >= 1024)
//! * Uniform + other head_dim → dequantize + SDPA fallback
//! * Mixed precision → dequantize + SDPA fallback (no GPU kernel today)
//!
//! # Algorithm overview
//!
//! **Keys** (inner-product optimised):
//!   1. Normalise each vector onto the unit sphere; store the L2 norm.
//!   2. Apply the orthogonal rotation Π: `r = Π · k_norm`.
//!   3. Nearest-centroid scalar quantisation of every coordinate using the
//!      Lloyd-Max codebook for the Beta distribution (MSE at `b-1` bits).
//!   4. Compute the residual `e = k_norm - Π^T · codebook[idx]` and project it
//!      through a Gaussian matrix J: sign(J · e) gives 1-bit QJL signs.
//!
//! **Values** (MSE optimised, no QJL stage):
//!   1. Normalise + store norm.
//!   2. Rotate then quantise with the full `b`-bit codebook.
//!
//! **Outlier-aware mixed-bit** (optional):
//!   Per-row, the top-`k` coordinates by magnitude are flagged as "outliers"
//!   and stored at a higher bit-width in a separate sub-vector.
//!
//! # What is NOT in this module
//!
//! - The mlx-rs `Array` integration code.
//! - The `TurboQuantKvCache` struct (see `KvLayerCache` in qwen3_native).
//! - The pmetal-metal `TurboQuantTransform` (InlineArray.matmul replaces it).
//! - The fused-attention path (we use standard SDPA).
//!
//! # Module-split deferral (audit 2026-04-25)
//!
//! At ~5000 LOC this file is large for a "single module." A clean split into
//! `turboquant/{config, codebook, state, cache, encode, attention, qjl}` was
//! considered during the April audit; we deferred it because:
//!
//! 1. Many `pub(super)` boundaries currently rely on intra-file visibility,
//!    so a split needs careful re-export plumbing without widening surfaces.
//! 2. The test fixtures depend on private constructors via `mod tests`
//!    super-imports — splitting them would mean either re-export-for-test
//!    cuts or a fixture-only sub-crate.
//! 3. The recently-added [`crate::turboquant_dispatch`] already extracted the
//!    cross-architecture decode dispatcher, which was the highest-coupling
//!    surface and the most legitimate split candidate.
//!
//! The split work is real but not blocking; capture it as a follow-up so the
//! audit can report Phase F as documented-not-implemented.

use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use crate::InlineArray;
use crate::compat::Dtype;

// ── Constants ────────────────────────────────────────────────────────────────

/// Deterministic seed — same as the mlx-rs reference implementation.
const TURBOQUANT_SEED: u64 = 0x5442_5155_414e_544d;
/// Vectors with L2 norm below this are treated as zero.
const ZERO_EPSILON: f32 = 1e-12;
/// Defensive upper bound on encoded residual L2 norms, used to prevent Inf/NaN
/// from upstream fp16 corruption from reaching the QJL term in the score and
/// attention kernels. Derived from `||k_rot||=1` + triangle inequality plus a
/// conservative margin over the Beta-codebook reconstruction norm; realistic
/// values are below 1.0 for any bit-width b≥2. Any residual norm above this
/// cap would already violate Theorem 2's distortion bound — clipping is safe.
const MAX_RESIDUAL_NORM: f32 = 4.0;

fn turboquant_trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

fn turboquant_wht_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_USE_WHT")
        .map(|value| value != "0")
        .unwrap_or(true)
}

/// Returns true when the active dim should use signed-FWHT instead of dense
/// `[d×d]` matmul for rotation and QJL projection.
///
/// FWHT is `O(d log d)`; dense matmul is `O(d²)`. For every transformer KV
/// head_dim in current use (32, 64, 96, 128, 192, 256, 320…) the FWHT
/// alternative wins on both compute and memory. We restrict to power-of-two
/// dims ≥ 4 because the underlying Walsh-Hadamard kernel requires it; non-pow2
/// (192, 80, etc.) keep the dense path.
fn dim_uses_fwht(dim: usize) -> bool {
    turboquant_wht_enabled() && dim >= 4 && dim.is_power_of_two()
}

fn turboquant_q8_fullbyte_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_Q8_FULLBYTE")
        .map(|value| value != "0")
        .unwrap_or(false)
}

fn trace_turboquant_bridge(message: &str) {
    if turboquant_trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][BRIDGE] {message}");
    }
}

fn eval_stage_micros(array: &InlineArray) -> u128 {
    let start = Instant::now();
    array.eval();
    crate::inline_array::synchronize();
    start.elapsed().as_micros()
}

mod config;
pub use config::{DEFAULT_RECENT_WINDOW, TurboQuantConfig, TurboQuantTensorConfig};
use config::HOT_EVICTION_CHUNK;

mod core;
pub use core::TurboQuantCore;

mod state;
pub use state::TurboQuantState;
use state::TensorRuntime;

mod bits;
pub use bits::PackedBits;
use bits::{packed_qjl_words, unpack_all};

// ═══════════════════════════════════════════════════════════════════════════
// Per-layer quantised storage
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-resident quantised key data for the Uniform (non-outlier) path.
///
/// All tensors live entirely on the GPU — no CPU round-trips during normal
/// operation.  Shape convention (accumulated over T steps):
///   indices:         [B, H, T, D]  uint8   — codebook index per coordinate
///   indices_t:       [B, H, D, T]  uint8   — score-friendly transposed view
///   q8_keybytes_t:   [B, H, D, T]  uint8   — q8-only packed index/sign view
///   q8_keybytes_seq: [B, H, T, D]  uint8   — q8 seq-major key shadow
///   q8_fullbyte_seq: [B, H, T, D]  uint8   — q8 seq-major pure-256-centroid key shadow
///   q8_kvbytes_seq:  [B, H, T, D]  uint16  — D256 q8 seq-major packed {key,value}
///   q8_slot_scales_seq: [B, H, T, 3]  f32  — [key_norm, residual_norm, value_norm]
///   norms:           [B, H, T, 1]  f32     — optional L2 norm before unit-sphere normalise
///   qjl_signs:       [B, H, T, ceil(D/32)]  uint32 packed sign words
///   qjl_signs_t:     [B, H, ceil(D/32), T]  uint32 transposed sign-word view
///   residual_norms:  [B, H, T, 1]  f32     — optional unscaled residual L2 norm
#[derive(Debug, Clone)]
struct GpuKeyStore {
    indices: InlineArray,
    indices_t: Option<InlineArray>,
    q8_keybytes_t: Option<InlineArray>,
    q8_keybytes_seq: Option<InlineArray>,
    q8_fullbyte_seq: Option<InlineArray>,
    q8_kvbytes_seq: Option<InlineArray>,
    q8_slot_scales_seq: Option<InlineArray>,
    norms: Option<InlineArray>,
    qjl_signs: InlineArray,
    qjl_signs_t: Option<InlineArray>,
    residual_norms: Option<InlineArray>,
}

impl GpuKeyStore {
    /// Concatenate a new step's GPU arrays along the T (axis 2) dimension.
    fn append(&mut self, new: GpuKeyStore) {
        self.indices = self.indices.kv_cache_append(&new.indices, 2);
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_t = match (self.q8_keybytes_t.take(), new.q8_keybytes_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_seq = match (self.q8_keybytes_seq.take(), new.q8_keybytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_fullbyte_seq = match (self.q8_fullbyte_seq.take(), new.q8_fullbyte_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_kvbytes_seq = match (self.q8_kvbytes_seq.take(), new.q8_kvbytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_slot_scales_seq = match (self.q8_slot_scales_seq.take(), new.q8_slot_scales_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
        self.qjl_signs = self.qjl_signs.kv_cache_append(&new.qjl_signs, 2);
        self.qjl_signs_t = match (self.qjl_signs_t.take(), new.qjl_signs_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.residual_norms = match (self.residual_norms.take(), new.residual_norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
    }

    fn cache_seq_capacity(&self) -> i32 {
        self.q8_kvbytes_seq
            .as_ref()
            .map(|arr| arr.dim(2))
            .or_else(|| self.q8_keybytes_seq.as_ref().map(|arr| arr.dim(2)))
            .or_else(|| self.indices_t.as_ref().map(|arr| arr.dim(3)))
            .unwrap_or_else(|| self.indices.dim(2))
    }

    fn indices_t_array(&self) -> InlineArray {
        self.indices_t
            .clone()
            .unwrap_or_else(|| self.indices.transpose_axes(&[0, 1, 3, 2]))
    }

    fn qjl_signs_t_array(&self) -> InlineArray {
        self.qjl_signs_t
            .clone()
            .unwrap_or_else(|| self.qjl_signs.transpose_axes(&[0, 1, 3, 2]))
    }

    fn qjl_words(&self) -> i32 {
        self.qjl_signs_t
            .as_ref()
            .map(|arr| arr.dim(2))
            .unwrap_or_else(|| self.qjl_signs.dim(3))
    }

    fn slot_scale_component_array(&self, component: i32) -> Option<InlineArray> {
        let slot_scales = self.q8_slot_scales_seq.as_ref()?;
        Some(slot_scales.slice(
            &[0, 0, 0, component],
            &[
                slot_scales.dim(0),
                slot_scales.dim(1),
                slot_scales.dim(2),
                component + 1,
            ],
        ))
    }

    fn key_norms_array(&self) -> Option<InlineArray> {
        self.norms
            .clone()
            .or_else(|| self.slot_scale_component_array(0))
    }

    fn residual_norms_array(&self) -> Option<InlineArray> {
        self.residual_norms
            .clone()
            .or_else(|| self.slot_scale_component_array(1))
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.indices);
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(q8_keybytes_t) = self.q8_keybytes_t.as_mut() {
            out.push(q8_keybytes_t);
        }
        if let Some(q8_keybytes_seq) = self.q8_keybytes_seq.as_mut() {
            out.push(q8_keybytes_seq);
        }
        if let Some(q8_fullbyte_seq) = self.q8_fullbyte_seq.as_mut() {
            out.push(q8_fullbyte_seq);
        }
        if let Some(q8_kvbytes_seq) = self.q8_kvbytes_seq.as_mut() {
            out.push(q8_kvbytes_seq);
        }
        if let Some(q8_slot_scales_seq) = self.q8_slot_scales_seq.as_mut() {
            out.push(q8_slot_scales_seq);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        out.push(&mut self.qjl_signs);
        if let Some(qjl_signs_t) = self.qjl_signs_t.as_mut() {
            out.push(qjl_signs_t);
        }
        if let Some(residual_norms) = self.residual_norms.as_mut() {
            out.push(residual_norms);
        }
    }
}

/// GPU-resident quantised value data for the Uniform path.
///
///   indices:  [B, H, T, D]  uint8
///   indices_t:[B, H, D, T]  uint8
///   norms:    [B, H, T, 1]  f32
#[derive(Debug, Clone)]
struct GpuValueStore {
    indices: Option<InlineArray>,
    indices_t: Option<InlineArray>,
    norms: Option<InlineArray>,
    d256_rot_values_seq: Option<InlineArray>,
}

impl GpuValueStore {
    fn append(&mut self, new: GpuValueStore) {
        self.indices = match (self.indices.take(), new.indices) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.d256_rot_values_seq = match (self.d256_rot_values_seq.take(), new.d256_rot_values_seq)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
    }

    fn indices_t_array(&self) -> Option<InlineArray> {
        self.indices_t.clone().or_else(|| {
            self.indices
                .as_ref()
                .map(|arr| arr.transpose_axes(&[0, 1, 3, 2]))
        })
    }

    fn norms_array(&self) -> Option<InlineArray> {
        self.norms.clone()
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        if let Some(indices) = self.indices.as_mut() {
            out.push(indices);
        }
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        if let Some(d256_rot_values_seq) = self.d256_rot_values_seq.as_mut() {
            out.push(d256_rot_values_seq);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-resident Mixed-precision K/V stores (Phase 3a)
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout pinned for upcoming Mixed attention kernels:
//   regular_indices    — [B, H, T, D_reg]               u8
//   regular_indices_t  — [B, H, D_reg, T]               u8   (S-innermost)
//   regular_qjl_signs  — [B, H, T, ceil(D_reg/32)]      u32
//   regular_qjl_signs_t — [B, H, ceil(D_reg/32), T]     u32  (S-innermost)
//   regular_norms      — [B, H, T, 1]                   f32
//   regular_residual_norms — [B, H, T, 1]               f32
//   regular_src_dim    — [B, H, T, D_reg]               u8   (scatter-back map)
//
// Outlier mirrors: same shapes with D_out instead of D_reg.
//
// Scatter tables (`*_src_dim`) hold the original-D positions of each
// sub-vector slot, sorted ascending. They are written once per token at
// encode time and gathered by the attention kernel to scatter regular and
// outlier contributions back into the [B, H, D_total] output. K and V
// stores have *independent* scatter tables — `select_outlier_mask` runs on
// each tensor's row magnitudes separately.

#[derive(Debug, Clone)]
struct GpuMixedKeyStore {
    regular_indices: InlineArray,
    regular_indices_t: Option<InlineArray>,
    regular_qjl_signs: InlineArray,
    regular_qjl_signs_t: Option<InlineArray>,
    regular_norms: InlineArray,
    regular_residual_norms: InlineArray,
    regular_src_dim: InlineArray,
    outlier_indices: InlineArray,
    outlier_indices_t: Option<InlineArray>,
    outlier_qjl_signs: InlineArray,
    outlier_qjl_signs_t: Option<InlineArray>,
    outlier_norms: InlineArray,
    outlier_residual_norms: InlineArray,
    outlier_src_dim: InlineArray,
}

impl GpuMixedKeyStore {
    fn append(&mut self, new: GpuMixedKeyStore) {
        self.regular_indices = self.regular_indices.kv_cache_append(&new.regular_indices, 2);
        self.regular_indices_t = match (self.regular_indices_t.take(), new.regular_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_qjl_signs = self
            .regular_qjl_signs
            .kv_cache_append(&new.regular_qjl_signs, 2);
        self.regular_qjl_signs_t = match (self.regular_qjl_signs_t.take(), new.regular_qjl_signs_t)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_norms = self.regular_norms.kv_cache_append(&new.regular_norms, 2);
        self.regular_residual_norms = self
            .regular_residual_norms
            .kv_cache_append(&new.regular_residual_norms, 2);
        self.regular_src_dim = self.regular_src_dim.kv_cache_append(&new.regular_src_dim, 2);
        self.outlier_indices = self.outlier_indices.kv_cache_append(&new.outlier_indices, 2);
        self.outlier_indices_t = match (self.outlier_indices_t.take(), new.outlier_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_qjl_signs = self
            .outlier_qjl_signs
            .kv_cache_append(&new.outlier_qjl_signs, 2);
        self.outlier_qjl_signs_t = match (self.outlier_qjl_signs_t.take(), new.outlier_qjl_signs_t)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_norms = self.outlier_norms.kv_cache_append(&new.outlier_norms, 2);
        self.outlier_residual_norms = self
            .outlier_residual_norms
            .kv_cache_append(&new.outlier_residual_norms, 2);
        self.outlier_src_dim = self.outlier_src_dim.kv_cache_append(&new.outlier_src_dim, 2);
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.regular_indices);
        if let Some(indices_t) = self.regular_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.regular_qjl_signs);
        if let Some(signs_t) = self.regular_qjl_signs_t.as_mut() {
            out.push(signs_t);
        }
        out.push(&mut self.regular_norms);
        out.push(&mut self.regular_residual_norms);
        out.push(&mut self.regular_src_dim);
        out.push(&mut self.outlier_indices);
        if let Some(indices_t) = self.outlier_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.outlier_qjl_signs);
        if let Some(signs_t) = self.outlier_qjl_signs_t.as_mut() {
            out.push(signs_t);
        }
        out.push(&mut self.outlier_norms);
        out.push(&mut self.outlier_residual_norms);
        out.push(&mut self.outlier_src_dim);
    }
}

#[derive(Debug, Clone)]
struct GpuMixedValueStore {
    regular_indices: InlineArray,
    regular_indices_t: Option<InlineArray>,
    regular_norms: InlineArray,
    regular_src_dim: InlineArray,
    outlier_indices: InlineArray,
    outlier_indices_t: Option<InlineArray>,
    outlier_norms: InlineArray,
    outlier_src_dim: InlineArray,
}

impl GpuMixedValueStore {
    fn append(&mut self, new: GpuMixedValueStore) {
        self.regular_indices = self.regular_indices.kv_cache_append(&new.regular_indices, 2);
        self.regular_indices_t = match (self.regular_indices_t.take(), new.regular_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.regular_norms = self.regular_norms.kv_cache_append(&new.regular_norms, 2);
        self.regular_src_dim = self.regular_src_dim.kv_cache_append(&new.regular_src_dim, 2);
        self.outlier_indices = self.outlier_indices.kv_cache_append(&new.outlier_indices, 2);
        self.outlier_indices_t = match (self.outlier_indices_t.take(), new.outlier_indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.outlier_norms = self.outlier_norms.kv_cache_append(&new.outlier_norms, 2);
        self.outlier_src_dim = self.outlier_src_dim.kv_cache_append(&new.outlier_src_dim, 2);
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.regular_indices);
        if let Some(indices_t) = self.regular_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.regular_norms);
        out.push(&mut self.regular_src_dim);
        out.push(&mut self.outlier_indices);
        if let Some(indices_t) = self.outlier_indices_t.as_mut() {
            out.push(indices_t);
        }
        out.push(&mut self.outlier_norms);
        out.push(&mut self.outlier_src_dim);
    }
}

/// Quantised key store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedKeyStore {
    // GPU-native store (Uniform path only).  When Some, dequantize uses GPU ops.
    gpu: Option<GpuKeyStore>,

    // GPU-native store for the Mixed path. Populated alongside the CPU
    // `PackedBits` fields below when a Mixed config is active and the GPU
    // encode succeeded. Phase 3a — present but currently consumed only by
    // round-trip tests; Phase 3c will wire it into attention.
    gpu_mixed: Option<GpuMixedKeyStore>,

    // CPU fallback: regular (non-outlier) sub-vector data.
    pub regular_indices: PackedBits,
    pub regular_qjl_signs: PackedBits,
    pub regular_norms: Vec<f32>,
    pub regular_residual_norms: Vec<f32>,

    // Outlier sub-vector data (None when config is Uniform).
    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_qjl_signs: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
    pub outlier_residual_norms: Option<Vec<f32>>,
}

impl QuantizedKeyStore {
    fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits.saturating_sub(1),
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits.saturating_sub(1),
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => {
                Some(outlier_bits.saturating_sub(1))
            }
        };

        Self {
            gpu: None,
            gpu_mixed: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_qjl_signs: PackedBits::new(1),
            regular_norms: Vec::new(),
            regular_residual_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_qjl_signs: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
            outlier_residual_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    fn extend(
        &mut self,
        encoded: &EncodedKeyRows,
        outlier_encoded: Option<&EncodedKeyRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
        self.regular_indices.extend_from_slice(&encoded.mse_indices);
        self.regular_qjl_signs.extend_from_slice(&encoded.qjl_signs);
        self.regular_norms.extend_from_slice(&encoded.norms);
        self.regular_residual_norms
            .extend_from_slice(&encoded.residual_norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant key outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant key outlier indices missing")
                .extend_from_slice(&outlier.mse_indices);
            self.outlier_qjl_signs
                .as_mut()
                .expect("TurboQuant key outlier QJL signs missing")
                .extend_from_slice(&outlier.qjl_signs);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant key outlier norms missing")
                .extend_from_slice(&outlier.norms);
            self.outlier_residual_norms
                .as_mut()
                .expect("TurboQuant key outlier residual_norms missing")
                .extend_from_slice(&outlier.residual_norms);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_qjl_signs.byte_len()
            + self.regular_norms.len() * 4
            + self.regular_residual_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_qjl_signs.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
            + self
                .outlier_residual_norms
                .as_ref()
                .map_or(0, |v| v.len() * 4)
    }
}

/// Quantised value store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedValueStore {
    // GPU-native store (Uniform path only).
    gpu: Option<GpuValueStore>,

    // GPU-native store for the Mixed path. See `QuantizedKeyStore.gpu_mixed`.
    gpu_mixed: Option<GpuMixedValueStore>,

    pub regular_indices: PackedBits,
    pub regular_norms: Vec<f32>,

    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
}

impl QuantizedValueStore {
    fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits,
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits,
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => Some(outlier_bits),
        };

        Self {
            gpu: None,
            gpu_mixed: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    fn extend(
        &mut self,
        encoded: &EncodedValueRows,
        outlier_encoded: Option<&EncodedValueRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
        self.regular_indices.extend_from_slice(&encoded.indices);
        self.regular_norms.extend_from_slice(&encoded.norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant value outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant value outlier indices missing")
                .extend_from_slice(&outlier.indices);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant value outlier norms missing")
                .extend_from_slice(&outlier.norms);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Complete quantised KV entry for one attention layer
// ═══════════════════════════════════════════════════════════════════════════

/// Compressed KV cache for one attention layer.
///
/// Stores all cached positions as bit-packed indices + f32 metadata.
/// Backed by [`TurboQuantState`] for dequantisation.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Compressed keys — inner-product optimised (MSE + QJL).
    pub keys: Option<QuantizedKeyStore>,
    /// Compressed values — MSE optimised.
    pub values: Option<QuantizedValueStore>,
    /// Layout from the first append (batch, heads, key_dim, value_dim).
    layout: Option<CacheLayout>,
    /// Total cached positions (cold + hot). Public for compatibility with
    /// callers that still read it directly. Maintained as the invariant
    /// `offset == cold_offset + hot_offset` after every append/rollback.
    pub offset: usize,
    /// Tokens currently sitting compressed in the cold stores. The GPU
    /// uniform attention kernels score against `cold_offset` slots.
    cold_offset: usize,
    /// Tokens currently held uncompressed in the fp16/bf16 hot ring.
    hot_offset: usize,
    /// Hot-ring keys, shape `[B, H_kv, hot_capacity, D_k]`. Native dtype
    /// (whatever was passed in on first append). `None` when the recent
    /// window is disabled or no tokens are currently uncompressed.
    hot_keys: Option<InlineArray>,
    /// Hot-ring values, same shape semantics as `hot_keys`.
    hot_values: Option<InlineArray>,
    /// Native dtype of the hot ring (taken from the first append). When the
    /// cold side is dequantized for the mixed-attention path, the cold
    /// f32/bf16 output is cast to this dtype before concatenation.
    hot_dtype: Option<i32>,
    /// Config used to build this cache.
    pub config: TurboQuantConfig,
    /// Shared pre-computed matrices and codebooks.
    pub state: Option<Arc<TurboQuantState>>,
}

#[derive(Debug, Clone, Copy)]
struct CacheLayout {
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum UniformAttentionBenchMode {
    Split,
    SpecializedQ8D128TwoPass,
    SpecializedQ8D256TwoPass,
    SpecializedQ8D256FullbytePass1,
    SpecializedQ8D256FullbytePass2,
    SpecializedQ8D256FullbyteSplitDenseV,
    SpecializedQ8D256FullbyteLocalSoftmax,
}

impl QuantizedKvCache {
    /// Create an empty cache.  `state` should be `None` on first use; call
    /// `append` to populate.
    pub fn new(config: TurboQuantConfig) -> Self {
        Self {
            keys: None,
            values: None,
            layout: None,
            offset: 0,
            cold_offset: 0,
            hot_offset: 0,
            hot_keys: None,
            hot_values: None,
            hot_dtype: None,
            config,
            state: None,
        }
    }

    /// Create with a pre-built shared state (avoids re-building QR/Lloyd-Max).
    pub fn with_state(config: TurboQuantConfig, state: Arc<TurboQuantState>) -> Self {
        let mut cache = Self::new(config);
        cache.state = Some(state);
        cache
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// True when no positions have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Reset to empty (retains pre-built state and config).
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.layout = None;
        self.hot_keys = None;
        self.hot_values = None;
        self.hot_dtype = None;
        self.offset = 0;
        self.cold_offset = 0;
        self.hot_offset = 0;
    }

    /// Number of tokens currently held uncompressed in the hot ring.
    pub fn hot_len(&self) -> usize {
        self.hot_offset
    }

    /// Number of tokens that have been compressed into the cold stores.
    pub fn cold_len(&self) -> usize {
        self.cold_offset
    }

    /// Hot-ring capacity = `recent_window + HOT_EVICTION_CHUNK` when the
    /// window is enabled, `0` when disabled (legacy compress-everything mode).
    fn hot_capacity(&self) -> usize {
        self.config
            .recent_window
            .map(|w| w + HOT_EVICTION_CHUNK)
            .unwrap_or(0)
    }

    /// Append new keys and values.
    ///
    /// `keys` and `values` must have shape `[B, H, S, D]` as f32 or bf16.
    ///
    /// For the Uniform quantisation config the entire pipeline runs on-GPU:
    /// normalise → rotate → argmin codebook → QJL projection → sign.
    /// No GPU→CPU transfer happens.  Results are stored as `InlineArray`s and
    /// concatenated along the T axis on subsequent calls.
    ///
    /// For the Mixed (outlier-aware) config the CPU path is used (outlier mask
    /// selection requires a per-row top-k sort that is not trivially vectorisable).
    ///
    /// Returns an error string on shape mismatch.
    ///
    /// Dispatches to one of two paths based on `config.recent_window`:
    /// - `None` (legacy): every appended token is compressed immediately.
    /// - `Some(N)`: the newest `N` tokens stay in an uncompressed fp16/bf16
    ///   hot ring; older history is evicted into the cold compressed stores
    ///   in `HOT_EVICTION_CHUNK` batches. Short-context callers therefore
    ///   pay zero compression overhead.
    pub fn append(&mut self, keys: &InlineArray, values: &InlineArray) -> Result<(), String> {
        let layout = self.ensure_layout(keys, values)?;
        let seq_len = keys.dim(2) as usize;
        if seq_len == 0 {
            return Ok(());
        }

        match self.config.recent_window {
            None => {
                self.compress_into_cold(keys, values, layout, seq_len)?;
                self.offset = self.cold_offset + self.hot_offset;
                Ok(())
            }
            Some(window) => self.append_with_recent_window(keys, values, layout, seq_len, window),
        }
    }

    /// Compress and accumulate `keys`/`values` (shape `[B, H, S, D]`) directly
    /// into the cold stores, advancing `self.cold_offset` by `seq_len`.
    /// Mirrors the legacy compress-immediately path.
    fn compress_into_cold(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
        layout: CacheLayout,
        seq_len: usize,
    ) -> Result<(), String> {
        let config = self.config;
        let state = self.state.get_or_insert_with(|| {
            Arc::new(TurboQuantState::new(
                layout.key_dim,
                layout.value_dim,
                config,
            ))
        });
        let state = Arc::clone(state);

        // Cast to f32 once — needed for both GPU and CPU paths.
        let keys_f32 = keys.as_dtype(10 /* float32 */);
        let values_f32 = values.as_dtype(10 /* float32 */);

        let ks = self
            .keys
            .get_or_insert_with(|| QuantizedKeyStore::new(config.keys));
        let vs = self
            .values
            .get_or_insert_with(|| QuantizedValueStore::new(config.values));

        // ── GPU path (Uniform only) ───────────────────────────────────────
        let gpu_keys_ok = matches!(config.keys, TurboQuantTensorConfig::Uniform { .. });
        let gpu_vals_ok = matches!(config.values, TurboQuantTensorConfig::Uniform { .. });

        if gpu_keys_ok && gpu_vals_ok {
            if let Some((new_ks_gpu, new_vs_gpu)) =
                gpu_quantize_kv(&state, &keys_f32, &values_f32, config)
            {
                match ks.gpu.as_mut() {
                    None => ks.gpu = Some(new_ks_gpu),
                    Some(g) => g.append(new_ks_gpu),
                }
                match vs.gpu.as_mut() {
                    None => vs.gpu = Some(new_vs_gpu),
                    Some(g) => g.append(new_vs_gpu),
                }
                self.cold_offset += seq_len;
                return Ok(());
            }
            // GPU path failed — fall through to CPU.
        }

        // ── CPU fallback path ─────────────────────────────────────────────
        let key_rows = inline_array_to_bshd_rows(&keys_f32)?;
        let value_rows = inline_array_to_bshd_rows(&values_f32)?;

        let rows_per_seq = layout.batch * layout.heads;
        debug_assert_eq!(key_rows.len(), rows_per_seq * seq_len * layout.key_dim);

        let encoded_keys = encode_key_rows(&state.keys, layout.key_dim, &key_rows);
        let encoded_values = encode_value_rows(&state.values, layout.value_dim, &value_rows);

        ks.extend(
            &encoded_keys.regular,
            encoded_keys.outlier.as_ref(),
            encoded_keys.outlier_mask.as_ref(),
        );
        vs.extend(
            &encoded_values.regular,
            encoded_values.outlier.as_ref(),
            encoded_values.outlier_mask.as_ref(),
        );

        // ── Mixed GPU path (additive, Phase 3a) ───────────────────────────
        // Mirrors the Uniform GPU path above but populates gpu_mixed instead
        // of gpu, alongside (not in lieu of) the CPU PackedBits stores. The
        // CPU stores remain authoritative for now; Phase 3c will flip
        // attention dispatch to read from gpu_mixed.
        let mixed_keys = matches!(config.keys, TurboQuantTensorConfig::Mixed { .. });
        let mixed_vals = matches!(config.values, TurboQuantTensorConfig::Mixed { .. });
        if mixed_keys && mixed_vals {
            if let Some((new_ks_gpu, new_vs_gpu)) =
                gpu_quantize_kv_mixed(&state, &keys_f32, &values_f32, config)
            {
                match ks.gpu_mixed.as_mut() {
                    None => ks.gpu_mixed = Some(new_ks_gpu),
                    Some(g) => g.append(new_ks_gpu),
                }
                match vs.gpu_mixed.as_mut() {
                    None => vs.gpu_mixed = Some(new_vs_gpu),
                    Some(g) => g.append(new_vs_gpu),
                }
            }
        }

        self.cold_offset += seq_len;
        Ok(())
    }

    /// Append into the hot fp16/bf16 ring, evicting oldest tokens to the cold
    /// stores once the ring exceeds `window + HOT_EVICTION_CHUNK`.
    fn append_with_recent_window(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
        layout: CacheLayout,
        seq_len: usize,
        window: usize,
    ) -> Result<(), String> {
        // Lock in the hot dtype on first append. All subsequent writes go
        // through `as_dtype` so the ring always carries a single dtype.
        if self.hot_dtype.is_none() {
            self.hot_dtype = Some(keys.dtype_raw());
        }
        let hot_dtype = self.hot_dtype.unwrap();
        let capacity = self.hot_capacity().max(seq_len);

        // Lazy-allocate (or grow) the hot ring.
        if self.hot_keys.is_none() {
            self.hot_keys = Some(InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            ));
            self.hot_values = Some(InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            ));
        } else if self.hot_offset + seq_len > capacity {
            // One-shot prefill larger than the current ring — grow to fit.
            let need = self.hot_offset + seq_len;
            let new_cap = need.max(capacity);
            let prev_keys = self.hot_keys.take().unwrap();
            let prev_values = self.hot_values.take().unwrap();
            let mut new_keys = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    new_cap as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            );
            let mut new_values = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    new_cap as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            );
            if self.hot_offset > 0 {
                let kept_keys = prev_keys.slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                let kept_values = prev_values.slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
                new_keys = new_keys.slice_set(
                    &kept_keys,
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                new_values = new_values.slice_set(
                    &kept_values,
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
            }
            self.hot_keys = Some(new_keys);
            self.hot_values = Some(new_values);
        }

        // Write the new chunk into the ring at `hot_offset..hot_offset+seq_len`.
        let start = self.hot_offset;
        let stop = start + seq_len;
        let keys_typed = keys.as_dtype(hot_dtype);
        let values_typed = values.as_dtype(hot_dtype);
        {
            let hot_keys = self.hot_keys.as_mut().expect("hot_keys allocated above");
            *hot_keys = hot_keys.slice_set(
                &keys_typed,
                &[0, 0, start as i32, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    stop as i32,
                    layout.key_dim as i32,
                ],
            );
        }
        {
            let hot_values = self.hot_values.as_mut().expect("hot_values allocated above");
            *hot_values = hot_values.slice_set(
                &values_typed,
                &[0, 0, start as i32, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    stop as i32,
                    layout.value_dim as i32,
                ],
            );
        }
        self.hot_offset += seq_len;
        self.offset = self.cold_offset + self.hot_offset;

        // Drain the oldest tokens once the ring fills past `window + chunk`.
        // Eviction is amortized: each call moves `min(overflow, chunk)` tokens
        // instead of one-token-at-a-time shuffles.
        while self.hot_offset > window + HOT_EVICTION_CHUNK {
            let evict_seq = self
                .hot_offset
                .saturating_sub(window)
                .min(HOT_EVICTION_CHUNK);
            self.evict_oldest_to_cold(layout, evict_seq)?;
        }
        Ok(())
    }

    /// Move the leading `evict_seq` tokens out of the hot ring into cold,
    /// then shuffle the remaining tail back to position 0 of the buffer.
    /// Caller must guarantee `evict_seq <= self.hot_offset`.
    fn evict_oldest_to_cold(
        &mut self,
        layout: CacheLayout,
        evict_seq: usize,
    ) -> Result<(), String> {
        if evict_seq == 0 {
            return Ok(());
        }
        let remain = self.hot_offset - evict_seq;
        let hot_dtype = self
            .hot_dtype
            .expect("hot_dtype set when hot ring is non-empty");

        // Phase 1: extract the leading slice we want to compress and the
        // trailing slice we want to keep, into owned values. The immutable
        // borrows of `hot_keys`/`hot_values` are dropped at the end of this
        // block before any `&mut self` call below.
        let (evict_keys, evict_values, kept) = {
            let hot_keys = self.hot_keys.as_ref().ok_or_else(|| {
                "TurboQuant hot keys missing during evict".to_string()
            })?;
            let hot_values = self.hot_values.as_ref().ok_or_else(|| {
                "TurboQuant hot values missing during evict".to_string()
            })?;
            let evict_keys = hot_keys.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    evict_seq as i32,
                    layout.key_dim as i32,
                ],
            );
            let evict_values = hot_values.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    evict_seq as i32,
                    layout.value_dim as i32,
                ],
            );
            let kept = if remain > 0 {
                let kept_keys = hot_keys.slice(
                    &[0, 0, evict_seq as i32, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                let kept_values = hot_values.slice(
                    &[0, 0, evict_seq as i32, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
                Some((kept_keys, kept_values))
            } else {
                None
            };
            (evict_keys, evict_values, kept)
        };

        // Phase 2: mutate. The borrows above are dropped.
        self.compress_into_cold(&evict_keys, &evict_values, layout, evict_seq)?;
        if let Some((kept_keys, kept_values)) = kept {
            let capacity = self.hot_capacity().max(remain);
            let mut new_keys = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            );
            let mut new_values = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            );
            new_keys = new_keys.slice_set(
                &kept_keys,
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    remain as i32,
                    layout.key_dim as i32,
                ],
            );
            new_values = new_values.slice_set(
                &kept_values,
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    remain as i32,
                    layout.value_dim as i32,
                ],
            );
            self.hot_keys = Some(new_keys);
            self.hot_values = Some(new_values);
        } else {
            // Hot ring fully drained — drop the buffers until the next append.
            self.hot_keys = None;
            self.hot_values = None;
        }
        self.hot_offset = remain;
        self.offset = self.cold_offset + self.hot_offset;
        Ok(())
    }

    /// Dequantise and return all cached keys as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32). The returned tensor includes both the compressed
    /// cold history and the uncompressed hot tail (the latter cast to f32).
    ///
    /// Uses the GPU path when keys were quantised on-GPU; otherwise falls back
    /// to the CPU decode path.
    pub fn dequantize_keys(&self) -> Option<InlineArray> {
        let layout = self.layout?;

        let cold = if self.cold_offset > 0 {
            let ks = self.keys.as_ref()?;
            let state = self.state.as_ref()?;
            if let Some(ref g) = ks.gpu {
                let TurboQuantTensorConfig::Uniform { bits } = self.config.keys else {
                    unreachable!("GPU store only exists for Uniform config")
                };
                Some(gpu_dequantize_keys(g, &state.keys, bits)?)
            } else {
                let rows = decode_key_rows(&state.keys, layout.key_dim, ks);
                Some(f32_rows_to_bhsd_array(
                    &rows,
                    layout.batch,
                    layout.heads,
                    self.cold_offset,
                    layout.key_dim,
                ))
            }
        } else {
            None
        };

        let hot = if self.hot_offset > 0 {
            let hot_keys = self.hot_keys.as_ref()?;
            let active = hot_keys.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    self.hot_offset as i32,
                    layout.key_dim as i32,
                ],
            );
            // Cold is f32; cast hot to f32 so concat dtypes match.
            Some(active.as_dtype(10))
        } else {
            None
        };

        let mut combined = match (cold, hot) {
            (Some(c), Some(h)) => c.concatenate_2(&h, 2),
            (Some(c), None) => c,
            (None, Some(h)) => h,
            (None, None) => return None,
        };
        let mut to_eval = vec![&mut combined];
        crate::inline_array::eval_and_detach_many(&mut to_eval);
        Some(combined)
    }

    /// Dequantise and return all cached values as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32). Includes both compressed cold history and the
    /// uncompressed hot tail (cast to f32).
    pub fn dequantize_values(&self) -> Option<InlineArray> {
        let layout = self.layout?;

        let cold = if self.cold_offset > 0 {
            let vs = self.values.as_ref()?;
            let state = self.state.as_ref()?;
            if let Some(ref g) = vs.gpu {
                let TurboQuantTensorConfig::Uniform { bits } = self.config.values else {
                    unreachable!("GPU store only exists for Uniform config")
                };
                Some(gpu_dequantize_values(g, &state.values, bits)?)
            } else {
                let rows = decode_value_rows(&state.values, layout.value_dim, vs);
                Some(f32_rows_to_bhsd_array(
                    &rows,
                    layout.batch,
                    layout.heads,
                    self.cold_offset,
                    layout.value_dim,
                ))
            }
        } else {
            None
        };

        let hot = if self.hot_offset > 0 {
            let hot_values = self.hot_values.as_ref()?;
            let active = hot_values.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    self.hot_offset as i32,
                    layout.value_dim as i32,
                ],
            );
            Some(active.as_dtype(10))
        } else {
            None
        };

        let mut combined = match (cold, hot) {
            (Some(c), Some(h)) => c.concatenate_2(&h, 2),
            (Some(c), None) => c,
            (None, Some(h)) => h,
            (None, None) => return None,
        };
        let mut to_eval = vec![&mut combined];
        crate::inline_array::eval_and_detach_many(&mut to_eval);
        Some(combined)
    }

    /// Evaluate and detach GPU-resident cache arrays to keep graph chains short.
    pub fn eval_and_detach_gpu_state(&mut self) {
        let mut to_eval = Vec::new();
        if let Some(keys) = &mut self.keys {
            if let Some(gpu) = &mut keys.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
            if let Some(gpu) = &mut keys.gpu_mixed {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if let Some(values) = &mut self.values {
            if let Some(gpu) = &mut values.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
            if let Some(gpu) = &mut values.gpu_mixed {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if let Some(hot) = self.hot_keys.as_mut() {
            to_eval.push(hot);
        }
        if let Some(hot) = self.hot_values.as_mut() {
            to_eval.push(hot);
        }
        if !to_eval.is_empty() {
            crate::inline_array::eval_and_detach_many(&mut to_eval);
        }
    }

    /// Append a single-token KV chunk and compute attention output.
    ///
    /// Dispatch:
    /// - **Hot-only** (`cold_offset == 0`, common for short prompts when the
    ///   recent fp16 window is enabled): run standard SDPA against the active
    ///   prefix of the hot ring. No quantization round-trip on the decode path.
    /// - **Cold-only** (`hot_offset == 0`, recent window disabled or fully
    ///   evicted): try the optimized GPU TurboQuant attention kernels.
    /// - **Mixed**: dequantize the cold cache, concat with the active hot
    ///   suffix, run standard SDPA. Correctness-first; a future v2 may
    ///   score hot directly + cold compressedly in one kernel.
    pub fn append_and_compute_attention(
        &mut self,
        queries: &InlineArray,
        keys: &InlineArray,
        values: &InlineArray,
        scale: f32,
    ) -> Result<InlineArray, String> {
        if queries.ndim() != 4
            || keys.ndim() != 4
            || values.ndim() != 4
            || queries.dim(2) != 1
            || keys.dim(2) != 1
            || values.dim(2) != 1
        {
            return Err(
                "TurboQuant direct attention requires [B, H, 1, D] single-token decode inputs"
                    .to_string(),
            );
        }

        let layout = self.ensure_layout(keys, values)?;
        self.append(keys, values)?;
        let query_dtype = queries.dtype_raw();
        let queries_f32 = if query_dtype == 10 {
            queries.clone()
        } else {
            queries.as_dtype(10)
        };

        // Hot-only: standard fused SDPA against the fp16/bf16 hot ring.
        // Skips both the GPU TurboQuant kernels and any dequantize allocation.
        if self.cold_offset == 0 && self.hot_offset > 0 {
            let hot_keys = self
                .hot_keys
                .as_ref()
                .ok_or_else(|| "TurboQuant hot keys missing".to_string())?;
            let hot_values = self
                .hot_values
                .as_ref()
                .ok_or_else(|| "TurboQuant hot values missing".to_string())?;
            let active_keys = hot_keys
                .slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                )
                .as_dtype(10);
            let active_values = hot_values
                .slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                )
                .as_dtype(10);

            let q_heads = queries.dim(1) as usize;
            let kv_heads = layout.heads;
            let (keys_for_attn, values_for_attn) = if q_heads == kv_heads {
                (active_keys, active_values)
            } else {
                let groups = q_heads / kv_heads;
                if groups * kv_heads != q_heads {
                    return Err(format!(
                        "TurboQuant GQA mismatch: query heads {q_heads} not divisible by kv heads {kv_heads}"
                    ));
                }
                (
                    active_keys.repeat(groups as i32, 1),
                    active_values.repeat(groups as i32, 1),
                )
            };
            let output = crate::decode::sdpa_causal_like_mlx(
                &queries_f32,
                &keys_for_attn,
                &values_for_attn,
                scale,
                queries.dim(2),
            );
            return Ok(if query_dtype == 10 {
                output
            } else {
                output.as_dtype(query_dtype)
            });
        }

        // Cold-only: optimized GPU TurboQuant kernels. The score-against-cold
        // assumption inside `try_gpu_uniform_attention` is only correct when
        // the hot ring is empty.
        if self.hot_offset == 0 {
            if let Some(output) =
                self.try_gpu_uniform_attention(&queries_f32, layout, scale, query_dtype)
            {
                return Ok(if query_dtype == 10 || output.dtype_raw() == query_dtype {
                    output
                } else {
                    output.as_dtype(query_dtype)
                });
            }
        }

        // Mixed (or cold-only fallback): dequantize the full cache (cold +
        // hot tail) and run standard SDPA. dequantize_keys/values returns
        // concat(cold_f32, hot_f32) automatically.
        let full_keys = self
            .dequantize_keys()
            .ok_or_else(|| "TurboQuant failed to dequantize keys".to_string())?;
        let full_values = self
            .dequantize_values()
            .ok_or_else(|| "TurboQuant failed to dequantize values".to_string())?;

        let q_heads = queries.dim(1) as usize;
        let kv_heads = layout.heads;
        let (keys_for_attn, values_for_attn) = if q_heads == kv_heads {
            (full_keys, full_values)
        } else {
            let groups = q_heads / kv_heads;
            if groups * kv_heads != q_heads {
                return Err(format!(
                    "TurboQuant GQA mismatch: query heads {q_heads} not divisible by kv heads {kv_heads}"
                ));
            }
            (
                full_keys.repeat(groups as i32, 1),
                full_values.repeat(groups as i32, 1),
            )
        };

        let queries_f32 = queries.as_dtype(10);
        let output = crate::decode::sdpa_causal_like_mlx(
            &queries_f32,
            &keys_for_attn,
            &values_for_attn,
            scale,
            queries.dim(2),
        );
        Ok(if queries.dtype_raw() == 10 {
            output
        } else {
            output.as_dtype(queries.dtype_raw())
        })
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_core_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
        mode: UniformAttentionBenchMode,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let kv_heads_i32 = layout.heads as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq || kv_heads_i32 <= 0 {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();

        match mode {
            UniformAttentionBenchMode::SpecializedQ8D128TwoPass => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 128
                    || value_dim != 128
                    || qjl_words != 4
                {
                    return None;
                }
                let key_indices =
                    ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                let key_qjl_signs =
                    ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                let value_indices =
                    vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
                InlineArray::turboquant_attention_q8_d128_2pass(
                    query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_indices,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256TwoPass => self
                .try_gpu_uniform_attention_q8_d256_precomputed(
                    query_rot,
                    Some(query_proj),
                    q_heads,
                    scale,
                ),
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass1 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass2 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                let (partials, sums, maxs) = self
                    .bench_gpu_uniform_attention_state_precomputed_fullbyte(
                        query_rot, q_heads, scale,
                    )?;
                InlineArray::turboquant_attention_q8_d256_pass2_merge(
                    &partials,
                    &sums,
                    &maxs,
                    q_rows as u32,
                    sums.dim(1) as u32,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteSplitDenseV => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    let scores = InlineArray::turboquant_score_q8_d256_fullbyte(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )?;
                    let weights = scores.softmax(-1);
                    InlineArray::turboquant_weighted_sum_d256_dense_values(
                        &weights,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteLocalSoftmax => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::Split => {
                let scores = self
                    .bench_gpu_uniform_scores_precomputed(query_rot, query_proj, q_heads, scale)?;
                let weights = scores.softmax(-1);
                InlineArray::turboquant_weighted_decode(
                    &weights,
                    &vs.indices_t_array()?,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    value_dim as u32,
                    1u32 << value_bits,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let key_rot = key_core.inverse_rotation_arr.as_ref()?;
        let key_proj = key_core.inverse_qjl_arr.as_ref()?;
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((query_rows.matmul(key_rot), query_rows.matmul(key_proj)))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms_wht(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        if key_dim != 256 {
            return None;
        }
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((
            key_core.rotate_rows_wht(&query_rows)?,
            key_core.project_rows_wht(&query_rows)?,
        ))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_output_inverse_rotate_wht(
        &self,
        decoded_rot: &InlineArray,
    ) -> Option<InlineArray> {
        let state = self.state.as_ref()?;
        let value_core = match &state.values {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let dim = decoded_rot.dim(1);
        if dim != 256 {
            return None;
        }
        value_core.inverse_rotate_rows_wht(decoded_rot)
    }
    fn try_gpu_uniform_attention_q8_d256_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: Option<&InlineArray>,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || value_bits != 8
            || key_dim != 256
            || value_dim != 256
            || n_seq < 1024
            || q_rows <= 0
            || q_heads <= 0
            || (q_heads % layout.heads as i32) != 0
            || (q_heads / layout.heads as i32) > 8
            || cache_seq_capacity < n_seq
        {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            let qjl_words = ks.qjl_words();
            if qjl_words != 8 {
                return None;
            }
            let query_proj = query_proj?;
            if let (Some(key_bytes), Some(slot_scales), Some(value_rot_dense)) = (
                ks.q8_keybytes_seq.as_ref(),
                ks.q8_slot_scales_seq.as_ref(),
                vs.d256_rot_values_seq.as_ref(),
            ) {
                InlineArray::turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else if let (Some(kv_bytes), Some(slot_scales)) =
                (ks.q8_kvbytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                if let Some(value_rot_dense) = vs.d256_rot_values_seq.as_ref() {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        value_core.codebook_arr(value_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                }
            } else if let (Some(key_bytes), Some(slot_scales)) =
                (ks.q8_keybytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                InlineArray::turboquant_attention_q8_d256_packed_keys_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices
                        .as_ref()?
                        .reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else {
                InlineArray::turboquant_attention_q8_d256_2pass(
                    query_rot,
                    query_proj,
                    &ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]),
                    &ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]),
                    &ks.key_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    &ks.residual_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]),
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        let qjl_words = ks.qjl_words();
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        if key_bits == 8
            && key_dim == 256
            && qjl_words == 8
            && q_heads > 0
            && (q_heads % layout.heads as i32) == 0
            && (q_heads / layout.heads as i32) <= 8
        {
            if let Some(scores) = InlineArray::turboquant_score_q8_d256(
                query_rot,
                query_proj,
                &ks.indices_t_array(),
                &ks.qjl_signs_t_array(),
                &key_norms,
                &key_residual_norms,
                key_core.codebook_arr(key_bits.saturating_sub(1))?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            ) {
                return Some(scores);
            }
        }
        InlineArray::turboquant_score(
            query_rot,
            query_proj,
            &ks.indices_t_array(),
            &ks.qjl_signs_t_array(),
            &key_norms,
            &key_residual_norms,
            key_core.codebook_arr(key_bits.saturating_sub(1))?,
            key_dim as u32,
            qjl_words as u32,
            key_core.codebook_arr(key_bits.saturating_sub(1))?.dim(0) as u32,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
            scale,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq <= 0
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales)) =
            (ks.q8_fullbyte_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
        {
            InlineArray::turboquant_score_q8_d256_fullbyte(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_state_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<(InlineArray, InlineArray, InlineArray)> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || value_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq < 1024
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_decode(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let value_dim = layout.value_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.cold_offset as i32;
        let indices_t = vs.indices_t_array()?;
        InlineArray::turboquant_weighted_decode(
            weights,
            &indices_t,
            &vs.norms_array()?.reshape(&[kv_rows, indices_t.dim(3)]),
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            indices_t.dim(3) as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_sum_dense_values(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let value_dim = layout.value_dim as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.cold_offset as i32;
        let value_rot_dense = vs.d256_rot_values_seq.as_ref()?;
        let cache_seq_capacity = value_rot_dense.dim(2);
        if value_dim != 256 || q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        InlineArray::turboquant_weighted_sum_d256_dense_values(
            weights,
            &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    fn try_gpu_uniform_attention(
        &self,
        queries_f32: &InlineArray,
        layout: CacheLayout,
        scale: f32,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };

        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let value_dim = layout.value_dim as i32;
        let q_rows = batch * q_heads;
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }

        let trace_timing = turboquant_trace_enabled();
        let query_ready_us = if trace_timing {
            eval_stage_micros(queries_f32)
        } else {
            0
        };
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        let can_try_q8_fullbyte = turboquant_q8_fullbyte_enabled()
            && key_bits == 8
            && value_bits == 8
            && key_dim == 256
            && value_dim == 256
            && n_seq >= 1024
            && ks.q8_fullbyte_seq.is_some()
            && ks.q8_slot_scales_seq.is_some()
            && vs.d256_rot_values_seq.is_some();
        let mut project_us = 0;
        // Fused rotate+project: saves one dispatch per layer by doing
        // `input @ [inv_rot | inv_qjl]` as a single [N, 2*dim] matmul
        // instead of two separate [N, dim] matmuls. Only applied when
        // both outputs are needed (i.e., the q8 fullbyte fast path is
        // not taken). Falls back to sequential calls if the stacked
        // matrix wasn't built.
        let (query_rot, mut query_proj) = if !can_try_q8_fullbyte {
            if let Some((rot, proj)) = key_core.rotate_and_project_array(&query_rows) {
                (rot, Some(proj))
            } else {
                let rot = key_core.rotate_array(&query_rows)?;
                let proj = key_core.project_array(&query_rows)?;
                (rot, Some(proj))
            }
        } else {
            (key_core.rotate_array(&query_rows)?, None)
        };
        let rotate_us = if trace_timing {
            eval_stage_micros(&query_rot)
        } else {
            0
        };
        if let Some(proj) = query_proj.as_ref() {
            if trace_timing {
                project_us = eval_stage_micros(proj);
            }
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();
        if can_try_q8_fullbyte {
            if let Some(decoded_rot) =
                self.try_gpu_uniform_attention_q8_d256_precomputed(&query_rot, None, q_heads, scale)
            {
                let decode_us = if trace_timing {
                    eval_stage_micros(&decoded_rot)
                } else {
                    0
                };
                let output_rows =
                    value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                let inverse_rotate_us = if trace_timing {
                    eval_stage_micros(&output_rows)
                } else {
                    0
                };
                if trace_timing {
                    trace_turboquant_bridge(&format!(
                        "gpu_uniform_q8_d256_fullbyte_densev_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                        n_seq,
                        q_rows,
                        query_ready_us,
                        rotate_us,
                        project_us,
                        decode_us,
                        inverse_rotate_us,
                    ));
                }
                return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
            }
        }
        if query_proj.is_none() {
            let projected = key_core.project_array(&query_rows)?;
            if trace_timing {
                project_us = eval_stage_micros(&projected);
            }
            query_proj = Some(projected);
        }
        let query_proj = query_proj.as_ref()?;
        let key_codebook = key_core.codebook_arr(key_bits.saturating_sub(1))?;
        if let Some(decoded_rot) = self.try_gpu_uniform_attention_q8_d256_precomputed(
            &query_rot,
            Some(query_proj),
            q_heads,
            scale,
        ) {
            let decode_us = if trace_timing {
                eval_stage_micros(&decoded_rot)
            } else {
                0
            };
            let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
            let inverse_rotate_us = if trace_timing {
                eval_stage_micros(&output_rows)
            } else {
                0
            };
            if trace_timing {
                trace_turboquant_bridge(&format!(
                    "gpu_uniform_q8_d256_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                    n_seq,
                    q_rows,
                    query_ready_us,
                    rotate_us,
                    project_us,
                    decode_us,
                    inverse_rotate_us,
                ));
            }
            return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
        }
        if key_bits == 8 && value_bits == 8 && key_dim == 128 && value_dim == 128 && n_seq >= 1024 {
            let key_indices = ks
                .indices_t_array()
                .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
            let value_indices =
                vs.indices_t_array()?
                    .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
            let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);

            if q_heads > 8 {
                if let Some(key_bytes) = ks.q8_keybytes_t.as_ref() {
                    let key_bytes = key_bytes.reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                    if let Some(decoded_rot) =
                        InlineArray::turboquant_attention_q8_d128_packed_keys_2pass(
                            &query_rot,
                            query_proj,
                            &key_bytes,
                            &key_norms,
                            &key_residual_norms,
                            key_codebook,
                            &value_indices,
                            &value_norms,
                            value_core.codebook_arr(value_bits)?,
                            q_rows as u32,
                            n_seq as u32,
                            cache_seq_capacity as u32,
                            q_heads as u32,
                            layout.heads as u32,
                            scale,
                        )
                    {
                        let decode_us = if trace_timing {
                            eval_stage_micros(&decoded_rot)
                        } else {
                            0
                        };
                        let output_rows =
                            value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                        let inverse_rotate_us = if trace_timing {
                            eval_stage_micros(&output_rows)
                        } else {
                            0
                        };
                        if trace_timing {
                            trace_turboquant_bridge(&format!(
                                "gpu_uniform_q8_d128_packed_keys_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                                n_seq,
                                q_rows,
                                query_ready_us,
                                rotate_us,
                                project_us,
                                decode_us,
                                inverse_rotate_us,
                            ));
                        }
                        return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                    }
                }
            } else if qjl_words == 4 {
                let key_qjl_signs =
                    ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                if let Some(decoded_rot) = InlineArray::turboquant_attention_q8_d128_2pass(
                    &query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    key_codebook,
                    &value_indices,
                    &value_norms,
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                ) {
                    let decode_us = if trace_timing {
                        eval_stage_micros(&decoded_rot)
                    } else {
                        0
                    };
                    let output_rows =
                        value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                    let inverse_rotate_us = if trace_timing {
                        eval_stage_micros(&output_rows)
                    } else {
                        0
                    };
                    if trace_timing {
                        trace_turboquant_bridge(&format!(
                            "gpu_uniform_q8_d128_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                            n_seq,
                            q_rows,
                            query_ready_us,
                            rotate_us,
                            project_us,
                            decode_us,
                            inverse_rotate_us,
                        ));
                    }
                    return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                }
            }
        }

        let scores =
            self.bench_gpu_uniform_scores_precomputed(&query_rot, query_proj, q_heads, scale)?;
        let score_us = if trace_timing {
            eval_stage_micros(&scores)
        } else {
            0
        };
        let weights = scores.softmax(-1);
        let softmax_us = if trace_timing {
            eval_stage_micros(&weights)
        } else {
            0
        };
        let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);
        let decoded_rot = InlineArray::turboquant_weighted_decode(
            &weights,
            &vs.indices_t_array()?,
            &value_norms,
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )?;
        let decode_us = if trace_timing {
            eval_stage_micros(&decoded_rot)
        } else {
            0
        };
        let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
        let inverse_rotate_us = if trace_timing {
            eval_stage_micros(&output_rows)
        } else {
            0
        };
        if trace_timing {
            trace_turboquant_bridge(&format!(
                "gpu_uniform_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score={} softmax={} decode={} inverse_rotate={}",
                n_seq,
                q_rows,
                query_ready_us,
                rotate_us,
                project_us,
                score_us,
                softmax_us,
                decode_us,
                inverse_rotate_us
            ));
        }
        Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]))
    }

    fn ensure_layout(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
    ) -> Result<CacheLayout, String> {
        // Validate shape: [B, H, S, D]
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(format!(
                "TurboQuant: expected 4-D keys/values, got ndim {} / {}",
                keys.ndim(),
                values.ndim()
            ));
        }

        let b = keys.dim(0) as usize;
        let h = keys.dim(1) as usize;
        let kd = keys.dim(3) as usize;
        let vd = values.dim(3) as usize;

        if let Some(existing) = self.layout {
            if existing.batch != b
                || existing.heads != h
                || existing.key_dim != kd
                || existing.value_dim != vd
            {
                return Err(format!(
                    "TurboQuant: layout mismatch — expected [{b},{h},*,{kd}] / [{b},{h},*,{vd}]"
                ));
            }
            return Ok(existing);
        }

        let layout = CacheLayout {
            batch: b,
            heads: h,
            key_dim: kd,
            value_dim: vd,
        };
        self.layout = Some(layout);
        Ok(layout)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public convenience constructors
// ═══════════════════════════════════════════════════════════════════════════

/// Build a shared [`TurboQuantState`] for the given dimensions and config.
///
/// This is the expensive step (~100 ms per unique dim).  Call once at model
/// load time and share the `Arc` across all layers.
///
/// ## GPU-attention coverage gaps
///
/// Not every `(config, dim)` combination has a fused GPU attention path; see
/// [`TurboQuantState::has_gpu_attention_support`] for the full predicate.
/// Mixed-precision configs (`outlier_count > 0`) currently fall back to
/// dequantize+SDPA at attention time. We log a warning at state-build time
/// when that fallback will be hit so the cost is visible during model load
/// rather than being discovered as a perf regression at first decode.
pub fn build_state(
    key_dim: usize,
    value_dim: usize,
    config: TurboQuantConfig,
) -> Arc<TurboQuantState> {
    let state = Arc::new(TurboQuantState::new(key_dim, value_dim, config));
    if !state.has_gpu_attention_support() {
        eprintln!(
            "[turboquant] WARNING: state for key_dim={key_dim} value_dim={value_dim} \
             will fall back to dequantize+SDPA at attention time. \
             Reason: {}",
            state.gpu_attention_unsupported_reason()
        );
    }
    state
}

/// Look up the dispatch reason for a particular state — returned to callers
/// who want to surface "your config is on the slow path" diagnostics in
/// model-load logs without re-implementing the predicate.
pub fn gpu_attention_unsupported_reason_for(
    state: &TurboQuantState,
) -> Option<&'static str> {
    if state.has_gpu_attention_support() {
        None
    } else {
        Some(state.gpu_attention_unsupported_reason())
    }
}

/// Create a [`QuantizedKvCache`] that will lazily build its state on first use.
pub fn new_cache(config: TurboQuantConfig) -> QuantizedKvCache {
    QuantizedKvCache::new(config)
}

/// Create a [`QuantizedKvCache`] with a pre-built shared state.
pub fn new_cache_with_state(
    config: TurboQuantConfig,
    state: Arc<TurboQuantState>,
) -> QuantizedKvCache {
    QuantizedKvCache::with_state(config, state)
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-native quantise / dequantise
// ═══════════════════════════════════════════════════════════════════════════

/// Quantise keys and values entirely on GPU.
///
/// Returns `None` if the GPU path is unavailable (e.g. missing codebook arr).
/// On success returns `(GpuKeyStore, GpuValueStore)` — both shapes `[B, H, S, *]`.
///
/// Algorithm:
///   1. Normalise onto unit sphere; keep per-vector L2 norm.
///   2. Rotate: x_norm @ rotation.T  → rotated   [B, H, S, D]
///   3. Nearest-centroid: argmin over squared diffs to codebook [C] → indices [B, H, S, D]
///   4. (Keys) Reconstruct MSE approx, compute residual, project via J, take sign.
fn gpu_quantize_kv(
    state: &TurboQuantState,
    keys: &InlineArray,   // [B, H, S, Dk]  f32
    values: &InlineArray, // [B, H, S, Dv]  f32
    config: TurboQuantConfig,
) -> Option<(GpuKeyStore, GpuValueStore)> {
    let TurboQuantTensorConfig::Uniform { bits: key_bits } = config.keys else {
        return None;
    };
    let TurboQuantTensorConfig::Uniform { bits: val_bits } = config.values else {
        return None;
    };
    let key_mse_bits = key_bits.saturating_sub(1);

    let k_core = match &state.keys {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };
    let v_core = match &state.values {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // ── Keys ─────────────────────────────────────────────────────────────
    // 1. L2 norm along D axis, keepdims → [B, H, S, 1]
    let key_norms = keys.norm_l2(-1, true);
    // 2. Normalise: x / max(norm, eps)
    let eps = InlineArray::from_f32(ZERO_EPSILON);
    let safe_norms_k = key_norms.maximum(&eps);
    let k_norm = keys.divide(&safe_norms_k);

    // 3. Rotate: k_norm @ rotation.T  (CPU: matmul_rows(rotation, dim, input) = input @ rotation.T)
    //    rotation.T == inverse_rotation, so we matmul with inverse_rotation_arr.
    let k_rot = k_core.rotate_array(&k_norm)?;

    // 4. GPU nearest-centroid → [B, H, S, D] uint32
    let k_indices = k_core.gpu_quantize_mse(&k_rot, key_mse_bits)?;

    // 5. Reconstruct MSE approximation in the rotated space.
    let k_mse_recon_rot = k_core.gpu_reconstruct_mse(&k_indices, key_mse_bits)?;

    // 6. Residual norms: rotation-invariant, so compute directly in rotated space.
    //    residual_rot = k_rot - k_mse_recon_rot  →  norm_l2  [B, H, S, 1]
    //    Defensive clip to [0, MAX_RESIDUAL_NORM]: IEEE fmax/fmin sanitize NaN
    //    to 0 (via maximum with 0) and cap Inf to the bound. Real residual
    //    norms sit well below this range.
    let k_residual_rot = k_rot.subtract(&k_mse_recon_rot);
    let k_residual_norms_raw = k_residual_rot.norm_l2(-1, true);
    let zero_bound = InlineArray::from_f32(0.0f32);
    let upper_bound = InlineArray::from_f32(MAX_RESIDUAL_NORM);
    let k_residual_norms = k_residual_norms_raw
        .maximum(&zero_bound)
        .minimum(&upper_bound);

    // 7. QJL: project the residual in the **unrotated** space.
    //    residual_unrot = k_mse_recon_rot @ rotation_arr  (inverse-rotate the rotated reconstruction)
    //    then: residual_unrot = k_norm - inv_rotate(k_mse_recon_rot)
    //    QJL: residual_unrot @ inverse_qjl_arr  (= residual @ qjl.T)
    let k_mse_recon_unrot = k_core.inverse_rotate_array(&k_mse_recon_rot)?;
    let k_residual_unrot = k_norm.subtract(&k_mse_recon_unrot);
    let k_qjl_proj = k_core.project_array(&k_residual_unrot)?;
    let qjl_shape = k_qjl_proj.shape();
    let qjl_ndim = qjl_shape.len();
    let qjl_rows: i32 = qjl_shape[..qjl_ndim - 1].iter().product();
    let packed_dim = packed_qjl_words(k_core.dim) as i32;
    let k_qjl_proj_2d = if qjl_ndim == 2 {
        k_qjl_proj.clone()
    } else {
        k_qjl_proj.reshape(&[qjl_rows, k_core.dim as i32])
    };
    let k_qjl_signs = InlineArray::turboquant_pack_sign_bits(
        &k_qjl_proj_2d,
        k_core.dim as u32,
        packed_dim as u32,
        qjl_rows as u32,
    )?;
    let k_qjl_signs = if qjl_ndim == 2 {
        k_qjl_signs
    } else {
        let mut packed_shape: Vec<i32> = qjl_shape[..qjl_ndim - 1].to_vec();
        packed_shape.push(packed_dim);
        k_qjl_signs.reshape(&packed_shape)
    };
    let use_q8_seq_shadow = key_bits == 8 && k_core.dim == 256 && v_core.dim == 256;
    let k_indices_t = (!use_q8_seq_shadow).then(|| k_indices.transpose_axes(&[0, 1, 3, 2]));
    let k_qjl_signs_t = (!use_q8_seq_shadow).then(|| k_qjl_signs.transpose_axes(&[0, 1, 3, 2]));
    let q8_pack_inputs = if key_bits == 8 {
        let kv_rows = (keys.dim(0) * keys.dim(1)) as u32;
        let seq = keys.dim(2) as u32;
        let indices_t_3d = if let Some(indices_t) = k_indices_t.as_ref() {
            indices_t.reshape(&[kv_rows as i32, k_core.dim as i32, seq as i32])
        } else {
            k_indices.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                k_core.dim as i32,
                seq as i32,
            ])
        };
        let qjl_signs_t_3d = if let Some(qjl_signs_t) = k_qjl_signs_t.as_ref() {
            qjl_signs_t.reshape(&[kv_rows as i32, packed_dim, seq as i32])
        } else {
            k_qjl_signs.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                packed_dim,
                seq as i32,
            ])
        };
        Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d))
    } else {
        None
    };
    let q8_keybytes_t = if use_q8_seq_shadow {
        None
    } else if let Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d)) = q8_pack_inputs.as_ref() {
        InlineArray::turboquant_pack_q8_keybytes(
            indices_t_3d,
            qjl_signs_t_3d,
            k_core.dim as u32,
            packed_dim as u32,
            *kv_rows,
            *seq,
        )
        .map(|packed| packed.reshape(&[keys.dim(0), keys.dim(1), k_core.dim as i32, keys.dim(2)]))
    } else {
        None
    };
    let q8_keybytes_seq = if use_q8_seq_shadow {
        q8_pack_inputs
            .as_ref()
            .and_then(|(kv_rows, seq, indices_t_3d, qjl_signs_t_3d)| {
                InlineArray::turboquant_pack_q8_keybytes_seq(
                    indices_t_3d,
                    qjl_signs_t_3d,
                    k_core.dim as u32,
                    packed_dim as u32,
                    *kv_rows,
                    *seq,
                )
                .map(|packed| {
                    packed.reshape(&[keys.dim(0), keys.dim(1), keys.dim(2), k_core.dim as i32])
                })
            })
    } else {
        None
    };
    let q8_fullbyte_seq = if use_q8_seq_shadow && turboquant_q8_fullbyte_enabled() {
        k_core
            .gpu_quantize_mse(&k_rot, 8)
            .map(|indices| indices.as_dtype(Dtype::Uint8.as_i32()))
    } else {
        None
    };

    // ── Values ────────────────────────────────────────────────────────────
    let (v_indices, v_indices_t, val_norms, d256_rot_values_seq) = if use_q8_seq_shadow {
        (
            None,
            None,
            None,
            Some(
                v_core
                    .rotate_array(values)?
                    .as_dtype(Dtype::Bfloat16.as_i32()),
            ),
        )
    } else {
        let val_norms = values.norm_l2(-1, true);
        let safe_norms_v = val_norms.maximum(&eps);
        let v_norm = values.divide(&safe_norms_v);

        // v_norm @ rotation.T = v_norm @ inverse_rotation_arr
        let v_rot = v_core.rotate_array(&v_norm)?;
        let v_indices = v_core.gpu_quantize_mse(&v_rot, val_bits)?;
        let v_indices_t = Some(v_indices.transpose_axes(&[0, 1, 3, 2]));
        (Some(v_indices), v_indices_t, Some(val_norms), None)
    };
    let q8_kvbytes_seq = None;
    let q8_slot_scales_seq = if use_q8_seq_shadow {
        let key_scales = key_norms.concatenate_2(&k_residual_norms, 3);
        let value_scales = InlineArray::ones(
            &[values.dim(0), values.dim(1), values.dim(2), 1],
            Dtype::Float32.as_i32(),
        );
        Some(key_scales.concatenate_2(&value_scales, 3))
    } else {
        None
    };

    Some((
        GpuKeyStore {
            indices: k_indices,
            indices_t: k_indices_t,
            q8_keybytes_t,
            q8_keybytes_seq,
            q8_fullbyte_seq,
            q8_kvbytes_seq,
            q8_slot_scales_seq,
            norms: (!use_q8_seq_shadow).then_some(key_norms),
            qjl_signs: k_qjl_signs,
            qjl_signs_t: k_qjl_signs_t,
            residual_norms: (!use_q8_seq_shadow).then_some(k_residual_norms),
        },
        GpuValueStore {
            indices: v_indices,
            indices_t: v_indices_t,
            norms: val_norms,
            d256_rot_values_seq,
        },
    ))
}

/// Encode a single-tensor sub-vector (regular *or* outlier slice) into the
/// rotated-MSE + QJL representation that the Mixed-precision attention
/// kernels will read. Mirrors the keys half of `gpu_quantize_kv` (lines
/// 2571-2630) but parameterised by an arbitrary `TurboQuantCore` so it
/// works for the regular_core (D_reg, dense rotation) and outlier_core
/// (D_out, FWHT rotation when D_out is pow2) without duplication.
fn gpu_encode_key_subvector(
    rows: &InlineArray, // [B, H, S, sub_dim] f32
    core: &TurboQuantCore,
    key_bits: u8,
) -> Option<MixedKeySubvectorEncoding> {
    let mse_bits = key_bits.saturating_sub(1);
    let eps = InlineArray::from_f32(ZERO_EPSILON);

    let norms = rows.norm_l2(-1, true);
    let safe_norms = norms.maximum(&eps);
    let normalized = rows.divide(&safe_norms);

    let rotated = core.rotate_array(&normalized)?;
    let indices_u32 = core.gpu_quantize_mse(&rotated, mse_bits)?;
    // Phase 3a: keep indices as u32 for round-trip clarity. Phase 3c will
    // introduce a u8 packed shadow alongside (matching Uniform's q8_keybytes).
    let indices = indices_u32.clone();

    let recon_rot = core.gpu_reconstruct_mse(&indices_u32, mse_bits)?;
    let residual_rot = rotated.subtract(&recon_rot);
    let residual_norms_raw = residual_rot.norm_l2(-1, true);
    let zero_bound = InlineArray::from_f32(0.0f32);
    let upper_bound = InlineArray::from_f32(MAX_RESIDUAL_NORM);
    let residual_norms = residual_norms_raw
        .maximum(&zero_bound)
        .minimum(&upper_bound);

    let recon_unrot = core.inverse_rotate_array(&recon_rot)?;
    let residual_unrot = normalized.subtract(&recon_unrot);
    let qjl_proj = core.project_array(&residual_unrot)?;
    let qjl_shape = qjl_proj.shape();
    let qjl_ndim = qjl_shape.len();
    let qjl_rows: i32 = qjl_shape[..qjl_ndim - 1].iter().product();
    let packed_dim = packed_qjl_words(core.dim) as i32;
    let qjl_proj_2d = if qjl_ndim == 2 {
        qjl_proj.clone()
    } else {
        qjl_proj.reshape(&[qjl_rows, core.dim as i32])
    };
    let qjl_signs = InlineArray::turboquant_pack_sign_bits(
        &qjl_proj_2d,
        core.dim as u32,
        packed_dim as u32,
        qjl_rows as u32,
    )?;
    let qjl_signs = if qjl_ndim == 2 {
        qjl_signs
    } else {
        let mut packed_shape: Vec<i32> = qjl_shape[..qjl_ndim - 1].to_vec();
        packed_shape.push(packed_dim);
        qjl_signs.reshape(&packed_shape)
    };

    let indices_t = indices.transpose_axes(&[0, 1, 3, 2]);
    let qjl_signs_t = qjl_signs.transpose_axes(&[0, 1, 3, 2]);

    Some(MixedKeySubvectorEncoding {
        indices,
        indices_t: Some(indices_t),
        qjl_signs,
        qjl_signs_t: Some(qjl_signs_t),
        norms,
        residual_norms,
    })
}

struct MixedKeySubvectorEncoding {
    indices: InlineArray,
    indices_t: Option<InlineArray>,
    qjl_signs: InlineArray,
    qjl_signs_t: Option<InlineArray>,
    norms: InlineArray,
    residual_norms: InlineArray,
}

/// Encode a single-tensor sub-vector for the *value* path (norm + rotated
/// MSE indices, no QJL residual term — mirrors the values half of
/// `gpu_quantize_kv` at lines 2701-2723).
fn gpu_encode_value_subvector(
    rows: &InlineArray, // [B, H, S, sub_dim] f32
    core: &TurboQuantCore,
    val_bits: u8,
) -> Option<MixedValueSubvectorEncoding> {
    let eps = InlineArray::from_f32(ZERO_EPSILON);

    let norms = rows.norm_l2(-1, true);
    let safe_norms = norms.maximum(&eps);
    let normalized = rows.divide(&safe_norms);

    let rotated = core.rotate_array(&normalized)?;
    let indices_u32 = core.gpu_quantize_mse(&rotated, val_bits)?;
    let indices = indices_u32.clone();
    let indices_t = indices.transpose_axes(&[0, 1, 3, 2]);

    Some(MixedValueSubvectorEncoding {
        indices,
        indices_t: Some(indices_t),
        norms,
    })
}

struct MixedValueSubvectorEncoding {
    indices: InlineArray,
    indices_t: Option<InlineArray>,
    norms: InlineArray,
}

/// Compute the Mixed-precision outlier partition on the GPU.
///
/// Returns `(regular_src_dim, outlier_src_dim)` of shape `[B, H, S, D_reg]`
/// and `[B, H, S, D_out]` respectively, both `int32`. Each row's entries
/// hold the original-D positions of the corresponding sub-vector slot,
/// **sorted ascending**. The kernel reads these tables to scatter regular
/// and outlier contributions back into the `[B, H, D_total]` output.
///
/// Pipeline:
///   1. `argpartition(-|x|, outlier_count, axis=-1)` → `[B,H,S,D]` with the
///      first `outlier_count` entries being outlier positions (unsorted).
///   2. Slice into `[outlier_idxs_unsorted, regular_idxs_unsorted]`.
///   3. Stable-sort each subset ascending by re-running argsort and
///      `take_along_axis` (positions are unique ints, no tie-break needed).
fn gpu_compute_outlier_partition(
    rows: &InlineArray, // [B, H, S, D_total] f32
    outlier_count: usize,
    total_dim: usize,
) -> Option<(InlineArray, InlineArray)> {
    if outlier_count == 0 || outlier_count >= total_dim {
        return None;
    }
    let abs = rows.abs();
    let neg_abs = abs.negative();
    let part = neg_abs.argpartition(outlier_count as i32, -1);

    let b = part.dim(0);
    let h = part.dim(1);
    let s = part.dim(2);
    let outlier_count_i32 = outlier_count as i32;
    let total_dim_i32 = total_dim as i32;

    let outlier_unsorted = part.slice(&[0, 0, 0, 0], &[b, h, s, outlier_count_i32]);
    let regular_unsorted = part.slice(&[0, 0, 0, outlier_count_i32], &[b, h, s, total_dim_i32]);

    let outlier_perm = outlier_unsorted.argsort(-1);
    let outlier_src_dim = outlier_unsorted.take_along_axis(&outlier_perm, -1);

    let regular_perm = regular_unsorted.argsort(-1);
    let regular_src_dim = regular_unsorted.take_along_axis(&regular_perm, -1);

    Some((regular_src_dim, outlier_src_dim))
}

/// Build a `GpuMixedKeyStore` for one append step from `[B, H, S, D]` f32
/// keys + values. Mirrors `gpu_quantize_kv` for the Mixed branch.
fn gpu_quantize_kv_mixed(
    state: &TurboQuantState,
    keys: &InlineArray,   // [B, H, S, Dk] f32
    values: &InlineArray, // [B, H, S, Dv] f32
    config: TurboQuantConfig,
) -> Option<(GpuMixedKeyStore, GpuMixedValueStore)> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits: kr_bits,
        outlier_bits: ko_bits,
        outlier_count: k_oc,
    } = config.keys
    else {
        return None;
    };
    let TurboQuantTensorConfig::Mixed {
        regular_bits: vr_bits,
        outlier_bits: vo_bits,
        outlier_count: v_oc,
    } = config.values
    else {
        return None;
    };

    let (k_reg_core, k_out_core) = match &state.keys {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core, outlier_core),
        _ => return None,
    };
    let (v_reg_core, v_out_core) = match &state.values {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core, outlier_core),
        _ => return None,
    };

    let k_total_dim = keys.dim(3) as usize;
    let v_total_dim = values.dim(3) as usize;

    // ── Keys ─────────────────────────────────────────────────────────────
    let (k_reg_src, k_out_src) = gpu_compute_outlier_partition(keys, k_oc, k_total_dim)?;
    let k_reg_rows = keys.take_along_axis(&k_reg_src, -1);
    let k_out_rows = keys.take_along_axis(&k_out_src, -1);

    let k_reg_enc = gpu_encode_key_subvector(&k_reg_rows, k_reg_core, kr_bits)?;
    let k_out_enc = gpu_encode_key_subvector(&k_out_rows, k_out_core, ko_bits)?;

    // Cast scatter tables to u8 for storage (D_total ≤ 256 fits comfortably).
    let k_reg_src_u8 = k_reg_src.as_dtype(Dtype::Uint8.as_i32());
    let k_out_src_u8 = k_out_src.as_dtype(Dtype::Uint8.as_i32());

    // ── Values ───────────────────────────────────────────────────────────
    let (v_reg_src, v_out_src) = gpu_compute_outlier_partition(values, v_oc, v_total_dim)?;
    let v_reg_rows = values.take_along_axis(&v_reg_src, -1);
    let v_out_rows = values.take_along_axis(&v_out_src, -1);

    let v_reg_enc = gpu_encode_value_subvector(&v_reg_rows, v_reg_core, vr_bits)?;
    let v_out_enc = gpu_encode_value_subvector(&v_out_rows, v_out_core, vo_bits)?;

    let v_reg_src_u8 = v_reg_src.as_dtype(Dtype::Uint8.as_i32());
    let v_out_src_u8 = v_out_src.as_dtype(Dtype::Uint8.as_i32());

    let mut kstore = GpuMixedKeyStore {
        regular_indices: k_reg_enc.indices,
        regular_indices_t: k_reg_enc.indices_t,
        regular_qjl_signs: k_reg_enc.qjl_signs,
        regular_qjl_signs_t: k_reg_enc.qjl_signs_t,
        regular_norms: k_reg_enc.norms,
        regular_residual_norms: k_reg_enc.residual_norms,
        regular_src_dim: k_reg_src_u8,
        outlier_indices: k_out_enc.indices,
        outlier_indices_t: k_out_enc.indices_t,
        outlier_qjl_signs: k_out_enc.qjl_signs,
        outlier_qjl_signs_t: k_out_enc.qjl_signs_t,
        outlier_norms: k_out_enc.norms,
        outlier_residual_norms: k_out_enc.residual_norms,
        outlier_src_dim: k_out_src_u8,
    };
    let mut vstore = GpuMixedValueStore {
        regular_indices: v_reg_enc.indices,
        regular_indices_t: v_reg_enc.indices_t,
        regular_norms: v_reg_enc.norms,
        regular_src_dim: v_reg_src_u8,
        outlier_indices: v_out_enc.indices,
        outlier_indices_t: v_out_enc.indices_t,
        outlier_norms: v_out_enc.norms,
        outlier_src_dim: v_out_src_u8,
    };

    // The Mixed encode chain branches at two non-deterministic-on-re-eval
    // points: (1) `argpartition` returns an unspecified permutation within
    // each bucket, so the gather (rows = take_along_axis(keys, src_dim)) and
    // the stored `src_dim` table need to commit to the *same* permutation;
    // (2) `gpu_quantize_mse`'s argmin is sensitive to MLX scheduler-driven
    // re-evals on the codebook-distance argmin under f32 noise. Both are
    // correct in isolation but compose into a hidden invariant — the stored
    // `indices` and the residual_norms derived from `recon = take(codebook,
    // indices)` must come from the same materialisation. A single eval+detach
    // here freezes the whole chain at one consistent point. ~1ms per encoded
    // chunk; matches the cost of the existing `eval_and_detach_gpu_state`
    // barrier on the Uniform path.
    let mut to_eval: Vec<&mut InlineArray> = Vec::new();
    kstore.collect_for_detach(&mut to_eval);
    vstore.collect_for_detach(&mut to_eval);
    crate::inline_array::eval_and_detach_many(&mut to_eval);
    Some((kstore, vstore))
}

/// Phase 3b — Score `queries` against a Mixed-precision GPU key store using the
/// dormant `mlx_inline_turboquant_mixed_score` kernel.
///
/// **Layout-oracle contract.** This helper validates the Phase 3a storage
/// layout against the existing C++ score kernel so that Phase 3c's fused
/// attention kernels can reuse the same memory shape with confidence. It is
/// **not** a production scoring path: the kernel takes a single
/// `[N, D_sub]` query slice per sub-vector and reuses it for every cache
/// slot, so the result is only correct when every slot's outlier mask
/// matches `kstore.regular_src_dim[..,0,..]` / `outlier_src_dim[..,0,..]`.
/// Phase 3c's kernels gather Q per-slot from the full `[N, D_total]` query
/// instead.
///
/// Inputs:
///   - `queries`: `[B, q_heads, 1, D_total]` f32 — the un-rotated, un-projected
///     query (matches the shape produced by attention dispatch).
///   - `kstore`: GPU-resident encoded keys.
///   - `n_seq`: how many of the `T` cached slots to score against (≤ T).
///
/// Returns scores `[N, n_seq]` (N = B · q_heads).
#[allow(dead_code)] // Phase 3b oracle — wired by tests; Phase 3c attention path supersedes it.
#[allow(clippy::too_many_arguments)]
fn try_gpu_mixed_score(
    state: &TurboQuantState,
    config: &TurboQuantConfig,
    kstore: &GpuMixedKeyStore,
    queries: &InlineArray,
    q_heads: i32,
    kv_heads: i32,
    n_seq: i32,
    scale: f32,
) -> Option<InlineArray> {
    let (reg_core, out_core) = match &state.keys {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.keys
    else {
        return None;
    };
    if q_heads <= 0 || kv_heads <= 0 || (q_heads % kv_heads) != 0 {
        return None;
    }

    let reg_codebook = reg_core.codebook_arr(regular_bits.saturating_sub(1))?;
    let out_codebook = out_core.codebook_arr(outlier_bits.saturating_sub(1))?;

    let b = queries.dim(0);
    let d_reg = kstore.regular_indices.dim(3);
    let d_out = kstore.outlier_indices.dim(3);
    let t = kstore.regular_indices.dim(2);
    if n_seq <= 0 || n_seq > t {
        return None;
    }
    let n_rows = b * q_heads;
    let kv_rows = b * kv_heads;
    let groups = q_heads / kv_heads;

    // Slice the slot-0 mask (shape [B, kv_heads, 1, D_*]) and broadcast across
    // q_heads via the GQA grouping. `repeat(groups, axis=1)` produces the
    // q_head-sized mask in the same kv_head→q_head order the kernel expects
    // (q_head = kv_head * groups + g).
    let reg_src_slot0 = kstore
        .regular_src_dim
        .slice(&[0, 0, 0, 0], &[b, kv_heads, 1, d_reg])
        .as_dtype(Dtype::Int32.as_i32());
    let out_src_slot0 = kstore
        .outlier_src_dim
        .slice(&[0, 0, 0, 0], &[b, kv_heads, 1, d_out])
        .as_dtype(Dtype::Int32.as_i32());
    let reg_src_q = reg_src_slot0.repeat(groups, 1);
    let out_src_q = out_src_slot0.repeat(groups, 1);

    let q_reg = queries.take_along_axis(&reg_src_q, -1);
    let q_out = queries.take_along_axis(&out_src_q, -1);

    let q_reg_2d = q_reg.reshape(&[n_rows, d_reg]);
    let q_out_2d = q_out.reshape(&[n_rows, d_out]);

    let q_reg_rot = reg_core.rotate_array(&q_reg_2d)?;
    let q_reg_proj = reg_core.project_array(&q_reg_2d)?;
    let q_out_rot = out_core.rotate_array(&q_out_2d)?;
    let q_out_proj = out_core.project_array(&q_out_2d)?;

    let reg_norms_flat = kstore.regular_norms.reshape(&[kv_rows, t]);
    let reg_residual_flat = kstore.regular_residual_norms.reshape(&[kv_rows, t]);
    let out_norms_flat = kstore.outlier_norms.reshape(&[kv_rows, t]);
    let out_residual_flat = kstore.outlier_residual_norms.reshape(&[kv_rows, t]);

    let reg_qjl_words = kstore.regular_qjl_signs.dim(3);
    let out_qjl_words = kstore.outlier_qjl_signs.dim(3);

    InlineArray::turboquant_mixed_score(
        &q_reg_rot,
        &q_reg_proj,
        &kstore.regular_indices,
        &kstore.regular_qjl_signs,
        &reg_norms_flat,
        &reg_residual_flat,
        reg_codebook,
        &q_out_rot,
        &q_out_proj,
        &kstore.outlier_indices,
        &kstore.outlier_qjl_signs,
        &out_norms_flat,
        &out_residual_flat,
        out_codebook,
        d_reg as u32,
        reg_qjl_words as u32,
        reg_codebook.dim(0) as u32,
        d_out as u32,
        out_qjl_words as u32,
        out_codebook.dim(0) as u32,
        n_rows as u32,
        n_seq as u32,
        t as u32,
        q_heads as u32,
        kv_heads as u32,
        scale,
    )
}

/// Dequantise GPU-stored keys back to `[B, H, T, Dk]` f32.
///
/// Formula (per coordinate):
///   k̃ = (codebook[idx] + (√(π/2)/D) · (J^T · sign) · residual_norm) · norm  [inv-rotated]
fn gpu_dequantize_keys(
    store: &GpuKeyStore,
    runtime: &TensorRuntime,
    key_bits: u8,
) -> Option<InlineArray> {
    let key_mse_bits = key_bits.saturating_sub(1);
    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // 1. Reconstruct MSE centroids in the rotated domain: take(codebook, indices) → [B,H,T,D].
    let mse_recon_rot = core.gpu_reconstruct_mse(&store.indices, key_mse_bits)?;

    // 2. Inverse-rotate back to input space.
    //    CPU: inverse_rotate_rows = matmul_rows(inverse_rotation, dim, input) = input @ inverse_rotation.T = input @ rotation.
    //    So GPU: recon_rot @ rotation_arr.
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. QJL correction.
    //    CPU: inverse_project_rows(signs) = matmul_rows(inverse_qjl, dim, signs) = signs @ inverse_qjl.T = signs @ qjl.
    //    The GPU store keeps packed uint32 sign words, so unpack to {-1,+1}
    //    before the matmul with qjl_arr.
    let packed_shape = store.qjl_signs.shape();
    let packed_ndim = packed_shape.len();
    let packed_rows: i32 = packed_shape[..packed_ndim - 1].iter().product();
    let packed_words = packed_shape[packed_ndim - 1];
    let packed_signs = if packed_ndim == 2 {
        store.qjl_signs.clone()
    } else {
        store.qjl_signs.reshape(&[packed_rows, packed_words])
    };
    let unpacked_qjl_2d = InlineArray::turboquant_unpack_sign_bits(
        &packed_signs,
        core.dim as u32,
        packed_words as u32,
        packed_rows as u32,
    )?;
    let unpacked_qjl = if packed_ndim == 2 {
        unpacked_qjl_2d
    } else {
        let mut unpacked_shape: Vec<i32> = packed_shape[..packed_ndim - 1].to_vec();
        unpacked_shape.push(core.dim as i32);
        unpacked_qjl_2d.reshape(&unpacked_shape)
    };
    let qjl_correction = core.inverse_project_array(&unpacked_qjl)?;
    let dim = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim);
    // residual_norms: [B,H,T,1] keepdims — broadcast along D.
    let residual_norms = store.residual_norms_array()?;
    let scale = residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);

    // 4. Base + QJL correction, rescale by original L2 norm.
    // norms: [B,H,T,1] keepdims — broadcast along D.
    let combined = mse_base.add(&correction);
    Some(combined.multiply(&store.key_norms_array()?))
}

/// Dequantise GPU-stored values back to `[B, H, T, Dv]` f32.
fn gpu_dequantize_values(
    store: &GpuValueStore,
    runtime: &TensorRuntime,
    val_bits: u8,
) -> Option<InlineArray> {
    if let Some(d256_rot_values_seq) = store.d256_rot_values_seq.as_ref() {
        let core = match runtime {
            TensorRuntime::Uniform { core, .. } => core,
            TensorRuntime::Mixed { .. } => return None,
        };
        let dense_rot = d256_rot_values_seq.as_dtype(Dtype::Float32.as_i32());
        return core.inverse_rotate_array(&dense_rot);
    }

    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    if dim_uses_fwht(core.dim) {
        let indices_arr = store.indices.as_ref()?;
        let norms_arr = store.norms.as_ref()?;
        let shape = indices_arr.shape();
        let total = shape.iter().product::<i32>() as usize;
        let rows = shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let indices: Vec<u16> = inline_array_to_f32_vec(indices_arr, total)?
            .into_iter()
            .map(|v| v as u16)
            .collect();
        let norms = inline_array_to_f32_vec(norms_arr, rows)?;
        let reconstructed = decode_value_component_rows_raw(core, &indices, &norms, val_bits);
        return Some(InlineArray::from_f32_slice(&reconstructed, shape));
    }

    // 1. Reconstruct MSE centroids in rotated space.
    let mse_recon_rot = core.gpu_reconstruct_mse(store.indices.as_ref()?, val_bits)?;

    // 2. Inverse-rotate: recon_rot @ rotation_arr (same derivation as keys).
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. Rescale by stored L2 norms [B,H,T,1].
    Some(mse_base.multiply(store.norms.as_ref()?))
}

/// Dequantise a single key sub-vector (regular *or* outlier) for the
/// Mixed-precision path. Mirrors `gpu_dequantize_keys` body but
/// parameterised by an arbitrary `TurboQuantCore` and stored arrays.
//
// Phase 3a only exercises this from tests; Phase 3c will wire it into the
// non-test attention dispatch. Suppress the dead-code warning meanwhile.
#[allow(dead_code)]
fn gpu_dequantize_key_subvector(
    indices_u8: &InlineArray,        // [B, H, T, D_sub] u8
    qjl_signs: &InlineArray,         // [B, H, T, ceil(D_sub/32)] u32
    norms: &InlineArray,             // [B, H, T, 1] f32
    residual_norms: &InlineArray,    // [B, H, T, 1] f32
    core: &TurboQuantCore,
    key_bits: u8,
) -> Option<InlineArray> {
    let mse_bits = key_bits.saturating_sub(1);

    // Phase 3a: indices stored as u32; cast no-op when already u32.
    let indices_u32 = indices_u8.as_dtype(Dtype::Uint32.as_i32());
    let mse_recon_rot = core.gpu_reconstruct_mse(&indices_u32, mse_bits)?;
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    let packed_shape = qjl_signs.shape();
    let packed_ndim = packed_shape.len();
    let packed_rows: i32 = packed_shape[..packed_ndim - 1].iter().product();
    let packed_words = packed_shape[packed_ndim - 1];
    let packed_signs = if packed_ndim == 2 {
        qjl_signs.clone()
    } else {
        qjl_signs.reshape(&[packed_rows, packed_words])
    };
    let unpacked_2d = InlineArray::turboquant_unpack_sign_bits(
        &packed_signs,
        core.dim as u32,
        packed_words as u32,
        packed_rows as u32,
    )?;
    let unpacked = if packed_ndim == 2 {
        unpacked_2d
    } else {
        let mut shape: Vec<i32> = packed_shape[..packed_ndim - 1].to_vec();
        shape.push(core.dim as i32);
        unpacked_2d.reshape(&shape)
    };
    let qjl_correction = core.inverse_project_array(&unpacked)?;
    let dim_f = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim_f);
    let scale = residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);
    let combined = mse_base.add(&correction);
    Some(combined.multiply(norms))
}

/// Dequantise a single value sub-vector for the Mixed-precision path.
#[allow(dead_code)]
fn gpu_dequantize_value_subvector(
    indices_u8: &InlineArray,
    norms: &InlineArray,
    core: &TurboQuantCore,
    val_bits: u8,
) -> Option<InlineArray> {
    let indices_u32 = indices_u8.as_dtype(Dtype::Uint32.as_i32());
    let mse_recon_rot = core.gpu_reconstruct_mse(&indices_u32, val_bits)?;
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;
    Some(mse_base.multiply(norms))
}

/// Dequantise a `GpuMixedKeyStore` back to `[B, H, T, D_total]` f32 by
/// dequantising each sub-vector and scattering through the per-row
/// scatter tables.
#[allow(dead_code)]
fn gpu_dequantize_keys_mixed(
    store: &GpuMixedKeyStore,
    runtime: &TensorRuntime,
    config: &TurboQuantConfig,
) -> Option<InlineArray> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.keys
    else {
        return None;
    };
    let (reg_core, out_core) = match runtime {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };

    let regular_recon = gpu_dequantize_key_subvector(
        &store.regular_indices,
        &store.regular_qjl_signs,
        &store.regular_norms,
        &store.regular_residual_norms,
        reg_core,
        regular_bits,
    )?;
    let outlier_recon = gpu_dequantize_key_subvector(
        &store.outlier_indices,
        &store.outlier_qjl_signs,
        &store.outlier_norms,
        &store.outlier_residual_norms,
        out_core,
        outlier_bits,
    )?;

    let b = store.regular_indices.dim(0);
    let h = store.regular_indices.dim(1);
    let t = store.regular_indices.dim(2);
    let d_reg = store.regular_indices.dim(3);
    let d_out = store.outlier_indices.dim(3);
    let d_total = d_reg + d_out;

    let regular_src_i32 = store.regular_src_dim.as_dtype(Dtype::Int32.as_i32());
    let outlier_src_i32 = store.outlier_src_dim.as_dtype(Dtype::Int32.as_i32());

    let zero = InlineArray::zeros(&[b, h, t, d_total], Dtype::Float32.as_i32());
    let with_regular = zero.put_along_axis_op(&regular_src_i32, &regular_recon, -1);
    let merged = with_regular.put_along_axis_op(&outlier_src_i32, &outlier_recon, -1);
    Some(merged)
}

/// Dequantise a `GpuMixedValueStore` back to `[B, H, T, D_total]` f32.
#[allow(dead_code)]
fn gpu_dequantize_values_mixed(
    store: &GpuMixedValueStore,
    runtime: &TensorRuntime,
    config: &TurboQuantConfig,
) -> Option<InlineArray> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.values
    else {
        return None;
    };
    let (reg_core, out_core) = match runtime {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };

    let regular_recon =
        gpu_dequantize_value_subvector(&store.regular_indices, &store.regular_norms, reg_core, regular_bits)?;
    let outlier_recon =
        gpu_dequantize_value_subvector(&store.outlier_indices, &store.outlier_norms, out_core, outlier_bits)?;

    let b = store.regular_indices.dim(0);
    let h = store.regular_indices.dim(1);
    let t = store.regular_indices.dim(2);
    let d_reg = store.regular_indices.dim(3);
    let d_out = store.outlier_indices.dim(3);
    let d_total = d_reg + d_out;

    let regular_src_i32 = store.regular_src_dim.as_dtype(Dtype::Int32.as_i32());
    let outlier_src_i32 = store.outlier_src_dim.as_dtype(Dtype::Int32.as_i32());

    let zero = InlineArray::zeros(&[b, h, t, d_total], Dtype::Float32.as_i32());
    let with_regular = zero.put_along_axis_op(&regular_src_i32, &regular_recon, -1);
    let merged = with_regular.put_along_axis_op(&outlier_src_i32, &outlier_recon, -1);
    Some(merged)
}

// ═══════════════════════════════════════════════════════════════════════════
// Encoding (quantise) helpers
// ═══════════════════════════════════════════════════════════════════════════

struct EncodedKeyRows {
    mse_indices: Vec<u16>,
    qjl_signs: Vec<u16>,
    norms: Vec<f32>,
    residual_norms: Vec<f32>,
}

struct EncodedValueRows {
    indices: Vec<u16>,
    norms: Vec<f32>,
}

struct BatchedKeyRows {
    regular: EncodedKeyRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedKeyRows>,
}

struct BatchedValueRows {
    regular: EncodedValueRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedValueRows>,
}

fn encode_key_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedKeyRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedKeyRows {
                regular: encode_key_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedKeyRows {
                regular: encode_key_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_key_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

fn encode_value_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedValueRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedValueRows {
                regular: encode_value_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedValueRows {
                regular: encode_value_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_value_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

/// Two-stage key encoder: MSE at (bits-1) + QJL on residual.
#[allow(clippy::needless_range_loop)]
fn encode_key_component_rows(core: &TurboQuantCore, rows: &[f32], key_bits: u8) -> EncodedKeyRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    // Step 1: Normalise onto unit sphere. Non-finite inputs (NaN/Inf) or
    // degenerate zero rows are zeroed out — the MSE quantizer's binary
    // search does not tolerate NaN.
    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        if !norm.is_finite() || norm <= ZERO_EPSILON {
            norms[row_idx] = 0.0;
            // `normalized` already initialised to zero.
            continue;
        }
        norms[row_idx] = norm;
        let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
        for (dst, &src) in dst.iter_mut().zip(row.iter()) {
            let n = src / norm;
            *dst = if n.is_finite() { n } else { 0.0 };
        }
    }

    // Step 2: MSE quantise at (bits-1).
    let mse_bits = key_bits.saturating_sub(1);
    let mut mse_indices = quantize_mse_rows(core, &normalized, mse_bits);

    // Step 3: Reconstruct MSE approximation.
    let decoded_mse = if mse_bits == 0 {
        vec![0.0; rows.len()]
    } else {
        reconstruct_mse_rows(core, &mse_indices, mse_bits)
    };

    // Step 4: Compute residual = normalized - decoded_mse.
    let mut residual = vec![0.0f32; rows.len()];
    let mut residual_norms = vec![0.0f32; num_rows];
    for row_idx in 0..num_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        if norms[row_idx] <= ZERO_EPSILON {
            mse_indices[start..end].fill(0);
            continue;
        }
        let res_row = &mut residual[start..end];
        for ((dst, &lhs), &rhs) in res_row
            .iter_mut()
            .zip(normalized[start..end].iter())
            .zip(decoded_mse[start..end].iter())
        {
            *dst = lhs - rhs;
        }
        let raw_norm = l2_norm(res_row);
        residual_norms[row_idx] = if raw_norm.is_finite() {
            raw_norm.clamp(0.0, MAX_RESIDUAL_NORM)
        } else {
            0.0
        };
    }

    // Step 5: QJL — project residual and take signs.
    let projected = core.project_rows(&residual);
    let mut qjl_signs: Vec<u16> = projected
        .iter()
        .map(|&v| if v >= 0.0 { 1 } else { 0 })
        .collect();

    // Zero-vector rows get all-zero signs.
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            qjl_signs[start..end].fill(0);
        }
    }

    EncodedKeyRows {
        mse_indices,
        qjl_signs,
        norms,
        residual_norms,
    }
}

/// MSE-only value encoder.
#[allow(clippy::needless_range_loop)]
fn encode_value_component_rows(
    core: &TurboQuantCore,
    rows: &[f32],
    value_bits: u8,
) -> EncodedValueRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        if !norm.is_finite() || norm <= ZERO_EPSILON {
            norms[row_idx] = 0.0;
            continue;
        }
        norms[row_idx] = norm;
        let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
        for (dst, &src) in dst.iter_mut().zip(row.iter()) {
            let n = src / norm;
            *dst = if n.is_finite() { n } else { 0.0 };
        }
    }

    let mut indices = quantize_mse_rows(core, &normalized, value_bits);
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            indices[start..end].fill(0);
        }
    }

    EncodedValueRows { indices, norms }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decoding (dequantise) helpers
// ═══════════════════════════════════════════════════════════════════════════

fn decode_key_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedKeyStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_key_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_key_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                *regular_bits,
            );
            let outlier = decode_key_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant key outlier indices missing"),
                ),
                &unpack_all(
                    store
                        .outlier_qjl_signs
                        .as_ref()
                        .expect("TurboQuant key outlier QJL signs missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant key outlier norms missing"),
                store
                    .outlier_residual_norms
                    .as_ref()
                    .expect("TurboQuant key outlier residual_norms missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant key outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

fn decode_value_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedValueStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_value_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_value_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *regular_bits,
            );
            let outlier = decode_value_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant value outlier indices missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant value outlier norms missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant value outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

/// Reconstruct key rows from MSE indices + QJL signs + norms.
///
/// Formula (per row):
///   k̃ = Π^T · codebook[idx] · norm + (√(π/2)/D) · Π^T · J^T · sign · residual_norm · norm
#[allow(clippy::needless_range_loop)]
fn decode_key_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    qjl_signs: &[u16],
    norms: &[f32],
    residual_norms: &[f32],
    key_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mse_bits = key_bits.saturating_sub(1);

    // MSE base reconstruction (rotate back from codebook centroids).
    let mut reconstructed = if mse_bits == 0 {
        vec![0.0; total_rows * core.dim]
    } else {
        reconstruct_mse_rows(core, indices, mse_bits)
    };

    // QJL correction term — only if any residual is non-trivial.
    if residual_norms.iter().any(|&rn| rn > ZERO_EPSILON) {
        let qjl_signs_f32: Vec<f32> = qjl_signs
            .iter()
            .map(|&v| if v == 0 { -1.0 } else { 1.0 })
            .collect();
        let qjl_correction = core.inverse_project_rows(&qjl_signs_f32);

        for row_idx in 0..total_rows {
            let residual_norm = residual_norms[row_idx];
            if residual_norm <= ZERO_EPSILON {
                continue;
            }
            let scale = ((PI / 2.0).sqrt() * residual_norm) / (core.dim as f32);
            let start = row_idx * core.dim;
            let end = start + core.dim;
            for (val, &correction) in reconstructed[start..end]
                .iter_mut()
                .zip(qjl_correction[start..end].iter())
            {
                *val += scale * correction;
            }
        }
    }

    // Rescale by stored norm.
    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}

/// Reconstruct value rows from MSE indices + norms.
#[allow(clippy::needless_range_loop)]
fn decode_value_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    norms: &[f32],
    value_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mut reconstructed = reconstruct_mse_rows(core, indices, value_bits);

    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}

// ═══════════════════════════════════════════════════════════════════════════
// Core quantisation primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Rotate then nearest-centroid quantise: returns a per-coordinate index.
fn quantize_mse_rows(core: &TurboQuantCore, normalized: &[f32], bits: u8) -> Vec<u16> {
    if bits == 0 {
        return vec![0; normalized.len()];
    }
    let rotated = core.rotate_rows(normalized);
    let codebook = core.codebook(bits);
    rotated
        .iter()
        .map(|&v| nearest_centroid_index(v, codebook) as u16)
        .collect()
}

/// Look up centroids then inverse-rotate to reconstruct approximate vectors.
fn reconstruct_mse_rows(core: &TurboQuantCore, indices: &[u16], bits: u8) -> Vec<f32> {
    if bits == 0 {
        return vec![0.0; indices.len()];
    }
    let codebook = core.codebook(bits);
    let rotated: Vec<f32> = indices.iter().map(|&i| codebook[usize::from(i)]).collect();
    core.inverse_rotate_rows(&rotated)
}

/// Binary search for the nearest centroid (codebook is sorted ascending).
fn nearest_centroid_index(value: f32, codebook: &[f32]) -> usize {
    match codebook.binary_search_by(|probe| probe.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= codebook.len() => codebook.len() - 1,
        Err(i) => {
            let left = codebook[i - 1];
            let right = codebook[i];
            if (value - left).abs() <= (right - value).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Outlier selection / scattering
// ═══════════════════════════════════════════════════════════════════════════

/// Identify the top-k highest-magnitude coordinates as outliers.
fn select_outlier_mask(row: &[f32], outlier_count: usize) -> Vec<u16> {
    let mut ranked: Vec<usize> = (0..row.len()).collect();
    ranked.sort_unstable_by(|&lhs, &rhs| {
        row[rhs]
            .abs()
            .total_cmp(&row[lhs].abs())
            .then_with(|| lhs.cmp(&rhs))
    });
    let mut mask = vec![0u16; row.len()];
    for dim_idx in ranked.into_iter().take(outlier_count) {
        mask[dim_idx] = 1;
    }
    mask
}

/// Partition rows into regular and outlier sub-vectors.
///
/// Returns `(outlier_mask, regular_rows, outlier_rows)`.
fn split_rows_by_outliers(
    rows: &[f32],
    total_dim: usize,
    outlier_count: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let num_rows = rows.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut mask_all = Vec::with_capacity(rows.len());
    let mut regular_rows = Vec::with_capacity(num_rows * regular_dim);
    let mut outlier_rows = Vec::with_capacity(num_rows * outlier_count);

    for row in rows.chunks(total_dim) {
        let mask = select_outlier_mask(row, outlier_count);
        for (&v, &is_outlier) in row.iter().zip(mask.iter()) {
            if is_outlier == 1 {
                outlier_rows.push(v);
            } else {
                regular_rows.push(v);
            }
        }
        mask_all.extend_from_slice(&mask);
    }

    (mask_all, regular_rows, outlier_rows)
}

/// Re-interleave regular and outlier sub-vectors using the stored mask.
fn scatter_mixed_rows(
    outlier_mask: &[u16],
    total_dim: usize,
    outlier_count: usize,
    regular_rows: &[f32],
    outlier_rows: &[f32],
) -> Vec<f32> {
    let num_rows = outlier_mask.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut merged = vec![0.0f32; outlier_mask.len()];

    for row_idx in 0..num_rows {
        let mask_row = &outlier_mask[row_idx * total_dim..(row_idx + 1) * total_dim];
        let mut reg_cur = 0usize;
        let mut out_cur = 0usize;
        for dim_idx in 0..total_dim {
            let dst = &mut merged[row_idx * total_dim + dim_idx];
            if mask_row[dim_idx] == 1 {
                *dst = outlier_rows[row_idx * outlier_count + out_cur];
                out_cur += 1;
            } else {
                *dst = regular_rows[row_idx * regular_dim + reg_cur];
                reg_cur += 1;
            }
        }
    }

    merged
}

mod math;
pub use math::{beta_codebook, generate_rademacher_signs, signed_fwht_forward};
pub(crate) use math::{generate_random_orthogonal, generate_random_projection};
#[cfg(test)]
use math::build_beta_codebook;
use math::{
    f32_rows_to_bhsd_array, inline_array_to_bshd_rows, inline_array_to_f32_vec, l2_norm,
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod layout_invariants {
    //! Pin the layout-derived dimensions so a silent off-by-one in a helper
    //! cannot misalign the C++ score kernels' reads. Mirrors the contract in
    //! `cpp/bridge/turboquant.h`'s top-of-file invariants block.
    use super::packed_qjl_words;

    #[test]
    fn qjl_words_match_pack_invariants() {
        // Exact 32-multiples — kernel expects no padding word.
        assert_eq!(packed_qjl_words(32), 1);
        assert_eq!(packed_qjl_words(64), 2);
        assert_eq!(packed_qjl_words(128), 4);
        assert_eq!(packed_qjl_words(256), 8);
        // Non-multiples — caller must round up because each sign-bit lands
        // in some 32-bit lane and the kernel reads `qjl_words` lanes.
        assert_eq!(packed_qjl_words(33), 2);
        assert_eq!(packed_qjl_words(96), 3);
        assert_eq!(packed_qjl_words(192), 6);
    }

    #[test]
    fn d256_qjl_words_is_eight() {
        // The fused d256 attention kernel hardcodes `qjl_words == 8`. If we
        // ever change the QJL packing this test fails fast with a clear
        // pointer to the C++ kernel that needs an update.
        assert_eq!(packed_qjl_words(256), 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_direct_attention_case_with(
        dim: usize,
        heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        // These helpers exercise the cold compressed GPU store directly, so
        // disable the recent-fp16 window — otherwise short prefills sit in
        // the hot ring and never touch the cold path.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (seed_cache, queries, step_keys, step_values, scale, b, h, d)
    }

    fn make_uniform_direct_attention_case() -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        make_uniform_direct_attention_case_with(16, 2, 3)
    }

    fn make_uniform_gqa_direct_attention_case_with(
        dim: usize,
        q_heads: i32,
        kv_heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
        i32,
    ) {
        // These helpers exercise the cold compressed GPU store directly, so
        // disable the recent-fp16 window — otherwise short prefills sit in
        // the hot ring and never touch the cold path.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * kv_heads * prefill * d) as usize;
        let step_kv_len = (b * kv_heads * d) as usize;
        let query_len = (b * q_heads * d) as usize;
        let prefill_keys = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.2),
            &[b, kv_heads, prefill, d],
        );
        let prefill_values = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.7),
            &[b, kv_heads, prefill, d],
        );
        let queries = InlineArray::from_f32_slice(&make_data(query_len, 1.3), &[b, q_heads, 1, d]);
        let step_keys =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 1.9), &[b, kv_heads, 1, d]);
        let step_values =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 2.4), &[b, kv_heads, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (
            seed_cache,
            queries,
            step_keys,
            step_values,
            scale,
            b,
            q_heads,
            kv_heads,
            d,
        )
    }

    /// Build a Mixed-precision (`preset_q3_5`) direct-attention fixture. Matches
    /// the Uniform helper's shape contract so the same `manual_single_token_attention`
    /// reference can compare against both. Mixed paths currently dequantize+SDPA
    /// inside `append_and_compute_attention`; once the fused Mixed kernel lands
    /// (Phase 3), this fixture pins the expected numerics so the kernel must
    /// match the dequantize+SDPA baseline at < 1e-4.
    fn make_mixed_direct_attention_case_with(
        dim: usize,
        heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        // Cold-only path: disable the recent fp16 window so every appended token
        // takes the Mixed-quantized cold store and exercises the full
        // dequantize_keys/dequantize_values + SDPA fallback (the path the fused
        // Mixed kernel will replace).
        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        (seed_cache, queries, step_keys, step_values, scale, b, h, d)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn manual_single_token_attention(
        queries: &mut InlineArray,
        keys: &mut InlineArray,
        values: &mut InlineArray,
        batch: i32,
        heads: i32,
        seq: i32,
        dim: i32,
        scale: f32,
    ) -> Vec<f32> {
        let q = queries
            .to_f32_vec((batch * heads * dim) as usize)
            .expect("queries to_f32");
        let k = keys
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("keys to_f32");
        let v = values
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("values to_f32");

        let rows = (batch * heads) as usize;
        let seq_usize = seq as usize;
        let dim_usize = dim as usize;
        let mut out = vec![0.0f32; rows * dim_usize];

        for row in 0..rows {
            let q_base = row * dim_usize;
            let q_row = &q[q_base..q_base + dim_usize];

            let mut scores = vec![0.0f32; seq_usize];
            for t in 0..seq_usize {
                let k_base = (row * seq_usize + t) * dim_usize;
                let k_row = &k[k_base..k_base + dim_usize];
                let dot = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>();
                scores[t] = dot * scale;
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
            for score in &mut scores {
                *score /= sum_exp.max(f32::MIN_POSITIVE);
            }

            let out_row = &mut out[q_base..q_base + dim_usize];
            for t in 0..seq_usize {
                let v_base = (row * seq_usize + t) * dim_usize;
                let v_row = &v[v_base..v_base + dim_usize];
                let weight = scores[t];
                for (dst, val) in out_row.iter_mut().zip(v_row.iter()) {
                    *dst += weight * *val;
                }
            }
        }

        out
    }

    #[test]
    fn packed_bits_round_trip() {
        let values = [1u16, 6, 3, 0, 7, 2, 4];
        let mut packed = PackedBits::new(3);
        packed.extend_from_slice(&values);
        let round_trip: Vec<u16> = (0..values.len()).map(|i| packed.get(i)).collect();
        assert_eq!(round_trip, values);

        packed.truncate(4);
        let truncated: Vec<u16> = (0..4).map(|i| packed.get(i)).collect();
        assert_eq!(truncated, values[..4]);
    }

    #[test]
    fn beta_codebook_is_sorted_and_correct_length() {
        let codebook = build_beta_codebook(128, 4);
        assert_eq!(codebook.len(), 16);
        assert!(codebook.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn beta_codebook_memo_matches_direct() {
        // Memoized public entry must match the underlying computation byte-for-byte.
        let direct = build_beta_codebook(128, 3);
        let memoed = beta_codebook(128, 3);
        assert_eq!(direct.len(), memoed.len());
        for (a, b) in direct.iter().zip(memoed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        // Second call hits the cache and returns identical contents.
        let memoed_again = beta_codebook(128, 3);
        assert_eq!(*memoed, *memoed_again);
    }

    #[test]
    fn codebook_range_within_unit_interval() {
        let codebook = build_beta_codebook(128, 4);
        assert!(codebook.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn nearest_centroid_boundary_cases() {
        let cb = vec![-0.5f32, 0.0, 0.5];
        assert_eq!(nearest_centroid_index(-2.0, &cb), 0);
        assert_eq!(nearest_centroid_index(2.0, &cb), 2);
        assert_eq!(nearest_centroid_index(0.0, &cb), 1);
        assert_eq!(nearest_centroid_index(0.26, &cb), 2);
    }

    #[test]
    fn turboquant_handles_zero_rows() {
        let core = TurboQuantCore::new(8, 4);
        let encoded = encode_key_component_rows(&core, &[0.0; 8], 4);
        assert_eq!(encoded.norms, vec![0.0]);
        assert_eq!(encoded.residual_norms, vec![0.0]);
        assert!(encoded.mse_indices.iter().all(|&v| v == 0));
        assert!(encoded.qjl_signs.iter().all(|&v| v == 0));
    }

    #[test]
    fn turboquant_state_constructs_without_panic() {
        let config = TurboQuantConfig::uniform(4, 4);
        let _state = TurboQuantState::new(64, 64, config);
    }

    #[test]
    fn mixed_config_effective_bits() {
        let config = TurboQuantTensorConfig::mixed(2, 4, 32);
        assert_eq!(config.effective_bits(128), 2.5);
        assert_eq!(config.regular_dim(128), 96);
        assert_eq!(config.outlier_count(), 32);
    }

    #[test]
    fn select_outlier_mask_marks_top_k() {
        let row = [0.1f32, 0.9, 0.5, 0.8, 0.2];
        let mask = select_outlier_mask(&row, 2);
        // Top 2 by magnitude: index 1 (0.9) and index 3 (0.8)
        assert_eq!(mask[1], 1);
        assert_eq!(mask[3], 1);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[2], 0);
        assert_eq!(mask[4], 0);
    }

    #[test]
    fn scatter_round_trips_split() {
        let rows = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mask, regular, outlier) = split_rows_by_outliers(&rows, 3, 1);
        let merged = scatter_mixed_rows(&mask, 3, 1, &regular, &outlier);
        assert_eq!(merged.len(), rows.len());
        // Merged must contain the same values (possibly reordered by scatter).
        let mut orig_sorted = rows.clone();
        let mut merged_sorted = merged.clone();
        orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        merged_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(orig_sorted, merged_sorted);
    }

    #[test]
    fn encode_value_norm_preserved() {
        let core = TurboQuantCore::new(8, 4);
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let encoded = encode_value_component_rows(&core, &v, 4);
        assert_eq!(encoded.norms.len(), 1);
        let expected_norm = l2_norm(&v);
        assert!((encoded.norms[0] - expected_norm).abs() < 1e-5);
    }

    #[test]
    fn turboquant_presets_match_schedule() {
        let q2_5 = TurboQuantConfig::preset_q2_5(128);
        let q3_5 = TurboQuantConfig::preset_q3_5(128);
        assert_eq!(q2_5, TurboQuantConfig::mixed(2, 4, 32, 2, 4, 32));
        assert_eq!(q3_5, TurboQuantConfig::mixed(3, 5, 32, 3, 5, 32));
    }

    /// GPU round-trip: append via GPU path then dequantize via GPU path.
    ///
    /// We verify two things:
    ///   1. The GPU path is actually taken (store.gpu is Some).
    ///   2. The GPU dequantised output is close to the CPU dequantised output
    ///      (same algorithm, both paths should produce bitwise-close results
    ///      modulo f32 ordering differences).
    #[test]
    fn turboquant_gpu_path_round_trip() {
        // Small dim so the test is fast.
        let dim = 16usize;
        // Disable the hot window — this test asserts that data lands in the
        // cold GPU store, which only happens once the recent fp16 ring is off
        // (or has overflowed past `recent_window + HOT_EVICTION_CHUNK`).
        let config = TurboQuantConfig::uniform(4, 4).with_recent_window(None);
        let b = 1i32;
        let h = 2i32;
        let s = 3i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;

        // Build deterministic input vectors.
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.1 - total as f32 * 0.05).sin())
            .collect();
        // Upload as [B, H, S, D] f32.
        let keys_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        // ── CPU reference path (use Mixed config to force CPU) ────────────
        // Disable the hot window so the test exercises the cold-only legacy
        // path it was originally written against.
        let cpu_config = TurboQuantConfig {
            keys: TurboQuantTensorConfig::Mixed {
                regular_bits: 3,
                outlier_bits: 4,
                outlier_count: 4,
            },
            values: TurboQuantTensorConfig::Mixed {
                regular_bits: 4,
                outlier_bits: 4,
                outlier_count: 4,
            },
            recent_window: None,
        };
        let mut cpu_cache = QuantizedKvCache::new(cpu_config);
        cpu_cache.append(&keys_arr, &vals_arr).expect("CPU append");
        // Verify CPU path taken (no GPU store).
        assert!(
            cpu_cache.keys.as_ref().unwrap().gpu.is_none(),
            "Expected CPU path for Mixed config"
        );

        // ── GPU path ──────────────────────────────────────────────────────
        let mut gpu_cache = QuantizedKvCache::new(config);
        gpu_cache.append(&keys_arr, &vals_arr).expect("GPU append");

        // Verify GPU path was taken.
        assert!(
            gpu_cache.keys.as_ref().unwrap().gpu.is_some(),
            "GPU store should be Some for Uniform config"
        );
        assert!(
            gpu_cache.values.as_ref().unwrap().gpu.is_some(),
            "GPU value store should be Some for Uniform config"
        );

        // Dequantise — should succeed.
        let dk = gpu_cache.dequantize_keys().expect("GPU dequantize_keys");
        let dv = gpu_cache
            .dequantize_values()
            .expect("GPU dequantize_values");

        // Verify output shapes: [B, H, T, D].
        assert_eq!(dk.shape(), &[b, h, s, d], "dequantized keys shape mismatch");
        assert_eq!(
            dv.shape(),
            &[b, h, s, d],
            "dequantized values shape mismatch"
        );

        // Output should be finite (not NaN/Inf).
        let dk_vals = dk
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dk to_f32");
        let dv_vals = dv
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dv to_f32");
        assert!(
            dk_vals.iter().all(|v| v.is_finite()),
            "dequantized keys contain non-finite"
        );
        assert!(
            dv_vals.iter().all(|v| v.is_finite()),
            "dequantized values contain non-finite"
        );

        // Verify output is within reasonable range (quantisation introduces error but
        // should not explode — reconstructed vectors should be roughly same magnitude as input).
        let input_max = data.iter().cloned().fold(0.0f32, f32::max).abs();
        let dk_max = dk_vals.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(
            dk_max < input_max * 3.0,
            "dequantized keys magnitude unreasonably large"
        );
    }

    #[test]
    fn gpu_packed_qjl_sign_words_round_trip_encodes_zero_as_positive_bit() {
        let projected = InlineArray::from_f32_slice(&[-2.0f32, 0.0, 3.0, -0.5], &[1, 4]);
        let packed =
            InlineArray::turboquant_pack_sign_bits(&projected, 4, 1, 1).expect("pack sign bits");
        let mut unpacked =
            InlineArray::turboquant_unpack_sign_bits(&packed, 4, 1, 1).expect("unpack sign bits");
        let values = unpacked.to_f32_vec(4).expect("unpacked to_f32");
        assert_eq!(values, vec![-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn turboquant_q8_d256_gpu_store_uses_seq_shadow_without_transposed_shadow() {
        let (seed_cache, _, _, _, _, _, _, _, _) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let gpu = seed_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("gpu key store");
        assert!(
            gpu.q8_keybytes_t.is_none(),
            "d256 q8 path should not keep transposed q8 shadow"
        );
        assert!(
            gpu.q8_keybytes_seq.is_some(),
            "d256 q8 path should keep seq-major packed key shadow"
        );
        assert!(
            gpu.q8_kvbytes_seq.is_none(),
            "d256 q8 path should not keep packed kv shadow when dense rotated values are present"
        );
        assert!(
            gpu.q8_slot_scales_seq.is_some(),
            "d256 q8 path should keep seq-major slot scale shadow"
        );
        assert!(
            gpu.indices_t.is_none(),
            "d256 q8 path should not keep transposed key indices"
        );
        assert!(
            gpu.qjl_signs_t.is_none(),
            "d256 q8 path should not keep transposed qjl sign words"
        );
        assert!(
            gpu.norms.is_none(),
            "d256 q8 path should source key norms from slot scales"
        );
        assert!(
            gpu.residual_norms.is_none(),
            "d256 q8 path should source residual norms from slot scales"
        );
        let gpu_values = seed_cache
            .values
            .as_ref()
            .and_then(|v| v.gpu.as_ref())
            .expect("gpu value store");
        assert!(
            gpu_values.indices_t.is_none(),
            "d256 q8 path should not keep transposed value indices"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d128_long_context() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case_with(128, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "long-context direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d256_long_context_gqa() {
        let (seed_cache, queries, step_keys, step_values, scale, b, q_heads, kv_heads, d) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let full_values = ref_cache.dequantize_values().expect("dequantize values");
        let repeated_keys = full_keys.repeat(q_heads / kv_heads, 1);
        let repeated_values = full_values.repeat(q_heads / kv_heads, 1);
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut repeated_keys.clone(),
            &mut repeated_values.clone(),
            b,
            q_heads,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * q_heads * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 gqa direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    /// Phase 3 parity baseline. Mixed-precision (`preset_q3_5`) currently
    /// flows through `dequantize_keys + dequantize_values + sdpa_causal_like_mlx`
    /// — a "correctness-first" fallback. This test pins the expected output
    /// against a from-scratch scalar reference so that when the fused Mixed
    /// kernel lands it must produce the same numerics.
    ///
    /// Tolerance matches the Uniform parity tests (1e-4): both paths
    /// dequantize the same Mixed-quantized data, so the only delta here is
    /// MLX's fused SDPA vs scalar SDPA on identical f32 tensors.
    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_mixed() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_mixed_direct_attention_case_with(128, 2, 31);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            32,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "mixed direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    /// Drives the d256 fullbyte+dense-values direct attention path against a
    /// from-scratch CPU reference at the given sequence length. Pulled out of
    /// the original single-shape test so the seq-sweep variant below can hit
    /// 1024/2048/4096 without copy-pasting 80 LOC of fixture setup.
    fn run_d256_fullbyte_dense_values_parity_at(seq: i32) {
        let batch = 1i32;
        let q_heads = 4i32;
        let kv_heads = 2i32;
        let groups = q_heads / kv_heads;
        let dim = 256i32;
        let kv_rows = batch * kv_heads;
        let q_rows = batch * q_heads;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let query_rot_vec: Vec<f32> = (0..(q_rows * dim) as usize)
            .map(|i| ((i as f32) * 0.013 + 0.4).sin() * 0.7)
            .collect();
        let key_indices_vec: Vec<u8> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| (((i * 17) + 23) & 0xff) as u8)
            .collect();
        let slot_scales_vec: Vec<f32> = (0..(kv_rows * seq * 3) as usize)
            .map(|i| match i % 3 {
                0 => 0.5 + (((i / 3) % 11) as f32) * 0.03125,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let key_codebook_vec: Vec<f32> = (0..256).map(|i| ((i as f32) - 127.5) / 96.0).collect();
        let value_dense_vec: Vec<f32> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| ((i as f32) * 0.009 - 0.7).cos() * 0.5)
            .collect();

        let query_rot = InlineArray::from_f32_slice(&query_rot_vec, &[q_rows, dim]);
        let key_indices = InlineArray::from_u8_slice(&key_indices_vec, &[kv_rows, seq, dim]);
        let slot_scales = InlineArray::from_f32_slice(&slot_scales_vec, &[kv_rows, seq, 3]);
        let key_codebook = InlineArray::from_f32_slice(&key_codebook_vec, &[256]);
        let value_dense = InlineArray::from_f32_slice(&value_dense_vec, &[kv_rows, seq, dim]);

        let mut direct = InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
            &query_rot,
            &key_indices,
            &slot_scales,
            &key_codebook,
            &value_dense,
            q_rows as u32,
            seq as u32,
            seq as u32,
            q_heads as u32,
            kv_heads as u32,
            scale,
        )
        .expect("fullbyte direct attention");

        let mut keys = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        let mut values = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        for qh in 0..q_heads as usize {
            let kvh = qh / groups as usize;
            for t in 0..seq as usize {
                let scale_base = (kvh * seq as usize + t) * 3;
                let key_norm = slot_scales_vec[scale_base];
                let key_base = (kvh * seq as usize + t) * dim as usize;
                let out_base = (qh * seq as usize + t) * dim as usize;
                for d_idx in 0..dim as usize {
                    let idx = key_indices_vec[key_base + d_idx] as usize;
                    keys[out_base + d_idx] = key_codebook_vec[idx] * key_norm;
                    values[out_base + d_idx] = value_dense_vec[key_base + d_idx];
                }
            }
        }

        let mut queries = query_rot.reshape(&[batch, q_heads, 1, dim]);
        let mut keys = InlineArray::from_f32_slice(&keys, &[batch, q_heads, seq, dim]);
        let mut values = InlineArray::from_f32_slice(&values, &[batch, q_heads, seq, dim]);
        let reference_vals = manual_single_token_attention(
            &mut queries,
            &mut keys,
            &mut values,
            batch,
            q_heads,
            seq,
            dim,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((batch * q_heads * dim) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 fullbyte direct attention diverged from manual reference at seq={seq}: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_attention_q8_d256_fullbyte_dense_values_matches_manual_reference() {
        run_d256_fullbyte_dense_values_parity_at(1024);
    }

    /// Denser parity sweep over n_seq for the d256 fullbyte+dense-values path.
    ///
    /// The single-shape test above pins correctness at the n_seq=1024 dispatch
    /// boundary. This sweep extends coverage to 2048 and 4096 so any future
    /// touch-ups to the shared d256 pass-2 merge kernel (e.g. when the
    /// mixed-precision fused attention kernel from Phase 3 reuses it) catch
    /// regressions across the long-context envelope, not just at the threshold.
    #[test]
    fn turboquant_attention_q8_d256_fullbyte_dense_values_parity_seq_sweep() {
        for seq in [1024, 2048, 4096] {
            run_d256_fullbyte_dense_values_parity_at(seq);
        }
    }

    #[test]
    fn turboquant_direct_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache;
        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");
        let vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    /// CPU reference for `[B, H, S, D]` causal multi-token attention. Returns
    /// the output flattened in `[B, H, S, D]` order. Only used by the prefill
    /// dispatch tests below — production paths go through MLX SDPA directly.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn manual_causal_multi_token_attention(
        queries: &mut InlineArray,
        keys: &mut InlineArray,
        values: &mut InlineArray,
        batch: i32,
        heads: i32,
        seq_q: i32,
        seq_kv: i32,
        dim: i32,
        scale: f32,
    ) -> Vec<f32> {
        let q = queries
            .to_f32_vec((batch * heads * seq_q * dim) as usize)
            .expect("queries to_f32");
        let k = keys
            .to_f32_vec((batch * heads * seq_kv * dim) as usize)
            .expect("keys to_f32");
        let v = values
            .to_f32_vec((batch * heads * seq_kv * dim) as usize)
            .expect("values to_f32");

        let rows = (batch * heads) as usize;
        let sq = seq_q as usize;
        let sk = seq_kv as usize;
        let dim_us = dim as usize;
        let causal_offset = sk - sq;
        let mut out = vec![0.0f32; rows * sq * dim_us];

        for row in 0..rows {
            for qi in 0..sq {
                let q_pos = causal_offset + qi;
                let q_base = (row * sq + qi) * dim_us;
                let q_row = &q[q_base..q_base + dim_us];

                let mut scores = vec![f32::NEG_INFINITY; sk];
                for t in 0..=q_pos {
                    let k_base = (row * sk + t) * dim_us;
                    let k_row = &k[k_base..k_base + dim_us];
                    let dot = q_row
                        .iter()
                        .zip(k_row.iter())
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum::<f32>();
                    scores[t] = dot * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - max_score).exp();
                        sum_exp += *score;
                    } else {
                        *score = 0.0;
                    }
                }
                for score in &mut scores {
                    *score /= sum_exp.max(f32::MIN_POSITIVE);
                }

                let out_row = &mut out[q_base..q_base + dim_us];
                for t in 0..=q_pos {
                    let v_base = (row * sk + t) * dim_us;
                    let v_row = &v[v_base..v_base + dim_us];
                    let weight = scores[t];
                    for (dst, val) in out_row.iter_mut().zip(v_row.iter()) {
                        *dst += weight * *val;
                    }
                }
            }
        }

        out
    }

    /// Prefill with non-empty cache: dispatch must take the `prev > 0`
    /// fallback path (`append → dequantize → SDPA`) and produce results
    /// matching a CPU reference run over the concatenated history.
    ///
    /// This exercises the previously-untested prefill branch in
    /// `turboquant_dispatch::turboquant_attention_step` (audit Phase G).
    #[test]
    fn turboquant_dispatch_prefill_with_existing_cache_matches_reference() {
        use crate::turboquant_dispatch::turboquant_attention_step;

        let dim = 16usize;
        let heads = 2i32;
        let prev_len = 3i32;
        let new_len = 5i32;
        let total = prev_len + new_len;
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (d as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * heads * prev_len * d) as usize;
        let new_kv_len = (b * heads * new_len * d) as usize;
        let q_len = (b * heads * new_len * d) as usize;
        let total_kv_len = (b * heads * total * d) as usize;

        let prev_keys =
            InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.2), &[b, heads, prev_len, d]);
        let prev_values =
            InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.7), &[b, heads, prev_len, d]);
        let new_keys =
            InlineArray::from_f32_slice(&make_data(new_kv_len, 1.4), &[b, heads, new_len, d]);
        let new_values =
            InlineArray::from_f32_slice(&make_data(new_kv_len, 1.9), &[b, heads, new_len, d]);
        let queries =
            InlineArray::from_f32_slice(&make_data(q_len, 0.31), &[b, heads, new_len, d]);

        // Disable the recent-fp16 window so the dequantize fallback exercises
        // the cold compressed store, not the hot ring.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&prev_keys, &prev_values).expect("seed prefill");

        let mut output =
            turboquant_attention_step(&mut cache, &queries, &new_keys, &new_values, scale, prev_len, "TEST");
        let actual = output
            .to_f32_vec((b * heads * new_len * d) as usize)
            .expect("output to_f32");

        // Reference: stitch dequantized prev cache + raw new K/V, run causal
        // multi-token attention on the full concatenation. Matches what the
        // dispatch's `prev > 0` branch is *trying* to compute.
        let prev_keys_dq = cache
            .dequantize_keys()
            .expect("dequantize keys")
            .slice(&[0, 0, 0, 0], &[b, heads, total, d]);
        let prev_values_dq = cache
            .dequantize_values()
            .expect("dequantize values")
            .slice(&[0, 0, 0, 0], &[b, heads, total, d]);

        let mut ref_keys = prev_keys_dq;
        let mut ref_values = prev_values_dq;
        let mut ref_queries = queries.clone();
        let _ = ref_keys.to_f32_vec(total_kv_len);
        let _ = ref_values.to_f32_vec(total_kv_len);
        let expected = manual_causal_multi_token_attention(
            &mut ref_queries,
            &mut ref_keys,
            &mut ref_values,
            b,
            heads,
            new_len,
            total,
            d,
            scale,
        );

        let max_abs_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 5e-3,
            "prefill-with-history dispatch diverged from CPU reference: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_reference_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_keys_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        assert_eq!(full_keys.shape(), &[b, h, 4, d]);
        let vals = full_keys
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("keys to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_values_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        assert_eq!(full_values.shape(), &[b, h, 4, d]);
        let vals = full_values
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("values to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    /// Verify that multiple appends accumulate correctly in the GPU store.
    #[test]
    fn turboquant_gpu_multi_append() {
        let dim = 8usize;
        let config = TurboQuantConfig::uniform(4, 4);
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;

        let make_data = |seed: f32| -> Vec<f32> {
            (0..b * h * d)
                .map(|i| (i as f32 * 0.15 + seed).sin())
                .collect()
        };

        let mut cache = QuantizedKvCache::new(config);

        // Append 3 steps individually.
        for step in 0..3 {
            let data = make_data(step as f32);
            let arr = InlineArray::from_f32_slice(&data, &[b, h, 1, d]);
            cache.append(&arr, &arr).expect("append step");
        }

        assert_eq!(cache.offset, 3, "Should have 3 cached positions");

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        // Shape should be [B, H, 3, D].
        assert_eq!(dk.shape(), &[b, h, 3, d]);
    }

    // ─── Hot/cold split (Phase C bridge mirror) ──────────────────────────────

    /// With the recent-fp16 window enabled, short prompts must stay
    /// uncompressed in the hot ring. The cold stores never come online.
    #[test]
    fn hot_window_keeps_short_context_uncompressed() {
        let dim = 16usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(Some(64));
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let prefill = 32i32;
        let total = (b * h * prefill * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("append");

        assert_eq!(cache.hot_len(), prefill as usize);
        assert_eq!(cache.cold_len(), 0);
        assert_eq!(cache.offset, prefill as usize);
        assert!(cache.keys.is_none(), "cold key store should not exist yet");
        assert!(cache.values.is_none(), "cold value store should not exist yet");

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        assert_eq!(dk.shape(), &[b, h, prefill, d]);
    }

    /// Once the hot ring exceeds `recent_window + HOT_EVICTION_CHUNK`,
    /// the oldest tokens spill into the cold compressed stores. The
    /// invariant `offset == cold_offset + hot_offset` must hold.
    #[test]
    fn hot_window_evicts_to_cold_after_overflow() {
        let dim = 16usize;
        // Tiny window so we can exercise eviction without enormous tensors.
        // Eviction triggers when hot_offset > window + HOT_EVICTION_CHUNK.
        // With window = 4, that's > 4 + 1024 = > 1028.
        let window = 4usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(Some(window));
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let total_tokens = 1100i32; // > window + HOT_EVICTION_CHUNK
        let total = (b * h * total_tokens * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, total_tokens, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, total_tokens, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("prefill");

        assert_eq!(cache.offset, total_tokens as usize);
        assert_eq!(cache.cold_len() + cache.hot_len(), total_tokens as usize);
        assert!(cache.cold_len() > 0, "expected some tokens in cold");
        assert!(
            cache.hot_len() <= window + HOT_EVICTION_CHUNK,
            "hot ring should not exceed window+chunk after eviction"
        );

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        assert_eq!(dk.shape(), &[b, h, total_tokens, d]);
    }

    /// Legacy mode (`recent_window: None`) compresses every appended token
    /// immediately; the hot ring must stay disabled.
    #[test]
    fn legacy_recent_window_none_compresses_immediately() {
        let dim = 16usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let prefill = 8i32;
        let total = (b * h * prefill * d) as usize;
        let data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.09).cos()).collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("append");

        assert_eq!(cache.cold_len(), prefill as usize);
        assert_eq!(cache.hot_len(), 0);
        assert!(cache.hot_keys.is_none());
        assert!(cache.hot_values.is_none());
    }

    // ─── Defensive residual-norm clamp (A1 from audit) ───────────────────────
    //
    // Pathological inputs (NaN / ±Inf from upstream fp16 corruption) must not
    // propagate into the QJL term. The CPU encode path uses an explicit
    // `is_finite` + `clamp` guard; the GPU path composes `maximum(0).minimum(MAX)`.
    //
    // These tests exercise the CPU path directly since the GPU op graph
    // requires a live Metal device and is covered by the broader GPU integration
    // tests above.

    #[test]
    fn residual_norm_clamp_sanitizes_nan_input() {
        // One row of NaN should produce finite residual norm (0) — the row is
        // treated as zero by the upstream `norm <= ZERO_EPSILON` check, but if
        // it slipped past that the clamp still catches it.
        let core = TurboQuantCore::new(16, 4);
        let mut row = vec![0.1f32; 16];
        row[0] = f32::NAN;
        let encoded = encode_key_component_rows(&core, &row, 4);
        assert_eq!(encoded.residual_norms.len(), 1);
        assert!(
            encoded.residual_norms[0].is_finite(),
            "residual_norm must be finite even with NaN input, got {}",
            encoded.residual_norms[0]
        );
        assert!(
            (0.0..=MAX_RESIDUAL_NORM).contains(&encoded.residual_norms[0]),
            "residual_norm must be in [0, MAX], got {}",
            encoded.residual_norms[0]
        );
    }

    #[test]
    fn residual_norm_clamp_caps_inf_input() {
        let core = TurboQuantCore::new(16, 4);
        let mut row = vec![1.0f32; 16];
        row[5] = f32::INFINITY;
        let encoded = encode_key_component_rows(&core, &row, 4);
        assert!(
            encoded.residual_norms[0].is_finite(),
            "Inf input must not leak into residual_norm"
        );
        assert!(encoded.residual_norms[0] <= MAX_RESIDUAL_NORM);
    }

    // ─── Round-trip correctness (T1 from audit) ──────────────────────────────
    //
    // Verify the key invariants from turboquant.pdf Theorem 1 + Theorem 2 on
    // the deterministic CPU encode/decode path:
    //
    //   1. Round-trip reconstruction error has per-row MSE bounded by a
    //      constant that shrinks with bit-width (distortion bound).
    //   2. The inner product <q, decode(encode(k))> is approximately unbiased
    //      for <q, k> averaged across many random q, k pairs.

    fn seeded_gaussian_rows(num_rows: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Box-Muller over a xorshift stream — deterministic, no external crate.
        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }
        let mut state = seed.max(1);
        let mut u = || -> f32 {
            // Uniform (0, 1).
            let raw = xorshift64(&mut state);
            ((raw >> 40) as f32 + 1.0) / ((1u64 << 24) as f32)
        };
        let total = num_rows * dim;
        let mut out = Vec::with_capacity(total);
        while out.len() < total {
            let u1 = u();
            let u2 = u();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            out.push(r * theta.cos());
            if out.len() < total {
                out.push(r * theta.sin());
            }
        }
        out.truncate(total);
        out
    }

    fn decode_cpu_key(
        core: &TurboQuantCore,
        encoded: &EncodedKeyRows,
        bits: u8,
        _num_rows: usize,
    ) -> Vec<f32> {
        // Delegate to the production CPU decode path so the test exercises the
        // same arithmetic as live inference.
        decode_key_component_rows_raw(
            core,
            &encoded.mse_indices,
            &encoded.qjl_signs,
            &encoded.norms,
            &encoded.residual_norms,
            bits,
        )
    }

    #[test]
    fn turboquant_cpu_round_trip_error_bound_shrinks_with_bits() {
        // Distortion bound from Theorem 1: per-row MSE scales like 1/2^(2*(b-1))
        // for the MSE stage, with QJL residual correction adding an unbiased
        // zero-mean term. Across enough random rows, the average squared error
        // should be strictly smaller at higher bit widths.
        let dim = 64;
        let num_rows = 128;
        let data = seeded_gaussian_rows(num_rows, dim, 0xA1B2_C3D4_E5F6_0789);

        let mut errors = Vec::new();
        for &bits in &[3u8, 5u8, 7u8] {
            let core = TurboQuantCore::new(dim, bits);
            let encoded = encode_key_component_rows(&core, &data, bits);
            let decoded = decode_cpu_key(&core, &encoded, bits, num_rows);
            // Per-element MSE, averaged across all rows.
            let mse: f32 = data
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / (num_rows * dim) as f32;
            assert!(
                mse.is_finite(),
                "MSE must be finite for bits={}, got {}",
                bits,
                mse
            );
            errors.push((bits, mse));
        }

        // Monotonicity: more bits ⇒ less error. This is the concrete form of
        // the distortion bound at the statistical level.
        for window in errors.windows(2) {
            let (b_lo, mse_lo) = window[0];
            let (b_hi, mse_hi) = window[1];
            assert!(
                mse_hi < mse_lo,
                "Error at {} bits ({}) should beat {} bits ({})",
                b_hi,
                mse_hi,
                b_lo,
                mse_lo
            );
        }

        // Sanity floor: highest bit-width should reconstruct to a small
        // absolute error — Gaussian data normalized to unit sphere with 7-bit
        // MSE + QJL should land well below 0.5 per-element MSE.
        let (_, worst_case_at_7_bits) = errors.last().copied().unwrap();
        assert!(
            worst_case_at_7_bits < 0.5,
            "7-bit TurboQuant MSE {} is unexpectedly high",
            worst_case_at_7_bits
        );
    }

    #[test]
    fn turboquant_cpu_inner_product_is_approximately_unbiased() {
        // Paper Theorem 2: E[<q, k̂>] = <q, k> for keys encoded via the 2-stage
        // MSE + QJL path. With enough independent (q, k) pairs, the mean
        // reconstructed inner product should track the ground-truth mean.
        let dim = 128;
        let num_rows = 256;
        let bits = 5u8;
        let core = TurboQuantCore::new(dim, bits);

        let keys = seeded_gaussian_rows(num_rows, dim, 0x1111_2222_3333_4444);
        let queries = seeded_gaussian_rows(num_rows, dim, 0x5555_6666_7777_8888);

        let encoded = encode_key_component_rows(&core, &keys, bits);
        let decoded = decode_cpu_key(&core, &encoded, bits, num_rows);

        let mut sum_gt = 0.0f64;
        let mut sum_est = 0.0f64;
        let mut sum_abs_rel_err = 0.0f64;
        let mut valid = 0usize;
        for row_idx in 0..num_rows {
            let start = row_idx * dim;
            let k_row = &keys[start..start + dim];
            let q_row = &queries[start..start + dim];
            let k_hat = &decoded[start..start + dim];
            let gt: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
            let est: f32 = q_row.iter().zip(k_hat.iter()).map(|(a, b)| a * b).sum();
            sum_gt += gt as f64;
            sum_est += est as f64;
            if gt.abs() > 1e-3 {
                sum_abs_rel_err += ((est - gt) / gt).abs() as f64;
                valid += 1;
            }
        }
        let mean_gt = sum_gt / num_rows as f64;
        let mean_est = sum_est / num_rows as f64;
        let _ = (sum_abs_rel_err, valid); // per-row rel err is high-variance at low bits

        // Unbiasedness: the sample mean of the reconstructed inner product
        // should match the ground-truth mean to within the CLT-expected
        // standard error. For 256 rows of 128-dim Gaussian vectors the
        // per-row variance is O(1), so the standard error of the mean is
        // O(1/sqrt(256)) = 0.0625. We allow 4x headroom to keep the test
        // stable across platforms.
        let diff = (mean_est - mean_gt).abs();
        assert!(
            diff < 0.25,
            "Mean reconstructed inner product {} diverges from ground truth {} (diff {})",
            mean_est,
            mean_gt,
            diff
        );
    }

    /// Phase 3a — direct round-trip: bypass `QuantizedKvCache`, encode and
    /// decode via the new GPU Mixed helpers, verify reconstruction error
    /// stays within the q3_5 codebook budget. Catches encode/decode bugs
    /// without dragging in cache-state interactions.
    #[test]
    fn turboquant_gpu_quantize_kv_mixed_round_trip_no_cache() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.137).sin() + if i % 17 == 0 { 3.0 } else { 0.0 })
            .collect();
        let arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let state = TurboQuantState::new(dim, dim, config);

        let (kstore, vstore) = gpu_quantize_kv_mixed(&state, &arr, &arr, config)
            .expect("gpu_quantize_kv_mixed");

        let mut dec_keys = gpu_dequantize_keys_mixed(&kstore, &state.keys, &config)
            .expect("dequantize keys");
        let mut dec_vals = gpu_dequantize_values_mixed(&vstore, &state.values, &config)
            .expect("dequantize values");
        let recon_keys = dec_keys.to_f32_vec(total).expect("dec keys to_f32");
        let recon_vals = dec_vals.to_f32_vec(total).expect("dec vals to_f32");

        let max_key_err = recon_keys
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o).abs())
            .fold(0.0f32, f32::max);
        let max_val_err = recon_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o).abs())
            .fold(0.0f32, f32::max);
        // q3_5 on rows with outliers in [-3, 3] reconstructs well below 1.5.
        // Tighter bounds risk legitimate codebook variance; this gates real
        // bugs (lazy-graph corruption gave err ≈ 5.5 before the eval barrier).
        assert!(
            max_key_err < 1.5,
            "key reconstruction error {max_key_err} too large"
        );
        assert!(
            max_val_err < 1.5,
            "val reconstruction error {max_val_err} too large"
        );
    }

    /// Phase 3a sanity: gather sub-vectors, then scatter back. Round-trip
    /// must equal the original input.
    #[test]
    fn turboquant_gpu_mixed_partition_gather_scatter_round_trip() {
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = 128i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| (i as f32) * 0.137 + if i % 17 == 0 { 3.0 } else { 0.0 })
            .collect();
        let arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let (reg_src, out_src) =
            gpu_compute_outlier_partition(&arr, 32, 128).expect("partition");
        let reg = arr.take_along_axis(&reg_src, -1);
        let out = arr.take_along_axis(&out_src, -1);

        let reg_src_i32 = reg_src.as_dtype(Dtype::Int32.as_i32());
        let out_src_i32 = out_src.as_dtype(Dtype::Int32.as_i32());

        let zero = InlineArray::zeros(&[b, h, s, d], Dtype::Float32.as_i32());
        let with_reg = zero.put_along_axis_op(&reg_src_i32, &reg, -1);
        let mut merged = with_reg.put_along_axis_op(&out_src_i32, &out, -1);

        let merged_vec = merged.to_f32_vec(total).expect("merged to_f32");
        let max_diff = merged_vec
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "partition+gather+scatter round-trip max_abs_diff = {max_diff}"
        );
    }

    /// Phase 3a invariant: Mixed sub-vector encode/decode helpers must
    /// produce bit-identical results to the existing Uniform helpers when
    /// pointed at the same core + same data. Catches drift between the
    /// two pipelines.
    #[test]
    fn turboquant_gpu_mixed_subvector_matches_uniform_when_no_partition() {
        let dim = 96usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.13).sin() * 0.5)
            .collect();
        let rows = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let config = TurboQuantConfig::uniform(3, 4).with_recent_window(None);
        let state = TurboQuantState::new(dim, dim, config);
        let core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core.as_ref(),
            _ => panic!("expected Uniform"),
        };

        // Uniform path
        let (kstore_uni, _) =
            gpu_quantize_kv(&state, &rows, &vals, config).expect("uniform encode");
        let mut dec_uni = gpu_dequantize_keys(&kstore_uni, &state.keys, 3).expect("uniform decode");
        let recon_uni = dec_uni.to_f32_vec(total).expect("dec uni to_f32");

        // Mixed sub-vector path on the SAME core
        let enc_mix = gpu_encode_key_subvector(&rows, core, 3).expect("mix encode");
        let mut dec_mix = gpu_dequantize_key_subvector(
            &enc_mix.indices,
            &enc_mix.qjl_signs,
            &enc_mix.norms,
            &enc_mix.residual_norms,
            core,
            3,
        )
        .expect("mix decode");
        let recon_mix = dec_mix.to_f32_vec(total).expect("dec mix to_f32");

        let max_diff = recon_uni
            .iter()
            .zip(recon_mix.iter())
            .map(|(u, m)| (u - m).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Uniform vs Mixed helpers diverge: {max_diff}"
        );
    }

    /// Phase 3a — round-trip: append Mixed-precision rows, then dequantize
    /// from both the CPU `PackedBits` store (existing) and the new
    /// `GpuMixedKeyStore`/`GpuMixedValueStore`. Both paths share encode-time
    /// math (rotation, codebooks, QJL) but the partition is independently
    /// computed (CPU sort vs GPU argpartition+argsort), so we tolerate small
    /// ordering-induced ties at the boundary of outlier vs regular slots.
    #[test]
    fn turboquant_gpu_mixed_storage_round_trip_matches_cpu_dequantize() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);

        // Deterministic input — sawtooth + sin gives non-trivial outlier
        // structure (some dims with much larger magnitude than others),
        // which is what the Mixed encode is designed for.
        let data: Vec<f32> = (0..total)
            .map(|i| {
                let phase = (i as f32) * 0.137;
                let outlier_kick = if i % 17 == 0 { 3.0 } else { 0.0 };
                phase.sin() + outlier_kick
            })
            .collect();
        let keys_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys_arr, &vals_arr).expect("append");

        let kstore_cpu = cache.keys.as_ref().expect("keys");
        let vstore_cpu = cache.values.as_ref().expect("values");
        let kstore_gpu = kstore_cpu.gpu_mixed.as_ref().expect("gpu_mixed keys populated");
        let vstore_gpu = vstore_cpu.gpu_mixed.as_ref().expect("gpu_mixed values populated");
        let state = cache.state.as_ref().expect("state");

        let mut cpu_keys = cache.dequantize_keys().expect("CPU dequantize_keys");
        let mut cpu_values = cache.dequantize_values().expect("CPU dequantize_values");
        let mut gpu_keys = gpu_dequantize_keys_mixed(kstore_gpu, &state.keys, &config)
            .expect("gpu_dequantize_keys_mixed");
        let mut gpu_values = gpu_dequantize_values_mixed(vstore_gpu, &state.values, &config)
            .expect("gpu_dequantize_values_mixed");

        let cpu_keys_vec = cpu_keys.to_f32_vec(total).expect("cpu keys to_f32");
        let cpu_values_vec = cpu_values.to_f32_vec(total).expect("cpu values to_f32");
        let gpu_keys_vec = gpu_keys.to_f32_vec(total).expect("gpu keys to_f32");
        let gpu_values_vec = gpu_values.to_f32_vec(total).expect("gpu values to_f32");

        let max_err = |recon: &[f32], orig: &[f32]| {
            recon.iter()
                .zip(orig.iter())
                .map(|(r, o)| (r - o).abs())
                .fold(0.0f32, f32::max)
        };
        let cpu_err_k = max_err(&cpu_keys_vec, &data);
        let gpu_err_k = max_err(&gpu_keys_vec, &data);
        let cpu_err_v = max_err(&cpu_values_vec, &data);
        let gpu_err_v = max_err(&gpu_values_vec, &data);

        // Quality gate: both paths reconstruct the original input within
        // q3_5 codebook error. The GPU path's lazy-graph eval barrier in
        // gpu_quantize_kv_mixed is load-bearing — without it, the partition
        // chain re-evaluates inconsistently and gpu_err_k goes from ~0.75 to
        // ~5.5. This test is the regression gate for that fix.
        assert!(gpu_err_k < 1.5, "GPU keys reconstruction error {gpu_err_k}");
        assert!(gpu_err_v < 1.5, "GPU values reconstruction error {gpu_err_v}");
        // Pairwise diff is bounded by the sum of per-path errors plus slack
        // for the independent outlier-mask choices on borderline ties.
        let pairwise_k = max_err(&cpu_keys_vec, &gpu_keys_vec);
        let pairwise_v = max_err(&cpu_values_vec, &gpu_values_vec);
        assert!(
            pairwise_k < (cpu_err_k + gpu_err_k) * 1.5 + 0.1,
            "pairwise key diff {pairwise_k} disproportionate (cpu {cpu_err_k}, gpu {gpu_err_k})"
        );
        assert!(
            pairwise_v < (cpu_err_v + gpu_err_v) * 1.5 + 0.1,
            "pairwise val diff {pairwise_v} disproportionate (cpu {cpu_err_v}, gpu {gpu_err_v})"
        );
    }

    /// Phase 3b layout oracle: the dormant `mlx_inline_turboquant_mixed_score`
    /// kernel reads the Phase 3a Mixed store directly. If our `[B,H,T,D_*]`
    /// layout, dtype choice, or qjl_words count is off by one, this kernel
    /// returns garbage. We compare its output against a reference computed
    /// by dequantising the same store and dot-producting with the query.
    ///
    /// Constraint: T=1 because the kernel's signature carries a single
    /// `[N, D_sub]` query slice — Phase 3c attention kernels gather Q
    /// per-slot from the full `[N, D_total]` query. See `try_gpu_mixed_score`
    /// docstring.
    #[test]
    fn turboquant_gpu_mixed_score_matches_dequantize_dot_product() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32; // q_heads = kv_heads = 4 (no GQA grouping in this test)
        let s = 1i32; // T=1 — see oracle constraint above.
        let d = dim as i32;
        let total_keys = (b * h * s * d) as usize;
        let q_total = (b * h * d) as usize;

        // Synthetic K with deliberate outliers so the regular/outlier split
        // exercises both codebooks. Phase wraps at 0.137 rad/element.
        let key_data: Vec<f32> = (0..total_keys)
            .map(|i| {
                let phase = (i as f32) * 0.137;
                let outlier_kick = if i % 17 == 0 { 3.0 } else { 0.0 };
                phase.sin() + outlier_kick
            })
            .collect();
        let q_data: Vec<f32> = (0..q_total)
            .map(|i| ((i as f32) * 0.211).cos())
            .collect();
        let keys_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);
        let queries_arr = InlineArray::from_f32_slice(&q_data, &[b, h, 1, d]);

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys_arr, &vals_arr).expect("append");

        let kstore = cache
            .keys
            .as_ref()
            .and_then(|ks| ks.gpu_mixed.as_ref())
            .expect("gpu_mixed kstore populated");
        let state = cache.state.as_ref().expect("state");

        let scale = 1.0f32 / (dim as f32).sqrt();
        let mut gpu_scores = try_gpu_mixed_score(state, &config, kstore, &queries_arr, h, h, s, scale)
            .expect("try_gpu_mixed_score");

        // Reference: dequantise K → recon_keys [B, H, T, D]; then
        // scores[n, t] = sum_d Q[n, d] * recon_keys[kv_row, t, d] * scale
        // with N = B·q_heads = B·H (kv_heads == q_heads here).
        let mut recon_keys = gpu_dequantize_keys_mixed(kstore, &state.keys, &config)
            .expect("gpu_dequantize_keys_mixed");
        let recon_vec = recon_keys.to_f32_vec(total_keys).expect("recon to_f32");
        let n = (b * h) as usize;
        let t = s as usize;
        let mut ref_scores = vec![0.0f32; n * t];
        for row in 0..n {
            // GQA mapping: kv_row = batch * kv_heads + (q_head / groups).
            // groups=1 here, so kv_row == row.
            let kv_row = row;
            for slot in 0..t {
                let mut acc = 0.0f32;
                for di in 0..(dim) {
                    let q = q_data[row * dim + di];
                    let k = recon_vec[(kv_row * t + slot) * dim + di];
                    acc += q * k;
                }
                ref_scores[row * t + slot] = acc * scale;
            }
        }

        let gpu_vec = gpu_scores.to_f32_vec(n * t).expect("gpu_scores to_f32");
        let max_diff = ref_scores
            .iter()
            .zip(gpu_vec.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0f32, f32::max);
        let abs_max_ref = ref_scores
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        // The kernel and the reference compute the same expression in
        // different orders (kernel: rotated-space dot product + qjl-residual
        // term; reference: full-space dot product over dequantised K).
        // Floating-point reassociation gives roundoff ≈ 1e-3 of the score
        // magnitude. A real layout bug (off-by-one stride, wrong dtype, mask
        // mis-aligned with codebook lookup) flips this into >0.1·abs_max_ref.
        let tol = 5e-3 * abs_max_ref;
        assert!(
            max_diff < tol,
            "mixed_score layout drift: max_diff={max_diff}, tol={tol} (abs_max_ref={abs_max_ref})"
        );
    }
}

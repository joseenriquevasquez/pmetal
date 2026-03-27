//! TurboQuant KV cache.
//!
//! This implements a practical TurboQuant-inspired KV cache for MLX tensors:
//! - vectors are normalized onto the unit sphere and their norms are stored
//! - keys use the paper's two-stage inner-product quantizer
//! - values use the MSE-optimized scalar codebook path
//!
//! The implementation keeps the paper's data flow and storage layout while
//! batching the dense rotation and QJL transforms through Metal when available.
//! CPU fallback remains in place for unsupported dimensions or runtime Metal
//! failures, so the cache stays usable end-to-end across environments.

use std::{f32::consts::PI, sync::Arc};

use mlx_rs::{Array, Dtype, error::Exception};
use pmetal_metal::{MetalContext, TurboQuantTransform};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tracing::debug;

use crate::kernels::{AttentionMaskType, FusedAttentionConfig};

/// Deterministic seed used for TurboQuant rotations and QJL projections.
const TURBOQUANT_SEED: u64 = 0x5442_5155_414e_544d;
const ZERO_EPSILON: f32 = 1e-12;
const LLOYD_MAX_ITERS: usize = 64;
const LLOYD_MAX_TOLERANCE: f64 = 1e-7;
const LLOYD_GRID_POINTS: usize = 8192;

/// Per-tensor TurboQuant precision configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantTensorConfig {
    /// Single TurboQuant instance across the full vector.
    Uniform {
        /// Total effective bits per channel.
        bits: u8,
    },
    /// Two independent TurboQuant instances split by outlier mask.
    Mixed {
        /// Bit-width for non-outlier channels.
        regular_bits: u8,
        /// Bit-width for outlier channels.
        outlier_bits: u8,
        /// Number of outlier channels per row.
        outlier_count: usize,
    },
}

impl TurboQuantTensorConfig {
    /// Create a uniform TurboQuant tensor config.
    pub const fn uniform(bits: u8) -> Self {
        Self::Uniform { bits }
    }

    /// Create a mixed-bit TurboQuant tensor config.
    pub const fn mixed(regular_bits: u8, outlier_bits: u8, outlier_count: usize) -> Self {
        Self::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        }
    }

    fn assert_valid(self, total_dim: usize, label: &str) {
        match self {
            Self::Uniform { bits } => {
                assert!(
                    (1..=8).contains(&bits),
                    "TurboQuant {label} bits must be in 1..=8"
                );
            }
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => {
                assert!(
                    (1..=8).contains(&regular_bits),
                    "TurboQuant {label} regular_bits must be in 1..=8"
                );
                assert!(
                    (1..=8).contains(&outlier_bits),
                    "TurboQuant {label} outlier_bits must be in 1..=8"
                );
                assert!(
                    outlier_count > 0 && outlier_count < total_dim,
                    "TurboQuant {label} outlier_count must be in 1..{total_dim}"
                );
                assert!(
                    outlier_bits >= regular_bits,
                    "TurboQuant {label} outlier_bits must be >= regular_bits"
                );
            }
        }
    }

    /// Number of outlier channels per row.
    pub fn outlier_count(self) -> usize {
        match self {
            Self::Uniform { .. } => 0,
            Self::Mixed { outlier_count, .. } => outlier_count,
        }
    }

    /// Number of regular channels per row.
    pub fn regular_dim(self, total_dim: usize) -> usize {
        total_dim - self.outlier_count()
    }

    /// Effective average bits per channel.
    pub fn effective_bits(self, total_dim: usize) -> f32 {
        match self {
            Self::Uniform { bits } => bits as f32,
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => {
                let regular_dim = total_dim - outlier_count;
                ((regular_dim * usize::from(regular_bits))
                    + (outlier_count * usize::from(outlier_bits))) as f32
                    / total_dim as f32
            }
        }
    }

    fn describe(self, total_dim: usize) -> String {
        match self {
            Self::Uniform { bits } => format!("{bits}b"),
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => format!(
                "{:.2}b ({}/{total_dim}@{outlier_bits}b, rest@{regular_bits}b)",
                self.effective_bits(total_dim),
                outlier_count
            ),
        }
    }
}

/// Full TurboQuant K/V configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantConfig {
    /// Key-cache quantization strategy.
    pub keys: TurboQuantTensorConfig,
    /// Value-cache quantization strategy.
    pub values: TurboQuantTensorConfig,
}

impl TurboQuantConfig {
    /// Create a uniform K/V TurboQuant config.
    pub const fn uniform(key_bits: u8, value_bits: u8) -> Self {
        Self {
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
        }
    }

    /// Create a mixed-bit K/V TurboQuant config.
    pub const fn mixed(
        key_regular_bits: u8,
        key_outlier_bits: u8,
        key_outlier_count: usize,
        value_regular_bits: u8,
        value_outlier_bits: u8,
        value_outlier_count: usize,
    ) -> Self {
        Self {
            keys: TurboQuantTensorConfig::mixed(
                key_regular_bits,
                key_outlier_bits,
                key_outlier_count,
            ),
            values: TurboQuantTensorConfig::mixed(
                value_regular_bits,
                value_outlier_bits,
                value_outlier_count,
            ),
        }
    }

    /// Outlier-aware 2.5-bit preset.
    ///
    /// The paper's page-18 arithmetic example is internally inconsistent. We
    /// keep the stated "outlier channel" strategy and use a mathematically
    /// correct mixed schedule: top 25% channels at 4 bits, the remainder at 2.
    pub fn preset_q2_5(total_dim: usize) -> Self {
        let outlier_count = recommended_outlier_count(total_dim);
        Self::mixed(2, 4, outlier_count, 2, 4, outlier_count)
    }

    /// Outlier-aware 3.5-bit preset.
    pub fn preset_q3_5(total_dim: usize) -> Self {
        let outlier_count = recommended_outlier_count(total_dim);
        Self::mixed(3, 5, outlier_count, 3, 5, outlier_count)
    }

    /// Human-readable description.
    pub fn describe(self, total_dim: usize) -> String {
        format!(
            "turboquant K@{},V@{}",
            self.keys.describe(total_dim),
            self.values.describe(total_dim)
        )
    }
}

fn recommended_outlier_count(total_dim: usize) -> usize {
    if total_dim <= 1 {
        0
    } else {
        total_dim.div_ceil(4).min(total_dim - 1)
    }
}

#[derive(Debug, Clone, Copy)]
struct TurboLayout {
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

#[derive(Debug, Clone)]
struct PackedBits {
    bits_per_value: u8,
    len: usize,
    bytes: Vec<u8>,
}

impl PackedBits {
    fn new(bits_per_value: u8) -> Self {
        Self {
            bits_per_value,
            len: 0,
            bytes: Vec::new(),
        }
    }

    #[cfg(test)]
    fn from_values(bits_per_value: u8, values: &[u16]) -> Self {
        let mut packed = Self::new(bits_per_value);
        packed.extend_from_slice(values);
        packed
    }

    fn extend_from_slice(&mut self, values: &[u16]) {
        if self.bits_per_value == 0 || values.is_empty() {
            self.len += values.len();
            return;
        }

        for &value in values {
            debug_assert!(u32::from(value) < (1u32 << self.bits_per_value));
            let bit_offset = self.len * usize::from(self.bits_per_value);
            let required_bits = bit_offset + usize::from(self.bits_per_value);
            let required_bytes = required_bits.div_ceil(8);
            if self.bytes.len() < required_bytes {
                self.bytes.resize(required_bytes, 0);
            }
            for bit in 0..self.bits_per_value {
                let bit_is_set = ((value >> bit) & 1) != 0;
                if bit_is_set {
                    let target_bit = bit_offset + usize::from(bit);
                    self.bytes[target_bit / 8] |= 1u8 << (target_bit % 8);
                }
            }
            self.len += 1;
        }
    }

    fn get(&self, index: usize) -> u16 {
        debug_assert!(index < self.len);
        if self.bits_per_value == 0 {
            return 0;
        }

        let bit_offset = index * usize::from(self.bits_per_value);
        let mut value = 0u16;
        for bit in 0..self.bits_per_value {
            let target_bit = bit_offset + usize::from(bit);
            let byte = self.bytes[target_bit / 8];
            let bit_is_set = ((byte >> (target_bit % 8)) & 1) != 0;
            if bit_is_set {
                value |= 1u16 << bit;
            }
        }
        value
    }

    fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        self.len = new_len;
        if self.bits_per_value == 0 {
            return;
        }

        let total_bits = self.len * usize::from(self.bits_per_value);
        let total_bytes = total_bits.div_ceil(8);
        self.bytes.truncate(total_bytes);
        if let Some(last) = self.bytes.last_mut() {
            let used_bits = total_bits % 8;
            if used_bits != 0 {
                *last &= (1u8 << used_bits) - 1;
            }
        }
    }
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Debug, Clone)]
struct TurboValueStore {
    regular_indices: PackedBits,
    regular_norms: Vec<f32>,
    outlier_mask: Option<PackedBits>,
    outlier_indices: Option<PackedBits>,
    outlier_norms: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct TurboKeyStore {
    regular_indices: PackedBits,
    regular_qjl_signs: PackedBits,
    regular_norms: Vec<f32>,
    regular_residual_norms: Vec<f32>,
    outlier_mask: Option<PackedBits>,
    outlier_indices: Option<PackedBits>,
    outlier_qjl_signs: Option<PackedBits>,
    outlier_norms: Option<Vec<f32>>,
    outlier_residual_norms: Option<Vec<f32>>,
}

impl TurboValueStore {
    fn new(config: TurboQuantTensorConfig, _total_dim: usize) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits,
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits,
        };
        let outlier_bits = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => Some(outlier_bits),
        };

        Self {
            regular_indices: PackedBits::new(regular_bits),
            regular_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    fn extend(&mut self, encoded: &EncodedTurboValueRows) {
        self.regular_indices
            .extend_from_slice(&encoded.regular.indices);
        self.regular_norms
            .extend(encoded.regular.norms.iter().copied());

        if let Some(mask) = &encoded.outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant value outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = &encoded.outlier {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant value outlier indices missing")
                .extend_from_slice(&outlier.indices);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant value outlier norms missing")
                .extend(outlier.norms.iter().copied());
        }
    }

    fn truncate(&mut self, keep_rows: usize, total_dim: usize, config: TurboQuantTensorConfig) {
        self.regular_indices
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_norms.truncate(keep_rows);

        if let Some(mask) = &mut self.outlier_mask {
            mask.truncate(keep_rows * total_dim);
        }
        if let Some(outlier_indices) = &mut self.outlier_indices {
            outlier_indices.truncate(keep_rows * config.outlier_count());
        }
        if let Some(outlier_norms) = &mut self.outlier_norms {
            outlier_norms.truncate(keep_rows);
        }
    }

    fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_norms.len() * std::mem::size_of::<f32>()
            + self.outlier_mask.as_ref().map_or(0, PackedBits::byte_len)
            + self
                .outlier_indices
                .as_ref()
                .map_or(0, PackedBits::byte_len)
            + self
                .outlier_norms
                .as_ref()
                .map_or(0, |norms| norms.len() * std::mem::size_of::<f32>())
    }
}

impl TurboKeyStore {
    fn new(config: TurboQuantTensorConfig, _total_dim: usize) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits.saturating_sub(1),
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits.saturating_sub(1),
        };
        let outlier_bits = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => {
                Some(outlier_bits.saturating_sub(1))
            }
        };

        Self {
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

    fn extend(&mut self, encoded: &EncodedTurboKeyRows) {
        self.regular_indices
            .extend_from_slice(&encoded.regular.mse_indices);
        self.regular_qjl_signs
            .extend_from_slice(&encoded.regular.qjl_signs);
        self.regular_norms
            .extend(encoded.regular.norms.iter().copied());
        self.regular_residual_norms
            .extend(encoded.regular.residual_norms.iter().copied());

        if let Some(mask) = &encoded.outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant key outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = &encoded.outlier {
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
                .extend(outlier.norms.iter().copied());
            self.outlier_residual_norms
                .as_mut()
                .expect("TurboQuant key outlier residual norms missing")
                .extend(outlier.residual_norms.iter().copied());
        }
    }

    fn truncate(&mut self, keep_rows: usize, total_dim: usize, config: TurboQuantTensorConfig) {
        self.regular_indices
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_qjl_signs
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_norms.truncate(keep_rows);
        self.regular_residual_norms.truncate(keep_rows);

        if let Some(mask) = &mut self.outlier_mask {
            mask.truncate(keep_rows * total_dim);
        }
        if let Some(outlier_indices) = &mut self.outlier_indices {
            outlier_indices.truncate(keep_rows * config.outlier_count());
        }
        if let Some(outlier_qjl_signs) = &mut self.outlier_qjl_signs {
            outlier_qjl_signs.truncate(keep_rows * config.outlier_count());
        }
        if let Some(outlier_norms) = &mut self.outlier_norms {
            outlier_norms.truncate(keep_rows);
        }
        if let Some(outlier_residual_norms) = &mut self.outlier_residual_norms {
            outlier_residual_norms.truncate(keep_rows);
        }
    }

    fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_qjl_signs.byte_len()
            + self.regular_norms.len() * std::mem::size_of::<f32>()
            + self.regular_residual_norms.len() * std::mem::size_of::<f32>()
            + self.outlier_mask.as_ref().map_or(0, PackedBits::byte_len)
            + self
                .outlier_indices
                .as_ref()
                .map_or(0, PackedBits::byte_len)
            + self
                .outlier_qjl_signs
                .as_ref()
                .map_or(0, PackedBits::byte_len)
            + self
                .outlier_norms
                .as_ref()
                .map_or(0, |norms| norms.len() * std::mem::size_of::<f32>())
            + self
                .outlier_residual_norms
                .as_ref()
                .map_or(0, |norms| norms.len() * std::mem::size_of::<f32>())
    }
}

#[derive(Clone)]
struct TurboQuantMetalCore {
    rotation: TurboQuantTransform,
    inverse_rotation: TurboQuantTransform,
    qjl_projection: TurboQuantTransform,
    inverse_qjl_projection: TurboQuantTransform,
}

impl std::fmt::Debug for TurboQuantMetalCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboQuantMetalCore")
            .field("dim", &self.rotation.dim())
            .finish()
    }
}

impl TurboQuantMetalCore {
    fn new(
        ctx: Arc<MetalContext>,
        dim: usize,
        rotation: &[f32],
        inverse_rotation: &[f32],
        qjl_projection: &[f32],
        inverse_qjl_projection: &[f32],
    ) -> pmetal_metal::Result<Self> {
        Ok(Self {
            rotation: TurboQuantTransform::with_context(ctx.clone(), rotation, dim)?,
            inverse_rotation: TurboQuantTransform::with_context(
                ctx.clone(),
                inverse_rotation,
                dim,
            )?,
            qjl_projection: TurboQuantTransform::with_context(ctx.clone(), qjl_projection, dim)?,
            inverse_qjl_projection: TurboQuantTransform::with_context(
                ctx,
                inverse_qjl_projection,
                dim,
            )?,
        })
    }
}

pub(crate) struct TurboQuantCore {
    dim: usize,
    rotation: Vec<f32>,
    inverse_rotation: Vec<f32>,
    qjl_projection: Vec<f32>,
    inverse_qjl_projection: Vec<f32>,
    codebooks: Vec<Vec<f32>>,
    metal: Option<TurboQuantMetalCore>,
}

impl std::fmt::Debug for TurboQuantCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboQuantCore")
            .field("dim", &self.dim)
            .field("codebook_bits", &(self.codebooks.len().saturating_sub(1)))
            .field("metal_enabled", &self.metal.is_some())
            .finish()
    }
}

impl TurboQuantCore {
    fn new(dim: usize, max_mse_bits: u8) -> Self {
        let mut rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ ((dim as u64) << 32));
        let rotation = generate_random_orthogonal(dim, &mut rng);
        let inverse_rotation = transpose_square_matrix(&rotation, dim);
        let qjl_projection = generate_random_projection(dim, &mut rng);
        let inverse_qjl_projection = transpose_square_matrix(&qjl_projection, dim);

        let mut codebooks = vec![Vec::new(); usize::from(max_mse_bits) + 1];
        for bits in 1..=max_mse_bits {
            codebooks[usize::from(bits)] = build_beta_codebook(dim, bits);
        }

        let metal = match MetalContext::global().and_then(|ctx| {
            TurboQuantMetalCore::new(
                ctx,
                dim,
                &rotation,
                &inverse_rotation,
                &qjl_projection,
                &inverse_qjl_projection,
            )
        }) {
            Ok(metal) => Some(metal),
            Err(error) => {
                debug!(
                    dim,
                    "TurboQuant Metal backend unavailable, using CPU fallback: {error}"
                );
                None
            }
        };

        Self {
            dim,
            rotation,
            inverse_rotation,
            qjl_projection,
            inverse_qjl_projection,
            codebooks,
            metal,
        }
    }

    fn codebook(&self, bits: u8) -> &[f32] {
        &self.codebooks[usize::from(bits)]
    }

    fn rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_rows(
            "rotation",
            &self.rotation,
            self.metal.as_ref().map(|m| &m.rotation),
            input,
        )
    }

    fn inverse_rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_rows(
            "inverse-rotation",
            &self.inverse_rotation,
            self.metal.as_ref().map(|m| &m.inverse_rotation),
            input,
        )
    }

    fn project_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_rows(
            "qjl-projection",
            &self.qjl_projection,
            self.metal.as_ref().map(|m| &m.qjl_projection),
            input,
        )
    }

    fn inverse_project_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_rows(
            "inverse-qjl-projection",
            &self.inverse_qjl_projection,
            self.metal.as_ref().map(|m| &m.inverse_qjl_projection),
            input,
        )
    }

    fn apply_rows(
        &self,
        stage: &'static str,
        matrix: &[f32],
        metal: Option<&TurboQuantTransform>,
        input: &[f32],
    ) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if let Some(transform) = metal {
            match transform.apply_rows(input) {
                Ok(output) => return output,
                Err(error) => {
                    debug!(
                        stage,
                        dim = self.dim,
                        rows = input.len() / self.dim,
                        "TurboQuant Metal transform failed, falling back to CPU: {error}"
                    );
                }
            }
        }
        matmul_rows(matrix, self.dim, input)
    }
}

#[derive(Debug, Clone)]
enum TurboQuantTensorRuntime {
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

#[derive(Debug, Clone)]
pub(crate) struct TurboQuantRuntime {
    key_dim: usize,
    value_dim: usize,
    keys: TurboQuantTensorRuntime,
    values: TurboQuantTensorRuntime,
}

impl TurboQuantRuntime {
    fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Self {
        config.keys.assert_valid(key_dim, "keys");
        config.values.assert_valid(value_dim, "values");

        let mut core_cache = std::collections::HashMap::<(usize, u8), Arc<TurboQuantCore>>::new();
        let mut get_core = |subdim: usize, max_mse_bits: u8| {
            core_cache
                .entry((subdim, max_mse_bits))
                .or_insert_with(|| Arc::new(TurboQuantCore::new(subdim, max_mse_bits)))
                .clone()
        };

        let keys = build_tensor_runtime(key_dim, config.keys, true, &mut get_core);
        let values = build_tensor_runtime(value_dim, config.values, false, &mut get_core);

        Self {
            key_dim,
            value_dim,
            keys,
            values,
        }
    }
}

fn build_tensor_runtime<F>(
    total_dim: usize,
    config: TurboQuantTensorConfig,
    keys: bool,
    get_core: &mut F,
) -> TurboQuantTensorRuntime
where
    F: FnMut(usize, u8) -> Arc<TurboQuantCore>,
{
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            let max_mse_bits = if keys { bits.saturating_sub(1) } else { bits };
            TurboQuantTensorRuntime::Uniform {
                config,
                core: get_core(total_dim, max_mse_bits),
            }
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = total_dim - outlier_count;
            let regular_max_bits = if keys {
                regular_bits.saturating_sub(1)
            } else {
                regular_bits
            };
            let outlier_max_bits = if keys {
                outlier_bits.saturating_sub(1)
            } else {
                outlier_bits
            };
            TurboQuantTensorRuntime::Mixed {
                config,
                regular_core: get_core(regular_dim, regular_max_bits),
                outlier_core: get_core(outlier_count, outlier_max_bits),
            }
        }
    }
}

/// TurboQuant KV cache.
///
/// Keys use the inner-product quantizer from the paper and values use the
/// MSE-optimized quantizer.
#[derive(Debug)]
pub struct TurboQuantKvCache {
    keys: Option<TurboKeyStore>,
    values: Option<TurboValueStore>,
    layout: Option<TurboLayout>,
    offset: usize,
    config: TurboQuantConfig,
    dtype: Dtype,
    runtime: Option<Arc<TurboQuantRuntime>>,
}

impl TurboQuantKvCache {
    /// Create a new TurboQuant KV cache.
    ///
    /// `key_bits` and `value_bits` are the total effective bits per channel.
    /// Keys reserve one of those bits for the QJL residual stage.
    pub fn new(key_bits: u8, value_bits: u8) -> Self {
        Self::new_with_config(TurboQuantConfig::uniform(key_bits, value_bits))
    }

    /// Create a TurboQuant cache from an explicit configuration.
    pub fn new_with_config(config: TurboQuantConfig) -> Self {
        Self {
            keys: None,
            values: None,
            layout: None,
            offset: 0,
            config,
            dtype: Dtype::Float16,
            runtime: None,
        }
    }

    pub(crate) fn new_with_runtime(
        config: TurboQuantConfig,
        runtime: Arc<TurboQuantRuntime>,
    ) -> Self {
        let mut cache = Self::new_with_config(config);
        cache.runtime = Some(runtime);
        cache
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// Returns `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// RoPE offset for new tokens.
    pub fn rope_offset(&self) -> i32 {
        self.offset as i32
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.layout = None;
        self.offset = 0;
    }

    pub(crate) fn can_direct_attention(
        &self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        attn_config: &FusedAttentionConfig,
    ) -> bool {
        queries.shape().len() == 4
            && keys.shape().len() == 4
            && values.shape().len() == 4
            && queries.dim(2) == 1
            && keys.dim(2) == 1
            && values.dim(2) == 1
            && queries.dim(0) == keys.dim(0)
            && keys.dim(0) == values.dim(0)
            && keys.dim(1) == values.dim(1)
            && queries.dim(3) == keys.dim(3)
            && matches!(
                attn_config.mask_type,
                AttentionMaskType::None
                    | AttentionMaskType::Causal
                    | AttentionMaskType::SlidingWindow(_)
            )
    }

    fn append(&mut self, keys: &Array, values: &Array) -> Result<TurboLayout, Exception> {
        self.dtype = keys.dtype();
        let layout = self.ensure_layout(keys, values)?;
        let rows_per_seq = layout.batch * layout.heads;
        let seq_len = keys.dim(2) as usize;

        let key_rows = array_rows_in_bshd_order(keys)?;
        let value_rows = array_rows_in_bshd_order(values)?;

        let config = self.config;
        let runtime = self.runtime.get_or_insert_with(|| {
            Arc::new(TurboQuantRuntime::new(
                layout.key_dim,
                layout.value_dim,
                config,
            ))
        });

        let encoded_keys = encode_key_rows_for_runtime(&runtime.keys, layout.key_dim, &key_rows);
        let encoded_values =
            encode_value_rows_for_runtime(&runtime.values, layout.value_dim, &value_rows);

        let key_store = self
            .keys
            .get_or_insert_with(|| TurboKeyStore::new(self.config.keys, layout.key_dim));
        key_store.extend(&encoded_keys);

        let value_store = self
            .values
            .get_or_insert_with(|| TurboValueStore::new(self.config.values, layout.value_dim));
        value_store.extend(&encoded_values);

        self.offset += seq_len;
        debug_assert_eq!(
            key_store.regular_norms.len(),
            self.offset * rows_per_seq,
            "TurboQuant key store row count drifted"
        );

        Ok(layout)
    }

    /// Append a new `[B, H, S, D]` KV chunk and return the dequantized cache.
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.append(keys, values)?;
        Ok((self.dequantize_keys()?, self.dequantize_values()?))
    }

    /// Append a new `[B, H, S, D]` KV chunk and compute direct attention output
    /// from the compressed cache for single-token decode.
    pub fn append_and_compute_attention(
        &mut self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        attn_config: &FusedAttentionConfig,
    ) -> Result<Array, Exception> {
        if !self.can_direct_attention(queries, keys, values, attn_config) {
            return Err(Exception::custom(
                "TurboQuant direct attention requires [B, H, 1, D] single-token decode inputs"
                    .to_string(),
            ));
        }

        let layout = self.append(keys, values)?;
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant runtime missing"))?;
        let key_store = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant key store missing"))?;
        let value_store = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant value store missing"))?;

        direct_attention_output(
            queries,
            layout,
            self.offset,
            runtime,
            key_store,
            value_store,
            self.config,
            attn_config,
        )
    }

    /// Whether the cache supports trim.
    pub fn is_trimmable(&self) -> bool {
        true
    }

    /// Trim `n` tokens from the logical tail.
    pub fn trim(&mut self, n: usize) -> usize {
        let trimmed = n.min(self.offset);
        self.rollback(trimmed);
        trimmed
    }

    /// Roll back the last `n` cached tokens.
    pub fn rollback(&mut self, n: usize) {
        if n == 0 || self.offset == 0 {
            return;
        }

        let layout = match self.layout {
            Some(layout) => layout,
            None => return,
        };
        let keep_seq = self.offset.saturating_sub(n);
        let keep_rows = keep_seq * layout.batch * layout.heads;

        if let Some(keys) = &mut self.keys {
            keys.truncate(keep_rows, layout.key_dim, self.config.keys);
        }
        if let Some(values) = &mut self.values {
            values.truncate(keep_rows, layout.value_dim, self.config.values);
        }

        self.offset = keep_seq;
        if self.offset == 0 {
            self.keys = None;
            self.values = None;
            self.layout = None;
        }
    }

    /// Estimated storage used by the cache payload.
    pub fn memory_usage(&self) -> usize {
        let key_bytes = self.keys.as_ref().map_or(0, TurboKeyStore::memory_usage);
        let value_bytes = self
            .values
            .as_ref()
            .map_or(0, TurboValueStore::memory_usage);
        key_bytes + value_bytes
    }

    fn ensure_layout(&mut self, keys: &Array, values: &Array) -> Result<TurboLayout, Exception> {
        if keys.shape().len() != 4 || values.shape().len() != 4 {
            return Err(Exception::custom(format!(
                "TurboQuant KV cache expects [B, H, S, D], got {:?} vs {:?}",
                keys.shape(),
                values.shape()
            )));
        }
        if keys.dim(0) != values.dim(0)
            || keys.dim(1) != values.dim(1)
            || keys.dim(2) != values.dim(2)
        {
            return Err(Exception::custom(format!(
                "TurboQuant KV cache requires matching [B, H, S] axes, got {:?} vs {:?}",
                keys.shape(),
                values.shape()
            )));
        }

        let layout = TurboLayout {
            batch: keys.dim(0) as usize,
            heads: keys.dim(1) as usize,
            key_dim: keys.dim(3) as usize,
            value_dim: values.dim(3) as usize,
        };

        self.config.keys.assert_valid(layout.key_dim, "keys");
        self.config.values.assert_valid(layout.value_dim, "values");

        if self.runtime.as_ref().is_some_and(|runtime| {
            runtime.key_dim != layout.key_dim || runtime.value_dim != layout.value_dim
        }) && self.offset == 0
        {
            self.runtime = Some(Arc::new(TurboQuantRuntime::new(
                layout.key_dim,
                layout.value_dim,
                self.config,
            )));
        }

        match self.layout {
            Some(existing)
                if existing.batch != layout.batch
                    || existing.heads != layout.heads
                    || existing.key_dim != layout.key_dim
                    || existing.value_dim != layout.value_dim =>
            {
                Err(Exception::custom(format!(
                    "TurboQuant KV cache layout changed from {:?} to {:?}",
                    existing, layout
                )))
            }
            Some(existing) => Ok(existing),
            None => {
                self.layout = Some(layout);
                Ok(layout)
            }
        }
    }

    fn dequantize_keys(&self) -> Result<Array, Exception> {
        let layout = self
            .layout
            .ok_or_else(|| Exception::custom("TurboQuant key layout missing"))?;
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant runtime missing"))?;
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant key store missing"))?;

        let decoded = decode_key_rows_for_runtime(&runtime.keys, layout.key_dim, keys);

        let array = Array::from_slice(
            &decoded,
            &[
                layout.batch as i32,
                self.offset as i32,
                layout.heads as i32,
                layout.key_dim as i32,
            ],
        );
        array.transpose_axes(&[0, 2, 1, 3])?.as_dtype(self.dtype)
    }

    fn dequantize_values(&self) -> Result<Array, Exception> {
        let layout = self
            .layout
            .ok_or_else(|| Exception::custom("TurboQuant value layout missing"))?;
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant runtime missing"))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant value store missing"))?;

        let decoded = decode_value_rows_for_runtime(&runtime.values, layout.value_dim, values);

        let array = Array::from_slice(
            &decoded,
            &[
                layout.batch as i32,
                self.offset as i32,
                layout.heads as i32,
                layout.value_dim as i32,
            ],
        );
        array.transpose_axes(&[0, 2, 1, 3])?.as_dtype(self.dtype)
    }
}

fn direct_attention_output(
    queries: &Array,
    layout: TurboLayout,
    total_seq: usize,
    runtime: &TurboQuantRuntime,
    key_store: &TurboKeyStore,
    value_store: &TurboValueStore,
    config: TurboQuantConfig,
    attn_config: &FusedAttentionConfig,
) -> Result<Array, Exception> {
    let batch = queries.dim(0) as usize;
    let num_heads = queries.dim(1) as usize;
    let key_dim = queries.dim(3) as usize;
    let value_dim = layout.value_dim;
    let query_rows = array_rows_in_bshd_order(&queries.as_dtype(Dtype::Float32)?)?;
    let start_seq = attention_start_seq(total_seq, attn_config.mask_type);
    let num_groups = (attn_config.num_heads / attn_config.num_kv_heads).max(1) as usize;

    debug_assert_eq!(layout.batch, batch);
    debug_assert_eq!(layout.key_dim, key_dim);

    let mut row_cache = std::collections::HashMap::<(usize, usize), Vec<usize>>::new();
    let mut decoded_value_cache = std::collections::HashMap::<(usize, usize), Vec<f32>>::new();
    let mut output_rows = vec![0.0f32; batch * num_heads * value_dim];

    for batch_idx in 0..batch {
        for query_head in 0..num_heads {
            let kv_head = if layout.heads == num_heads {
                query_head
            } else {
                query_head / num_groups
            };
            let row_indices = row_cache
                .entry((batch_idx, kv_head))
                .or_insert_with(|| {
                    collect_attention_rows(batch_idx, kv_head, layout, total_seq, start_seq)
                })
                .clone();
            let query_row = &query_rows[(batch_idx * num_heads + query_head) * key_dim
                ..(batch_idx * num_heads + query_head + 1) * key_dim];

            let scores = direct_attention_scores_for_query(
                &runtime.keys,
                config.keys,
                key_dim,
                key_store,
                query_row,
                &row_indices,
            );
            let weights = scaled_softmax(&scores, attn_config.scale, attn_config.logit_softcapping);

            let decoded_values = decoded_value_cache
                .entry((batch_idx, kv_head))
                .or_insert_with(|| {
                    decode_selected_value_rows(
                        &runtime.values,
                        config.values,
                        value_dim,
                        value_store,
                        &row_indices,
                    )
                })
                .clone();

            let output_row = &mut output_rows[(batch_idx * num_heads + query_head) * value_dim
                ..(batch_idx * num_heads + query_head + 1) * value_dim];
            for (weight, value_row) in weights.iter().zip(decoded_values.chunks_exact(value_dim)) {
                for (dst, value) in output_row.iter_mut().zip(value_row.iter()) {
                    *dst += *weight * *value;
                }
            }
        }
    }

    let output = Array::from_slice(
        &output_rows,
        &[batch as i32, num_heads as i32, 1, value_dim as i32],
    );
    if queries.dtype() == Dtype::Float32 {
        Ok(output)
    } else {
        output.as_dtype(queries.dtype())
    }
}

fn attention_start_seq(total_seq: usize, mask_type: AttentionMaskType) -> usize {
    match mask_type {
        AttentionMaskType::SlidingWindow(window) => total_seq.saturating_sub(window as usize),
        AttentionMaskType::None | AttentionMaskType::Causal => 0,
    }
}

fn collect_attention_rows(
    batch_idx: usize,
    kv_head: usize,
    layout: TurboLayout,
    total_seq: usize,
    start_seq: usize,
) -> Vec<usize> {
    (start_seq..total_seq)
        .map(|seq_idx| ((batch_idx * total_seq) + seq_idx) * layout.heads + kv_head)
        .collect()
}

fn scaled_softmax(scores: &[f32], scale: f32, softcap: Option<f32>) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut scaled: Vec<f32> = scores.iter().map(|score| score * scale).collect();
    if let Some(cap) = softcap {
        for value in &mut scaled {
            *value = cap * (*value / cap).tanh();
        }
    }

    let max_score = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores: Vec<f32> = scaled
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect();
    let denom: f32 = exp_scores.iter().sum();
    if denom <= ZERO_EPSILON {
        let uniform = 1.0 / exp_scores.len() as f32;
        exp_scores.fill(uniform);
        return exp_scores;
    }
    for value in &mut exp_scores {
        *value /= denom;
    }
    exp_scores
}

fn direct_attention_scores_for_query(
    runtime: &TurboQuantTensorRuntime,
    config: TurboQuantTensorConfig,
    total_dim: usize,
    store: &TurboKeyStore,
    query_row: &[f32],
    row_indices: &[usize],
) -> Vec<f32> {
    match runtime {
        TurboQuantTensorRuntime::Uniform { core, .. } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            let q_rot = core.rotate_rows(query_row);
            let q_proj = core.project_rows(query_row);
            row_indices
                .iter()
                .map(|row| {
                    score_key_component_row(
                        core,
                        &store.regular_indices,
                        &store.regular_qjl_signs,
                        &store.regular_norms,
                        &store.regular_residual_norms,
                        bits,
                        *row,
                        &q_rot,
                        &q_proj,
                    )
                })
                .collect()
        }
        TurboQuantTensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!("mixed runtime must carry mixed config");
            };
            let regular_dim = total_dim - outlier_count;
            let masks = collect_mask_rows(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant key outlier mask missing"),
                row_indices,
                total_dim,
            );
            let (regular_queries, outlier_queries) =
                split_query_by_masks(query_row, total_dim, outlier_count, &masks);
            let regular_rot = regular_core.rotate_rows(&regular_queries);
            let regular_proj = regular_core.project_rows(&regular_queries);
            let outlier_rot = outlier_core.rotate_rows(&outlier_queries);
            let outlier_proj = outlier_core.project_rows(&outlier_queries);

            row_indices
                .iter()
                .enumerate()
                .map(|(local_idx, row)| {
                    let regular_slice =
                        &regular_rot[local_idx * regular_dim..(local_idx + 1) * regular_dim];
                    let regular_proj_slice =
                        &regular_proj[local_idx * regular_dim..(local_idx + 1) * regular_dim];
                    let outlier_slice =
                        &outlier_rot[local_idx * outlier_count..(local_idx + 1) * outlier_count];
                    let outlier_proj_slice =
                        &outlier_proj[local_idx * outlier_count..(local_idx + 1) * outlier_count];

                    score_key_component_row(
                        regular_core,
                        &store.regular_indices,
                        &store.regular_qjl_signs,
                        &store.regular_norms,
                        &store.regular_residual_norms,
                        regular_bits,
                        *row,
                        regular_slice,
                        regular_proj_slice,
                    ) + score_key_component_row(
                        outlier_core,
                        store
                            .outlier_indices
                            .as_ref()
                            .expect("TurboQuant key outlier indices missing"),
                        store
                            .outlier_qjl_signs
                            .as_ref()
                            .expect("TurboQuant key outlier QJL signs missing"),
                        store
                            .outlier_norms
                            .as_ref()
                            .expect("TurboQuant key outlier norms missing"),
                        store
                            .outlier_residual_norms
                            .as_ref()
                            .expect("TurboQuant key outlier residual norms missing"),
                        outlier_bits,
                        *row,
                        outlier_slice,
                        outlier_proj_slice,
                    )
                })
                .collect()
        }
    }
}

fn score_key_component_row(
    core: &TurboQuantCore,
    indices: &PackedBits,
    qjl_signs: &PackedBits,
    norms: &[f32],
    residual_norms: &[f32],
    key_bits: u8,
    row: usize,
    query_rot: &[f32],
    query_proj: &[f32],
) -> f32 {
    let norm = norms[row];
    if norm <= ZERO_EPSILON {
        return 0.0;
    }

    let mse_bits = key_bits.saturating_sub(1);
    let width = core.dim;
    let base = row * width;

    let mse_score = if mse_bits == 0 {
        0.0
    } else {
        let codebook = core.codebook(mse_bits);
        query_rot
            .iter()
            .enumerate()
            .map(|(idx, query)| *query * codebook[usize::from(indices.get(base + idx))])
            .sum::<f32>()
    };

    let residual_score = if residual_norms[row] > ZERO_EPSILON {
        let scale = ((PI / 2.0).sqrt() * residual_norms[row]) / (core.dim as f32);
        let qjl = query_proj
            .iter()
            .enumerate()
            .map(|(idx, query)| {
                let sign = if qjl_signs.get(base + idx) == 0 {
                    -1.0
                } else {
                    1.0
                };
                *query * sign
            })
            .sum::<f32>();
        scale * qjl
    } else {
        0.0
    };

    norm * (mse_score + residual_score)
}

fn decode_selected_value_rows(
    runtime: &TurboQuantTensorRuntime,
    config: TurboQuantTensorConfig,
    total_dim: usize,
    store: &TurboValueStore,
    row_indices: &[usize],
) -> Vec<f32> {
    match runtime {
        TurboQuantTensorRuntime::Uniform { core, .. } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            decode_value_component_rows_raw(
                core,
                &collect_packed_rows(&store.regular_indices, row_indices, total_dim),
                &collect_scalar_rows(&store.regular_norms, row_indices),
                bits,
            )
        }
        TurboQuantTensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!("mixed runtime must carry mixed config");
            };
            let regular_dim = total_dim - outlier_count;
            let regular = decode_value_component_rows_raw(
                regular_core,
                &collect_packed_rows(&store.regular_indices, row_indices, regular_dim),
                &collect_scalar_rows(&store.regular_norms, row_indices),
                regular_bits,
            );
            let outlier = decode_value_component_rows_raw(
                outlier_core,
                &collect_packed_rows(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant value outlier indices missing"),
                    row_indices,
                    outlier_count,
                ),
                &collect_scalar_rows(
                    store
                        .outlier_norms
                        .as_ref()
                        .expect("TurboQuant value outlier norms missing"),
                    row_indices,
                ),
                outlier_bits,
            );
            scatter_mixed_rows(
                &collect_mask_rows(
                    store
                        .outlier_mask
                        .as_ref()
                        .expect("TurboQuant value outlier mask missing"),
                    row_indices,
                    total_dim,
                ),
                total_dim,
                outlier_count,
                &regular,
                &outlier,
            )
        }
    }
}

fn collect_scalar_rows(values: &[f32], row_indices: &[usize]) -> Vec<f32> {
    row_indices.iter().map(|row| values[*row]).collect()
}

fn collect_mask_rows(mask: &PackedBits, row_indices: &[usize], width: usize) -> Vec<u16> {
    collect_packed_rows(mask, row_indices, width)
}

fn collect_packed_rows(bits: &PackedBits, row_indices: &[usize], width: usize) -> Vec<u16> {
    let mut values = Vec::with_capacity(row_indices.len() * width);
    for row in row_indices {
        let base = row * width;
        for offset in 0..width {
            values.push(bits.get(base + offset));
        }
    }
    values
}

fn split_query_by_masks(
    query_row: &[f32],
    total_dim: usize,
    outlier_count: usize,
    masks: &[u16],
) -> (Vec<f32>, Vec<f32>) {
    let num_rows = masks.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut regular_queries = Vec::with_capacity(num_rows * regular_dim);
    let mut outlier_queries = Vec::with_capacity(num_rows * outlier_count);

    for mask_row in masks.chunks_exact(total_dim) {
        for (value, is_outlier) in query_row.iter().zip(mask_row.iter()) {
            if *is_outlier == 1 {
                outlier_queries.push(*value);
            } else {
                regular_queries.push(*value);
            }
        }
    }

    (regular_queries, outlier_queries)
}

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

struct EncodedTurboKeyRows {
    regular: EncodedKeyRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedKeyRows>,
}

struct EncodedTurboValueRows {
    regular: EncodedValueRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedValueRows>,
}

fn encode_key_component_rows(core: &TurboQuantCore, rows: &[f32], key_bits: u8) -> EncodedKeyRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut residual_norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        norms[row_idx] = norm;
        if norm > ZERO_EPSILON {
            let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
            for (value, dst) in row.iter().zip(dst.iter_mut()) {
                *dst = *value / norm;
            }
        }
    }

    let mse_bits = key_bits.saturating_sub(1);
    let mut mse_indices = quantize_mse_rows(core, &normalized, mse_bits);
    let decoded_mse = if mse_bits == 0 {
        vec![0.0; rows.len()]
    } else {
        reconstruct_mse_rows(core, &mse_indices, mse_bits)
    };

    let mut residual = vec![0.0f32; rows.len()];
    for row_idx in 0..num_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        if norms[row_idx] <= ZERO_EPSILON {
            mse_indices[start..end].fill(0);
            continue;
        }
        let residual_row = &mut residual[start..end];
        for ((dst, lhs), rhs) in residual_row
            .iter_mut()
            .zip(normalized[start..end].iter())
            .zip(decoded_mse[start..end].iter())
        {
            *dst = lhs - rhs;
        }
        residual_norms[row_idx] = l2_norm(residual_row);
    }

    let projected = core.project_rows(&residual);
    let mut qjl_signs: Vec<u16> = projected
        .iter()
        .map(|value| if *value >= 0.0 { 1u16 } else { 0u16 })
        .collect();

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
        norms[row_idx] = norm;
        if norm > ZERO_EPSILON {
            let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
            for (value, dst) in row.iter().zip(dst.iter_mut()) {
                *dst = *value / norm;
            }
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

fn encode_key_rows_for_runtime(
    runtime: &TurboQuantTensorRuntime,
    total_dim: usize,
    rows: &[f32],
) -> EncodedTurboKeyRows {
    match runtime {
        TurboQuantTensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            EncodedTurboKeyRows {
                regular: encode_key_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TurboQuantTensorRuntime::Mixed {
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
                unreachable!("mixed runtime must carry mixed config");
            };
            let (outlier_mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            EncodedTurboKeyRows {
                regular: encode_key_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(outlier_mask),
                outlier: Some(encode_key_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

fn encode_value_rows_for_runtime(
    runtime: &TurboQuantTensorRuntime,
    total_dim: usize,
    rows: &[f32],
) -> EncodedTurboValueRows {
    match runtime {
        TurboQuantTensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            EncodedTurboValueRows {
                regular: encode_value_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TurboQuantTensorRuntime::Mixed {
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
                unreachable!("mixed runtime must carry mixed config");
            };
            let (outlier_mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            EncodedTurboValueRows {
                regular: encode_value_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(outlier_mask),
                outlier: Some(encode_value_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

fn decode_key_component_rows(
    core: &TurboQuantCore,
    indices: &PackedBits,
    qjl_signs: &PackedBits,
    norms: &[f32],
    residual_norms: &[f32],
    key_bits: u8,
) -> Vec<f32> {
    decode_key_component_rows_raw(
        core,
        &unpack_all(indices),
        &unpack_all(qjl_signs),
        norms,
        residual_norms,
        key_bits,
    )
}

fn decode_value_component_rows(
    core: &TurboQuantCore,
    indices: &PackedBits,
    norms: &[f32],
    value_bits: u8,
) -> Vec<f32> {
    decode_value_component_rows_raw(core, &unpack_all(indices), norms, value_bits)
}

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
    let mut reconstructed = if mse_bits == 0 {
        vec![0.0; total_rows * core.dim]
    } else {
        reconstruct_mse_rows(core, indices, mse_bits)
    };

    if residual_norms
        .iter()
        .any(|residual_norm| *residual_norm > ZERO_EPSILON)
    {
        let qjl_signs: Vec<f32> = qjl_signs
            .iter()
            .map(|value| if *value == 0 { -1.0 } else { 1.0 })
            .collect();
        let qjl = core.inverse_project_rows(&qjl_signs);
        for row_idx in 0..total_rows {
            let residual_norm = residual_norms[row_idx];
            if residual_norm <= ZERO_EPSILON {
                continue;
            }
            let scale = ((PI / 2.0).sqrt() * residual_norm) / (core.dim as f32);
            let start = row_idx * core.dim;
            let end = start + core.dim;
            for (value, correction) in reconstructed[start..end]
                .iter_mut()
                .zip(qjl[start..end].iter())
            {
                *value += scale * correction;
            }
        }
    }

    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
            continue;
        }
        for value in &mut reconstructed[start..end] {
            *value *= norm;
        }
    }

    reconstructed
}

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
            continue;
        }
        for value in &mut reconstructed[start..end] {
            *value *= norm;
        }
    }
    reconstructed
}

fn decode_key_rows_for_runtime(
    runtime: &TurboQuantTensorRuntime,
    total_dim: usize,
    store: &TurboKeyStore,
) -> Vec<f32> {
    match runtime {
        TurboQuantTensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            decode_key_component_rows(
                core,
                &store.regular_indices,
                &store.regular_qjl_signs,
                &store.regular_norms,
                &store.regular_residual_norms,
                *bits,
            )
        }
        TurboQuantTensorRuntime::Mixed {
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
                unreachable!("mixed runtime must carry mixed config");
            };
            let regular = decode_key_component_rows(
                regular_core,
                &store.regular_indices,
                &store.regular_qjl_signs,
                &store.regular_norms,
                &store.regular_residual_norms,
                *regular_bits,
            );
            let outlier = decode_key_component_rows(
                outlier_core,
                store
                    .outlier_indices
                    .as_ref()
                    .expect("TurboQuant key outlier indices missing"),
                store
                    .outlier_qjl_signs
                    .as_ref()
                    .expect("TurboQuant key outlier QJL signs missing"),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant key outlier norms missing"),
                store
                    .outlier_residual_norms
                    .as_ref()
                    .expect("TurboQuant key outlier residual norms missing"),
                *outlier_bits,
            );
            scatter_mixed_rows(
                &unpack_all(
                    store
                        .outlier_mask
                        .as_ref()
                        .expect("TurboQuant key outlier mask missing"),
                ),
                total_dim,
                *outlier_count,
                &regular,
                &outlier,
            )
        }
    }
}

fn decode_value_rows_for_runtime(
    runtime: &TurboQuantTensorRuntime,
    total_dim: usize,
    store: &TurboValueStore,
) -> Vec<f32> {
    match runtime {
        TurboQuantTensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!("uniform runtime must carry uniform config");
            };
            decode_value_component_rows(core, &store.regular_indices, &store.regular_norms, *bits)
        }
        TurboQuantTensorRuntime::Mixed {
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
                unreachable!("mixed runtime must carry mixed config");
            };
            let regular = decode_value_component_rows(
                regular_core,
                &store.regular_indices,
                &store.regular_norms,
                *regular_bits,
            );
            let outlier = decode_value_component_rows(
                outlier_core,
                store
                    .outlier_indices
                    .as_ref()
                    .expect("TurboQuant value outlier indices missing"),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant value outlier norms missing"),
                *outlier_bits,
            );
            scatter_mixed_rows(
                &unpack_all(
                    store
                        .outlier_mask
                        .as_ref()
                        .expect("TurboQuant value outlier mask missing"),
                ),
                total_dim,
                *outlier_count,
                &regular,
                &outlier,
            )
        }
    }
}

fn split_rows_by_outliers(
    rows: &[f32],
    total_dim: usize,
    outlier_count: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let num_rows = rows.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut outlier_mask = Vec::with_capacity(rows.len());
    let mut regular_rows = Vec::with_capacity(num_rows * regular_dim);
    let mut outlier_rows = Vec::with_capacity(num_rows * outlier_count);

    for row in rows.chunks(total_dim) {
        let mask = select_outlier_mask(row, outlier_count);
        for (value, is_outlier) in row.iter().zip(mask.iter()) {
            if *is_outlier == 1 {
                outlier_rows.push(*value);
            } else {
                regular_rows.push(*value);
            }
        }
        outlier_mask.extend(mask);
    }

    (outlier_mask, regular_rows, outlier_rows)
}

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
        let mut regular_cursor = 0usize;
        let mut outlier_cursor = 0usize;
        for dim_idx in 0..total_dim {
            let dst = &mut merged[row_idx * total_dim + dim_idx];
            if mask_row[dim_idx] == 1 {
                *dst = outlier_rows[row_idx * outlier_count + outlier_cursor];
                outlier_cursor += 1;
            } else {
                *dst = regular_rows[row_idx * regular_dim + regular_cursor];
                regular_cursor += 1;
            }
        }
    }

    merged
}

fn select_outlier_mask(row: &[f32], outlier_count: usize) -> Vec<u16> {
    let mut ranked_dims: Vec<usize> = (0..row.len()).collect();
    ranked_dims.sort_unstable_by(|lhs, rhs| {
        row[*rhs]
            .abs()
            .total_cmp(&row[*lhs].abs())
            .then_with(|| lhs.cmp(rhs))
    });

    let mut mask = vec![0u16; row.len()];
    for dim_idx in ranked_dims.into_iter().take(outlier_count) {
        mask[dim_idx] = 1;
    }
    mask
}

fn array_rows_in_bshd_order(array: &Array) -> Result<Vec<f32>, Exception> {
    let seq_major = array.as_type::<f32>()?.transpose_axes(&[0, 2, 1, 3])?;
    seq_major.eval()?;
    Ok(seq_major.as_slice::<f32>().to_vec())
}

fn quantize_mse_rows(core: &TurboQuantCore, normalized: &[f32], bits: u8) -> Vec<u16> {
    if bits == 0 {
        return vec![0; normalized.len()];
    }
    let rotated = core.rotate_rows(normalized);
    rotated
        .iter()
        .map(|value| nearest_centroid_index(*value, core.codebook(bits)) as u16)
        .collect()
}

fn reconstruct_mse_rows(core: &TurboQuantCore, indices: &[u16], bits: u8) -> Vec<f32> {
    if bits == 0 {
        return vec![0.0; indices.len()];
    }
    let codebook = core.codebook(bits);
    let rotated: Vec<f32> = indices
        .iter()
        .map(|index| codebook[usize::from(*index)])
        .collect();
    core.inverse_rotate_rows(&rotated)
}

fn unpack_all(bits: &PackedBits) -> Vec<u16> {
    (0..bits.len()).map(|index| bits.get(index)).collect()
}

fn nearest_centroid_index(value: f32, codebook: &[f32]) -> usize {
    match codebook.binary_search_by(|probe| probe.partial_cmp(&value).unwrap()) {
        Ok(index) => index,
        Err(0) => 0,
        Err(index) if index >= codebook.len() => codebook.len() - 1,
        Err(index) => {
            let left = codebook[index - 1];
            let right = codebook[index];
            if (value - left).abs() <= (right - value).abs() {
                index - 1
            } else {
                index
            }
        }
    }
}

fn build_beta_codebook(dim: usize, bits: u8) -> Vec<f32> {
    let centroid_count = 1usize << bits;
    let mut xs = Vec::with_capacity(LLOYD_GRID_POINTS);
    let mut weights = Vec::with_capacity(LLOYD_GRID_POINTS);
    let alpha = ((dim as f64) - 3.0) / 2.0;
    let step = 2.0 / (LLOYD_GRID_POINTS as f64);

    for idx in 0..LLOYD_GRID_POINTS {
        let x = -1.0 + ((idx as f64) + 0.5) * step;
        let weight = if dim <= 2 {
            1.0
        } else {
            (1.0 - x * x).max(0.0).powf(alpha)
        };
        xs.push(x);
        weights.push(weight);
    }

    let mut cumulative = Vec::with_capacity(LLOYD_GRID_POINTS);
    let mut total_weight = 0.0;
    for weight in &weights {
        total_weight += *weight;
        cumulative.push(total_weight);
    }

    let mut centroids = Vec::with_capacity(centroid_count);
    for bucket in 0..centroid_count {
        let target = ((bucket as f64) + 0.5) * total_weight / (centroid_count as f64);
        let index = cumulative.partition_point(|value| *value < target);
        centroids.push(xs[index.min(xs.len() - 1)]);
    }
    centroids.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());

    for _ in 0..LLOYD_MAX_ITERS {
        let mut boundaries = Vec::with_capacity(centroid_count + 1);
        boundaries.push(-1.0f64);
        for pair in centroids.windows(2) {
            boundaries.push((pair[0] + pair[1]) * 0.5);
        }
        boundaries.push(1.0f64);

        let mut updated = centroids.clone();
        let mut max_change = 0.0f64;
        for bucket in 0..centroid_count {
            let left = boundaries[bucket];
            let right = boundaries[bucket + 1];
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            for (&x, &weight) in xs.iter().zip(weights.iter()) {
                if x >= left && x < right {
                    weighted_sum += x * weight;
                    weight_sum += weight;
                }
            }
            if weight_sum > 0.0 {
                updated[bucket] = weighted_sum / weight_sum;
            } else {
                updated[bucket] = (left + right) * 0.5;
            }
            max_change = max_change.max((updated[bucket] - centroids[bucket]).abs());
        }
        centroids = updated;
        if max_change < LLOYD_MAX_TOLERANCE {
            break;
        }
    }

    centroids.into_iter().map(|value| value as f32).collect()
}

fn generate_random_projection(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut projection = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        projection.push(sample_standard_normal(rng));
    }
    projection
}

fn generate_random_orthogonal(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut q = vec![0.0f64; dim * dim];
    for column in 0..dim {
        let mut candidate = vec![0.0f64; dim];
        loop {
            for value in &mut candidate {
                *value = f64::from(sample_standard_normal(rng));
            }

            for prev_column in 0..column {
                let prev = &q[prev_column * dim..(prev_column + 1) * dim];
                let dot = dot_f64(&candidate, prev);
                for (value, prev_value) in candidate.iter_mut().zip(prev.iter()) {
                    *value -= dot * *prev_value;
                }
            }

            let norm = dot_f64(&candidate, &candidate).sqrt();
            if norm > 1e-8 {
                for (row, value) in candidate.iter().enumerate() {
                    q[column * dim + row] = *value / norm;
                }
                break;
            }
        }
    }

    let mut row_major = vec![0.0f32; dim * dim];
    for row in 0..dim {
        for column in 0..dim {
            row_major[row * dim + column] = q[column * dim + row] as f32;
        }
    }
    row_major
}

fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

fn dot_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn transpose_square_matrix(matrix: &[f32], dim: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; matrix.len()];
    for row in 0..dim {
        for column in 0..dim {
            transposed[column * dim + row] = matrix[row * dim + column];
        }
    }
    transposed
}

fn matmul_rows(matrix: &[f32], dim: usize, rows: &[f32]) -> Vec<f32> {
    let num_rows = rows.len() / dim;
    let mut output = vec![0.0f32; rows.len()];
    for row_idx in 0..num_rows {
        let src = &rows[row_idx * dim..(row_idx + 1) * dim];
        let dst = &mut output[row_idx * dim..(row_idx + 1) * dim];
        for out_dim in 0..dim {
            let matrix_row = &matrix[out_dim * dim..(out_dim + 1) * dim];
            let mut acc = 0.0f32;
            for (weight, value) in matrix_row.iter().zip(src.iter()) {
                acc += weight * value;
            }
            dst[out_dim] = acc;
        }
    }
    output
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// Convenience constructor for a TurboQuant KV cache.
pub fn create_turboquant_cache(key_bits: u8, value_bits: u8) -> TurboQuantKvCache {
    TurboQuantKvCache::new(key_bits, value_bits)
}

pub(crate) fn create_turboquant_runtime(
    key_dim: usize,
    value_dim: usize,
    config: TurboQuantConfig,
) -> Arc<TurboQuantRuntime> {
    Arc::new(TurboQuantRuntime::new(key_dim, value_dim, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bits_round_trip() {
        let values = [1u16, 6, 3, 0, 7, 2, 4];
        let mut packed = PackedBits::from_values(3, &values);
        let round_trip: Vec<u16> = (0..values.len()).map(|index| packed.get(index)).collect();
        assert_eq!(round_trip, values);

        packed.truncate(4);
        let truncated: Vec<u16> = (0..4).map(|index| packed.get(index)).collect();
        assert_eq!(truncated, values[..4]);
    }

    #[test]
    fn turboquant_handles_zero_rows() {
        let core = TurboQuantCore::new(8, 4);
        let encoded = encode_key_component_rows(&core, &[0.0; 8], 4);
        assert_eq!(encoded.norms, vec![0.0]);
        assert_eq!(encoded.residual_norms, vec![0.0]);
        assert!(encoded.mse_indices.iter().all(|value| *value == 0));
        assert!(encoded.qjl_signs.iter().all(|value| *value == 0));
    }

    #[test]
    fn mixed_tensor_config_reports_effective_bits() {
        let config = TurboQuantTensorConfig::mixed(2, 4, 32);
        assert_eq!(config.effective_bits(128), 2.5);
        assert_eq!(config.regular_dim(128), 96);
        assert_eq!(config.outlier_count(), 32);
    }

    #[test]
    fn turboquant_presets_match_outlier_schedule() {
        let q2_5 = TurboQuantConfig::preset_q2_5(128);
        let q3_5 = TurboQuantConfig::preset_q3_5(128);

        assert_eq!(q2_5, TurboQuantConfig::mixed(2, 4, 32, 2, 4, 32));
        assert_eq!(q3_5, TurboQuantConfig::mixed(3, 5, 32, 3, 5, 32));
    }

    #[test]
    fn beta_codebook_is_sorted() {
        let codebook = build_beta_codebook(128, 4);
        assert_eq!(codebook.len(), 16);
        assert!(codebook.windows(2).all(|window| window[0] <= window[1]));
    }
}

//! TurboQuant KV cache.
//!
//! This implements a practical TurboQuant-inspired KV cache for MLX tensors:
//! - vectors are normalized onto the unit sphere and their norms are stored
//! - keys use the paper's two-stage inner-product quantizer
//! - values use the MSE-optimized scalar codebook path
//!
//! The current implementation focuses on correctness and integration. It keeps
//! the paper's data flow and storage layout while leaving the rotation/QJL
//! matvecs on the host. This makes the cache usable end-to-end today and gives
//! us a stable API to accelerate with Metal kernels next.

use std::{f32::consts::PI, sync::Arc};

use mlx_rs::{
    Array, Dtype,
    error::Exception,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Deterministic seed used for TurboQuant rotations and QJL projections.
const TURBOQUANT_SEED: u64 = 0x5442_5155_414e_544d;
const ZERO_EPSILON: f32 = 1e-12;
const LLOYD_MAX_ITERS: usize = 64;
const LLOYD_MAX_TOLERANCE: f64 = 1e-7;
const LLOYD_GRID_POINTS: usize = 8192;

#[derive(Debug, Clone, Copy)]
struct TurboLayout {
    batch: usize,
    heads: usize,
    dim: usize,
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
}

#[derive(Debug, Clone)]
struct TurboValueStore {
    indices: PackedBits,
    norms: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TurboKeyStore {
    mse_indices: PackedBits,
    qjl_signs: PackedBits,
    norms: Vec<f32>,
    residual_norms: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct TurboQuantCore {
    dim: usize,
    rotation: Vec<f32>,
    qjl_projection: Vec<f32>,
    codebooks: Vec<Vec<f32>>,
}

impl TurboQuantCore {
    fn new(dim: usize, max_mse_bits: u8) -> Self {
        let mut rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ ((dim as u64) << 32));
        let rotation = generate_random_orthogonal(dim, &mut rng);
        let qjl_projection = generate_random_projection(dim, &mut rng);

        let mut codebooks = vec![Vec::new(); usize::from(max_mse_bits) + 1];
        for bits in 1..=max_mse_bits {
            codebooks[usize::from(bits)] = build_beta_codebook(dim, bits);
        }

        Self {
            dim,
            rotation,
            qjl_projection,
            codebooks,
        }
    }

    fn codebook(&self, bits: u8) -> &[f32] {
        &self.codebooks[usize::from(bits)]
    }

    fn rotate(&self, input: &[f32]) -> Vec<f32> {
        matvec(&self.rotation, self.dim, input)
    }

    fn inverse_rotate(&self, input: &[f32]) -> Vec<f32> {
        matvec_transposed(&self.rotation, self.dim, input)
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
    key_bits: u8,
    value_bits: u8,
    dtype: Dtype,
    core: Option<Arc<TurboQuantCore>>,
}

impl TurboQuantKvCache {
    /// Create a new TurboQuant KV cache.
    ///
    /// `key_bits` and `value_bits` are the total effective bits per channel.
    /// Keys reserve one of those bits for the QJL residual stage.
    pub fn new(key_bits: u8, value_bits: u8) -> Self {
        assert!(
            (1..=8).contains(&key_bits),
            "TurboQuant key_bits must be in 1..=8"
        );
        assert!(
            (1..=8).contains(&value_bits),
            "TurboQuant value_bits must be in 1..=8"
        );

        Self {
            keys: None,
            values: None,
            layout: None,
            offset: 0,
            key_bits,
            value_bits,
            dtype: Dtype::Float16,
            core: None,
        }
    }

    pub(crate) fn new_with_core(
        key_bits: u8,
        value_bits: u8,
        core: Arc<TurboQuantCore>,
    ) -> Self {
        let mut cache = Self::new(key_bits, value_bits);
        cache.core = Some(core);
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

    /// Append a new `[B, H, S, D]` KV chunk and return the dequantized cache.
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.dtype = keys.dtype();
        let layout = self.ensure_layout(keys, values)?;
        let rows_per_seq = layout.batch * layout.heads;
        let seq_len = keys.dim(2) as usize;

        let key_rows = array_rows_in_bshd_order(keys)?;
        let value_rows = array_rows_in_bshd_order(values)?;

        let max_mse_bits = self.max_mse_bits();
        let core = self
            .core
            .get_or_insert_with(|| Arc::new(TurboQuantCore::new(layout.dim, max_mse_bits)));

        let mut encoded_keys = Vec::with_capacity(key_rows.len());
        let mut encoded_values = Vec::with_capacity(value_rows.len());
        for row in key_rows.chunks(layout.dim) {
            encoded_keys.push(encode_key_row(core, row, self.key_bits));
        }
        for row in value_rows.chunks(layout.dim) {
            encoded_values.push(encode_value_row(core, row, self.value_bits));
        }

        let key_store = self
            .keys
            .get_or_insert_with(|| TurboKeyStore {
                mse_indices: PackedBits::new(self.key_bits.saturating_sub(1)),
                qjl_signs: PackedBits::new(1),
                norms: Vec::new(),
                residual_norms: Vec::new(),
            });
        for encoded in &encoded_keys {
            key_store.mse_indices.extend_from_slice(&encoded.mse_indices);
            key_store.qjl_signs.extend_from_slice(&encoded.qjl_signs);
            key_store.norms.push(encoded.norm);
            key_store.residual_norms.push(encoded.residual_norm);
        }

        let value_store = self
            .values
            .get_or_insert_with(|| TurboValueStore {
                indices: PackedBits::new(self.value_bits),
                norms: Vec::new(),
            });
        for encoded in &encoded_values {
            value_store.indices.extend_from_slice(&encoded.indices);
            value_store.norms.push(encoded.norm);
        }

        self.offset += seq_len;
        debug_assert_eq!(
            key_store.norms.len(),
            self.offset * rows_per_seq,
            "TurboQuant key store row count drifted"
        );

        Ok((self.dequantize_keys()?, self.dequantize_values()?))
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
            keys.mse_indices.truncate(keep_rows * layout.dim);
            keys.qjl_signs.truncate(keep_rows * layout.dim);
            keys.norms.truncate(keep_rows);
            keys.residual_norms.truncate(keep_rows);
        }
        if let Some(values) = &mut self.values {
            values.indices.truncate(keep_rows * layout.dim);
            values.norms.truncate(keep_rows);
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
        let key_bytes = self.keys.as_ref().map_or(0, |keys| {
            keys.mse_indices.byte_len()
                + keys.qjl_signs.byte_len()
                + keys.norms.len() * std::mem::size_of::<f32>()
                + keys.residual_norms.len() * std::mem::size_of::<f32>()
        });
        let value_bytes = self.values.as_ref().map_or(0, |values| {
            values.indices.byte_len() + values.norms.len() * std::mem::size_of::<f32>()
        });
        key_bytes + value_bytes
    }

    fn max_mse_bits(&self) -> u8 {
        self.value_bits.max(self.key_bits.saturating_sub(1))
    }

    fn ensure_layout(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<TurboLayout, Exception> {
        if keys.shape() != values.shape() {
            return Err(Exception::custom(format!(
                "TurboQuant KV cache requires matching K/V shapes, got {:?} vs {:?}",
                keys.shape(),
                values.shape()
            )));
        }
        if keys.shape().len() != 4 {
            return Err(Exception::custom(format!(
                "TurboQuant KV cache expects [B, H, S, D], got {:?}",
                keys.shape()
            )));
        }

        let layout = TurboLayout {
            batch: keys.dim(0) as usize,
            heads: keys.dim(1) as usize,
            dim: keys.dim(3) as usize,
        };

        if self
            .core
            .as_ref()
            .is_some_and(|core| core.dim != layout.dim)
            && self.offset == 0
        {
            self.core = Some(Arc::new(TurboQuantCore::new(
                layout.dim,
                self.max_mse_bits(),
            )));
        }

        match self.layout {
            Some(existing)
                if existing.batch != layout.batch
                    || existing.heads != layout.heads
                    || existing.dim != layout.dim =>
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
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant core missing"))?;
        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant key store missing"))?;

        let total_rows = self.offset * layout.batch * layout.heads;
        let mut decoded = vec![0f32; total_rows * layout.dim];
        for row in 0..total_rows {
            let row_out = &mut decoded[row * layout.dim..(row + 1) * layout.dim];
            decode_key_row(core, keys, row, self.key_bits, row_out);
        }

        let array = Array::from_slice(
            &decoded,
            &[
                layout.batch as i32,
                self.offset as i32,
                layout.heads as i32,
                layout.dim as i32,
            ],
        );
        array.transpose_axes(&[0, 2, 1, 3])?.as_dtype(self.dtype)
    }

    fn dequantize_values(&self) -> Result<Array, Exception> {
        let layout = self
            .layout
            .ok_or_else(|| Exception::custom("TurboQuant value layout missing"))?;
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant core missing"))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("TurboQuant value store missing"))?;

        let total_rows = self.offset * layout.batch * layout.heads;
        let mut decoded = vec![0f32; total_rows * layout.dim];
        for row in 0..total_rows {
            let row_out = &mut decoded[row * layout.dim..(row + 1) * layout.dim];
            decode_value_row(core, values, row, self.value_bits, row_out);
        }

        let array = Array::from_slice(
            &decoded,
            &[
                layout.batch as i32,
                self.offset as i32,
                layout.heads as i32,
                layout.dim as i32,
            ],
        );
        array.transpose_axes(&[0, 2, 1, 3])?.as_dtype(self.dtype)
    }
}

struct EncodedKeyRow {
    mse_indices: Vec<u16>,
    qjl_signs: Vec<u16>,
    norm: f32,
    residual_norm: f32,
}

struct EncodedValueRow {
    indices: Vec<u16>,
    norm: f32,
}

fn encode_key_row(core: &TurboQuantCore, row: &[f32], key_bits: u8) -> EncodedKeyRow {
    let norm = l2_norm(row);
    if norm <= ZERO_EPSILON {
        return EncodedKeyRow {
            mse_indices: vec![0; core.dim],
            qjl_signs: vec![0; core.dim],
            norm: 0.0,
            residual_norm: 0.0,
        };
    }

    let normalized: Vec<f32> = row.iter().map(|value| *value / norm).collect();
    let mse_bits = key_bits.saturating_sub(1);
    let mse_indices = quantize_mse(core, &normalized, mse_bits);
    let decoded_mse = if mse_bits == 0 {
        vec![0.0; core.dim]
    } else {
        reconstruct_mse(core, &mse_indices, mse_bits)
    };

    let residual: Vec<f32> = normalized
        .iter()
        .zip(decoded_mse.iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect();
    let residual_norm = l2_norm(&residual);
    let qjl_projection = matvec(&core.qjl_projection, core.dim, &residual);
    let qjl_signs = qjl_projection
        .iter()
        .map(|value| if *value >= 0.0 { 1u16 } else { 0u16 })
        .collect();

    EncodedKeyRow {
        mse_indices,
        qjl_signs,
        norm,
        residual_norm,
    }
}

fn encode_value_row(core: &TurboQuantCore, row: &[f32], value_bits: u8) -> EncodedValueRow {
    let norm = l2_norm(row);
    if norm <= ZERO_EPSILON {
        return EncodedValueRow {
            indices: vec![0; core.dim],
            norm: 0.0,
        };
    }

    let normalized: Vec<f32> = row.iter().map(|value| *value / norm).collect();
    EncodedValueRow {
        indices: quantize_mse(core, &normalized, value_bits),
        norm,
    }
}

fn decode_key_row(
    core: &TurboQuantCore,
    store: &TurboKeyStore,
    row: usize,
    key_bits: u8,
    out: &mut [f32],
) {
    let norm = store.norms[row];
    if norm <= ZERO_EPSILON {
        out.fill(0.0);
        return;
    }

    let mse_bits = key_bits.saturating_sub(1);
    let mut reconstructed = if mse_bits == 0 {
        vec![0.0; core.dim]
    } else {
        let indices = unpack_row(&store.mse_indices, row, core.dim);
        reconstruct_mse(core, &indices, mse_bits)
    };

    let residual_norm = store.residual_norms[row];
    if residual_norm > ZERO_EPSILON {
        let qjl_row = unpack_row(&store.qjl_signs, row, core.dim);
        let qjl_signs: Vec<f32> = qjl_row
            .iter()
            .map(|value| if *value == 0 { -1.0 } else { 1.0 })
            .collect();
        let qjl = matvec_transposed(&core.qjl_projection, core.dim, &qjl_signs);
        let scale = ((PI / 2.0).sqrt() * residual_norm) / (core.dim as f32);
        for (value, correction) in reconstructed.iter_mut().zip(qjl.iter()) {
            *value += scale * correction;
        }
    }

    for (dst, value) in out.iter_mut().zip(reconstructed.iter()) {
        *dst = norm * *value;
    }
}

fn decode_value_row(
    core: &TurboQuantCore,
    store: &TurboValueStore,
    row: usize,
    value_bits: u8,
    out: &mut [f32],
) {
    let norm = store.norms[row];
    if norm <= ZERO_EPSILON {
        out.fill(0.0);
        return;
    }

    let indices = unpack_row(&store.indices, row, core.dim);
    let reconstructed = reconstruct_mse(core, &indices, value_bits);
    for (dst, value) in out.iter_mut().zip(reconstructed.iter()) {
        *dst = norm * *value;
    }
}

fn array_rows_in_bshd_order(array: &Array) -> Result<Vec<f32>, Exception> {
    let seq_major = array.as_type::<f32>()?.transpose_axes(&[0, 2, 1, 3])?;
    seq_major.eval()?;
    Ok(seq_major.as_slice::<f32>().to_vec())
}

fn quantize_mse(core: &TurboQuantCore, normalized: &[f32], bits: u8) -> Vec<u16> {
    if bits == 0 {
        return vec![0; core.dim];
    }
    let rotated = core.rotate(normalized);
    rotated
        .iter()
        .map(|value| nearest_centroid_index(*value, core.codebook(bits)) as u16)
        .collect()
}

fn reconstruct_mse(core: &TurboQuantCore, indices: &[u16], bits: u8) -> Vec<f32> {
    if bits == 0 {
        return vec![0.0; core.dim];
    }
    let codebook = core.codebook(bits);
    let rotated: Vec<f32> = indices
        .iter()
        .map(|index| codebook[usize::from(*index)])
        .collect();
    core.inverse_rotate(&rotated)
}

fn unpack_row(bits: &PackedBits, row: usize, width: usize) -> Vec<u16> {
    let base = row * width;
    (0..width).map(|offset| bits.get(base + offset)).collect()
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

fn matvec(matrix: &[f32], dim: usize, vector: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; dim];
    for row in 0..dim {
        let mut acc = 0.0f32;
        let matrix_row = &matrix[row * dim..(row + 1) * dim];
        for (weight, value) in matrix_row.iter().zip(vector.iter()) {
            acc += weight * value;
        }
        output[row] = acc;
    }
    output
}

fn matvec_transposed(matrix: &[f32], dim: usize, vector: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; dim];
    for row in 0..dim {
        let value = vector[row];
        let matrix_row = &matrix[row * dim..(row + 1) * dim];
        for column in 0..dim {
            output[column] += matrix_row[column] * value;
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

pub(crate) fn create_turboquant_core(dim: usize, key_bits: u8, value_bits: u8) -> Arc<TurboQuantCore> {
    Arc::new(TurboQuantCore::new(
        dim,
        value_bits.max(key_bits.saturating_sub(1)),
    ))
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
        let encoded = encode_key_row(&core, &[0.0; 8], 4);
        assert_eq!(encoded.norm, 0.0);
        assert_eq!(encoded.residual_norm, 0.0);
    }

    #[test]
    fn beta_codebook_is_sorted() {
        let codebook = build_beta_codebook(128, 4);
        assert_eq!(codebook.len(), 16);
        assert!(codebook.windows(2).all(|window| window[0] <= window[1]));
    }
}

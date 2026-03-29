//! Shared utilities for model architecture implementations.

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

/// Create a causal attention mask of shape [seq_len, seq_len].
///
/// Returns an additive mask where masked (future) positions hold `-inf`
/// and valid (past/current) positions hold `0.0`, matching the convention
/// expected by all attention kernels in this crate.
///
/// # Arguments
/// * `seq_len` - Sequence length (mask will be square)
///
/// # Returns
/// Float32 array of shape [seq_len, seq_len]
pub fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    // Lower-triangular matrix: 1.0 where position is valid (past or current)
    let lower_tri = ops::tri(seq_len, seq_len, 0, Dtype::Float32);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    // Where lower_tri == 0 (future positions), set -inf; otherwise 0.0
    let mask = lower_tri.equal(&zero);
    Ok(ops::where_fn(&mask, &neg_inf, &zero))
}

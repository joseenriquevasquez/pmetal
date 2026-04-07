//! Cross-entropy loss computation with optimizations for large vocabularies.
//!
//! This module provides efficient cross-entropy loss computation with:
//! - Chunked computation for large vocabularies (>65536 tokens)
//! - Logit softcapping for Gemma2 models
//! - Logit scaling for Cohere models
//! - Proper ignore_index handling with masking

use crate::ArrayDtypeExt;
use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

/// Maximum vocabulary size for single-pass computation.
/// For larger vocabularies, we use chunked computation.
const MAX_FUSED_SIZE: i32 = 65536;

/// Configuration for cross-entropy loss computation.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyConfig {
    /// Index to ignore in loss computation (e.g., -100 for padding).
    pub ignore_index: Option<i64>,
    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,
    /// Logit softcapping value for Gemma2 models (0.0 to disable).
    /// Applies: logits = softcap * tanh(logits / softcap)
    pub logit_softcapping: f32,
    /// Logit scaling value for Cohere models (0.0 or 1.0 to disable).
    /// Applies: logits = scale * logits
    pub logit_scaling: f32,
}

impl CrossEntropyConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ignore index.
    pub fn with_ignore_index(mut self, index: i64) -> Self {
        self.ignore_index = Some(index);
        self
    }

    /// Set label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Set logit softcapping for Gemma2.
    pub fn with_softcapping(mut self, softcap: f32) -> Self {
        self.logit_softcapping = softcap;
        self
    }

    /// Set logit scaling for Cohere.
    pub fn with_scaling(mut self, scale: f32) -> Self {
        self.logit_scaling = scale;
        self
    }
}

/// Apply logit transformations (softcapping and/or scaling).
fn apply_logit_transforms(logits: &Array, softcapping: f32, scaling: f32) -> Array {
    let mut result = logits.clone();

    // Apply scaling first (Cohere): logits = scale * logits
    if scaling != 0.0 && scaling != 1.0 {
        let scale_arr = Array::from_f32(scaling);
        result = result.multiply(&scale_arr);
    }

    // Apply softcapping (Gemma2): logits = softcap * tanh(logits / softcap)
    if softcapping != 0.0 {
        let softcap_arr = Array::from_f32(softcapping);
        let scaled = result.divide(&softcap_arr);
        let tanh_scaled = ops::tanh(&scaled);
        result = tanh_scaled.multiply(&softcap_arr);
    }

    result
}

/// Compute stable logsumexp along last axis: max(x) + log(sum(exp(x - max(x))))
fn stable_logsumexp(logits: &Array) -> Array {
    // Get max along last axis, keeping dims for broadcasting
    let max_logits = logits.max_axis(-1, true);
    let shifted = logits.subtract(&max_logits);
    let exp_shifted = shifted.exp();
    let sum_exp = exp_shifted.sum_axis(-1, true);
    let log_sum = sum_exp.log();
    max_logits.add(&log_sum)
}

/// Compute chunked logsumexp for large vocabularies.
///
/// For vocab > 65536, splits into chunks and uses the identity:
/// logsumexp([a, b, c, ...]) = logsumexp([logsumexp(chunk_a), logsumexp(chunk_b), ...])
fn chunked_logsumexp(logits: &Array, vocab_size: i32) -> Array {
    if vocab_size <= MAX_FUSED_SIZE {
        return stable_logsumexp(logits);
    }

    let n_rows = logits.dim(0);

    // Number of chunks needed
    let n_chunks = (vocab_size + MAX_FUSED_SIZE - 1) / MAX_FUSED_SIZE;

    // Compute logsumexp for each chunk
    let mut chunk_logsumexps = Vec::with_capacity(n_chunks as usize);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * MAX_FUSED_SIZE;
        let end = ((chunk_idx + 1) * MAX_FUSED_SIZE).min(vocab_size);
        let chunk = logits.slice(&[0, start], &[n_rows, end]);
        let chunk_lse = stable_logsumexp(&chunk);
        chunk_logsumexps.push(chunk_lse);
    }

    // Concatenate chunk logsumexps along last axis
    let refs: Vec<&Array> = chunk_logsumexps.iter().collect();
    let stacked = ops::concatenate_axis(&refs, -1);

    // Final logsumexp over the chunk results
    stable_logsumexp(&stacked)
}

/// Gather target logits from logits tensor.
fn gather_target_logits(logits: &Array, targets: &Array) -> Array {
    // For each row, get logits[targets[row]]
    // Shape: logits [n, vocab], targets [n] -> result [n]
    let targets_expanded = targets.reshape(&[-1, 1]);
    // take_along_axis requires matching dtype for indices (int32)
    let targets_i32 = targets_expanded.as_dtype(Dtype::Int32.as_i32());
    let gathered = logits.take_along_axis(&targets_i32, -1);
    gathered.squeeze(-1)
}

/// Compute cross-entropy loss with chunking support for large vocabularies.
///
/// CE(x, y) = logsumexp(x) - x[y]
///
/// For ignored indices (e.g., -100), the loss is set to 0.
pub fn fast_cross_entropy_loss(
    logits: &Array,
    targets: &Array,
    config: &CrossEntropyConfig,
) -> Result<Array, Exception> {
    let shape = logits.shape();
    let ndim = shape.len();

    // Handle 3D input [batch, seq, vocab] -> [batch*seq, vocab]
    let (flat_logits, flat_targets) = if ndim == 3 {
        let batch = shape[0];
        let seq_len = shape[1];
        let vocab = shape[2];
        let reshaped_logits = logits.reshape(&[batch * seq_len, vocab]);
        let reshaped_targets = targets.reshape(&[-1]);
        (reshaped_logits, reshaped_targets)
    } else if ndim == 2 {
        (logits.clone(), targets.reshape(&[-1]))
    } else {
        return Err(Exception::custom(format!(
            "Expected 2D or 3D logits, got {}D",
            ndim
        )));
    };

    let vocab_size = flat_logits.dim(-1);
    let n_tokens = flat_logits.dim(0);

    // Apply logit transformations if configured
    let transformed_logits = if config.logit_softcapping != 0.0 || config.logit_scaling != 0.0 {
        apply_logit_transforms(&flat_logits, config.logit_softcapping, config.logit_scaling)
    } else {
        flat_logits.clone()
    };

    // Compute logsumexp (chunked if vocab is large)
    let logsumexp = chunked_logsumexp(&transformed_logits, vocab_size);

    // Get target logits (with transformation if needed)
    let target_logits = gather_target_logits(&transformed_logits, &flat_targets);

    // CE = logsumexp - target_logit
    let logsumexp_squeezed = logsumexp.squeeze(-1);
    let per_token_loss = logsumexp_squeezed.subtract(&target_logits);

    // Handle ignore_index masking
    let targets_i32 = flat_targets.as_dtype(Dtype::Int32.as_i32());
    let masked_loss = if let Some(ignore_idx) = config.ignore_index {
        let ignore_arr = Array::from_i32(ignore_idx as i32);
        let valid_mask = targets_i32.not_equal(&ignore_arr);
        let valid_mask_f32 = valid_mask.as_dtype(Dtype::Float32.as_i32());
        per_token_loss.multiply(&valid_mask_f32)
    } else {
        per_token_loss
    };

    // Count valid tokens for mean computation
    let n_valid = if let Some(ignore_idx) = config.ignore_index {
        let ignore_arr = Array::from_i32(ignore_idx as i32);
        let valid_mask = targets_i32.not_equal(&ignore_arr);
        let valid_mask_f32 = valid_mask.as_dtype(Dtype::Float32.as_i32());
        valid_mask_f32.sum_all()
    } else {
        Array::from_i32(n_tokens).as_dtype(Dtype::Float32.as_i32())
    };

    // Mean loss over valid tokens - protect against division by zero
    let total_loss = masked_loss.sum_all();
    let n_valid_safe = n_valid.maximum(&Array::from_f32(1.0));
    Ok(total_loss.divide(&n_valid_safe))
}

/// Compute cross-entropy loss for language model training.
///
/// This is a convenience wrapper around `fast_cross_entropy_loss` with a simpler API.
///
/// # Arguments
/// * `logits` - Predicted logits of shape [batch, seq_len, vocab_size]
/// * `targets` - Target token indices of shape [batch, seq_len]
/// * `ignore_index` - Index to ignore in loss computation (e.g., padding token -100)
/// * `label_smoothing` - Label smoothing factor (0.0 to disable)
///
/// # Returns
/// Scalar loss value.
pub fn cross_entropy_loss(
    logits: &Array,
    targets: &Array,
    ignore_index: Option<i64>,
    label_smoothing: f32,
) -> Result<Array, Exception> {
    let mut config = CrossEntropyConfig::new();
    if let Some(idx) = ignore_index {
        config = config.with_ignore_index(idx);
    }
    config = config.with_label_smoothing(label_smoothing);

    // Note: label smoothing is not yet implemented in fast_cross_entropy_loss
    // For now, use the basic implementation if label smoothing is requested
    if label_smoothing > 0.0 {
        return cross_entropy_loss_with_smoothing(logits, targets, ignore_index, label_smoothing);
    }

    fast_cross_entropy_loss(logits, targets, &config)
}

/// Cross-entropy loss with label smoothing support.
fn cross_entropy_loss_with_smoothing(
    logits: &Array,
    targets: &Array,
    ignore_index: Option<i64>,
    label_smoothing: f32,
) -> Result<Array, Exception> {
    let vocab_size = logits.dim(-1);
    let flat_logits = logits.reshape(&[-1, vocab_size]);
    let flat_targets = targets.reshape(&[-1]);

    // Per-token loss via log_softmax + gather
    let log_probs = flat_logits.log_softmax(-1); // [N, vocab]

    // Gather log-prob at label positions
    let targets_i32 = flat_targets.as_dtype(Dtype::Int32.as_i32());
    let gathered_idx = targets_i32.reshape(&[-1, 1]);
    let nll = log_probs.take_along_axis(&gathered_idx, -1).squeeze(-1); // [N]

    // Label smoothing: blend NLL with uniform distribution
    // loss = (1 - smoothing) * nll + smoothing * (-mean(log_probs))
    let uniform_loss = log_probs.negative().mean_axis(-1, false); // [N]
    let alpha = Array::from_f32(1.0 - label_smoothing);
    let beta = Array::from_f32(label_smoothing);
    let per_token_loss = nll
        .negative()
        .multiply(&alpha)
        .add(&uniform_loss.multiply(&beta));

    // Handle ignore_index masking
    if let Some(ignore_idx) = ignore_index {
        let ignore_arr = Array::from_i32(ignore_idx as i32);
        let mask = flat_targets
            .as_dtype(Dtype::Int32.as_i32())
            .not_equal(&ignore_arr);
        let mask_f32 = mask.as_dtype(Dtype::Float32.as_i32());
        let count = mask_f32.sum_all();
        let safe_count = count.maximum(&Array::from_f32(1.0));
        let masked_loss = per_token_loss.multiply(&mask_f32);
        Ok(masked_loss.sum_all().divide(&safe_count))
    } else {
        Ok(per_token_loss.mean_axis(0, false))
    }
}

/// Compute perplexity from cross-entropy loss.
pub fn perplexity(loss: &Array) -> Result<Array, Exception> {
    Ok(loss.exp())
}

/// Cross-entropy loss specifically optimized for Gemma2 models.
///
/// Applies logit softcapping with default cap of 30.0.
pub fn gemma2_cross_entropy_loss(
    logits: &Array,
    targets: &Array,
    ignore_index: Option<i64>,
) -> Result<Array, Exception> {
    let mut config = CrossEntropyConfig::new().with_softcapping(30.0);
    if let Some(idx) = ignore_index {
        config = config.with_ignore_index(idx);
    }
    fast_cross_entropy_loss(logits, targets, &config)
}

/// Cross-entropy loss specifically optimized for Cohere models.
///
/// Applies logit scaling.
pub fn cohere_cross_entropy_loss(
    logits: &Array,
    targets: &Array,
    ignore_index: Option<i64>,
    logit_scale: f32,
) -> Result<Array, Exception> {
    let mut config = CrossEntropyConfig::new().with_scaling(logit_scale);
    if let Some(idx) = ignore_index {
        config = config.with_ignore_index(idx);
    }
    fast_cross_entropy_loss(logits, targets, &config)
}

// ============================================================================
// Length-Based Masking
// ============================================================================
//
// Instead of storing full attention masks [batch, seq, seq], use sequence lengths
// and compute masks on-the-fly. This saves O(n²) -> O(n) memory for sequence masking.

/// Create a sequence mask from lengths
///
/// This is more memory efficient than storing full attention masks:
/// - Storage: O(batch) for lengths vs O(batch * seq * seq) for attention masks
/// - Computation: O(batch * seq) for mask generation
///
/// # Arguments
/// * `lengths` - Sequence lengths [batch]
/// * `max_seq_len` - Maximum sequence length
///
/// # Returns
/// Boolean mask [batch, max_seq_len] where positions < length are True.
pub fn create_mask_from_lengths(lengths: &Array, max_seq_len: i32) -> Result<Array, Exception> {
    // positions = arange(max_seq_len)[None, :]  -> [1, seq_len]
    let positions = ops::arange(max_seq_len, Dtype::Int32);
    let positions = positions.reshape(&[1, max_seq_len]);

    // lengths[:, None] -> [batch, 1]
    let lengths_i32 = lengths.as_dtype(Dtype::Int32.as_i32());
    let lengths_expanded = lengths_i32.expand_dims(-1);

    // mask = positions < lengths -> [batch, seq_len]
    Ok(positions.less(&lengths_expanded))
}

/// Cross-entropy loss with length-based masking
///
/// This is more memory efficient than using explicit attention masks:
/// - Uses sequence lengths instead of full [batch, seq, seq] masks
/// - Computes masking on-the-fly
///
/// # Arguments
/// * `logits` - Predicted logits [batch, seq_len, vocab_size]
/// * `targets` - Target token indices [batch, seq_len]
/// * `lengths` - Sequence lengths [batch] (actual length of each sequence)
/// * `label_smoothing` - Label smoothing factor
///
/// # Returns
/// Scalar loss value, averaged over valid (non-padded) tokens.
pub fn cross_entropy_loss_with_lengths(
    logits: &Array,
    targets: &Array,
    lengths: &Array,
    label_smoothing: f32,
) -> Result<Array, Exception> {
    let batch_size = logits.dim(0);
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(-1);

    // Flatten for loss computation
    let flat_logits = logits.reshape(&[-1, vocab_size]);
    let flat_targets = targets.reshape(&[-1]);

    // Compute per-token log probs
    let log_probs = flat_logits.log_softmax(-1);
    let targets_i32 = flat_targets.as_dtype(Dtype::Int32.as_i32());
    let gathered_idx = targets_i32.reshape(&[-1, 1]);
    let nll = log_probs.take_along_axis(&gathered_idx, -1).squeeze(-1); // [N]

    // Per-token loss
    let per_token_loss = if label_smoothing > 0.0 {
        let uniform_loss = log_probs.negative().mean_axis(-1, false);
        let alpha = Array::from_f32(1.0 - label_smoothing);
        let beta = Array::from_f32(label_smoothing);
        nll.negative()
            .multiply(&alpha)
            .add(&uniform_loss.multiply(&beta))
    } else {
        nll.negative()
    };
    let per_token_loss = per_token_loss.reshape(&[batch_size, seq_len]);

    // Create mask from lengths [batch, seq_len]
    let mask = create_mask_from_lengths(lengths, seq_len)?;
    let mask_f32 = mask.as_dtype(Dtype::Float32.as_i32());

    // Apply mask and compute mean
    let masked_loss = per_token_loss.multiply(&mask_f32);
    let total_loss = masked_loss.sum_all();
    let num_valid_tokens = mask_f32.sum_all();

    // Average loss over valid tokens
    Ok(total_loss.divide(&num_valid_tokens))
}

/// Compute log probabilities with length-based masking (for DPO/preference learning).
///
/// This is a pattern for efficient log prob computation:
/// - Uses `take_along_axis` for efficient gathering
/// - Length-based masking instead of attention masks
/// - Returns sum of log probs per sequence
///
/// # Arguments
/// * `logits` - Model output logits [batch, seq_len, vocab_size]
/// * `targets` - Target token indices [batch, seq_len]
/// * `lengths` - Sequence lengths [batch]
///
/// # Returns
/// Sum of log probabilities per sequence [batch].
pub fn compute_log_probs_with_lengths(
    logits: &Array,
    targets: &Array,
    lengths: &Array,
) -> Result<Array, Exception> {
    let batch = logits.dim(0);
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(2);

    // Shift for next-token prediction
    let pred_logits = logits.slice(&[0, 0, 0], &[batch, seq_len - 1, vocab_size]);
    let target_labels = targets.slice(&[0, 1], &[batch, seq_len]);

    // Compute log softmax
    let log_probs = pred_logits.log_softmax(-1);

    // Gather target log probs using take_along_axis
    let target_i32 = target_labels.as_dtype(Dtype::Int32.as_i32());
    let gather_indices = target_i32.expand_dims(-1);
    let gathered = log_probs.take_along_axis(&gather_indices, -1);
    let gathered = gathered.squeeze(-1);

    // Create mask from lengths (adjusted for shift: lengths - 1)
    let one = Array::from_i32(1);
    let lengths_i32 = lengths.as_dtype(Dtype::Int32.as_i32());
    let adjusted_lengths = lengths_i32.subtract(&one);
    let mask = create_mask_from_lengths(&adjusted_lengths, seq_len - 1)?;
    let mask_f32 = mask.as_dtype(Dtype::Float32.as_i32());

    // Apply mask and sum
    let masked_log_probs = gathered.multiply(&mask_f32);
    Ok(masked_log_probs.sum_axis(1, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Array, Dtype, random};

    #[test]
    fn test_cross_entropy_basic() {
        // Logits: [batch=1, seq=2, vocab=4]
        let logits = Array::from_f32_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, // token 0
                4.0, 3.0, 2.0, 1.0, // token 1
            ],
            &[1, 2, 4],
        );
        // Targets: [batch=1, seq=2]
        let targets = Array::from_i32_slice(&[3_i32, 0]).reshape(&[1, 2]);

        let loss = cross_entropy_loss(&logits, &targets, None, 0.0).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        // Should be close to 0 since we're predicting the argmax
        let val = loss_eval.item_f32();
        assert!(val < 1.5);
    }

    #[test]
    fn test_cross_entropy_with_ignore() {
        let logits = Array::from_f32_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, // token 0 - predict class 3
                4.0, 3.0, 2.0, 1.0, // token 1 - predict class 0
                1.0, 1.0, 1.0, 1.0, // token 2 - ignored
            ],
            &[1, 3, 4],
        );
        let targets = Array::from_i32_slice(&[3_i32, 0, -100]).reshape(&[1, 3]);

        let loss = cross_entropy_loss(&logits, &targets, Some(-100), 0.0).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        let value = loss_eval.item_f32();
        assert!(value.is_finite());
    }

    #[test]
    fn test_fast_cross_entropy_config() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], &[2, 4]);
        let targets = Array::from_i32_slice(&[3_i32, 0]).reshape(&[2]);

        let config = CrossEntropyConfig::new().with_ignore_index(-100);
        let loss = fast_cross_entropy_loss(&logits, &targets, &config).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        assert!(loss_eval.item_f32().is_finite());
    }

    #[test]
    fn test_logit_softcapping() {
        let logits = Array::from_f32_slice(&[100.0_f32, -100.0, 0.0, 50.0], &[1, 4]);
        let targets = Array::from_i32_slice(&[0_i32]).reshape(&[1]);

        let config = CrossEntropyConfig::new().with_softcapping(30.0);
        let loss = fast_cross_entropy_loss(&logits, &targets, &config).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        assert!(loss_eval.item_f32().is_finite());
    }

    #[test]
    fn test_gemma2_cross_entropy() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let targets = Array::from_i32_slice(&[3_i32, 0]).reshape(&[2]);

        let loss = gemma2_cross_entropy_loss(&logits, &targets, None).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        assert!(loss_eval.item_f32().is_finite());
    }

    #[test]
    fn test_perplexity() {
        let loss = Array::from_f32(2.0);
        let ppl = perplexity(&loss).unwrap();
        let ppl_eval = ppl.clone();
        ppl_eval.eval();

        let expected = 2.0_f32.exp();
        let value = ppl_eval.item_f32();
        assert!((value - expected).abs() < 0.01);
    }

    #[test]
    fn test_stable_logsumexp() {
        let logits = Array::from_f32_slice(&[1000.0_f32, 1001.0, 1002.0, 1003.0], &[1, 4]);
        let lse = stable_logsumexp(&logits);
        let lse_eval = lse.clone();
        lse_eval.eval();

        let value = lse_eval.item_f32();
        assert!(value.is_finite());
        assert!((value - 1003.44).abs() < 0.1);
    }

    #[test]
    fn test_chunked_logsumexp_small_vocab() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        let lse = chunked_logsumexp(&logits, 4);
        let lse_eval = lse.clone();
        lse_eval.eval();

        let expected = stable_logsumexp(&logits);
        let exp_eval = expected.clone();
        exp_eval.eval();

        let lse_val = lse_eval.item_f32();
        let expected_val = exp_eval.item_f32();
        assert!((lse_val - expected_val).abs() < 0.001);
    }

    #[test]
    fn test_create_mask_from_lengths() {
        let lengths = Array::from_i32_slice(&[2_i32, 4, 3]).reshape(&[3]);
        let max_len = 5;

        let mask = create_mask_from_lengths(&lengths, max_len).unwrap();
        let mask_f32 = mask.as_dtype(Dtype::Float32.as_i32());
        let mut mf_eval = mask_f32.clone();
        mf_eval.eval();

        // Expected mask shape: [3, 5]
        assert_eq!(mask.shape(), &[3, 5]);

        let n = 15;
        let mask_vals = mf_eval.to_f32_vec(n).unwrap_or_default();
        assert_eq!(mask_vals.len(), 15);

        // Row 0: length=2, so [1,1,0,0,0]
        assert!((mask_vals[0] - 1.0).abs() < 0.01);
        assert!((mask_vals[1] - 1.0).abs() < 0.01);
        assert!((mask_vals[2] - 0.0).abs() < 0.01);

        // Row 1: length=4
        assert!((mask_vals[5] - 1.0).abs() < 0.01);
        assert!((mask_vals[8] - 1.0).abs() < 0.01);
        assert!((mask_vals[9] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_entropy_loss_with_lengths() {
        let batch_size: i32 = 2;
        let seq_len: i32 = 4;
        let vocab_size: i32 = 10;

        let logits_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size) as usize)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let logits = Array::from_f32_slice(&logits_data, &[batch_size, seq_len, vocab_size]);

        let labels =
            Array::from_i32_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8]).reshape(&[batch_size, seq_len]);

        let lengths = Array::from_i32_slice(&[3i32, 4]).reshape(&[batch_size]);

        let loss = cross_entropy_loss_with_lengths(&logits, &labels, &lengths, 0.0).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        let loss_val = loss_eval.item_f32();
        assert!(
            loss_val.is_finite(),
            "Loss should be finite, got {}",
            loss_val
        );
        assert!(loss_val > 0.0, "Loss should be positive");
    }

    #[test]
    fn test_cross_entropy_division_by_zero_protection() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
        let targets = Array::from_i32_slice(&[-100_i32]).reshape(&[1]);

        let config = CrossEntropyConfig::new().with_ignore_index(-100);
        let loss = fast_cross_entropy_loss(&logits, &targets, &config).unwrap();
        let loss_eval = loss.clone();
        loss_eval.eval();

        let loss_val = loss_eval.item_f32();
        assert!(
            loss_val.is_finite(),
            "Loss should be finite when all tokens ignored, got {}",
            loss_val
        );
    }
}

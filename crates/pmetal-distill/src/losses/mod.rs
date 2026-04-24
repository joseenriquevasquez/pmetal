//! Loss functions for knowledge distillation.
//!
//! GPU-first implementations using Metal kernels for optimal Apple Silicon performance.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! This module provides various loss functions used in knowledge distillation:
//! - KL Divergence (forward and reverse)
//! - Jensen-Shannon Divergence
//! - Soft Cross-Entropy
//! - MSE on logits
//! - Hidden state alignment losses
//!
//! # GPU Acceleration
//!
//! When the `metal` feature is enabled (default), all loss implementations
//! automatically use custom Metal kernels with these optimizations:
//! - Online softmax: O(1) memory per token instead of materializing full probability tensors
//! - Fused operations: temperature scaling + softmax + loss in single kernel pass
//! - SIMD parallelization: Optimized for large vocabularies (>1024 tokens)
//!
//! # Example
//!
//! ```rust,ignore
//! use pmetal_distill::losses::{KlDivergenceLoss, DistillLoss};
//!
//! let loss = KlDivergenceLoss::new();
//!
//! // GPU acceleration is automatic - no API changes needed
//! let result = loss.compute(&teacher_logits, &student_logits, 2.0)?;
//! ```

mod attention_transfer;
pub mod hidden_state;
mod hinge_ranking;
mod jensen_shannon;
mod kl_divergence;
mod logistic_ranking;
mod mse;
mod soft_cross_entropy;
mod tvd;

pub use attention_transfer::AttentionTransferLoss;
pub use hidden_state::HiddenStateLoss;
pub use hinge_ranking::HingeRankingLoss;
pub use jensen_shannon::JensenShannonLoss;
pub use kl_divergence::KlDivergenceLoss;
pub use logistic_ranking::LogisticRankingLoss;
pub use mse::MseLoss;
pub use soft_cross_entropy::SoftCrossEntropyLoss;
pub use tvd::TvdLoss;

use crate::Result;
use pmetal_bridge::compat::{Array, Dtype, ops};

/// Default number of top-k teacher tokens to use when vocab sizes differ.
///
/// 128 covers the vast majority of teacher probability mass in practice
/// (top-128 tokens typically account for >99.9% of mass in peaked distributions).
pub const SPARSE_TOPK_DEFAULT: i32 = 128;

/// Trait for distillation loss functions.
pub trait DistillLoss: Send + Sync {
    /// Compute the distillation loss between teacher and student outputs.
    ///
    /// # Arguments
    /// * `teacher_logits` - Logits from the teacher model
    /// * `student_logits` - Logits from the student model
    /// * `temperature` - Temperature for softmax scaling
    ///
    /// # Returns
    /// The computed loss value as a scalar array.
    fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        self.compute_weighted(teacher_logits, student_logits, temperature, None)
    }

    /// Compute weighted distillation loss.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits
    /// * `student_logits` - Student logits
    /// * `temperature` - Softmax temperature
    /// * `weights` - Per-token weights `[batch, seq]`
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array>;

    /// Compute token-level masked distillation loss.
    ///
    /// Tokens where `mask == 0` are excluded from the mean so that padding
    /// positions and special tokens do not dilute the gradient signal.
    ///
    /// The default implementation calls `compute_weighted` with `weights = None`,
    /// multiplies per-token losses by the mask, then normalises by the number of
    /// unmasked tokens (floored at 1 to avoid NaN on all-zero masks).
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits `[batch, seq, vocab]`
    /// * `student_logits` - Student logits `[batch, seq, vocab]`
    /// * `temperature` - Softmax temperature
    /// * `mask` - Binary mask `[batch, seq]` where 1 = include, 0 = exclude
    fn compute_masked(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        mask: &Array,
    ) -> Result<Array> {
        let loss = self.compute_weighted(teacher_logits, student_logits, temperature, None)?;
        let masked = loss.multiply(mask);
        let sum = masked.sum_all();
        let count = mask.sum_all();
        let safe_count = ops::maximum(&count, &Array::from_f32(1.0));
        Ok(sum.divide(&safe_count))
    }

    /// Get the name of this loss function.
    fn name(&self) -> &'static str;
}

/// Check if GPU acceleration is available for distillation losses.
///
/// Returns true if Metal is available and the device supports GPU-accelerated
/// distillation loss computation.
#[cfg(feature = "metal")]
pub fn is_gpu_available() -> bool {
    pmetal_metal::context::MetalContext::global().is_ok()
}

#[cfg(not(feature = "metal"))]
pub fn is_gpu_available() -> bool {
    false
}

/// Combined distillation loss with hard and soft targets.
pub struct CombinedLoss {
    /// Soft target loss function.
    soft_loss: Box<dyn DistillLoss>,
    /// Alpha for blending (final = alpha * soft + (1-alpha) * hard).
    alpha: f32,
    /// Temperature for soft targets.
    temperature: f32,
}

impl CombinedLoss {
    /// Create a new combined loss.
    pub fn new(soft_loss: Box<dyn DistillLoss>, alpha: f32, temperature: f32) -> Self {
        Self {
            soft_loss,
            alpha,
            temperature,
        }
    }

    /// Compute combined loss with hard labels.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher model outputs
    /// * `student_logits` - Student model outputs
    /// * `labels` - Ground truth labels for hard loss
    pub fn compute_with_labels(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        labels: &Array,
    ) -> Result<Array> {
        // Soft loss (temperature-scaled KL/CE between teacher and student)
        let soft = self
            .soft_loss
            .compute(teacher_logits, student_logits, self.temperature)?;

        // Hard loss (cross-entropy with ground truth)
        let hard = cross_entropy_with_logits(student_logits, labels)?;

        // Weighted combination
        // Note: soft loss is scaled by T^2 to maintain gradient magnitude
        let t_squared = self.temperature * self.temperature;
        let soft_scaled = soft.multiply(&Array::from_f32(self.alpha * t_squared));
        let hard_scaled = hard.multiply(&Array::from_f32(1.0 - self.alpha));

        Ok(soft_scaled.add(&hard_scaled))
    }
}

/// Standard cross-entropy loss with logits and integer labels.
fn cross_entropy_with_logits(logits: &Array, labels: &Array) -> Result<Array> {
    // Log-softmax for numerical stability
    let log_probs = logits.log_softmax(-1);

    // Gather the log probabilities at label positions
    // Shape: logits [batch, seq, vocab], labels [batch, seq]
    let vocab_size = logits.dim(2);

    // Flatten for gather operation
    let log_probs_flat = log_probs.reshape(&[-1, vocab_size]);
    let labels_flat = labels.reshape(&[-1]);

    // Get log prob at each label position using MLX take_along_axis
    let gathered = gather_at_indices(&log_probs_flat, &labels_flat)?;

    // Mean negative log probability
    let neg_log_probs = gathered.negative();
    let loss = neg_log_probs.mean_all();

    Ok(loss)
}

/// Align teacher and student logits when their vocab sizes differ.
///
/// Uses the default top-k of [`SPARSE_TOPK_DEFAULT`] (128).  For a custom `k`,
/// call [`align_vocab_with_k`] directly.
///
/// When teacher and student have different vocabulary sizes (cross-architecture
/// distillation), aligning via sparse top-k is more principled than truncation
/// or padding: it preserves the distribution information where it matters most
/// and works regardless of which model has the larger vocabulary.
///
/// # Returns
///
/// `(teacher_aligned, student_aligned, mismatched)`.  `mismatched` is `true`
/// when vocab sizes differed and the sparse path was taken.
pub fn align_vocab(teacher_logits: &Array, student_logits: &Array) -> Result<(Array, Array, bool)> {
    align_vocab_with_k(teacher_logits, student_logits, SPARSE_TOPK_DEFAULT)
}

/// Align teacher and student logits when their vocab sizes differ, with a configurable top-k.
///
/// When teacher and student have different vocabulary sizes (cross-architecture
/// distillation), aligning via sparse top-k is more principled than truncation
/// or padding: it preserves the distribution information where it matters most
/// and works regardless of which model has the larger vocabulary.
///
/// # Algorithm
///
/// 1. Identify the teacher's top-k token indices along the last axis.
/// 2. Gather teacher logits at those k positions → shape `[..., k]`.
/// 3. Gather student logits at those same k positions → shape `[..., k]`.
///    Teacher indices that exceed the student vocab are masked to `-1e9` so
///    they contribute negligible softmax probability.
///
/// The returned pair has matching last dimensions so any loss function can be
/// applied without modification.  The Metal GPU path must be bypassed for
/// mismatched vocab because the fused kernels require equal vocab sizes.
///
/// # Parameters
///
/// * `top_k` – number of teacher tokens to retain per position.  Clamped to
///   `teacher_vocab` if larger.  Must be ≥ 1.
///
/// # Returns
///
/// `(teacher_aligned, student_aligned, mismatched)`.  `mismatched` is `true`
/// when vocab sizes differed and the sparse path was taken.
pub fn align_vocab_with_k(
    teacher_logits: &Array,
    student_logits: &Array,
    top_k: i32,
) -> Result<(Array, Array, bool)> {
    let teacher_vocab = teacher_logits.dim(-1) as usize;
    let student_vocab = student_logits.dim(-1) as usize;

    if teacher_vocab == student_vocab {
        return Ok((teacher_logits.clone(), student_logits.clone(), false));
    }

    // argpartition requires kth < size; cap at teacher_vocab - 1 so we never
    // request kth == vocab_size (which would be out-of-range).
    let k = top_k
        .max(1)
        .min((teacher_vocab as i32).saturating_sub(1).max(1));

    tracing::debug!(
        teacher_vocab,
        student_vocab,
        k,
        "vocab mismatch: using sparse top-k distillation"
    );

    // Argpartition descending: negate teacher logits then argpartition so the
    // first k indices correspond to the k largest teacher logit positions.
    // O(V) vs O(V log V) for argsort — significant win at vocab size ~150k.
    let neg_teacher = teacher_logits.negative();
    let partitioned_indices = ops::argpartition_axis(&neg_teacher, k as i32, -1);

    // Slice first k positions along the last axis.
    //
    // Reshape to 2D [N, vocab], slice to [N, k], reshape back to leading_dims + [k].
    // This is rank-agnostic and works correctly for tensors of any rank.
    let shape = partitioned_indices.shape().to_vec();
    let ndim = shape.len();
    let vocab_dim = shape[ndim - 1] as usize;
    let n: i32 = shape[..ndim - 1].iter().product();

    let part_2d = partitioned_indices.reshape(&[n, vocab_dim as i32]);
    let top_k_2d = part_2d.slice(&[0, 0], &[n, k]);
    // Rebuild leading shape + [k]
    let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
    out_shape.push(k);
    let top_k_indices = top_k_2d.reshape(&out_shape);

    // Gather teacher logits at the top-k token positions.
    let teacher_aligned = teacher_logits.take_along_axis(&top_k_indices, -1);

    // Clamp teacher indices to the student vocab range for gathering.
    // Out-of-range positions are masked below so clamping to 0 is safe here.
    let student_vocab_minus1 = Array::from_i32((student_vocab as i32) - 1);
    let zero = Array::from_i32(0);
    let clamped =
        ops::minimum(&top_k_indices, &student_vocab_minus1).as_dtype(Dtype::Int32.as_i32());
    let clamped = ops::maximum(&clamped, &zero);

    // Gather student logits at clamped positions.
    let student_gathered = student_logits.take_along_axis(&clamped, -1);

    // Mask out positions where the teacher index fell outside the student vocab.
    // Those positions receive -1e9 so their softmax weight is negligible (~0).
    let student_vocab_arr = Array::from_i32(student_vocab as i32);
    let out_of_range = top_k_indices
        .as_dtype(Dtype::Int32.as_i32())
        .greater_equal(&student_vocab_arr);
    let neg_large = Array::from_f32(-1e9_f32);
    let student_aligned = ops::where_fn(&out_of_range, &neg_large, &student_gathered);

    Ok((teacher_aligned, student_aligned, true))
}

/// Softmax along specified axis.
pub fn softmax(x: &Array, axis: i32) -> Result<Array> {
    Ok(x.softmax(axis))
}

/// Gather values from a 2D array at specified column indices.
///
/// Uses MLX take operation for GPU-accelerated gather.
fn gather_at_indices(values: &Array, indices: &Array) -> Result<Array> {
    // values: [N, V], indices: [N] -> output: [N]
    let n = values.dim(0);
    let v = values.dim(1);

    // Create row indices [0, 1, 2, ..., N-1] via GPU arange — stays in the
    // MLX compute graph and avoids a CPU Vec<i32> allocation.
    let row_indices_arr = Array::arange(n, Dtype::Int32.as_i32());

    // Compute flat indices: row * V + col
    let v_arr = Array::from_i32(v);
    let flat_indices = row_indices_arr.multiply(&v_arr).add(indices);

    // Flatten values and gather
    let values_flat = values.reshape(&[-1]);
    let gathered = values_flat.take_axis(&flat_indices, 0);

    Ok(gathered)
}

/// Combine per-token losses produced by a fused Metal kernel with an optional
/// per-token weight tensor and return a scalar reduction.
///
/// When `weights` is `None`, returns the unweighted mean.
/// When `weights` is `Some`, returns `sum(loss * w) / max(sum(w), 1e-8)`; the
/// weight tensor must have `num_tokens` total elements (typically shape
/// `[batch, seq]`).
///
/// This is the shared path used by `KlDivergenceLoss`, `JensenShannonLoss`,
/// `SoftCrossEntropyLoss` etc. to consume the per-token output of
/// `FusedDistill::forward`. Weighted reduction runs on the CPU (≤ a few KB of
/// f32 data per step), which is negligible next to the O(N·V) kernel work that
/// already stayed on the GPU.
#[cfg(feature = "metal")]
pub(crate) fn reduce_per_token_with_weights(
    per_token_losses: &pmetal_metal::buffer::MetalBuffer<f32>,
    weights: Option<&Array>,
    num_tokens: usize,
) -> Result<Array> {
    let losses = per_token_losses.as_slice();
    if losses.is_empty() {
        return Ok(Array::from_f32(0.0));
    }

    let Some(w) = weights else {
        let sum: f32 = losses.iter().sum();
        return Ok(Array::from_f32(sum / losses.len() as f32));
    };

    let w_size: usize = w.shape().iter().map(|&d| d as usize).product();
    if w_size != num_tokens {
        return Err(crate::DistillError::Other(format!(
            "weight tensor has {} elements but expected {} (one per token)",
            w_size, num_tokens
        )));
    }

    let w_flat = w.reshape(&[num_tokens as i32]);
    w_flat.eval();
    let w_slice = w_flat.as_slice::<f32>();

    let mut weighted_sum = 0.0_f32;
    let mut total_weight = 0.0_f32;
    for (loss, weight) in losses.iter().zip(w_slice.iter()) {
        weighted_sum += loss * weight;
        total_weight += weight;
    }
    let safe_weight = total_weight.abs().max(1e-8);
    Ok(Array::from_f32(weighted_sum / safe_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_softmax() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let probs = softmax(&logits, -1).unwrap();
        let probs_data: Vec<f32> = probs.clone().to_f32_vec(3).unwrap();

        // Check probabilities sum to 1
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check relative ordering
        assert!(probs_data[2] > probs_data[1]);
        assert!(probs_data[1] > probs_data[0]);
    }

    #[test]
    #[serial]
    fn test_log_softmax() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let log_probs = logits.log_softmax(-1);
        let log_probs_data: Vec<f32> = log_probs.clone().to_f32_vec(3).unwrap();

        // All log probs should be <= 0
        for lp in &log_probs_data {
            assert!(*lp <= 0.0);
        }

        // exp(log_softmax) should equal softmax
        let probs = log_probs.exp();
        let probs_data: Vec<f32> = probs.clone().to_f32_vec(3).unwrap();
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    #[serial]
    fn test_gather_at_indices() {
        let values = Array::from_f32_slice(&[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let indices = Array::from_i32_slice(&[1_i32, 2]);
        let result = gather_at_indices(&values, &indices).unwrap();
        let result_data: Vec<f32> = result.clone().to_f32_vec(2).unwrap();

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - 0.2).abs() < 1e-5); // row 0, col 1
        assert!((result_data[1] - 0.6).abs() < 1e-5); // row 1, col 2
    }

    #[test]
    #[serial]
    fn test_gpu_availability_check() {
        // Should return true on Apple Silicon with Metal feature
        let available = is_gpu_available();
        println!("GPU acceleration available: {}", available);
    }

    // -----------------------------------------------------------------
    // align_vocab / cross-vocab tests
    // -----------------------------------------------------------------

    /// Same vocab size → passthrough, no copy.
    #[test]
    #[serial]
    fn test_align_vocab_same_size_passthrough() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let (t, s, mismatched) = align_vocab(&logits, &logits).unwrap();
        assert!(!mismatched, "same vocab should not be mismatched");
        // Shapes preserved
        assert_eq!(t.shape(), logits.shape());
        assert_eq!(s.shape(), logits.shape());
    }

    /// Teacher vocab > student vocab (the Qwen3-4B → Qwen3.5-0.8B scenario).
    /// After alignment both tensors must have the same last dimension (k).
    #[test]
    #[serial]
    fn test_align_vocab_teacher_larger_2d() {
        // teacher: [1, 8], student: [1, 6]  (teacher larger)
        let teacher_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let student_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 8]);
        let student = Array::from_f32_slice(&student_data, &[1, 6]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.dim(-1), k);
        assert_eq!(sa.dim(-1), k);
        // Teacher leading dims preserved
        assert_eq!(ta.shape()[..ta.shape().len() - 1], [1]);
    }

    /// Student vocab > teacher vocab.
    #[test]
    #[serial]
    fn test_align_vocab_student_larger_2d() {
        // teacher: [1, 6], student: [1, 8]  (student larger)
        let teacher_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let student_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 6]);
        let student = Array::from_f32_slice(&student_data, &[1, 8]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.dim(-1), k);
        assert_eq!(sa.dim(-1), k);
    }

    /// 3-D tensors [batch, seq, vocab] — the common real-world shape.
    /// This is the critical regression test: the previous `(.., ..k)` indexing
    /// would slice the sequence dimension, producing wrong shapes.
    #[test]
    #[serial]
    fn test_align_vocab_3d_batch_seq_vocab() {
        let batch = 2_i32;
        let seq = 3_i32;
        let teacher_vocab = 10_i32;
        let student_vocab = 8_i32;
        let k = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| i as f32)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| i as f32)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        // Shape must be [batch, seq, k] — NOT [batch, k, teacher_vocab]
        assert_eq!(
            ta.shape(),
            &[batch, seq, k],
            "teacher_aligned shape wrong: {:?}",
            ta.shape()
        );
        assert_eq!(
            sa.shape(),
            &[batch, seq, k],
            "student_aligned shape wrong: {:?}",
            sa.shape()
        );
    }

    /// Verify that top-k indices actually correspond to the largest teacher logits.
    #[test]
    #[serial]
    fn test_align_vocab_selects_top_logits() {
        // teacher: [1, 6] with clear ordering
        // token 5 has highest logit (10.0), token 4 second (9.0), etc.
        let teacher_data = [1.0_f32, 2.0, 3.0, 4.0, 9.0, 10.0];
        let student_data = [0.1_f32, 0.2, 0.3, 0.4]; // only 4 tokens
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 6]);
        let student = Array::from_f32_slice(&student_data, &[1, 4]);

        let (ta, _sa, _) = align_vocab_with_k(&teacher, &student, 3).unwrap();
        ta.eval();

        let ta_vals: Vec<f32> = ta.clone().to_f32_vec(3).unwrap();
        // All three selected values must be from {4.0, 9.0, 10.0}
        // i.e. all >= 4.0 (the 4th-highest)
        for v in &ta_vals {
            assert!(*v >= 4.0 - 1e-5, "unexpected value in top-3: {}", v);
        }
    }

    /// Positions where the teacher index falls outside the student vocab must be
    /// masked to -1e9 in the student-aligned tensor.
    #[test]
    #[serial]
    fn test_align_vocab_out_of_range_masked() {
        // teacher has 8 tokens; student has only 4.
        // The highest teacher logits (indices 7, 6, 5, 4) all exceed student_vocab=4,
        // so all 4 selected student positions must be masked.
        let teacher_data = [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]; // indices 4-7 top
        let student_data = [0.1_f32, 0.2, 0.3, 0.4];
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 8]);
        let student = Array::from_f32_slice(&student_data, &[1, 4]);

        let (_ta, sa, _) = align_vocab_with_k(&teacher, &student, 4).unwrap();
        sa.eval();

        let sa_vals: Vec<f32> = sa.clone().to_f32_vec(4).unwrap();
        // Every entry must be ≤ -1e8 (masked)
        for v in &sa_vals {
            assert!(*v <= -1e8, "expected masked value (-1e9), got {}", v);
        }
    }

    /// Simulate the real cross-architecture case: teacher_vocab=151936, student_vocab=152080.
    /// Here student_vocab > teacher_vocab so no masking needed.
    /// Verifies the function completes without panic and returns sensible shapes.
    #[test]
    #[serial]
    fn test_align_vocab_qwen3_to_qwen35_shapes() {
        // Use small stand-ins for the real vocab sizes but with the same relationship
        // (student > teacher), scaled down for test speed.
        let teacher_vocab = 120_i32;
        let student_vocab = 130_i32;
        let batch = 1_i32;
        let seq = 2_i32;
        let k = 32_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 50) as f32 - 25.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i % 50) as f32 - 25.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.shape(), &[batch, seq, k]);
        assert_eq!(sa.shape(), &[batch, seq, k]);

        // All teacher-aligned values must be finite
        let ta_vals: Vec<f32> = ta.clone().to_f32_vec(64).unwrap();
        assert!(ta_vals.iter().all(|v| v.is_finite()));

        // student-aligned: indices < student_vocab so no masking expected
        let sa_vals: Vec<f32> = sa.clone().to_f32_vec(64).unwrap();
        assert!(sa_vals.iter().all(|v| v.is_finite()));
    }
}

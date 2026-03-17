//! Logistic ranking loss for knowledge distillation.
//!
//! A softer alternative to hinge ranking loss using softplus:
//!
//! For each pair (i, j) where P_teacher[i] > P_teacher[j]:
//!   loss += softplus(-(logit_student[i] - logit_student[j]))
//!         = log(1 + exp(-(logit_student[i] - logit_student[j])))
//!
//! Provides smoother gradients than hinge for more stable training.

use crate::Result;
use mlx_rs::Array;

use super::DistillLoss;

/// Logistic ranking loss for smooth pairwise token ordering.
///
/// Uses softplus (log(1 + exp(-x))) instead of relu for smoother gradients.
/// Operates on student logits directly (not probabilities) for stability.
pub struct LogisticRankingLoss {
    /// Number of top teacher tokens to consider for pairwise ranking.
    top_k: i32,
    /// Minimum teacher probability difference to form a valid pair.
    eps: f32,
}

impl LogisticRankingLoss {
    /// Create a new logistic ranking loss with default top_k=32.
    pub fn new() -> Self {
        Self {
            top_k: 32,
            eps: 1e-6,
        }
    }

    /// Set the number of top-k teacher tokens for pairwise ranking.
    pub fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }
}

impl Default for LogisticRankingLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for LogisticRankingLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let temp = Array::from_f32(temperature);
        let k = self.top_k;

        // Teacher probabilities for ranking
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let teacher_probs = super::softmax(&teacher_scaled, -1)?;

        // Top-k indices
        let vocab = teacher_logits.dim(-1);
        let k = k.min((vocab - 1).max(0)); // argpartition requires kth < axis size
        let neg_teacher = teacher_probs.negative()?;
        let partitioned = mlx_rs::ops::argpartition_axis(&neg_teacher, k, -1)?;
        use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
        let top_k_idx = partitioned.index((Ellipsis, ..k));

        // Gather teacher probs at top-k
        let t_gathered = teacher_probs.take_along_axis(&top_k_idx, -1)?;

        // Student LOGITS (not probs) at same positions — for softplus stability
        let student_scaled = student_logits.divide(&temp)?;
        let s_gathered = student_scaled.take_along_axis(&top_k_idx, -1)?;

        // Flatten to [N, K]
        let shape = teacher_logits.shape();
        let ndim = shape.len();
        let n: i32 = shape[..ndim - 1].iter().product();

        let t_flat = t_gathered.reshape(&[n, k])?;
        let s_flat = s_gathered.reshape(&[n, k])?;

        // Pairwise: [N, K, 1] vs [N, 1, K]
        let t_i = t_flat.reshape(&[n, k, 1])?;
        let t_j = t_flat.reshape(&[n, 1, k])?;
        let s_i = s_flat.reshape(&[n, k, 1])?;
        let s_j = s_flat.reshape(&[n, 1, k])?;

        let teacher_margin = t_i.subtract(&t_j)?;
        let student_diff = s_i.subtract(&s_j)?; // logit difference

        // Valid pairs
        let eps_arr = Array::from_f32(self.eps);
        let valid_mask = teacher_margin
            .gt(&eps_arr)?
            .as_dtype(mlx_rs::Dtype::Float32)?;

        // Softplus(-diff) = log(1 + exp(-diff))
        // Numerically stable: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        let neg_diff = student_diff.negative()?;
        let abs_diff = mlx_rs::ops::abs(&student_diff)?;
        let zero = Array::from_f32(0.0);
        let softplus = mlx_rs::ops::maximum(&neg_diff, &zero)?
            .add(&mlx_rs::ops::log1p(&abs_diff.negative()?.exp()?)?)?;

        // Masked mean
        let masked = softplus.multiply(&valid_mask)?;
        let pair_sum = masked.sum_axes(&[-1, -2], None)?;
        let pair_count = valid_mask.sum_axes(&[-1, -2], None)?;
        let safe_count = mlx_rs::ops::maximum(&pair_count, &Array::from_f32(1.0))?;
        let per_token = pair_sum.divide(&safe_count)?;

        if let Some(w) = weights {
            let w_flat = w.reshape(&[-1])?;
            let weighted = per_token.multiply(&w_flat)?;
            let sum = weighted.sum(false)?;
            let w_sum = w_flat.sum(false)?;
            let safe_sum = mlx_rs::ops::maximum(&w_sum, &Array::from_f32(1.0))?;
            Ok(sum.divide(&safe_sum)?)
        } else {
            Ok(per_token.mean(None)?)
        }
    }

    fn name(&self) -> &'static str {
        "logistic_ranking"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_ranking_higher_loss() {
        let teacher = Array::from_slice(&[10.0f32, 0.0, -10.0], &[1, 1, 3]);
        let student_good = Array::from_slice(&[8.0f32, 0.0, -8.0], &[1, 1, 3]);
        let student_bad = Array::from_slice(&[-10.0f32, 0.0, 10.0], &[1, 1, 3]);

        let loss = LogisticRankingLoss::new().with_top_k(3);
        let good = loss.compute(&teacher, &student_good, 1.0).unwrap();
        let bad = loss.compute(&teacher, &student_bad, 1.0).unwrap();
        good.eval().unwrap();
        bad.eval().unwrap();

        let good_val: f32 = good.item();
        let bad_val: f32 = bad.item();
        assert!(
            bad_val > good_val,
            "Reversed ranking → higher loss: good={good_val}, bad={bad_val}"
        );
    }
}

//! Hinge ranking loss for knowledge distillation.
//!
//! Encourages the student to preserve the teacher's relative token ordering
//! via a pairwise margin-based loss:
//!
//! For each pair (i, j) where P_teacher[i] > P_teacher[j]:
//!   loss += max(0, margin - (P_student[i] - P_student[j]))

use crate::Result;
use mlx_rs::Array;

use super::DistillLoss;

/// Hinge ranking loss for pairwise token ordering.
///
/// For the top-k teacher tokens, creates all K*(K-1)/2 ordered pairs and
/// penalizes the student whenever its probability ranking disagrees with
/// the teacher's.
pub struct HingeRankingLoss {
    /// Number of top teacher tokens to consider for pairwise ranking.
    top_k: i32,
    /// Minimum teacher probability difference to form a valid pair.
    eps: f32,
}

impl HingeRankingLoss {
    /// Create a new hinge ranking loss with default top_k=32.
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

impl Default for HingeRankingLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for HingeRankingLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let temp = Array::from_f32(temperature);
        let k = self.top_k;

        // Teacher probabilities
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let teacher_probs = super::softmax(&teacher_scaled, -1)?;

        // Top-k indices via argpartition (O(V) not O(V log V))
        let vocab = teacher_logits.dim(-1);
        let k = k.min((vocab - 1).max(0)); // argpartition requires kth < axis size
        let neg_teacher = teacher_probs.negative()?;
        let partitioned = mlx_rs::ops::argpartition_axis(&neg_teacher, k, -1)?;
        use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
        let top_k_idx = partitioned.index((Ellipsis, ..k));

        // Gather teacher probs at top-k
        let t_gathered = teacher_probs.take_along_axis(&top_k_idx, -1)?;

        // Student probs at same positions
        let student_scaled = student_logits.divide(&temp)?;
        let student_probs = super::softmax(&student_scaled, -1)?;
        let s_gathered = student_probs.take_along_axis(&top_k_idx, -1)?;

        // Flatten to [N, K] where N = batch * seq
        let shape = teacher_logits.shape();
        let ndim = shape.len();
        let n: i32 = shape[..ndim - 1].iter().product();

        let t_flat = t_gathered.reshape(&[n, k])?;
        let s_flat = s_gathered.reshape(&[n, k])?;

        // Pairwise: [N, K, 1] - [N, 1, K] → [N, K, K]
        let t_i = t_flat.reshape(&[n, k, 1])?;
        let t_j = t_flat.reshape(&[n, 1, k])?;
        let s_i = s_flat.reshape(&[n, k, 1])?;
        let s_j = s_flat.reshape(&[n, 1, k])?;

        let teacher_margin = t_i.subtract(&t_j)?;
        let student_diff = s_i.subtract(&s_j)?;

        // Valid pairs: teacher[i] > teacher[j] + eps
        let eps_arr = Array::from_f32(self.eps);
        let valid_mask = teacher_margin
            .gt(&eps_arr)?
            .as_dtype(mlx_rs::Dtype::Float32)?;

        // Hinge: relu(margin - student_diff)
        let violation = teacher_margin.subtract(&student_diff)?;
        let zero = Array::from_f32(0.0);
        let hinge = mlx_rs::ops::maximum(&violation, &zero)?;

        // Masked mean over pairs → per-token scalar
        let masked = hinge.multiply(&valid_mask)?;
        let pair_sum = masked.sum_axes(&[-1, -2], None)?; // [N]
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
        "hinge_ranking"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_zero_loss() {
        let logits = Array::from_slice(&[3.0f32, 2.0, 1.0, 0.5, 3.0, 2.0, 1.0, 0.5], &[1, 2, 4]);
        let loss = HingeRankingLoss::new().with_top_k(4);
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        result.eval().unwrap();
        let val: f32 = result.item();
        assert!(
            val.abs() < 1e-4,
            "Same distributions → zero hinge, got {val}"
        );
    }

    #[test]
    fn reversed_ranking_positive_loss() {
        let teacher = Array::from_slice(&[10.0f32, 0.0, -10.0], &[1, 1, 3]);
        let student = Array::from_slice(&[-10.0f32, 0.0, 10.0], &[1, 1, 3]);
        let loss = HingeRankingLoss::new().with_top_k(3);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval().unwrap();
        let val: f32 = result.item();
        assert!(val > 0.1, "Reversed ranking → positive hinge, got {val}");
    }
}

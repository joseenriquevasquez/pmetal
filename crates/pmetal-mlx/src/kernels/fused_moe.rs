//! MLX MoE combine operations.
//!
//! Pure MLX ops for MoE expert output combination. The previous Metal side-channel
//! (`fused_moe_combine`) has been removed — the 6 MLX ops are already async on GPU
//! and adding synchronization barriers (4x eval() + waitUntilCompleted) made the
//! Metal path 5-20x slower than the MLX ops it replaced.

use mlx_rs::Array;
use mlx_rs::error::Exception;

/// MoE combine: residual + weighted expert sum + sigmoid-gated shared expert.
///
/// Computes:
/// ```text
/// y = (expert_outs * weights.unsqueeze(-1)).sum(-2)  // weighted sum
/// shared_gate = sigmoid(shared_gate_logit)
/// y += shared_gate * shared_out
/// result = x + y  // residual
/// ```
///
/// All ops run asynchronously on GPU via MLX's lazy evaluation graph.
///
/// # Arguments
/// * `residual` - Input hidden states `[batch_seq, D]` or `[D]`
/// * `expert_outs` - Expert outputs `[batch_seq, K, D]`
/// * `expert_weights` - Routing weights `[batch_seq, K]`
/// * `shared_out` - Shared expert output `[batch_seq, D]`
/// * `shared_gate_logit` - Shared expert gate logit `[batch_seq, 1]` or scalar
/// * `k` - Number of active experts (top-k)
/// * `batch_seq` - Batch * sequence length
pub fn moe_combine_mlx(
    residual: &Array,
    expert_outs: &Array,
    expert_weights: &Array,
    shared_out: &Array,
    shared_gate_logit: &Array,
    k: i32,
    batch_seq: i32,
) -> Result<Array, Exception> {
    let y = expert_outs
        .multiply(&expert_weights.reshape(&[batch_seq, k, 1])?)?
        .sum_axis(-2, false)?;

    // Shared expert with gate
    let shared_gate = mlx_rs::nn::sigmoid(shared_gate_logit)?;
    let shared_y = shared_gate.multiply(shared_out)?;

    let result = y.add(&shared_y)?;
    result.add(residual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_moe_combine_mlx_basic() {
        let dim = 64i32;
        let k = 4i32;
        let batch_seq = 1i32;

        let residual = mlx_rs::random::normal::<f32>(&[batch_seq, dim], None, None, None).unwrap();
        let expert_outs =
            mlx_rs::random::normal::<f32>(&[batch_seq, k, dim], None, None, None).unwrap();
        let expert_weights = Array::from_slice(&[0.3f32, 0.25, 0.25, 0.2], &[batch_seq, k]);
        let shared_out =
            mlx_rs::random::normal::<f32>(&[batch_seq, dim], None, None, None).unwrap();
        let shared_gate_logit = Array::from_f32(0.5);

        let result = moe_combine_mlx(
            &residual,
            &expert_outs,
            &expert_weights,
            &shared_out,
            &shared_gate_logit,
            k,
            batch_seq,
        )
        .unwrap();

        result.eval().unwrap();
        assert_eq!(result.shape(), &[batch_seq, dim]);

        // Verify no NaN
        let data: Vec<f32> = result.as_slice().to_vec();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "NaN at index {}", i);
        }
    }

    #[test]
    #[serial]
    fn test_moe_combine_mlx_batched() {
        let dim = 32i32;
        let k = 2i32;
        let batch_seq = 4i32;

        let residual = mlx_rs::random::normal::<f32>(&[batch_seq, dim], None, None, None).unwrap();
        let expert_outs =
            mlx_rs::random::normal::<f32>(&[batch_seq, k, dim], None, None, None).unwrap();
        let expert_weights =
            mlx_rs::random::normal::<f32>(&[batch_seq, k], None, None, None).unwrap();
        let shared_out =
            mlx_rs::random::normal::<f32>(&[batch_seq, dim], None, None, None).unwrap();
        let shared_gate_logit =
            mlx_rs::random::normal::<f32>(&[batch_seq, 1], None, None, None).unwrap();

        let result = moe_combine_mlx(
            &residual,
            &expert_outs,
            &expert_weights,
            &shared_out,
            &shared_gate_logit,
            k,
            batch_seq,
        )
        .unwrap();

        result.eval().unwrap();
        assert_eq!(result.shape(), &[batch_seq, dim]);
    }

}

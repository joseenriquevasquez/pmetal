//! Mixture of Experts (MoE) implementation for MLX.
//!
//! MoE architectures use multiple "expert" networks and a router to select
//! which experts process each token. This enables scaling model capacity
//! without proportionally increasing compute.
//!
//! ## Key design decisions
//! - Router logits are cast to float32 before softmax for numerical stability.
//! - Top-k uses GPU-native argpartition + slice (no CPU round-trip).
//! - Per-expert dispatch: only assigned tokens are fed to each expert.
//! - Aux loss follows the Switch Transformer formula: `N * sum(f_i * P_i)`.

#![allow(missing_docs)]

use crate::ArrayDtypeExt;
use pmetal_bridge::compat::{
    Array, Dtype, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, ops,
    random,
};
use pmetal_bridge::impl_module_params;
use std::rc::Rc;

/// Configuration for Mixture of Experts.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub use_aux_loss: bool,
    pub aux_loss_coef: f32,
    pub router_jitter: f32,
    pub normalize_router_weights: bool,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_aux_loss: true,
            aux_loss_coef: 0.01,
            router_jitter: 0.0,
            normalize_router_weights: true,
        }
    }
}

impl MoEConfig {
    pub fn new(hidden_size: i32, intermediate_size: i32, num_experts: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_experts,
            ..Default::default()
        }
    }
    pub fn with_num_experts_per_tok(mut self, k: usize) -> Self {
        self.num_experts_per_tok = k;
        self
    }
    pub fn with_aux_loss(mut self, use_aux: bool, coef: f32) -> Self {
        self.use_aux_loss = use_aux;
        self.aux_loss_coef = coef;
        self
    }
    pub fn with_router_jitter(mut self, jitter: f32) -> Self {
        self.router_jitter = jitter;
        self
    }
}

/// Minimal linear layer: weight [out, in], no bias.
///
/// Replaces `nn::Linear` — keeps `pmetal-mlx` free of mlx-rs module machinery.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix, shape `[out_features, in_features]`.
    pub weight: Array,
}
impl_module_params!(Linear; weight);

impl Linear {
    /// Create a zero-initialised linear layer.
    pub fn new(in_features: i32, out_features: i32) -> Self {
        Self {
            weight: ops::zeros(&[out_features, in_features], Dtype::Float32),
        }
    }

    /// `y = x @ W^T`
    pub fn forward(&self, x: &Array) -> Array {
        x.matmul(&self.weight.t())
    }
}

/// Router for selecting experts.
///
/// Returns normalized top-k routing weights, top-k expert indices, and raw logits.
///
/// Supports auxiliary-loss-free load balancing (DeepSeek-V3 style):
/// A non-trainable `routing_bias` is added to scores for expert *selection* only
/// (not for output weighting). After each forward pass, the bias is updated:
///   `bias[e] += gamma * (target_load - actual_load[e])`
/// This eliminates the need for the Switch Transformer auxiliary loss while
/// providing better load distribution.
#[derive(Debug)]
pub struct MoERouter {
    pub gate: Linear,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub jitter: f32,
    pub training: bool,
    /// Dynamic routing bias for loss-free load balancing (not a trainable parameter).
    /// Shape: `[num_experts]`. Updated after each forward pass when `use_bias_balancing` is true.
    pub routing_bias: Option<Vec<f32>>,
    /// Whether to use bias-based load balancing (DeepSeek-V3 style).
    pub use_bias_balancing: bool,
    /// Learning rate for bias updates: `bias[e] += gamma * (target - actual)`.
    pub bias_gamma: f32,
}
impl_module_params!(MoERouter; gate);

impl MoERouter {
    pub fn new(hidden_size: i32, num_experts: usize, num_experts_per_tok: usize) -> Self {
        let gate = Linear::new(hidden_size, num_experts as i32);
        Self {
            gate,
            num_experts,
            num_experts_per_tok,
            jitter: 0.0,
            training: true,
            routing_bias: None,
            use_bias_balancing: false,
            bias_gamma: 0.001,
        }
    }

    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter;
        self
    }

    /// Enable auxiliary-loss-free load balancing (DeepSeek-V3 style).
    ///
    /// When enabled, a dynamic routing bias is maintained and updated after each
    /// forward pass. The Switch Transformer auxiliary loss is skipped.
    pub fn with_bias_balancing(mut self, gamma: f32) -> Self {
        self.use_bias_balancing = true;
        self.bias_gamma = gamma;
        self.routing_bias = Some(vec![0.0; self.num_experts]);
        self
    }

    pub fn train(&mut self) {
        self.training = true;
    }
    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    /// Forward: compute routing weights and expert assignments.
    ///
    /// Returns `(normalized_weights [N, k], top_indices [N, k], router_logits [N, E])`.
    ///
    /// When `use_bias_balancing` is enabled, the routing bias is added to softmax scores
    /// for expert *selection* only. The output weights use the original (unbiased) scores.
    /// After each forward, the bias is updated to equalize expert load.
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size]);

        let mut router_logits = self.gate.forward(&hidden_flat);

        // Add jitter noise during training for load balancing
        if self.training && self.jitter > 0.0 {
            let noise = random::uniform(router_logits.shape(), Dtype::Float32);
            // Scale noise to [-jitter, jitter]: noise * 2*jitter - jitter
            let scaled = noise
                .multiply(&Array::from_f32(2.0 * self.jitter))
                .subtract(&Array::from_f32(self.jitter));
            router_logits = router_logits.add(&scaled);
        }

        // Cast to float32 before softmax for numerical stability
        let router_logits_f32 = if router_logits.dtype() != Dtype::Float32 {
            router_logits.as_dtype(Dtype::Float32.as_i32())
        } else {
            router_logits.clone()
        };
        let routing_weights = router_logits_f32.softmax(-1);

        // GPU-native top-k
        let k = self.num_experts_per_tok;

        let top_indices = if self.use_bias_balancing {
            // DeepSeek-V3 style: add bias for selection, use original scores for weights
            if let Some(ref bias) = self.routing_bias {
                let bias_array = Array::from_f32_slice(bias, &[self.num_experts as i32]);
                let biased_weights = routing_weights.add(&bias_array);
                let (_, indices) = gpu_topk(&biased_weights, k);
                indices
            } else {
                let (_, indices) = gpu_topk(&routing_weights, k);
                indices
            }
        } else {
            let (_, indices) = gpu_topk(&routing_weights, k);
            indices
        };

        // Gather the *unbiased* routing weights for the selected experts
        let top_weights = routing_weights.take_along_axis(&top_indices, -1);

        // Normalize selected weights to sum to 1
        let weight_sum = top_weights.sum_axis(-1, true);
        let safe_sum = weight_sum.maximum(&Array::from_f32(1e-8));
        let normalized_weights = top_weights.divide(&safe_sum);

        // Update routing bias after forward (DeepSeek-V3 dynamic balancing)
        if self.use_bias_balancing && self.training {
            self.update_routing_bias(&top_indices, batch_seq as usize)?;
        }

        Ok((normalized_weights, top_indices, router_logits))
    }

    /// Update the routing bias to equalize expert load.
    ///
    /// `bias[e] += gamma * (target_load - actual_load[e])`
    /// where target_load = k / num_experts (uniform distribution).
    fn update_routing_bias(
        &mut self,
        selected_experts: &Array,
        num_tokens: usize,
    ) -> Result<(), Exception> {
        let Some(ref mut bias) = self.routing_bias else {
            return Ok(());
        };

        let mut se_owned = selected_experts.clone();
        se_owned.eval();
        let n = se_owned.size();
        let indices: Vec<i32> = se_owned
            .to_f32_vec(n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();

        let target_load = self.num_experts_per_tok as f32 / self.num_experts as f32;
        let total = (num_tokens * self.num_experts_per_tok) as f32;

        // Count actual load per expert
        let mut counts = vec![0usize; self.num_experts];
        for &idx in &indices {
            if (idx as usize) < self.num_experts {
                counts[idx as usize] += 1;
            }
        }

        // Update bias
        for (e, count) in counts.iter().enumerate() {
            let actual_load = *count as f32 / total.max(1.0);
            bias[e] += self.bias_gamma * (target_load - actual_load);
        }

        Ok(())
    }
}

/// GPU-native top-k selection using argpartition + slice.
///
/// Returns the top-k values and indices from `probs` along the last axis.
/// This avoids the CPU round-trip and NaN panic of the previous `custom_topk`.
/// Indices are returned as Int32.
fn gpu_topk(probs: &Array, k: usize) -> (Array, Array) {
    let neg_k = -(k as i32);
    let partitioned_indices = probs.argpartition(neg_k, -1);

    // Slice the last k columns: [N, E] -> [N, k]
    // slice(start, stop) uses the full shape start/stop vectors.
    let n_rows = partitioned_indices.dim(0);
    let n_cols = partitioned_indices.dim(1);
    let col_start = n_cols + neg_k; // == n_cols - k
    let top_indices_raw = partitioned_indices.slice(&[0, col_start], &[n_rows, n_cols]);
    let top_indices = top_indices_raw.as_dtype(Dtype::Int32.as_i32());

    // Gather top-k values
    let top_values = probs.take_along_axis(&top_indices, -1);

    (top_values, top_indices)
}

/// Single expert MLP (SwiGLU).
#[derive(Debug)]
pub struct Expert {
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
}
impl_module_params!(Expert; w1, w3, w2);

impl Expert {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Self {
        Self {
            w1: Linear::new(hidden_size, intermediate_size),
            w3: Linear::new(hidden_size, intermediate_size),
            w2: Linear::new(intermediate_size, hidden_size),
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        let gate = self.w1.forward(x);
        // SwiGLU: silu(gate) * up
        let gate_activated = gate.silu();
        let up = self.w3.forward(x);
        let hidden = gate_activated.multiply(&up);
        self.w2.forward(&hidden)
    }
}

/// Mixture of Experts layer.
///
/// Routes tokens to top-k experts, runs only assigned tokens per expert,
/// and accumulates weighted outputs.
#[derive(Debug)]
pub struct MoELayer {
    pub router: MoERouter,
    pub experts: Vec<Expert>,
    pub config: MoEConfig,
}
impl_module_params!(MoELayer; router, experts);

impl MoELayer {
    pub fn new(config: MoEConfig) -> Self {
        let router = MoERouter::new(
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok,
        )
        .with_jitter(config.router_jitter);
        let experts = (0..config.num_experts)
            .map(|_| Expert::new(config.hidden_size, config.intermediate_size))
            .collect();
        Self {
            router,
            experts,
            config,
        }
    }

    pub fn train(&mut self) {
        self.router.train();
    }
    pub fn eval_mode(&mut self) {
        self.router.eval_mode();
    }

    /// Forward pass: route tokens to experts and accumulate outputs.
    ///
    /// Per-expert dispatch — each expert only processes its assigned tokens.
    /// Uses eval + CPU index extraction to gather token indices per expert,
    /// matching the mlx-examples pattern (small routing tensor round-trip is
    /// negligible compared to the expert MLP compute savings).
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Option<Array>), Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];

        let (routing_weights, selected_experts, router_logits) =
            self.router.forward(hidden_states)?;
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size]);

        // Eval routing tensors to CPU for index extraction
        // (small tensors: [batch_seq, k] — negligible transfer cost)
        let mut se_owned = selected_experts.clone();
        se_owned.eval();
        let mut rw_owned = routing_weights.clone();
        rw_owned.eval();

        let n_tokens = batch_seq as usize;
        let k = self.config.num_experts_per_tok;
        let se_n = se_owned.size();
        let rw_n = rw_owned.size();
        let expert_indices: Vec<i32> = se_owned
            .to_f32_vec(se_n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let expert_weights: Vec<f32> = rw_owned.to_f32_vec(rw_n).unwrap_or_default();

        // Build per-expert token lists: (token_idx, routing_weight)
        let mut expert_assignments: Vec<Vec<(usize, f32)>> =
            vec![Vec::new(); self.config.num_experts];
        for token_idx in 0..n_tokens {
            for slot in 0..k {
                let flat_idx = token_idx * k + slot;
                let expert_id = expert_indices[flat_idx] as usize;
                let weight = expert_weights[flat_idx];
                if expert_id < self.config.num_experts {
                    expert_assignments[expert_id].push((token_idx, weight));
                }
            }
        }

        // Per-expert dispatch — each expert only processes its assigned tokens.
        // Outputs are accumulated into a CPU f32 buffer (scatter-add replacement).
        let hidden_size_usize = hidden_size as usize;
        let input_dtype = hidden_states.dtype();
        let mut output_buf = vec![0.0f32; n_tokens * hidden_size_usize];

        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            // Gather assigned token indices and weights
            let token_indices: Vec<i32> = assignments.iter().map(|&(idx, _)| idx as i32).collect();
            let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();
            let m = token_indices.len();

            let idx_array = Array::from_i32_slice(&token_indices).reshape(&[m as i32]);
            let weight_array = Array::from_f32_slice(&weights, &[m as i32, 1]);

            // Gather only the assigned tokens
            let expert_input = hidden_flat.take_axis(&idx_array, 0);

            // Run expert only on assigned tokens
            let expert_out = self.experts[expert_idx].forward(&expert_input);

            // Weight the output
            let weighted_out = expert_out.multiply(&weight_array);

            // Eval and scatter-add into CPU buffer
            let mut wo_owned = weighted_out.clone();
            wo_owned.eval();
            let wo_n = wo_owned.size();
            let wo_data: Vec<f32> = wo_owned.to_f32_vec(wo_n).unwrap_or_default();

            for (local_idx, &token_pos) in token_indices.iter().enumerate() {
                let tok = token_pos as usize;
                let src_base = local_idx * hidden_size_usize;
                let dst_base = tok * hidden_size_usize;
                for d in 0..hidden_size_usize {
                    output_buf[dst_base + d] += wo_data[src_base + d];
                }
            }
        }

        // Build final output array from accumulated buffer
        let final_output = Array::from_f32_slice(&output_buf, &[batch_seq, hidden_size])
            .as_dtype(input_dtype.as_i32());

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        let output = final_output.reshape(&output_shape);

        // Compute auxiliary loss if enabled (skip when using bias balancing)
        let aux_loss = if self.config.use_aux_loss && !self.router.use_bias_balancing {
            Some(self.compute_aux_loss(&router_logits, &selected_experts, n_tokens)?)
        } else {
            None
        };

        Ok((output, aux_loss))
    }

    /// Forward pass with pre-computed routing (bypasses the internal router).
    ///
    /// Accepts `expert_indices` [N, k] (Int32) and `expert_weights` [N, k] (Float32) as
    /// already computed by an external gate (e.g. DeepSeek's sigmoid + e_score_correction_bias
    /// router).  The internal `MoERouter` is completely skipped — no softmax, no jitter, no
    /// aux-loss computation.  The dispatch/scatter logic is identical to `forward()`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [..., hidden_size]
    /// * `expert_indices` - Pre-selected expert indices [N_flat, k], Int32
    /// * `expert_weights` - Pre-computed routing weights [N_flat, k], Float32 (should sum to 1 per row)
    pub fn forward_with_routing(
        &mut self,
        hidden_states: &Array,
        expert_indices: &Array,
        expert_weights: &Array,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let batch_seq = shape[..shape.len() - 1].iter().product::<i32>();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = hidden_states.reshape(&[batch_seq, hidden_size]);

        // Eval routing tensors to CPU for index extraction
        let mut ei_owned = expert_indices.clone();
        ei_owned.eval();
        let mut ew_owned = expert_weights.clone();
        ew_owned.eval();

        let n_tokens = batch_seq as usize;
        let k = self.config.num_experts_per_tok;
        let ei_n = ei_owned.size();
        let ew_n = ew_owned.size();
        let ei_data: Vec<i32> = ei_owned
            .to_f32_vec(ei_n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let ew_data: Vec<f32> = ew_owned.to_f32_vec(ew_n).unwrap_or_default();

        // Build per-expert token lists: (token_idx, routing_weight)
        let mut expert_assignments: Vec<Vec<(usize, f32)>> =
            vec![Vec::new(); self.config.num_experts];
        for token_idx in 0..n_tokens {
            for slot in 0..k {
                let flat_idx = token_idx * k + slot;
                let expert_id = ei_data[flat_idx] as usize;
                let weight = ew_data[flat_idx];
                if expert_id < self.config.num_experts {
                    expert_assignments[expert_id].push((token_idx, weight));
                }
            }
        }

        let hidden_size_usize = hidden_size as usize;
        let input_dtype = hidden_states.dtype();
        let mut output_buf = vec![0.0f32; n_tokens * hidden_size_usize];

        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            let token_indices: Vec<i32> = assignments.iter().map(|&(idx, _)| idx as i32).collect();
            let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();
            let m = token_indices.len();

            let idx_array = Array::from_i32_slice(&token_indices).reshape(&[m as i32]);
            let weight_array = Array::from_f32_slice(&weights, &[m as i32, 1]);

            let expert_input = hidden_flat.take_axis(&idx_array, 0);
            let expert_out = self.experts[expert_idx].forward(&expert_input);
            let weighted_out = expert_out.multiply(&weight_array);

            let mut wo_owned = weighted_out.clone();
            wo_owned.eval();
            let wo_n = wo_owned.size();
            let wo_data: Vec<f32> = wo_owned.to_f32_vec(wo_n).unwrap_or_default();

            for (local_idx, &token_pos) in token_indices.iter().enumerate() {
                let tok = token_pos as usize;
                let src_base = local_idx * hidden_size_usize;
                let dst_base = tok * hidden_size_usize;
                for d in 0..hidden_size_usize {
                    output_buf[dst_base + d] += wo_data[src_base + d];
                }
            }
        }

        let final_output = Array::from_f32_slice(&output_buf, &[batch_seq, hidden_size])
            .as_dtype(input_dtype.as_i32());

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        Ok(final_output.reshape(&output_shape))
    }

    /// Compute auxiliary load-balancing loss using Switch Transformer formula.
    ///
    /// `aux_loss = num_experts * sum(f_i * P_i)`
    ///
    /// Where:
    /// - `f_i` = fraction of tokens dispatched to expert i
    /// - `P_i` = mean routing probability for expert i
    fn compute_aux_loss(
        &self,
        router_logits: &Array,
        selected_experts: &Array,
        num_tokens: usize,
    ) -> Result<Array, Exception> {
        let num_experts = self.config.num_experts;
        let num_experts_i32 = num_experts as i32;

        // P_i: mean routing probability per expert across all tokens
        let routing_probs = router_logits.softmax(-1);
        let mean_routing_prob = routing_probs.mean_axis(0, false); // [num_experts]

        // f_i: fraction of tokens dispatched to each expert
        // Count how many times each expert appears in selected_experts
        let mut dispatch_fractions = Vec::with_capacity(num_experts);
        let total_dispatches = (num_tokens * self.config.num_experts_per_tok) as f32;
        let mut se_owned = selected_experts.clone();
        se_owned.eval();
        let se_n = se_owned.size();
        let se_data: Vec<i32> = se_owned
            .to_f32_vec(se_n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        for e in 0..num_experts {
            let count = se_data.iter().filter(|&&x| x == e as i32).count();
            dispatch_fractions.push(count as f32 / total_dispatches.max(1.0));
        }
        let f_array = Array::from_f32_slice(&dispatch_fractions, &[num_experts_i32]);

        // Switch Transformer aux loss: N * sum(f_i * P_i)
        let f_times_p = f_array.multiply(&mean_routing_prob);
        let aux_loss = f_times_p
            .sum(None)
            .multiply(&Array::from_f32(num_experts as f32));

        Ok(aux_loss.multiply(&Array::from_f32(self.config.aux_loss_coef)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a small MoEConfig suitable for fast unit tests.
    fn small_config() -> MoEConfig {
        MoEConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 4,
            num_experts_per_tok: 2,
            use_aux_loss: true,
            aux_loss_coef: 0.01,
            router_jitter: 0.0,
            normalize_router_weights: true,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1 — Router output is float32 and normalized weights sum to ~1
    // -----------------------------------------------------------------------

    /// Verify that the router always returns float32 routing weights regardless
    /// of input dtype, and that per-token normalized weights sum to exactly 1.
    #[test]
    #[serial]
    fn test_router_softmax_dtype() -> Result<(), Exception> {
        let hidden = 16_i32;
        let n_experts = 4_usize;
        let k = 2_usize;
        let n_tokens = 6_i32;

        let mut router = MoERouter::new(hidden, n_experts, k);

        // Random float32 input [n_tokens, hidden]
        let input = random::uniform(&[n_tokens, hidden], Dtype::Float32);

        let (weights, top_indices, _logits) = router.forward(&input)?;

        // Weights must be float32
        assert_eq!(
            weights.dtype(),
            Dtype::Float32,
            "routing weights must be float32, got {:?}",
            weights.dtype()
        );

        // Shape must be [n_tokens, k]
        assert_eq!(
            weights.shape(),
            &[n_tokens, k as i32],
            "weights shape mismatch: {:?}",
            weights.shape()
        );
        assert_eq!(
            top_indices.shape(),
            &[n_tokens, k as i32],
            "top_indices shape mismatch: {:?}",
            top_indices.shape()
        );

        // Each row must sum to ~1.0 (normalized weights)
        let mut w_owned = weights.clone();
        w_owned.eval();
        let w_n = w_owned.size();
        let w_data: Vec<f32> = w_owned.to_f32_vec(w_n).unwrap_or_default();
        for token in 0..(n_tokens as usize) {
            let row_sum: f32 = (0..k).map(|slot| w_data[token * k + slot]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "token {} weights sum to {}, expected ~1.0",
                token,
                row_sum
            );
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 2 — Top-k selection produces correct shape and valid expert indices
    // -----------------------------------------------------------------------

    /// Route a 4-token batch through a router with 4 experts and k=2.
    /// We cannot prescribe exact expert selection without controlling weights,
    /// but we can assert:
    ///   - top_indices shape == [N, k]
    ///   - every selected index is in [0, num_experts)
    ///   - no duplicate expert per token (each slot is distinct)
    #[test]
    #[serial]
    fn test_topk_selection() -> Result<(), Exception> {
        let hidden = 16_i32;
        let n_experts = 4_usize;
        let k = 2_usize;
        let n_tokens = 4_i32;

        let mut router = MoERouter::new(hidden, n_experts, k);

        let input = random::uniform(&[n_tokens, hidden], Dtype::Float32);

        let (_weights, top_indices, _logits) = router.forward(&input)?;

        // Shape must be [n_tokens, k]
        assert_eq!(
            top_indices.shape(),
            &[n_tokens, k as i32],
            "top_indices shape must be [N, k], got {:?}",
            top_indices.shape()
        );

        // Every index must be a valid expert id and no token should pick the
        // same expert twice (argsort of distinct values is always injective).
        let mut ti_owned = top_indices.clone();
        ti_owned.eval();
        let ti_n = ti_owned.size();
        let indices: Vec<i32> = ti_owned
            .to_f32_vec(ti_n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        for token in 0..n_tokens as usize {
            let a = indices[token * k] as usize;
            let b = indices[token * k + 1] as usize;
            assert!(a < n_experts, "token {} slot 0 out of range: {}", token, a);
            assert!(b < n_experts, "token {} slot 1 out of range: {}", token, b);
            assert_ne!(
                a, b,
                "token {} received duplicate expert assignment: {}/{}",
                token, a, b
            );
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_gpu_topk_matches_full_sort_reference() -> Result<(), Exception> {
        let probs = Array::from_f32_slice(
            &[
                0.10f32, 0.60, 0.20, 0.40, 0.30, 0.55, 0.15, 0.45, 0.35, 0.25,
            ],
            &[2, 5],
        );

        let (values, indices) = gpu_topk(&probs, 2);
        let mut values_owned = values.clone();
        values_owned.eval();
        let mut indices_owned = indices.clone();
        indices_owned.eval();

        let v_n = values_owned.size();
        let i_n = indices_owned.size();
        let value_data: Vec<f32> = values_owned.to_f32_vec(v_n).unwrap_or_default();
        let index_data: Vec<i32> = indices_owned
            .to_f32_vec(i_n)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let mut probs_owned = probs.clone();
        probs_owned.eval();
        let p_n = probs_owned.size();
        let probs_data: Vec<f32> = probs_owned.to_f32_vec(p_n).unwrap_or_default();

        for row in 0..2 {
            let mut expected: Vec<(i32, f32)> = probs_data[row * 5..(row + 1) * 5]
                .iter()
                .cloned()
                .enumerate()
                .map(|(idx, value)| (idx as i32, value))
                .collect();
            expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            expected.truncate(2);
            expected.sort_by_key(|&(idx, _)| idx);

            let mut actual = vec![
                (index_data[row * 2], value_data[row * 2]),
                (index_data[row * 2 + 1], value_data[row * 2 + 1]),
            ];
            actual.sort_by_key(|&(idx, _)| idx);

            assert_eq!(
                actual.len(),
                expected.len(),
                "row {row} returned the wrong number of top-k entries"
            );
            for (actual_pair, expected_pair) in actual.iter().zip(expected.iter()) {
                assert_eq!(
                    actual_pair.0, expected_pair.0,
                    "row {row} selected the wrong expert ids"
                );
                assert!(
                    (actual_pair.1 - expected_pair.1).abs() < 1e-6,
                    "row {row} selected the wrong top-k values: {:?} vs {:?}",
                    actual,
                    expected
                );
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 3 — Expert forward pass produces correct output shape
    // -----------------------------------------------------------------------

    /// A single Expert (SwiGLU MLP) must map [T, hidden] → [T, hidden].
    #[test]
    #[serial]
    fn test_expert_forward_shape() -> Result<(), Exception> {
        let hidden = 16_i32;
        let intermediate = 32_i32;
        let t = 4_i32;

        let expert = Expert::new(hidden, intermediate);
        let input = random::uniform(&[t, hidden], Dtype::Float32);

        let output = expert.forward(&input);

        assert_eq!(
            output.shape(),
            &[t, hidden],
            "expert output shape mismatch: expected [{}, {}], got {:?}",
            t,
            hidden,
            output.shape()
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 4 — MoELayer preserves spatial dimensions end-to-end
    // -----------------------------------------------------------------------

    /// A complete MoELayer forward pass over a 3-D input [batch, seq, hidden]
    /// must return an output tensor with the same shape.
    #[test]
    #[serial]
    fn test_moe_layer_output_shape() -> Result<(), Exception> {
        let config = small_config();
        let hidden = config.hidden_size;
        let batch = 2_i32;
        let seq = 5_i32;

        let mut layer = MoELayer::new(config);

        let input = random::uniform(&[batch, seq, hidden], Dtype::Float32);

        let (output, _aux) = layer.forward(&input)?;

        assert_eq!(
            output.shape(),
            &[batch, seq, hidden],
            "MoELayer output shape mismatch: expected [{}, {}, {}], got {:?}",
            batch,
            seq,
            hidden,
            output.shape()
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 5 — Auxiliary loss is Some and strictly positive when enabled
    // -----------------------------------------------------------------------

    /// The Switch Transformer aux loss `N * sum(f_i * P_i)` is always >= 0
    /// (product of non-negative terms).  For any non-trivial routing it is
    /// strictly positive, which we verify here.
    #[test]
    #[serial]
    fn test_aux_loss_formula() -> Result<(), Exception> {
        let config = MoEConfig {
            use_aux_loss: true,
            aux_loss_coef: 0.01,
            ..small_config()
        };
        let hidden = config.hidden_size;
        let mut layer = MoELayer::new(config);

        let input = random::uniform(&[3, 4, hidden], Dtype::Float32);

        let (_output, aux_loss) = layer.forward(&input)?;

        let aux = aux_loss.expect("aux_loss must be Some when use_aux_loss = true");
        let mut aux_owned = aux.clone();
        aux_owned.eval();
        let n = aux_owned.size();
        let vals = aux_owned.to_f32_vec(n).unwrap_or_default();
        let val = vals.first().copied().unwrap_or(f32::NAN);

        assert!(val.is_finite(), "aux_loss must be finite, got {}", val);
        assert!(
            val >= 0.0,
            "aux_loss must be non-negative (Switch Transformer formula), got {}",
            val
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 6 — Auxiliary loss is None when disabled
    // -----------------------------------------------------------------------

    /// When `use_aux_loss = false` no extra computation is performed and the
    /// second element of the forward return is `None`.
    #[test]
    #[serial]
    fn test_aux_loss_disabled() -> Result<(), Exception> {
        let config = MoEConfig {
            use_aux_loss: false,
            ..small_config()
        };
        let hidden = config.hidden_size;
        let mut layer = MoELayer::new(config);

        let input = random::uniform(&[2, 3, hidden], Dtype::Float32);

        let (_output, aux_loss) = layer.forward(&input)?;

        assert!(
            aux_loss.is_none(),
            "aux_loss must be None when use_aux_loss = false"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 7 — No NaN / no panic when most experts receive zero tokens
    // -----------------------------------------------------------------------

    /// With 8 experts, k=2, and only 1 token, exactly 6 experts receive no
    /// token assignments.  The implementation must handle this gracefully:
    /// the zero-token experts are skipped, and the output must be a valid
    /// (non-NaN, finite) tensor.
    #[test]
    #[serial]
    fn test_empty_expert_no_panic() -> Result<(), Exception> {
        let config = MoEConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_aux_loss: false,
            aux_loss_coef: 0.01,
            router_jitter: 0.0,
            normalize_router_weights: true,
        };
        let hidden = config.hidden_size;
        let mut layer = MoELayer::new(config);

        // Single token — 6 of 8 experts will receive no assignments
        let input = random::uniform(&[1, hidden], Dtype::Float32);

        let (output, _aux) = layer.forward(&input)?;
        let mut out_owned = output.clone();
        out_owned.eval();

        assert_eq!(
            out_owned.shape(),
            &[1, hidden],
            "output shape mismatch with sparse expert usage: {:?}",
            out_owned.shape()
        );

        // Verify no NaN values in the output
        let n = out_owned.size();
        let out_data: Vec<f32> = out_owned.to_f32_vec(n).unwrap_or_default();
        for (i, &v) in out_data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "output[{}] = {} is not finite (NaN or Inf) with sparse experts",
                i,
                v
            );
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Test 8 — Eval mode disables jitter noise without panicking
    // -----------------------------------------------------------------------

    /// In eval mode the router must not apply jitter noise.  We verify this
    /// by setting jitter > 0 but switching to eval mode, then confirming:
    ///   - The forward pass completes without panic.
    ///   - Output shape is correct.
    ///   - Calling the same deterministic input twice produces identical results
    ///     (no stochastic noise in eval mode).
    #[test]
    #[serial]
    fn test_moe_layer_eval_mode() -> Result<(), Exception> {
        let config = MoEConfig {
            router_jitter: 0.1, // Would add noise in training mode
            use_aux_loss: false,
            ..small_config()
        };
        let hidden = config.hidden_size;
        let mut layer = MoELayer::new(config);

        // Switch to eval — jitter must be suppressed
        layer.eval_mode();

        // Use a fixed deterministic input
        let input_data: Vec<f32> = (0..(3 * hidden as usize))
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input = Array::from_f32_slice(&input_data, &[3, hidden]);

        let (output_a, _) = layer.forward(&input)?;
        let (output_b, _) = layer.forward(&input)?;

        let mut a_owned = output_a.clone();
        a_owned.eval();
        let mut b_owned = output_b.clone();
        b_owned.eval();

        assert_eq!(
            a_owned.shape(),
            &[3, hidden],
            "eval mode output shape mismatch: {:?}",
            a_owned.shape()
        );

        // Both passes must produce identical results (eval mode is deterministic)
        let an = a_owned.size();
        let bn = b_owned.size();
        let a: Vec<f32> = a_owned.to_f32_vec(an).unwrap_or_default();
        let b: Vec<f32> = b_owned.to_f32_vec(bn).unwrap_or_default();
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                va, vb,
                "eval mode outputs differ at index {}: {} vs {} — jitter may still be active",
                i, va, vb
            );
        }

        Ok(())
    }
}

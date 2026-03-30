//! Integration tests for training with Metal FlashAttention.
//!
//! These tests verify the complete training flow works correctly with our
//! efficient O(n) backward pass implementation.

#![allow(clippy::expect_fun_call)]

use pmetal_bridge::compat::{Array, Dtype, random};
use pmetal_mlx::kernels::{
    FusedAttentionConfig, compute_attention_gradients, differentiable_attention,
    init_training_context, with_training_mode,
};

fn random_tensor(shape: &[i32]) -> Array {
    random::uniform_range(0.0, 1.0, shape, Dtype::Float32)
}

#[test]
fn test_training_attention_forward_backward() {
    // Initialize training context
    let ctx = init_training_context().expect("Failed to init training context");

    // Enable training mode
    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.enable_training();
    }

    let batch = 1;
    let n_heads = 4;
    let n_kv_heads = 4;
    let seq_len = 32;
    let head_dim = 64;

    let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
    let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
    let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

    let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

    // Forward pass
    let mut output = differentiable_attention(0, &queries, &keys, &values, &config)
        .expect("Forward pass failed");

    output.eval();
    assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);

    // Simulate upstream gradient
    let d_output = random_tensor(&[batch, n_heads, seq_len, head_dim]);

    // Backward pass
    let (mut d_q, mut d_k, mut d_v) =
        compute_attention_gradients(0, &d_output).expect("Backward pass failed");

    d_q.eval();
    d_k.eval();
    d_v.eval();

    // Verify gradient shapes
    assert_eq!(d_q.shape(), &[batch, n_heads, seq_len, head_dim]);
    assert_eq!(d_k.shape(), &[batch, n_kv_heads, seq_len, head_dim]);
    assert_eq!(d_v.shape(), &[batch, n_kv_heads, seq_len, head_dim]);

    // Verify gradients are non-zero (basic sanity check)
    let mut d_q_sum = d_q.abs().sum(None);
    let mut d_k_sum = d_k.abs().sum(None);
    let mut d_v_sum = d_v.abs().sum(None);

    d_q_sum.eval();
    d_k_sum.eval();
    d_v_sum.eval();

    assert!(d_q_sum.item::<f32>() > 0.0, "d_q should be non-zero");
    assert!(d_k_sum.item::<f32>() > 0.0, "d_k should be non-zero");
    assert!(d_v_sum.item::<f32>() > 0.0, "d_v should be non-zero");

    // Cleanup
    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.disable_training();
    }
}

#[test]
fn test_with_training_mode_helper() {
    // Use head_dim=64 which has the implemented backward kernels
    let result = with_training_mode(|| {
        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 4;
        let seq_len = 16;
        let head_dim = 64; // Use 64 to match implemented kernels

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

        // Forward pass
        let mut output = differentiable_attention(0, &queries, &keys, &values, &config)?;
        output.eval();

        // Backward pass
        let d_output = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let (mut d_q, _d_k, _d_v) = compute_attention_gradients(0, &d_output)?;
        d_q.eval();

        let mut loss = output.sum(None);
        Ok(loss.item::<f32>())
    });

    assert!(
        result.is_ok(),
        "Training mode helper should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_multi_layer_training() {
    // Test training with multiple attention layers (like a real model)
    let ctx = init_training_context().expect("Failed to init training context");

    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.enable_training();
    }

    let batch = 1;
    let n_heads = 4;
    let n_kv_heads = 2; // GQA
    let seq_len = 32;
    let head_dim = 64;
    let num_layers = 4;

    let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

    // Simulate forward pass through multiple layers
    let mut outputs = Vec::new();
    for layer_id in 0..num_layers {
        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let mut output = differentiable_attention(layer_id, &queries, &keys, &values, &config)
            .expect(&format!("Forward pass failed for layer {}", layer_id));
        output.eval();
        outputs.push(output);
    }

    // Verify all outputs have correct shape
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(
            output.shape(),
            &[batch, n_heads, seq_len, head_dim],
            "Layer {} output shape mismatch",
            i
        );
    }

    // Simulate backward pass (in reverse order, like real backprop)
    for layer_id in (0..num_layers).rev() {
        let d_output = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let (mut d_q, mut d_k, mut d_v) = compute_attention_gradients(layer_id, &d_output)
            .expect(&format!("Backward pass failed for layer {}", layer_id));

        d_q.eval();
        d_k.eval();
        d_v.eval();

        // Verify gradient shapes (accounting for GQA)
        assert_eq!(d_q.shape(), &[batch, n_heads, seq_len, head_dim]);
        assert_eq!(d_k.shape(), &[batch, n_kv_heads, seq_len, head_dim]);
        assert_eq!(d_v.shape(), &[batch, n_kv_heads, seq_len, head_dim]);
    }

    // Cleanup
    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.disable_training();
    }
}

#[test]
fn test_inference_mode_no_cache() {
    // Test that inference mode doesn't store caches
    let ctx = init_training_context().expect("Failed to init training context");

    // Keep training mode disabled
    {
        let ctx_guard = ctx.lock().unwrap();
        assert!(!ctx_guard.is_training(), "Should start in inference mode");
    }

    let batch = 1;
    let n_heads = 4;
    let n_kv_heads = 4;
    let seq_len = 16;
    let head_dim = 32;

    let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
    let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
    let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

    let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

    // Forward pass in inference mode
    let mut output = differentiable_attention(0, &queries, &keys, &values, &config)
        .expect("Forward pass failed");

    output.eval();
    assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);

    // Verify no cache was stored
    {
        let ctx_guard = ctx.lock().unwrap();
        assert!(
            !ctx_guard.has_cache(0),
            "Cache should not be stored in inference mode"
        );
    }
}

#[test]
fn test_gqa_gradients() {
    // Test Grouped Query Attention gradient computation
    let ctx = init_training_context().expect("Failed to init training context");

    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.enable_training();
    }

    let batch = 2;
    let n_heads = 8;
    let n_kv_heads = 2; // 4 groups
    let seq_len = 64;
    let head_dim = 64;

    let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
    let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
    let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

    let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

    // Forward pass
    let mut output = differentiable_attention(0, &queries, &keys, &values, &config)
        .expect("Forward pass failed");
    output.eval();

    // Output should have full head count
    assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);

    // Backward pass
    let d_output = random_tensor(&[batch, n_heads, seq_len, head_dim]);
    let (mut d_q, mut d_k, mut d_v) =
        compute_attention_gradients(0, &d_output).expect("Backward pass failed");

    d_q.eval();
    d_k.eval();
    d_v.eval();

    // Gradient shapes should match input shapes
    assert_eq!(d_q.shape(), &[batch, n_heads, seq_len, head_dim]);
    assert_eq!(d_k.shape(), &[batch, n_kv_heads, seq_len, head_dim]);
    assert_eq!(d_v.shape(), &[batch, n_kv_heads, seq_len, head_dim]);

    // Cleanup
    {
        let mut ctx_guard = ctx.lock().unwrap();
        ctx_guard.disable_training();
    }
}

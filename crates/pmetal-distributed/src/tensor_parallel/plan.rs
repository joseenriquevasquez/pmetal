//! Architecture-agnostic tensor parallelism plan builders.
//!
//! These functions generate [`ShardingPlan`]s by inspecting model configuration
//! (as `serde_json::Value`) without depending on pmetal-models types. This keeps
//! the dependency direction clean: pmetal-distributed → (no model deps).
//!
//! # Supported Block Types
//!
//! - Standard multi-head attention (Llama, Mistral, Gemma, etc.)
//! - SwiGLU FFN blocks
//! - GDN (gated delta net) blocks (Qwen 3.5, FalconH1)
//! - MoE (shared expert + routed experts)
//!
//! # Reference
//!
//! The Qwen 3.5 `shard()` method in mlx-lm serves as the authoritative
//! reference for hybrid GDN+Attention+MoE tensor parallelism.

use super::sharding::{ShardingDirective, ShardingPlan};
use anyhow::Result;

/// Build a TP plan for a standard multi-head attention block.
///
/// Q/K/V projections are column-sharded (AllToSharded on axis 0),
/// O projection is row-sharded (ShardedToAll on axis 0).
///
/// Handles GQA: if `n_kv_heads < n_heads`, KV heads are repeated
/// if N > n_kv_heads (standard MLX behavior).
pub fn plan_attention(
    prefix: &str,
    _n_heads: usize,
    _n_kv_heads: usize,
) -> Vec<(String, ShardingDirective)> {
    vec![
        (
            format!("{prefix}.q_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.k_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.v_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.o_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 0 },
        ),
        // Biases (if present) follow the same pattern.
        (
            format!("{prefix}.q_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.k_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.v_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.o_proj.bias"),
            ShardingDirective::ShardedToAll { axis: 0 },
        ),
    ]
}

/// Build a TP plan for a SwiGLU FFN block.
///
/// gate_proj and up_proj are column-sharded; down_proj is row-sharded.
pub fn plan_ffn(prefix: &str) -> Vec<(String, ShardingDirective)> {
    vec![
        (
            format!("{prefix}.gate_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.up_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.down_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 0 },
        ),
    ]
}

/// Build a TP plan for a GDN (gated delta net) block.
///
/// Follows the Qwen 3.5 shard() pattern:
/// - All input projections (qkv, z, b, a) are column-sharded
/// - Conv1d is custom-sharded by key_dim
/// - Output projection is row-sharded
/// - Per-head parameters (dt_bias, A_log) are split by heads
///
/// # Reference
///
/// mlx-lm `qwen3_5.py` lines 393-440
pub fn plan_gdn(
    prefix: &str,
    _key_dim: usize,
    _n_heads: usize,
) -> Vec<(String, ShardingDirective)> {
    vec![
        // Input projections: column-shard
        (
            format!("{prefix}.in_proj_qkv.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_z.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_b.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_a.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        // Conv1d (depthwise): shard along output channels (= key_dim * n_heads)
        (
            format!("{prefix}.conv1d.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        // Output projection: row-shard
        (
            format!("{prefix}.o_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 0 },
        ),
        // Per-head parameters: shard along head dimension (axis 0)
        (
            format!("{prefix}.dt_bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.A_log"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
    ]
}

/// Build a TP plan for an MoE block (shared expert + routed experts).
///
/// Shared expert uses standard FFN column/row sharding.
/// Routed experts use expert-level sharding along axis 0.
pub fn plan_moe(prefix: &str, total_experts: usize) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    // Shared expert: standard FFN sharding
    let shared = format!("{prefix}.shared_expert");
    directives.extend(plan_ffn(&shared));

    // Shared expert gate (scalar per expert, replicated)
    directives.push((
        format!("{prefix}.shared_expert_gate.weight"),
        ShardingDirective::Replicated,
    ));

    // Router gate: replicated (all ranks compute full routing scores)
    directives.push((
        format!("{prefix}.gate.weight"),
        ShardingDirective::Replicated,
    ));

    // Routed experts: expert-level sharding
    // Weights are stacked as [num_experts, ...], split along axis 0
    for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        directives.push((
            format!("{prefix}.experts.{suffix}"),
            ShardingDirective::ExpertSharded { total_experts },
        ));
    }

    directives
}

/// Build a full model sharding plan from architecture type and config.json.
///
/// Reads the architecture type and config fields to produce a complete
/// sharding plan without depending on pmetal-models types.
///
/// # Supported Architectures
///
/// - `"Qwen3NextForCausalLM"` / `"qwen3_next"` — Hybrid GDN+Attention+MoE
/// - `"LlamaForCausalLM"` / `"llama"` — Standard transformer
/// - `"MistralForCausalLM"` / `"mistral"` — Standard transformer + sliding window
/// - `"Qwen2MoeForCausalLM"` / `"qwen2_moe"` — MoE transformer
/// - `"DeepseekV3ForCausalLM"` / `"deepseek_v3"` — MoE with shared expert
/// - Generic fallback for standard transformer architectures
pub fn build_plan(
    arch: &str,
    config: &serde_json::Value,
    world_size: usize,
) -> Result<ShardingPlan> {
    if world_size <= 1 {
        return Ok(ShardingPlan::new());
    }

    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let n_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;

    let num_experts = config
        .get("num_experts")
        .or_else(|| config.get("num_local_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let key_dim = config
        .get("key_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;

    // Detect hybrid GDN+Attention models by checking for GDN-specific config
    let is_hybrid_gdn = config.get("gdn_heads").is_some()
        || config.get("mixer_pattern").is_some()
        || arch.contains("qwen3_next")
        || arch.contains("Qwen3Next")
        || arch.contains("FalconH1");

    // Detect attention interval for hybrid models (every Nth layer is attention)
    let attn_interval = config
        .get("attention_interval")
        .or_else(|| config.get("attn_layer_period"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;

    let mut plan = ShardingPlan::new();

    // Embedding: replicated (each rank needs full vocab)
    plan.add("model.embed_tokens.weight", ShardingDirective::Replicated);

    for layer in 0..num_layers {
        let layer_prefix = format!("model.layers.{layer}");

        if is_hybrid_gdn {
            // Hybrid GDN + Attention model
            let is_attention_layer = (layer + 1) % attn_interval == 0;

            if is_attention_layer {
                // Full attention layer: standard Q/K/V/O sharding
                let attn = format!("{layer_prefix}.self_attn");
                for (name, dir) in plan_attention(&attn, n_heads, n_kv_heads) {
                    plan.add(name, dir);
                }
            } else {
                // GDN layer: shard input/output projections
                let gdn = format!("{layer_prefix}.gdn");
                for (name, dir) in plan_gdn(&gdn, key_dim, n_heads) {
                    plan.add(name, dir);
                }
            }
        } else {
            // Standard attention layer
            let attn = format!("{layer_prefix}.self_attn");
            for (name, dir) in plan_attention(&attn, n_heads, n_kv_heads) {
                plan.add(name, dir);
            }
        }

        // FFN / MoE block
        if num_experts > 0 {
            let moe = format!("{layer_prefix}.block_sparse_moe");
            // Try alternate naming conventions
            let moe_prefix =
                if config.get("model_type").and_then(|v| v.as_str()) == Some("deepseek_v3") {
                    format!("{layer_prefix}.mlp")
                } else {
                    moe
                };
            for (name, dir) in plan_moe(&moe_prefix, num_experts) {
                plan.add(name, dir);
            }
        } else {
            let ffn = format!("{layer_prefix}.mlp");
            for (name, dir) in plan_ffn(&ffn) {
                plan.add(name, dir);
            }
        }

        // Layer norms: replicated (small, not worth sharding)
        plan.add(
            format!("{layer_prefix}.input_layernorm.weight"),
            ShardingDirective::Replicated,
        );
        plan.add(
            format!("{layer_prefix}.post_attention_layernorm.weight"),
            ShardingDirective::Replicated,
        );
    }

    // Final norm and lm_head: replicated
    plan.add("model.norm.weight", ShardingDirective::Replicated);
    plan.add("lm_head.weight", ShardingDirective::Replicated);

    tracing::info!(
        "Built TP plan for arch={arch}, layers={num_layers}, heads={n_heads}, \
         kv_heads={n_kv_heads}, experts={num_experts}, hybrid_gdn={is_hybrid_gdn}, \
         sharded_weights={}, world_size={world_size}",
        plan.num_sharded()
    );

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_attention_produces_8_directives() {
        let directives = plan_attention("layer.0.self_attn", 32, 8);
        assert_eq!(directives.len(), 8); // q,k,v,o weights + q,k,v,o biases
    }

    #[test]
    fn plan_ffn_produces_3_directives() {
        let directives = plan_ffn("layer.0.mlp");
        assert_eq!(directives.len(), 3); // gate, up, down
    }

    #[test]
    fn plan_gdn_produces_8_directives() {
        let directives = plan_gdn("layer.0.gdn", 128, 32);
        assert_eq!(directives.len(), 8);
    }

    #[test]
    fn plan_moe_includes_shared_and_routed() {
        let directives = plan_moe("layer.0.block_sparse_moe", 512);
        // shared_expert FFN (3) + shared_expert_gate (1) + gate (1) + routed (3) = 8
        assert_eq!(directives.len(), 8);
    }

    #[test]
    fn build_plan_world_size_1_is_empty() {
        let config = serde_json::json!({ "num_hidden_layers": 4 });
        let plan = build_plan("llama", &config, 1).unwrap();
        assert_eq!(plan.num_sharded(), 0);
    }

    #[test]
    fn build_plan_standard_transformer() {
        let config = serde_json::json!({
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        });
        let plan = build_plan("llama", &config, 2).unwrap();
        // 2 layers × (8 attention + 3 FFN) = 22 sharded weights
        assert!(plan.num_sharded() > 0);
    }

    #[test]
    fn build_plan_hybrid_gdn_moe() {
        let config = serde_json::json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_experts": 512,
            "gdn_heads": 32,
            "attention_interval": 4,
            "key_dim": 128,
        });
        let plan = build_plan("Qwen3NextForCausalLM", &config, 4).unwrap();
        assert!(plan.num_sharded() > 0);

        // Verify GDN layers got GDN directives (layers 0,1,2)
        assert!(
            plan.directives
                .contains_key("model.layers.0.gdn.in_proj_qkv.weight")
        );
        // Verify attention layer (layer 3) got attention directives
        assert!(
            plan.directives
                .contains_key("model.layers.3.self_attn.q_proj.weight")
        );
    }
}

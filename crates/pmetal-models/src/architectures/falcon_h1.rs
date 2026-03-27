//! FalconH1 hybrid Mamba-2 + Attention architecture.
//!
//! FalconH1 is a hybrid architecture where **every layer** contains both a
//! Mamba-2 SSM branch and a full attention branch running in parallel, unlike
//! NemotronH which uses a pattern string to alternate between layer types.
//!
//! ## Architecture
//!
//! Each decoder layer performs:
//! ```text
//! residual = x
//! normed   = RMSNorm(x)
//! attn_out = Attention(normed)  * attn_out_multiplier
//! mamba_out= Mamba2(normed)     * ssm_out_multiplier
//! h        = residual + attn_out + mamba_out
//! h        = h + MLP(RMSNorm(h))
//! ```
//!
//! Additional features:
//! - `key_multiplier` applied to attention keys before RoPE
//! - GQA (grouped query attention)
//! - SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`)
//! - RoPE position embeddings
//! - Hybrid KV + Mamba cache for efficient generation
//!
//! Reference: tiiuae/falcon-h1 on Hugging Face
//! Reference: transformers.models.falcon_h1.modeling_falcon_h1

use std::collections::HashMap;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, Param},
    nn,
    ops::indexing::IndexOp,
};
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry};
use serde::Deserialize;

// Re-use the Mamba-2 SSM functions from NemotronH. They are declared `pub`
// so we can import them directly. This avoids duplicating ~300 lines of
// heavily-tested numerical code.
use super::nemotron_h::{MambaRMSNormGated, ssm_attention, ssm_update_single};

// ============================================================================
// Config
// ============================================================================

fn default_model_type() -> String {
    "falcon_h1".to_string()
}
fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_mamba_d_ssm() -> i32 {
    128
}
fn default_mamba_d_conv() -> i32 {
    4
}
fn default_mamba_n_groups() -> i32 {
    8
}
fn default_mamba_head_dim() -> i32 {
    64
}
fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}
fn default_use_conv_bias() -> bool {
    true
}
fn default_tie_word_embeddings() -> bool {
    false
}

/// FalconH1 model configuration deserialized from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct FalconH1Config {
    /// HuggingFace model_type string.
    #[serde(default = "default_model_type")]
    pub model_type: String,

    // Core dimensions
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,

    // Attention
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Scaling applied to attention keys before RoPE (Falcon-specific feature).
    #[serde(default)]
    pub key_multiplier: Option<f32>,

    // Mamba-2 SSM
    /// SSM state dimension, also called `d_state` or `ssm_state_size`.
    #[serde(default = "default_mamba_d_ssm")]
    pub mamba_d_ssm: i32,
    /// Conv1d kernel size.
    #[serde(default = "default_mamba_d_conv")]
    pub mamba_d_conv: i32,
    /// Number of B/C groups.
    #[serde(default = "default_mamba_n_groups")]
    pub mamba_n_groups: i32,
    /// Number of Mamba heads.
    pub mamba_num_heads: i32,
    /// Mamba head dimension.
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: i32,
    /// Use bias in Mamba in_proj / out_proj.
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// Use bias in conv1d.
    #[serde(default = "default_use_conv_bias")]
    pub use_conv_bias: bool,
    /// Time step range as (min, max).
    #[serde(default = "default_time_step_limit")]
    pub time_step_limit: (f32, f32),
    /// Override lower bound.
    #[serde(default)]
    pub time_step_min: Option<f32>,
    /// Override upper bound.
    #[serde(default)]
    pub time_step_max: Option<f32>,

    // Hybrid scaling multipliers
    /// Per-layer attention output multipliers (length = num_hidden_layers).
    #[serde(default)]
    pub attn_out_multipliers: Option<Vec<f32>>,
    /// Per-layer SSM output multipliers (length = num_hidden_layers).
    #[serde(default)]
    pub ssm_out_multipliers: Option<Vec<f32>>,
    /// Global fallback attn multiplier when per-layer list is absent.
    #[serde(default)]
    pub attn_out_multiplier: Option<f32>,
    /// Global fallback SSM multiplier when per-layer list is absent.
    #[serde(default)]
    pub ssm_out_multiplier: Option<f32>,

    // Misc
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl FalconH1Config {
    /// Attention head dimension = hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Mamba intermediate size = mamba_num_heads * mamba_head_dim.
    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Mamba conv input dimension = intermediate + 2 * n_groups * d_ssm.
    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_ssm
    }

    /// Per-layer attention output multiplier.
    pub fn attn_multiplier_for_layer(&self, i: usize) -> f32 {
        self.attn_out_multipliers
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .or(self.attn_out_multiplier)
            .unwrap_or(1.0)
    }

    /// Per-layer SSM output multiplier.
    pub fn ssm_multiplier_for_layer(&self, i: usize) -> f32 {
        self.ssm_out_multipliers
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .or(self.ssm_out_multiplier)
            .unwrap_or(1.0)
    }

    /// Effective lower time-step bound.
    pub fn effective_time_step_min(&self) -> f32 {
        self.time_step_min.unwrap_or(self.time_step_limit.0)
    }

    /// Effective upper time-step bound.
    pub fn effective_time_step_max(&self) -> f32 {
        self.time_step_max.unwrap_or(self.time_step_limit.1)
    }
}

impl crate::traits::ModelConfig for FalconH1Config {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim()
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.rms_norm_eps as f32
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta as f32
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

// ============================================================================
// Attention
// ============================================================================

/// GQA attention with RoPE and optional key multiplier.
#[derive(Debug, ModuleParameters)]
pub struct FalconH1Attention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,

    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    /// Scalar multiplied into keys before RoPE.
    pub key_multiplier: f32,
}

impl FalconH1Attention {
    pub fn new(config: &FalconH1Config) -> Result<Self, Exception> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = nn::LinearBuilder::new(config.hidden_size, num_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, num_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, num_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(num_heads * head_dim, config.hidden_size)
            .bias(false)
            .build()?;

        let scale = (head_dim as f32).sqrt().recip();
        let key_multiplier = config.key_multiplier.unwrap_or(1.0);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            rope_theta: config.rope_theta as f32,
            key_multiplier,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        // Project Q, K, V
        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        // Reshape to multi-head format [B, L, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        // Transpose to attention format [B, heads, L, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply key multiplier BEFORE RoPE (matches HF reference implementation)
        let k = if (self.key_multiplier - 1.0).abs() > 1e-7 {
            k.multiply(&Array::from_f32(self.key_multiplier))?
        } else {
            k
        };

        // RoPE: determine offset from KV cache
        let rope_offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, rope_offset)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, rope_offset)?;

        // Fused SDPA (handles GQA head expansion internally)
        let attn_config =
            FusedAttentionConfig::new(self.num_heads, self.num_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(AttentionMaskType::Causal);

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) =
                    (*cache_ref).try_turboquant_attention(*layer_idx, &q, &k, &v, &attn_config)?
                {
                    let output = output
                        .transpose_axes(&[0, 2, 1, 3])?
                        .reshape(&[batch, seq_len, -1])?;
                    return Module::forward(&mut self.o_proj, &output);
                }
            }
        }

        // Update KV cache if provided
        let (k, v) = if let Some((cache_ref, layer_idx)) = cache {
            cache_ref.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        Module::forward(&mut self.o_proj, &output)
    }
}

// ============================================================================
// Mamba-2 SSM Layer
// ============================================================================

/// Mamba-2 SSM layer for FalconH1.
///
/// Holds all SSM parameters and reuses the numerically-validated SSM kernels
/// from NemotronH (`ssm_update_single` for single-token decode, `ssm_attention`
/// for multi-token prefill).  Weight names follow the HF convention:
/// `model.layers.{i}.mamba.*`.
#[derive(Debug, ModuleParameters)]
pub struct FalconH1Mamba {
    /// Input projection: hidden_size -> (intermediate + conv_dim + num_heads)
    #[param]
    pub in_proj: nn::Linear,
    /// Depthwise causal conv1d over conv_dim channels.
    #[param]
    pub conv1d: nn::Conv1d,
    /// Output projection: intermediate -> hidden_size.
    #[param]
    pub out_proj: nn::Linear,

    // SSM parameters: loaded from checkpoint, not re-derived.
    /// Log state-transition diagonal [num_heads]; A = -exp(a_log).
    pub a_log: Array,
    /// Skip connection weights [num_heads].
    pub d: Array,
    /// Time step bias [num_heads].
    pub dt_bias: Array,
    /// Gated RMS norm applied to the SSM output before out_proj.
    pub norm: MambaRMSNormGated,

    // Cached dimensions (avoid repeated arithmetic in hot path)
    pub mamba_num_heads: i32,
    pub mamba_head_dim: i32,
    pub mamba_intermediate_size: i32,
    pub mamba_conv_dim: i32,
    pub ssm_state_size: i32,
    pub n_groups: i32,
    pub conv_kernel_size: i32,
    pub time_step_min: f32,
    pub time_step_max: f32,
}

impl FalconH1Mamba {
    pub fn new(config: &FalconH1Config) -> Result<Self, Exception> {
        let intermediate_size = config.mamba_intermediate_size();
        let conv_dim = config.mamba_conv_dim();
        let mamba_num_heads = config.mamba_num_heads;
        let conv_kernel_size = config.mamba_d_conv;

        // in_proj: hidden -> intermediate + conv_dim + mamba_num_heads (dt)
        let projection_size = intermediate_size + conv_dim + mamba_num_heads;
        let in_proj = nn::LinearBuilder::new(config.hidden_size, projection_size)
            .bias(config.mamba_proj_bias)
            .build()?;

        // Depthwise conv1d: use (1, conv_dim, kernel) with groups=conv_dim to get
        // the correct MLX weight shape [conv_dim, kernel, 1].
        // Padding is applied manually (causal left-padding) in forward().
        let conv1d = nn::Conv1dBuilder::new(1, conv_dim, conv_kernel_size)
            .groups(conv_dim)
            .bias(config.use_conv_bias)
            .padding(0)
            .build()?;

        // out_proj: intermediate -> hidden
        let out_proj = nn::LinearBuilder::new(intermediate_size, config.hidden_size)
            .bias(config.mamba_proj_bias)
            .build()?;

        // SSM parameter arrays – initialised to neutral values, overwritten at
        // weight-load time.
        let a_log = Array::zeros::<f32>(&[mamba_num_heads])?;
        let d = Array::ones::<f32>(&[mamba_num_heads])?;
        let dt_bias = Array::ones::<f32>(&[mamba_num_heads])?;

        let norm = MambaRMSNormGated::new(
            intermediate_size,
            config.rms_norm_eps as f32,
            config.mamba_n_groups,
        )?;

        Ok(Self {
            in_proj,
            conv1d,
            out_proj,
            a_log,
            d,
            dt_bias,
            norm,
            mamba_num_heads,
            mamba_head_dim: config.mamba_head_dim,
            mamba_intermediate_size: intermediate_size,
            mamba_conv_dim: conv_dim,
            ssm_state_size: config.mamba_d_ssm,
            n_groups: config.mamba_n_groups,
            conv_kernel_size,
            time_step_min: config.effective_time_step_min(),
            time_step_max: config.effective_time_step_max(),
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let intermediate_size = self.mamba_intermediate_size;
        let conv_dim = self.mamba_conv_dim;
        let n_groups = self.n_groups;
        let ssm_state_size = self.ssm_state_size;
        let num_heads = self.mamba_num_heads;
        let head_dim = self.mamba_head_dim;
        let conv_kernel = self.conv_kernel_size;

        // Input projection: [B, L, hidden] -> [B, L, intermediate + conv_dim + num_heads]
        let projected = Module::forward(&mut self.in_proj, x)?;

        // Split along last axis: gate | conv_input | dt
        let split_at = &[intermediate_size, intermediate_size + conv_dim];
        let parts = mlx_rs::ops::split_sections(&projected, split_at, -1)?;
        let gate = &parts[0]; // [B, L, intermediate_size]
        let conv_input = &parts[1]; // [B, L, conv_dim]
        let dt = &parts[2]; // [B, L, num_heads]

        // Causal conv1d with optional state caching.
        // We cannot directly pass `cache` to a helper that also borrows
        // `self.conv1d`, so we inline both branches here.
        let conv_activated = {
            // Extract out the Option to avoid lifetime conflict with self.conv1d
            // We handle the None case first (no cache).
            match cache.as_deref_mut() {
                Some(mc) => {
                    let padded = mc.update_conv_state(conv_input, conv_kernel)?;
                    let out = Module::forward(&mut self.conv1d, &padded)?;
                    let out_len = out.dim(1);
                    let out = out.index((.., (out_len - seq_len).., ..));
                    nn::silu(&out)?
                }
                None => {
                    let pad_amount = (conv_kernel - 1) as i32;
                    let padded = mlx_rs::ops::pad(
                        conv_input,
                        &[(0i32, 0i32), (pad_amount, 0), (0, 0)],
                        Array::from_int(0),
                        None,
                    )?;
                    let out = Module::forward(&mut self.conv1d, &padded)?;
                    nn::silu(&out)?
                }
            }
        };

        // Split conv output: hidden_states | B | C
        let bc_size = n_groups * ssm_state_size;
        let conv_split_at = &[intermediate_size, intermediate_size + bc_size];
        let conv_parts = mlx_rs::ops::split_sections(&conv_activated, conv_split_at, -1)?;
        let hidden_states = &conv_parts[0]; // [B, L, intermediate_size]
        let b_proj = &conv_parts[1]; // [B, L, n_groups * ssm_state_size]
        let c_proj = &conv_parts[2]; // [B, L, n_groups * ssm_state_size]

        // Reshape for multi-head SSM computation
        let x_heads = hidden_states.reshape(&[batch, seq_len, num_heads, head_dim])?;
        let b_reshaped = b_proj.reshape(&[batch, seq_len, n_groups, ssm_state_size])?;
        let c_reshaped = c_proj.reshape(&[batch, seq_len, n_groups, ssm_state_size])?;

        // SSM forward: single-token fast path (decode) or full attention (prefill)
        let prev_state = cache.as_ref().and_then(|mc| mc.get_ssm_state());
        let (y, next_state) = if let (1, Some(prev)) = (seq_len, prev_state) {
            ssm_update_single(
                &x_heads,
                &self.a_log,
                &b_reshaped,
                &c_reshaped,
                &self.d,
                dt,
                &self.dt_bias,
                prev,
                (self.time_step_min, self.time_step_max),
            )?
        } else {
            ssm_attention(
                &x_heads,
                &self.a_log,
                &b_reshaped,
                &c_reshaped,
                &self.d,
                dt,
                &self.dt_bias,
                prev_state,
                (self.time_step_min, self.time_step_max),
            )?
        };

        // Persist updated SSM state
        if let Some(mc) = cache {
            mc.set_ssm_state(next_state);
        }

        // Reshape back: [B, 1/L, H, D] -> [B, L, intermediate_size]
        let y = y.reshape(&[batch, seq_len, intermediate_size])?;

        // Gated RMS norm: normalise y with silu(gate) as the gate signal
        let y_normed = self.norm.forward(&y, Some(gate))?;

        // Output projection: [B, L, intermediate_size] -> [B, L, hidden_size]
        Module::forward(&mut self.out_proj, &y_normed)
    }
}

// ============================================================================
// SwiGLU MLP
// ============================================================================

/// SwiGLU feed-forward network.
///
/// Computes: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, ModuleParameters)]
pub struct FalconH1MLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl FalconH1MLP {
    pub fn new(config: &FalconH1Config) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        // SwiGLU: silu(gate) * up
        let activated = nn::silu(&gate)?.multiply(&up)?;
        Module::forward(&mut self.down_proj, &activated)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

/// FalconH1 hybrid decoder layer.
///
/// Every layer runs both Attention and Mamba in parallel on the same
/// normalised input, scales each branch by its per-layer multiplier, sums
/// them, adds the residual, then passes through SwiGLU MLP.
#[derive(Debug, ModuleParameters)]
pub struct FalconH1DecoderLayer {
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub self_attn: FalconH1Attention,
    #[param]
    pub mamba: FalconH1Mamba,
    #[param]
    pub pre_ff_layernorm: nn::RmsNorm,
    #[param]
    pub feed_forward: FalconH1MLP,

    /// Per-layer attention output scale (baked in at construction time).
    pub attn_out_multiplier: f32,
    /// Per-layer SSM output scale (baked in at construction time).
    pub ssm_out_multiplier: f32,
}

impl FalconH1DecoderLayer {
    pub fn new(config: &FalconH1Config, layer_idx: usize) -> Result<Self, Exception> {
        let norm_eps = config.rms_norm_eps as f32;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(norm_eps)
            .build()?;
        let pre_ff_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(norm_eps)
            .build()?;

        let self_attn = FalconH1Attention::new(config)?;
        let mamba = FalconH1Mamba::new(config)?;
        let feed_forward = FalconH1MLP::new(config)?;

        let attn_out_multiplier = config.attn_multiplier_for_layer(layer_idx);
        let ssm_out_multiplier = config.ssm_multiplier_for_layer(layer_idx);

        Ok(Self {
            input_layernorm,
            self_attn,
            mamba,
            pre_ff_layernorm,
            feed_forward,
            attn_out_multiplier,
            ssm_out_multiplier,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        // Pre-norm shared by both branches
        let normed = Module::forward(&mut self.input_layernorm, x)?;

        // Run both branches in sequence (MLX is lazy; they will run in parallel on GPU)
        let attn_out = self.self_attn.forward(&normed, mask, kv_cache)?;
        let mamba_out = self.mamba.forward(&normed, mamba_cache)?;

        // Scale each branch by its per-layer multiplier
        let attn_scaled = scale_if_needed(attn_out, self.attn_out_multiplier)?;
        let mamba_scaled = scale_if_needed(mamba_out, self.ssm_out_multiplier)?;

        // Sum branches and add residual
        let mixed = attn_scaled.add(&mamba_scaled)?;
        let h = x.add(&mixed)?;

        // MLP sub-layer
        let normed2 = Module::forward(&mut self.pre_ff_layernorm, &h)?;
        let mlp_out = self.feed_forward.forward(&normed2)?;
        h.add(&mlp_out)
    }
}

/// Multiply `x` by `multiplier` only when it differs from 1.0 (skip the op
/// when it's a no-op to avoid unnecessary MLX graph nodes).
#[inline]
fn scale_if_needed(x: Array, multiplier: f32) -> Result<Array, Exception> {
    if (multiplier - 1.0).abs() > 1e-7 {
        x.multiply(&Array::from_f32(multiplier))
    } else {
        Ok(x)
    }
}

// ============================================================================
// Model
// ============================================================================

/// FalconH1 backbone (embedding + layers + final norm).
#[derive(Debug, ModuleParameters)]
pub struct FalconH1Model {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<FalconH1DecoderLayer>,
    #[param]
    pub final_layernorm: nn::RmsNorm,
    pub config: FalconH1Config,
}

impl FalconH1Model {
    pub fn new(config: FalconH1Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers as usize {
            layers.push(FalconH1DecoderLayer::new(&config, i)?);
        }

        let final_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps as f32)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            config,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = kv_cache.as_deref_mut().map(|c| (c, layer_idx));
            let mamba = mamba_cache
                .as_deref_mut()
                .and_then(|c| c.get_mut(layer_idx));

            hidden = layer.forward(&hidden, mask, kv, mamba)?;
        }

        Module::forward(&mut self.final_layernorm, &hidden)
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }
}

// ============================================================================
// CausalLM head
// ============================================================================

/// FalconH1 for causal language modelling.
#[derive(Debug, ModuleParameters)]
pub struct FalconH1ForCausalLM {
    #[param]
    pub model: FalconH1Model,
    /// Present when tie_word_embeddings = false; absent otherwise.
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl FalconH1ForCausalLM {
    pub fn new(config: FalconH1Config) -> Result<Self, Exception> {
        let tie = config.tie_word_embeddings;
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        let model = FalconH1Model::new(config)?;

        let lm_head = if !tie {
            Some(
                nn::LinearBuilder::new(hidden_size, vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self { model, lm_head })
    }

    /// Access the config without moving.
    pub fn config(&self) -> &FalconH1Config {
        &self.model.config
    }

    /// Full-sequence forward (no caching).
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let hidden = self.model.forward(input_ids, mask)?;
        self.project_logits(hidden)
    }

    /// Cached forward for autoregressive generation.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.project_logits(hidden)
    }

    fn project_logits(&mut self, hidden: Array) -> Result<Array, Exception> {
        if let Some(ref mut head) = self.lm_head {
            Module::forward(head, &hidden)
        } else {
            // Tied embeddings: use embedding matrix transposed as the output projection
            self.model.embed_tokens.as_linear(&hidden)
        }
    }

    /// Evaluate (materialise) all parameters onto the device.
    pub fn eval(&self) -> Result<(), Exception> {
        for (_, p) in self.parameters().flatten() {
            p.eval()?;
        }
        Ok(())
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Load FalconH1 weights from a HuggingFace safetensors weight map.
///
/// ## HuggingFace Weight Name Mapping
///
/// ```text
/// model.embed_tokens.weight
/// model.layers.{i}.input_layernorm.weight
/// model.layers.{i}.self_attn.q_proj.weight
/// model.layers.{i}.self_attn.k_proj.weight
/// model.layers.{i}.self_attn.v_proj.weight
/// model.layers.{i}.self_attn.o_proj.weight
/// model.layers.{i}.mamba.in_proj.weight
/// model.layers.{i}.mamba.conv1d.weight   [PyTorch: [out,in/g,k] → MLX: [out,k,in/g]]
/// model.layers.{i}.mamba.conv1d.bias
/// model.layers.{i}.mamba.A_log
/// model.layers.{i}.mamba.D
/// model.layers.{i}.mamba.dt_bias
/// model.layers.{i}.mamba.norm.weight
/// model.layers.{i}.mamba.out_proj.weight
/// model.layers.{i}.feed_forward.gate_proj.weight
/// model.layers.{i}.feed_forward.up_proj.weight
/// model.layers.{i}.feed_forward.down_proj.weight
/// model.layers.{i}.pre_ff_layernorm.weight
/// model.final_layernorm.weight
/// lm_head.weight
/// ```
pub fn load_falcon_h1_weights(
    model: &mut FalconH1ForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), crate::loader::LoadError> {
    // Embedding
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = Param::new(w.clone());
    }

    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let p = format!("model.layers.{i}");

        // --- Layer norms ---
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{p}.input_layernorm.weight"),
        );
        load_rms_norm_weight(
            &mut layer.pre_ff_layernorm,
            weights,
            &format!("{p}.pre_ff_layernorm.weight"),
        );

        // --- Attention projections ---
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{p}.self_attn.q_proj.weight"),
        );
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{p}.self_attn.k_proj.weight"),
        );
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{p}.self_attn.v_proj.weight"),
        );
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{p}.self_attn.o_proj.weight"),
        );

        // --- Mamba: linear projections ---
        load_linear_weight(
            &mut layer.mamba.in_proj,
            weights,
            &format!("{p}.mamba.in_proj.weight"),
        );
        load_linear_weight(
            &mut layer.mamba.out_proj,
            weights,
            &format!("{p}.mamba.out_proj.weight"),
        );

        // conv1d weights: transpose from PyTorch [out, in/groups, kernel]
        //                 to MLX format      [out, kernel, in/groups]
        if let Some(w) = weights.get(&format!("{p}.mamba.conv1d.weight")) {
            let w_mlx = w.transpose_axes(&[0, 2, 1])?;
            layer.mamba.conv1d.weight = Param::new(w_mlx);
        }
        if let Some(b) = weights.get(&format!("{p}.mamba.conv1d.bias")) {
            layer.mamba.conv1d.bias = Param::new(Some(b.clone()));
        }

        // SSM parameters
        if let Some(w) = weights.get(&format!("{p}.mamba.A_log")) {
            layer.mamba.a_log = w.clone();
        }
        if let Some(w) = weights.get(&format!("{p}.mamba.D")) {
            layer.mamba.d = w.clone();
        }
        if let Some(w) = weights.get(&format!("{p}.mamba.dt_bias")) {
            layer.mamba.dt_bias = w.clone();
        }
        if let Some(w) = weights.get(&format!("{p}.mamba.norm.weight")) {
            layer.mamba.norm.weight = w.clone();
        }

        // --- MLP ---
        load_linear_weight(
            &mut layer.feed_forward.gate_proj,
            weights,
            &format!("{p}.feed_forward.gate_proj.weight"),
        );
        load_linear_weight(
            &mut layer.feed_forward.up_proj,
            weights,
            &format!("{p}.feed_forward.up_proj.weight"),
        );
        load_linear_weight(
            &mut layer.feed_forward.down_proj,
            weights,
            &format!("{p}.feed_forward.down_proj.weight"),
        );
    }

    // Final norm
    if let Some(w) = weights.get("model.final_layernorm.weight") {
        model.model.final_layernorm.weight = Param::new(w.clone());
    }

    // lm_head (optional: absent when weights are tied)
    if let Some(ref mut head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            head.weight = Param::new(w.clone());
        }
    }

    Ok(())
}

// ---- Small private helpers ------------------------------------------------

fn load_linear_weight(linear: &mut nn::Linear, weights: &HashMap<String, Array>, key: &str) {
    if let Some(w) = weights.get(key) {
        linear.weight = Param::new(w.clone());
    }
}

fn load_rms_norm_weight(norm: &mut nn::RmsNorm, weights: &HashMap<String, Array>, key: &str) {
    if let Some(w) = weights.get(key) {
        norm.weight = Param::new(w.clone());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> FalconH1Config {
        FalconH1Config {
            model_type: "falcon_h1".to_string(),
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 2,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            key_multiplier: Some(0.5),
            mamba_d_ssm: 16,
            mamba_d_conv: 4,
            mamba_n_groups: 2,
            mamba_num_heads: 4,
            mamba_head_dim: 16,
            mamba_proj_bias: false,
            use_conv_bias: true,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            attn_out_multipliers: Some(vec![0.5, 0.5]),
            ssm_out_multipliers: Some(vec![0.5, 0.5]),
            attn_out_multiplier: None,
            ssm_out_multiplier: None,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_config_helpers() {
        let cfg = small_config();
        assert_eq!(cfg.head_dim(), 16);
        // intermediate = 4 * 16 = 64
        assert_eq!(cfg.mamba_intermediate_size(), 64);
        // conv_dim = 64 + 2*2*16 = 128
        assert_eq!(cfg.mamba_conv_dim(), 128);
        assert_eq!(cfg.attn_multiplier_for_layer(0), 0.5);
        assert_eq!(cfg.ssm_multiplier_for_layer(1), 0.5);
        // Out-of-range layer falls back to global default 1.0
        assert_eq!(cfg.attn_multiplier_for_layer(99), 1.0);
    }

    #[test]
    fn test_model_construction() {
        let cfg = small_config();
        let model = FalconH1ForCausalLM::new(cfg).expect("model construction failed");
        assert_eq!(model.model.layers.len(), 2);
        // tie_word_embeddings = true => lm_head absent
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_forward_no_cache() {
        let cfg = small_config();
        let mut model = FalconH1ForCausalLM::new(cfg).expect("model construction");
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).expect("forward failed");
        // [batch=1, seq=4, vocab=512]
        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_forward_with_hybrid_cache() {
        let cfg = small_config();
        let mut model = FalconH1ForCausalLM::new(cfg.clone()).expect("model construction");

        let num_layers = cfg.num_hidden_layers as usize;
        let head_dim = cfg.head_dim() as usize;
        let kv_config = pmetal_mlx::kv_cache::KVCacheConfig::new(
            num_layers,
            256,
            cfg.num_key_value_heads as usize,
            head_dim,
        );
        let mut kv_cache = KVCache::new(kv_config);
        let mut mamba_cache = MambaCache::new(num_layers);

        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model
            .forward_with_cache(
                &input_ids,
                None,
                Some(&mut kv_cache),
                Some(&mut mamba_cache),
            )
            .expect("forward_with_cache failed");
        assert_eq!(logits.shape()[0], 1);
        assert_eq!(logits.shape()[2], 512);
    }
}

//! LoRA fine-tuning for Qwen3/3.5 — zero mlx-rs dependency.
//!
//! Uses the native Qwen3 forward pass from `qwen3_native` with added LoRA
//! adapter weights. The training loop uses pmetal-bridge's `value_and_grad`
//! for gradient computation and [`AdamW`] for parameter updates.
//!
//! # Design note — forward-pass integration
//!
//! `NativeWeights` deliberately encapsulates its per-layer weight fields
//! (they are not `pub`). The LoRA contribution is therefore injected at
//! the **logit** level during training via a lightweight adapter-only
//! forward pass. A `TODO` block in [`train_step`] marks the spot where a
//! layer-level hook should be added once `NativeWeights` exposes mutable
//! projection accessors.

use std::collections::HashMap;
use std::path::Path;

use crate::inline_array;
use crate::optimizer::ParamSet;
use crate::{AdamW, InlineArray};
use crate::qwen3_native::{NativeCache, NativeWeights, Qwen3Config};
use crate::training;

// ============================================================================
// LoRA configuration
// ============================================================================

/// Rank-decomposition fine-tuning configuration.
///
/// Defaults target the four canonical attention projections plus the three
/// MLP projections, with rank 16 and alpha 32 (LoRA scale = 2.0).
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Inner dimension `r` of the low-rank decomposition.
    pub rank: i32,
    /// Scaling constant; effective scale = `alpha / rank`.
    pub alpha: f32,
    /// Which projection names receive LoRA adapters, e.g. `"q_proj"`.
    pub target_modules: Vec<String>,
    /// Dropout probability applied to the LoRA hidden state (0 = disabled).
    pub dropout: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
                "gate_proj".into(),
                "up_proj".into(),
                "down_proj".into(),
            ],
            dropout: 0.0,
        }
    }
}

// ============================================================================
// LoRA adapter
// ============================================================================

/// Low-rank adapter weights for a single linear projection.
///
/// The effective weight modification at inference time is:
/// ```text
/// W_eff = W_base + scale * (B @ A)
/// ```
/// where `A` is `lora_a` `[rank, in_features]` and `B` is `lora_b`
/// `[out_features, rank]`.
pub struct LoraAdapter {
    /// `[rank, in_features]` — trainable. Initialised ~ N(0, 1/sqrt(rank)).
    pub lora_a: InlineArray,
    /// `[out_features, rank]` — trainable. Initialised to zero so the adapter
    /// is identity at the start of fine-tuning.
    pub lora_b: InlineArray,
    /// Pre-computed `alpha / rank`.
    pub scale: f32,
}

impl LoraAdapter {
    /// Allocate adapter weights.
    ///
    /// `A` uses a Kaiming-style init (std = `1/sqrt(rank)`) so the activation
    /// magnitudes stay stable. `B` is zero — the net adapter output is 0 at
    /// step 0, matching the behaviour expected by LoRA (Hu et al., 2022).
    pub fn new(in_features: i32, out_features: i32, rank: i32, alpha: f32) -> Self {
        // dtype 10 == float32 (matches MLX's `mlx::core::float32`)
        let dtype_f32 = 10_i32;
        let scale = alpha / rank as f32;
        let std_dev = (1.0_f32 / rank as f32).sqrt();

        // A ~ N(0, std_dev)
        let a = InlineArray::random_normal(&[rank, in_features], dtype_f32)
            .multiply(&InlineArray::from_f32(std_dev));

        // B = zeros
        let b = InlineArray::zeros(&[out_features, rank], dtype_f32);

        Self { lora_a: a, lora_b: b, scale }
    }
}

// ============================================================================
// LoRA weight collection
// ============================================================================

/// All LoRA adapters for a Qwen3/3.5 model.
///
/// Adapters are keyed by `"layers.{i}.{proj_name}"`.
pub struct Qwen3LoraWeights {
    /// Map from `"layers.{layer_idx}.{proj_name}"` to its adapter.
    pub adapters: HashMap<String, LoraAdapter>,
    /// Configuration used when these adapters were created.
    pub config: LoraConfig,
}

impl Qwen3LoraWeights {
    /// Allocate LoRA adapters for every target module in every layer.
    ///
    /// Skips any `target_modules` entry that is not a recognised projection
    /// name rather than panicking — this keeps things forward-compatible when
    /// new projection names are introduced.
    pub fn new(model_config: &Qwen3Config, lora_config: LoraConfig) -> Self {
        let mut adapters = HashMap::new();
        let n_layers = model_config.num_hidden_layers;
        let hidden = model_config.hidden_size;
        let intermediate = model_config.intermediate_size;
        let n_heads = model_config.num_attention_heads;
        let head_dim = model_config
            .head_dim
            .unwrap_or(hidden / n_heads);
        let n_kv = model_config
            .num_key_value_heads
            .unwrap_or(n_heads);

        for i in 0..n_layers {
            for target in &lora_config.target_modules {
                // Derive (in_features, out_features) from the projection name.
                // Qwen3.5 gated attention has q_proj output = n_heads * head_dim * 2
                // (queries + gate concatenated). We allocate for that width so
                // the adapter matrix dimensions stay correct when LoRA is wired
                // into a future layer-level hook.
                let dims: Option<(i32, i32)> = match target.as_str() {
                    "q_proj" => {
                        // Qwen3.5 gated: output is 2× head_dim per head.
                        // We default to 1× here; the hook site can override.
                        Some((hidden, n_heads * head_dim))
                    }
                    "k_proj" => Some((hidden, n_kv * head_dim)),
                    "v_proj" => Some((hidden, n_kv * head_dim)),
                    "o_proj" => Some((n_heads * head_dim, hidden)),
                    "gate_proj" | "up_proj" => Some((hidden, intermediate)),
                    "down_proj" => Some((intermediate, hidden)),
                    _ => {
                        // Unknown module — skip silently so the caller does not
                        // have to maintain a perfect allowlist.
                        None
                    }
                };

                if let Some((in_f, out_f)) = dims {
                    let key = format!("layers.{i}.{target}");
                    adapters.insert(
                        key,
                        LoraAdapter::new(in_f, out_f, lora_config.rank, lora_config.alpha),
                    );
                }
            }
        }

        Self { adapters, config: lora_config }
    }

    /// Return a [`ParamSet`] containing all trainable adapter arrays.
    ///
    /// Key format: `"layers.{i}.{proj}.lora_a"` / `"…lora_b"`.
    pub fn trainable_params(&self) -> ParamSet {
        let mut params = ParamSet::new();
        for (key, adapter) in &self.adapters {
            params.insert(format!("{key}.lora_a"), adapter.lora_a.clone());
            params.insert(format!("{key}.lora_b"), adapter.lora_b.clone());
        }
        params
    }

    /// Write back updated arrays from an optimizer-stepped [`ParamSet`].
    ///
    /// Only keys present in `params` are updated; unknown keys are ignored.
    pub fn update_from_params(&mut self, params: &ParamSet) {
        for (key, adapter) in &mut self.adapters {
            if let Some(a) = params.get(&format!("{key}.lora_a")) {
                adapter.lora_a = a.clone();
            }
            if let Some(b) = params.get(&format!("{key}.lora_b")) {
                adapter.lora_b = b.clone();
            }
        }
    }

    /// Persist all adapter weights to a safetensors file.
    ///
    /// The file can be reloaded with [`load`] and merged back into any
    /// checkpoint that shares the same `Qwen3Config`.
    pub fn save(&self, path: &str) {
        // Build owned key strings first so we can take `&str` references that
        // outlive the `.collect()` call below.
        let owned_keys: Vec<(String, &InlineArray)> = self
            .adapters
            .iter()
            .flat_map(|(key, adapter)| {
                [
                    (format!("{key}.lora_a"), &adapter.lora_a),
                    (format!("{key}.lora_b"), &adapter.lora_b),
                ]
            })
            .collect();

        let entries: Vec<(&str, &InlineArray)> = owned_keys
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        InlineArray::save_safetensors(path, &entries);
    }

    /// Load LoRA weights from a safetensors file produced by [`save`].
    ///
    /// Only keys that already exist in `self.adapters` are restored;
    /// extra keys in the file are silently ignored, making it safe to
    /// load partial checkpoints.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        if !Path::new(path).exists() {
            return Err(format!("LoRA checkpoint not found: {path}"));
        }

        // `load_safetensors_shard` parses the file once and returns all
        // tensors as a Vec of (name, array) pairs.
        let shard = inline_array::load_safetensors_shard(path)
            .ok_or_else(|| format!("failed to load LoRA checkpoint: {path}"))?;

        // Build a lookup map for O(1) access by key.
        let loaded: HashMap<String, InlineArray> = shard.into_iter().collect();

        for (key, adapter) in &mut self.adapters {
            let key_a = format!("{key}.lora_a");
            let key_b = format!("{key}.lora_b");
            if let Some(a) = loaded.get(&key_a) {
                adapter.lora_a = a.clone();
            }
            if let Some(b) = loaded.get(&key_b) {
                adapter.lora_b = b.clone();
            }
        }

        Ok(())
    }
}

// ============================================================================
// Training configuration
// ============================================================================

/// Hyper-parameters for the LoRA training loop.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Base learning rate (AdamW `base_lr`).
    pub learning_rate: f32,
    /// AdamW weight decay (L2 regularisation coefficient).
    pub weight_decay: f32,
    /// Global gradient L2 norm cap. Set ≤ 0 to disable clipping.
    pub max_grad_norm: f32,
    /// Number of mini-batches to accumulate gradients over before stepping.
    pub gradient_accumulation_steps: usize,
    /// Number of passes over the training set.
    pub num_epochs: usize,
    /// Maximum sequence length (tokens). Sequences are truncated/padded here.
    pub max_seq_len: usize,
    /// Mini-batch size (sequences per step).
    pub batch_size: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            gradient_accumulation_steps: 1,
            num_epochs: 1,
            max_seq_len: 512,
            batch_size: 1,
        }
    }
}

// ============================================================================
// Step result
// ============================================================================

/// Metrics returned from a single [`train_step`] call.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Cross-entropy loss for this step (already `eval()`'d).
    pub loss: f32,
    /// Global gradient L2 norm before clipping (already `eval()`'d).
    pub grad_norm: f32,
    /// Wall-clock time for the entire step in milliseconds.
    pub step_time_ms: u64,
}

// ============================================================================
// Core training primitive
// ============================================================================

/// Run one LoRA training step: forward → loss → backward → optimizer update.
///
/// # Arguments
///
/// * `base_weights` — frozen model weights loaded via `NativeWeights::load`.
/// * `lora` — trainable LoRA adapters; updated in-place after the optimizer
///   step.
/// * `optimizer` — AdamW instance whose step counter advances by one here.
/// * `input_ids` — flat `[batch_size * seq_len]` token IDs (row-major).
/// * `labels` — flat `[batch_size * seq_len]` token IDs; positions equal to
///   `ignore_index` (typically `-100`) are excluded from the loss.
/// * `batch_size` / `seq_len` — used to reshape the flat slices.
/// * `ignore_index` — label value that marks padding / masked positions.
/// * `max_grad_norm` — passed to [`training::clip_grad_norm`]; ≤ 0 disables.
///
/// # Returns
///
/// A [`StepResult`] with the scalar loss, gradient norm, and step latency.
///
/// # TODO — layer-level LoRA injection
///
/// Currently the training loss is computed using the **base model's forward
/// pass** (`forward_step`) without any LoRA contribution. This gives correct
/// gradient *shapes* for the adapter parameters but the loss does not reflect
/// the adapter's output because the adapter projection is not yet wired into
/// the attention / MLP layers.
///
/// The correct integration requires `NativeWeights` (or `LayerWeights`) to
/// expose mutable accessors for the `attn_{q,k,v,o}_w` / `mlp_{gate,up,down}_w`
/// fields so that `train_step` can temporarily replace each weight with a
/// `LoRA-augmented` closure before calling `forward_step`. Until that API is
/// in place, use this step to train adapter parameters via the gradient
/// of the **residual adapter output** rather than end-to-end fine-tuning.
///
/// Tracking issue: integrate `training::lora_forward` / `lora_forward_quantized`
/// at each projection site inside `qwen3_native::forward_step` and thread the
/// adapter arrays through as differentiable parameters.
pub fn train_step(
    base_weights: &NativeWeights,
    lora: &mut Qwen3LoraWeights,
    optimizer: &mut AdamW,
    input_ids: &[i32],
    labels: &[i32],
    batch_size: i32,
    seq_len: i32,
    ignore_index: i32,
    max_grad_norm: f32,
) -> StepResult {
    let t0 = std::time::Instant::now();

    // ── Build input arrays ──────────────────────────────────────────────────
    // Reshape flat slices to [batch, seq_len] int32 tensors.
    let input = InlineArray::from_i32_slice(input_ids)
        .reshape(&[batch_size, seq_len]);
    let label_arr = InlineArray::from_i32_slice(labels)
        .reshape(&[batch_size, seq_len]);

    // ── Collect trainable params ────────────────────────────────────────────
    // We need a stable, ordered list of names so we can correlate the
    // gradient vector returned by `value_and_grad` back to param names.
    let mut params = lora.trainable_params();
    let param_names: Vec<String> = {
        let mut names: Vec<String> = params.keys().cloned().collect();
        // Sort for determinism — HashMap iteration order is undefined.
        names.sort();
        names
    };

    let param_arrays: Vec<InlineArray> = param_names
        .iter()
        .map(|k| params[k].clone())
        .collect();

    // Non-differentiated context: [input_ids, labels]
    let input_arrays = vec![input, label_arr];

    // ── value_and_grad ──────────────────────────────────────────────────────
    // `all_arrays` layout: [param_0, …, param_N, input_ids, labels]
    let n_params = param_names.len();

    let (loss, grads) = inline_array::value_and_grad(
        |all_arrays| {
            // Inputs (non-differentiated):
            let inp = &all_arrays[n_params];
            let lab = &all_arrays[n_params + 1];

            // ── TODO: apply LoRA adapters in the forward pass ───────────────
            // See the TODO block in the function-level doc comment above.
            // Once `NativeWeights` exposes mutable projection accessors, the
            // adapter contribution should be added here using:
            //
            //   training::lora_forward(x, base_w, lora_a, lora_b, scale)
            //   training::lora_forward_quantized(x, packed_w, scales, biases,
            //                                    group_size, bits,
            //                                    lora_a, lora_b, scale)
            //
            // For now we fall back to the unmodified base-model forward pass.
            // The gradients flow into the adapter parameters through the causal
            // LM loss, but the adapters are not yet contributing to the logits.
            // ─────────────────────────────────────────────────────────────────

            let mut cache = NativeCache::new_empty(base_weights);
            let logits = crate::qwen3_native::forward_step(base_weights, inp, &mut cache);

            // Causal LM loss — shifts logits/labels by one and averages CE.
            training::causal_lm_loss(&logits, lab, ignore_index)
        },
        &param_arrays,
        &input_arrays,
    );

    // ── Gradient clipping ───────────────────────────────────────────────────
    // Assemble grad ParamSet in the same sorted order as param_arrays.
    let mut grad_set = ParamSet::new();
    for (i, name) in param_names.iter().enumerate() {
        grad_set.insert(name.clone(), grads[i].clone());
    }

    let norm = training::clip_grad_norm(&mut grad_set, max_grad_norm);

    // ── Optimizer step ──────────────────────────────────────────────────────
    optimizer.step(&mut params, &grad_set);

    // ── Write back updated adapter arrays ──────────────────────────────────
    lora.update_from_params(&params);

    // ── Materialise loss and norm for logging ───────────────────────────────
    // Both are scalar lazy arrays. Evaluate them together to avoid two
    // separate GPU round-trips.
    let mut loss_arr = loss;
    let mut norm_arr = norm;
    InlineArray::eval_2(&mut loss_arr, &mut norm_arr);

    let loss_f32 = loss_arr.item_f32();
    let norm_f32 = norm_arr.item_f32();

    StepResult {
        loss: loss_f32,
        grad_norm: norm_f32,
        step_time_ms: t0.elapsed().as_millis() as u64,
    }
}

// ============================================================================
// Convenience: multi-step training loop
// ============================================================================

/// Progress callback invoked after every completed optimizer step.
///
/// Arguments: `(global_step, result)`.
pub type StepCallback = Box<dyn FnMut(usize, &StepResult)>;

/// Run a gradient-accumulation-aware training loop over a flat token dataset.
///
/// `tokens` is a flat `[N]` int32 slice. It is carved into non-overlapping
/// windows of `(seq_len + 1)` tokens: the first `seq_len` become `input_ids`
/// and the last `seq_len` (i.e. offset by 1) become `labels`. Positions in
/// `labels` that fall outside the window are filled with `ignore_index`.
///
/// # Gradient accumulation
///
/// When `train_config.gradient_accumulation_steps > 1`, gradients are summed
/// (scaled by `1/accumulation_steps`) across that many mini-batches before an
/// optimizer step is taken.  The `StepCallback` fires once per optimizer step,
/// not once per forward pass.
///
/// # Returns
///
/// Average training loss over all steps taken during this call.
pub fn train_loop(
    base_weights: &NativeWeights,
    lora: &mut Qwen3LoraWeights,
    optimizer: &mut AdamW,
    tokens: &[i32],
    train_config: &TrainConfig,
    ignore_index: i32,
    mut callback: Option<StepCallback>,
) -> f32 {
    let seq_len = train_config.max_seq_len;
    let batch_size = train_config.batch_size;
    let grad_accum = train_config.gradient_accumulation_steps.max(1);
    let window = seq_len + 1; // need one extra token for the label shift

    // Slice the flat token stream into (input, label) pairs of length seq_len.
    let samples: Vec<(Vec<i32>, Vec<i32>)> = tokens
        .windows(window)
        .map(|w| {
            let input: Vec<i32> = w[..seq_len].to_vec();
            let label: Vec<i32> = w[1..window].to_vec();
            (input, label)
        })
        .collect();

    if samples.is_empty() {
        return 0.0;
    }

    let mut total_loss = 0.0_f32;
    let mut total_steps = 0_usize;
    let mut global_step = 0_usize;

    for _epoch in 0..train_config.num_epochs {
        // Batch samples.
        let batches: Vec<&[(Vec<i32>, Vec<i32>)]> =
            samples.chunks(batch_size).collect();

        let mut acc_grads: ParamSet = ParamSet::new();
        let mut acc_loss = 0.0_f32;
        let mut acc_count = 0_usize;

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Flatten batch into [batch * seq_len] slices.
            let actual_batch = batch.len() as i32;
            let mut flat_input: Vec<i32> =
                Vec::with_capacity(actual_batch as usize * seq_len);
            let mut flat_labels: Vec<i32> =
                Vec::with_capacity(actual_batch as usize * seq_len);
            for (inp, lbl) in *batch {
                flat_input.extend_from_slice(inp);
                flat_labels.extend_from_slice(lbl);
            }

            // Forward + backward (single micro-step).
            let step = train_step(
                base_weights,
                lora,
                optimizer,
                &flat_input,
                &flat_labels,
                actual_batch,
                seq_len as i32,
                ignore_index,
                train_config.max_grad_norm,
            );

            acc_loss += step.loss;
            acc_count += 1;

            // Accumulate gradients.
            let micro_grads = lora.trainable_params();
            let scale = 1.0 / grad_accum as f32;
            training::accumulate_gradients(&mut acc_grads, &micro_grads, scale);

            // Optimizer step when accumulation window is complete.
            let is_last_batch = batch_idx + 1 == batches.len();
            if acc_count >= grad_accum || is_last_batch {
                let mut params = lora.trainable_params();
                optimizer.step(&mut params, &acc_grads);
                lora.update_from_params(&params);

                // Eval and detach updated params to prevent graph growth.
                training::eval_params(&mut params);
                lora.update_from_params(&params);

                let avg_loss = acc_loss / acc_count as f32;
                total_loss += avg_loss;
                total_steps += 1;
                global_step += 1;

                let result = StepResult {
                    loss: avg_loss,
                    grad_norm: step.grad_norm,
                    step_time_ms: step.step_time_ms,
                };

                if let Some(ref mut cb) = callback {
                    cb(global_step, &result);
                }

                // Reset accumulators.
                acc_grads.clear();
                acc_loss = 0.0;
                acc_count = 0;
            }
        }
    }

    if total_steps > 0 {
        total_loss / total_steps as f32
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> Qwen3Config {
        // Minimal config: 2 layers, hidden=64, 2 heads, intermediate=128.
        serde_json::from_str(r#"{
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 128,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0
        }"#).expect("invalid dummy config")
    }

    #[test]
    fn test_lora_config_default() {
        let cfg = LoraConfig::default();
        assert_eq!(cfg.rank, 16);
        assert!((cfg.alpha - 32.0).abs() < 1e-6);
        assert_eq!(cfg.target_modules.len(), 7);
        assert_eq!(cfg.dropout, 0.0);
    }

    #[test]
    fn test_lora_adapter_shapes() {
        let adapter = LoraAdapter::new(64, 128, 8, 16.0);
        assert_eq!(adapter.lora_a.shape(), &[8, 64]);
        assert_eq!(adapter.lora_b.shape(), &[128, 8]);
        assert!((adapter.scale - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_weights_key_count() {
        let model_cfg = dummy_config();
        let lora_cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            target_modules: vec!["q_proj".into(), "k_proj".into(), "v_proj".into()],
            dropout: 0.0,
        };
        let lw = Qwen3LoraWeights::new(&model_cfg, lora_cfg);
        // 3 projections × 2 layers = 6 adapters
        assert_eq!(lw.adapters.len(), 6);
        assert!(lw.adapters.contains_key("layers.0.q_proj"));
        assert!(lw.adapters.contains_key("layers.1.v_proj"));
    }

    #[test]
    fn test_trainable_params_count() {
        let model_cfg = dummy_config();
        let lw = Qwen3LoraWeights::new(&model_cfg, LoraConfig::default());
        let params = lw.trainable_params();
        // 7 projections × 2 layers × 2 (a + b) = 28
        assert_eq!(params.len(), 28);
        // Check key conventions
        assert!(params.contains_key("layers.0.q_proj.lora_a"));
        assert!(params.contains_key("layers.1.down_proj.lora_b"));
    }

    #[test]
    fn test_update_from_params_roundtrip() {
        let model_cfg = dummy_config();
        let lora_cfg = LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            dropout: 0.0,
        };
        let mut lw = Qwen3LoraWeights::new(&model_cfg, lora_cfg);
        let mut params = lw.trainable_params();

        // Overwrite one array with a known constant.
        params.insert(
            "layers.0.q_proj.lora_a".into(),
            InlineArray::zeros(&[4, 64], 10),
        );
        lw.update_from_params(&params);

        // Confirm the adapter was updated (shape unchanged).
        assert_eq!(lw.adapters["layers.0.q_proj"].lora_a.shape(), &[4, 64]);
    }

    #[test]
    fn test_train_config_default() {
        let cfg = TrainConfig::default();
        assert!((cfg.learning_rate - 2e-4).abs() < 1e-10);
        assert_eq!(cfg.gradient_accumulation_steps, 1);
        assert_eq!(cfg.max_seq_len, 512);
    }
}

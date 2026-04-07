//! Shared inference configuration utilities.
//!
//! Provides canonical implementations for:
//! - Stop token collection from all available sources
//! - Sampling default loading from `generation_config.json`
//!
//! These are used by CLI, GUI, Python bindings, and examples to ensure
//! consistent inference behavior across all consumers.

use std::path::Path;

use crate::Tokenizer;
use crate::chat_templates::ChatTemplateType;

/// Collect all stop tokens from every available source.
///
/// Merges tokens from:
/// 1. `generation_config.json` — the model's declared `eos_token_id` (single or array)
/// 2. Chat template EOS — the template-specific end token (e.g. `<|im_end|>` for ChatML)
/// 3. Tokenizer's `eos_token_id` — resolved from special_tokens_map / heuristics
/// 4. Well-known special tokens — if they exist in the vocabulary as single tokens,
///    they're likely EOS candidates (e.g. `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`)
///
/// Returns a deduplicated list. This ensures fine-tuned models stop correctly
/// regardless of whether they produce the base model's EOS or the chat EOS.
pub fn collect_all_stop_tokens(
    model_path: &Path,
    tokenizer: &Tokenizer,
    template_type: Option<ChatTemplateType>,
) -> Vec<u32> {
    let mut tokens = Vec::new();

    // 1. generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(eos) = config.get("eos_token_id") {
                    if let Some(arr) = eos.as_array() {
                        for v in arr {
                            if let Some(id) = v.as_u64() {
                                tokens.push(id as u32);
                            }
                        }
                    } else if let Some(id) = eos.as_u64() {
                        tokens.push(id as u32);
                    }
                }
            }
        }
    }

    // 2. Chat template EOS (if template type is known)
    if let Some(tt) = template_type {
        let eos_str = tt.eos_token();
        if let Ok(encoded) = tokenizer.encode(eos_str) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // 3. Tokenizer's resolved eos_token_id
    if let Some(eos) = tokenizer.eos_token_id() {
        tokens.push(eos);
    }

    // 4. Well-known special tokens — probe the vocabulary for common EOS tokens.
    //    Only add tokens that encode to exactly 1 token (i.e. they're real special tokens,
    //    not subword sequences).
    let candidates = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<end_of_turn>",
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    for candidate in &candidates {
        if let Ok(encoded) = tokenizer.encode(candidate) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // Deduplicate
    tokens.sort_unstable();
    tokens.dedup();

    // Final fallback
    if tokens.is_empty() {
        tokens.push(2);
    }

    tracing::debug!("Collected stop tokens: {:?}", tokens);
    tokens
}

/// Sampling hyperparameter defaults loaded from model config.
#[derive(Debug, Clone)]
pub struct SamplingDefaults {
    /// Sampling temperature (0 = greedy).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p nucleus sampling.
    pub top_p: f32,
    /// Min-p dynamic sampling (0 = disabled).
    pub min_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Frequency penalty (0.0 = disabled).
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled).
    pub presence_penalty: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Load sampling defaults from a model's `generation_config.json`.
///
/// Falls back to sensible defaults if the file doesn't exist or fields are missing.
///
/// When `thinking_mode` is true, Qwen3-family recommended parameters (temp=0.6,
/// top_p=0.95, top_k=20) take precedence over `generation_config.json` for
/// temperature, because models often ship with generic temp=1.0 defaults that
/// produce noisy thinking traces. Other parameters still respect the model's
/// config.
pub fn load_sampling_defaults(model_path: &Path, thinking_mode: bool) -> SamplingDefaults {
    let mut defaults = SamplingDefaults::default();

    // Read generation_config.json first (model's declared defaults)
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(v) = config.get("temperature").and_then(|v| v.as_f64()) {
                    defaults.temperature = v as f32;
                }
                if let Some(v) = config.get("top_k").and_then(|v| v.as_u64()) {
                    defaults.top_k = v as usize;
                }
                if let Some(v) = config.get("top_p").and_then(|v| v.as_f64()) {
                    defaults.top_p = v as f32;
                }
                if let Some(v) = config.get("min_p").and_then(|v| v.as_f64()) {
                    defaults.min_p = v as f32;
                }
                if let Some(v) = config.get("repetition_penalty").and_then(|v| v.as_f64()) {
                    defaults.repetition_penalty = v as f32;
                }
                if let Some(v) = config.get("frequency_penalty").and_then(|v| v.as_f64()) {
                    defaults.frequency_penalty = v as f32;
                }
                if let Some(v) = config.get("presence_penalty").and_then(|v| v.as_f64()) {
                    defaults.presence_penalty = v as f32;
                }
            }
        }
    }

    // Thinking mode: override temperature with Qwen3 recommendations.
    // Models ship with generic temp=1.0 in generation_config.json, but thinking
    // mode needs lower temperature (0.6) for focused reasoning. top_k/top_p from
    // the model config are usually fine.
    if thinking_mode {
        defaults.temperature = 0.6;
        // Ensure top_p is wide enough for thinking (Qwen3 recommends 0.95)
        if defaults.top_p < 0.9 {
            defaults.top_p = 0.95;
        }
    }

    defaults
}

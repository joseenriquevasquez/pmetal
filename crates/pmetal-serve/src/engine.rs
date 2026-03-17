//! Core inference engine that wraps model + tokenizer + generation.

use crate::error::{ServeError, ServeResult};
use crate::types::ChatMessage;
use mlx_rs::Array;
use mlx_rs::ops::indexing::argmax_axis;
use pmetal_data::chat_templates::{ChatTemplate, detect_chat_template};
use pmetal_models::dispatcher::DynamicModel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// Per-request metrics
// ────────────────────────────────────────────────────────────────────────────

/// Timing and throughput metrics for a single generation request.
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    /// Time from request start to the first generated token (ms).
    pub first_token_latency_ms: f64,
    /// Total time from start to last token (ms).
    pub total_latency_ms: f64,
    /// Generated tokens per second (completion_tokens / total_latency).
    pub tokens_per_second: f64,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of completion tokens.
    pub completion_tokens: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Inference engine
// ────────────────────────────────────────────────────────────────────────────

/// The inference engine encapsulates model, tokenizer, and generation parameters.
pub struct InferenceEngine {
    /// The loaded model (behind a std Mutex — DynamicModel is !Send).
    model: Arc<Mutex<ModelState>>,
    /// The tokenizer.
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// Detected chat template.
    chat_template: ChatTemplate,
    /// Model name/ID for API responses.
    model_id: String,
    /// Maximum sequence length for KV cache.
    max_seq_len: usize,
    /// Stop token IDs.
    stop_token_ids: Vec<u32>,
    /// Model creation timestamp.
    created_at: i64,
}

/// Model + cache state that must be accessed sequentially.
struct ModelState {
    model: DynamicModel,
}

// SAFETY: DynamicModel contains `dyn` trait objects that are `!Send` by default,
// but MLX operations are thread-safe within the unified memory model on Apple Silicon.
// We protect all access with a std::sync::Mutex, ensuring single-threaded access.
#[allow(unsafe_code)]
unsafe impl Send for ModelState {}
#[allow(unsafe_code)]
unsafe impl Sync for ModelState {}

impl InferenceEngine {
    /// Create a new inference engine from a loaded model and tokenizer.
    pub fn new(
        model: DynamicModel,
        tokenizer: tokenizers::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
    ) -> ServeResult<Self> {
        let chat_template = detect_chat_template(model_path, &model_id);

        // Collect stop tokens from tokenizer's special tokens
        let mut stop_token_ids = Vec::new();
        if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
            stop_token_ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("<|eot_id|>") {
            stop_token_ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
            stop_token_ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("</s>") {
            stop_token_ids.push(id);
        }
        if let Some(id) = tokenizer.token_to_id("<|end|>") {
            stop_token_ids.push(id);
        }

        let created_at = chrono::Utc::now().timestamp();

        Ok(Self {
            model: Arc::new(Mutex::new(ModelState { model })),
            tokenizer: Arc::new(tokenizer),
            chat_template,
            model_id,
            max_seq_len,
            stop_token_ids,
            created_at,
        })
    }

    /// Model ID for API responses.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Creation timestamp.
    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    /// Format chat messages using the detected template.
    pub fn format_chat(&self, messages: &[ChatMessage]) -> String {
        let msgs: Vec<pmetal_data::chat_templates::Message> = messages
            .iter()
            .map(|m| pmetal_data::chat_templates::Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();
        let formatted = self.chat_template.apply(&msgs);
        formatted.text
    }

    /// Tokenize a prompt string.
    pub fn tokenize(&self, text: &str) -> ServeResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> ServeResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))
    }

    /// Generate tokens from input IDs (non-streaming).
    ///
    /// Returns `(generated_tokens, finish_reason, metrics)`.
    pub async fn generate(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        _temperature: f32,
        _top_p: Option<f32>,
        extra_stop_tokens: &[u32],
    ) -> ServeResult<(Vec<u32>, String, RequestMetrics)> {
        let prompt_tokens = input_ids.len();
        let start = Instant::now();

        let mut state = self.model.lock().map_err(|_| ServeError::Busy)?;
        let model = &mut state.model;
        let mut cache = model.create_cache(self.max_seq_len);

        // Build input array [1, seq_len]
        let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
        let seq_len = input_ids.len() as i32;
        let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

        // Prefill
        let mut logits = model
            .forward_with_cache(&input_arr, None, Some(&mut cache))
            .map_err(ServeError::Model)?;
        logits.eval().map_err(ServeError::Model)?;

        let mut generated = Vec::with_capacity(max_tokens);
        let mut finish_reason = "length".to_string();
        let mut first_token_time: Option<f64> = None;

        for _ in 0..max_tokens {
            let next_token = self.sample_greedy(&logits)?;

            // Record first-token latency on the first iteration.
            if first_token_time.is_none() {
                first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
            }

            if self.stop_token_ids.contains(&next_token) || extra_stop_tokens.contains(&next_token)
            {
                finish_reason = "stop".to_string();
                break;
            }

            generated.push(next_token);

            // Decode step
            let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
            logits = model
                .forward_with_cache(&next_input, None, Some(&mut cache))
                .map_err(ServeError::Model)?;
            logits.eval().map_err(ServeError::Model)?;
        }

        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let completion_tokens = generated.len();
        let tokens_per_second = if total_latency_ms > 0.0 {
            completion_tokens as f64 / (total_latency_ms / 1000.0)
        } else {
            0.0
        };

        let metrics = RequestMetrics {
            first_token_latency_ms: first_token_time.unwrap_or(total_latency_ms),
            total_latency_ms,
            tokens_per_second,
            prompt_tokens,
            completion_tokens,
        };

        Ok((generated, finish_reason, metrics))
    }

    /// Generate tokens one at a time, invoking `on_token` for each.
    ///
    /// This is the streaming variant: the caller receives each token
    /// synchronously as it is produced. Returns `(finish_reason, metrics)`.
    pub async fn generate_streaming<F>(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        _temperature: f32,
        _top_p: Option<f32>,
        extra_stop_tokens: &[u32],
        mut on_token: F,
    ) -> ServeResult<(String, RequestMetrics)>
    where
        F: FnMut(u32) -> ServeResult<()>,
    {
        let prompt_tokens = input_ids.len();
        let start = Instant::now();

        let mut state = self.model.lock().map_err(|_| ServeError::Busy)?;
        let model = &mut state.model;
        let mut cache = model.create_cache(self.max_seq_len);

        // Build input array [1, seq_len]
        let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
        let seq_len = input_ids.len() as i32;
        let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

        // Prefill
        let mut logits = model
            .forward_with_cache(&input_arr, None, Some(&mut cache))
            .map_err(ServeError::Model)?;
        logits.eval().map_err(ServeError::Model)?;

        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();
        let mut first_token_time: Option<f64> = None;

        for _ in 0..max_tokens {
            let next_token = self.sample_greedy(&logits)?;

            if first_token_time.is_none() {
                first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
            }

            if self.stop_token_ids.contains(&next_token) || extra_stop_tokens.contains(&next_token)
            {
                finish_reason = "stop".to_string();
                break;
            }

            on_token(next_token)?;
            completion_tokens += 1;

            // Decode step
            let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
            logits = model
                .forward_with_cache(&next_input, None, Some(&mut cache))
                .map_err(ServeError::Model)?;
            logits.eval().map_err(ServeError::Model)?;
        }

        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tokens_per_second = if total_latency_ms > 0.0 {
            completion_tokens as f64 / (total_latency_ms / 1000.0)
        } else {
            0.0
        };

        let metrics = RequestMetrics {
            first_token_latency_ms: first_token_time.unwrap_or(total_latency_ms),
            total_latency_ms,
            tokens_per_second,
            prompt_tokens,
            completion_tokens,
        };

        Ok((finish_reason, metrics))
    }

    /// Greedy sampling: argmax of the last position's logits.
    fn sample_greedy(&self, logits: &Array) -> ServeResult<u32> {
        // logits: [1, seq_len, vocab_size]
        // Squeeze batch dim, flatten, extract last row, argmax
        let squeezed = logits.squeeze_axes(&[0]).map_err(ServeError::Model)?;
        let seq_len = squeezed.dim(0) as usize;
        let vocab_size = squeezed.dim(1) as usize;

        let flat = squeezed.reshape(&[-1]).map_err(ServeError::Model)?;
        flat.eval().map_err(ServeError::Model)?;
        let data = flat.as_slice::<f32>();

        let start = (seq_len - 1) * vocab_size;
        let last_row = &data[start..start + vocab_size];
        let last_logits = Array::from_slice(last_row, &[vocab_size as i32]);

        let token_arr = argmax_axis(&last_logits, 0, false).map_err(ServeError::Model)?;
        token_arr.eval().map_err(ServeError::Model)?;
        Ok(token_arr.item::<u32>())
    }
}

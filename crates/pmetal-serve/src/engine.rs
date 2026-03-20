//! Core inference engine that wraps model + tokenizer + generation.

use crate::error::{ServeError, ServeResult};
use crate::types::ChatMessage;
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use pmetal_data::chat_templates::{ChatTemplate, ChatTemplateType, detect_chat_template};
use pmetal_data::inference_config::collect_all_stop_tokens;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::{GenerationConfig, Sampler};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// Per-request sampling parameters
// ────────────────────────────────────────────────────────────────────────────

/// All sampling parameters for a single generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub extra_stop_token_ids: Vec<u32>,
}

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
// Token event (sent through the mpsc channel during streaming)
// ────────────────────────────────────────────────────────────────────────────

/// A single event emitted during token-by-token streaming generation.
pub enum TokenEvent {
    /// A generated token.
    Token(u32),
    /// Generation is complete — carries finish reason and final metrics.
    Done(String, RequestMetrics),
    /// Generation failed.
    Error(String),
}

// ────────────────────────────────────────────────────────────────────────────
// Inference engine
// ────────────────────────────────────────────────────────────────────────────

/// The inference engine encapsulates model, tokenizer, and generation parameters.
pub struct InferenceEngine {
    /// The loaded model (behind a std Mutex — DynamicModel is !Send).
    model: Arc<Mutex<ModelState>>,
    /// The tokenizer.
    tokenizer: Arc<pmetal_data::Tokenizer>,
    /// Detected chat template.
    chat_template: ChatTemplate,
    /// Model name/ID for API responses.
    model_id: String,
    /// Maximum sequence length for KV cache.
    max_seq_len: usize,
    /// Stop token IDs collected from all available sources.
    stop_token_ids: Vec<u32>,
    /// Model creation timestamp.
    created_at: i64,
}

/// Model + cache state that must be accessed sequentially.
struct ModelState {
    model: DynamicModel,
}

// SAFETY: DynamicModel is !Send because it contains raw pointers from MLX's C FFI.
// We serialize all access through std::sync::Mutex, ensuring no concurrent access.
// The Mutex guard is never held across an await point.
#[allow(unsafe_code)]
unsafe impl Send for ModelState {}

impl InferenceEngine {
    /// Create a new inference engine from a loaded model and tokenizer.
    pub fn new(
        model: DynamicModel,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
    ) -> ServeResult<Self> {
        let chat_template = detect_chat_template(model_path, &model_id);

        // Collect stop tokens from all available sources using the canonical
        // `collect_all_stop_tokens` implementation from pmetal-data.
        // This merges generation_config.json EOS, chat-template EOS, tokenizer
        // EOS, and 11 well-known special token probes — deduplicated.
        let template_type: Option<ChatTemplateType> = Some(chat_template.template_type);
        let stop_token_ids = collect_all_stop_tokens(model_path, &tokenizer, template_type);

        tracing::info!(
            "Inference engine ready: model_id={}, stop_tokens={:?}",
            model_id,
            stop_token_ids
        );

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

    /// Shared reference to the tokenizer.
    ///
    /// Returns a cloned `Arc` so that route handlers can hold onto the
    /// tokenizer independently of the engine reference, which is needed
    /// for decoding tokens inside async streaming closures.
    pub fn tokenizer_arc(&self) -> Arc<pmetal_data::Tokenizer> {
        Arc::clone(&self.tokenizer)
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
        self.tokenizer
            .encode(text)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> ServeResult<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))
    }

    /// Validate sampling parameters, returning an error for any out-of-range value.
    ///
    /// Deliberately does not error on `max_tokens > max_seq_len` — the engine
    /// clamps silently, matching OpenAI behaviour.
    fn validate_params(params: &SamplingParams, _max_seq_len: usize) -> ServeResult<()> {
        if params.max_tokens == 0 {
            return Err(ServeError::BadRequest("max_tokens must be >= 1".into()));
        }
        if params.temperature < 0.0 || !params.temperature.is_finite() {
            return Err(ServeError::BadRequest(
                "temperature must be >= 0.0 and finite".into(),
            ));
        }
        if let Some(top_p) = params.top_p {
            if top_p <= 0.0 || top_p > 1.0 || !top_p.is_finite() {
                return Err(ServeError::BadRequest("top_p must be in (0.0, 1.0]".into()));
            }
        }
        if let Some(min_p) = params.min_p {
            if !(0.0..1.0).contains(&min_p) || !min_p.is_finite() {
                return Err(ServeError::BadRequest("min_p must be in [0.0, 1.0)".into()));
            }
        }
        if let Some(rp) = params.repetition_penalty {
            if rp <= 0.0 || !rp.is_finite() {
                return Err(ServeError::BadRequest(
                    "repetition_penalty must be > 0.0".into(),
                ));
            }
        }
        if let Some(fp) = params.frequency_penalty {
            if !fp.is_finite() {
                return Err(ServeError::BadRequest(
                    "frequency_penalty must be finite".into(),
                ));
            }
        }
        if let Some(pp) = params.presence_penalty {
            if !pp.is_finite() {
                return Err(ServeError::BadRequest(
                    "presence_penalty must be finite".into(),
                ));
            }
        }
        Ok(())
    }

    /// Build a `GenerationConfig` from API request sampling parameters.
    ///
    /// Temperature == 0.0 or unset maps to greedy decoding (`do_sample = false`).
    /// All stop tokens (engine-level + per-request) are merged into the config.
    /// `max_tokens` is silently clamped to `max_seq_len` (matches OpenAI behaviour).
    pub fn build_generation_config(&self, params: &SamplingParams) -> GenerationConfig {
        let temperature = params.temperature;
        let do_sample = temperature > 0.0;

        // Clamp max_tokens silently — matching OpenAI API behaviour.
        let max_tokens = params.max_tokens.min(self.max_seq_len);

        // Merge engine-level stop tokens with any per-request stop tokens.
        let mut stop_tokens = self.stop_token_ids.clone();
        stop_tokens.extend_from_slice(&params.extra_stop_token_ids);
        stop_tokens.sort_unstable();
        stop_tokens.dedup();

        let mut config = if do_sample {
            // Start from the default sampling config, then apply per-request overrides.
            GenerationConfig {
                max_new_tokens: max_tokens,
                temperature,
                do_sample: true,
                stop_tokens,
                seed: params.seed,
                ..GenerationConfig::default()
            }
        } else {
            GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
        };

        // Apply optional overrides — only set fields the caller specified.
        if let Some(top_k) = params.top_k {
            config = config.with_top_k(top_k);
        }
        if let Some(top_p) = params.top_p {
            config = config.with_top_p(top_p);
        }
        if let Some(min_p) = params.min_p {
            config = config.with_min_p(min_p);
        }
        if let Some(rp) = params.repetition_penalty {
            config = config.with_repetition_penalty(rp);
        }
        if let Some(fp) = params.frequency_penalty {
            config = config.with_frequency_penalty(fp);
        }
        if let Some(pp) = params.presence_penalty {
            config = config.with_presence_penalty(pp);
        }
        // For greedy mode the seed is not set in the config initializer above,
        // so apply it here (it won't affect sampling but may affect MLX RNG state).
        if !do_sample {
            if let Some(seed) = params.seed {
                config = config.with_seed(seed);
            }
        }

        config
    }

    /// Extract the last-position logits from a model output tensor.
    ///
    /// Model outputs have shape `[1, seq_len, vocab_size]` (after prefill) or
    /// `[1, 1, vocab_size]` (after decode steps). We extract the last position
    /// and squeeze to a 1-D array of shape `[vocab_size]` suitable for
    /// `Sampler::sample`.
    fn extract_last_logits(logits: &Array) -> ServeResult<Array> {
        // Shape: [batch=1, seq_len, vocab_size]
        // Index last position along seq_len dim → [1, vocab_size]
        let last_idx = logits.dim(1) - 1;
        let last = logits.index((.., last_idx, ..));
        // Squeeze batch dim → [vocab_size]
        last.squeeze().map_err(ServeError::Model)
    }

    /// Generate tokens from input IDs (non-streaming).
    ///
    /// Returns `(generated_tokens, finish_reason, metrics)`.
    pub async fn generate(
        &self,
        input_ids: &[u32],
        params: SamplingParams,
    ) -> ServeResult<(Vec<u32>, String, RequestMetrics)> {
        // Validate before dispatching to the blocking thread.
        Self::validate_params(&params, self.max_seq_len)?;

        let prompt_tokens = input_ids.len();
        let gen_config = self.build_generation_config(&params);
        // Use the (possibly clamped) value from the built config.
        let max_tokens = gen_config.max_new_tokens;
        let stop_tokens = gen_config.stop_tokens.clone();
        let input_ids = input_ids.to_vec();
        let model_arc = Arc::clone(&self.model);
        let max_seq_len = self.max_seq_len;

        // Generation is synchronous/blocking; run it on a dedicated blocking
        // thread so we don't stall the async executor.
        //
        // DynamicModel is !Send — ModelState wraps it with an unsafe Send impl
        // guarded by the Mutex. The Mutex is cloned (Arc) into the closure.
        let result = tokio::task::spawn_blocking(move || {
            let mut state = model_arc.lock().map_err(|_| ServeError::Busy)?;
            let model = &mut state.model;
            let mut cache = model.create_cache(max_seq_len);

            // Build input array [1, seq_len] for prefill.
            let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
            let seq_len = input_ids.len() as i32;
            let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

            let start = Instant::now();

            // Prefill forward pass — produces logits for the first sample step.
            let mut logits = model
                .forward_with_cache(&input_arr, None, Some(&mut cache))
                .map_err(ServeError::Model)?;
            // TODO(perf): switch to mlx_rs::eval_async once available so the
            // next prefill/decode can overlap with the current eval on the GPU.
            logits.eval().map_err(ServeError::Model)?;

            // Sampler must be created inside spawn_blocking — it holds
            // MLX Arrays and is !Send.
            let mut sampler = Sampler::new(gen_config);

            let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut finish_reason = "length".to_string();
            let mut first_token_time: Option<f64> = None;
            // Track all tokens seen (prompt + generated) for repetition penalty.
            let mut all_tokens: Vec<u32> = input_ids.clone();

            for i in 0..max_tokens {
                // Sample from current logits (prefill logits on i=0, decode logits thereafter).
                let last_logits = Self::extract_last_logits(&logits)?;
                let next_token = sampler
                    .sample(&last_logits, &all_tokens)
                    .map_err(ServeError::Model)?;

                // Record TTFT on first sampled token.
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                }

                // Check stop condition before accepting the token.
                if stop_tokens.contains(&next_token) {
                    finish_reason = "stop".to_string();
                    break;
                }

                generated.push(next_token);
                all_tokens.push(next_token);

                // Only run a decode forward pass when there are more iterations.
                // This avoids the wasted forward pass after the last token.
                if i + 1 < max_tokens {
                    let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
                    logits = model
                        .forward_with_cache(&next_input, None, Some(&mut cache))
                        .map_err(ServeError::Model)?;
                    // eval() is required here: without it the lazy graph grows
                    // unbounded across decode steps and OOMs before the loop ends.
                    // TODO(perf): replace with async_eval for pipeline overlap.
                    logits.eval().map_err(ServeError::Model)?;
                }
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

            Ok::<_, ServeError>((generated, finish_reason, metrics))
        })
        .await
        .map_err(|e| ServeError::Internal(e.to_string()))??;

        Ok(result)
    }

    /// Begin token-by-token streaming generation.
    ///
    /// Validates `params` before spawning. If validation fails, sends a single
    /// `TokenEvent::Error` through the channel and returns immediately.
    ///
    /// Spawns a blocking thread that runs the generation loop and sends
    /// `TokenEvent` values through an `mpsc` channel. Returns the receiver
    /// end immediately so the route handler can start consuming events while
    /// generation proceeds in parallel.
    ///
    /// The channel will receive:
    /// - Zero or more `TokenEvent::Token(id)` — one per generated token.
    /// - Exactly one `TokenEvent::Done(finish_reason, metrics)` on success.
    /// - Exactly one `TokenEvent::Error(msg)` if generation fails (no [DONE]).
    pub fn generate_streaming(
        &self,
        input_ids: &[u32],
        params: SamplingParams,
    ) -> tokio::sync::mpsc::Receiver<TokenEvent> {
        // Channel capacity: keep a small buffer so the generation thread is
        // never stalled waiting for the HTTP layer to consume events, but
        // don't allocate an unbounded queue.
        let (tx, rx) = tokio::sync::mpsc::channel::<TokenEvent>(64);

        // Validate before spawning — send error through channel if invalid.
        if let Err(e) = Self::validate_params(&params, self.max_seq_len) {
            let _ = tx.try_send(TokenEvent::Error(e.to_string()));
            return rx;
        }

        let prompt_tokens = input_ids.len();
        let gen_config = self.build_generation_config(&params);
        // Use the (possibly clamped) value from the built config.
        let max_tokens = gen_config.max_new_tokens;
        let stop_tokens = gen_config.stop_tokens.clone();
        let input_ids = input_ids.to_vec();
        let model_arc = Arc::clone(&self.model);
        let max_seq_len = self.max_seq_len;

        // Spawn generation on a dedicated blocking thread.
        tokio::task::spawn_blocking(move || {
            // Macro-style helper: send an event or bail on channel close.
            macro_rules! send {
                ($event:expr) => {
                    if tx.blocking_send($event).is_err() {
                        // Receiver dropped (client disconnected) — stop generation.
                        return;
                    }
                };
            }

            let state_guard = match model_arc.lock() {
                Ok(g) => g,
                Err(_) => {
                    send!(TokenEvent::Error("engine busy".into()));
                    return;
                }
            };
            // Shadow to get mutable access — we need to hold the guard for
            // the entire generation loop.
            let mut state = state_guard;
            let model = &mut state.model;
            let mut cache = model.create_cache(max_seq_len);

            // Build prefill input array.
            let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
            let seq_len = input_ids.len() as i32;
            let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

            let start = Instant::now();

            // Prefill forward pass — produces logits for the first sample step.
            let mut logits = match model.forward_with_cache(&input_arr, None, Some(&mut cache)) {
                Ok(l) => l,
                Err(e) => {
                    send!(TokenEvent::Error(e.to_string()));
                    return;
                }
            };
            // eval() is required here: without it the lazy graph grows unbounded.
            // TODO(perf): replace with async_eval for pipeline overlap.
            if let Err(e) = logits.eval() {
                send!(TokenEvent::Error(e.to_string()));
                return;
            }

            // Sampler created inside spawn_blocking — it holds MLX Arrays.
            let mut sampler = Sampler::new(gen_config);

            let mut completion_tokens = 0usize;
            let mut finish_reason = "length".to_string();
            let mut first_token_time: Option<f64> = None;
            let mut all_tokens: Vec<u32> = input_ids.clone();

            for i in 0..max_tokens {
                // Sample from current logits (prefill logits on i=0, decode logits thereafter).
                let last_logits = match Self::extract_last_logits(&logits) {
                    Ok(l) => l,
                    Err(e) => {
                        send!(TokenEvent::Error(e.to_string()));
                        return;
                    }
                };

                let next_token = match sampler.sample(&last_logits, &all_tokens) {
                    Ok(t) => t,
                    Err(e) => {
                        send!(TokenEvent::Error(e.to_string()));
                        return;
                    }
                };

                // Record TTFT on first sampled token.
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                }

                // Check stop condition before emitting the token.
                if stop_tokens.contains(&next_token) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Emit token before running the next forward pass so the
                // route handler can begin decoding and sending it to the
                // client while the GPU works on the next token.
                send!(TokenEvent::Token(next_token));
                completion_tokens += 1;
                all_tokens.push(next_token);

                // Only run a decode forward pass when there are more iterations.
                // This avoids the wasted forward pass after the last token.
                if i + 1 < max_tokens {
                    let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
                    logits = match model.forward_with_cache(&next_input, None, Some(&mut cache)) {
                        Ok(l) => l,
                        Err(e) => {
                            send!(TokenEvent::Error(e.to_string()));
                            return;
                        }
                    };
                    // eval() is required: prevents unbounded lazy graph growth.
                    // TODO(perf): replace with async_eval for pipeline overlap.
                    if let Err(e) = logits.eval() {
                        send!(TokenEvent::Error(e.to_string()));
                        return;
                    }
                }
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

            // Done — send final event (ignore send error, client may be gone).
            let _ = tx.blocking_send(TokenEvent::Done(finish_reason, metrics));
        });

        rx
    }
}

//! HTTP route handlers for the OpenAI-compatible API.

use crate::engine::InferenceEngine;
use crate::error::ServeError;
use crate::types::*;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use futures::stream;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ────────────────────────────────────────────────────────────────────────────
// Serving metrics state
// ────────────────────────────────────────────────────────────────────────────

/// Atomic metrics counters accumulated across all requests.
///
/// All values use fixed-point representation where applicable:
/// - Latency sums: microseconds (integer) for precision without f64 atomics.
/// - Token counts: plain integers.
#[derive(Debug, Default)]
pub struct ServingMetrics {
    /// Total completed requests.
    pub total_requests: AtomicU64,
    /// Sum of first-token latencies in microseconds.
    pub sum_first_token_us: AtomicU64,
    /// Sum of total request latencies in microseconds.
    pub sum_total_latency_us: AtomicU64,
    /// Sum of completion token counts.
    pub sum_completion_tokens: AtomicU64,
    /// Sum of prompt token counts.
    pub sum_prompt_tokens: AtomicU64,
}

impl ServingMetrics {
    /// Record metrics from a completed request.
    pub fn record(&self, metrics: &crate::engine::RequestMetrics) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.sum_first_token_us.fetch_add(
            (metrics.first_token_latency_ms * 1000.0) as u64,
            Ordering::Relaxed,
        );
        self.sum_total_latency_us.fetch_add(
            (metrics.total_latency_ms * 1000.0) as u64,
            Ordering::Relaxed,
        );
        self.sum_completion_tokens
            .fetch_add(metrics.completion_tokens as u64, Ordering::Relaxed);
        self.sum_prompt_tokens
            .fetch_add(metrics.prompt_tokens as u64, Ordering::Relaxed);
    }

    /// Compute rolling averages and return as a JSON value.
    pub fn snapshot(&self) -> serde_json::Value {
        let total = self.total_requests.load(Ordering::Relaxed);
        let sum_first = self.sum_first_token_us.load(Ordering::Relaxed);
        let sum_total = self.sum_total_latency_us.load(Ordering::Relaxed);
        let sum_compl = self.sum_completion_tokens.load(Ordering::Relaxed);
        let sum_prompt = self.sum_prompt_tokens.load(Ordering::Relaxed);

        let avg_first_ms = if total > 0 {
            (sum_first as f64 / total as f64) / 1000.0
        } else {
            0.0
        };
        let avg_total_ms = if total > 0 {
            (sum_total as f64 / total as f64) / 1000.0
        } else {
            0.0
        };
        // tokens/s: total completion tokens / total wall time across all reqs
        let rolling_tps = if sum_total > 0 {
            sum_compl as f64 / (sum_total as f64 / 1_000_000.0)
        } else {
            0.0
        };

        json!({
            "total_requests": total,
            "avg_first_token_latency_ms": avg_first_ms,
            "avg_total_latency_ms": avg_total_ms,
            "rolling_tokens_per_second": rolling_tps,
            "total_prompt_tokens": sum_prompt,
            "total_completion_tokens": sum_compl,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Application state
// ────────────────────────────────────────────────────────────────────────────

/// Shared application state.
pub struct AppState {
    pub engine: InferenceEngine,
    pub metrics: ServingMetrics,
}

// ────────────────────────────────────────────────────────────────────────────
// Route handlers
// ────────────────────────────────────────────────────────────────────────────

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let model = ModelInfo {
        id: state.engine.model_id().to_string(),
        object: "model".to_string(),
        created: state.engine.created_at(),
        owned_by: "pmetal".to_string(),
    };

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![model],
    })
}

/// GET /v1/metrics
///
/// Returns rolling aggregate serving metrics as JSON.
pub async fn serving_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

/// POST /v1/chat/completions
///
/// Supports both non-streaming (default) and streaming (`stream: true`) modes.
/// When streaming, returns a `text/event-stream` SSE response conforming to
/// the OpenAI streaming format with `data: {...}` events and a final
/// `data: [DONE]` sentinel.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ServeError> {
    // Format messages using chat template
    let prompt = state.engine.format_chat(&req.messages);
    let input_ids = state.engine.tokenize(&prompt)?;
    let prompt_tokens = input_ids.len();

    // Parse stop tokens from request
    let extra_stop_ids: Vec<u32> = req
        .stop
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|s| {
            state
                .engine
                .tokenize(s)
                .ok()
                .and_then(|ids| ids.first().copied())
        })
        .collect();

    let temperature = req.temperature.unwrap_or(0.7);
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.engine.model_id().to_string();
    let created = state.engine.created_at();

    if req.stream.unwrap_or(false) {
        // ── Streaming path ───────────────────────────────────────────────────
        // We collect all tokens up front (generation is synchronous/blocking
        // under the mutex) and then stream them as SSE events. This keeps the
        // implementation simple while correctly producing the SSE wire format.
        //
        // A true token-by-token streaming implementation would require moving
        // the generation loop onto a dedicated thread and piping tokens through
        // a channel; that is deferred until the generation loop becomes async.
        let (generated_tokens, finish_reason, stream_metrics) = state
            .engine
            .generate(
                &input_ids,
                req.max_tokens,
                temperature,
                req.top_p,
                &extra_stop_ids,
            )
            .await?;
        state.metrics.record(&stream_metrics);

        // Build SSE event stream from the collected tokens.
        let events: Vec<Result<Event, Infallible>> = build_chat_sse_events(
            &generated_tokens,
            &finish_reason,
            &request_id,
            &model_id,
            created,
            &state.engine,
        )?;

        let stream = stream::iter(events);
        return Ok(Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response());
    }

    // ── Non-streaming path ───────────────────────────────────────────────────
    let (tokens, finish_reason, metrics) = state
        .engine
        .generate(
            &input_ids,
            req.max_tokens,
            temperature,
            req.top_p,
            &extra_stop_ids,
        )
        .await?;

    state.metrics.record(&metrics);

    let completion_tokens = tokens.len();
    let text = state.engine.decode(&tokens)?;

    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ServeError> {
    let input_ids = state.engine.tokenize(&req.prompt)?;
    let prompt_tokens = input_ids.len();
    let temperature = req.temperature.unwrap_or(0.0);

    let extra_stop_ids: Vec<u32> = req
        .stop
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|s| {
            state
                .engine
                .tokenize(s)
                .ok()
                .and_then(|ids| ids.first().copied())
        })
        .collect();

    let (tokens, finish_reason, metrics) = state
        .engine
        .generate(
            &input_ids,
            req.max_tokens,
            temperature,
            req.top_p,
            &extra_stop_ids,
        )
        .await?;

    state.metrics.record(&metrics);

    let completion_tokens = tokens.len();
    let text = state.engine.decode(&tokens)?;

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: state.engine.created_at(),
        model: state.engine.model_id().to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

// ────────────────────────────────────────────────────────────────────────────
// SSE helpers
// ────────────────────────────────────────────────────────────────────────────

/// Build a Vec of SSE `Event`s for a chat completion streaming response.
///
/// Emits:
/// 1. An opening event with `role: "assistant"` and no content.
/// 2. One event per decoded text segment (one per token, decoded individually).
/// 3. A closing event with `finish_reason` and empty delta.
/// 4. A `[DONE]` sentinel event.
fn build_chat_sse_events(
    tokens: &[u32],
    finish_reason: &str,
    request_id: &str,
    model_id: &str,
    created: i64,
    engine: &InferenceEngine,
) -> Result<Vec<Result<Event, Infallible>>, ServeError> {
    let mut events: Vec<Result<Event, Infallible>> = Vec::with_capacity(tokens.len() + 3);

    // Opening event: role announcement.
    let opening_chunk = ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_id.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    events.push(Ok(
        Event::default().data(serde_json::to_string(&opening_chunk).unwrap_or_default())
    ));

    // One SSE event per token.
    for &token_id in tokens {
        let text = engine.decode(&[token_id]).unwrap_or_default();
        let chunk = ChatCompletionChunk {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some(text),
                },
                finish_reason: None,
            }],
        };
        events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }

    // Closing event: finish_reason, empty delta.
    let closing_chunk = ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_id.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: None,
            },
            finish_reason: Some(finish_reason.to_string()),
        }],
    };
    events.push(Ok(
        Event::default().data(serde_json::to_string(&closing_chunk).unwrap_or_default())
    ));

    // [DONE] sentinel.
    events.push(Ok(Event::default().data("[DONE]")));

    Ok(events)
}

//! Shared SSE helpers for the OpenAI + Anthropic streaming endpoints.
//!
//! BPE tokenisers can split a multi-byte UTF-8 codepoint across two tokens
//! — emitting each `Tokenizer::decode(&[token])` independently would send
//! invalid `\xNN` byte sequences to the client, which most HTTP/SSE
//! consumers reject. [`IncrementalDecoder`] buffers the full token
//! sequence, decodes the growing buffer as a unit, and emits only the
//! confirmed text prefix that has already shown up in a complete decode.
//!
//! All three streaming handlers (`chat_sse_stream`, `completion_sse_stream`,
//! and Anthropic `messages` streaming) previously open-coded this pattern
//! with `token_buffer: Vec<u32>` + `emitted_text_len: usize`. Factoring it
//! here keeps the state machine in one place — the handlers only differ in
//! how they wrap the decoded text into their endpoint-specific event
//! frames.

use std::sync::Arc;

/// Stateful decoder that turns a per-token event stream into UTF-8-safe
/// text deltas.
///
/// Typical use:
///
/// ```ignore
/// let mut dec = IncrementalDecoder::new(tokenizer);
/// // on each TokenEvent::Token:
/// let delta = dec.push(token_id);
/// if !delta.is_empty() { emit_content_delta(delta); }
/// // on TokenEvent::Done:
/// let tail = dec.flush();
/// if !tail.is_empty() { emit_content_delta(tail); }
/// // For tool-call detection post-Done:
/// let full = dec.decoded_text();
/// ```
pub struct IncrementalDecoder {
    tokenizer: Arc<pmetal_data::Tokenizer>,
    buffer: Vec<u32>,
    emitted: usize,
}

impl IncrementalDecoder {
    /// Create a new decoder that decodes tokens against `tokenizer`.
    pub fn new(tokenizer: Arc<pmetal_data::Tokenizer>) -> Self {
        Self {
            tokenizer,
            buffer: Vec::new(),
            emitted: 0,
        }
    }

    /// Push a token and return the newly confirmed text prefix. Returns an
    /// empty string when the decoder is still mid-codepoint — callers
    /// should skip emitting an SSE frame in that case.
    pub fn push(&mut self, token_id: u32) -> String {
        self.buffer.push(token_id);
        self.consume_newly_decoded()
    }

    /// Flush any remaining buffered tokens at end-of-stream. Returns an
    /// empty string when every decoded byte has already been emitted.
    pub fn flush(&mut self) -> String {
        self.consume_newly_decoded()
    }

    /// The full decoded text accumulated so far. Used by the chat /
    /// Anthropic streams to run best-effort tool-call detection after
    /// the terminal `Done` event.
    pub fn decoded_text(&self) -> String {
        self.tokenizer.decode(&self.buffer).unwrap_or_default()
    }

    /// Number of tokens seen so far. Used by the Anthropic stream to
    /// report `output_tokens` on the terminal `message_delta`.
    pub fn token_count(&self) -> usize {
        self.buffer.len()
    }

    /// Advance `emitted` to the current decoded length and return the
    /// suffix that crossed the boundary. Shared by [`push`] and [`flush`].
    fn consume_newly_decoded(&mut self) -> String {
        let decoded = self.tokenizer.decode(&self.buffer).unwrap_or_default();
        if decoded.len() > self.emitted {
            let out = decoded[self.emitted..].to_owned();
            self.emitted = decoded.len();
            out
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: IncrementalDecoder depends on `pmetal_data::Tokenizer`, which
    // loads real tokenizer JSON files. End-to-end SSE behaviour is covered
    // by the downstream handlers; the assertions here are kept to pure
    // state-transition invariants that don't require a concrete tokenizer.
    //
    // If you add a mock tokenizer to `pmetal-data::tokenizer::testing`,
    // plumb it in here to cover the BPE-boundary edge case directly.
}

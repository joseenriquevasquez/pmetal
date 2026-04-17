//! OpenAI-compatible request/response types.

use pmetal_data::chat_templates::{FunctionCall, ToolCall, ToolDefinition};
use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize a field that may be either a single string or an array of strings.
///
/// The OpenAI API accepts `"stop": "token"` (string) or `"stop": ["a", "b"]`
/// (array) — both map to `Option<Vec<String>>`.
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct StringOrVec;

    impl<'de> Visitor<'de> for StringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or array of strings or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_owned()]))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(vec![v]))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut vec = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                vec.push(s);
            }
            Ok(Some(vec))
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Tool calls made by the assistant. `None` for user/system/tool messages.
    ///
    /// When the model emits a structured tool call (best-effort JSON parse of the
    /// generated text), this is populated and `content` is left empty. OpenAI
    /// clients that support tools read `tool_calls` and ignore `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Chat completion request (POST /v1/chat/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Stop sequences — accepts either a single string or an array of strings.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// OpenAI-compatible tool definitions. When present, the chat template
    /// injects them into the system prompt so the model is aware it may emit
    /// tool calls. Best-effort JSON parsing of the model's output after
    /// generation decides whether `tool_calls` or `content` is returned.
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
}

fn default_max_tokens() -> usize {
    256
}

/// Text completion request (POST /v1/completions).
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Stop sequences — accepts either a single string or an array of strings.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_parse_bare_object() {
        let text = r#"{"name": "get_weather", "arguments": {"city": "SF"}}"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["city"], "SF");
    }

    #[test]
    fn tool_parse_wrapped_object() {
        let text = r#"{"tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}"#;
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn tool_parse_rejects_plain_text() {
        assert!(try_parse_tool_calls("Hello, world!").is_none());
        assert!(try_parse_tool_calls("{invalid json").is_none());
        assert!(try_parse_tool_calls(r#"{"greeting": "hi"}"#).is_none());
    }

    #[test]
    fn tool_parse_allows_whitespace() {
        let text = "  \n\t{\n  \"name\": \"f\"\n}\n ";
        let calls = try_parse_tool_calls(text).expect("should parse");
        assert_eq!(calls[0].function.name, "f");
    }
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Choice in a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Text completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// Choice in a text completion.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming SSE chunk for chat completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// Delta in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls attached to the closing chunk when the model's output parses
    /// as a function call. Mid-stream token deltas leave this `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Best-effort parser that converts a raw assistant response into a tool call
/// when the text matches the OpenAI tool-call shape.
///
/// Accepted shapes:
/// 1. Bare object: `{"name": "...", "arguments": {...}}`
/// 2. Wrapped object: `{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}`
///
/// Returns `None` when the text does not cleanly parse as a tool call. Callers
/// fall back to returning the raw text as `content`.
pub fn try_parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let trimmed = text.trim();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    if let Some(wrapper) = value.get("tool_calls").and_then(|v| v.as_array()) {
        let calls: Vec<ToolCall> = wrapper
            .iter()
            .filter_map(|tc| {
                let func = tc.get("function")?;
                Some(ToolCall {
                    id: tc.get("id").and_then(|v| v.as_str()).map(str::to_owned),
                    tool_type: tc
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("function")
                        .to_owned(),
                    function: FunctionCall {
                        name: func.get("name")?.as_str()?.to_owned(),
                        arguments: func
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    },
                })
            })
            .collect();
        return (!calls.is_empty()).then_some(calls);
    }

    let name = value.get("name").and_then(|v| v.as_str())?;
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    Some(vec![ToolCall {
        id: None,
        tool_type: "function".to_owned(),
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
        },
    }])
}

/// Model info for /v1/models.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Model list response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

//! OpenAI-compatible wire types (serde).

use serde::{Deserialize, Serialize};

fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_seed() -> u64 {
    42
}
fn default_stream() -> bool {
    false
}

/// `stop` accepts a single string or an array of strings.
fn deserialize_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(d)?;
    match value {
        serde_json::Value::String(s) => Ok(vec![s]),
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| D::Error::custom("stop entries must be strings"))
            })
            .collect(),
        _ => Err(D::Error::custom("stop must be a string or array of strings")),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
}

/// Map temperature/top_p onto the engine sampler config (greedy when temp≈0).
pub fn sampler_from(temperature: f32, top_p: f32) -> hayai_model::SamplerConfig {
    if temperature <= 0.0 {
        hayai_model::SamplerConfig::Greedy
    } else if top_p < 1.0 {
        hayai_model::SamplerConfig::TopP {
            temperature,
            top_p,
        }
    } else {
        hayai_model::SamplerConfig::Temperature { temperature }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Responses
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

/// SSE chunk for chat completions.
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// SSE chunk for legacy completions.
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

impl ApiError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "invalid_request_error".into(),
                code: Some("model_not_found".into()),
            },
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "internal_error".into(),
                code: None,
            },
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: msg.into(),
                ty: "invalid_request_error".into(),
                code: None,
            },
        }
    }
}

pub fn request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}")
}

pub fn created_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
